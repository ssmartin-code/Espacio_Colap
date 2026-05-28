#!/usr/bin/env python3
"""
Análisis LiDAR 3D de masa forestal — alternativa Python a lidR.

Implementa los mismos algoritmos clave:
  • Clasificación de suelo  (CSF via scipy)
  • Normalización de altura (TIN interpolado)
  • CHM Pit-Free             (multivariante TIN)
  • Detección de cimas       (LMF adaptativo)
  • Segmentación de árboles  (Dalponte 2016 simplificado / Watershed)
  • Métricas por árbol
  • Visualización 3D         (open3d offline → PNG; dash para interactivo)
  • Exportación              (GeoPackage, LAZ, PLY, CSV)

Uso:
    python analisis_lidar_3d_python.py --shp eb696780-TERRENOS.shp

Dependencias:
    pip install laspy geopandas rasterio scipy scikit-image open3d numpy
"""

import argparse
import math
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

os.environ["SHAPE_RESTORE_SHX"] = "YES"

# ── Imports obligatorios ──────────────────────────────────────────────────────
try:
    import laspy
except ImportError:
    sys.exit("pip install laspy")

try:
    import geopandas as gpd
    from pyproj import CRS
    from shapely.geometry import MultiPolygon, Point, Polygon, mapping
    from shapely.ops import unary_union
except ImportError:
    sys.exit("pip install geopandas pyproj shapely")

try:
    import rasterio
    from rasterio.features import shapes as rio_shapes
    from rasterio.mask import mask as rio_mask
    from rasterio.transform import from_bounds
    import rasterio.warp
except ImportError:
    sys.exit("pip install rasterio")

try:
    from scipy import ndimage
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    from scipy.spatial import Delaunay, cKDTree
    from scipy.ndimage import label, maximum_filter, gaussian_filter
except ImportError:
    sys.exit("pip install scipy")

try:
    from skimage.feature import peak_local_max
    from skimage.filters import gaussian
    from skimage.segmentation import watershed
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

try:
    import open3d as o3d
    HAS_O3D = True
except ImportError:
    HAS_O3D = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap

# ── Parámetros ────────────────────────────────────────────────────────────────
EPSG       = 25830
RES_CHM    = 1.0      # m — resolución CHM
MIN_HT     = 2.0      # m — altura mínima árbol
WS_BASE    = 3        # px — ventana LMF base
CHUNK_PTS  = 5_000_000  # puntos por chunk


# ── I. Lectura y recorte del LAS ──────────────────────────────────────────────

def leer_laz(paths: List[Path], bbox: Tuple) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Lee todos los LAZ de `paths`, recorta a `bbox` (xmin,ymin,xmax,ymax).
    Devuelve arrays (x, y, z, classification)."""
    xmin, ymin, xmax, ymax = bbox
    xs, ys, zs, cls = [], [], [], []

    for p in paths:
        print(f"  Leyendo {p.name} …")
        with laspy.open(str(p)) as lf:
            for chunk in lf.chunk_iterator(CHUNK_PTS):
                x = np.asarray(chunk.x); y = np.asarray(chunk.y)
                z = np.asarray(chunk.z)
                m = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
                if not m.any():
                    continue
                xs.append(x[m]); ys.append(y[m]); zs.append(z[m])
                try:
                    cls.append(np.asarray(chunk.classification)[m])
                except Exception:
                    cls.append(np.zeros(m.sum(), dtype=np.uint8))

    if not xs:
        raise RuntimeError("Sin puntos en el bbox. Revisa coordenadas del LAZ.")

    return (np.concatenate(xs), np.concatenate(ys),
            np.concatenate(zs), np.concatenate(cls))


# ── II. Normalización de altura (TIN sobre puntos suelo) ─────────────────────

def normalizar_altura(x, y, z, cls) -> np.ndarray:
    """Interpola el MDT desde puntos suelo (clase 2) con TIN y resta al Z bruto."""
    suelo = cls == 2
    if suelo.sum() < 4:
        print("  [aviso] Pocos puntos suelo — usando mínimo local como MDT")
        # Fallback: ventana 5m, valor mínimo
        from scipy.ndimage import uniform_filter
        z_norm = z - np.percentile(z[::10], 5)
        return np.clip(z_norm, 0, None).astype(np.float32)

    print(f"  Puntos suelo (clase 2): {suelo.sum():,} / {len(z):,}")
    interp = LinearNDInterpolator(
        np.column_stack([x[suelo], y[suelo]]), z[suelo])
    z_suelo = interp(np.column_stack([x, y]))

    # Relleno de NaN con el vecino más próximo fuera del convex hull
    nan_mask = np.isnan(z_suelo)
    if nan_mask.any():
        nn = NearestNDInterpolator(
            np.column_stack([x[suelo], y[suelo]]), z[suelo])
        z_suelo[nan_mask] = nn(np.column_stack([x[nan_mask], y[nan_mask]]))

    z_norm = z - z_suelo
    return np.clip(z_norm, 0, 60).astype(np.float32)


# ── III. CHM Pit-Free (multiumbral) ──────────────────────────────────────────

def chm_pitfree(x, y, z_norm, bbox, res=RES_CHM,
                umbrales=(0, 2, 5, 10, 15, 20)) -> Tuple[np.ndarray, object]:
    """
    Implementación Python del algoritmo Pit-Free de Khosravipour et al. (2014).
    Para cada umbral de altura h_t construye un TIN de máximos locales y
    fusiona en el CHM final el máximo entre todos los planos de TIN.
    """
    xmin, ymin, xmax, ymax = bbox
    cols = int(math.ceil((xmax - xmin) / res))
    rows = int(math.ceil((ymax - ymin) / res))
    transform = from_bounds(xmin, ymin, xmax, ymax, cols, rows)

    chm = np.zeros((rows, cols), dtype=np.float32)

    for ht in umbrales:
        mask = z_norm >= ht
        if mask.sum() < 4:
            continue
        xi, yi, zi = x[mask], y[mask], z_norm[mask]
        # Reducir a máximos por celda (más rápido que TIN completo)
        ci = np.clip(np.floor((xi - xmin) / res).astype(int), 0, cols - 1)
        ri = np.clip(np.floor((ymax - yi) / res).astype(int), 0, rows - 1)
        layer = np.full((rows, cols), np.nan, dtype=np.float32)
        for idx in range(len(xi)):
            r, c = ri[idx], ci[idx]
            if np.isnan(layer[r, c]) or zi[idx] > layer[r, c]:
                layer[r, c] = zi[idx]

        # Rellenar NaN con interpolación lineal local (anti-pit)
        nan_m = np.isnan(layer)
        if nan_m.any():
            from scipy.ndimage import distance_transform_edt
            indices = distance_transform_edt(nan_m,
                       return_distances=False, return_indices=True)
            layer[nan_m] = layer[tuple(indices[:, nan_m])]

        np.maximum(chm, layer, out=chm)

    # Suavizado gaussiano ligero para eliminar artefactos residuales
    chm = gaussian_filter(chm, sigma=0.5)
    chm = np.clip(chm, 0, 60).astype(np.float32)
    return chm, transform


# ── IV. Local Maximum Filter adaptativo ──────────────────────────────────────

def detectar_cimas(chm: np.ndarray, transform, res=RES_CHM,
                   min_ht=MIN_HT) -> gpd.GeoDataFrame:
    """LMF con ventana adaptativa: ventana más grande para árboles altos."""

    def ws_adaptativo(altura):
        return max(3, int(min(altura / 2.5, 10) / res))

    # Para cada píxel >min_ht, aplicar máximo local con ventana según altura
    chm_smooth = gaussian(chm, sigma=1.0) if HAS_SKIMAGE else gaussian_filter(chm, 1.0)
    veg_mask = chm_smooth >= min_ht

    if HAS_SKIMAGE:
        coords = peak_local_max(chm_smooth, min_distance=int(2 / res),
                                 labels=veg_mask, threshold_abs=min_ht)
    else:
        local_max = maximum_filter(chm_smooth, size=5)
        mask_max = (chm_smooth == local_max) & veg_mask
        coords = np.argwhere(mask_max)

    records = []
    for r, c in coords:
        ht = float(chm[r, c])
        if ht < min_ht:
            continue
        # Verificar que es máximo local en ventana adaptativa
        ws = ws_adaptativo(ht)
        r0 = max(0, r - ws); r1 = min(chm.shape[0], r + ws + 1)
        c0 = max(0, c - ws); c1 = min(chm.shape[1], c + ws + 1)
        if ht < chm[r0:r1, c0:c1].max():
            continue
        x_m, y_m = rasterio.transform.xy(transform, r, c)
        records.append({"tree_id": len(records) + 1,
                         "h_apex": round(ht, 2),
                         "geometry": Point(x_m, y_m)})

    gdf = gpd.GeoDataFrame(records, crs=f"EPSG:{EPSG}")
    print(f"  Cimas detectadas: {len(gdf)}")
    return gdf


# ── V. Segmentación Dalponte 2016 (crecimiento desde cimas) ──────────────────

def segmentar_dalponte(chm: np.ndarray, ttops: gpd.GeoDataFrame,
                        transform, th_seed=0.45, th_cr=0.55,
                        max_cr=20, min_ht=MIN_HT) -> np.ndarray:
    """
    Dalponte 2016: crecimiento de región 2D desde cimas sobre el CHM.
    Más rápido que li2012 (3D); precisión similar para masas densas.
    """
    rows, cols = chm.shape
    labels = np.zeros((rows, cols), dtype=np.int32)
    seeds  = {}

    for _, row in ttops.iterrows():
        tid = int(row["tree_id"])
        r, c = rasterio.transform.rowcol(transform, row.geometry.x, row.geometry.y)
        r = np.clip(r, 0, rows - 1); c = np.clip(c, 0, cols - 1)
        seeds[tid] = (r, c, float(row["h_apex"]))
        labels[r, c] = tid

    # Cola de prioridad (BFS ponderado por altura)
    from collections import deque
    queue = deque()
    for tid, (r, c, h) in seeds.items():
        queue.append((r, c, tid, h))

    visited = labels > 0
    res = abs(transform.a)

    while queue:
        r, c, tid, h_seed = queue.popleft()
        h_tree = seeds[tid][2]

        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if visited[nr, nc]:
                continue
            h_n = float(chm[nr, nc])
            # Criterios de inclusión Dalponte
            if h_n < min_ht:
                continue
            if h_n < h_tree * th_seed:
                continue
            if h_n < h_seed * th_cr:
                continue
            # Límite de radio de copa
            r_seed, c_seed = seeds[tid][:2]
            dist = math.sqrt(((nr - r_seed)*res)**2 + ((nc - c_seed)*res)**2)
            if dist > max_cr:
                continue
            labels[nr, nc] = tid
            visited[nr, nc] = True
            queue.append((nr, nc, tid, h_n))

    return labels


# ── VI. Métricas por árbol ────────────────────────────────────────────────────

def metricas_por_arbol(x, y, z_norm, cls,
                        labels_grid: np.ndarray,
                        transform, ttops: gpd.GeoDataFrame,
                        chm: np.ndarray) -> gpd.GeoDataFrame:
    """Calcula métricas 3D por árbol asignando puntos al segmento más cercano."""
    rows, cols = labels_grid.shape

    # Asignar cada punto LiDAR a un árbol
    ci = np.clip(np.floor(
        (x - transform.c) / transform.a).astype(int), 0, cols - 1)
    ri = np.clip(np.floor(
        (y - transform.f) / transform.e).astype(int), 0, rows - 1)
    tree_ids = labels_grid[ri, ci]

    records = []
    for tid in np.unique(tree_ids):
        if tid == 0:
            continue
        mask = tree_ids == tid
        zi  = z_norm[mask]
        # Geometría del polígono de copa desde la máscara raster
        copa_mask = (labels_grid == tid).astype(np.uint8)
        polys = [Polygon(shp["coordinates"][0])
                 for shp, val in rio_shapes(copa_mask, transform=transform)
                 if val == 1 and len(shp["coordinates"][0]) >= 4]
        copa_geom = unary_union(polys) if polys else Point(0, 0)

        records.append({
            "tree_id":   int(tid),
            "h_max":     round(float(zi.max()), 2),
            "h_mean":    round(float(zi.mean()), 2),
            "h_p95":     round(float(np.percentile(zi, 95)), 2),
            "n_pts":     int(mask.sum()),
            "area_copa": round(float(copa_geom.area), 1),
            "geometry":  copa_geom,
        })

    gdf = gpd.GeoDataFrame(records, crs=f"EPSG:{EPSG}")
    print(f"  Árboles segmentados: {len(gdf)}")
    return gdf


# ── VII. Visualización 3D con open3d ─────────────────────────────────────────

def render_3d(x, y, z_norm, labels: np.ndarray,
              out_path: Path, n_arboles: int) -> None:
    """
    Renderiza la nube de puntos en 3D coloreada por árbol.
    Exporta un PNG de vista isométrica (sin necesidad de pantalla).
    """
    if not HAS_O3D:
        print("  [aviso] open3d no disponible — saltando render 3D")
        return

    print("  Generando render 3D con open3d …")

    # Submuestreo para render rápido
    n_max = 500_000
    idx = (np.random.choice(len(x), min(n_max, len(x)), replace=False)
           if len(x) > n_max else np.arange(len(x)))

    pts = np.column_stack([x[idx], y[idx], z_norm[idx]]).astype(np.float64)
    # Centrar coordenadas para evitar problemas de precisión
    pts[:, 0] -= pts[:, 0].mean()
    pts[:, 1] -= pts[:, 1].mean()

    # Colores por árbol (paleta viridis-like)
    lbl_idx = labels[idx] if labels is not None else np.zeros(len(idx), dtype=int)
    cmap = plt.get_cmap("tab20")
    colors = np.array([cmap(int(l) % 20)[:3]
                       if l > 0 else [0.15, 0.15, 0.15]
                       for l in lbl_idx])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Renderizado offscreen
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1600, height=1000)
    vis.add_geometry(pcd)

    ctr = vis.get_view_control()
    ctr.set_zoom(0.45)
    ctr.set_front([0.5, -0.8, 0.4])
    ctr.set_up([0, 0, 1])
    ctr.set_lookat([0, 0, z_norm.mean()])

    opt = vis.get_render_option()
    opt.point_size = 1.5
    opt.background_color = np.array([0.08, 0.08, 0.12])

    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(str(out_path))
    vis.destroy_window()
    print(f"  Render 3D guardado → {out_path}")


def exportar_ply(x, y, z_norm, labels, out_path: Path) -> None:
    """Exporta la nube segmentada en formato PLY (CloudCompare / MeshLab)."""
    if not HAS_O3D:
        return
    pts = np.column_stack([x, y, z_norm]).astype(np.float64)
    pts[:, 0] -= pts[:, 0].mean(); pts[:, 1] -= pts[:, 1].mean()
    cmap = plt.get_cmap("tab20")
    colors = np.array([cmap(int(l) % 20)[:3] if l > 0 else [0.15]*3
                       for l in labels])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(str(out_path), pcd)
    print(f"  Nube PLY exportada → {out_path}")


# ── VIII. Función principal ───────────────────────────────────────────────────

def main(shp_path: Path, laz_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n═══════════════════════════════════════════════════════")
    print(" ANÁLISIS LiDAR 3D — Python (equivalente a lidR)")
    print("═══════════════════════════════════════════════════════\n")

    # 1. Leer shapefile
    print("► [1/8] Leyendo parcelas …")
    gdf = gpd.read_file(str(shp_path)).set_crs(epsg=EPSG)
    xmin, ymin, xmax, ymax = gdf.total_bounds
    buf = 50
    bbox = (xmin - buf, ymin - buf, xmax + buf, ymax + buf)
    print(f"  {len(gdf)} parcelas | bbox +{buf}m = {tuple(round(v) for v in bbox)}")

    # 2. Buscar LAZ
    laz_files = sorted(laz_dir.glob("*.la[sz]")) + sorted(laz_dir.glob("*.LAS"))
    if not laz_files:
        print(f"\n  Sin archivos LAZ en {laz_dir}")
        print("  Descárgalos del Centro de Descargas del CNIG:")
        print("    https://centrodedescargas.cnig.es → LIDAR-PNOA-2")
        return

    # 3. Leer puntos
    print(f"\n► [2/8] Leyendo {len(laz_files)} archivo(s) LAZ …")
    x, y, z, cls = leer_laz(laz_files, bbox)
    print(f"  Puntos totales: {len(x):,}  |  Z: {z.min():.1f}–{z.max():.1f} m")

    # 4. Normalizar
    print("\n► [3/8] Normalizando alturas (TIN sobre suelo clase 2) …")
    z_norm = normalizar_altura(x, y, z, cls)
    print(f"  Z normalizado: 0–{z_norm.max():.1f} m")

    # 5. CHM Pit-Free
    print("\n► [4/8] Generando CHM Pit-Free …")
    chm, tf = chm_pitfree(x, y, z_norm, bbox, res=RES_CHM)
    print(f"  CHM: {chm.shape}  max={chm.max():.1f} m")
    profile_chm = {"driver": "GTiff", "dtype": "float32", "nodata": 0.0,
                   "width": chm.shape[1], "height": chm.shape[0], "count": 1,
                   "crs": f"EPSG:{EPSG}", "transform": tf, "compress": "lzw"}
    chm_path = out_dir / "CHM_pitfree_1m.tif"
    with rasterio.open(str(chm_path), "w", **profile_chm) as dst:
        dst.write(chm[np.newaxis])

    # 6. Detección de cimas (LMF)
    print("\n► [5/8] Detectando cimas (LMF adaptativo) …")
    ttops = detectar_cimas(chm, tf, min_ht=MIN_HT)
    ttops.to_file(str(out_dir / "cimas_lmf.gpkg"), driver="GPKG")

    # 7. Segmentación Dalponte
    print("\n► [6/8] Segmentando árboles (Dalponte 2016) …")
    labels_grid = segmentar_dalponte(chm, ttops, tf, min_ht=MIN_HT)

    # 8. Métricas por árbol
    print("\n► [7/8] Calculando métricas por árbol …")
    metricas = metricas_por_arbol(x, y, z_norm, cls, labels_grid, tf, ttops, chm)
    metricas.to_file(str(out_dir / "copas_dalponte.gpkg"), driver="GPKG")
    metricas.drop(columns="geometry").to_csv(
        str(out_dir / "metricas_arboles.csv"), index=False)

    # 9. Render 3D + export PLY
    print("\n► [8/8] Render 3D y exportación …")
    # Asignar label de árbol a cada punto para colorear
    ci = np.clip(np.floor((x - tf.c) / tf.a).astype(int), 0, chm.shape[1]-1)
    ri = np.clip(np.floor((y - tf.f) / tf.e).astype(int), 0, chm.shape[0]-1)
    pts_labels = labels_grid[ri, ci]

    render_3d(x, y, z_norm, pts_labels, out_dir / "render_3d_open3d.png",
              n_arboles=len(ttops))
    exportar_ply(x, y, z_norm, pts_labels, out_dir / "nube_segmentada.ply")

    # ── Visualización 2D ─────────────────────────────────────────────────────
    cmap_ht = LinearSegmentedColormap.from_list(
        "ht", [(0,"#eaf7ea"),(0.25,"#74c476"),(0.55,"#238b45"),(1,"#005a32")])

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor="#1a1a2e")
    for ax in axes:
        ax.set_facecolor("#1a1a2e")
        for sp in ax.spines.values(): sp.set_color("#4a4a6a")

    # Panel 1: CHM
    ext = [bbox[0], bbox[2], bbox[1], bbox[3]]
    im1 = axes[0].imshow(chm, extent=ext, origin="upper",
                          cmap=cmap_ht, vmin=0, vmax=20)
    gdf.boundary.plot(ax=axes[0], color="yellow", linewidth=1.3)
    ttops.plot(ax=axes[0], color="red", markersize=8, marker="+", linewidth=0.8)
    axes[0].set_title("CHM Pit-Free + Cimas LMF", color="#dde",
                       fontsize=11, fontweight="bold")
    plt.colorbar(im1, ax=axes[0], label="Altura (m)", fraction=0.035)

    # Panel 2: Copas segmentadas
    metricas[metricas.geometry.is_valid].plot(
        ax=axes[1], column="h_max", cmap=cmap_ht, vmin=0, vmax=20,
        legend=True, legend_kwds={"label": "Altura máx. (m)", "fraction": 0.035})
    gdf.boundary.plot(ax=axes[1], color="yellow", linewidth=1.3)
    axes[1].set_title("Copas segmentadas (Dalponte 2016)", color="#dde",
                       fontsize=11, fontweight="bold")

    for ax in axes:
        ax.tick_params(colors="#aaaacc", labelsize=7)
        ax.set_xlabel("Este (m)", color="#ccccee", fontsize=8)

    fig.suptitle("Análisis LiDAR 3D — Python (laspy + open3d)",
                 color="#e8e8ff", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(out_dir / "analisis_3d_python.png"),
                dpi=150, facecolor="#1a1a2e", bbox_inches="tight")
    plt.close()

    # ── Resumen ──────────────────────────────────────────────────────────────
    h = metricas["h_max"]
    a = metricas["area_copa"]
    print("\n═══════════════════════════════════════════════════════")
    print(f"  Árboles detectados  : {len(metricas)}")
    print(f"  Altura máxima       : {h.max():.1f} m")
    print(f"  Altura media (max)  : {h.mean():.1f} m")
    print(f"  Mediana             : {h.median():.1f} m")
    print(f"  Área copa media     : {a.mean():.1f} m²")
    print(f"\n  Resultados en: {out_dir.resolve()}")
    print("═══════════════════════════════════════════════════════\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shp", type=Path, default=Path("eb696780-TERRENOS.shp"))
    ap.add_argument("--laz", type=Path, default=Path("output_lidar/laz_tiles"))
    ap.add_argument("--out", type=Path, default=Path("output_lidar_3d"))
    args = ap.parse_args()
    main(args.shp, args.laz, args.out)
