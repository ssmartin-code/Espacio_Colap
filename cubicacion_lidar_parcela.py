#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
 Inventario forestal de parcela a partir de PNOA-LiDAR
=============================================================================
 Pipeline reutilizable para cualquier parcela de España. A partir de:
   - un límite de parcela (GML de Catastro, SHP, GPKG...)
   - una nube de puntos LiDAR PNOA (.laz / .las), idealmente clasificada y RGB

 produce, paso a paso y guardando cada resultado:

   [1] Mapa de alturas de la vegetación (CHM normalizado)      -> 01_altura/
   [2] Segmentación de copas individuales                      -> 02_copas/
   [3] Clasificación de especies pinar vs alcornoque           -> 03_especies/
   [4] Cubicación aproximada de la madera                      -> 04_cubicacion/

 Diseñado para ser CALIBRABLE: las allometrías de volumen y los umbrales de
 clasificación están centralizados en CONFIG y documentados.

 Uso típico (Windows):
   python cubicacion_lidar_parcela.py ^
     --gml  "H:\\...\\Parcela_Catastral\\GML_Parcelas.gml" ^
     --laz  "H:\\...\\PNOA_Lidar\\PNOA_2019_CYL_C_304-4536_ORT-CLA-RGB.laz" ^
     --out  "H:\\...\\Estudio"

 Dependencias:
   pip install laspy[lazrs] numpy scipy rasterio geopandas shapely
               scikit-image scikit-learn matplotlib pandas pyogrio
=============================================================================
"""

from __future__ import annotations
import os
import sys
import json
import argparse
from dataclasses import dataclass, field, asdict

# --- Resolver conflicto PROJ con PostgreSQL/PostGIS en Windows ---------------
# Si PostgreSQL tiene una versión antigua de proj.db, rasterio/pyproj fallan.
# Forzamos el uso del proj.db de la instalación de Python (conda/pip).
_conda_root = os.path.dirname(sys.executable)
_proj_candidates = [
    os.path.join(_conda_root, "Library", "share", "proj"),   # miniconda base
    os.path.join(_conda_root, "..", "Library", "share", "proj"),  # conda env
    os.path.join(_conda_root, "share", "proj"),               # Linux/Mac
]
for _p in _proj_candidates:
    _p = os.path.normpath(_p)
    if os.path.isfile(os.path.join(_p, "proj.db")):
        os.environ["PROJ_DATA"] = _p
        os.environ["PROJ_LIB"]  = _p
        break
else:
    try:
        import pyproj as _pp
        _p = _pp.datadir.get_data_dir()
        os.environ.setdefault("PROJ_DATA", _p)
        os.environ.setdefault("PROJ_LIB",  _p)
    except Exception:
        pass
os.environ.setdefault("CPL_LOG_ERRORS", "OFF")   # silencia avisos DLL 32-bit
# -----------------------------------------------------------------------------

# Catastro a veces entrega SHP sin .shx; esto permite reconstruirlo.
os.environ.setdefault("SHAPE_RESTORE_SHX", "YES")

import numpy as np
import pandas as pd
import laspy
import rasterio
from rasterio.transform import from_origin
from rasterio.features import rasterize as rio_rasterize
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon, box, shape, mapping
from shapely.ops import unary_union
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, maximum_filter
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import regionprops, find_contours
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

# rasterio trae su propio proj.db (compatible con SU versión de PROJ). Lo usamos
# SOLO al escribir rásters, vía rasterio.Env, sin tocar el PROJ de pyproj.
_RIO_PROJ = os.path.join(os.path.dirname(rasterio.__file__), "proj_data")
if not os.path.isfile(os.path.join(_RIO_PROJ, "proj.db")):
    _RIO_PROJ = None


def _wkt_desde_epsg(epsg):
    """Obtiene el WKT del CRS usando pyproj (que sí funciona), evitando que
    rasterio/GDAL tenga que resolver el código EPSG contra su proj.db."""
    try:
        import pyproj
        return pyproj.CRS.from_epsg(epsg).to_wkt()
    except Exception:
        return None


def escribir_geotiff(path, array, transform, epsg, nx, ny):
    """Escribe un GeoTIFF de forma robusta frente a conflictos de PROJ.

    1) Intenta con el CRS en formato WKT (derivado por pyproj) — no requiere
       que GDAL resuelva el código EPSG contra su proj.db.
    2) Si falla, escribe el ráster SIN CRS (asignable luego en QGIS) y avisa.
    """
    base = dict(driver="GTiff", height=ny, width=nx, count=1,
                dtype="float32", transform=transform, nodata=np.nan)
    env_kw = dict(PROJ_DATA=_RIO_PROJ, PROJ_LIB=_RIO_PROJ) if _RIO_PROJ else {}
    wkt = _wkt_desde_epsg(epsg)
    try:
        crs = rasterio.crs.CRS.from_wkt(wkt) if wkt else f"EPSG:{epsg}"
        with rasterio.Env(**env_kw):
            with rasterio.open(path, "w", crs=crs, **base) as dst:
                dst.write(array.astype(np.float32), 1)
        return True
    except Exception as e:
        log(f"    AVISO PROJ: no se pudo incrustar el CRS ({type(e).__name__}).")
        with rasterio.open(path, "w", crs=None, **base) as dst:
            dst.write(array.astype(np.float32), 1)
        log(f"    -> GeoTIFF escrito SIN CRS. Asigna EPSG:{epsg} en QGIS "
            f"(Capa > Establecer SRC) si lo necesitas georreferenciado.")
        return False


def escribir_vector(gdf, path, epsg):
    """Escribe un GeoPackage de forma robusta frente a conflictos de PROJ."""
    try:
        gdf.to_file(path, driver="GPKG")
        return True
    except Exception as e:
        log(f"    AVISO PROJ al escribir {os.path.basename(path)} "
            f"({type(e).__name__}); reintento sin CRS incrustado.")
        g2 = gdf.copy()
        try:
            g2 = g2.set_crs(None, allow_override=True)
        except Exception:
            pass
        g2.to_file(path, driver="GPKG")
        log(f"    -> {os.path.basename(path)} escrito SIN CRS "
            f"(asigna EPSG:{epsg} en QGIS si lo necesitas).")
        return False


# =============================================================================
#  CONFIGURACIÓN  (todo lo "calibrable" vive aquí)
# =============================================================================

@dataclass
class EspecieParams:
    """Parámetros allométricos por especie. CALIBRAR con tarifas locales/IFN."""
    nombre: str
    # Altura -> DBH (diámetro normal, cm):  DBH = dbh_k * (H ** dbh_p)
    dbh_k: float
    dbh_p: float
    # Volumen de fuste (m3) = vol_a * DBH_cm**vol_b * H_m**vol_c
    # (tarifa de doble entrada, forma clásica IFN; coeficientes orientativos)
    vol_a: float
    vol_b: float
    vol_c: float
    color: str  # color para los mapas


# Coeficientes POR DEFECTO — orientativos, estilo IFN. Sustituir por tarifas
# regionales calibradas cuando se disponga de ellas (p.ej. IFN4 / Junta CyL).
ESPECIES = {
    "pino": EspecieParams(
        nombre="Pinar (Pinus pinaster/pinea)",
        dbh_k=1.60, dbh_p=1.00,          # pino esbelto: DBH≈1.6·H
        vol_a=2.5e-5, vol_b=1.85, vol_c=0.95,
        color="#1f6b3b",
    ),
    "alcornoque": EspecieParams(
        nombre="Alcornocal (Quercus suber)",
        dbh_k=2.60, dbh_p=1.00,          # copa ancha y trabada: DBH≈2.6·H
        vol_a=3.8e-5, vol_b=1.90, vol_c=0.80,
        color="#9c5a2e",
    ),
}


@dataclass
class Config:
    # --- Sistema de referencia (PNOA = ETRS89 UTM; CyL/Salamanca = zona 30N) ---
    epsg: int = 25830
    # --- Rasterización ---
    res: float = 0.5                 # resolución del CHM (m)
    buffer_parcela: float = 15.0     # margen alrededor de la parcela (m)
    # --- Detección/segmentación de árboles ---
    altura_min: float = 2.0          # altura mínima para considerar arbolado (m)
    suavizado_sigma: float = 0.6     # sigma gaussiano sobre el CHM (px)
    # ventana de detección de cimas adaptativa (radio en m según altura)
    # --- Clasificación de especies ---
    usar_kmeans: bool = True
    # --- Clases LAS estándar ---
    clase_suelo: int = 2
    clases_vegetacion: tuple = (3, 4, 5)   # baja, media, alta
    especies: dict = field(default_factory=lambda: ESPECIES)


CFG = Config()


# =============================================================================
#  Utilidades
# =============================================================================

def log(msg: str):
    print(msg, flush=True)


def crea_dirs(out_dir: str) -> dict:
    sub = {
        "altura":     os.path.join(out_dir, "01_altura"),
        "copas":      os.path.join(out_dir, "02_copas"),
        "especies":   os.path.join(out_dir, "03_especies"),
        "cubicacion": os.path.join(out_dir, "04_cubicacion"),
    }
    for d in [out_dir] + list(sub.values()):
        os.makedirs(d, exist_ok=True)
    return sub


def ventana_lmf(h: float) -> float:
    """Radio (m) de la ventana de máximo local, adaptativo a la altura."""
    if h < 5:   return 1.0
    if h < 10:  return 1.5
    if h < 20:  return 2.5
    return 3.5


# =============================================================================
#  PASO 0 — Leer límite de parcela
# =============================================================================

def leer_parcela(path: str, epsg: int) -> gpd.GeoDataFrame:
    log(f"\n[0] Leyendo parcela: {path}")
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        log(f"    Parcela sin CRS -> asigno EPSG:{epsg}")
        gdf = gdf.set_crs(epsg=epsg)
    elif gdf.crs.to_epsg() != epsg:
        log(f"    Reproyecto parcela {gdf.crs.to_epsg()} -> {epsg}")
        gdf = gdf.to_crs(epsg=epsg)
    minx, miny, maxx, maxy = gdf.total_bounds
    log(f"    {len(gdf)} polígono(s) | bbox: {minx:.0f},{miny:.0f} — {maxx:.0f},{maxy:.0f}")
    log(f"    Superficie total: {gdf.geometry.area.sum()/1e4:.2f} ha")
    return gdf


# =============================================================================
#  PASO 1 — Leer LiDAR y normalizar alturas -> CHM (mapa de alturas)
# =============================================================================

def leer_laz(path: str, bbox, epsg: int):
    """Lee la nube recortada al bbox. Devuelve dict de arrays."""
    log(f"\n[1] Leyendo LiDAR: {path}")
    minx, miny, maxx, maxy = bbox
    with laspy.open(path) as fh:
        hdr = fh.header
        log(f"    {hdr.point_count:,} puntos | formato {hdr.point_format.id} "
            f"| escala {hdr.scales}")
        tiene_rgb = all(d in hdr.point_format.dimension_names
                        for d in ("red", "green", "blue"))
        las = fh.read()

    x, y, z = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)
    m = (x >= minx) & (x <= maxx) & (y >= miny) & (y <= maxy)
    log(f"    Puntos dentro del bbox de la parcela: {m.sum():,}")

    out = {
        "x": x[m], "y": y[m], "z": z[m],
        "cls": np.asarray(las.classification)[m].astype(np.int16),
        "rn":  np.asarray(las.return_number)[m] if "return_number" in las.point_format.dimension_names else None,
    }
    if tiene_rgb:
        # PNOA suele venir en 16 bits -> normalizo a 0..255
        r = np.asarray(las.red)[m].astype(np.float32)
        g = np.asarray(las.green)[m].astype(np.float32)
        b = np.asarray(las.blue)[m].astype(np.float32)
        escala = 257.0 if max(r.max(), g.max(), b.max()) > 255 else 1.0
        out["r"], out["g"], out["b"] = r/escala, g/escala, b/escala
        log("    RGB disponible (se usará para clasificar especies)")
    else:
        out["r"] = out["g"] = out["b"] = None
        log("    Sin RGB: la clasificación de especies usará solo estructura")
    return out


def construir_dtm(x, y, z, cls, bbox, res):
    """Modelo Digital del Terreno por interpolación de puntos suelo (clase 2)."""
    minx, miny, maxx, maxy = bbox
    nx = int(np.ceil((maxx - minx) / res))
    ny = int(np.ceil((maxy - miny) / res))
    transform = from_origin(minx, maxy, res, res)

    suelo = cls == CFG.clase_suelo
    if suelo.sum() < 10:
        log("    AVISO: pocos puntos de suelo; uso mínimos locales como terreno")
        suelo = np.ones_like(cls, dtype=bool)

    gx = minx + (np.arange(nx) + 0.5) * res
    gy = maxy - (np.arange(ny) + 0.5) * res
    GX, GY = np.meshgrid(gx, gy)

    dtm = griddata((x[suelo], y[suelo]), z[suelo], (GX, GY),
                   method="linear")
    nan = np.isnan(dtm)
    if nan.any():
        dtm[nan] = griddata((x[suelo], y[suelo]), z[suelo],
                            (GX[nan], GY[nan]), method="nearest")
    return dtm, transform, (nx, ny)


def construir_chm(x, y, z, dtm, transform, shape_xy, res):
    """CHM = max(altura sobre el terreno) por celda."""
    nx, ny = shape_xy
    minx = transform.c
    maxy = transform.f
    col = np.clip(((x - minx) / res).astype(int), 0, nx - 1)
    row = np.clip(((maxy - y) / res).astype(int), 0, ny - 1)
    h = z - dtm[row, col]                      # altura normalizada por punto
    h = np.where(h < 0, 0, h)

    chm = np.zeros((ny, nx), dtype=np.float32)
    np.maximum.at(chm, (row, col), h.astype(np.float32))
    return chm, h, row, col


def paso1_altura(pts, parcela, dirs, cfg: Config):
    log("\n========== PASO 1 · MAPA DE ALTURAS ==========")
    minx, miny, maxx, maxy = parcela.total_bounds
    bbox = (minx - cfg.buffer_parcela, miny - cfg.buffer_parcela,
            maxx + cfg.buffer_parcela, maxy + cfg.buffer_parcela)

    dtm, transform, shape_xy = construir_dtm(
        pts["x"], pts["y"], pts["z"], pts["cls"], bbox, cfg.res)
    chm, h_pt, row, col = construir_chm(
        pts["x"], pts["y"], pts["z"], dtm, transform, shape_xy, cfg.res)
    pts["h"], pts["row"], pts["col"] = h_pt, row, col

    # Recorte al polígono real de la parcela
    nx, ny = shape_xy
    mask = rio_rasterize(
        [(g, 1) for g in parcela.geometry], out_shape=(ny, nx),
        transform=transform, fill=0, dtype="uint8").astype(bool)
    chm_parcela = np.where(mask, chm, np.nan)

    # Exportar GeoTIFF del CHM (con manejo robusto del CRS / PROJ)
    tif = os.path.join(dirs["altura"], "CHM_alturas.tif")
    escribir_geotiff(tif, chm_parcela, transform, cfg.epsg, nx, ny)
    log(f"    CHM -> {tif}")

    valid = chm_parcela[np.isfinite(chm_parcela)]
    if valid.size:
        log(f"    Altura veg.: min {valid.min():.1f} | "
            f"media {valid[valid>=cfg.altura_min].mean():.1f} | "
            f"máx {valid.max():.1f} m")

    # PNG con sombreado
    _mapa_altura_png(chm_parcela, transform, parcela,
                     os.path.join(dirs["altura"], "mapa_alturas.png"), cfg)
    return chm, chm_parcela, transform, mask, shape_xy


def _mapa_altura_png(chm, transform, parcela, out_png, cfg):
    ny, nx = chm.shape
    extent = (transform.c, transform.c + nx*cfg.res,
              transform.f - ny*cfg.res, transform.f)
    fig, ax = plt.subplots(figsize=(10, 9), facecolor="#15151f")
    ax.set_facecolor("#15151f")
    ls = LightSource(azdeg=315, altdeg=45)
    base = np.nan_to_num(chm)
    hs = ls.hillshade(base, vert_exag=2, dx=cfg.res, dy=cfg.res)
    ax.imshow(hs, cmap="gray", extent=extent, alpha=0.5)
    im = ax.imshow(chm, cmap="YlGn", extent=extent, alpha=0.85,
                   vmin=0, vmax=np.nanpercentile(chm, 99))
    for g in parcela.geometry:
        _plot_geom(ax, g, color="yellow", lw=1.8)
    cb = fig.colorbar(im, ax=ax, shrink=0.7)
    cb.set_label("Altura vegetación (m)", color="#ccccee")
    cb.ax.yaxis.set_tick_params(color="#ccccee")
    plt.setp(plt.getp(cb.ax, "yticklabels"), color="#ccccee")
    ax.set_title("Mapa de alturas de la vegetación (CHM)", color="#ddddee")
    ax.tick_params(colors="#aaaacc")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, facecolor="#15151f")
    plt.close()
    log(f"    Mapa -> {out_png}")


def _plot_geom(ax, geom, **kw):
    geoms = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
    for g in geoms:
        xs, ys = g.exterior.xy
        ax.plot(xs, ys, **kw)


# =============================================================================
#  PASO 2 — Detección de cimas y segmentación de copas
# =============================================================================

def paso2_copas(chm, transform, mask, dirs, cfg: Config):
    log("\n========== PASO 2 · SEGMENTACIÓN DE COPAS ==========")
    chm_s = gaussian_filter(np.nan_to_num(chm), cfg.suavizado_sigma)
    chm_s = np.where(mask, chm_s, 0)

    # --- Detección de cimas con ventana adaptativa ---
    # Aproximamos la ventana variable usando un radio acorde a la altura media.
    h_med = chm_s[chm_s >= cfg.altura_min].mean() if (chm_s >= cfg.altura_min).any() else 5
    rad_px = max(1, int(round(ventana_lmf(h_med) / cfg.res)))
    coords = peak_local_max(chm_s, min_distance=rad_px,
                            threshold_abs=cfg.altura_min, exclude_border=False)
    log(f"    Cimas detectadas: {len(coords)} (ventana ≈ {rad_px} px)")

    # --- Segmentación watershed desde las cimas ---
    markers = np.zeros(chm_s.shape, dtype=np.int32)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i
    suelo = chm_s < cfg.altura_min
    labels = watershed(-chm_s, markers, mask=~suelo)
    n_arb = len(np.unique(labels)) - 1
    log(f"    Copas segmentadas: {n_arb}")

    # --- Polígonos de copa + métricas geométricas ---
    res = cfg.res
    minx, maxy = transform.c, transform.f
    registros = []
    for rp in regionprops(labels, intensity_image=chm_s):
        if rp.area * res * res < 1.0:          # descarta copas < 1 m²
            continue
        # contorno -> polígono en coordenadas del terreno
        m_local = (labels == rp.label)
        cont = find_contours(m_local.astype(float), 0.5)
        if not cont:
            continue
        cc = max(cont, key=len)
        poly_xy = [(minx + (c + 0.5) * res, maxy - (r + 0.5) * res)
                   for r, c in cc]
        if len(poly_xy) < 4:
            continue
        poly = Polygon(poly_xy)
        if not poly.is_valid or poly.area < 1.0:
            poly = poly.buffer(0)
            if poly.is_empty:
                continue
        r0, c0 = rp.coords[np.argmax(rp.image_intensity[rp.image])] \
            if False else (int(rp.centroid[0]), int(rp.centroid[1]))
        # cima real (máximo del CHM dentro de la copa)
        rr, cc2 = np.where(m_local)
        idx = np.argmax(chm_s[rr, cc2])
        cima_r, cima_c = rr[idx], cc2[idx]
        h_max = float(chm_s[cima_r, cima_c])
        area = float(poly.area)
        diam = 2.0 * np.sqrt(area / np.pi)
        registros.append({
            "tree_id": int(rp.label),
            "h_max": h_max,
            "area_copa": area,
            "diam_copa": diam,
            "esbeltez": h_max / diam if diam > 0 else np.nan,
            "cima_x": minx + (cima_c + 0.5) * res,
            "cima_y": maxy - (cima_r + 0.5) * res,
            "geometry": poly,
        })

    gdf = gpd.GeoDataFrame(registros, crs=f"EPSG:{cfg.epsg}")
    gpkg = os.path.join(dirs["copas"], "copas_segmentadas.gpkg")
    escribir_vector(gdf, gpkg, cfg.epsg)
    cimas = gpd.GeoDataFrame(
        gdf[["tree_id", "h_max"]].copy(),
        geometry=[Point(xy) for xy in zip(gdf.cima_x, gdf.cima_y)],
        crs=f"EPSG:{cfg.epsg}")
    escribir_vector(cimas, os.path.join(dirs["copas"], "cimas_arboles.gpkg"), cfg.epsg)
    log(f"    Copas -> {gpkg}")
    return gdf, labels


# =============================================================================
#  PASO 3 — Métricas por árbol + clasificación de especies
# =============================================================================

def metricas_por_arbol(gdf, labels, pts, cfg: Config):
    """Añade estadísticos de color (RGB) y estructura por copa."""
    tiene_rgb = pts.get("r") is not None
    row, col, h = pts["row"], pts["col"], pts["h"]
    lab_pt = labels[row, col]

    exg, vert_skew = {}, {}
    for tid in gdf.tree_id:
        m = (lab_pt == tid) & (h >= cfg.altura_min)
        if m.sum() < 3:
            exg[tid], vert_skew[tid] = np.nan, np.nan
            continue
        if tiene_rgb:
            r, g, b = pts["r"][m], pts["g"][m], pts["b"][m]
            s = r + g + b + 1e-6
            # Excess Green normalizado: alto = más verde (conífera densa)
            exg[tid] = float(np.mean((2*g - r - b) / s))
        else:
            exg[tid] = np.nan
        hh = h[m]
        # asimetría vertical: copas cónicas (pino) concentran masa abajo
        mu, sd = hh.mean(), hh.std() + 1e-6
        vert_skew[tid] = float(np.mean(((hh - mu) / sd) ** 3))

    gdf["exg"] = gdf.tree_id.map(exg)
    gdf["vert_skew"] = gdf.tree_id.map(vert_skew)
    return gdf


def paso3_especies(gdf, labels, pts, parcela, dirs, transform, cfg: Config):
    log("\n========== PASO 3 · ESPECIES (pinar vs alcornoque) ==========")
    gdf = metricas_por_arbol(gdf, labels, pts, cfg)

    # --- Reglas estructurales -> "puntuación de conífera" ---
    # Pino: esbelto (h/diam alto), copa estrecha, asimetría vertical positiva,
    #       y (si hay RGB) verde intenso.
    feats = gdf[["esbeltez", "vert_skew", "exg"]].copy()
    feats["exg"] = feats["exg"].fillna(feats["exg"].median() if feats["exg"].notna().any() else 0)
    feats = feats.fillna(feats.median(numeric_only=True))

    if cfg.usar_kmeans and len(gdf) >= 4:
        X = StandardScaler().fit_transform(feats.values)
        km = KMeans(n_clusters=2, n_init=10, random_state=42).fit(X)
        lab = km.labels_
        # El clúster con mayor esbeltez media = pino
        esb0 = gdf.esbeltez[lab == 0].mean()
        esb1 = gdf.esbeltez[lab == 1].mean()
        cl_pino = 0 if esb0 >= esb1 else 1
        gdf["especie"] = np.where(lab == cl_pino, "pino", "alcornoque")
        metodo = "KMeans(2) sobre [esbeltez, asimetría vertical, verdor RGB]"
    else:
        # Fallback puramente por reglas
        umbral = gdf.esbeltez.median()
        gdf["especie"] = np.where(gdf.esbeltez >= umbral, "pino", "alcornoque")
        metodo = "regla por mediana de esbeltez"

    n_pino = (gdf.especie == "pino").sum()
    n_alc = (gdf.especie == "alcornoque").sum()
    log(f"    Método: {metodo}")
    log(f"    Pinar: {n_pino} árboles | Alcornocal: {n_alc} árboles")

    gpkg = os.path.join(dirs["especies"], "copas_especies.gpkg")
    escribir_vector(gdf, gpkg, cfg.epsg)
    _mapa_especies_png(gdf, parcela,
                       os.path.join(dirs["especies"], "mapa_especies.png"), cfg)
    log(f"    Especies -> {gpkg}")
    return gdf


def _mapa_especies_png(gdf, parcela, out_png, cfg):
    fig, ax = plt.subplots(figsize=(10, 9), facecolor="#15151f")
    ax.set_facecolor("#15151f")
    for esp, p in cfg.especies.items():
        sub = gdf[gdf.especie == esp]
        for g in sub.geometry:
            _fill_geom(ax, g, color=p.color)
        ax.scatter([], [], c=p.color, label=f"{p.nombre} ({len(sub)})", s=80)
    for g in parcela.geometry:
        _plot_geom(ax, g, color="yellow", lw=1.8)
    ax.legend(loc="upper right", facecolor="#1f1f2e", labelcolor="#ddddee")
    ax.set_title("Clasificación de especies por copa", color="#ddddee")
    ax.set_aspect("equal")
    ax.tick_params(colors="#aaaacc")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, facecolor="#15151f")
    plt.close()
    log(f"    Mapa -> {out_png}")


def _fill_geom(ax, geom, **kw):
    geoms = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
    for g in geoms:
        xs, ys = g.exterior.xy
        ax.fill(xs, ys, edgecolor="white", linewidth=0.3, **kw)


# =============================================================================
#  PASO 4 — Cubicación de la madera
# =============================================================================

def paso4_cubicacion(gdf, parcela, dirs, cfg: Config):
    log("\n========== PASO 4 · CUBICACIÓN DE MADERA ==========")
    dbh, vol = [], []
    for _, row in gdf.iterrows():
        p = cfg.especies[row.especie]
        h = max(row.h_max, 0.1)
        d = p.dbh_k * (h ** p.dbh_p)                 # DBH estimado (cm)
        v = p.vol_a * (d ** p.vol_b) * (h ** p.vol_c)  # volumen fuste (m3)
        dbh.append(d)
        vol.append(v)
    gdf["dbh_cm"] = dbh
    gdf["vol_m3"] = vol

    sup_ha = parcela.geometry.area.sum() / 1e4
    resumen = (gdf.groupby("especie")
               .agg(n_arboles=("vol_m3", "size"),
                    h_media=("h_max", "mean"),
                    h_max=("h_max", "max"),
                    dbh_medio=("dbh_cm", "mean"),
                    vol_total_m3=("vol_m3", "sum"))
               .reset_index())
    resumen["vol_ha_m3"] = resumen["vol_total_m3"] / sup_ha

    vol_total = gdf.vol_m3.sum()
    csv = os.path.join(dirs["cubicacion"], "cubicacion_por_arbol.csv")
    df = gdf.drop(columns="geometry")
    df.to_csv(csv, index=False)
    resumen.to_csv(os.path.join(dirs["cubicacion"], "resumen_cubicacion.csv"),
                   index=False)

    log(f"    Superficie parcela: {sup_ha:.2f} ha")
    for _, r in resumen.iterrows():
        log(f"    · {r.especie:11s}: {int(r.n_arboles):4d} árboles | "
            f"DBH medio {r.dbh_medio:4.1f} cm | "
            f"V {r.vol_total_m3:7.1f} m³ ({r.vol_ha_m3:5.1f} m³/ha)")
    log(f"    VOLUMEN TOTAL ESTIMADO: {vol_total:.1f} m³  "
        f"({vol_total/sup_ha:.1f} m³/ha)")
    log(f"    Detalle -> {csv}")

    # Informe de texto
    _informe(resumen, vol_total, sup_ha,
             os.path.join(dirs["cubicacion"], "informe_cubicacion.txt"), cfg)
    return gdf, resumen


def _informe(resumen, vol_total, sup_ha, out_txt, cfg):
    lines = [
        "=" * 60,
        " INFORME DE CUBICACIÓN — Inventario LiDAR de parcela",
        "=" * 60,
        f" Superficie analizada : {sup_ha:.2f} ha",
        f" Volumen total fuste  : {vol_total:.1f} m³  ({vol_total/sup_ha:.1f} m³/ha)",
        "",
        " Desglose por especie:",
    ]
    for _, r in resumen.iterrows():
        lines.append(f"   - {r.especie}: {int(r.n_arboles)} árboles, "
                     f"{r.vol_total_m3:.1f} m³ ({r.vol_ha_m3:.1f} m³/ha)")
    lines += [
        "",
        " NOTA METODOLÓGICA:",
        "  · DBH estimado desde altura LiDAR (modelo H-D por especie).",
        "  · Volumen por tarifa de doble entrada V=a·DBH^b·H^c (estilo IFN).",
        "  · Coeficientes ORIENTATIVOS — calibrar con tarifas locales/IFN4.",
        "  · El alcornoque se explota por CORCHO; el volumen de fuste aquí",
        "    calculado es maderable de referencia, no producto comercial.",
        "=" * 60,
    ]
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"    Informe -> {out_txt}")


# =============================================================================
#  ORQUESTADOR
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Inventario LiDAR de parcela")
    ap.add_argument("--gml", required=True, help="Límite de parcela (GML/SHP/GPKG)")
    ap.add_argument("--laz", required=True, help="Nube LiDAR PNOA (.laz/.las)")
    ap.add_argument("--out", required=True, help="Carpeta de resultados")
    ap.add_argument("--epsg", type=int, default=CFG.epsg)
    ap.add_argument("--res", type=float, default=CFG.res, help="Resolución CHM (m)")
    ap.add_argument("--altura-min", type=float, default=CFG.altura_min)
    ap.add_argument("--sin-kmeans", action="store_true",
                    help="Clasificar especies solo por reglas (sin KMeans)")
    args = ap.parse_args()

    CFG.epsg = args.epsg
    CFG.res = args.res
    CFG.altura_min = args.altura_min
    CFG.usar_kmeans = not args.sin_kmeans

    log("=" * 60)
    log(" INVENTARIO FORESTAL LiDAR DE PARCELA")
    log("=" * 60)
    dirs = crea_dirs(args.out)

    parcela = leer_parcela(args.gml, CFG.epsg)
    minx, miny, maxx, maxy = parcela.total_bounds
    bbox = (minx - CFG.buffer_parcela, miny - CFG.buffer_parcela,
            maxx + CFG.buffer_parcela, maxy + CFG.buffer_parcela)
    pts = leer_laz(args.laz, bbox, CFG.epsg)

    chm, chm_p, transform, mask, shape_xy = paso1_altura(pts, parcela, dirs, CFG)
    gdf, labels = paso2_copas(chm, transform, mask, dirs, CFG)
    gdf = paso3_especies(gdf, labels, pts, parcela, dirs, transform, CFG)
    gdf, resumen = paso4_cubicacion(gdf, parcela, dirs, CFG)

    # Guarda la configuración usada (trazabilidad / reutilización)
    cfg_dump = {k: v for k, v in asdict(CFG).items() if k != "especies"}
    cfg_dump["especies"] = {k: asdict(v) for k, v in CFG.especies.items()}
    with open(os.path.join(args.out, "config_usada.json"), "w", encoding="utf-8") as f:
        json.dump(cfg_dump, f, indent=2, ensure_ascii=False)

    log("\n" + "=" * 60)
    log(f" PROCESO COMPLETADO. Resultados en: {args.out}")
    log("=" * 60)


if __name__ == "__main__":
    main()
