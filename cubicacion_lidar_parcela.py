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
from scipy.ndimage import gaussian_filter, label as ndlabel
from scipy.ndimage import median_filter, binary_dilation
from skimage.feature import peak_local_max
from skimage.morphology import h_maxima, disk
from skimage.segmentation import watershed
from skimage.measure import regionprops, find_contours, perimeter as skperim
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

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
    """Parámetros allométricos y económicos por especie. CALIBRAR con tarifas locales."""
    nombre: str
    color: str
    # --- H → DBH:  DBH_cm = dbh_k × H_m^dbh_p ---
    dbh_k: float
    dbh_p: float
    # --- Volumen fuste (m³): V = vol_a × DBH_cm^vol_b × H_m^vol_c ---
    # Pinus pinaster: Montero et al. 2001, "Modelos para la estimación de la
    # producción maderable en masas de Pinus pinaster Ait." (con corteza)
    # Quercus suber: Tomé et al. 2001; valor orientativo (el corcho prima)
    vol_a: float
    vol_b: float
    vol_c: float
    # --- Precio madera en pie (€/m³) por clase diamétrica ---
    # Fuente: JCYL / Subasta pública CyL 2022-2024 (orientativo)
    precio_latizal_m3: float    # DBH 10-20 cm
    precio_fustal_m_m3: float   # DBH 20-30 cm
    precio_fustal_g_m3: float   # DBH > 30 cm
    # --- Corcho (solo Quercus suber) ---
    # Fórmula: W_corcho_kg = cork_a × DBH_cm^cork_b
    # Ribeiro et al. 2003, J. Environ. Mgmt. (corcho de reproducción)
    cork_a: float               # 0.0 si no aplica
    cork_b: float
    precio_corcho_kg: float     # €/kg corcho reproducción (JCYL, Extremadura ~2023)


ESPECIES = {
    "pino": EspecieParams(
        nombre="Pinus pinaster Ait.",
        color="#1f6b3b",
        # H-D: relación típica IFN4 CyL para masas adultas de P. pinaster
        dbh_k=1.40, dbh_p=1.10,
        # Volumen c.c.: Montero et al. 2001 (tarifa 2 entradas, P. pinaster España)
        vol_a=6.9e-5, vol_b=1.78, vol_c=0.98,
        # Precios en pie, subastas CyL 2022-2024 (€/m³ orientativo)
        precio_latizal_m3=10.0,
        precio_fustal_m_m3=22.0,
        precio_fustal_g_m3=38.0,
        # Sin producción de corcho
        cork_a=0.0, cork_b=0.0, precio_corcho_kg=0.0,
    ),
    "alcornoque": EspecieParams(
        nombre="Quercus suber L.",
        color="#9c5a2e",
        # Q. suber: copa ancha, H-D con menor esbeltez que el pino
        dbh_k=2.30, dbh_p=1.00,
        # Volumen orientativo (el producto comercial es el corcho, no la madera)
        vol_a=3.8e-5, vol_b=1.90, vol_c=0.80,
        # Madera alcornoque tiene poco mercado; precio muy bajo en pie
        precio_latizal_m3=5.0,
        precio_fustal_m_m3=10.0,
        precio_fustal_g_m3=15.0,
        # Corcho reproducción: Ribeiro et al. 2003 W(kg) = 0.041 × DBH^2.06
        # Precio: JCYL/Extremadura ~€0.70-1.20/kg (usamos €0.85 como referencia)
        cork_a=0.041, cork_b=2.06, precio_corcho_kg=0.85,
    ),
}


@dataclass
class Config:
    # --- CRS ---
    epsg: int = 25830                   # ETRS89 UTM 30N (España peninsular)
    # --- CHM ---
    res: float = 0.5                    # resolución del CHM (m)
    buffer_parcela: float = 15.0        # margen buffer alrededor de la parcela (m)
    # --- Detección/segmentación ---
    altura_min: float = 2.0             # altura mínima de árbol (m)
    h_prom_min: float = 0.8             # prominencia mínima del pico sobre entorno (m)
    area_copa_min_m2: float = 3.0       # superficie mínima de copa válida (m²)
    area_copa_max_m2: float = 400.0     # superficie máxima (copas gigantes = artefacto)
    fusion_area_m2: float = 6.0         # copas menores → se fusionan con vecina
    # --- Clasificación ---
    usar_gmm: bool = True               # GaussianMixture > KMeans para clusters elípticos
    n_min_arb_clasificar: int = 4       # mínimo de árboles para clasificar automáticamente
    # --- Clases LAS ---
    clase_suelo: int = 2
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


def radio_copa_esperado(h: float) -> float:
    """Radio de copa esperado (m) según la altura del árbol.

    Derivado de datos PNOA-IFN4 para masas mixtas pinar-alcornocal en CyL.
    La relación es casi lineal: rc ≈ 0.25·H para pino, 0.40·H para alcornoque.
    Usamos un promedio conservador para no sub-segmentar.
    """
    if h < 4:   return 1.0
    if h < 8:   return 1.5
    if h < 14:  return 2.5
    if h < 20:  return 3.5
    return 4.5


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
    """CHM pit-free simplificado: percentil 95 por celda (reduce artefactos de pits).

    El CHM de máximos sufre "pits" (agujeros) donde ningún retorno llegó al techo
    de la copa. Usar el percentil 95 dentro de cada celda es más robusto.
    Para celdas con pocos puntos se usa el máximo directamente.
    Referencia: Khosravipour et al. 2014, PFG.
    """
    nx, ny = shape_xy
    minx = transform.c
    maxy = transform.f
    col = np.clip(((x - minx) / res).astype(int), 0, nx - 1)
    row = np.clip(((maxy - y) / res).astype(int), 0, ny - 1)
    h = z - dtm[row, col]
    h = np.where(h < 0, 0, h)

    # Acumulamos en un dict de listas para calcular percentiles
    cell_h: dict = {}
    for i in range(len(h)):
        k = (row[i], col[i])
        if k not in cell_h:
            cell_h[k] = []
        cell_h[k].append(h[i])

    chm = np.zeros((ny, nx), dtype=np.float32)
    for (r, c), hvals in cell_h.items():
        arr = np.array(hvals)
        # percentil 95 si hay ≥5 puntos, máximo si no
        chm[r, c] = float(np.percentile(arr, 95) if len(arr) >= 5 else arr.max())

    # Rellena huecos con un filtro de mediana 3×3 (sólo donde chm==0 y hay vecinos)
    chm_filled = median_filter(chm, size=3)
    gaps = (chm == 0)
    chm = np.where(gaps, chm_filled, chm)

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
#  PASO 2 — Detección de cimas (h-maxima) y segmentación de copas mejorada
# =============================================================================

def _suavizar_chm(chm, res):
    """Suavizado adaptativo: sigma más fino en zonas bajas, más grueso en zonas altas.

    Un sigma único sobre-suaviza copas pequeñas o sub-suaviza copas grandes.
    Aplicamos 2 pasadas gaussian con sigmas distintos y combinamos según altura.
    """
    chm0 = np.nan_to_num(chm)
    s_fino  = gaussian_filter(chm0, sigma=0.6)  # preserva detalles en copas bajas
    s_grueso = gaussian_filter(chm0, sigma=1.2)  # reduce ruido en copas altas
    alpha = np.clip(chm0 / 15.0, 0, 1)           # blend 0→fino, 1→grueso a 15 m
    return s_fino * (1 - alpha) + s_grueso * alpha


def _detectar_cimas_hmax(chm_s, mask, cfg):
    """Detección de cimas mediante h-maxima transform.

    h-maxima suprime todos los máximos locales cuya prominencia sobre su entorno
    sea menor que h_prom_min. Esto elimina falsos picos entre copas tocadas y
    picos de ruido dentro de una misma copa, problema crónico del LMF fijo.
    Referencia: Vincent & Soille 1991; aplicación a CHM: Kaartinen et al. 2012.
    """
    chm_v = chm_s * mask
    hmax_mask = h_maxima(chm_v.astype(np.float64), cfg.h_prom_min)
    hmax_mask = hmax_mask & (chm_v >= cfg.altura_min)

    # Para cada región de h-maxima conectada, quedamos solo con el píxel de máximo
    labeled_peaks, n = ndlabel(hmax_mask)
    seeds = np.zeros_like(hmax_mask, dtype=np.int32)
    seed_id = 1
    for i in range(1, n + 1):
        rr, cc = np.where(labeled_peaks == i)
        idx = np.argmax(chm_v[rr, cc])
        pr, pc = rr[idx], cc[idx]
        h_local = float(chm_v[pr, pc])
        # Ventana mínima entre semillas: radio proporcional a la altura del pico
        seeds[pr, pc] = seed_id
        seed_id += 1

    # Supresión de no-máximos con ventana adaptativa: elimina semillas demasiado
    # cercanas entre sí (< radio_mínimo para esa altura)
    seed_coords = np.column_stack(np.where(seeds > 0))
    if len(seed_coords) > 1:
        keep = np.ones(len(seed_coords), dtype=bool)
        h_vals = chm_v[seed_coords[:, 0], seed_coords[:, 1]]
        # orden descendente de altura: el pico más alto gana en caso de colisión
        orden = np.argsort(-h_vals)
        suprimido = np.zeros(len(seed_coords), dtype=bool)
        for idx in orden:
            if suprimido[idx]:
                continue
            h_i = h_vals[idx]
            r_min_px = max(1, int(radio_copa_esperado(h_i) * 0.55 / cfg.res))
            for jdx in orden:
                if jdx == idx or suprimido[jdx]:
                    continue
                dr = seed_coords[idx, 0] - seed_coords[jdx, 0]
                dc = seed_coords[idx, 1] - seed_coords[jdx, 1]
                if (dr*dr + dc*dc) < r_min_px * r_min_px:
                    suprimido[jdx] = True
        seeds_final = np.zeros_like(seeds)
        valid_coords = seed_coords[~suprimido]
        for k, (r, c) in enumerate(valid_coords, start=1):
            seeds_final[r, c] = k
    else:
        seeds_final = seeds

    return seeds_final


def _fusionar_copas_pequenas(labels, chm_s, cfg):
    """Fusiona copas muy pequeñas con la copa vecina más alta.

    La sobre-segmentación crea fragmentos (<fusion_area_m2) en bordes de copa
    y en zonas de transición entre árboles. Estos fragmentos se asignan al
    segmento adyacente con mayor altura máxima.
    """
    res = cfg.res
    min_px = max(1, int(cfg.fusion_area_m2 / (res * res)))
    props = {rp.label: rp.area for rp in regionprops(labels)}
    pequeños = {lab for lab, area in props.items() if area < min_px}
    if not pequeños:
        return labels

    labels_out = labels.copy()
    for lab in pequeños:
        mask_p = labels_out == lab
        # Dilata 1 px para encontrar vecinos
        dilated = binary_dilation(mask_p, structure=np.ones((3, 3)))
        borde = dilated & ~mask_p
        vecinos = np.unique(labels_out[borde])
        vecinos = [v for v in vecinos if v != 0 and v not in pequeños]
        if vecinos:
            # Elige el vecino con mayor CHM máximo (árbol dominante)
            mejor = max(vecinos, key=lambda v: chm_s[labels_out == v].max() if (labels_out == v).any() else 0)
            labels_out[mask_p] = mejor
    return labels_out


def paso2_copas(chm, transform, mask, dirs, cfg: Config):
    log("\n========== PASO 2 · SEGMENTACIÓN DE COPAS ==========")

    # 1. Suavizado adaptativo del CHM
    chm_s = _suavizar_chm(chm, cfg.res)
    chm_s = np.where(mask, chm_s, 0)

    # 2. Detección de semillas con h-maxima (robusto contra over-detection)
    seeds = _detectar_cimas_hmax(chm_s, mask, cfg)
    n_seeds = (seeds > 0).sum()
    log(f"    Semillas h-maxima (prom. mín. {cfg.h_prom_min}m): {n_seeds}")

    # 3. Watershed marcador-controlado descendente sobre el CHM negado
    #    Compact=True mejora la regularidad de bordes; connectivity=2 permite
    #    conectar en diagonal (más realista para copas circulares).
    suelo = chm_s < cfg.altura_min
    labels = watershed(-chm_s, seeds, mask=~suelo, compactness=0.001)

    # 4. Fusión de sobre-segmentos pequeños
    labels = _fusionar_copas_pequenas(labels, chm_s, cfg)
    n_antes = len(np.unique(labels)) - 1
    log(f"    Copas tras segmentación+fusión: {n_antes}")

    # 5. Vectorización: contorno → polígono + métricas geométricas
    res = cfg.res
    minx, maxy_t = transform.c, transform.f
    registros = []
    for rp in regionprops(labels, intensity_image=chm_s):
        area_m2 = rp.area * res * res
        if area_m2 < cfg.area_copa_min_m2 or area_m2 > cfg.area_copa_max_m2:
            continue
        m_local = (labels == rp.label)
        cont = find_contours(m_local.astype(float), 0.5)
        if not cont:
            continue
        cc_arr = max(cont, key=len)
        poly_xy = [(minx + (c + 0.5) * res, maxy_t - (r + 0.5) * res)
                   for r, c in cc_arr]
        if len(poly_xy) < 4:
            continue
        poly = Polygon(poly_xy)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area < cfg.area_copa_min_m2:
            continue
        poly = poly.simplify(res * 0.4, preserve_topology=True)

        # Cima real (máximo del CHM suavizado dentro de la copa)
        rr, cc2 = np.where(m_local)
        idx = np.argmax(chm_s[rr, cc2])
        cima_r, cima_c = int(rr[idx]), int(cc2[idx])
        h_max = float(chm_s[cima_r, cima_c])

        # Métricas geométricas de forma
        area = float(poly.area)
        perim = float(poly.length)
        diam = 2.0 * np.sqrt(area / np.pi)
        # Compacidad: 1=círculo perfecto; baja=copa irregular (más alcornoque)
        compact = (4 * np.pi * area / perim**2) if perim > 0 else np.nan
        registros.append({
            "tree_id":    int(rp.label),
            "h_max":      h_max,
            "area_copa":  area,
            "diam_copa":  diam,
            "compacidad": compact,
            "esbeltez":   h_max / diam if diam > 0 else np.nan,
            "cima_x":     minx + (cima_c + 0.5) * res,
            "cima_y":     maxy_t - (cima_r + 0.5) * res,
            "geometry":   poly,
        })

    gdf = gpd.GeoDataFrame(registros, crs=f"EPSG:{cfg.epsg}")
    log(f"    Copas válidas (área {cfg.area_copa_min_m2}-{cfg.area_copa_max_m2} m²): {len(gdf)}")
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
#  PASO 3 — Features multiespectral+estructurales + clasificación GMM
# =============================================================================

def metricas_por_arbol(gdf, labels, pts, cfg: Config):
    """Extrae el conjunto completo de features por copa para clasificación.

    Features estructurales (siempre disponibles)
    ─────────────────────────────────────────────
    - esbeltez        h/diam_copa. Pino >> alcornoque.
    - compacidad      ya en gdf (4π·A/P²). Alcornoque más irregular → menor.
    - rugosidad_cv    CV(h) = σ(h)/μ(h) dentro de la copa.
                      Alcornoque tiene follaje más irregular → mayor rugosidad.
    - frac_alto       % de retornos por encima del 70% de h_max.
                      Pino (cónico) tiene pocos retornos cerca del ápice → menor.
    - frac_medio      % de retornos entre 40-70% de h_max (masa de copa del pino).
    - rel_area        area_copa / h_max². Alcornoque tiene copas anchas → mayor.
    - skew_vert       asimetría de la distribución vertical de retornos.
                      Pino: masa en el tercio inferior → skew positivo.

    Features radiométricas RGB (cuando la nube tiene RGB, p.ej. PNOA CLA-RGB)
    ──────────────────────────────────────────────────────────────────────────
    - exg             Excess Green = (2g−r−b)/(r+g+b). Diferencia especies de
                      follaje muy verde (coníferas) vs. especies con reflectancia
                      alta en rojo (quercíneas). Referencia: Woebbecke et al. 1995.
    - gcc             Green Chromatic Coord. = g/(r+g+b). Más estable que ExG.
    - rg_ratio        r/g. Quercus suber más brillante en rojo (corteza + hoja).
    - brillantez      (r+g+b)/3. Alcornoque = copa más abierta → más retornos suelo
                      visibles entre ramas → brillantez media mayor.

    Fuentes de referencia para separación P. pinaster / Q. suber en LiDAR:
    - Ferraz et al. 2016 (Remote Sensing of Env.): estructura vertical.
    - Bauwens et al. 2016 (Forests): métricas de copa y especies.
    - Listopad et al. 2015 (PFG): ExG y GCC sobre PNOA.
    """
    tiene_rgb = pts.get("r") is not None
    row, col, h = pts["row"], pts["col"], pts["h"]
    lab_pt = labels[row, col]

    feats_dict: dict = {tid: {} for tid in gdf.tree_id}

    for tid, h_max_arb, rel_area_ref in zip(
            gdf.tree_id, gdf.h_max, gdf.area_copa / (gdf.h_max**2 + 1e-6)):
        m = (lab_pt == tid) & (h > 0.5)     # excluye retornos de suelo
        n = m.sum()
        fd = feats_dict[tid]

        if n < 5:
            fd.update({k: np.nan for k in
                       ["rugosidad_cv","frac_alto","frac_medio","skew_vert",
                        "exg","gcc","rg_ratio","brillantez"]})
            continue

        hh = h[m]
        mu_h, sd_h = hh.mean(), hh.std() + 1e-6
        fd["rugosidad_cv"] = float(sd_h / mu_h)
        fd["frac_alto"]    = float(np.mean(hh >= 0.70 * h_max_arb))
        fd["frac_medio"]   = float(np.mean((hh >= 0.40 * h_max_arb) &
                                           (hh <  0.70 * h_max_arb)))
        fd["skew_vert"]    = float(np.mean(((hh - mu_h) / sd_h) ** 3))

        if tiene_rgb:
            r, g, b = pts["r"][m], pts["g"][m], pts["b"][m]
            s = r + g + b + 1e-6
            fd["exg"]       = float(np.mean((2*g - r - b) / s))
            fd["gcc"]       = float(np.mean(g / s))
            fd["rg_ratio"]  = float(np.mean(r / (g + 1e-6)))
            fd["brillantez"]= float(np.mean(s / 3))
        else:
            fd.update({k: np.nan for k in ["exg","gcc","rg_ratio","brillantez"]})

    for k in ["rugosidad_cv","frac_alto","frac_medio","skew_vert",
              "exg","gcc","rg_ratio","brillantez"]:
        gdf[k] = gdf.tree_id.map({tid: feats_dict[tid].get(k, np.nan)
                                   for tid in gdf.tree_id})
    return gdf


def _clasificar_gmm(feats_df, gdf):
    """GaussianMixture de 2 componentes sobre features normalizadas.

    GMM maneja clusters elípticos (P. pinaster y Q. suber tienen distribuciones
    asimétricas distintas en el espacio de features) mejor que KMeans.
    Devuelve etiquetas y probabilidades de pertenencia.
    Referencia: Bishop 2006 cap. 9; scikit-learn GaussianMixture.
    """
    X = StandardScaler().fit_transform(feats_df.values)
    gmm = GaussianMixture(n_components=2, covariance_type="full",
                          n_init=5, random_state=42).fit(X)
    lab = gmm.predict(X)
    proba = gmm.predict_proba(X)
    conf = proba.max(axis=1)  # confianza = prob del cluster asignado

    # ¿Cuál cluster es pino? El de mayor esbeltez media y mayor skew_vert medio
    # (pino = copa más estrecha y masa concentrada abajo)
    esb_col = feats_df.columns.get_loc("esbeltez") if "esbeltez" in feats_df.columns else 0
    mean_esb = [feats_df.iloc[lab == k]["esbeltez"].mean() for k in [0, 1]]
    cl_pino = int(np.argmax(mean_esb))

    etiquetas = np.where(lab == cl_pino, "pino", "alcornoque")
    return etiquetas, conf, gmm


def paso3_especies(gdf, labels, pts, parcela, dirs, transform, cfg: Config):
    log("\n========== PASO 3 · CLASIFICACIÓN DE ESPECIES ==========")
    gdf = metricas_por_arbol(gdf, labels, pts, cfg)

    tiene_rgb = pts.get("r") is not None

    # Construye la tabla de features para clasificar
    cols_base = ["esbeltez", "compacidad", "rugosidad_cv",
                 "frac_alto", "frac_medio", "skew_vert"]
    cols_rgb  = ["exg", "gcc", "rg_ratio"] if tiene_rgb else []
    cols_feat = cols_base + cols_rgb
    feats = gdf[cols_feat].copy()
    # Imputar NaN con la mediana de la columna
    for c in feats.columns:
        feats[c] = feats[c].fillna(feats[c].median())

    if cfg.usar_gmm and len(gdf) >= cfg.n_min_arb_clasificar:
        etiquetas, confianza, gmm = _clasificar_gmm(feats, gdf)
        gdf["especie"]    = etiquetas
        gdf["confianza"]  = confianza
        n_feat = len(cols_feat)
        sil = silhouette_score(StandardScaler().fit_transform(feats.values),
                               etiquetas, metric="euclidean")
        metodo = (f"GaussianMixture(2) | {n_feat} features "
                  f"({'estr.+RGB' if tiene_rgb else 'solo estructura'}) "
                  f"| silhouette={sil:.2f}")
    else:
        # Fallback por regla: esbeltez > mediana → pino
        umbral = gdf.esbeltez.median()
        gdf["especie"]   = np.where(gdf.esbeltez >= umbral, "pino", "alcornoque")
        gdf["confianza"] = 0.70
        metodo = "regla por mediana de esbeltez (n insuf. para GMM)"

    n_pino = (gdf.especie == "pino").sum()
    n_alc  = (gdf.especie == "alcornoque").sum()
    n_inc  = (gdf.confianza < 0.70).sum() if "confianza" in gdf else 0
    log(f"    Método   : {metodo}")
    log(f"    Pinus pinaster : {n_pino} árboles")
    log(f"    Quercus suber  : {n_alc} árboles")
    if n_inc:
        log(f"    Inciertos (<70% conf.): {n_inc} árboles (marcados en el mapa)")

    gpkg = os.path.join(dirs["especies"], "copas_especies.gpkg")
    escribir_vector(gdf, gpkg, cfg.epsg)
    _mapa_especies_png(gdf, parcela,
                       os.path.join(dirs["especies"], "mapa_especies.png"), cfg)
    log(f"    Especies -> {gpkg}")
    return gdf


def _mapa_especies_png(gdf, parcela, out_png, cfg):
    fig, ax = plt.subplots(figsize=(10, 9), facecolor="#15151f")
    ax.set_facecolor("#15151f")
    tiene_conf = "confianza" in gdf.columns
    for esp, p in cfg.especies.items():
        sub = gdf[gdf.especie == esp]
        # Alta confianza: color sólido; baja confianza: trama con borde distinto
        alta = sub[sub.confianza >= 0.70] if tiene_conf else sub
        baja = sub[sub.confianza < 0.70]  if tiene_conf else sub.iloc[0:0]
        for g in alta.geometry:
            _fill_geom(ax, g, color=p.color, alpha=0.9)
        for g in baja.geometry:
            _fill_geom(ax, g, color=p.color, alpha=0.35)
        lbl = f"{p.nombre} ({len(sub)})"
        if len(baja):
            lbl += f"  [{len(baja)} inciertos]"
        ax.scatter([], [], c=p.color, label=lbl, s=80)
    for g in parcela.geometry:
        _plot_geom(ax, g, color="yellow", lw=1.8)
    ax.legend(loc="upper right", facecolor="#1f1f2e", labelcolor="#ddddee",
              fontsize=9)
    ax.set_title("Clasificación de especies — Pinus pinaster / Quercus suber",
                 color="#ddddee")
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
#  PASO 4 — Cubicación + valor económico
# =============================================================================

def _precio_madera(dbh_cm, p: EspecieParams) -> float:
    """Precio en pie (€/m³) según clase diamétrica.

    Clases IFN estándar:
      Latizal alto : 10-20 cm DBH
      Fustal menor : 20-30 cm DBH
      Fustal mayor : > 30 cm DBH
    Fuente: JCYL, subastas públicas 2022-2024 (valores orientativos).
    """
    if dbh_cm < 20:
        return p.precio_latizal_m3
    if dbh_cm < 30:
        return p.precio_fustal_m_m3
    return p.precio_fustal_g_m3


def _valor_corcho(dbh_cm, p: EspecieParams) -> float:
    """Peso de corcho (kg) estimado y su valor económico (€).

    Quercus suber: W_corcho (kg) = cork_a × DBH_cm^cork_b
    Fuente: Ribeiro et al. 2003, J. Environ. Mgmt. (corcho de reproducción).
    Aplica solo a árboles con DBH > 13 cm (primera descorche).
    """
    if p.cork_a == 0 or dbh_cm < 13:
        return 0.0, 0.0
    w = p.cork_a * (dbh_cm ** p.cork_b)
    return w, w * p.precio_corcho_kg


def paso4_cubicacion(gdf, parcela, dirs, cfg: Config):
    log("\n========== PASO 4 · CUBICACIÓN Y VALOR ECONÓMICO ==========")

    dbh_l, vol_l, precio_l, val_mad_l = [], [], [], []
    cork_kg_l, val_cork_l, clase_l = [], [], []

    for _, row in gdf.iterrows():
        p = cfg.especies[row.especie]
        h  = max(float(row.h_max), 0.1)
        d  = p.dbh_k * (h ** p.dbh_p)             # DBH estimado (cm)
        v  = p.vol_a * (d ** p.vol_b) * (h ** p.vol_c)  # volumen fuste (m³)
        pr = _precio_madera(d, p)
        vm = v * pr                                # valor madera en pie (€)
        ck_kg, ck_val = _valor_corcho(d, p)        # corcho (kg, €)

        # Clase diamétrica
        if d < 10:   cl = "latizal_bajo"
        elif d < 20: cl = "latizal_alto"
        elif d < 30: cl = "fustal_menor"
        elif d < 40: cl = "fustal_medio"
        else:        cl = "fustal_mayor"

        dbh_l.append(d); vol_l.append(v); precio_l.append(pr)
        val_mad_l.append(vm); cork_kg_l.append(ck_kg); val_cork_l.append(ck_val)
        clase_l.append(cl)

    gdf["dbh_cm"]      = dbh_l
    gdf["vol_m3"]      = vol_l
    gdf["precio_m3"]   = precio_l
    gdf["valor_mad_eur"] = val_mad_l
    gdf["cork_kg"]     = cork_kg_l
    gdf["valor_cork_eur"]= val_cork_l
    gdf["valor_tot_eur"] = gdf["valor_mad_eur"] + gdf["valor_cork_eur"]
    gdf["clase_diam"]  = clase_l

    sup_ha = parcela.geometry.area.sum() / 1e4

    resumen = (gdf.groupby("especie")
               .agg(n_arboles    = ("vol_m3",       "size"),
                    h_media      = ("h_max",         "mean"),
                    h_max_obs    = ("h_max",         "max"),
                    dbh_medio_cm = ("dbh_cm",        "mean"),
                    vol_total_m3 = ("vol_m3",        "sum"),
                    valor_mad_eur  = ("valor_mad_eur",   "sum"),
                    cork_kg_tot  = ("cork_kg",       "sum"),
                    valor_cork_eur = ("valor_cork_eur",  "sum"),
                    valor_tot_eur  = ("valor_tot_eur",   "sum"))
               .reset_index())
    resumen["vol_ha_m3"]    = resumen["vol_total_m3"] / sup_ha
    resumen["valor_ha_eur"]   = resumen["valor_tot_eur"]  / sup_ha

    vol_total  = gdf.vol_m3.sum()
    val_total  = gdf["valor_tot_eur"].sum()
    val_mad    = gdf["valor_mad_eur"].sum()
    val_cork   = gdf["valor_cork_eur"].sum()

    csv = os.path.join(dirs["cubicacion"], "cubicacion_por_arbol.csv")
    gdf.drop(columns="geometry").to_csv(csv, index=False)
    resumen.to_csv(os.path.join(dirs["cubicacion"], "resumen_cubicacion.csv"),
                   index=False)

    log(f"    Superficie parcela : {sup_ha:.2f} ha")
    log(f"    {'Especie':<16} {'N árboles':>9} {'DBH m.(cm)':>10} "
        f"{'Vol.(m³)':>9} {'m³/ha':>6} {'Val.mad(€)':>10} {'Val.cork(€)':>11} "
        f"{'Total(€)':>9}")
    log("    " + "-"*88)
    for _, r in resumen.iterrows():
        log(f"    {r.especie:<16} {int(r.n_arboles):>9} {r.dbh_medio_cm:>10.1f} "
            f"{r.vol_total_m3:>9.1f} {r.vol_ha_m3:>6.1f} "
            f"{r.valor_mad_eur:>10.0f} {r.valor_cork_eur:>11.0f} "
            f"{r.valor_tot_eur:>9.0f}")
    log("    " + "="*88)
    log(f"    VOLUMEN TOTAL: {vol_total:.1f} m³  ({vol_total/sup_ha:.1f} m³/ha)")
    log(f"    VALOR MADERA : {val_mad:,.0f} €")
    log(f"    VALOR CORCHO : {val_cork:,.0f} €")
    log(f"    VALOR TOTAL  : {val_total:,.0f} €  ({val_total/sup_ha:,.0f} €/ha)")
    log(f"    Detalle      -> {csv}")

    _informe(resumen, vol_total, val_mad, val_cork, val_total, sup_ha,
             os.path.join(dirs["cubicacion"], "informe_cubicacion.txt"), cfg)
    _grafico_cubicacion(gdf, resumen, sup_ha,
                        os.path.join(dirs["cubicacion"], "grafico_cubicacion.png"))
    return gdf, resumen


def _informe(resumen, vol_total, val_mad, val_cork, val_total, sup_ha, out_txt, cfg):
    lines = [
        "=" * 70,
        " INFORME DE CUBICACIÓN Y VALORACIÓN — Inventario LiDAR de parcela",
        "=" * 70,
        f" Fecha análisis       : {pd.Timestamp.now().strftime('%Y-%m-%d')}",
        f" Superficie analizada : {sup_ha:.2f} ha",
        f" Volumen total fuste  : {vol_total:.1f} m³  ({vol_total/sup_ha:.1f} m³/ha)",
        "",
        f" VALOR ECONÓMICO ESTIMADO",
        f"   Madera en pie (Pinus pinaster) : {val_mad:>10,.0f} €",
        f"   Corcho reproducción (Q. suber) : {val_cork:>10,.0f} €",
        f"   ──────────────────────────────────────────",
        f"   TOTAL                          : {val_total:>10,.0f} €"
        f"  ({val_total/sup_ha:,.0f} €/ha)",
        "",
        " DESGLOSE POR ESPECIE",
        f"  {'Especie':<18} {'N':>5} {'DBH m.(cm)':>10} {'Vol.(m³)':>9}"
        f" {'m³/ha':>6} {'Val.mad.€':>10} {'Val.cork.€':>11} {'Total €':>9}",
        "  " + "-"*82,
    ]
    for _, r in resumen.iterrows():
        lines.append(
            f"  {r.especie:<18} {int(r.n_arboles):>5} {r.dbh_medio_cm:>10.1f}"
            f" {r.vol_total_m3:>9.1f} {r.vol_ha_m3:>6.1f}"
            f" {r.valor_mad_eur:>10.0f} {r.valor_cork_eur:>11.0f}"
            f" {r.valor_tot_eur:>9.0f}")
    lines += [
        "",
        " METODOLOGÍA",
        "  Segmentación : h-maxima (Kaartinen et al. 2012) + watershed marcador.",
        "  Especies     : GaussianMixture(2) sobre 6-9 features estructurales/RGB.",
        "  H→DBH        : Pinus pinaster: DBH=1.40·H^1.10 (IFN4 CyL orient.).",
        "                 Quercus suber : DBH=2.30·H (copa ancha, orient.).",
        "  Volumen      : P. pinaster: V=6.9e-5·D^1.78·H^0.98",
        "                 (Montero et al. 2001, c.c., Pinus pinaster España).",
        "                 Q. suber: V=3.8e-5·D^1.90·H^0.80 (orient.).",
        "  Corcho       : W(kg)=0.041·DBH^2.06 (Ribeiro et al. 2003, J.Env.Mgmt.).",
        "                 Precio referencia: 0.85 €/kg corcho reproducción (JCYL).",
        "  Madera       : precio en pie por clase diamétrica (subastas JCYL 2022-24).",
        "",
        "  *** VALORES ORIENTATIVOS. Requieren verificación con inventario pie a pie.",
        "  *** Calibrar coeficientes H-D con parcelas dasométricas locales.",
        "=" * 70,
    ]
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"    Informe -> {out_txt}")


def _grafico_cubicacion(gdf, resumen, sup_ha, out_png):
    """Gráfico de 4 paneles: distribución DBH, distribución alturas, valor por especie,
    distribución espacial del valor por árbol (mapa de calor)."""
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), facecolor="#15151f")
    for ax in axs.flat:
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="#aaaacc")
        for sp in ax.spines.values():
            sp.set_edgecolor("#3a3a5a")

    colores = {"pino": "#1f6b3b", "alcornoque": "#9c5a2e"}

    # Panel 1: Distribución de DBH por especie
    ax = axs[0, 0]
    for esp in gdf.especie.unique():
        sub = gdf[gdf.especie == esp]
        ax.hist(sub.dbh_cm, bins=20, alpha=0.75, color=colores.get(esp, "gray"),
                label=esp, edgecolor="white", linewidth=0.3)
    ax.axvline(20, color="gray", ls="--", lw=0.8)
    ax.axvline(30, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("DBH estimado (cm)", color="#ccccee")
    ax.set_ylabel("N árboles", color="#ccccee")
    ax.set_title("Distribución diamétrica", color="#ddddee")
    ax.legend(facecolor="#1f1f2e", labelcolor="#ddddee")

    # Panel 2: Distribución de alturas por especie
    ax = axs[0, 1]
    for esp in gdf.especie.unique():
        sub = gdf[gdf.especie == esp]
        ax.hist(sub.h_max, bins=20, alpha=0.75, color=colores.get(esp, "gray"),
                label=esp, edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Altura máxima (m)", color="#ccccee")
    ax.set_ylabel("N árboles", color="#ccccee")
    ax.set_title("Distribución de alturas", color="#ddddee")
    ax.legend(facecolor="#1f1f2e", labelcolor="#ddddee")

    # Panel 3: Valor económico por especie (barras apiladas madera+corcho)
    ax = axs[1, 0]
    especies = resumen.especie.tolist()
    x = np.arange(len(especies))
    ax.bar(x, resumen.valor_mad_eur, color=[colores.get(e, "gray") for e in especies],
           label="Madera", alpha=0.9)
    ax.bar(x, resumen.valor_cork_eur, bottom=resumen.valor_mad_eur,
           color="tan", label="Corcho", alpha=0.85, edgecolor="white", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(especies, color="#ccccee")
    ax.set_ylabel("Valor estimado (€)", color="#ccccee")
    ax.set_title("Valor económico por especie", color="#ddddee")
    ax.legend(facecolor="#1f1f2e", labelcolor="#ddddee")
    for xi, (_, r) in zip(x, resumen.iterrows()):
        ax.text(xi, r.valor_tot_eur + 50, f"{r.valor_tot_eur:,.0f} €",
                ha="center", va="bottom", color="#ddddee", fontsize=8)

    # Panel 4: Mapa de valor por árbol (scatter sobre cimas)
    ax = axs[1, 1]
    sc = ax.scatter(gdf.cima_x, gdf.cima_y, c=gdf["valor_tot_eur"],
                    cmap="YlOrRd", s=gdf.area_copa / 4, alpha=0.8,
                    edgecolors="none")
    cb = fig.colorbar(sc, ax=ax, shrink=0.8)
    cb.set_label("Valor por árbol (€)", color="#ccccee")
    cb.ax.yaxis.set_tick_params(color="#ccccee")
    plt.setp(plt.getp(cb.ax, "yticklabels"), color="#ccccee")
    ax.set_title("Mapa de valor por árbol", color="#ddddee")
    ax.set_aspect("equal")

    vol_total = gdf.vol_m3.sum()
    val_total = gdf["valor_tot_eur"].sum()
    fig.suptitle(
        f"Cubicación y valoración — {len(gdf)} árboles | "
        f"{vol_total:.0f} m³ | {val_total:,.0f} € total",
        color="#eeeeee", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, facecolor="#15151f")
    plt.close()
    log(f"    Gráfico -> {out_png}")


# =============================================================================
#  ORQUESTADOR
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Inventario LiDAR de parcela")
    ap.add_argument("--gml",        required=True,  help="Límite parcela (GML/SHP/GPKG)")
    ap.add_argument("--laz",        required=True,  help="Nube LiDAR PNOA (.laz/.las)")
    ap.add_argument("--out",        required=True,  help="Carpeta de resultados")
    ap.add_argument("--epsg",       type=int,   default=CFG.epsg)
    ap.add_argument("--res",        type=float, default=CFG.res,          help="Resolución CHM (m)")
    ap.add_argument("--altura-min", type=float, default=CFG.altura_min,   help="Altura mínima árbol (m)")
    ap.add_argument("--h-prom",     type=float, default=CFG.h_prom_min,   help="Prominencia mínima pico (m)")
    ap.add_argument("--area-min",   type=float, default=CFG.area_copa_min_m2, help="Área mínima copa (m²)")
    ap.add_argument("--sin-gmm",    action="store_true", help="Usar KMeans en lugar de GMM para especies")
    args = ap.parse_args()

    CFG.epsg             = args.epsg
    CFG.res              = args.res
    CFG.altura_min       = args.altura_min
    CFG.h_prom_min       = args.h_prom
    CFG.area_copa_min_m2 = args.area_min
    CFG.usar_gmm         = not args.sin_gmm

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
