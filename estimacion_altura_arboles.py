#!/usr/bin/env python3
"""
Estimación de alturas de árboles en parcelas usando LiDAR PNOA (IGN España).

Flujo de trabajo:
  1. Lee el shapefile de parcelas (ETRS89 UTM 30N / EPSG:25830)
  2. Descarga MDT (terreno) desde el WCS del IGN
  3. Descarga tiles LAZ del PNOA 2ª cobertura desde el CNIG
  4. Genera DSM desde primeros retornos del LiDAR
  5. Calcula CHM = DSM − MDT (altura normalizada sobre el suelo)
  6. Detecta árboles individuales: máximos locales + segmentación watershed
  7. Exporta CHM.tif, cimas_arboles.gpkg, copas_arboles.gpkg, resumen.csv

Uso:
    python estimacion_altura_arboles.py --shp ruta/al/parcelas.shp

Dependencias:
    pip install geopandas rasterio scipy scikit-image laspy requests pyproj
"""

import argparse
import io
import json
import math
import os
import struct
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# ── Forzar reconstrucción del .shx si falta
os.environ["SHAPE_RESTORE_SHX"] = "YES"

# ── Imports espaciales obligatorios ───────────────────────────────────────────
try:
    import geopandas as gpd
    from pyproj import CRS, Transformer
    from shapely.geometry import Point, box, mapping
    from shapely.ops import unary_union
except ImportError:
    sys.exit("Falta geopandas/pyproj/shapely.  pip install geopandas pyproj shapely")

try:
    import rasterio
    import rasterio.warp
    from rasterio.enums import Resampling
    from rasterio.features import geometry_mask
    from rasterio.mask import mask as rio_mask
    from rasterio.transform import from_bounds
except ImportError:
    sys.exit("Falta rasterio.  pip install rasterio")

try:
    import requests as _requests
    HAS_REQUESTS = True
except ImportError:
    import urllib.request as _urllib
    HAS_REQUESTS = False

try:
    from scipy import ndimage
    from scipy.ndimage import label, maximum_filter, minimum_filter
    HAS_SCIPY = True
except ImportError:
    sys.exit("Falta scipy.  pip install scipy")

try:
    from skimage.feature import peak_local_max
    from skimage.filters import gaussian
    from skimage.segmentation import watershed
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

try:
    import laspy
    HAS_LASPY = True
except ImportError:
    HAS_LASPY = False

# ── Parámetros configurables ──────────────────────────────────────────────────
EPSG_WORK        = 25830        # ETRS89 UTM 30N (CRS nativo del PNOA)
RESOLUTION_M     = 1.0          # Resolución del CHM en metros
MIN_TREE_HT_M    = 2.0          # Altura mínima para considerar árbol (m)
MAX_TREE_HT_M    = 60.0         # Altura máxima razonable (m)
SMOOTH_SIGMA     = 1.5          # Sigma del suavizado Gaussiano pre-detección
MIN_CROWN_AREA   = 1.5          # Área mínima de copa (m²)
BUFFER_M         = 50.0         # Buffer alrededor de las parcelas para descargar
TILE_SIZE_M      = 2000         # Tamaño de tile PNOA en metros

WCS_URL = "https://servicios.idee.es/wcs-inspire/mdt"
CNIG_CATALOG_URL = (
    "https://centrodedescargas.cnig.es/CentroDescargas/"
    "consultaCatalogoDatosRest.py"
)
CNIG_DOWNLOAD_URL = (
    "https://centrodedescargas.cnig.es/CentroDescargas/dameRutaFichero"
)


# ── Utilidades de descarga ────────────────────────────────────────────────────

def _get(url: str, params: dict = None, timeout: int = 120) -> bytes:
    """Descarga bytes desde una URL, usando requests o urllib como fallback."""
    if HAS_REQUESTS:
        resp = _requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.content
    else:
        if params:
            url = url + "?" + "&".join(f"{k}={v}" for k, v in params.items())
        with _urllib.urlopen(url, timeout=timeout) as r:
            return r.read()


# ── Lectura del shapefile ─────────────────────────────────────────────────────

def leer_shapefile(shp_path: Path) -> gpd.GeoDataFrame:
    """
    Lee el shapefile de parcelas. Si le falta el .prj asume EPSG:25830 (PNOA).
    Si le falta el .shx lo reconstruye (SHAPE_RESTORE_SHX=YES).
    """
    gdf = gpd.read_file(str(shp_path))
    if gdf.crs is None:
        print(f"  [aviso] Sin .prj → asignando CRS EPSG:{EPSG_WORK} (ETRS89 UTM 30N)")
        gdf = gdf.set_crs(epsg=EPSG_WORK)
    elif gdf.crs.to_epsg() != EPSG_WORK:
        print(f"  [info] Reproyectando de {gdf.crs.to_epsg()} a EPSG:{EPSG_WORK}")
        gdf = gdf.to_crs(epsg=EPSG_WORK)
    print(f"  Parcelas leídas: {len(gdf)} | Bounds: {gdf.total_bounds.round(1)}")
    return gdf


def bbox_con_buffer(gdf: gpd.GeoDataFrame, buffer: float = BUFFER_M) -> Tuple[float, ...]:
    """Devuelve (xmin, ymin, xmax, ymax) con un buffer alrededor de la extensión."""
    xmin, ymin, xmax, ymax = gdf.total_bounds
    return (xmin - buffer, ymin - buffer, xmax + buffer, ymax + buffer)


# ── Descarga MDT (terreno) desde IGN WCS ─────────────────────────────────────

def descargar_mdt(bbox: Tuple, out_path: Path, coverage: str = "Elevacion:MDT01") -> Path:
    """
    Descarga el MDT (Modelo Digital del Terreno) del IGN vía WCS 2.0.1.

    coverage:
        'Elevacion:MDT01' → MDT a 1 m (recomendado para árboles)
        'Elevacion:MDT05' → MDT a 5 m (más ligero, menos detalle)
    """
    xmin, ymin, xmax, ymax = bbox
    crs_uri = f"http://www.opengis.net/def/crs/EPSG/0/{EPSG_WORK}"

    params = {
        "SERVICE": "WCS",
        "VERSION": "2.0.1",
        "REQUEST": "GetCoverage",
        "COVERAGEID": coverage,
        "SUBSET": [
            f"X({xmin:.2f},{xmax:.2f})",
            f"Y({ymin:.2f},{ymax:.2f})",
        ],
        "SUBSETTINGCRS": crs_uri,
        "FORMAT": "image/tiff",
    }

    # requests no serializa lista en SUBSET → construimos URL manual
    qs = (
        f"SERVICE=WCS&VERSION=2.0.1&REQUEST=GetCoverage"
        f"&COVERAGEID={coverage}"
        f"&SUBSET=X({xmin:.2f},{xmax:.2f})"
        f"&SUBSET=Y({ymin:.2f},{ymax:.2f})"
        f"&SUBSETTINGCRS={crs_uri}"
        f"&FORMAT=image/tiff"
    )
    url = f"{WCS_URL}?{qs}"
    print(f"  Descargando MDT ({coverage}) …")
    data = _get(url)

    if data[:4] != b'II*\x00' and data[:4] != b'MM\x00*':
        # No es TIFF → posible error XML
        preview = data[:500].decode("utf-8", errors="replace")
        raise RuntimeError(f"WCS no devolvió TIFF:\n{preview}")

    out_path.write_bytes(data)
    print(f"  MDT guardado → {out_path} ({len(data)//1024} KB)")
    return out_path


# ── Descarga tiles LAZ del PNOA desde CNIG ────────────────────────────────────

def _tiles_necesarios(bbox: Tuple, tile_size: int = TILE_SIZE_M) -> List[Tuple[int, int]]:
    """Devuelve lista de esquinas (x_ll, y_ll) de los tiles que cubren bbox."""
    xmin, ymin, xmax, ymax = bbox
    xs = range(int(math.floor(xmin / tile_size)) * tile_size,
               int(math.floor(xmax / tile_size)) * tile_size + 1,
               tile_size)
    ys = range(int(math.floor(ymin / tile_size)) * tile_size,
               int(math.floor(ymax / tile_size)) * tile_size + 1,
               tile_size)
    return [(x, y) for x in xs for y in ys]


def _buscar_tiles_cnig(bbox: Tuple) -> List[dict]:
    """
    Consulta el catálogo del CNIG para obtener los tiles LAZ del PNOA 2ª cobertura
    que intersectan la bbox.  Devuelve lista de dicts con 'nombre', 'url'.
    """
    xmin, ymin, xmax, ymax = bbox
    # Convertir a geográficas (EPSG:4258 / 4326) para el catálogo
    t = Transformer.from_crs(EPSG_WORK, 4258, always_xy=True)
    lon_min, lat_min = t.transform(xmin, ymin)
    lon_max, lat_max = t.transform(xmax, ymax)

    params = {
        "codProducto": "LIDAR-PNOA-2",
        "lon": f"{(lon_min + lon_max)/2:.6f}",
        "lat": f"{(lat_min + lat_max)/2:.6f}",
        "radio": "5",          # radio de búsqueda en km
    }
    print(f"  Consultando catálogo CNIG (lon={params['lon']}, lat={params['lat']}) …")
    try:
        data = _get(CNIG_CATALOG_URL, params=params, timeout=30)
        tiles = json.loads(data)
        return tiles if isinstance(tiles, list) else []
    except Exception as exc:
        print(f"  [aviso] Catálogo CNIG no disponible: {exc}")
        return []


def descargar_tiles_laz(bbox: Tuple, out_dir: Path) -> List[Path]:
    """
    Descarga los archivos LAZ del PNOA 2ª cobertura para la bbox dada.
    Devuelve lista de rutas a los archivos descargados.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    archivos = []

    tiles_info = _buscar_tiles_cnig(bbox)
    if not tiles_info:
        # Fallback: intentar tiles por nomenclatura estándar
        print("  Usando nomenclatura estándar de tiles 2 km × 2 km …")
        archivos = _descargar_tiles_por_nomenclatura(bbox, out_dir)
    else:
        for tile in tiles_info:
            url = tile.get("urlDescarga") or tile.get("url")
            nombre = tile.get("nombre", "tile.laz")
            if not url:
                continue
            dest = out_dir / nombre
            if dest.exists():
                print(f"  Ya existe: {dest.name}")
                archivos.append(dest)
                continue
            try:
                print(f"  Descargando {nombre} …")
                data = _get(url, timeout=300)
                # Si es ZIP, extraemos el LAZ interior
                if data[:2] == b'PK':
                    with zipfile.ZipFile(io.BytesIO(data)) as zf:
                        for member in zf.namelist():
                            if member.lower().endswith((".laz", ".las")):
                                dest = out_dir / Path(member).name
                                dest.write_bytes(zf.read(member))
                                archivos.append(dest)
                else:
                    dest.write_bytes(data)
                    archivos.append(dest)
            except Exception as exc:
                print(f"  [aviso] No se pudo descargar {nombre}: {exc}")

    return archivos


def _nombre_tile_pnoa(x_ll: int, y_ll: int) -> str:
    """
    Construye el nombre de archivo LAZ del PNOA 2ª cobertura usando la
    nomenclatura del IGN: PNOA_<año>_<hoja>_<x>_<y>_ORT-CLA-APP_h02.laz
    La hoja MTN50 se aproxima según la hoja 1:50 000 del IGN.
    """
    # Esquina inferior-izquierda → aproximación al nombre de hoja MTN50
    # La hoja MTN50 depende de la ubicación exacta; aquí usamos el código
    # de la hoja a partir de las coordenadas geográficas del tile.
    t = Transformer.from_crs(EPSG_WORK, 4258, always_xy=True)
    lon, lat = t.transform(x_ll, y_ll)

    # Hoja MTN50: cada hoja cubre ~0.5° lon × 0.333° lat
    # Fila (1 = norte, desde lat 46°N): ceil((46 - lat) / 0.333)
    # Columna (1 = oeste, desde lon -10°W): ceil((lon - (-10)) / 0.5)
    fila = math.ceil((46.0 - lat) / (1.0 / 3.0))
    col  = math.ceil((lon - (-10.0)) / 0.5)
    hoja = (fila - 1) * 20 + col  # numeración aproximada
    return f"PNOA_2016_{hoja:04d}_{x_ll}_{y_ll}_ORT-CLA-APP_h02.laz"


def _descargar_tiles_por_nomenclatura(bbox: Tuple, out_dir: Path) -> List[Path]:
    """
    Intenta descargar tiles LAZ usando la nomenclatura estándar del PNOA y
    el servicio de descarga directo del CNIG.  Devuelve los que se obtengan.
    """
    corners = _tiles_necesarios(bbox)
    archivos = []
    base_ftp = "https://centrodedescargas.cnig.es/CentroDescargas/dameRutaFichero"

    for x_ll, y_ll in corners:
        nombre = _nombre_tile_pnoa(x_ll, y_ll)
        dest = out_dir / nombre
        if dest.exists():
            print(f"  Ya existe: {nombre}")
            archivos.append(dest)
            continue
        # El CNIG expone los ficheros LAZ a través de su buscador de series;
        # sin un ID de fichero válido no se puede descargar directamente aquí.
        # Los tiles de fallback se dejan como aviso al usuario.
        print(f"  [info] Tile esperado: {nombre} (usa el Centro de Descargas del CNIG)")

    return archivos


# ── Procesado del LiDAR (LAZ → DTM / DSM) ────────────────────────────────────

def laz_a_dtm_dsm(
    laz_paths: List[Path],
    bbox: Tuple,
    resolution: float = RESOLUTION_M,
    out_dir: Path = None,
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    A partir de archivos LAZ genera dos rasters dentro de la bbox:
      - dtm_laz.tif : interpolado desde puntos clasificados como suelo (clase 2)
      - dsm_laz.tif : primeros retornos (todos los puntos ≥ suelo)

    Devuelve (ruta_dtm, ruta_dsm) o (None, None) si no hay datos válidos.
    """
    if not HAS_LASPY:
        print("  [aviso] laspy no disponible → saltando procesado LAZ")
        return None, None

    xmin, ymin, xmax, ymax = bbox
    cols = int(math.ceil((xmax - xmin) / resolution))
    rows = int(math.ceil((ymax - ymin) / resolution))
    transform = from_bounds(xmin, ymin, xmax, ymax, cols, rows)

    dsm_grid = np.full((rows, cols), np.nan, dtype=np.float32)
    dtm_grid = np.full((rows, cols), np.nan, dtype=np.float32)
    dsm_count = np.zeros((rows, cols), dtype=np.int32)
    dtm_count = np.zeros((rows, cols), dtype=np.int32)

    total_pts = 0
    for laz_path in laz_paths:
        print(f"  Procesando {laz_path.name} …")
        try:
            with laspy.open(str(laz_path)) as lf:
                for chunk in lf.chunk_iterator(1_000_000):
                    x = np.asarray(chunk.x)
                    y = np.asarray(chunk.y)
                    z = np.asarray(chunk.z)

                    # Filtrar a bbox
                    mask = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
                    if not mask.any():
                        continue
                    x, y, z = x[mask], y[mask], z[mask]

                    # Coordenadas píxel (origen = esquina superior-izquierda)
                    ci = np.floor((x - xmin) / resolution).astype(int).clip(0, cols - 1)
                    ri = np.floor((ymax - y) / resolution).astype(int).clip(0, rows - 1)

                    # DSM: máximo de todos los retornos
                    for idx in range(len(x)):
                        r, c = ri[idx], ci[idx]
                        if np.isnan(dsm_grid[r, c]) or z[idx] > dsm_grid[r, c]:
                            dsm_grid[r, c] = z[idx]

                    # DTM: media de retornos clasificados como suelo (clase 2)
                    try:
                        cls = np.asarray(chunk.classification)[mask]
                        ground = cls == 2
                        if ground.any():
                            xg, yg, zg = x[ground], y[ground], z[ground]
                            cig = np.floor((xg - xmin) / resolution).astype(int).clip(0, cols - 1)
                            rig = np.floor((ymax - yg) / resolution).astype(int).clip(0, rows - 1)
                            for idx in range(len(xg)):
                                r, c = rig[idx], cig[idx]
                                if np.isnan(dtm_grid[r, c]):
                                    dtm_grid[r, c] = zg[idx]
                                    dtm_count[r, c] = 1
                                else:
                                    dtm_grid[r, c] += zg[idx]
                                    dtm_count[r, c] += 1
                    except Exception:
                        pass  # La clasificación puede no existir en algunos archivos

                    total_pts += len(x)
        except Exception as exc:
            print(f"  [aviso] Error al leer {laz_path.name}: {exc}")
            continue

    if total_pts == 0:
        print("  [aviso] Ningún punto LiDAR dentro de la bbox")
        return None, None

    print(f"  Puntos procesados dentro de bbox: {total_pts:,}")

    # Promediar DTM acumulado
    valid = dtm_count > 0
    dtm_grid[valid] = dtm_grid[valid] / dtm_count[valid]

    # Rellenar NaN por interpolación (vecino más próximo)
    dtm_grid = _rellenar_nan(dtm_grid)
    dsm_grid = _rellenar_nan(dsm_grid)

    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": cols,
        "height": rows,
        "count": 1,
        "crs": f"EPSG:{EPSG_WORK}",
        "transform": transform,
        "nodata": np.nan,
        "compress": "lzw",
    }

    dtm_path = out_dir / "dtm_laz.tif"
    dsm_path = out_dir / "dsm_laz.tif"

    with rasterio.open(dtm_path, "w", **profile) as dst:
        dst.write(dtm_grid[np.newaxis])
    with rasterio.open(dsm_path, "w", **profile) as dst:
        dst.write(dsm_grid[np.newaxis])

    print(f"  DTM (LAZ) guardado → {dtm_path}")
    print(f"  DSM (LAZ) guardado → {dsm_path}")
    return dtm_path, dsm_path


def _rellenar_nan(arr: np.ndarray) -> np.ndarray:
    """Rellena NaN con el valor del vecino válido más próximo."""
    mask_nan = np.isnan(arr)
    if not mask_nan.any():
        return arr
    indices = ndimage.distance_transform_edt(
        mask_nan, return_distances=False, return_indices=True
    )
    arr[mask_nan] = arr[tuple(indices[:, mask_nan])]
    return arr


# ── Cálculo del CHM ────────────────────────────────────────────────────────────

def calcular_chm(dtm_path: Path, dsm_path: Path, out_path: Path) -> Path:
    """
    CHM = DSM − DTM.  Reproyecta/re-muestrea si las resoluciones difieren.
    """
    with rasterio.open(str(dtm_path)) as dtm_src:
        dtm = dtm_src.read(1).astype(np.float32)
        profile = dtm_src.profile.copy()
        transform = dtm_src.transform
        shape = dtm.shape

    with rasterio.open(str(dsm_path)) as dsm_src:
        # Re-muestrear DSM a la misma cuadrícula que el DTM
        dsm = np.empty(shape, dtype=np.float32)
        rasterio.warp.reproject(
            source=rasterio.band(dsm_src, 1),
            destination=dsm,
            src_transform=dsm_src.transform,
            src_crs=dsm_src.crs,
            dst_transform=transform,
            dst_crs=profile["crs"],
            resampling=Resampling.bilinear,
        )

    chm = dsm - dtm
    # Eliminar valores físicamente imposibles
    chm = np.where((chm < 0) | (chm > MAX_TREE_HT_M), 0.0, chm)

    profile.update(dtype="float32", nodata=0.0, compress="lzw")
    with rasterio.open(str(out_path), "w", **profile) as dst:
        dst.write(chm[np.newaxis])

    print(f"  CHM calculado → {out_path}  (max={chm.max():.1f} m)")
    return out_path


def recortar_chm_a_parcelas(chm_path: Path, gdf: gpd.GeoDataFrame, out_path: Path) -> Path:
    """Enmascara el CHM a la unión de todas las parcelas del shapefile."""
    union_geom = [mapping(unary_union(gdf.geometry))]
    with rasterio.open(str(chm_path)) as src:
        chm_clip, transform = rio_mask(src, union_geom, crop=True, nodata=0.0)
        profile = src.profile.copy()
        profile.update(
            height=chm_clip.shape[1],
            width=chm_clip.shape[2],
            transform=transform,
        )
    with rasterio.open(str(out_path), "w", **profile) as dst:
        dst.write(chm_clip)
    print(f"  CHM recortado a parcelas → {out_path}")
    return out_path


# ── Detección de árboles ──────────────────────────────────────────────────────

def detectar_arboles(
    chm_path: Path,
    min_altura: float = MIN_TREE_HT_M,
    smooth_sigma: float = SMOOTH_SIGMA,
    min_crown_area: float = MIN_CROWN_AREA,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Detecta árboles individuales en el CHM mediante:
      1. Suavizado Gaussiano para reducir ruido
      2. Detección de máximos locales como cimas de árbol
      3. Segmentación Watershed para delinear copas

    Devuelve (cimas_gdf, copas_gdf) en EPSG:25830.
    """
    with rasterio.open(str(chm_path)) as src:
        chm = src.read(1).astype(np.float32)
        transform = src.transform
        crs = src.crs
        res = src.res[0]  # tamaño de píxel en metros

    # Suavizado
    if HAS_SKIMAGE:
        chm_smooth = gaussian(chm, sigma=smooth_sigma).astype(np.float32)
    else:
        chm_smooth = ndimage.gaussian_filter(chm, sigma=smooth_sigma).astype(np.float32)

    # Máscara de vegetación
    veg_mask = chm_smooth >= min_altura

    # Máximos locales (cimas)
    min_dist_px = max(1, int(1.5 / res))  # mínimo 1.5 m entre cimas
    if HAS_SKIMAGE:
        coords = peak_local_max(
            chm_smooth,
            min_distance=min_dist_px,
            labels=veg_mask,
            threshold_abs=min_altura,
        )
        maxima = np.zeros_like(chm_smooth, dtype=bool)
        if len(coords):
            maxima[coords[:, 0], coords[:, 1]] = True
    else:
        local_max = maximum_filter(chm_smooth, size=max(3, min_dist_px * 2 + 1))
        maxima = (chm_smooth == local_max) & veg_mask

    labeled_maxima, n_trees = label(maxima)
    print(f"  Cimas detectadas (pre-filtro): {n_trees}")

    if n_trees == 0:
        print("  [aviso] No se detectaron árboles en el CHM.")
        empty_cimas = gpd.GeoDataFrame(columns=["tree_id", "altura_m", "geometry"],
                                       crs=crs)
        empty_copas = gpd.GeoDataFrame(columns=["tree_id", "area_copa_m2", "geometry"],
                                       crs=crs)
        return empty_cimas, empty_copas

    # Watershed para delinear copas
    if HAS_SKIMAGE:
        chm_inv = -chm_smooth
        labels_ws = watershed(chm_inv, labeled_maxima, mask=veg_mask)
    else:
        # Fallback sin skimage: expandir cimas con distance transform
        labels_ws = ndimage.label(veg_mask)[0]

    # Construir GeoDataFrames de cimas y copas
    rows_max = coords.tolist() if HAS_SKIMAGE else list(zip(*np.where(maxima)))
    cimas_records = []
    copas_records = []

    for tree_id in range(1, n_trees + 1):
        # Cima: píxel de máximo local
        if HAS_SKIMAGE and len(coords):
            r, c = coords[tree_id - 1]
        else:
            rr, cc = np.where(labeled_maxima == tree_id)
            if len(rr) == 0:
                continue
            r, c = rr[0], cc[0]

        altura = float(chm[r, c])
        if altura < min_altura:
            continue

        # Coordenadas del píxel → coordenadas mapa
        x, y = rasterio.transform.xy(transform, r, c)
        cimas_records.append({
            "tree_id": tree_id,
            "altura_m": round(altura, 2),
            "geometry": Point(x, y),
        })

        # Copa: región del watershed para este árbol
        crown_mask = labels_ws == tree_id
        area = float(crown_mask.sum() * res * res)
        if area < min_crown_area:
            continue

        # Vectorizar la máscara de copa
        crown_shapes = list(
            rasterio.features.shapes(crown_mask.astype(np.uint8), transform=transform)
        )
        crown_polys = [
            __import__("shapely.geometry", fromlist=["shape"]).shape(geom)
            for geom, val in crown_shapes
            if val == 1
        ]
        if not crown_polys:
            continue
        copa_geom = unary_union(crown_polys)
        copas_records.append({
            "tree_id": tree_id,
            "area_copa_m2": round(area, 1),
            "geometry": copa_geom,
        })

    cimas_gdf = gpd.GeoDataFrame(cimas_records, crs=crs)
    copas_gdf = gpd.GeoDataFrame(copas_records, crs=crs)

    # Filtro espacial: solo árboles dentro de las parcelas
    print(f"  Árboles detectados: {len(cimas_gdf)}")
    return cimas_gdf, copas_gdf


# ── Exportar resultados ───────────────────────────────────────────────────────

def exportar_resultados(
    cimas_gdf: gpd.GeoDataFrame,
    copas_gdf: gpd.GeoDataFrame,
    out_dir: Path,
) -> None:
    """Guarda cimas_arboles.gpkg, copas_arboles.gpkg y resumen.csv."""
    if not cimas_gdf.empty:
        cimas_path = out_dir / "cimas_arboles.gpkg"
        cimas_gdf.to_file(str(cimas_path), driver="GPKG")
        print(f"  Cimas exportadas → {cimas_path}")

        copas_path = out_dir / "copas_arboles.gpkg"
        copas_gdf.to_file(str(copas_path), driver="GPKG")
        print(f"  Copas exportadas → {copas_path}")

        csv_path = out_dir / "resumen_arboles.csv"
        cimas_gdf.drop(columns="geometry").to_csv(str(csv_path), index=False)
        print(f"  Resumen CSV → {csv_path}")

        # Estadísticas
        h = cimas_gdf["altura_m"]
        print("\n  ── Estadísticas de altura ──────────────────────────")
        print(f"  Nº de árboles detectados : {len(h)}")
        print(f"  Altura mínima            : {h.min():.1f} m")
        print(f"  Altura máxima            : {h.max():.1f} m")
        print(f"  Altura media             : {h.mean():.1f} m")
        print(f"  Mediana                  : {h.median():.1f} m")
        print(f"  Desv. estándar           : {h.std():.1f} m")
        if not copas_gdf.empty:
            a = copas_gdf["area_copa_m2"]
            print(f"  Área copa media          : {a.mean():.1f} m²")
    else:
        print("  Sin árboles detectados; no se exportan capas.")


# ── Función principal ─────────────────────────────────────────────────────────

def main(shp_path: Path, out_dir: Path, usar_laz: bool = True) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print("\n══════════════════════════════════════════════════════════")
    print(" ESTIMACIÓN DE ALTURAS DE ÁRBOLES — PNOA LiDAR (IGN España)")
    print("══════════════════════════════════════════════════════════\n")

    # 1. Leer shapefile
    print("► Paso 1: Cargando shapefile de parcelas …")
    gdf = leer_shapefile(shp_path)
    bbox = bbox_con_buffer(gdf, buffer=BUFFER_M)
    print(f"  Bbox de trabajo (con buffer {BUFFER_M} m): {tuple(round(v,1) for v in bbox)}\n")

    # 2. Descargar MDT desde IGN WCS
    print("► Paso 2: Descargando MDT desde IGN WCS …")
    dtm_path = out_dir / "mdt_ign.tif"
    try:
        descargar_mdt(bbox, dtm_path, coverage="Elevacion:MDT01")
        dsm_path = None
    except Exception as exc:
        print(f"  [error] Descarga MDT01 fallida: {exc}")
        print("  Reintentando con MDT05 (5 m) …")
        try:
            descargar_mdt(bbox, dtm_path, coverage="Elevacion:MDT05")
            dsm_path = None
        except Exception as exc2:
            print(f"  [error] MDT05 también falló: {exc2}")
            print("  El MDT del IGN no está disponible ahora. "
                  "Puedes proporcionar un MDT propio con --dtm.")
            dtm_path = None

    # 3. Descargar tiles LAZ del PNOA
    dsm_path = None
    if usar_laz:
        print("\n► Paso 3: Descargando tiles LAZ del PNOA (CNIG) …")
        laz_dir = out_dir / "laz_tiles"
        laz_files = descargar_tiles_laz(bbox, laz_dir)

        if laz_files:
            print(f"\n► Paso 4: Procesando {len(laz_files)} archivos LAZ …")
            dtm_laz, dsm_path = laz_a_dtm_dsm(laz_files, bbox, out_dir=out_dir)
            if dtm_laz and dtm_path is None:
                dtm_path = dtm_laz  # usar DTM derivado del LAZ si no hay WCS
        else:
            print("  No se encontraron/descargaron tiles LAZ.")
            print("  ► Descarga manual:")
            print("    1. Ve a https://centrodedescargas.cnig.es")
            print("    2. Busca 'LIDAR PNOA 2' para las coordenadas:")
            for x_ll, y_ll in _tiles_necesarios(bbox):
                print(f"       Tile: X={x_ll}, Y={y_ll} (ETRS89 UTM 30N)")
            print("    3. Guarda los LAZ en:", laz_dir)
            print("    4. Vuelve a ejecutar el script.\n")
    else:
        print("\n► Paso 3: Omitiendo descarga LAZ (--no-laz)")

    # 4. Necesitamos DTM + DSM para el CHM
    if dtm_path is None or dsm_path is None:
        print("\n[aviso] Falta DTM o DSM para calcular el CHM.")
        if dtm_path is None:
            print("  Proporciona el MDT con: --dtm ruta/al/mdt.tif")
        if dsm_path is None:
            print("  Descarga los tiles LAZ del PNOA (ver instrucciones arriba)")
        print("\nSaliendo — sin datos suficientes para continuar.")
        return

    # 5. Calcular CHM
    print("\n► Paso 5: Calculando CHM …")
    chm_raw = out_dir / "chm_raw.tif"
    calcular_chm(dtm_path, dsm_path, chm_raw)

    chm_path = out_dir / "chm_parcelas.tif"
    recortar_chm_a_parcelas(chm_raw, gdf, chm_path)

    # 6. Detectar árboles
    print("\n► Paso 6: Detectando árboles …")
    cimas_gdf, copas_gdf = detectar_arboles(chm_path)

    # Filtro: solo árboles dentro de las parcelas
    if not cimas_gdf.empty:
        cimas_gdf = cimas_gdf[cimas_gdf.within(unary_union(gdf.geometry))]
        copas_gdf = copas_gdf[
            copas_gdf.geometry.intersects(unary_union(gdf.geometry))
        ]
        print(f"  Árboles dentro de parcelas: {len(cimas_gdf)}")

    # 7. Exportar
    print("\n► Paso 7: Exportando resultados …")
    exportar_resultados(cimas_gdf, copas_gdf, out_dir)
    print(f"\n✓ Proceso completado. Resultados en: {out_dir.resolve()}\n")


# ── Modo script con argumentos ────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Estimación de alturas de árboles con LiDAR PNOA (IGN España)"
    )
    p.add_argument(
        "--shp",
        type=Path,
        default=Path("eb696780-TERRENOS.shp"),
        help="Shapefile de parcelas (ETRS89 UTM 30N / EPSG:25830)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("output_lidar"),
        help="Directorio de salida (se crea si no existe)",
    )
    p.add_argument(
        "--dtm",
        type=Path,
        default=None,
        help="MDT propio en GeoTIFF (opcional; si se omite se descarga del IGN)",
    )
    p.add_argument(
        "--dsm",
        type=Path,
        default=None,
        help="MDS/DSM propio en GeoTIFF (opcional; si se omite se genera de los LAZ)",
    )
    p.add_argument(
        "--no-laz",
        action="store_true",
        help="No intentar descargar tiles LAZ del CNIG",
    )
    p.add_argument(
        "--min-altura",
        type=float,
        default=MIN_TREE_HT_M,
        help=f"Altura mínima de árbol en metros (defecto: {MIN_TREE_HT_M})",
    )
    p.add_argument(
        "--resolucion",
        type=float,
        default=RESOLUTION_M,
        help=f"Resolución del CHM en metros (defecto: {RESOLUTION_M})",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Sobreescribir globales con argumentos CLI
    RESOLUTION_M = args.resolucion
    MIN_TREE_HT_M = args.min_altura

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # Leer shapefile
    gdf = leer_shapefile(args.shp)

    # Si el usuario ya tiene DTM y/o DSM propios, saltamos las descargas
    if args.dtm and args.dsm:
        import shutil
        dtm_p = out_dir / "mdt_ign.tif"
        dsm_p = out_dir / "dsm_laz.tif"
        shutil.copy(args.dtm, dtm_p)
        shutil.copy(args.dsm, dsm_p)
        chm_raw = out_dir / "chm_raw.tif"
        calcular_chm(dtm_p, dsm_p, chm_raw)
        chm_path = out_dir / "chm_parcelas.tif"
        recortar_chm_a_parcelas(chm_raw, gdf, chm_path)
        cimas_gdf, copas_gdf = detectar_arboles(chm_path, min_altura=args.min_altura)
        exportar_resultados(cimas_gdf, copas_gdf, out_dir)
    else:
        main(args.shp, out_dir, usar_laz=not args.no_laz)
