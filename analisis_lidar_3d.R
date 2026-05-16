#!/usr/bin/env Rscript
# =============================================================================
# Análisis LiDAR 3D de masa forestal con lidR
# Equivalente de alto nivel al flujo de trabajo sobre nubes de puntos PNOA
# =============================================================================
# Instalar dependencias (solo la primera vez):
#   install.packages(c("lidR","terra","sf","future","ggplot2","viridis"),
#                    repos = "https://cloud.r-project.org")
# =============================================================================

suppressPackageStartupMessages({
  library(lidR)
  library(terra)
  library(sf)
  library(future)
  library(ggplot2)
})

# ── Configuración ─────────────────────────────────────────────────────────────
SHP_PATH    <- "eb696780-TERRENOS.shp"   # shapefile de parcelas
LAZ_DIR     <- "output_lidar/laz_tiles"  # tiles LAZ descargados del CNIG
OUT_DIR     <- "output_lidar_3d"         # carpeta de resultados
RES_CHM     <- 1.0                       # resolución CHM en metros
MIN_HT      <- 2.0                       # altura mínima árbol (m)
WS_LMF      <- 5                         # ventana detección cimas (m)

# Paralelismo: usa todos los núcleos disponibles menos uno
plan(multisession, workers = max(1L, availableCores() - 1L))

dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

# ── 1. Leer shapefile de parcelas ─────────────────────────────────────────────
cat("\n[1/7] Leyendo shapefile de parcelas...\n")
parcelas <- st_read(SHP_PATH, quiet = TRUE)
if (is.na(st_crs(parcelas))) {
  st_crs(parcelas) <- 25830          # ETRS89 UTM 30N si falta .prj
}
cat(sprintf("    %d parcelas | bbox: %.0f,%.0f — %.0f,%.0f\n",
            nrow(parcelas), st_bbox(parcelas)))

# ── 2. Cargar nube de puntos PNOA ─────────────────────────────────────────────
cat("\n[2/7] Cargando catálogo LiDAR...\n")
laz_files <- list.files(LAZ_DIR, pattern = "\\.(laz|las)$",
                         full.names = TRUE, recursive = TRUE)

if (length(laz_files) == 0) {
  stop(
    "No se encontraron archivos LAZ en: ", LAZ_DIR, "\n",
    "Descárgalos del Centro de Descargas del CNIG:\n",
    "  https://centrodedescargas.cnig.es\n",
    "  Producto: LIDAR-PNOA-2  |  Zona: UTM30N X≈728000, Y≈4818000"
  )
}

ctg <- readLAScatalog(LAZ_DIR)
crs(ctg) <- st_crs(25830)$wkt
opt_chunk_buffer(ctg)    <- 20    # buffer entre tiles para evitar bordes
opt_progress(ctg)        <- TRUE
opt_output_files(ctg)    <- file.path(OUT_DIR, "norm/{*}")

# Recortar al bbox de las parcelas (+ 50 m de margen)
bbox_par <- st_buffer(st_as_sfc(st_bbox(parcelas)), 50)
ctg_clip  <- clip_roi(ctg, bbox_par)

# ── 3. Clasificar suelo y normalizar alturas ───────────────────────────────────
cat("\n[3/7] Clasificando suelo (CSF) y normalizando...\n")
# CSF = Cloth Simulation Filter — estándar en PNOA 2ª cobertura
# Si los LAZ ya vienen clasificados (clase 2 = suelo), usa:
#   opt_filter(ctg) <- "-keep_class 2"
# y salta directamente a normalize_height()

las_norm <- normalize_height(ctg_clip,
                             algorithm = tin(),    # TIN sobre puntos suelo (clase 2)
                             use_class  = 2L)

# ── 4. Modelo Digital de Copas (CHM) — algoritmo pitfree ─────────────────────
cat("\n[4/7] Generando CHM con algoritmo Pit-Free...\n")
# pitfree: el mejor algoritmo disponible, elimina artefactos en huecos de copa
chm <- rasterize_canopy(
  las_norm,
  res       = RES_CHM,
  algorithm = pitfree(
    thresholds = c(0, 2, 5, 10, 15, 20),   # umbrales de altura para sub-nubes
    max_edge   = c(0, 1.5),                 # máxima arista TIN (m)
    subcircle  = 0.2                        # radio de sub-círculo anti-pit
  )
)
chm <- terra::focal(chm, w = matrix(1, 3, 3), fun = "mean", na.policy = "only")
writeRaster(chm, file.path(OUT_DIR, "CHM_pitfree_1m.tif"), overwrite = TRUE)
cat(sprintf("    CHM: %.1f – %.1f m\n", minmax(chm)[1], minmax(chm)[2]))

# ── 5. Detección de cimas (Local Maximum Filter) ──────────────────────────────
cat("\n[5/7] Detectando cimas de árbol (LMF adaptativo)...\n")
# LMF con ventana adaptativa: ventana más grande para árboles altos
ws_fun <- function(x) {          # x = altura en el CHM
  ifelse(x < 5, 2,
  ifelse(x < 10, 3,
  ifelse(x < 20, 5, 7)))
}
ttops <- locate_trees(chm,
                      algorithm = lmf(ws = ws_fun, hmin = MIN_HT))
cat(sprintf("    Cimas detectadas: %d\n", nrow(ttops)))

st_write(ttops, file.path(OUT_DIR, "cimas_lmf.gpkg"),
         delete_dsn = TRUE, quiet = TRUE)

# ── 6. Segmentación individual de árboles (Dalponte 2016) ────────────────────
cat("\n[6/7] Segmentando árboles individuales (Dalponte 2016)...\n")
# dalponte2016: crecimiento de región desde cimas hacia el CHM
# Alternativas: li2012 (3D, más lento), watershed (topográfico)
las_seg <- segment_trees(las_norm,
                          algorithm = dalponte2016(chm, ttops,
                                                    th_tree  = MIN_HT,
                                                    th_seed  = 0.45,
                                                    th_cr    = 0.55,
                                                    max_cr   = 20))

# ── 7. Métricas por árbol y exportación ───────────────────────────────────────
cat("\n[7/7] Calculando métricas por árbol...\n")
metricas <- crown_metrics(las_seg,
                           func = .stdtreemetrics,
                           geom = "convex")   # polígono convexo de copa

# Métricas estándar: Z (altura cima), npoints, convhull_area, etc.
# Añadir altura dominante (P95 de los retornos de cada árbol)
metricas_extra <- crown_metrics(las_seg,
  func = ~list(
    h_max    = max(Z),
    h_mean   = mean(Z),
    h_p95    = quantile(Z, 0.95),
    n_pts    = .N,
    densidad = .N / convhull_area
  ),
  geom = "convex"
)

st_write(metricas,       file.path(OUT_DIR, "copas_dalponte.gpkg"),
         delete_dsn = TRUE, quiet = TRUE)
st_write(metricas_extra, file.path(OUT_DIR, "metricas_arboles.gpkg"),
         delete_dsn = TRUE, quiet = TRUE)

# CSV resumen
df_csv <- as.data.frame(metricas_extra)
df_csv$geometry <- NULL
write.csv(df_csv, file.path(OUT_DIR, "metricas_arboles.csv"), row.names = FALSE)

# ── Estadísticas globales ──────────────────────────────────────────────────────
cat("\n══════════════════════════════════════════════════\n")
cat(" RESUMEN — ANÁLISIS LIDAR lidR\n")
cat("══════════════════════════════════════════════════\n")
cat(sprintf("  Árboles detectados  : %d\n",        nrow(metricas_extra)))
cat(sprintf("  Altura máxima       : %.1f m\n",    max(metricas_extra$h_max, na.rm = TRUE)))
cat(sprintf("  Altura media (p95)  : %.1f m\n",    mean(metricas_extra$h_p95, na.rm = TRUE)))
cat(sprintf("  Área copa media     : %.1f m²\n",   mean(metricas_extra$convhull_area, na.rm = TRUE)))
cat(sprintf("  Densidad media      : %.1f pts/m²\n",mean(metricas_extra$densidad, na.rm = TRUE)))
cat("══════════════════════════════════════════════════\n")

# ── Visualizaciones 2D (exportadas como PNG) ───────────────────────────────────
cat("\nGenerando visualizaciones...\n")

# 4a. CHM con cimas superpuestas
png(file.path(OUT_DIR, "mapa_CHM_cimas.png"), width = 1400, height = 1200, res = 150)
plot(chm, col = height.colors(50), main = "CHM pitfree + Cimas de árbol (LMF)")
plot(st_geometry(ttops), add = TRUE, col = "red", pch = 3, cex = 0.6)
plot(st_geometry(parcelas), add = TRUE, border = "yellow", lwd = 2)
dev.off()

# 4b. Copas segmentadas coloreadas por altura
png(file.path(OUT_DIR, "mapa_copas_segmentadas.png"), width = 1400, height = 1200, res = 150)
pal <- colorRampPalette(c("#edf8e9","#74c476","#238b45","#005a32"))(100)
plot(st_geometry(metricas_extra),
     col  = pal[cut(metricas_extra$h_max, 100, labels = FALSE)],
     border = "white", lwd = 0.3,
     main = "Copas individuales (Dalponte 2016) — color = altura máxima")
plot(st_geometry(parcelas), add = TRUE, border = "yellow", lwd = 2)
dev.off()

# 4c. Histograma de alturas
gg <- ggplot(df_csv, aes(x = h_max)) +
  geom_histogram(aes(fill = after_stat(x)), bins = 30, color = "white", linewidth = 0.2) +
  scale_fill_gradientn(colors = c("#edf8e9","#74c476","#238b45","#005a32"),
                        name = "Altura (m)") +
  geom_vline(xintercept = c(5, 10, 20), linetype = "dashed",
             color = "gray50", linewidth = 0.8) +
  annotate("text", x = c(5.3, 10.3, 20.3),
           y = Inf, label = c("5m","10m","20m"),
           vjust = 1.5, color = "gray40", size = 3) +
  labs(title  = "Distribución de alturas máximas por árbol (lidR)",
       x = "Altura máxima (m)", y = "Número de árboles") +
  theme_minimal(base_size = 13)
ggsave(file.path(OUT_DIR, "histograma_alturas_lidR.png"),
       gg, width = 8, height = 5, dpi = 150)

# ── 3D interactivo (requiere entorno gráfico) ──────────────────────────────────
# Para visualización 3D interactiva en RStudio / terminal gráfica:
#
#   plot(las_seg, color = "treeID", size = 2, bg = "black")
#
# Para exportar nube 3D a formato PLY (abrir en CloudCompare / MeshLab):
#   writeLAS(las_seg, file.path(OUT_DIR, "nube_segmentada.laz"))

cat("\n✓ Resultados en:", OUT_DIR, "\n")
cat("  CHM_pitfree_1m.tif       — modelo de alturas de copas\n")
cat("  cimas_lmf.gpkg           — cimas de árbol (LMF)\n")
cat("  copas_dalponte.gpkg      — polígonos de copa\n")
cat("  metricas_arboles.gpkg/.csv — métricas por árbol\n")
cat("  mapa_CHM_cimas.png\n")
cat("  mapa_copas_segmentadas.png\n")
cat("  histograma_alturas_lidR.png\n")
cat("\nPara visualización 3D interactiva ejecuta en R:\n")
cat("  library(lidR); las <- readLAS('output_lidar/laz_tiles/tu_tile.laz')\n")
cat("  las_n <- normalize_height(las, tin())\n")
cat("  las_s <- segment_trees(las_n, dalponte2016(chm, ttops))\n")
cat("  plot(las_s, color = 'treeID', size = 2, bg = 'black')\n")
