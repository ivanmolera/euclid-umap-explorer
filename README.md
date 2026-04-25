# Euclid UMAP Explorer

Aplicación web en Streamlit para explorar clusters de objetos astronómicos Euclid mediante UMAP, cruzando el catálogo PCA/morfológico con un catálogo de strong lenses.

La app está pensada para trabajar con datos pesados fuera del repositorio. Los catálogos e imágenes pueden leerse desde rutas locales o desde Google Cloud Storage usando rutas `gs://`.

## Funcionalidad

- Carga el catálogo PCA `representations_pca_40.parquet`.
- Detecta automáticamente columnas `feat_pca_*`.
- Extrae `object_id` desde `id_str` cuando hace falta.
- Carga el catálogo de strong lenses y cruza ambos catálogos por `object_id`.
- Ejecuta clusterización BIRCH solo cuando el usuario pulsa el botón.
- Permite seleccionar un cluster y componentes PCA para construir UMAP.
- Calcula UMAP con escalado previo mediante `StandardScaler`.
- Visualiza el embedding con Plotly.
- Marca visualmente no lentes, lentes, objeto canónico y objeto más anómalo.
- Permite seleccionar puntos del mapa y ver detalles del objeto.
- Muestra un indicador claro de `LENTE` / `NO LENTE`.
- Carga cutouts e imágenes de lentes bajo demanda desde `CUTOUT_BASE` y `LENS_IMG_BASE`.

## Variables de entorno

La app usa estas rutas configurables:

```bash
export PARQUET_PATH="gs://<bucket>/catalogues/morphology_catalogue/representations_pca_40.parquet"
export LENS_PATH="gs://<bucket>/catalogues/strong_lensing_catalogue/q1_discovery_engine_lens_catalog.csv"
export CUTOUT_BASE="gs://<bucket>/catalogues/morphology_catalogue/cutouts_jpg_gz_arcsinh_vis_only"
export LENS_IMG_BASE="gs://<bucket>/catalogues/strong_lensing_catalogue/lens"
export EUCLID_USE_LOCAL_CACHE=0
```

Opcionalmente:

```bash
export MORPH_PATH="gs://<bucket>/catalogues/morphology_catalogue/morphology_catalogue.parquet"
export EUCLID_CACHE_DIR="$HOME/.cache/euclid-umap-explorer"
```

`EUCLID_USE_LOCAL_CACHE=0` desactiva la copia local de catálogos. Es el valor recomendado cuando se leen datos desde `gs://`.

## Entorno local

Requiere Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Después configura las variables de entorno y ejecuta:

```bash
streamlit run app.py
```

## Datos

Los datos pesados no deben subirse al repositorio. Mantén fuera de GitHub:

- catálogos Parquet/CSV grandes,
- cutouts,
- imágenes de lentes,
- cualquier archivo generado por procesos de extracción o sincronización.

Si usas rutas locales, puedes configurar las variables de entorno para apuntar a tu filesystem. Si usas Cloud Run, usa un backend accesible por el servicio, por ejemplo Google Cloud Storage.

## Despliegue en Cloud Run

Ejemplo de despliegue:

```bash
gcloud run deploy euclid-umap-app \
  --source . \
  --region europe-west1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 900 \
  --set-env-vars=PARQUET_PATH=gs://<bucket>/catalogues/morphology_catalogue/representations_pca_40.parquet,LENS_PATH=gs://<bucket>/catalogues/strong_lensing_catalogue/q1_discovery_engine_lens_catalog.csv,CUTOUT_BASE=gs://<bucket>/catalogues/morphology_catalogue/cutouts_jpg_gz_arcsinh_vis_only,LENS_IMG_BASE=gs://<bucket>/catalogues/strong_lensing_catalogue/lens,EUCLID_USE_LOCAL_CACHE=0
```

El `Dockerfile` ejecuta Streamlit en el puerto indicado por Cloud Run mediante `$PORT`.

## Estructura

```text
.
├── app.py
├── requirements.txt
├── Dockerfile
├── README.md
└── data/
    ├── morphology/
    ├── strong_lenses/
    └── cutouts/
```
