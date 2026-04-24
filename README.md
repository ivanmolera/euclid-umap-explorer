# Euclid UMAP Explorer

Aplicación web en Streamlit para visualizar UMAP sobre clusters de objetos astronómicos Euclid usando el catálogo PCA `representations_pca_40.parquet` y el catálogo de strong lenses.

La app no espera catálogos ni cutouts dentro del repositorio. Los datos se leen desde Google Drive montado, usando las rutas configuradas en `app.py` o variables de entorno.

## Rutas de datos

Valores por defecto:

```python
MORPH_PATH = "/content/drive/MyDrive/catalogues/morphology_catalogue/morphology_catalogue.parquet"
CUTOUT_BASE = "/content/drive/MyDrive/catalogues/morphology_catalogue/cutouts_jpg_gz_arcsinh_vis_only"
PARQUET_PATH = "/content/drive/MyDrive/catalogues/morphology_catalogue/representations_pca_40.parquet"
LENS_PATH = "/content/drive/MyDrive/catalogues/strong_lensing_catalogue/q1_discovery_engine_lens_catalog.csv"
LENS_IMG_BASE = "/content/drive/MyDrive/catalogues/strong_lensing_catalogue/lens"
```

También pueden sobrescribirse con variables de entorno:

```bash
export MORPH_PATH="/content/drive/MyDrive/catalogues/morphology_catalogue/morphology_catalogue.parquet"
export CUTOUT_BASE="/content/drive/MyDrive/catalogues/morphology_catalogue/cutouts_jpg_gz_arcsinh_vis_only"
export PARQUET_PATH="/content/drive/MyDrive/catalogues/morphology_catalogue/representations_pca_40.parquet"
export LENS_PATH="/content/drive/MyDrive/catalogues/strong_lensing_catalogue/q1_discovery_engine_lens_catalog.csv"
export LENS_IMG_BASE="/content/drive/MyDrive/catalogues/strong_lensing_catalogue/lens"
```

## Funcionalidad

- Carga `representations_pca_40.parquet`.
- Extrae `object_id` desde `id_str` cuando hace falta.
- Detecta automáticamente columnas `feat_pca_*`.
- Carga el catálogo de lentes y, por defecto, usa candidatas `grade A`.
- Ejecuta clusterización BIRCH solo cuando el usuario pulsa el botón.
- Usa `StandardScaler.partial_fit` y BIRCH por lotes, siguiendo la lógica de los notebooks.
- Muestra un desplegable de clusters con `id`, número de objetos y número de lentes.
- Permite seleccionar componentes PCA para construir UMAP.
- Calcula UMAP con escalado previo y visualiza el embedding con Plotly.
- Marca visualmente lentes y no lentes.
- Muestra detalles del objeto seleccionado, incluyendo una lectura puntual de `MORPH_PATH`.
- Muestra imágenes del objeto seleccionado cuando existen en `CUTOUT_BASE` o `LENS_IMG_BASE`.

## Entorno local

Requiere Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Ejecutar

La ruta `/content/drive/...` es típica de Google Colab. Si ejecutas fuera de Colab, monta Google Drive o sobrescribe las variables de entorno anteriores para apuntar al mount real.

Por defecto la app copia los catálogos leídos desde Google Drive Desktop a una cache local fuera del repositorio, en `~/.cache/euclid-umap-explorer`, mostrando un estado y spinner durante la copia. Esto evita timeouts de `pyarrow` al leer parquet desde el filesystem virtual de Drive. Puedes cambiar la cache con `EUCLID_CACHE_DIR` o desactivarla con `EUCLID_USE_LOCAL_CACHE=0`.

```bash
streamlit run app.py
```

## Despliegue en Google Cloud Run

La versión actual asume acceso de filesystem a Google Drive montado. Para Cloud Run, lo recomendable será mover los catálogos y cutouts a Google Cloud Storage o usar URLs firmadas/Drive API antes del despliegue.

### Importante: uso de gcloud personal

Antes de ejecutar cualquier comando de despliegue, comprobar que `gcloud` está usando la configuración personal.

```bash
gcloud config configurations list
gcloud config list
gcloud auth list
```

La configuración activa debe ser:

```text
personal
```

La cuenta activa debe ser:

```text
ivan.molera@gmail.com
```

No usar ninguna configuración, cuenta ni proyecto que no pertenezca al entorno personal.

Si la configuración activa no es `personal`, activar:

```bash
gcloud config configurations activate personal
```

Después comprobar que el proyecto activo es el proyecto personal:

```bash
gcloud config get-value project
```

No ejecutar ningún comando `gcloud run deploy`, `gcloud builds submit` ni cambios de configuración si la cuenta activa o el proyecto no pertenecen al entorno personal.

Cuando la carga de datos esté migrada a un backend accesible por Cloud Run, el despliegue será:

```bash
gcloud run deploy euclid-umap-app \
  --source . \
  --region europe-west1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 900
```

## Datos pesados

No subir catálogos, imágenes ni cutouts al repositorio GitHub.
