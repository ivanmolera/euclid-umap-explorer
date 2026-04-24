# Euclid UMAP Explorer

Aplicación web en Streamlit para explorar interactivamente objetos astronómicos del catálogo morfológico de Euclid, calcular embeddings UMAP sobre variables PCA o morfológicas, y marcar candidatos o objetos confirmados de strong lensing cruzando por `object_id`.

## Estructura

```text
euclid-umap-explorer/
├── app.py
├── requirements.txt
├── Dockerfile
├── README.md
├── .gitignore
└── data/
    ├── morphology/
    │   └── .gitkeep
    ├── strong_lenses/
    │   └── .gitkeep
    └── cutouts/
        └── .gitkeep
```

## Entorno local

Requiere Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Datos

Los datos pesados no deben subirse a GitHub. Descárgalos manualmente desde Google Drive y colócalos en estas carpetas locales:

- Catálogo morfológico: `data/morphology/`
- Catálogo de strong lenses: `data/strong_lenses/`
- Imágenes/cutouts opcionales: `data/cutouts/`

Carpetas de origen:

- Morphology catalogue: https://drive.google.com/drive/folders/1WJRuWRKJsRn818H3zm2QBrfXttaYCfTw?usp=drive_link
- Strong lensing catalogue: https://drive.google.com/drive/folders/1oczyHPCJ6lPjqA2WUykQeQq_peWeO2Ru?usp=drive_link

Formatos soportados para catálogos: CSV, Parquet y Feather. El catálogo morfológico debe incluir `object_id`. El catálogo de lentes debe incluir `object_id` o una columna equivalente como `id`.

Para asociar imágenes, coloca archivos en `data/cutouts/` con nombre igual a `id_str` u `object_id`, por ejemplo:

```text
data/cutouts/123456789.png
data/cutouts/EUCLID_OBJECT_ID.jpg
```

## Ejecutar

```bash
streamlit run app.py
```

La app permite seleccionar catálogos locales, elegir cluster, seleccionar variables numéricas, ajustar `n_neighbors`, `min_dist` y limitar el número máximo de objetos visualizados.

## Despliegue en Google Cloud Run

### Importante: uso de gcloud personal

Antes de ejecutar cualquier comando de despliegue, comprobar que `gcloud` está usando la configuración personal, no la del trabajo.

Ejecutar:

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

Cuando la configuración y la cuenta sean correctas, desplegar con:

```bash
gcloud run deploy euclid-umap-app \
  --source . \
  --region europe-west1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 900
```

El contenedor ejecuta Streamlit en el puerto `$PORT`, como requiere Cloud Run.

## Notas de evolución

La carga de datos está encapsulada en funciones dentro de `app.py`. La primera versión asume ficheros locales descargados manualmente, pero se puede sustituir esa capa por lectores de Google Drive, Google Cloud Storage o URLs firmadas sin reescribir la lógica de visualización.
