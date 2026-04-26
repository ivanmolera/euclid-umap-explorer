# Euclid UMAP Explorer

Streamlit web application for exploring Euclid astronomical-object clusters with UMAP, using PCA representations and a strong-lensing catalogue.

The application reads catalogues and image assets at runtime from configurable paths. Google Cloud Storage paths are supported through `gs://` URIs.

## Data Sources

The analysis uses Euclid Q1 catalogue products published on Zenodo:

- [Euclid Quick Data Release (Q1): First visual morphology catalogue](https://zenodo.org/records/15106473)
- [Euclid Quick Data Release (Q1): The Strong Lensing Discovery Engine](https://zenodo.org/records/15025832)

Runtime data is expected to be available through the configured paths:

- PCA representations: `PARQUET_PATH`
- Strong-lensing catalogue: `LENS_PATH`
- Morphology cutouts: `CUTOUT_BASE`
- Strong-lens images: `LENS_IMG_BASE`
- Optional morphology catalogue: `MORPH_PATH`

The `data/` directories in this repository are placeholders for local development workflows.

## Features

- Loads the PCA catalogue `representations_pca_40.parquet`.
- Automatically detects `feat_pca_*` columns.
- Derives `object_id` from `id_str` when required.
- Loads a strong-lensing catalogue and joins it with the PCA catalogue through `object_id`.
- Allows the user to select lens grades used in the join (`A`, `B`, `C`).
- Runs BIRCH clustering with all available PCA components.
- Runs BIRCH clustering on demand.
- Allows the user to select a cluster, PCA components, and UMAP parameters.
- Scales selected features with `StandardScaler`.
- Computes a 2D UMAP embedding.
- Visualizes the embedding with Plotly.
- Distinguishes non-lenses, lens grades, canonical objects, and anomalous objects.
- Provides a visual cluster summary with canonical, anomalous, random, and lens examples.
- Computes compact lens vs non-lens PCA histograms for clusters containing lenses, limited to six selected UMAP components on screen.
- Supports point selection from the UMAP plot.
- Shows selected-object metadata, lens status, UMAP coordinates, selected PCA values, and available images.
- Loads morphology cutouts and lens images on demand from `CUTOUT_BASE` and `LENS_IMG_BASE`.

## Configuration

Set the required catalogue and image paths through environment variables:

```bash
export PARQUET_PATH="gs://<bucket>/catalogues/morphology_catalogue/representations_pca_40.parquet"
export LENS_PATH="gs://<bucket>/catalogues/strong_lensing_catalogue/q1_discovery_engine_lens_catalog.csv"
export CUTOUT_BASE="gs://<bucket>/catalogues/morphology_catalogue/cutouts_jpg_gz_arcsinh_vis_only"
export LENS_IMG_BASE="gs://<bucket>/catalogues/strong_lensing_catalogue/lens"
export EUCLID_USE_LOCAL_CACHE=0
```

Optional variables:

```bash
export MORPH_PATH="gs://<bucket>/catalogues/morphology_catalogue/morphology_catalogue.parquet"
export EUCLID_CACHE_DIR="$HOME/.cache/euclid-umap-explorer"
```

`MORPH_PATH` is used to display the full morphology-catalogue row for the selected object when available.

`EUCLID_USE_LOCAL_CACHE=0` disables local catalogue caching. This is the recommended setting for `gs://` paths.

## Local Setup

Python 3.11 is required.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Configure the environment variables and start Streamlit:

```bash
streamlit run app.py
```

## Cluster Summary

After BIRCH clustering, the cluster summary combines numerical metrics and visual examples for each cluster:

- clusters are ordered by the number of lenses found in each cluster, from highest to lowest,
- canonical object, closest to the cluster centroid in the selected PCA feature space,
- anomalous object, farthest from the cluster centroid,
- three deterministic random examples from the cluster,
- up to five lens examples, ordered by grade `A`, then `B`, then `C`.

Lens examples use the lens image when available and fall back to the morphology cutout otherwise.

For clusters containing lenses, the summary row includes an on-demand PCA histogram comparison between lens and non-lens objects.

## Cloud Run Deployment

Example deployment command:

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

The `Dockerfile` runs Streamlit on the Cloud Run `$PORT`.

## Repository Layout

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
