from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path
from typing import Iterable

import gcsfs
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import plotly.express as px
import streamlit as st
import umap
from PIL import Image
from sklearn.cluster import Birch
from sklearn.preprocessing import StandardScaler


APP_TITLE = "Euclid UMAP Explorer"

MORPH_PATH = os.getenv(
    "MORPH_PATH",
    "/content/drive/MyDrive/catalogues/morphology_catalogue/morphology_catalogue.parquet",
)
CUTOUT_BASE = os.getenv(
    "CUTOUT_BASE",
    "/content/drive/MyDrive/catalogues/morphology_catalogue/cutouts_jpg_gz_arcsinh_vis_only",
)
PARQUET_PATH = os.getenv(
    "PARQUET_PATH",
    "/content/drive/MyDrive/catalogues/morphology_catalogue/representations_pca_40.parquet",
)
LENS_PATH = os.getenv(
    "LENS_PATH",
    "/content/drive/MyDrive/catalogues/strong_lensing_catalogue/q1_discovery_engine_lens_catalog.csv",
)
LENS_IMG_BASE = os.getenv(
    "LENS_IMG_BASE",
    "/content/drive/MyDrive/catalogues/strong_lensing_catalogue/lens",
)
CACHE_DIR = Path(
    os.getenv("EUCLID_CACHE_DIR", Path.home() / ".cache" / "euclid-umap-explorer")
)
USE_LOCAL_CACHE = os.getenv("EUCLID_USE_LOCAL_CACHE", "1") != "0"

DEFAULT_CLUSTER_FEATURES = [
    "feat_pca_1",
    "feat_pca_12",
    "feat_pca_6",
    "feat_pca_10",
    "feat_pca_13",
    "feat_pca_27",
]


def inject_plot_cursor_css() -> None:
    st.markdown(
        """
        <style>
        .js-plotly-plot .plotly .draglayer .drag,
        .js-plotly-plot .plotly .draglayer .nsewdrag,
        .js-plotly-plot .plotly .cursor-crosshair,
        .js-plotly-plot .plotly .cursor-move,
        .js-plotly-plot .plotly .cursor-pointer {
            cursor: default !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def is_gcs_path(path: str) -> bool:
    return str(path).startswith("gs://")


@st.cache_resource(show_spinner=False)
def gcs_filesystem() -> gcsfs.GCSFileSystem:
    return gcsfs.GCSFileSystem()


def path_exists(path: str) -> bool:
    if is_gcs_path(path):
        return gcs_filesystem().exists(path)
    return Path(path).exists()


def join_data_path(base: str, *parts: str) -> str:
    clean_base = str(base).rstrip("/")
    clean_parts = [str(part).strip("/") for part in parts if str(part).strip("/")]
    if not clean_parts:
        return clean_base
    return f"{clean_base}/{'/'.join(clean_parts)}"


def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


def cache_target_for_path(path: str) -> Path:
    source = Path(path)
    stat = source.stat()
    cache_name = f"{source.stem}-{stat.st_size}-{int(stat.st_mtime)}{source.suffix}"
    return CACHE_DIR / cache_name


def copy_file_to_cache(source: Path, target: Path, progress=None) -> None:
    chunk_size = 8 * 1024 * 1024
    total_size = source.stat().st_size
    copied = 0
    tmp_target = target.with_name(f"{target.name}.tmp")

    try:
        with source.open("rb") as src, tmp_target.open("wb") as dst:
            while True:
                chunk = src.read(chunk_size)
                if not chunk:
                    break
                dst.write(chunk)
                copied += len(chunk)
                if progress is not None and total_size > 0:
                    progress.progress(
                        min(copied / total_size, 1.0),
                        text=f"{format_bytes(copied)} / {format_bytes(total_size)}",
                    )
        tmp_target.replace(target)
    except Exception:
        tmp_target.unlink(missing_ok=True)
        raise


def cached_input_path(path: str, progress=None):
    if is_gcs_path(path):
        return path

    source = Path(path)
    if not USE_LOCAL_CACHE or not source.exists() or source.is_dir():
        return source

    target = cache_target_for_path(path)
    if target.exists() and target.stat().st_size == source.stat().st_size:
        return target

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    copy_file_to_cache(source, target, progress=progress)
    return target


def prepare_catalog_cache(paths: list[str]) -> None:
    if not USE_LOCAL_CACHE:
        return

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    copied_any = False

    with st.status("Preparando catálogos en cache local", expanded=True) as status:
        st.write(f"Cache: `{CACHE_DIR}`")
        for path in paths:
            if is_gcs_path(path):
                st.write(f"Usando catálogo en Cloud Storage: `{path}`")
                continue

            source = Path(path)
            if not source.exists() or source.is_dir():
                continue

            size = source.stat().st_size
            target = cache_target_for_path(path)
            if target.exists() and target.stat().st_size == size:
                st.write(f"Ya en cache: `{source.name}` ({format_bytes(size)})")
                continue

            copied_any = True
            st.write(f"Copiando `{source.name}` ({format_bytes(size)}) desde el origen configurado...")
            progress = st.progress(0.0, text=f"0 B / {format_bytes(size)}")
            try:
                cached_input_path(path, progress=progress)
            except TimeoutError as exc:
                progress.empty()
                st.error(
                    "El origen de datos ha agotado el tiempo al servir este fichero. "
                    "Si usas una unidad sincronizada, márcalo como disponible sin conexión y vuelve a intentarlo."
                )
                status.update(label="No se pudo copiar un catálogo", state="error")
                raise exc
            progress.empty()
            st.write(f"Copiado: `{source.name}` ({format_bytes(size)})")

        if copied_any:
            status.update(label="Catálogos copiados a cache local", state="complete")
        else:
            status.update(label="Catálogos ya disponibles en cache local", state="complete")


def detect_pca_columns(df: pd.DataFrame) -> list[str]:
    def pca_index(column: str) -> int:
        try:
            return int(column.removeprefix("feat_pca_"))
        except ValueError:
            return 10_000

    return sorted(
        [column for column in df.columns if column.startswith("feat_pca_")],
        key=pca_index,
    )


def normalize_object_ids(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip()
    normalized = normalized.str.replace(r"\.0$", "", regex=True)
    normalized = normalized.str.strip("'\"")
    return normalized


def ensure_object_id_from_id_str(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "object_id" not in df.columns:
        if "id_str" not in df.columns:
            raise ValueError("El parquet PCA debe contener id_str u object_id.")
        df["object_id"] = df["id_str"].astype("string").str.split("_").str[-1]
    df["object_id"] = normalize_object_ids(df["object_id"])
    return df


@st.cache_data(show_spinner=False)
def load_pca_catalog(parquet_path: str) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_parquet(cached_input_path(parquet_path))
    df = ensure_object_id_from_id_str(df)
    feature_cols = detect_pca_columns(df)

    if not feature_cols:
        raise ValueError("No se encontraron columnas feat_pca_* en el parquet PCA.")

    keep_cols = ["id_str", "object_id", "hdf5_loc", *feature_cols]
    keep_cols = [column for column in keep_cols if column in df.columns]

    work_df = (
        df[keep_cols]
        .dropna(subset=feature_cols)
        .reset_index(drop=True)
        .copy()
    )
    return work_df, feature_cols


@st.cache_data(show_spinner=False)
def load_lens_catalog(lens_path: str, only_grade_a: bool) -> pd.DataFrame:
    lens_df = pd.read_csv(cached_input_path(lens_path), dtype={"object_id": "string"})
    if "object_id" not in lens_df.columns:
        raise ValueError("El catálogo de lentes debe contener object_id.")

    lens_df = lens_df.copy()
    lens_df["object_id"] = normalize_object_ids(lens_df["object_id"])

    if only_grade_a and "grade" in lens_df.columns:
        lens_df = lens_df[lens_df["grade"].astype(str).str.upper() == "A"].copy()

    columns = [column for column in ("object_id", "id_str", "grade") if column in lens_df.columns]
    return lens_df[columns].dropna(subset=["object_id"]).drop_duplicates("object_id")


def merge_lens_flags(work_df: pd.DataFrame, lens_df: pd.DataFrame) -> pd.DataFrame:
    lens_meta = lens_df.copy()
    rename_map = {}
    if "id_str" in lens_meta.columns:
        rename_map["id_str"] = "lens_id_str"
    if "grade" in lens_meta.columns:
        rename_map["grade"] = "lens_grade"
    lens_meta = lens_meta.rename(columns=rename_map)

    merged = work_df.merge(lens_meta, on="object_id", how="left")
    lens_object_ids = set(normalize_object_ids(lens_df["object_id"]).dropna())
    merged["is_lens"] = normalize_object_ids(merged["object_id"]).isin(lens_object_ids)

    if "id_str" in work_df.columns and "id_str" in lens_df.columns:
        lens_id_strs = set(lens_df["id_str"].astype("string").str.strip().dropna())
        merged["is_lens"] = merged["is_lens"] | (
            merged["id_str"].astype("string").str.strip().isin(lens_id_strs)
        )

    return merged


@st.cache_data(show_spinner=True)
def run_birch_clustering(
    parquet_path: str,
    lens_path: str,
    only_grade_a: bool,
    threshold: float,
    branching_factor: int,
    batch_size: int,
) -> tuple[pd.DataFrame, list[str]]:
    work_df, feature_cols = load_pca_catalog(parquet_path)
    lens_df = load_lens_catalog(lens_path, only_grade_a)

    scaler = StandardScaler()
    for start in range(0, len(work_df), batch_size):
        end = min(start + batch_size, len(work_df))
        x_batch = work_df.iloc[start:end][feature_cols].to_numpy(
            dtype=np.float32,
            copy=True,
        )
        scaler.partial_fit(x_batch)

    cluster_model = Birch(
        threshold=threshold,
        branching_factor=branching_factor,
        n_clusters=None,
        compute_labels=False,
    )
    for start in range(0, len(work_df), batch_size):
        end = min(start + batch_size, len(work_df))
        x_batch = work_df.iloc[start:end][feature_cols].to_numpy(
            dtype=np.float32,
            copy=True,
        )
        x_batch = scaler.transform(x_batch, copy=False)
        cluster_model.partial_fit(x_batch)

    cluster_model.partial_fit()

    labels = np.empty(len(work_df), dtype=np.int32)
    for start in range(0, len(work_df), batch_size):
        end = min(start + batch_size, len(work_df))
        x_batch = work_df.iloc[start:end][feature_cols].to_numpy(
            dtype=np.float32,
            copy=True,
        )
        x_batch = scaler.transform(x_batch, copy=False)
        labels[start:end] = cluster_model.predict(x_batch)

    clustered_df = work_df.copy()
    clustered_df["cluster"] = labels
    clustered_df = merge_lens_flags(clustered_df, lens_df)
    clustered_df.attrs["n_subclusters"] = len(cluster_model.subcluster_centers_)
    return clustered_df, feature_cols


def build_cluster_summary(clustered_df: pd.DataFrame) -> pd.DataFrame:
    summary_df = (
        clustered_df.groupby("cluster")
        .agg(
            n_objects=("object_id", "size"),
            n_lenses=("is_lens", "sum"),
        )
        .reset_index()
    )
    summary_df["lens_rate"] = summary_df["n_lenses"] / summary_df["n_objects"]
    summary_df = summary_df.sort_values(
        ["lens_rate", "n_lenses", "n_objects", "cluster"],
        ascending=[False, False, False, True],
    )
    return summary_df


def format_cluster_option(row: pd.Series) -> str:
    return (
        f"Cluster {int(row['cluster'])} | "
        f"{int(row['n_objects']):,} objetos | "
        f"{int(row['n_lenses']):,} lentes | "
        f"{row['lens_rate'] * 100:.3f}%"
    )


def sample_for_display(df: pd.DataFrame, max_objects: int) -> pd.DataFrame:
    if len(df) <= max_objects:
        return df.copy()

    working = df.copy()
    working["_sample_priority"] = 0

    if "is_lens" in working.columns:
        working.loc[working["is_lens"], "_sample_priority"] = 1
    if "is_canonical" in working.columns:
        working.loc[working["is_canonical"], "_sample_priority"] = 2
    if "is_anomaly" in working.columns:
        working.loc[working["is_anomaly"], "_sample_priority"] = 2

    priority_df = working[working["_sample_priority"] > 0].sort_values(
        ["_sample_priority"],
        ascending=False,
    )
    if len(priority_df) >= max_objects:
        return priority_df.head(max_objects).drop(columns=["_sample_priority"]).copy()

    remaining_df = working[working["_sample_priority"] == 0]
    n_remaining = max_objects - len(priority_df)
    sampled_remaining = remaining_df.sample(
        n=min(n_remaining, len(remaining_df)),
        random_state=42,
    )
    sampled = pd.concat([priority_df, sampled_remaining], ignore_index=True)
    return sampled.drop(columns=["_sample_priority"]).copy()


@st.cache_data(show_spinner=True)
def compute_umap_embedding(
    data: pd.DataFrame,
    selected_features: list[str],
    n_neighbors: int,
    min_dist: float,
) -> pd.DataFrame:
    clean = data.dropna(subset=selected_features).copy()
    if clean.empty:
        return clean

    n_neighbors = min(n_neighbors, max(2, len(clean) - 1))
    scaled = StandardScaler().fit_transform(clean[selected_features])
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=42,
    )
    embedding = reducer.fit_transform(scaled)
    clean["umap_1"] = embedding[:, 0]
    clean["umap_2"] = embedding[:, 1]
    return clean


def add_cluster_extreme_roles(
    data: pd.DataFrame,
    selected_features: list[str],
) -> pd.DataFrame:
    marked = data.copy()
    marked["is_canonical"] = False
    marked["is_anomaly"] = False
    marked["point_role"] = np.where(marked["is_lens"], "Lens", "No lens")

    clean = marked.dropna(subset=selected_features).copy()
    if len(clean) < 2:
        return marked

    scaled = StandardScaler().fit_transform(clean[selected_features])
    centroid = scaled.mean(axis=0, keepdims=True)
    distances = np.linalg.norm(scaled - centroid, axis=1)

    canonical_index = clean.index[int(np.argmin(distances))]
    anomaly_index = clean.index[int(np.argmax(distances))]

    marked.loc[canonical_index, "is_canonical"] = True
    marked.loc[anomaly_index, "is_anomaly"] = True
    marked.loc[canonical_index, "point_role"] = "Canonical"
    marked.loc[anomaly_index, "point_role"] = "Anomaly"
    marked["dist_to_cluster_centroid"] = np.nan
    marked.loc[clean.index, "dist_to_cluster_centroid"] = distances
    return marked


def build_umap_signature(
    selected_cluster: int,
    selected_features: list[str],
    n_neighbors: int,
    min_dist: float,
    max_objects: int,
    cluster_params: dict,
) -> tuple:
    return (
        PARQUET_PATH,
        LENS_PATH,
        bool(cluster_params["only_grade_a"]),
        float(cluster_params["threshold"]),
        int(cluster_params["branching_factor"]),
        int(cluster_params["batch_size"]),
        int(selected_cluster),
        tuple(selected_features),
        int(n_neighbors),
        round(float(min_dist), 4),
        int(max_objects),
    )


@st.cache_data(show_spinner=False)
def image_path_exists(path: str) -> bool:
    return path_exists(path)


def normalize_filename_object_id(object_id: object) -> str:
    value = str(object_id).strip()
    if value.endswith(".0"):
        value = value[:-2]
    if value.startswith("-"):
        return f"NEG{value[1:]}"
    return value.replace("-", "NEG")


def morphology_cutout_path(id_str: object, object_id: object = None) -> str | None:
    if pd.isna(id_str):
        return None

    parts = str(id_str).strip().split("_")
    if len(parts) == 2:
        tile_index = parts[0]
        object_id_part = parts[1]
    elif len(parts) >= 3:
        tile_index = parts[-2]
        object_id_part = parts[-1]
    else:
        return None

    candidate_object_ids = [object_id_part]
    if object_id is not None and not pd.isna(object_id):
        candidate_object_ids.append(object_id)

    for candidate_object_id in dict.fromkeys(
        normalize_filename_object_id(candidate) for candidate in candidate_object_ids
    ):
        filename = f"{tile_index}_{candidate_object_id}_gz_arcsinh_vis_only.jpg"
        path = join_data_path(CUTOUT_BASE, tile_index, filename)
        if image_path_exists(path):
            return path

    return None


def lens_image_path(lens_id_str: object) -> str | None:
    if pd.isna(lens_id_str):
        return None
    path = join_data_path(LENS_IMG_BASE, str(lens_id_str), "rgb_1.png")
    return path if image_path_exists(path) else None


@st.cache_data(show_spinner=False)
def load_morphology_object(morph_path: str, object_id: str) -> pd.DataFrame:
    if is_gcs_path(morph_path):
        return pd.DataFrame()

    path = cached_input_path(morph_path)
    if not Path(path).exists() or not object_id:
        return pd.DataFrame()

    dataset = ds.dataset(path, format="parquet")
    if "object_id" not in dataset.schema.names:
        return pd.DataFrame()

    field_type = dataset.schema.field("object_id").type
    filter_value: object = object_id
    if pa.types.is_integer(field_type):
        try:
            filter_value = int(object_id)
        except ValueError:
            return pd.DataFrame()

    table = dataset.to_table(filter=ds.field("object_id") == filter_value)
    if table.num_rows == 0 and not pa.types.is_string(field_type):
        table = dataset.to_table(filter=ds.field("object_id") == object_id)
    if table.num_rows == 0:
        return pd.DataFrame()

    return table.slice(0, 1).to_pandas()


def selected_point_index(event: object) -> int | None:
    if not event:
        return None

    try:
        points = event["selection"]["points"]
    except (KeyError, TypeError):
        return None

    if not points:
        return None

    customdata = points[0].get("customdata")
    if isinstance(customdata, Iterable) and not isinstance(customdata, str):
        customdata = list(customdata)[0] if customdata else None

    try:
        return int(customdata)
    except (TypeError, ValueError):
        return None


@st.cache_data(show_spinner=False)
def load_image_bytes(path: str) -> bytes:
    if is_gcs_path(path):
        with gcs_filesystem().open(path, "rb") as image_file:
            return image_file.read()
    return Path(path).read_bytes()


def show_image(path: str, caption: str) -> None:
    try:
        image = Image.open(BytesIO(load_image_bytes(path)))
    except Exception as exc:
        st.warning(f"No se pudo abrir la imagen: {exc}")
        return
    st.image(image, caption=caption, use_container_width=True)


def show_object_details(row: pd.Series, selected_features: list[str]) -> None:
    st.subheader("Objeto seleccionado")

    details = {
        "id_str": row.get("id_str", ""),
        "object_id": row.get("object_id", ""),
        "cluster": row.get("cluster", ""),
        "is_lens": bool(row.get("is_lens", False)),
        "is_canonical": bool(row.get("is_canonical", False)),
        "is_anomaly": bool(row.get("is_anomaly", False)),
        "lens_grade": row.get("lens_grade", ""),
        "dist_to_cluster_centroid": row.get("dist_to_cluster_centroid", np.nan),
        "umap_1": row.get("umap_1", np.nan),
        "umap_2": row.get("umap_2", np.nan),
    }
    st.dataframe(pd.DataFrame([details]), use_container_width=True, hide_index=True)

    st.markdown("**Componentes PCA usadas**")
    st.dataframe(
        pd.DataFrame(
            [{"feature": feature, "value": row.get(feature)} for feature in selected_features]
        ),
        use_container_width=True,
        hide_index=True,
    )

    morphology_df = load_morphology_object(MORPH_PATH, str(row.get("object_id", "")))
    if not morphology_df.empty:
        with st.expander("Fila del catálogo morfológico", expanded=False):
            morph_display = (
                morphology_df.iloc[0]
                .dropna()
                .astype(str)
                .reset_index()
                .rename(columns={"index": "campo", 0: "valor"})
            )
            st.dataframe(morph_display, use_container_width=True, hide_index=True)

    cutout_path = morphology_cutout_path(row.get("id_str"), row.get("object_id"))
    lens_path = lens_image_path(row.get("lens_id_str"))
    if lens_path is None and bool(row.get("is_lens", False)):
        lens_path = lens_image_path(row.get("id_str"))

    if cutout_path is None and lens_path is None:
        st.info("No se encontró imagen asociada en las rutas configuradas.")
        return

    if cutout_path is not None:
        show_image(cutout_path, "Cutout morfológico")
    if lens_path is not None:
        show_image(lens_path, "Imagen strong lens")


def validate_paths() -> pd.DataFrame:
    rows = [
        ("MORPH_PATH", MORPH_PATH),
        ("PARQUET_PATH", PARQUET_PATH),
        ("CUTOUT_BASE", CUTOUT_BASE),
        ("LENS_PATH", LENS_PATH),
        ("LENS_IMG_BASE", LENS_IMG_BASE),
    ]
    return pd.DataFrame(
        [
            {
                "name": name,
                "path": path,
                "exists": path_exists(path),
            }
            for name, path in rows
        ]
    )


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    inject_plot_cursor_css()
    st.title(APP_TITLE)

    with st.sidebar:
        st.header("Datos")
        with st.expander("Rutas configuradas", expanded=False):
            st.dataframe(validate_paths(), use_container_width=True, hide_index=True)
            st.caption(
                "La cache local solo copia catálogos individuales. "
                "CUTOUT_BASE y LENS_IMG_BASE no se copian; las imágenes se leen bajo demanda."
            )

        st.header("Lentes")
        only_grade_a = st.checkbox("Usar solo lentes grade A", value=True)

        st.header("Clusterización BIRCH")
        threshold = st.number_input("threshold", min_value=0.1, value=8.0, step=0.1)
        branching_factor = st.number_input(
            "branching_factor",
            min_value=2,
            value=2,
            step=1,
        )
        batch_size = st.number_input(
            "batch_size",
            min_value=1_000,
            max_value=250_000,
            value=25_000,
            step=1_000,
        )
        run_clustering = st.button("Ejecutar clusterización", type="primary")

    required = [PARQUET_PATH, LENS_PATH]
    missing = [path for path in required if not path_exists(path)]
    if missing:
        st.error(
            "No se encuentran los ficheros necesarios. Revisa "
            "las variables de entorno MORPH_PATH, PARQUET_PATH, LENS_PATH, "
            "CUTOUT_BASE y LENS_IMG_BASE."
        )
        st.code("\n".join(missing), language="text")
        st.stop()

    if run_clustering:
        st.session_state["cluster_ready"] = True
        st.session_state["cluster_params"] = {
            "only_grade_a": only_grade_a,
            "threshold": threshold,
            "branching_factor": int(branching_factor),
            "batch_size": int(batch_size),
        }

    if not st.session_state.get("cluster_ready"):
        st.info("Pulsa **Ejecutar clusterización** para generar los clusters BIRCH.")
        st.stop()

    # Solo cacheamos catálogos individuales. Las carpetas de imágenes se leen bajo demanda.
    prepare_catalog_cache([PARQUET_PATH, LENS_PATH])

    params = st.session_state["cluster_params"]
    try:
        clustered_df, pca_columns = run_birch_clustering(
            PARQUET_PATH,
            LENS_PATH,
            params["only_grade_a"],
            float(params["threshold"]),
            int(params["branching_factor"]),
            int(params["batch_size"]),
        )
    except TimeoutError as exc:
        st.error(
            "El origen de datos ha agotado el tiempo leyendo un catálogo. "
            "Si usas una unidad sincronizada, marca los ficheros como disponibles sin conexión o copia primero a la cache local."
        )
        st.exception(exc)
        st.stop()
    except OSError as exc:
        st.error(
            "No se pudo leer un catálogo desde las rutas configuradas. "
            "Si usas una unidad sincronizada, comprueba que los ficheros estén disponibles sin conexión."
        )
        st.exception(exc)
        st.stop()

    cluster_summary_df = build_cluster_summary(clustered_df)
    cluster_summary_df["option"] = cluster_summary_df.apply(format_cluster_option, axis=1)

    left_metric, middle_metric, right_metric = st.columns(3)
    left_metric.metric("Objetos clusterizados", f"{len(clustered_df):,}")
    middle_metric.metric("Clusters", f"{clustered_df['cluster'].nunique():,}")
    right_metric.metric("Lentes", f"{int(clustered_df['is_lens'].sum()):,}")

    with st.expander("Resumen de clusters", expanded=False):
        summary_display = cluster_summary_df.copy()
        summary_display["lens_rate"] = (summary_display["lens_rate"] * 100).round(3)
        st.dataframe(
            summary_display[["cluster", "n_objects", "n_lenses", "lens_rate"]],
            use_container_width=True,
            hide_index=True,
        )

    with st.sidebar:
        st.header("UMAP")
        selected_option = st.selectbox(
            "Cluster",
            cluster_summary_df["option"].tolist(),
        )
        selected_cluster = int(
            cluster_summary_df.loc[
                cluster_summary_df["option"] == selected_option,
                "cluster",
            ].iloc[0]
        )

        default_features = [
            feature for feature in DEFAULT_CLUSTER_FEATURES if feature in pca_columns
        ] or pca_columns[: min(6, len(pca_columns))]
        selected_features = st.multiselect(
            "Componentes PCA",
            pca_columns,
            default=default_features,
        )
        n_neighbors = st.slider("n_neighbors", 2, 100, 25)
        min_dist = st.slider("min_dist", 0.0, 1.0, 0.15, step=0.01)
        max_objects = st.slider("Máximo de objetos", 100, 100_000, 20_000, step=100)

    if not selected_features:
        st.warning("Selecciona al menos una componente PCA para construir UMAP.")
        st.stop()

    cluster_df = clustered_df[clustered_df["cluster"] == selected_cluster].copy()
    umap_signature = build_umap_signature(
        selected_cluster=selected_cluster,
        selected_features=selected_features,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        max_objects=int(max_objects),
        cluster_params=params,
    )
    stored_signature = st.session_state.get("umap_signature")
    needs_recalculation = stored_signature != umap_signature

    button_label = "Calcular UMAP" if stored_signature is None else "Recalcular UMAP"
    recalculate_umap = st.sidebar.button(
        button_label,
        type="primary" if needs_recalculation else "secondary",
        disabled=(not selected_features) or (not needs_recalculation and "umap_embedding_df" in st.session_state),
    )

    if recalculate_umap:
        cluster_df = add_cluster_extreme_roles(cluster_df, selected_features)
        display_df = sample_for_display(cluster_df, int(max_objects))

        if len(display_df) < 3:
            st.warning("Se necesitan al menos 3 objetos para calcular UMAP.")
            st.stop()

        embedding_df = compute_umap_embedding(
            display_df,
            selected_features,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
        )

        if embedding_df.empty:
            st.warning("No quedan objetos con valores completos para las componentes seleccionadas.")
            st.stop()

        embedding_df = embedding_df.reset_index(drop=True)
        embedding_df["point_index"] = embedding_df.index
        st.session_state["umap_embedding_df"] = embedding_df
        st.session_state["umap_signature"] = umap_signature
        needs_recalculation = False

    if needs_recalculation or "umap_embedding_df" not in st.session_state:
        st.info("Pulsa **Calcular UMAP** o **Recalcular UMAP** para actualizar la visualización.")
        st.stop()

    embedding_df = st.session_state["umap_embedding_df"]

    cluster_left, cluster_middle, cluster_right, cluster_fourth = st.columns(4)
    cluster_left.metric("Objetos del cluster", f"{len(cluster_df):,}")
    cluster_middle.metric("Objetos en UMAP", f"{len(embedding_df):,}")
    cluster_right.metric("Lentes en UMAP", f"{int(embedding_df['is_lens'].sum()):,}")
    cluster_fourth.metric("Extremos", "2")

    hover_columns = [
        column
        for column in (
            "id_str",
            "object_id",
            "cluster",
            "point_role",
            "is_lens",
            "lens_grade",
            "dist_to_cluster_centroid",
        )
        if column in embedding_df.columns
    ]

    fig = px.scatter(
        embedding_df,
        x="umap_1",
        y="umap_2",
        color="point_role",
        symbol="point_role",
        custom_data=["point_index"],
        hover_data=hover_columns,
        color_discrete_map={
            "No lens": "#4c78a8",
            "Lens": "#d62728",
            "Canonical": "#2ca02c",
            "Anomaly": "#111111",
        },
        symbol_map={
            "No lens": "circle",
            "Lens": "star",
            "Canonical": "diamond",
            "Anomaly": "x",
        },
        category_orders={
            "point_role": ["No lens", "Lens", "Canonical", "Anomaly"],
        },
        labels={"umap_1": "UMAP 1", "umap_2": "UMAP 2", "point_role": "Tipo"},
        height=680,
    )
    fig.update_traces(marker={"size": 7, "opacity": 0.72})
    fig.update_traces(
        marker={"size": 12, "opacity": 0.98, "line": {"width": 1.5, "color": "white"}},
        selector={"name": "Lens"},
    )
    fig.update_traces(
        marker={"size": 14, "opacity": 1.0, "line": {"width": 2, "color": "white"}},
        selector={"name": "Canonical"},
    )
    fig.update_traces(
        marker={"size": 14, "opacity": 1.0, "line": {"width": 2, "color": "#ffcc00"}},
        selector={"name": "Anomaly"},
    )
    for trace in fig.data:
        opacity = getattr(trace.marker, "opacity", None) or 1.0
        trace.selected = {"marker": {"opacity": opacity}}
        trace.unselected = {"marker": {"opacity": opacity}}

    fig.update_layout(
        title=f"Cluster {selected_cluster} | UMAP",
        legend_title_text="Objeto",
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        clickmode="event+select",
        dragmode="zoom",
        uirevision=st.session_state.get("umap_signature"),
    )

    plot_col, detail_col = st.columns([2, 1])
    with plot_col:
        event = st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displaylogo": False,
                "scrollZoom": True,
                "doubleClick": "reset",
            },
            on_select="rerun",
            selection_mode="points",
            key="umap_selection",
        )

    selected_index = selected_point_index(event)
    with detail_col:
        if selected_index is None:
            st.info("Selecciona un punto del mapa para ver sus detalles e imagen.")
        else:
            show_object_details(embedding_df.loc[selected_index], selected_features)


if __name__ == "__main__":
    main()
