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
import plotly.graph_objects as go
import streamlit as st
import umap
from PIL import Image
from plotly.subplots import make_subplots
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
    "feat_pca_6",
    "feat_pca_10",
    "feat_pca_13",
    "feat_pca_27",
]
DEFAULT_LENS_GRADES = ["A", "B"]
LENS_GRADE_OPTIONS = ["A", "B", "C"]
SUMMARY_RANDOM_OBJECTS = 3
SUMMARY_LENS_OBJECTS = 5
SUMMARY_THUMBNAIL_WIDTH = 96
SUMMARY_HISTOGRAM_BINS = 24
SUMMARY_HISTOGRAM_FEATURE_LIMIT = 6


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
        .lens-status {
            border-radius: 8px;
            font-weight: 700;
            margin: 0.25rem 0 0.85rem 0;
            padding: 0.7rem 0.85rem;
        }
        .lens-status--yes {
            background: #fff1f2;
            border: 1px solid #e11d48;
            color: #9f1239;
        }
        .lens-status--no {
            background: #f8fafc;
            border: 1px solid #94a3b8;
            color: #334155;
        }
        .lens-status__label {
            display: block;
            font-size: 1rem;
            line-height: 1.25;
        }
        .lens-status__meta {
            display: block;
            font-size: 0.82rem;
            font-weight: 500;
            line-height: 1.25;
            margin-top: 0.2rem;
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

    with st.status("Preparing catalogues in local cache", expanded=True) as status:
        st.write(f"Cache: `{CACHE_DIR}`")
        for path in paths:
            if is_gcs_path(path):
                st.write(f"Using catalogue in Cloud Storage: `{path}`")
                continue

            source = Path(path)
            if not source.exists() or source.is_dir():
                continue

            size = source.stat().st_size
            target = cache_target_for_path(path)
            if target.exists() and target.stat().st_size == size:
                st.write(f"Already cached: `{source.name}` ({format_bytes(size)})")
                continue

            copied_any = True
            st.write(
                f"Copying `{source.name}` ({format_bytes(size)}) from the configured source..."
            )
            progress = st.progress(0.0, text=f"0 B / {format_bytes(size)}")
            try:
                cached_input_path(path, progress=progress)
            except TimeoutError as exc:
                progress.empty()
                st.error(
                    "The data source timed out while serving this file. "
                    "If you are using a synchronized drive, make it available offline and try again."
                )
                status.update(label="Could not copy a catalogue", state="error")
                raise exc
            progress.empty()
            st.write(f"Copied: `{source.name}` ({format_bytes(size)})")

        if copied_any:
            status.update(label="Catalogues copied to local cache", state="complete")
        else:
            status.update(label="Catalogues already available in local cache", state="complete")


def detect_pca_columns(df: pd.DataFrame) -> list[str]:
    return detect_pca_column_names(df.columns)


def detect_pca_column_names(columns: Iterable[str]) -> list[str]:
    def pca_index(column: str) -> int:
        try:
            return int(column.removeprefix("feat_pca_"))
        except ValueError:
            return 10_000

    return sorted(
        [column for column in columns if str(column).startswith("feat_pca_")],
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
            raise ValueError("The PCA parquet must contain id_str or object_id.")
        df["object_id"] = df["id_str"].astype("string").str.split("_").str[-1]
    df["object_id"] = normalize_object_ids(df["object_id"])
    return df


@st.cache_data(show_spinner=False)
def load_pca_catalog(parquet_path: str) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_parquet(cached_input_path(parquet_path))
    df = ensure_object_id_from_id_str(df)
    feature_cols = detect_pca_columns(df)

    if not feature_cols:
        raise ValueError("No feat_pca_* columns were found in the PCA parquet file.")

    keep_cols = ["id_str", "object_id", "hdf5_loc", *feature_cols]
    keep_cols = [column for column in keep_cols if column in df.columns]

    work_df = (
        df[keep_cols]
        .dropna(subset=feature_cols)
        .reset_index(drop=True)
        .copy()
    )
    return work_df, feature_cols


def normalize_lens_grades(grades: Iterable[str]) -> tuple[str, ...]:
    return tuple(
        sorted({str(grade).strip().upper() for grade in grades if str(grade).strip()})
    )


@st.cache_data(show_spinner=False)
def load_lens_catalog(lens_path: str, selected_grades: tuple[str, ...]) -> pd.DataFrame:
    lens_df = pd.read_csv(cached_input_path(lens_path), dtype={"object_id": "string"})
    if "object_id" not in lens_df.columns:
        raise ValueError("The lens catalogue must contain object_id.")

    lens_df = lens_df.copy()
    lens_df["object_id"] = normalize_object_ids(lens_df["object_id"])

    if selected_grades and "grade" in lens_df.columns:
        lens_df = lens_df[
            lens_df["grade"].astype(str).str.strip().str.upper().isin(selected_grades)
        ].copy()

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
    selected_grades: tuple[str, ...],
    threshold: float,
    branching_factor: int,
    batch_size: int,
) -> tuple[pd.DataFrame, list[str]]:
    work_df, feature_cols = load_pca_catalog(parquet_path)
    lens_df = load_lens_catalog(lens_path, selected_grades)

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
        ["n_lenses", "lens_rate", "n_objects", "cluster"],
        ascending=[False, False, False, True],
    )
    return summary_df


def format_cluster_option(row: pd.Series) -> str:
    return (
        f"Cluster {int(row['cluster'])} | "
        f"{int(row['n_objects']):,} objects | "
        f"{int(row['n_lenses']):,} lenses | "
        f"{row['lens_rate'] * 100:.3f}%"
    )


def default_cluster_option_index(cluster_summary_df: pd.DataFrame) -> int:
    eligible_positions = np.flatnonzero(cluster_summary_df["n_lenses"].to_numpy() > 1)
    return int(eligible_positions[0]) if len(eligible_positions) else 0


def lens_grade_sort_key(series: pd.Series) -> pd.Series:
    grade_order = {grade: index for index, grade in enumerate(LENS_GRADE_OPTIONS)}
    return (
        series.astype("string")
        .str.strip()
        .str.upper()
        .map(grade_order)
        .fillna(len(grade_order))
        .astype(int)
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
        cluster_lens_grades(cluster_params),
        float(cluster_params["threshold"]),
        int(cluster_params["branching_factor"]),
        int(cluster_params["batch_size"]),
        int(selected_cluster),
        tuple(selected_features),
        int(n_neighbors),
        round(float(min_dist), 4),
        int(max_objects),
    )


def cluster_lens_grades(cluster_params: dict) -> tuple[str, ...]:
    if "lens_grades" in cluster_params:
        return normalize_lens_grades(cluster_params["lens_grades"])
    if cluster_params.get("only_grade_a", True):
        return ("A",)
    return tuple(LENS_GRADE_OPTIONS)


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
        st.warning(f"Could not open the image: {exc}")
        return
    st.image(image, caption=caption, use_container_width=True)


def object_image_path(row: pd.Series, prefer_lens_image: bool = False) -> str | None:
    if prefer_lens_image:
        lens_path = lens_image_path(row.get("lens_id_str"))
        if lens_path is None:
            lens_path = lens_image_path(row.get("id_str"))
        if lens_path is not None:
            return lens_path

    return morphology_cutout_path(row.get("id_str"), row.get("object_id"))


def show_thumbnail(
    row: pd.Series | None,
    caption: str,
    prefer_lens_image: bool = False,
) -> None:
    if row is None:
        st.caption(caption)
        st.caption("No object")
        return

    path = object_image_path(row, prefer_lens_image=prefer_lens_image)
    if path is None:
        st.caption(caption)
        st.caption("No image")
        return

    try:
        image = Image.open(BytesIO(load_image_bytes(path)))
    except Exception:
        st.caption(caption)
        st.caption("No image")
        return

    st.image(image, caption=caption, width=SUMMARY_THUMBNAIL_WIDTH)


def show_thumbnail_group(
    title: str,
    rows: list[pd.Series],
    captions: list[str],
    prefer_lens_image: bool = False,
) -> None:
    st.caption(title)
    count = max(len(rows), 1)
    for column, row, caption in zip(st.columns(count), rows or [None], captions or [""]):
        with column:
            show_thumbnail(row, caption, prefer_lens_image=prefer_lens_image)


def cluster_visual_rows(
    cluster_df: pd.DataFrame,
    summary_features: list[str],
    cluster_id: int,
) -> tuple[pd.Series | None, pd.Series | None, list[pd.Series], list[pd.Series]]:
    marked = add_cluster_extreme_roles(cluster_df, summary_features)

    canonical_rows = marked[marked["is_canonical"]]
    anomaly_rows = marked[marked["is_anomaly"]]
    canonical_row = canonical_rows.iloc[0] if not canonical_rows.empty else None
    anomaly_row = anomaly_rows.iloc[0] if not anomaly_rows.empty else None

    used_object_ids = {
        str(row.get("object_id"))
        for row in (canonical_row, anomaly_row)
        if row is not None and not pd.isna(row.get("object_id"))
    }
    random_pool = cluster_df[
        ~cluster_df["object_id"].astype("string").isin(used_object_ids)
    ]
    if random_pool.empty:
        random_pool = cluster_df
    random_rows = random_pool.sample(
        n=min(SUMMARY_RANDOM_OBJECTS, len(random_pool)),
        random_state=int(cluster_id) + 17,
    )

    lens_rows = cluster_df[cluster_df["is_lens"]].copy()
    if not lens_rows.empty:
        if "lens_grade" not in lens_rows.columns:
            lens_rows["lens_grade"] = ""
        lens_rows["_grade_order"] = lens_grade_sort_key(lens_rows["lens_grade"])
        lens_rows = lens_rows.sort_values(
            ["_grade_order", "lens_grade", "object_id"],
            na_position="last",
        ).drop(columns=["_grade_order"])
    lens_rows = lens_rows.head(SUMMARY_LENS_OBJECTS)

    return (
        canonical_row,
        anomaly_row,
        [row for _, row in random_rows.iterrows()],
        [row for _, row in lens_rows.iterrows()],
    )


def build_cluster_histogram_figure(
    cluster_df: pd.DataFrame,
    features: list[str],
) -> go.Figure:
    n_columns = 2
    n_rows = int(np.ceil(len(features) / n_columns))
    fig = make_subplots(
        rows=n_rows,
        cols=n_columns,
        subplot_titles=features,
        horizontal_spacing=0.08,
        vertical_spacing=0.22,
    )

    lens_df = cluster_df[cluster_df["is_lens"]]
    non_lens_df = cluster_df[~cluster_df["is_lens"]]

    for index, feature in enumerate(features):
        row = (index // n_columns) + 1
        column = (index % n_columns) + 1
        show_legend = index == 0

        fig.add_trace(
            go.Histogram(
                x=non_lens_df[feature].dropna(),
                name="No lens",
                marker={"color": "#4c78a8"},
                opacity=0.62,
                histnorm="probability",
                nbinsx=SUMMARY_HISTOGRAM_BINS,
                showlegend=show_legend,
            ),
            row=row,
            col=column,
        )
        fig.add_trace(
            go.Histogram(
                x=lens_df[feature].dropna(),
                name="Lens",
                marker={"color": "#d62728"},
                opacity=0.72,
                histnorm="probability",
                nbinsx=SUMMARY_HISTOGRAM_BINS,
                showlegend=show_legend,
            ),
            row=row,
            col=column,
        )

    fig.update_layout(
        barmode="overlay",
        height=max(260, 170 * n_rows),
        margin={"l": 20, "r": 10, "t": 45, "b": 25},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=True, zeroline=False, title_text="prob.")
    return fig


def render_cluster_histograms(
    cluster_id: int,
    cluster_df: pd.DataFrame,
    summary_features: list[str],
) -> None:
    n_lenses = int(cluster_df["is_lens"].sum())
    n_non_lenses = len(cluster_df) - n_lenses
    if n_lenses == 0:
        return

    state_key = f"cluster_histograms_visible_{cluster_id}"
    button_label = (
        "Update PCA histograms"
        if st.session_state.get(state_key)
        else "Compute PCA histograms"
    )
    if st.button(button_label, key=f"cluster_histograms_button_{cluster_id}"):
        st.session_state[state_key] = True

    if not st.session_state.get(state_key):
        return

    if n_non_lenses == 0:
        st.info("This cluster does not contain non-lens objects for comparison.")
        return

    fig = build_cluster_histogram_figure(cluster_df, summary_features)
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displaylogo": False, "responsive": True},
        key=f"cluster_histograms_chart_{cluster_id}",
    )


def render_cluster_visual_summary(
    clustered_df: pd.DataFrame,
    cluster_summary_df: pd.DataFrame,
    pca_columns: list[str],
    selected_features: list[str],
) -> None:
    summary_features = [
        feature for feature in selected_features if feature in pca_columns
    ] or [feature for feature in DEFAULT_CLUSTER_FEATURES if feature in pca_columns]
    summary_features = summary_features or pca_columns[: min(4, len(pca_columns))]
    histogram_features = summary_features[:SUMMARY_HISTOGRAM_FEATURE_LIMIT]

    for _, summary_row in cluster_summary_df.iterrows():
        cluster_id = int(summary_row["cluster"])
        cluster_df = clustered_df[clustered_df["cluster"] == cluster_id].copy()
        canonical_row, anomaly_row, random_rows, lens_rows = cluster_visual_rows(
            cluster_df,
            summary_features,
            cluster_id,
        )

        with st.container(border=True):
            stats_cols = st.columns([1, 1, 1, 1])
            stats_cols[0].metric("Cluster", cluster_id)
            stats_cols[1].metric("Objects", f"{int(summary_row['n_objects']):,}")
            stats_cols[2].metric("Lenses", f"{int(summary_row['n_lenses']):,}")
            stats_cols[3].metric("Density", f"{summary_row['lens_rate'] * 100:.3f}%")

            image_cols = st.columns([2, 3, 5])
            with image_cols[0]:
                show_thumbnail_group(
                    "Canonical / anomalous",
                    [canonical_row, anomaly_row],
                    ["Canonical", "Anomalous"],
                )
            with image_cols[1]:
                show_thumbnail_group(
                    "Random",
                    random_rows,
                    [f"Random {index + 1}" for index in range(len(random_rows))],
                )
            with image_cols[2]:
                lens_captions = []
                for row in lens_rows:
                    lens_grade = row.get("lens_grade", "")
                    if pd.isna(lens_grade) or not str(lens_grade).strip():
                        lens_captions.append("Grade ?")
                    else:
                        lens_captions.append(f"Grade {str(lens_grade).strip()}")
                show_thumbnail_group(
                    "Confirmed lenses",
                    lens_rows,
                    lens_captions,
                    prefer_lens_image=True,
                )
            render_cluster_histograms(cluster_id, cluster_df, histogram_features)


def show_lens_status(row: pd.Series) -> None:
    is_lens = bool(row.get("is_lens", False))
    lens_grade = row.get("lens_grade", "")
    lens_grade_text = ""
    if not pd.isna(lens_grade) and str(lens_grade).strip():
        lens_grade_text = f"Grade: {lens_grade}"

    if is_lens:
        label = "LENS"
        css_class = "lens-status--yes"
        meta = lens_grade_text or "Object present in the strong-lensing catalogue."
    else:
        label = "NOT LENS"
        css_class = "lens-status--no"
        meta = "Object not marked as a lens in the joined catalogue."

    st.markdown(
        f"""
        <div class="lens-status {css_class}">
            <span class="lens-status__label">{label}</span>
            <span class="lens-status__meta">{meta}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_object_details(row: pd.Series, selected_features: list[str]) -> None:
    st.subheader("Selected object")
    show_lens_status(row)

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

    st.markdown("**Selected PCA components**")
    st.dataframe(
        pd.DataFrame(
            [{"feature": feature, "value": row.get(feature)} for feature in selected_features]
        ),
        use_container_width=True,
        hide_index=True,
    )

    morphology_df = load_morphology_object(MORPH_PATH, str(row.get("object_id", "")))
    if not morphology_df.empty:
        with st.expander("Morphology catalogue row", expanded=False):
            morph_display = morphology_df.iloc[0].dropna().astype(str).reset_index()
            morph_display = (
                morph_display.rename(columns={"index": "field", morph_display.columns[-1]: "value"})
            )
            st.dataframe(morph_display, use_container_width=True, hide_index=True)

    cutout_path = morphology_cutout_path(row.get("id_str"), row.get("object_id"))
    lens_path = lens_image_path(row.get("lens_id_str"))
    if lens_path is None and bool(row.get("is_lens", False)):
        lens_path = lens_image_path(row.get("id_str"))

    if cutout_path is None and lens_path is None:
        st.info("No associated image was found in the configured paths.")
        return

    if cutout_path is not None:
        show_image(cutout_path, "Morphology cutout")
    if lens_path is not None:
        show_image(lens_path, "Strong-lens image")


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

    required = [PARQUET_PATH, LENS_PATH]
    missing = [path for path in required if not path_exists(path)]
    if missing:
        st.error(
            "Required files were not found. Check the "
            "MORPH_PATH, PARQUET_PATH, LENS_PATH, "
            "CUTOUT_BASE and LENS_IMG_BASE environment variables."
        )
        st.code("\n".join(missing), language="text")
        st.stop()

    with st.sidebar:
        st.header("Data")
        with st.expander("Configured paths", expanded=False):
            st.dataframe(validate_paths(), use_container_width=True, hide_index=True)
            st.caption(
                "The local cache only copies individual catalogues. "
                "CUTOUT_BASE and LENS_IMG_BASE are not copied; images are read on demand."
            )

        st.header("Lenses")
        selected_lens_grades = st.multiselect(
            "Lens grades",
            LENS_GRADE_OPTIONS,
            default=DEFAULT_LENS_GRADES,
        )
        selected_lens_grades = normalize_lens_grades(selected_lens_grades)

        st.header("BIRCH clustering")
        st.caption("BIRCH clustering always uses all available PCA components.")
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
        run_clustering = st.button("Run clustering", type="primary")

    if run_clustering:
        if not selected_lens_grades:
            st.warning("Select at least one lens grade before clustering.")
            st.stop()

        st.session_state["cluster_ready"] = True
        st.session_state["cluster_params"] = {
            "lens_grades": selected_lens_grades,
            "threshold": threshold,
            "branching_factor": int(branching_factor),
            "batch_size": int(batch_size),
        }

    if not st.session_state.get("cluster_ready"):
        st.info("Click **Run clustering** to generate the BIRCH clusters.")
        st.stop()

    # Only individual catalogues are cached. Image folders are read on demand.
    prepare_catalog_cache([PARQUET_PATH, LENS_PATH])

    params = st.session_state["cluster_params"]
    lens_grades = cluster_lens_grades(params)
    try:
        clustered_df, pca_columns = run_birch_clustering(
            PARQUET_PATH,
            LENS_PATH,
            lens_grades,
            float(params["threshold"]),
            int(params["branching_factor"]),
            int(params["batch_size"]),
        )
    except TimeoutError as exc:
        st.error(
            "The data source timed out while reading a catalogue. "
            "If you are using a synchronized drive, make the files available offline or copy them to local cache first."
        )
        st.exception(exc)
        st.stop()
    except OSError as exc:
        st.error(
            "Could not read a catalogue from the configured paths. "
            "If you are using a synchronized drive, check that the files are available offline."
        )
        st.exception(exc)
        st.stop()

    cluster_summary_df = build_cluster_summary(clustered_df)
    cluster_summary_df["option"] = cluster_summary_df.apply(format_cluster_option, axis=1)

    left_metric, middle_metric, right_metric = st.columns(3)
    left_metric.metric("Clustered objects", f"{len(clustered_df):,}")
    middle_metric.metric("Clusters", f"{clustered_df['cluster'].nunique():,}")
    right_metric.metric("Lenses", f"{int(clustered_df['is_lens'].sum()):,}")
    st.caption(f"Lens grades used: {', '.join(lens_grades)}")
    st.caption(
        "PCA components used for clustering: "
        f"all {len(pca_columns)} available components"
    )

    with st.sidebar:
        st.header("UMAP")
        selected_option = st.selectbox(
            "Cluster",
            cluster_summary_df["option"].tolist(),
            index=default_cluster_option_index(cluster_summary_df),
        )
        selected_cluster = int(
            cluster_summary_df.loc[
                cluster_summary_df["option"] == selected_option,
                "cluster",
            ].iloc[0]
        )

        default_features = [
            feature for feature in DEFAULT_CLUSTER_FEATURES if feature in pca_columns
        ] or pca_columns[: min(4, len(pca_columns))]
        selected_features = st.multiselect(
            "PCA components",
            pca_columns,
            default=default_features,
        )
        n_neighbors = st.slider("n_neighbors", 2, 100, 25)
        min_dist = st.slider("min_dist", 0.0, 1.0, 0.15, step=0.01)
        max_objects = st.slider("Maximum objects", 100, 100_000, 20_000, step=100)

    if not selected_features:
        st.warning("Select at least one PCA component to build UMAP.")
        st.stop()

    with st.expander("Cluster summary", expanded=False):
        summary_display = cluster_summary_df.copy()
        summary_display["lens_rate"] = (summary_display["lens_rate"] * 100).round(3)
        st.dataframe(
            summary_display[["cluster", "n_objects", "n_lenses", "lens_rate"]],
            use_container_width=True,
            hide_index=True,
        )
        render_cluster_visual_summary(
            clustered_df,
            cluster_summary_df,
            pca_columns,
            selected_features,
        )

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

    button_label = "Compute UMAP" if stored_signature is None else "Recompute UMAP"
    recalculate_umap = st.sidebar.button(
        button_label,
        type="primary" if needs_recalculation else "secondary",
        disabled=(not selected_features) or (not needs_recalculation and "umap_embedding_df" in st.session_state),
    )

    if recalculate_umap:
        cluster_df = add_cluster_extreme_roles(cluster_df, selected_features)
        display_df = sample_for_display(cluster_df, int(max_objects))

        if len(display_df) < 3:
            st.warning("At least 3 objects are required to compute UMAP.")
            st.stop()

        embedding_df = compute_umap_embedding(
            display_df,
            selected_features,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
        )

        if embedding_df.empty:
            st.warning("No objects remain with complete values for the selected components.")
            st.stop()

        embedding_df = embedding_df.reset_index(drop=True)
        embedding_df["point_index"] = embedding_df.index
        st.session_state["umap_embedding_df"] = embedding_df
        st.session_state["umap_signature"] = umap_signature
        needs_recalculation = False

    if needs_recalculation or "umap_embedding_df" not in st.session_state:
        st.info("Click **Compute UMAP** or **Recompute UMAP** to update the visualization.")
        st.stop()

    embedding_df = st.session_state["umap_embedding_df"]

    cluster_left, cluster_middle, cluster_right, cluster_fourth = st.columns(4)
    cluster_left.metric("Cluster objects", f"{len(cluster_df):,}")
    cluster_middle.metric("Objects in UMAP", f"{len(embedding_df):,}")
    cluster_right.metric("Lenses in UMAP", f"{int(embedding_df['is_lens'].sum()):,}")
    cluster_fourth.metric("Extremes", "2")

    embedding_df = embedding_df.copy()
    if "lens_grade" in embedding_df.columns:
        lens_grade_marker = (
            embedding_df["lens_grade"]
            .astype("string")
            .str.strip()
            .str.upper()
            .str[:1]
        )
        lens_grade_marker = lens_grade_marker.where(
            lens_grade_marker.isin(LENS_GRADE_OPTIONS),
            "?",
        )
    else:
        lens_grade_marker = pd.Series("?", index=embedding_df.index)
    embedding_df["lens_grade_marker"] = np.where(
        embedding_df["is_lens"],
        lens_grade_marker.fillna("?"),
        "",
    )

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
            "Lens": "circle",
            "Canonical": "diamond",
            "Anomaly": "x",
        },
        category_orders={
            "point_role": ["No lens", "Lens", "Canonical", "Anomaly"],
        },
        labels={"umap_1": "UMAP 1", "umap_2": "UMAP 2", "point_role": "Type"},
        height=680,
    )
    fig.update_traces(marker={"size": 7, "opacity": 0.72})
    fig.update_traces(
        marker={"size": 17, "opacity": 0.98, "line": {"width": 1.5, "color": "white"}},
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

    for lens_row in embedding_df[embedding_df["lens_grade_marker"] != ""].itertuples():
        fig.add_annotation(
            x=lens_row.umap_1,
            y=lens_row.umap_2,
            text=lens_row.lens_grade_marker,
            showarrow=False,
            font={"size": 10, "color": "white", "family": "Arial Black"},
            xanchor="center",
            yanchor="middle",
            captureevents=False,
        )

    fig.update_layout(
        title=f"Cluster {selected_cluster} | UMAP",
        legend_title_text="Object",
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
            st.info("Select a point on the map to view its details and image.")
        else:
            show_object_details(embedding_df.loc[selected_index], selected_features)


if __name__ == "__main__":
    main()
