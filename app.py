from __future__ import annotations

import base64
import hashlib
import html
import json
import logging
import multiprocessing as mp
import os
import pickle
import queue
import tempfile
import time
from io import BytesIO
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image


APP_TITLE = "Euclid UMAP Explorer"
APP_VERSION = "v0.1.6"
EUCLID_LOGO_PATH = Path(__file__).parent / "assets" / "euclid_logo.png"
EUCLID_FAVICON_PATH = Path(__file__).parent / "assets" / "favicon.png"

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
MAX_ALGORITHM_SECONDS = int(os.getenv("EUCLID_MAX_ALGORITHM_SECONDS", "600"))

DEFAULT_CLUSTER_FEATURES = [
    "feat_pca_6",
    "feat_pca_0",
    "feat_pca_1",
    "feat_pca_27",
    "feat_pca_12",
    "feat_pca_29",
]
DEFAULT_LENS_GRADES = ["A", "B", "C"]
LENS_GRADE_OPTIONS = ["A", "B", "C"]
SUMMARY_RANDOM_OBJECTS = 3
SUMMARY_LENS_OBJECTS = 5
SUMMARY_THUMBNAIL_WIDTH = 90
SUMMARY_HISTOGRAM_BINS = 24
SUMMARY_HISTOGRAM_FEATURE_LIMIT = 6
SUMMARY_DISTPLOT_MAX_POINTS_PER_GROUP = 5_000
PCA_FILTER_OPERATORS = [">", ">=", "<", "<=", "between"]
LENS_GRADE_HELP = (
    "Grade A: secure or almost secure lens candidates with clear lensing features "
    "(expert score > 2.0).\n\n"
    "Grade B: probable lens candidates requiring additional confirmation "
    "(expert score > 1.5).\n\n"
    "Grade C: possible lens candidates with lens-like morphology that may still "
    "be explained by other physical structures (expert score > 1.0)."
)
PARAMETER_HELP = {
    "threshold": (
        "BIRCH radius threshold. Larger values create broader clusters; "
        "smaller values split the data into more compact groups."
    ),
    "branching_factor": (
        "Maximum number of subclusters kept at each BIRCH tree node. Higher "
        "values can preserve more structure, with extra memory cost."
    ),
    "batch_size": (
        "Number of catalogue rows processed at once during BIRCH fitting and "
        "prediction. Larger batches can be faster but use more memory."
    ),
    "n_neighbors": (
        "UMAP neighborhood size. Lower values emphasize local structure; higher "
        "values preserve broader global structure."
    ),
    "min_dist": (
        "Minimum distance between nearby points in the UMAP layout. Lower values "
        "form tighter groups; higher values spread points out."
    ),
    "Maximum objects": (
        "Upper limit for objects drawn in the UMAP view, used to keep interaction "
        "responsive on large clusters."
    ),
}

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("euclid_umap_explorer")


def log_app_event(event_type: str, **fields: object) -> None:
    payload = {
        "app": APP_TITLE,
        "version": APP_VERSION,
        "event_type": event_type,
        **fields,
    }
    LOGGER.info(json.dumps(payload, default=str, sort_keys=True))


class AlgorithmTimeoutError(TimeoutError):
    pass


def _run_in_process_worker(
    result_queue: mp.Queue,
    result_path: str,
    function,
    args: tuple,
    kwargs: dict,
) -> None:
    try:
        result = function(*args, **kwargs)
        with open(result_path, "wb") as file:
            pickle.dump(result, file, protocol=pickle.HIGHEST_PROTOCOL)
        result_queue.put(("ok", None))
    except BaseException as exc:
        result_queue.put(("error", exc))


def run_with_timeout(function, *args, timeout_seconds: int, **kwargs):
    ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn")
    result_queue = ctx.Queue(maxsize=1)
    result_file = tempfile.NamedTemporaryFile(
        prefix="euclid_algorithm_result_",
        suffix=".pkl",
        delete=False,
    )
    result_path = result_file.name
    result_file.close()
    process = ctx.Process(
        target=_run_in_process_worker,
        args=(result_queue, result_path, function, args, kwargs),
    )
    try:
        process.start()
        process.join(timeout_seconds)

        if process.is_alive():
            process.terminate()
            process.join(5)
            if process.is_alive():
                process.kill()
                process.join()
            raise AlgorithmTimeoutError(
                f"{function.__name__} exceeded {format_duration(timeout_seconds)}."
            )

        try:
            status, payload = result_queue.get_nowait()
        except queue.Empty as exc:
            raise RuntimeError(
                f"{function.__name__} finished without returning a result."
            ) from exc

        if status == "error":
            raise payload

        with open(result_path, "rb") as file:
            return pickle.load(file)
    finally:
        try:
            os.unlink(result_path)
        except FileNotFoundError:
            pass


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
        .back-to-top {
            align-items: center;
            background: #ff5a52;
            border: 1px solid rgba(255, 255, 255, 0.35);
            border-radius: 999px;
            bottom: 1.4rem;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.28);
            color: white !important;
            display: flex;
            font-size: 1.25rem;
            font-weight: 800;
            height: 2.65rem;
            justify-content: center;
            position: fixed;
            right: 1.4rem;
            text-decoration: none !important;
            width: 2.65rem;
            z-index: 1000;
        }
        .back-to-top:hover {
            background: #e84840;
            color: white !important;
        }
        .concept-help {
            border-bottom: 1px dotted rgba(250, 250, 250, 0.72);
            cursor: help !important;
            display: inline-block;
            position: relative;
        }
        .concept-popover {
            background: #171a22;
            border: 1px solid rgba(255, 255, 255, 0.18);
            border-radius: 8px;
            box-shadow: 0 12px 32px rgba(0, 0, 0, 0.34);
            color: #f8fafc;
            display: none;
            font-size: 0.82rem;
            font-weight: 400;
            left: 0;
            line-height: 1.35;
            min-width: 240px;
            padding: 0.65rem 0.75rem;
            position: absolute;
            top: 1.45rem;
            z-index: 1000;
        }
        .concept-help:hover .concept-popover,
        .concept-help:focus .concept-popover {
            display: block;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_back_to_top_control() -> None:
    st.markdown(
        """
        <div id="euclid-page-top"></div>
        <a class="back-to-top" href="#euclid-page-top" title="Back to top">↑</a>
        """,
        unsafe_allow_html=True,
    )


def is_gcs_path(path: str) -> bool:
    return str(path).startswith("gs://")


@st.cache_resource(show_spinner=False)
def gcs_filesystem():
    import gcsfs

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


def _run_birch_clustering_impl(
    parquet_path: str,
    lens_path: str,
    selected_grades: tuple[str, ...],
    threshold: float,
    branching_factor: int,
    batch_size: int,
) -> tuple[pd.DataFrame, list[str]]:
    from sklearn.cluster import Birch
    from sklearn.preprocessing import StandardScaler

    started_at = time.perf_counter()
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
    duration_seconds = time.perf_counter() - started_at
    clustered_df.attrs["n_subclusters"] = len(cluster_model.subcluster_centers_)
    clustered_df.attrs["processing_seconds"] = duration_seconds
    log_app_event(
        "birch_clustering_computed",
        duration_seconds=round(duration_seconds, 3),
        n_objects=int(len(clustered_df)),
        n_features=int(len(feature_cols)),
        n_clusters=int(clustered_df["cluster"].nunique()),
        n_lenses=int(clustered_df["is_lens"].sum()),
        selected_grades=list(selected_grades),
        threshold=float(threshold),
        branching_factor=int(branching_factor),
        batch_size=int(batch_size),
    )
    return clustered_df, feature_cols


@st.cache_data(show_spinner=True)
def run_birch_clustering(
    parquet_path: str,
    lens_path: str,
    selected_grades: tuple[str, ...],
    threshold: float,
    branching_factor: int,
    batch_size: int,
) -> tuple[pd.DataFrame, list[str]]:
    return run_with_timeout(
        _run_birch_clustering_impl,
        parquet_path,
        lens_path,
        selected_grades,
        threshold,
        branching_factor,
        batch_size,
        timeout_seconds=MAX_ALGORITHM_SECONDS,
    )


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


def format_duration(seconds: float | int | None) -> str:
    if seconds is None:
        return "-"

    seconds = float(seconds)
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.1f} s"

    minutes, remaining_seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)} min {remaining_seconds:.0f} s"

    hours, remaining_minutes = divmod(minutes, 60)
    return f"{int(hours)} h {int(remaining_minutes)} min"


def default_cluster_option_index(cluster_summary_df: pd.DataFrame) -> int:
    eligible = cluster_summary_df[cluster_summary_df["n_lenses"] > 1].copy()
    if eligible.empty:
        return 0

    selected_index = (
        eligible.sort_values(
            ["lens_rate", "n_lenses", "n_objects", "cluster"],
            ascending=[False, False, False, True],
        )
        .index[0]
    )
    return int(cluster_summary_df.index.get_loc(selected_index))


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


def _compute_umap_embedding_impl(
    data: pd.DataFrame,
    selected_features: list[str],
    n_neighbors: int,
    min_dist: float,
) -> pd.DataFrame:
    import umap
    from sklearn.preprocessing import StandardScaler

    started_at = time.perf_counter()
    clean = data.dropna(subset=selected_features).copy()
    if clean.empty:
        duration_seconds = time.perf_counter() - started_at
        clean.attrs["processing_seconds"] = duration_seconds
        log_app_event(
            "umap_computed",
            duration_seconds=round(duration_seconds, 3),
            n_objects=0,
            n_features=int(len(selected_features)),
            n_neighbors=int(n_neighbors),
            min_dist=float(min_dist),
        )
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
    duration_seconds = time.perf_counter() - started_at
    clean.attrs["processing_seconds"] = duration_seconds
    log_app_event(
        "umap_computed",
        duration_seconds=round(duration_seconds, 3),
        n_objects=int(len(clean)),
        n_features=int(len(selected_features)),
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
    )
    return clean


@st.cache_data(show_spinner=True)
def compute_umap_embedding(
    data: pd.DataFrame,
    selected_features: list[str],
    n_neighbors: int,
    min_dist: float,
) -> pd.DataFrame:
    return run_with_timeout(
        _compute_umap_embedding_impl,
        data,
        selected_features,
        n_neighbors,
        min_dist,
        timeout_seconds=MAX_ALGORITHM_SECONDS,
    )


def add_cluster_extreme_roles(
    data: pd.DataFrame,
    selected_features: list[str],
) -> pd.DataFrame:
    from sklearn.preprocessing import StandardScaler

    marked = data.copy()
    marked["is_canonical"] = False
    marked["is_anomaly"] = False
    marked["point_role"] = np.where(marked["is_lens"], "Lens candidate", "Unknown")

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


def normalize_pca_filters(raw_filters: list[dict], pca_columns: list[str]) -> tuple[dict, ...]:
    valid_columns = set(pca_columns)
    normalized = []
    for raw_filter in raw_filters:
        if not raw_filter.get("enabled", True):
            continue

        feature = raw_filter.get("feature")
        operator = raw_filter.get("operator")
        if feature not in valid_columns or operator not in PCA_FILTER_OPERATORS:
            continue

        if operator == "between":
            lower = float(raw_filter.get("lower", 0.0))
            upper = float(raw_filter.get("upper", 0.0))
            lower, upper = sorted((lower, upper))
            normalized.append(
                {
                    "feature": feature,
                    "operator": operator,
                    "lower": lower,
                    "upper": upper,
                    "enabled": True,
                }
            )
        else:
            normalized.append(
                {
                    "feature": feature,
                    "operator": operator,
                    "value": float(raw_filter.get("value", 0.0)),
                    "enabled": True,
                }
            )
    return tuple(normalized)


def pca_filter_signature(pca_filters: tuple[dict, ...]) -> tuple:
    signature = []
    for pca_filter in pca_filters:
        if pca_filter["operator"] == "between":
            signature.append(
                (
                    pca_filter["feature"],
                    pca_filter["operator"],
                    round(float(pca_filter["lower"]), 6),
                    round(float(pca_filter["upper"]), 6),
                )
            )
        else:
            signature.append(
                (
                    pca_filter["feature"],
                    pca_filter["operator"],
                    round(float(pca_filter["value"]), 6),
                )
            )
    return tuple(signature)


def format_pca_filter(pca_filter: dict) -> str:
    if pca_filter["operator"] == "between":
        return (
            f"{pca_filter['feature']} between "
            f"{pca_filter['lower']:.4g} and {pca_filter['upper']:.4g}"
        )
    return f"{pca_filter['feature']} {pca_filter['operator']} {pca_filter['value']:.4g}"


def apply_pca_filters(data: pd.DataFrame, pca_filters: tuple[dict, ...]) -> pd.DataFrame:
    if not pca_filters:
        return data.copy()

    mask = pd.Series(True, index=data.index)
    for pca_filter in pca_filters:
        values = data[pca_filter["feature"]]
        operator = pca_filter["operator"]
        if operator == ">":
            mask &= values > pca_filter["value"]
        elif operator == ">=":
            mask &= values >= pca_filter["value"]
        elif operator == "<":
            mask &= values < pca_filter["value"]
        elif operator == "<=":
            mask &= values <= pca_filter["value"]
        elif operator == "between":
            mask &= values.between(
                pca_filter["lower"],
                pca_filter["upper"],
                inclusive="both",
            )
    return data[mask].copy()


def add_pca_filter() -> None:
    st.session_state["pca_filter_count"] = (
        st.session_state.get("pca_filter_count", 0) + 1
    )


def render_pca_filter_controls(pca_columns: list[str]) -> list[dict]:
    with st.expander("PCA value filters", expanded=False):
        st.caption("Filters are combined with AND before UMAP is computed.")
        if "pca_filter_count" not in st.session_state:
            st.session_state["pca_filter_count"] = 0

        button_col, clear_col = st.columns(2)
        with button_col:
            st.button("Add filter", on_click=add_pca_filter)
        with clear_col:
            if st.session_state.get("pca_filter_count", 0):
                if st.button("Clear filters", key="clear_pca_filters"):
                    st.session_state["pca_filter_count"] = 0
                    st.rerun()

        filter_count = st.number_input(
            "Number of filters",
            min_value=0,
            max_value=12,
            step=1,
            key="pca_filter_count",
        )

        raw_filters = []
        for index in range(int(filter_count)):
            st.markdown(f"**Filter {index + 1}**")
            enabled = st.checkbox(
                "Enabled",
                value=True,
                key=f"pca_filter_{index}_enabled",
            )
            feature = st.selectbox(
                "Component",
                pca_columns,
                index=min(index, len(pca_columns) - 1),
                key=f"pca_filter_{index}_feature",
            )
            operator = st.selectbox(
                "Operator",
                PCA_FILTER_OPERATORS,
                key=f"pca_filter_{index}_operator",
            )

            if operator == "between":
                lower_col, upper_col = st.columns(2)
                with lower_col:
                    lower = st.number_input(
                        "Lower",
                        value=0.0,
                        step=0.1,
                        format="%.6f",
                        key=f"pca_filter_{index}_lower",
                    )
                with upper_col:
                    upper = st.number_input(
                        "Upper",
                        value=1.0,
                        step=0.1,
                        format="%.6f",
                        key=f"pca_filter_{index}_upper",
                    )
                raw_filters.append(
                    {
                        "feature": feature,
                        "operator": operator,
                        "lower": lower,
                        "upper": upper,
                        "enabled": enabled,
                    }
                )
            else:
                value = st.number_input(
                    "Value",
                    value=0.0,
                    step=0.1,
                    format="%.6f",
                    key=f"pca_filter_{index}_value",
                )
                raw_filters.append(
                    {
                        "feature": feature,
                        "operator": operator,
                        "value": value,
                        "enabled": enabled,
                    }
                )

    return raw_filters


def build_umap_signature(
    selected_cluster: int,
    selected_features: list[str],
    pca_filters: tuple[dict, ...],
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
        pca_filter_signature(pca_filters),
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
    import pyarrow as pa
    import pyarrow.dataset as ds
    import pyarrow.fs as pafs

    if not object_id:
        return pd.DataFrame()

    if is_gcs_path(morph_path):
        filesystem, path = pafs.FileSystem.from_uri(morph_path)
        dataset = ds.dataset(path, format="parquet", filesystem=filesystem)
    else:
        path = cached_input_path(morph_path)
        if not Path(path).exists():
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


def normalize_search_object_id(object_id: str) -> str:
    value = str(object_id).strip()
    if not value:
        raise ValueError("Enter an object_id.")
    try:
        return str(int(value))
    except ValueError as exc:
        raise ValueError("object_id must be an integer value.") from exc


def is_valid_search_object_id(object_id: str) -> bool:
    try:
        normalize_search_object_id(object_id)
    except ValueError:
        return False
    return True


def serializable_table_value(value: object) -> object:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def table_row_to_dict(row: object) -> dict[str, object]:
    columns = getattr(row, "colnames", None)
    if columns is None and hasattr(row, "columns"):
        columns = row.columns
    return {column: serializable_table_value(row[column]) for column in columns}


@st.cache_data(show_spinner=False, ttl=24 * 60 * 60)
def fetch_euclid_object_summary(
    object_id: str,
    search_radius_arcmin: float = 0.10,
    instrument: str = "VIS",
) -> dict[str, object]:
    from astroquery.esa.euclid import Euclid

    normalized_object_id = normalize_search_object_id(object_id)

    object_query = f"""
SELECT
    object_id,
    right_ascension,
    declination,
    segmentation_area,
    ellipticity,
    kron_radius,
    flux_detection_total,
    flux_vis_sersic,
    vis_det,
    det_quality_flag
FROM catalogue.mer_catalogue
WHERE object_id = {normalized_object_id}
"""

    object_job = Euclid.launch_job_async(object_query, verbose=False)
    object_result = object_job.get_results()
    if len(object_result) == 0:
        raise ValueError(f"object_id {normalized_object_id} was not found in catalogue.mer_catalogue.")

    object_row = object_result[0]
    object_summary = table_row_to_dict(object_row)
    ra = float(object_row["right_ascension"])
    dec = float(object_row["declination"])

    search_radius_deg = max(float(search_radius_arcmin) / 60.0, 0.5 / 60.0)
    mosaic_query = f"""
SELECT
    file_name,
    file_path,
    instrument_name,
    filter_name,
    product_type,
    tile_index
FROM q1.mosaic_product
WHERE instrument_name = '{instrument}'
  AND INTERSECTS(CIRCLE({ra}, {dec}, {search_radius_deg}), fov) = 1
ORDER BY file_name
"""

    mosaic_job = Euclid.launch_job_async(mosaic_query, verbose=False)
    mosaic_result = mosaic_job.get_results()
    if len(mosaic_result) == 0:
        raise ValueError(f"No {instrument} mosaic product was found for object_id {normalized_object_id}.")

    mosaic_row = mosaic_result[0]
    mosaic_summary = table_row_to_dict(mosaic_row)
    file_path = f"{mosaic_row['file_path']}/{mosaic_row['file_name']}"
    local_cutout_path = morphology_cutout_path(
        f"{mosaic_summary.get('tile_index')}_{normalized_object_id}",
        normalized_object_id,
    )

    return {
        "object_id": normalized_object_id,
        "object_summary": object_summary,
        "mosaic_summary": mosaic_summary,
        "file_path": file_path,
        "cutout_path": local_cutout_path,
    }


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
    started_at = time.perf_counter()
    if is_gcs_path(path):
        with gcs_filesystem().open(path, "rb") as image_file:
            image_bytes = image_file.read()
        source = "gcs"
    else:
        image_bytes = Path(path).read_bytes()
        source = "local"

    log_app_event(
        "image_loaded",
        duration_seconds=round(time.perf_counter() - started_at, 3),
        source=source,
        bytes=int(len(image_bytes)),
        suffix=Path(str(path)).suffix.lower(),
    )
    return image_bytes


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
        image_bytes = load_image_bytes(path)
        image = Image.open(BytesIO(image_bytes))
    except Exception:
        st.caption(caption)
        st.caption("No image")
        return

    object_id = row.get("object_id", "")
    id_str = row.get("id_str", "")
    image_type = "png" if str(path).lower().endswith(".png") else "jpeg"
    image_src = f"data:image/{image_type};base64,{base64.b64encode(image_bytes).decode()}"
    modal_key = hashlib.sha1(f"{caption}|{object_id}|{id_str}".encode()).hexdigest()[:12]
    modal_id = f"thumb-{modal_key}"
    escaped_caption = html.escape(str(caption))
    escaped_object_id = html.escape(str(object_id))
    escaped_id_str = html.escape(str(id_str))
    id_str_html = (
        f"<div class='thumb-modal-meta'>id_str: {escaped_id_str}</div>"
        if not pd.isna(id_str) and str(id_str).strip()
        else ""
    )

    st.markdown(
        f"""
        <style>
        #{modal_id} {{
            display: none;
        }}
        #{modal_id}:target {{
            align-items: center;
            background: rgba(0, 0, 0, 0.82);
            display: flex;
            inset: 0;
            justify-content: center;
            padding: 2rem;
            position: fixed;
            z-index: 10000;
        }}
        #{modal_id} .thumb-modal-panel {{
            background: #0f1117;
            border: 1px solid rgba(255, 255, 255, 0.18);
            border-radius: 8px;
            padding: 1rem;
        }}
        #{modal_id} .thumb-modal-meta {{
            color: #f8fafc;
            font-size: 0.9rem;
            margin-bottom: 0.45rem;
        }}
        #{modal_id} img {{
            display: block;
            height: 400px;
            object-fit: contain;
            width: 400px;
        }}
        .thumb-link img {{
            cursor: pointer;
            display: block;
            height: {SUMMARY_THUMBNAIL_WIDTH}px;
            object-fit: cover;
            width: {SUMMARY_THUMBNAIL_WIDTH}px;
        }}
        .thumb-caption {{
            color: rgba(250, 250, 250, 0.72);
            font-size: 0.82rem;
            margin-top: 0.2rem;
            text-align: center;
            width: {SUMMARY_THUMBNAIL_WIDTH}px;
        }}
        </style>
        <a class="thumb-link" href="#{modal_id}" title="Open enlarged image">
            <img src="{image_src}" width="{SUMMARY_THUMBNAIL_WIDTH}" height="{SUMMARY_THUMBNAIL_WIDTH}" />
        </a>
        <div class="thumb-caption">{escaped_caption}</div>
        <div id="{modal_id}">
            <a href="#" style="position: fixed; inset: 0;" aria-label="Close"></a>
            <div class="thumb-modal-panel">
                <div class="thumb-modal-meta"><strong>{escaped_caption}</strong></div>
                <div class="thumb-modal-meta">object_id: {escaped_object_id}</div>
                {id_str_html}
                <img src="{image_src}" alt="{escaped_caption}" />
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_thumbnail_group(
    title: str,
    rows: list[pd.Series],
    captions: list[str],
    prefer_lens_image: bool = False,
) -> None:
    render_thumbnail_group_title(title)
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


def sample_distplot_values(values: pd.Series, random_state: int) -> list[float]:
    clean_values = values.dropna()
    if len(clean_values) > SUMMARY_DISTPLOT_MAX_POINTS_PER_GROUP:
        clean_values = clean_values.sample(
            n=SUMMARY_DISTPLOT_MAX_POINTS_PER_GROUP,
            random_state=random_state,
        )
    return clean_values.astype(float).tolist()


def feature_bin_size(values: list[float]) -> float:
    if len(values) < 2:
        return 1.0
    value_range = max(values) - min(values)
    if value_range <= 0:
        return 1.0
    return value_range / SUMMARY_HISTOGRAM_BINS


def can_show_kde(values: list[float]) -> bool:
    return len(values) > 1 and len(set(values)) > 1


def build_cluster_distplot_figure(
    cluster_df: pd.DataFrame,
    feature: str,
    feature_index: int,
) -> object | None:
    import plotly.figure_factory as ff

    lens_df = cluster_df[cluster_df["is_lens"]]
    non_lens_df = cluster_df[~cluster_df["is_lens"]]
    non_lens_values = sample_distplot_values(
        non_lens_df[feature],
        random_state=feature_index + 101,
    )
    lens_values = sample_distplot_values(
        lens_df[feature],
        random_state=feature_index + 701,
    )

    hist_data = []
    group_labels = []
    colors = []
    if non_lens_values:
        hist_data.append(non_lens_values)
        group_labels.append("Unknown")
        colors.append("#4c78a8")
    if lens_values:
        hist_data.append(lens_values)
        group_labels.append("Lens candidate")
        colors.append("#d62728")
    if not hist_data:
        return None

    all_values = [value for values in hist_data for value in values]
    show_curve = all(can_show_kde(values) for values in hist_data)
    fig = ff.create_distplot(
        hist_data,
        group_labels,
        bin_size=feature_bin_size(all_values),
        colors=colors,
        curve_type="kde",
        show_curve=show_curve,
        show_hist=True,
        show_rug=True,
        histnorm="probability density",
    )
    fig.update_layout(
        title={"text": feature, "x": 0.5, "xanchor": "center"},
        height=300,
        margin={"l": 28, "r": 12, "t": 46, "b": 58},
        legend={"orientation": "h", "yanchor": "top", "y": -0.18, "x": 0},
        barmode="overlay",
        yaxis={"title": "density", "showgrid": True, "zeroline": False},
        yaxis2={"showgrid": True, "showticklabels": False, "zeroline": False},
    )
    fig.update_traces(opacity=0.72, selector={"type": "histogram"})
    fig.update_traces(line={"width": 2.0}, selector={"mode": "lines"})
    fig.update_xaxes(showgrid=False, zeroline=False)
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

    chart_columns = st.columns(2)
    for index, feature in enumerate(summary_features):
        fig = build_cluster_distplot_figure(cluster_df, feature, index)
        if fig is None:
            continue
        with chart_columns[index % 2]:
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displaylogo": False, "responsive": True},
                key=f"cluster_distplot_chart_{cluster_id}_{feature}",
            )


def render_thumbnail_group_title(title: str) -> None:
    if title != "Canonical / anomalous":
        st.caption(title)
        return

    st.markdown(
        """
        <div style="color: rgba(250, 250, 250, 0.72); font-size: 0.82rem; margin-bottom: 0.35rem;">
            <span class="concept-help" tabindex="0">Canonical
                <span class="concept-popover">
                    Object closest to the cluster centroid in the selected PCA feature space.
                </span>
            </span>
            <span style="margin: 0 0.2rem;">/</span>
            <span class="concept-help" tabindex="0">Anomalous
                <span class="concept-popover">
                    Object farthest from the cluster centroid in the selected PCA feature space.
                </span>
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_help_label(label: str, help_text: str) -> None:
    escaped_label = html.escape(label)
    escaped_help = html.escape(help_text).replace("\n", "<br>")
    st.markdown(
        f"""
        <div style="font-size: 0.88rem; font-weight: 600; margin-bottom: 0.2rem;">
            <span class="concept-help" tabindex="0">
                {escaped_label}
                <span class="concept-popover">{escaped_help}</span>
            </span>
        </div>
        """,
        unsafe_allow_html=True,
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
                    "Labelled lens candidates in the cluster",
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
        label = "LENS CANDIDATE"
        css_class = "lens-status--yes"
        meta = lens_grade_text or "Object present in the strong-lensing catalogue."
    else:
        label = "UNKNOWN"
        css_class = "lens-status--no"
        meta = "Object not marked as a lens candidate in the joined catalogue."

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

    cutout_path = morphology_cutout_path(row.get("id_str"), row.get("object_id"))
    lens_path = lens_image_path(row.get("lens_id_str"))
    if lens_path is None and bool(row.get("is_lens", False)):
        lens_path = lens_image_path(row.get("id_str"))

    if cutout_path is None and lens_path is None:
        st.info("No associated image was found in the configured paths.")
    else:
        if cutout_path is not None:
            show_image(cutout_path, "Morphology cutout")
        if lens_path is not None:
            show_image(lens_path, "Strong-lens image")


def show_morphology_catalogue_row(row: pd.Series) -> None:
    morphology_df = load_morphology_object(MORPH_PATH, str(row.get("object_id", "")))
    if not morphology_df.empty:
        st.markdown("**Morphology catalogue row**")
        morph_display = morphology_df.iloc[0].dropna().astype(str).reset_index()
        morph_display = (
            morph_display.rename(columns={"index": "field", morph_display.columns[-1]: "value"})
        )
        st.dataframe(morph_display, use_container_width=True, hide_index=True)


def show_selected_pca_components(row: pd.Series, selected_features: list[str]) -> None:
    st.markdown("**Selected PCA components**")
    st.dataframe(
        pd.DataFrame(
            [{"feature": feature, "value": row.get(feature)} for feature in selected_features]
        ),
        use_container_width=True,
        hide_index=True,
    )


def render_euclid_object_search(object_id: str) -> None:
    started_at = time.perf_counter()
    with st.spinner(f"Searching Euclid object {object_id}..."):
        result = fetch_euclid_object_summary(object_id)

    object_summary = result["object_summary"]
    mosaic_summary = result["mosaic_summary"]
    morphology_df = load_morphology_object(MORPH_PATH, str(result["object_id"]))
    log_app_event(
        "object_search_completed",
        duration_seconds=round(time.perf_counter() - started_at, 3),
        has_precomputed_cutout=bool(result.get("cutout_path")),
        has_morphology_row=not morphology_df.empty,
        instrument=str(mosaic_summary.get("instrument_name", "")),
        tile_index=str(mosaic_summary.get("tile_index", "")),
    )

    with st.container(border=True):
        st.subheader("Object search")
        metric_cols = st.columns([2.4, 1, 1, 1])
        metric_cols[0].metric("object_id", str(result["object_id"]))
        metric_cols[1].metric("RA", f"{float(object_summary['right_ascension']):.6f}")
        metric_cols[2].metric("Dec", f"{float(object_summary['declination']):.6f}")
        metric_cols[3].metric("Tile", str(mosaic_summary.get("tile_index", "")))

        image_col, summary_col = st.columns([1, 1])
        with image_col:
            cutout_path = result.get("cutout_path")
            if cutout_path:
                show_image(
                    str(cutout_path),
                    "Euclid VIS cutout",
                )
            else:
                st.info(
                    "No precomputed JPEG cutout was found for this object. "
                    "No FITS file was downloaded."
                )

            st.markdown("**Mosaic summary**")
            mosaic_display = pd.DataFrame(
                [{"field": field, "value": value} for field, value in mosaic_summary.items()]
            )
            st.dataframe(mosaic_display, use_container_width=True, hide_index=True)
        with summary_col:
            st.markdown("**Object summary**")
            object_display = pd.DataFrame(
                [{"field": field, "value": value} for field, value in object_summary.items()]
            )
            st.dataframe(object_display, use_container_width=True, hide_index=True)

            st.markdown("**Morphology catalogue features**")
            if morphology_df.empty:
                st.info("No morphology catalogue row was found for this object_id.")
            else:
                morphology_display = morphology_df.iloc[0].dropna().astype(str).reset_index()
                morphology_display = morphology_display.rename(
                    columns={
                        "index": "field",
                        morphology_display.columns[-1]: "value",
                    }
                )
                st.dataframe(morphology_display, use_container_width=True, hide_index=True)


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


def request_clustering() -> None:
    st.session_state["cluster_requested"] = True
    st.session_state["cluster_summary_expanded"] = True


def collapse_cluster_summary() -> None:
    st.session_state["cluster_summary_expanded"] = False


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon=str(EUCLID_FAVICON_PATH), layout="wide")
    inject_plot_cursor_css()
    render_back_to_top_control()
    st.title(APP_TITLE)
    loading_placeholder = st.empty()
    loading_placeholder.info("Loading application...")

    required = [PARQUET_PATH, LENS_PATH]
    with st.spinner("Loading application..."):
        missing = [path for path in required if not path_exists(path)]
    if missing:
        loading_placeholder.empty()
        st.error(
            "Required files were not found. Check the "
            "MORPH_PATH, PARQUET_PATH, LENS_PATH, "
            "CUTOUT_BASE and LENS_IMG_BASE environment variables."
        )
        st.code("\n".join(missing), language="text")
        st.stop()
    loading_placeholder.empty()

    with st.sidebar:
        logo_left, logo_center, logo_right = st.columns([1, 2, 1])
        with logo_center:
            st.image(str(EUCLID_LOGO_PATH), use_container_width=True)
        st.caption(f"Version {APP_VERSION}")

        st.header("Data")
        st.markdown(
            """
This analysis uses Euclid Q1 catalogue products available at:

- [The Strong Lensing Discovery Engine](https://zenodo.org/records/15025832)
- [First visual morphology catalogue](https://zenodo.org/records/15106473)
            """
        )
        search_input_col, search_button_col = st.columns([3, 1])
        with search_input_col:
            object_id_search_value = st.text_input(
                "object_id",
                placeholder="object_id",
                label_visibility="collapsed",
                key="object_id_search_value",
            )
        search_object_id = object_id_search_value.strip()
        search_submitted = search_button_col.button(
            "Search",
            disabled=not is_valid_search_object_id(search_object_id),
        )
        if search_submitted:
            st.session_state["euclid_search_object_id"] = search_object_id

        st.header("Lens candidates")
        render_help_label("Lens grades", LENS_GRADE_HELP)
        selected_lens_grades = st.multiselect(
            "Lens grades",
            LENS_GRADE_OPTIONS,
            default=DEFAULT_LENS_GRADES,
            label_visibility="collapsed",
        )
        selected_lens_grades = normalize_lens_grades(selected_lens_grades)

        st.header("BIRCH clustering")
        birch_expanded = not st.session_state.get("cluster_ready") and not st.session_state.get(
            "cluster_requested"
        )
        with st.expander("BIRCH parameters", expanded=birch_expanded):
            st.caption("All available PCA components are used for the initial clustering")
            render_help_label("threshold", PARAMETER_HELP["threshold"])
            threshold = st.number_input(
                "threshold",
                min_value=0.1,
                value=8.0,
                step=0.1,
                label_visibility="collapsed",
            )
            render_help_label("branching_factor", PARAMETER_HELP["branching_factor"])
            branching_factor = st.number_input(
                "branching_factor",
                min_value=2,
                value=2,
                step=1,
                label_visibility="collapsed",
            )
            render_help_label("batch_size", PARAMETER_HELP["batch_size"])
            batch_size = st.number_input(
                "batch_size",
                min_value=1_000,
                max_value=250_000,
                value=25_000,
                step=1_000,
                label_visibility="collapsed",
            )
            run_clustering = st.button(
                "Run clustering",
                type="primary",
                on_click=request_clustering,
            )

    searched_object_id = st.session_state.get("euclid_search_object_id")
    if searched_object_id:
        try:
            render_euclid_object_search(searched_object_id)
        except ValueError as exc:
            log_app_event("object_search_failed", error_type=type(exc).__name__)
            st.warning(str(exc))
        except ImportError as exc:
            log_app_event("object_search_failed", error_type=type(exc).__name__)
            st.error("The Euclid object search dependencies are not installed.")
            st.exception(exc)
        except Exception as exc:
            log_app_event("object_search_failed", error_type=type(exc).__name__)
            st.error("Could not retrieve the Euclid cutout for this object_id.")
            st.exception(exc)

    clustering_requested = st.session_state.pop("cluster_requested", False)
    if run_clustering or clustering_requested:
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
        st.session_state["cluster_summary_expanded"] = True
        log_app_event(
            "birch_clustering_requested",
            selected_grades=list(selected_lens_grades),
            threshold=float(threshold),
            branching_factor=int(branching_factor),
            batch_size=int(batch_size),
        )

    if not st.session_state.get("cluster_ready"):
        st.info("Click **Run clustering** button to clusterize data.")
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
    except AlgorithmTimeoutError as exc:
        log_app_event("birch_clustering_timeout", timeout_seconds=MAX_ALGORITHM_SECONDS)
        st.error(
            "BIRCH clustering was cancelled because it exceeded the "
            f"{format_duration(MAX_ALGORITHM_SECONDS)} execution limit. "
            "Try a less expensive configuration before running it again."
        )
        st.exception(exc)
        st.stop()
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

    with st.sidebar:
        st.header("PCA components")
        default_features = [
            feature for feature in DEFAULT_CLUSTER_FEATURES if feature in pca_columns
        ] or pca_columns[: min(4, len(pca_columns))]
        selected_features = st.multiselect(
            "PCA components",
            pca_columns,
            default=default_features,
        )
        raw_pca_filters = render_pca_filter_controls(pca_columns)

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

        with st.expander("UMAP parameters", expanded=True):
            render_help_label("n_neighbors", PARAMETER_HELP["n_neighbors"])
            n_neighbors = st.slider(
                "n_neighbors",
                2,
                100,
                25,
                label_visibility="collapsed",
            )
            render_help_label("min_dist", PARAMETER_HELP["min_dist"])
            min_dist = st.slider(
                "min_dist",
                0.0,
                1.0,
                0.15,
                step=0.01,
                label_visibility="collapsed",
            )
            render_help_label("Maximum objects", PARAMETER_HELP["Maximum objects"])
            max_objects = st.slider(
                "Maximum objects",
                100,
                100_000,
                20_000,
                step=100,
                label_visibility="collapsed",
            )

    if not selected_features:
        st.warning("Select at least one PCA component to build UMAP.")
        st.stop()
    pca_filters = normalize_pca_filters(raw_pca_filters, pca_columns)

    with st.expander(
        "Clustering summary",
        expanded=st.session_state.get("cluster_summary_expanded", False),
    ):
        st.markdown(
            f"""
            <div style="display: flex; align-items: baseline; gap: 0.45rem; margin-bottom: 0.75rem;">
                <span style="color: rgba(250, 250, 250, 0.72); font-size: 0.95rem;">
                    Execution time:
                </span>
                <span style="font-size: 1.8rem; font-weight: 600; line-height: 1;">
                    {format_duration(clustered_df.attrs.get("processing_seconds"))}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
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
    filtered_cluster_df = apply_pca_filters(cluster_df, pca_filters)
    if pca_filters:
        st.caption(
            "Active PCA filters: "
            + "; ".join(format_pca_filter(pca_filter) for pca_filter in pca_filters)
        )

    umap_signature = build_umap_signature(
        selected_cluster=selected_cluster,
        selected_features=selected_features,
        pca_filters=pca_filters,
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
        disabled=(not selected_features)
        or len(filtered_cluster_df) < 3
        or (not needs_recalculation and "umap_embedding_df" in st.session_state),
        on_click=collapse_cluster_summary,
    )

    if len(filtered_cluster_df) < 3:
        st.warning("At least 3 objects must remain after PCA filters to compute UMAP.")
        st.stop()

    if recalculate_umap:
        filtered_cluster_df = add_cluster_extreme_roles(filtered_cluster_df, selected_features)
        display_df = sample_for_display(filtered_cluster_df, int(max_objects))
        log_app_event(
            "umap_requested",
            cluster=int(selected_cluster),
            cluster_objects=int(len(cluster_df)),
            filtered_objects=int(len(filtered_cluster_df)),
            display_objects=int(len(display_df)),
            n_features=int(len(selected_features)),
            n_pca_filters=int(len(pca_filters)),
            n_neighbors=int(n_neighbors),
            min_dist=float(min_dist),
            max_objects=int(max_objects),
        )

        if len(display_df) < 3:
            st.warning("At least 3 objects are required to compute UMAP.")
            st.stop()

        try:
            embedding_df = compute_umap_embedding(
                display_df,
                selected_features,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
            )
        except AlgorithmTimeoutError as exc:
            log_app_event(
                "umap_timeout",
                timeout_seconds=MAX_ALGORITHM_SECONDS,
                cluster=int(selected_cluster),
                display_objects=int(len(display_df)),
                n_features=int(len(selected_features)),
            )
            st.error(
                "UMAP was cancelled because it exceeded the "
                f"{format_duration(MAX_ALGORITHM_SECONDS)} execution limit. "
                "Try reducing the maximum number of objects, using fewer PCA components, "
                "or adjusting the UMAP parameters before running it again."
            )
            st.exception(exc)
            st.stop()

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

    (
        cluster_left,
        cluster_filtered,
        cluster_middle,
        cluster_right,
        cluster_fourth,
    ) = st.columns(5)
    cluster_left.metric("Cluster objects", f"{len(cluster_df):,}")
    cluster_filtered.metric("After filters", f"{len(filtered_cluster_df):,}")
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

    import plotly.express as px

    fig = px.scatter(
        embedding_df,
        x="umap_1",
        y="umap_2",
        color="point_role",
        symbol="point_role",
        custom_data=["point_index"],
        hover_data=hover_columns,
        color_discrete_map={
            "Unknown": "#4c78a8",
            "Lens candidate": "#d62728",
            "Canonical": "#2ca02c",
            "Anomaly": "#111111",
        },
        symbol_map={
            "Unknown": "circle",
            "Lens candidate": "circle",
            "Canonical": "diamond",
            "Anomaly": "x",
        },
        category_orders={
            "point_role": ["Unknown", "Lens candidate", "Canonical", "Anomaly"],
        },
        labels={"umap_1": "UMAP 1", "umap_2": "UMAP 2", "point_role": "Type"},
        height=680,
    )
    fig.update_traces(marker={"size": 7, "opacity": 0.72})
    fig.update_traces(
        marker={"size": 17, "opacity": 0.98, "line": {"width": 1.5, "color": "white"}},
        selector={"name": "Lens candidate"},
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

    with st.expander("UMAP summary", expanded=True):
        st.markdown(
            f"""
            <div style="display: flex; align-items: baseline; gap: 0.45rem; margin-bottom: 0.75rem;">
                <span style="color: rgba(250, 250, 250, 0.72); font-size: 0.95rem;">
                    Execution time:
                </span>
                <span style="font-size: 1.8rem; font-weight: 600; line-height: 1;">
                    {format_duration(embedding_df.attrs.get("processing_seconds"))}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
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
        selected_row = (
            embedding_df.loc[selected_index]
            if selected_index is not None
            else None
        )
        with detail_col:
            if selected_row is None:
                st.info("Select a point on the map to view its details and image.")
            else:
                show_object_details(selected_row, selected_features)

        if selected_row is not None:
            morphology_col, pca_col = st.columns([1, 1])
            with morphology_col:
                show_morphology_catalogue_row(selected_row)
            with pca_col:
                show_selected_pca_components(selected_row, selected_features)


if __name__ == "__main__":
    main()
