from __future__ import annotations

import time

import numpy as np
import pandas as pd
import streamlit as st

from .catalogs import (
    load_lens_catalog,
    load_pca_catalog,
    merge_lens_flags,
    normalize_lens_grades,
)
from .config import (
    LENS_GRADE_OPTIONS,
    LENS_PATH,
    MAX_ALGORITHM_SECONDS,
    PARQUET_PATH,
    PCA_FILTER_OPERATORS,
)
from .runtime import log_app_event, run_with_timeout

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
