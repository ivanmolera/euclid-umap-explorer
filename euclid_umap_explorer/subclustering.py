from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from .analysis import sample_for_display
from .config import MAX_ALGORITHM_SECONDS
from .runtime import log_app_event, run_with_timeout


def _compute_hierarchical_subclusters_impl(
    data: pd.DataFrame,
    selected_features: list[str],
    n_subclusters: int,
    max_objects: int,
) -> pd.DataFrame:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler

    started_at = time.perf_counter()
    clean = data.dropna(subset=selected_features).copy()
    if clean.empty:
        clean.attrs["processing_seconds"] = time.perf_counter() - started_at
        return clean

    clean = sample_for_display(clean, max_objects)
    if len(clean) < 2:
        clean["hierarchical_subcluster"] = 0
        clean.attrs["processing_seconds"] = time.perf_counter() - started_at
        return clean

    n_subclusters = min(max(2, int(n_subclusters)), len(clean))

    scaled = StandardScaler().fit_transform(clean[selected_features])
    model = AgglomerativeClustering(
        n_clusters=n_subclusters,
        metric="euclidean",
        linkage="ward",
    )
    clean["hierarchical_subcluster"] = model.fit_predict(scaled).astype(int)

    duration_seconds = time.perf_counter() - started_at
    clean.attrs["processing_seconds"] = duration_seconds
    log_app_event(
        "hierarchical_subclustering_computed",
        duration_seconds=round(duration_seconds, 3),
        n_objects=int(len(clean)),
        n_features=int(len(selected_features)),
        n_subclusters=int(n_subclusters),
        max_objects=int(max_objects),
    )
    return clean


@st.cache_data(show_spinner=True)
def compute_hierarchical_subclusters(
    data: pd.DataFrame,
    selected_features: list[str],
    n_subclusters: int,
    max_objects: int,
) -> pd.DataFrame:
    return run_with_timeout(
        _compute_hierarchical_subclusters_impl,
        data,
        selected_features,
        n_subclusters,
        max_objects,
        timeout_seconds=MAX_ALGORITHM_SECONDS,
    )


def build_subclustering_signature(
    umap_signature: tuple,
    selected_features: list[str],
    n_subclusters: int,
    max_objects: int,
) -> tuple:
    return (
        umap_signature,
        tuple(selected_features),
        int(n_subclusters),
        int(max_objects),
    )


def build_subcluster_summary(subclustered_df: pd.DataFrame) -> pd.DataFrame:
    if subclustered_df.empty or "hierarchical_subcluster" not in subclustered_df.columns:
        return pd.DataFrame()

    summary_df = (
        subclustered_df.groupby("hierarchical_subcluster")
        .agg(
            n_objects=("object_id", "size"),
            n_lenses=("is_lens", "sum"),
        )
        .reset_index()
        .sort_values(
            ["n_lenses", "n_objects", "hierarchical_subcluster"],
            ascending=[False, False, True],
        )
    )
    summary_df["lens_rate"] = summary_df["n_lenses"] / summary_df["n_objects"]
    return summary_df
