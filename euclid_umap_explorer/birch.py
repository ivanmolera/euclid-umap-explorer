from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .catalogs import load_lens_catalog, load_pca_catalog, merge_lens_flags
from .config import MAX_ALGORITHM_SECONDS
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
