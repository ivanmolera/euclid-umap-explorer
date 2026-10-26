from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .catalogs import load_lens_catalog, load_pca_catalog, merge_lens_flags
from .config import MAX_ALGORITHM_SECONDS
from .runtime import log_app_event, run_with_timeout


@dataclass
class BirchProjection:
    feature_cols: tuple[str, ...]
    centers: np.ndarray
    center_labels: np.ndarray
    scaler_mean: np.ndarray | None
    scaler_scale: np.ndarray | None
    distance_cutoffs: dict[int, float]
    global_distance_cutoff: float


def _prepare_birch_features(
    work_df, parquet_path, selected_features, scaling, batch_size, source=None
):
    from .birch_cache import (
        load_prepared_features,
        prepared_cache_path,
        save_prepared_features,
        source_signature,
    )

    path = None
    try:
        source = source if source is not None else source_signature(parquet_path)
        path = prepared_cache_path(source, batch_size, selected_features, scaling)
        cached = load_prepared_features(
            path, source, work_df["id_str"], batch_size, selected_features, scaling
        )
        if cached is not None:
            log_app_event("birch_prepared_features_loaded", cache_path=path)
            return cached
    except Exception as exc:
        log_app_event("birch_prepared_cache_read_failed", error_type=type(exc).__name__)

    values = work_df[list(selected_features)].to_numpy(dtype=np.float32, copy=True)
    mean = scale = None
    if scaling == "standard":
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        # Keep the original fitting order and batch boundaries for reproducibility.
        for start in range(0, len(values), batch_size):
            scaler.partial_fit(values[start : start + batch_size])
        for start in range(0, len(values), batch_size):
            batch = values[start : start + batch_size]
            batch[:] = scaler.transform(batch, copy=False)
        mean, scale = scaler.mean_.copy(), scaler.scale_.copy()

    if path is not None:
        try:
            if source_signature(parquet_path) == source:
                save_prepared_features(
                    path, source, work_df["id_str"], batch_size, selected_features,
                    scaling, values, mean, scale,
                )
        except Exception as exc:
            log_app_event("birch_prepared_cache_write_failed", error_type=type(exc).__name__)
    return values, mean, scale


def _run_birch_clustering_impl(
    parquet_path: str,
    lens_path: str,
    selected_grades: tuple[str, ...],
    threshold: float,
    branching_factor: int,
    batch_size: int,
    selected_features: tuple[str, ...],
    scaling: str,
    *,
    source: dict | None = None,
) -> tuple[pd.DataFrame, list[str], BirchProjection]:
    from sklearn.cluster import Birch

    started_at = time.perf_counter()
    work_df, feature_cols = load_pca_catalog(parquet_path)
    lens_df = load_lens_catalog(lens_path, selected_grades)
    load_seconds = time.perf_counter() - started_at

    missing_features = [
        feature for feature in selected_features if feature not in feature_cols
    ]
    if missing_features:
        raise ValueError(
            "BIRCH features are missing from the PCA catalogue: "
            + ", ".join(missing_features)
        )
    if not selected_features:
        raise ValueError("Select at least one PCA feature for BIRCH clustering.")
    if scaling not in {"none", "standard"}:
        raise ValueError(f"Unsupported BIRCH feature scaling: {scaling}")
    if batch_size < 1:
        raise ValueError("BIRCH batch size must be positive.")

    birch_feature_cols = list(selected_features)
    phase_started = time.perf_counter()
    values, scaler_mean, scaler_scale = _prepare_birch_features(
        work_df, parquet_path, selected_features, scaling, batch_size, source
    )
    preparation_seconds = time.perf_counter() - phase_started

    phase_started = time.perf_counter()
    cluster_model = Birch(
        threshold=threshold,
        branching_factor=branching_factor,
        n_clusters=None,
        compute_labels=False,
    )
    for start in range(0, len(work_df), batch_size):
        cluster_model.partial_fit(values[start : start + batch_size])

    cluster_model.partial_fit()
    fit_seconds = time.perf_counter() - phase_started
    centers = cluster_model.subcluster_centers_.astype(np.float32, copy=True)
    center_labels = cluster_model.subcluster_labels_.astype(np.int32, copy=True)
    if not np.array_equal(center_labels, np.arange(len(centers))):
        raise RuntimeError("BIRCH subcluster labels do not match their centers.")

    labels = np.empty(len(work_df), dtype=np.int32)
    training_distances = np.empty(len(work_df), dtype=np.float32)
    phase_started = time.perf_counter()
    for start in range(0, len(work_df), batch_size):
        end = min(start + batch_size, len(work_df))
        x_batch = values[start:end]
        labels[start:end] = cluster_model.predict(x_batch)
        training_distances[start:end] = np.linalg.norm(
            x_batch - centers[labels[start:end]], axis=1
        )
    predict_seconds = time.perf_counter() - phase_started

    phase_started = time.perf_counter()
    global_distance_cutoff = float(np.percentile(training_distances, 95))
    distance_stats = pd.DataFrame(
        {"cluster": labels, "distance": training_distances}
    ).groupby("cluster").agg(
        n_objects=("distance", "size"),
        p95_distance=("distance", lambda values: values.quantile(0.95)),
    )
    distance_cutoffs = {
        int(cluster): (
            float(row["p95_distance"])
            if int(row["n_objects"]) >= 20
            else global_distance_cutoff
        )
        for cluster, row in distance_stats.iterrows()
    }
    projection = BirchProjection(
        feature_cols=tuple(birch_feature_cols),
        centers=centers,
        center_labels=center_labels,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        distance_cutoffs=distance_cutoffs,
        global_distance_cutoff=global_distance_cutoff,
    )

    clustered_df = work_df.copy()
    clustered_df["cluster"] = labels
    clustered_df = merge_lens_flags(clustered_df, lens_df)
    duration_seconds = time.perf_counter() - started_at
    clustered_df.attrs["n_subclusters"] = len(cluster_model.subcluster_centers_)
    clustered_df.attrs["processing_seconds"] = duration_seconds
    log_app_event(
        "birch_clustering_computed",
        duration_seconds=round(duration_seconds, 3),
        load_seconds=round(load_seconds, 3),
        preparation_seconds=round(preparation_seconds, 3),
        fit_seconds=round(fit_seconds, 3),
        predict_seconds=round(predict_seconds, 3),
        summary_seconds=round(time.perf_counter() - phase_started, 3),
        n_objects=int(len(clustered_df)),
        n_features=int(len(birch_feature_cols)),
        features=birch_feature_cols,
        scaling=scaling,
        n_clusters=int(clustered_df["cluster"].nunique()),
        n_lenses=int(clustered_df["is_lens"].sum()),
        selected_grades=list(selected_grades),
        threshold=float(threshold),
        branching_factor=int(branching_factor),
        batch_size=int(batch_size),
    )
    return clustered_df, feature_cols, projection


def _assign_excluded_artifacts_impl(
    full_parquet_path: str,
    included_id_strs: set[str],
    projection: BirchProjection,
    batch_size: int,
) -> pd.DataFrame:
    from sklearn.metrics import pairwise_distances_argmin_min

    if batch_size < 1:
        raise ValueError("Artifact assignment batch size must be positive.")
    started_at = time.perf_counter()
    full_df, available_features = load_pca_catalog(full_parquet_path)
    if not included_id_strs.issubset(set(full_df["id_str"].dropna().astype(str))):
        raise ValueError("The clustered PCA catalogue is not a subset of the full catalogue.")
    missing_features = set(projection.feature_cols) - set(available_features)
    if missing_features:
        raise ValueError(
            "PCA features are missing from the unfiltered catalogue: "
            + ", ".join(sorted(missing_features))
        )

    excluded = full_df.loc[
        ~full_df["id_str"].astype("string").isin(included_id_strs),
        ["id_str", "object_id", *projection.feature_cols],
    ].copy()
    if excluded.empty:
        return pd.DataFrame(
            columns=["id_str", "object_id", "cluster", "distance", "is_far"]
        )

    assignments = np.empty(len(excluded), dtype=np.int32)
    distances = np.empty(len(excluded), dtype=np.float32)
    for start in range(0, len(excluded), batch_size):
        end = min(start + batch_size, len(excluded))
        x_batch = excluded.iloc[start:end][list(projection.feature_cols)].to_numpy(
            dtype=np.float32, copy=True
        )
        if projection.scaler_mean is not None and projection.scaler_scale is not None:
            x_batch -= projection.scaler_mean
            x_batch /= projection.scaler_scale
        nearest, distance = pairwise_distances_argmin_min(x_batch, projection.centers)
        assignments[start:end] = projection.center_labels[nearest]
        distances[start:end] = distance

    excluded = excluded[["id_str", "object_id"]].reset_index(drop=True)
    excluded["cluster"] = assignments
    excluded["distance"] = distances
    cutoffs = excluded["cluster"].map(projection.distance_cutoffs).fillna(
        projection.global_distance_cutoff
    )
    excluded["is_far"] = excluded["distance"] > cutoffs
    log_app_event(
        "birch_excluded_artifacts_assigned",
        duration_seconds=round(time.perf_counter() - started_at, 3),
        n_excluded=int(len(excluded)),
        n_far=int(excluded["is_far"].sum()),
    )
    return excluded


def assign_excluded_artifacts(
    full_parquet_path: str,
    included_id_strs: set[str],
    projection: BirchProjection,
    batch_size: int = 2048,
) -> pd.DataFrame:
    return run_with_timeout(
        _assign_excluded_artifacts_impl,
        full_parquet_path,
        included_id_strs,
        projection,
        batch_size,
        timeout_seconds=MAX_ALGORITHM_SECONDS,
    )


def run_birch_clustering(
    parquet_path: str,
    lens_path: str,
    selected_grades: tuple[str, ...],
    threshold: float,
    branching_factor: int,
    batch_size: int,
    selected_features: tuple[str, ...],
    scaling: str,
) -> tuple[pd.DataFrame, list[str], BirchProjection]:
    return run_with_timeout(
        _run_birch_clustering_with_cache_impl,
        parquet_path,
        lens_path,
        selected_grades,
        threshold,
        branching_factor,
        batch_size,
        selected_features,
        scaling,
        timeout_seconds=MAX_ALGORITHM_SECONDS,
    )


def _run_birch_clustering_with_cache_impl(
    parquet_path: str,
    lens_path: str,
    selected_grades: tuple[str, ...],
    threshold: float,
    branching_factor: int,
    batch_size: int,
    selected_features: tuple[str, ...],
    scaling: str,
) -> tuple[pd.DataFrame, list[str], BirchProjection]:
    from .birch_cache import (
        cache_path,
        legacy_cache_path,
        load_cached_projection,
        ordered_id_digest,
        save_cached_projection,
        source_signature,
    )

    started_at = time.perf_counter()
    signature = None
    parameters = (threshold, branching_factor, batch_size, selected_features, scaling)
    path = cache_path(parquet_path, *parameters)
    try:
        signature = source_signature(parquet_path)
        loaded_path = path
        cached = load_cached_projection(path, signature, *parameters)
        if cached is None:
            legacy_path = legacy_cache_path(parquet_path, *parameters)
            if legacy_path is not None:
                cached = load_cached_projection(legacy_path, signature, *parameters)
                loaded_path = legacy_path
        if cached is not None:
            labels, projection, row_digest = cached
            work_df, feature_cols = load_pca_catalog(parquet_path)
            if (
                len(labels) == len(work_df)
                and ordered_id_digest(work_df["id_str"]) == row_digest
                and set(selected_features).issubset(feature_cols)
                and source_signature(parquet_path) == signature
            ):
                work_df["cluster"] = labels
                lens_df = load_lens_catalog(lens_path, selected_grades)
                clustered_df = merge_lens_flags(work_df, lens_df)
                duration = time.perf_counter() - started_at
                clustered_df.attrs["processing_seconds"] = duration
                clustered_df.attrs["n_subclusters"] = len(projection.centers)
                log_app_event(
                    "birch_precomputed_loaded",
                    duration_seconds=round(duration, 3),
                    n_objects=len(clustered_df),
                    n_clusters=len(projection.centers),
                    cache_path=loaded_path,
                )
                return clustered_df, feature_cols, projection
            log_app_event("birch_cache_rejected", reason="catalogue rows differ")
    except Exception as exc:
        log_app_event("birch_cache_read_failed", error_type=type(exc).__name__)

    clustered_df, feature_cols, projection = _run_birch_clustering_impl(
        parquet_path,
        lens_path,
        selected_grades,
        threshold,
        branching_factor,
        batch_size,
        selected_features,
        scaling,
        source=signature,
    )
    if signature is not None:
        try:
            if source_signature(parquet_path) != signature:
                log_app_event("birch_cache_write_skipped", reason="catalogue changed")
                return clustered_df, feature_cols, projection
            save_cached_projection(
                path,
                signature,
                clustered_df["id_str"],
                clustered_df["cluster"],
                projection,
                *parameters,
            )
            log_app_event("birch_precomputed_saved", cache_path=path)
        except Exception as exc:
            log_app_event("birch_cache_write_failed", error_type=type(exc).__name__)
    return clustered_df, feature_cols, projection
