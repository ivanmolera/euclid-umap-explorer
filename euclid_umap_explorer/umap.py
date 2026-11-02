from __future__ import annotations

import time

import pandas as pd

from .analysis import cluster_lens_grades, pca_filter_signature
from .config import (
    DEFAULT_BIRCH_FEATURES,
    DEFAULT_BIRCH_SCALING,
    DEFAULT_SEMISUPERVISED_UMAP_TARGET_WEIGHT,
    LENS_PATH,
    MAX_ALGORITHM_SECONDS,
    PARQUET_PATH,
)
from .runtime import log_app_event, run_with_timeout
from .physical import physical_filter_signature


def _compute_umap_embedding_impl(
    data: pd.DataFrame,
    selected_features: list[str],
    n_neighbors: int,
    min_dist: float,
    n_components: int = 2,
) -> pd.DataFrame:
    import umap
    from sklearn.preprocessing import StandardScaler

    if n_components not in (2, 3):
        raise ValueError("UMAP supports 2D or 3D projections.")
    started_at = time.perf_counter()
    clean = data.dropna(subset=selected_features).copy()
    clean = clean.drop(columns=["umap_1", "umap_2", "umap_3"], errors="ignore")
    clean.attrs["n_components"] = n_components
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
        n_components=n_components,
        init="random" if len(clean) <= n_components + 1 else "spectral",
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=42,
    )
    embedding = reducer.fit_transform(scaled)
    for axis in range(n_components):
        clean[f"umap_{axis + 1}"] = embedding[:, axis]
    duration_seconds = time.perf_counter() - started_at
    clean.attrs["processing_seconds"] = duration_seconds
    log_app_event(
        "umap_computed",
        n_components=n_components,
        duration_seconds=round(duration_seconds, 3),
        n_objects=int(len(clean)),
        n_features=int(len(selected_features)),
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
    )
    return clean


def compute_umap_embedding(
    data: pd.DataFrame,
    selected_features: list[str],
    n_neighbors: int,
    min_dist: float,
    n_components: int = 2,
) -> pd.DataFrame:
    return run_with_timeout(
        _compute_umap_embedding_impl,
        data,
        selected_features,
        n_neighbors,
        min_dist,
        n_components,
        timeout_seconds=MAX_ALGORITHM_SECONDS,
    )


def _lens_grade_targets(data: pd.DataFrame) -> pd.Series:
    if "lens_grade" not in data.columns:
        return pd.Series(-1, index=data.index, dtype=int)

    grade_targets = (
        data["lens_grade"]
        .astype("string")
        .str.strip()
        .str.upper()
        .map({"A": 2, "B": 1, "C": 0})
        .fillna(-1)
        .astype(int)
    )
    return grade_targets


def _compute_semisupervised_umap_embedding_impl(
    data: pd.DataFrame,
    selected_features: list[str],
    n_neighbors: int,
    min_dist: float,
    target_weight: float,
    n_components: int = 2,
) -> pd.DataFrame:
    import umap
    from sklearn.preprocessing import StandardScaler

    if n_components not in (2, 3):
        raise ValueError("UMAP supports 2D or 3D projections.")
    started_at = time.perf_counter()
    clean = data.dropna(subset=selected_features).copy()
    clean = clean.drop(columns=["semi_umap_1", "semi_umap_2", "semi_umap_3"], errors="ignore")
    clean.attrs["n_components"] = n_components
    if clean.empty:
        duration_seconds = time.perf_counter() - started_at
        clean.attrs["processing_seconds"] = duration_seconds
        log_app_event(
            "semisupervised_umap_computed",
            duration_seconds=round(duration_seconds, 3),
            n_objects=0,
            n_features=int(len(selected_features)),
            n_neighbors=int(n_neighbors),
            min_dist=float(min_dist),
            target_weight=float(target_weight),
        )
        return clean

    n_neighbors = min(n_neighbors, max(2, len(clean) - 1))
    scaled = StandardScaler().fit_transform(clean[selected_features])
    targets = _lens_grade_targets(clean)
    reducer = umap.UMAP(
        n_components=n_components,
        init="random" if len(clean) <= n_components + 1 else "spectral",
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        target_metric="categorical",
        target_weight=target_weight,
        random_state=42,
    )
    embedding = reducer.fit_transform(scaled, y=targets.to_numpy())
    for axis in range(n_components):
        clean[f"semi_umap_{axis + 1}"] = embedding[:, axis]
    clean["semi_supervised_target"] = targets
    clean["semi_supervised_label"] = targets.map(
        {
            2: "Grade A",
            1: "Grade B",
            0: "Grade C",
            -1: "Unknown",
        }
    )

    duration_seconds = time.perf_counter() - started_at
    clean.attrs["processing_seconds"] = duration_seconds
    log_app_event(
        "semisupervised_umap_computed",
        n_components=n_components,
        duration_seconds=round(duration_seconds, 3),
        n_objects=int(len(clean)),
        n_features=int(len(selected_features)),
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        target_weight=float(target_weight),
        labelled_objects=int((targets >= 0).sum()),
    )
    return clean


def compute_semisupervised_umap_embedding(
    data: pd.DataFrame,
    selected_features: list[str],
    n_neighbors: int,
    min_dist: float,
    target_weight: float = DEFAULT_SEMISUPERVISED_UMAP_TARGET_WEIGHT,
    n_components: int = 2,
) -> pd.DataFrame:
    return run_with_timeout(
        _compute_semisupervised_umap_embedding_impl,
        data,
        selected_features,
        n_neighbors,
        min_dist,
        target_weight,
        n_components,
        timeout_seconds=MAX_ALGORITHM_SECONDS,
    )


def build_umap_signature(
    selected_cluster: int,
    selected_features: list[str],
    pca_filters: tuple[dict, ...],
    n_neighbors: int,
    min_dist: float,
    max_objects: int,
    cluster_params: dict,
    physical_filters: tuple[dict, ...] = (),
    n_components: int = 2,
) -> tuple:
    return (
        cluster_params.get("parquet_path", PARQUET_PATH),
        bool(cluster_params.get("straight_line_artifacts_filter", False)),
        LENS_PATH,
        cluster_lens_grades(cluster_params),
        float(cluster_params["threshold"]),
        int(cluster_params["branching_factor"]),
        int(cluster_params["batch_size"]),
        tuple(cluster_params.get("birch_features", DEFAULT_BIRCH_FEATURES)),
        str(cluster_params.get("birch_scaling", DEFAULT_BIRCH_SCALING)),
        int(selected_cluster),
        tuple(selected_features),
        pca_filter_signature(pca_filters),
        physical_filter_signature(physical_filters),
        int(n_neighbors),
        round(float(min_dist), 4),
        int(max_objects),
        int(n_components),
    )
