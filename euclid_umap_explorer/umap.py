from __future__ import annotations

import time

import pandas as pd

from .analysis import cluster_lens_grades, pca_filter_signature
from .config import LENS_PATH, MAX_ALGORITHM_SECONDS, PARQUET_PATH
from .runtime import log_app_event, run_with_timeout


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
) -> pd.DataFrame:
    import umap
    from sklearn.preprocessing import StandardScaler

    started_at = time.perf_counter()
    clean = data.dropna(subset=selected_features).copy()
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
        )
        return clean

    n_neighbors = min(n_neighbors, max(2, len(clean) - 1))
    scaled = StandardScaler().fit_transform(clean[selected_features])
    targets = _lens_grade_targets(clean)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        target_metric="categorical",
        random_state=42,
    )
    embedding = reducer.fit_transform(scaled, y=targets.to_numpy())
    clean["semi_umap_1"] = embedding[:, 0]
    clean["semi_umap_2"] = embedding[:, 1]
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
        duration_seconds=round(duration_seconds, 3),
        n_objects=int(len(clean)),
        n_features=int(len(selected_features)),
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        labelled_objects=int((targets >= 0).sum()),
    )
    return clean


def compute_semisupervised_umap_embedding(
    data: pd.DataFrame,
    selected_features: list[str],
    n_neighbors: int,
    min_dist: float,
) -> pd.DataFrame:
    return run_with_timeout(
        _compute_semisupervised_umap_embedding_impl,
        data,
        selected_features,
        n_neighbors,
        min_dist,
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
