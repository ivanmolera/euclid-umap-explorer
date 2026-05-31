from __future__ import annotations

import time

import pandas as pd
import plotly.graph_objects as go

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


def build_dendrogram_figure(
    data: pd.DataFrame,
    selected_features: list[str],
    max_objects: int,
    truncate_clusters: int,
) -> go.Figure:
    from scipy.cluster.hierarchy import dendrogram, linkage
    from sklearn.preprocessing import StandardScaler

    clean = data.dropna(subset=selected_features).copy()
    clean = sample_for_display(clean, max_objects)
    if len(clean) < 3:
        raise ValueError("At least 3 objects are required to build a dendrogram.")

    scaled = StandardScaler().fit_transform(clean[selected_features])
    linkage_matrix = linkage(scaled, method="ward", metric="euclidean")
    dendrogram_data = dendrogram(
        linkage_matrix,
        no_plot=True,
        truncate_mode="lastp",
        p=min(int(truncate_clusters), len(clean)),
        show_leaf_counts=True,
    )

    fig = go.Figure()
    for x_values, y_values in zip(
        dendrogram_data["icoord"],
        dendrogram_data["dcoord"],
    ):
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines",
                line={"color": "#5b7fb0", "width": 1.4},
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.update_layout(
        title=(
            f"Hierarchical dendrogram preview "
            f"({len(clean):,} sampled objects, Ward linkage)"
        ),
        xaxis_title="Truncated leaves",
        yaxis_title="Distance",
        margin={"l": 10, "r": 10, "t": 45, "b": 35},
        height=320,
    )
    fig.update_xaxes(showticklabels=False)
    return fig
