from __future__ import annotations

import numpy as np
import pandas as pd

from .catalogs import normalize_lens_grades
from .config import LENS_GRADE_OPTIONS, PCA_FILTER_OPERATORS

DEFAULT_PCA_SELECTION_PRESET = "Lens-displaced PCA components"
PCA_SELECTION_PRESETS = [
    "Manual selection",
    "Top 10 PCA by explained variance",
    "Top 10 PCA by Random Forest importance",
    "Top 10 PCA by mutual information with lens candidate labels",
    "Lens-displaced PCA components",
    "All 40 PCA baseline",
]

PCA_TOP_10_BY_EXPLAINED_VARIANCE = [
    "feat_pca_0",
    "feat_pca_1",
    "feat_pca_2",
    "feat_pca_3",
    "feat_pca_4",
    "feat_pca_5",
    "feat_pca_6",
    "feat_pca_7",
    "feat_pca_8",
    "feat_pca_9",
]
PCA_TOP_10_BY_RANDOM_FOREST_IMPORTANCE = [
    "feat_pca_6",
    "feat_pca_0",
    "feat_pca_1",
    "feat_pca_27",
    "feat_pca_12",
    "feat_pca_29",
    "feat_pca_10",
    "feat_pca_22",
    "feat_pca_13",
    "feat_pca_32",
]
PCA_TOP_10_BY_MUTUAL_INFORMATION = [
    "feat_pca_6",
    "feat_pca_27",
    "feat_pca_12",
    "feat_pca_32",
    "feat_pca_0",
    "feat_pca_29",
    "feat_pca_10",
    "feat_pca_1",
    "feat_pca_13",
    "feat_pca_22",
]
PCA_LENS_DISPLACED_COMPONENTS = [
    "feat_pca_6",
    "feat_pca_0",
    "feat_pca_12",
    "feat_pca_1",
    "feat_pca_27",
    "feat_pca_10",
    "feat_pca_8",
    "feat_pca_13",
]
PCA_PRESET_FEATURES = {
    "Top 10 PCA by explained variance": PCA_TOP_10_BY_EXPLAINED_VARIANCE,
    "Top 10 PCA by Random Forest importance": PCA_TOP_10_BY_RANDOM_FOREST_IMPORTANCE,
    "Top 10 PCA by mutual information with lens candidate labels": (
        PCA_TOP_10_BY_MUTUAL_INFORMATION
    ),
    "Lens-displaced PCA components": PCA_LENS_DISPLACED_COMPONENTS,
}


def build_cluster_summary(clustered_df: pd.DataFrame) -> pd.DataFrame:
    total_objects = len(clustered_df)
    total_lenses = int(clustered_df["is_lens"].sum())
    global_lens_rate = total_lenses / total_objects if total_objects else 0.0

    summary_df = (
        clustered_df.groupby("cluster")
        .agg(
            n_objects=("object_id", "size"),
            n_lenses=("is_lens", "sum"),
        )
        .reset_index()
    )
    summary_df["lens_rate"] = summary_df["n_lenses"] / summary_df["n_objects"]
    summary_df["enrichment"] = (
        summary_df["lens_rate"] / global_lens_rate if global_lens_rate else 0.0
    )
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
        f"{row['lens_rate'] * 100:.3f}% | "
        f"{row.get('enrichment', 0.0):.2f}x enrichment"
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


def pca_features_for_preset(
    pca_columns: list[str],
    preset: str,
) -> list[str]:
    if preset == "All 40 PCA baseline":
        return list(pca_columns)

    available_columns = set(pca_columns)
    preset_features = PCA_PRESET_FEATURES.get(preset, [])
    return [feature for feature in preset_features if feature in available_columns]


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


def cluster_lens_grades(cluster_params: dict) -> tuple[str, ...]:
    if "lens_grades" in cluster_params:
        return normalize_lens_grades(cluster_params["lens_grades"])
    if cluster_params.get("only_grade_a", True):
        return ("A",)
    return tuple(LENS_GRADE_OPTIONS)
