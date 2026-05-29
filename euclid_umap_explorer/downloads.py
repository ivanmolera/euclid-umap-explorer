from __future__ import annotations

from collections.abc import Iterable
from io import StringIO

import pandas as pd

from .analysis import add_cluster_extreme_roles
from .catalogs import load_morphology_object, load_morphology_objects, normalize_object_ids
from .config import DOWNLOAD_MAX_UMAP_ROWS, MORPH_PATH


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buffer = StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def cluster_summary_download_df(
    clustered_df: pd.DataFrame,
    cluster_summary_df: pd.DataFrame,
    summary_features: list[str],
) -> pd.DataFrame:
    rows = []
    for _, summary_row in cluster_summary_df.iterrows():
        cluster_id = int(summary_row["cluster"])
        cluster_df = clustered_df[clustered_df["cluster"] == cluster_id].copy()
        marked_df = add_cluster_extreme_roles(cluster_df, summary_features)

        canonical = ""
        anomalous = ""
        canonical_rows = marked_df[marked_df["is_canonical"]]
        anomaly_rows = marked_df[marked_df["is_anomaly"]]
        if not canonical_rows.empty:
            canonical = str(canonical_rows.iloc[0].get("object_id", ""))
        if not anomaly_rows.empty:
            anomalous = str(anomaly_rows.iloc[0].get("object_id", ""))

        rows.append(
            {
                "cluster": cluster_id,
                "n_objects": int(summary_row["n_objects"]),
                "n_lenses": int(summary_row["n_lenses"]),
                "lens_rate": float(summary_row["lens_rate"]),
                "canonical": canonical,
                "anomalous": anomalous,
            }
        )

    return pd.DataFrame(rows)


def selected_point_indices(event: object) -> list[int]:
    if not event:
        return []

    try:
        points = event["selection"]["points"]
    except (KeyError, TypeError):
        return []

    indices = []
    for point in points or []:
        customdata = point.get("customdata")
        if isinstance(customdata, Iterable) and not isinstance(customdata, str):
            customdata = list(customdata)[0] if customdata else None
        try:
            indices.append(int(customdata))
        except (TypeError, ValueError):
            continue

    return indices


def morphology_row_for_object(object_id: object) -> dict[str, object]:
    morphology_df = load_morphology_object(MORPH_PATH, str(object_id))
    if morphology_df.empty:
        return {}
    return morphology_df.iloc[0].dropna().to_dict()


def format_ra_hms(ra_degrees: float) -> str:
    total_seconds = (float(ra_degrees) % 360.0) / 15.0 * 3600.0
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    return f"{hours:02d}h {minutes:02d}m {seconds:06.3f}s"


def format_dec_hms(dec_degrees: float) -> str:
    return format_dec_dms(dec_degrees)


def format_dec_dms(dec_degrees: float) -> str:
    sign = "-" if float(dec_degrees) < 0 else "+"
    total_seconds = abs(float(dec_degrees)) * 3600.0
    degrees = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    return f"{sign}{degrees:02d}° {minutes:02d}′ {seconds:06.3f}″"


def table_rows_with_coordinate_formats(
    section: str,
    values: dict[str, object],
) -> list[dict[str, object]]:
    rows = []
    for field, value in values.items():
        rows.append({"section": section, "field": field, "value": value})
        if field == "right_ascension":
            try:
                rows.append(
                    {
                        "section": section,
                        "field": "right_ascension_hms",
                        "value": format_ra_hms(float(value)),
                    }
                )
            except (TypeError, ValueError):
                pass
        if field == "declination":
            try:
                rows.append(
                    {
                        "section": section,
                        "field": "declination_dms",
                        "value": format_dec_dms(float(value)),
                    }
                )
            except (TypeError, ValueError):
                pass
    return rows


def umap_download_df(
    embedding_df: pd.DataFrame,
    selected_features: list[str],
    selected_indices: list[int] | None = None,
    max_rows: int = DOWNLOAD_MAX_UMAP_ROWS,
) -> pd.DataFrame:
    if selected_indices:
        base_df = embedding_df.loc[
            embedding_df.index.intersection(selected_indices)
        ].copy()
    else:
        base_df = embedding_df.copy()

    base_df = base_df.head(max_rows)
    base_df = base_df.copy()
    base_df["object_id"] = normalize_object_ids(base_df["object_id"])
    morphology_df = load_morphology_objects(MORPH_PATH, base_df["object_id"])
    if not morphology_df.empty:
        export_df = base_df.merge(
            morphology_df,
            on="object_id",
            how="left",
            suffixes=("", "_morphology"),
        )
    else:
        export_df = base_df

    if "right_ascension" in export_df.columns:
        export_df["right_ascension_hms"] = export_df["right_ascension"].map(
            lambda value: format_ra_hms(float(value)) if pd.notna(value) else ""
        )
    if "declination" in export_df.columns:
        export_df["declination_dms"] = export_df["declination"].map(
            lambda value: format_dec_dms(float(value)) if pd.notna(value) else ""
        )

    preferred_columns = [
        column
        for column in (
            "object_id",
            "id_str",
            "right_ascension",
            "right_ascension_hms",
            "declination",
            "declination_dms",
            "umap_1",
            "umap_2",
            "semi_umap_1",
            "semi_umap_2",
            "cluster",
            "hierarchical_subcluster",
            "is_lens",
            "lens_grade",
            "point_role",
            "semi_supervised_label",
            *selected_features,
        )
        if column in export_df.columns
    ]
    remaining_columns = [
        column for column in export_df.columns if column not in preferred_columns
    ]
    return export_df[preferred_columns + remaining_columns]


def object_search_download_df(
    object_summary: dict[str, object],
    morphology_df: pd.DataFrame,
    mosaic_summary: dict[str, object],
) -> pd.DataFrame:
    rows = []
    rows.extend(table_rows_with_coordinate_formats("Object summary", object_summary))
    rows.extend(
        table_rows_with_coordinate_formats(
            "Morphology catalogue features",
            {} if morphology_df.empty else morphology_df.iloc[0].dropna().to_dict(),
        )
    )
    rows.extend(
        {"section": "Mosaic summary", "field": field, "value": value}
        for field, value in mosaic_summary.items()
    )

    return pd.DataFrame(rows)
