from __future__ import annotations

from collections.abc import Iterable
from math import isfinite

import numpy as np
import pandas as pd

from .catalogs import normalize_object_ids


PHYSICAL_MEASUREMENT_GROUPS = (
    (
        "Physical characterization",
        (
            "phz_median",
            "phz_pp_median_stellarmass",
            "concentration",
            "asymmetry",
            "smoothness",
            "gini",
            "moment_20",
            "sersic_sersic_vis_index",
        ),
    ),
)

PHYSICAL_ANALYSIS_FIELDS = {
    "phz_median": "Photometric redshift",
    "phz_pp_median_stellarmass": "Stellar mass (log10 M_sun)",
    "concentration": "Concentration (CAS)",
    "asymmetry": "Asymmetry (CAS)",
    "smoothness": "Smoothness (CAS)",
    "gini": "Gini",
    "moment_20": "M20",
    "sersic_sersic_vis_index": "Sersic index",
}

PHYSICAL_QUALITY_FLAG_FIELDS = (
    "phz_flags",
    "phys_param_flags",
    "sersic_visnir_flags",
)

PHYSICAL_QUERY_FIELDS = (
    *PHYSICAL_ANALYSIS_FIELDS,
    *PHYSICAL_QUALITY_FLAG_FIELDS,
)

PHYSICAL_FILTER_OPERATORS = ("between", ">=", "<=")


def clean_physical_measurements(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize identifiers and non-finite numerical values."""
    if data.empty:
        return data.copy()

    clean = data.copy()
    if "object_id" in clean.columns:
        clean["object_id"] = normalize_object_ids(clean["object_id"])
    numeric_columns = clean.select_dtypes(include=[np.number]).columns
    if len(numeric_columns):
        clean[numeric_columns] = clean[numeric_columns].replace(
            [np.inf, -np.inf], np.nan
        )
    return clean


def analysis_ready_physical_measurements(data: pd.DataFrame) -> pd.DataFrame:
    """Return finite measurements with documented quality flags applied."""
    clean = clean_physical_measurements(data)
    if clean.empty:
        return clean

    if "phz_flags" in clean.columns and "phz_median" in clean.columns:
        clean.loc[clean["phz_flags"] != 0, "phz_median"] = np.nan

    if "phys_param_flags" in clean.columns:
        physical_parameter_columns = (
            "phz_pp_median_redshift",
            "phz_pp_median_stellarmass",
            "phz_pp_median_luminosity",
            "phz_pp_median_sfr",
        )
        invalid = clean["phys_param_flags"] != 0
        for column in physical_parameter_columns:
            if column in clean.columns:
                clean.loc[invalid, column] = np.nan

    if "sersic_visnir_flags" in clean.columns and "sersic_sersic_vis_index" in clean.columns:
        clean.loc[
            clean["sersic_visnir_flags"] != 0,
            "sersic_sersic_vis_index",
        ] = np.nan

    return clean


def physical_measurement_display_rows(
    row: pd.Series | dict[str, object],
    *,
    excluded_fields: Iterable[str] = (),
) -> list[dict[str, object]]:
    values = row.to_dict() if isinstance(row, pd.Series) else dict(row)
    clean = analysis_ready_physical_measurements(pd.DataFrame([values]))
    if not clean.empty:
        values = clean.iloc[0].to_dict()
    excluded = set(excluded_fields)
    rows: list[dict[str, object]] = []

    def is_missing(value: object) -> bool:
        if value is None or pd.isna(value):
            return True
        return isinstance(value, (float, np.floating)) and not np.isfinite(value)

    for section, fields in PHYSICAL_MEASUREMENT_GROUPS:
        for field in fields:
            value = values.get(field)
            if field in excluded or is_missing(value):
                continue
            rows.append({"section": section, "field": field, "value": value})

    return rows


def available_physical_analysis_fields(data: pd.DataFrame) -> list[str]:
    clean = analysis_ready_physical_measurements(data)
    return [
        field
        for field in PHYSICAL_ANALYSIS_FIELDS
        if field in clean.columns and pd.to_numeric(clean[field], errors="coerce").notna().any()
    ]


def build_physical_summary(
    data: pd.DataFrame,
    *,
    total_objects: int | None = None,
) -> pd.DataFrame:
    clean = analysis_ready_physical_measurements(data)
    denominator = int(total_objects) if total_objects is not None else len(clean)
    rows = []
    for field in available_physical_analysis_fields(clean):
        values = pd.to_numeric(clean[field], errors="coerce").dropna()
        rows.append(
            {
                "field": field,
                "measurement": PHYSICAL_ANALYSIS_FIELDS[field],
                "valid_objects": int(len(values)),
                "coverage_%": 100.0 * len(values) / denominator if denominator else 0.0,
                "q25": float(values.quantile(0.25)),
                "median": float(values.median()),
                "q75": float(values.quantile(0.75)),
            }
        )
    return pd.DataFrame(rows)


def build_grouped_physical_summary(
    physical_data: pd.DataFrame,
    memberships: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    if physical_data.empty or memberships.empty or group_column not in memberships.columns:
        return pd.DataFrame()

    membership = memberships[["object_id", group_column]].copy()
    membership["object_id"] = normalize_object_ids(membership["object_id"])
    membership = membership.drop_duplicates("object_id")
    clean = analysis_ready_physical_measurements(physical_data)
    merged = membership.merge(clean, on="object_id", how="left")

    rows = []
    for group_value, group_df in merged.groupby(group_column, dropna=False):
        group_summary = build_physical_summary(group_df, total_objects=len(group_df))
        if group_summary.empty:
            continue
        group_summary.insert(0, group_column, group_value)
        rows.append(group_summary)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def normalize_physical_filters(
    raw_filters: Iterable[dict[str, object]],
    available_fields: Iterable[str],
) -> tuple[dict[str, object], ...]:
    available = set(available_fields)
    normalized = []
    for raw_filter in raw_filters:
        field = str(raw_filter.get("field", ""))
        operator = str(raw_filter.get("operator", "between"))
        if not raw_filter.get("enabled", True) or field not in available:
            continue
        if operator not in PHYSICAL_FILTER_OPERATORS:
            continue

        if operator == "between":
            lower = float(raw_filter.get("lower", 0.0))
            upper = float(raw_filter.get("upper", 0.0))
            if not isfinite(lower) or not isfinite(upper):
                continue
            lower, upper = sorted((lower, upper))
            normalized.append(
                {"field": field, "operator": operator, "lower": lower, "upper": upper}
            )
        else:
            value = float(raw_filter.get("value", 0.0))
            if isfinite(value):
                normalized.append({"field": field, "operator": operator, "value": value})
    return tuple(normalized)


def physical_filter_signature(filters: tuple[dict[str, object], ...]) -> tuple:
    signature = []
    for item in filters:
        if item["operator"] == "between":
            signature.append(
                (
                    item["field"],
                    item["operator"],
                    round(float(item["lower"]), 8),
                    round(float(item["upper"]), 8),
                )
            )
        else:
            signature.append(
                (
                    item["field"],
                    item["operator"],
                    round(float(item["value"]), 8),
                )
            )
    return tuple(signature)


def format_physical_filter(item: dict[str, object]) -> str:
    label = PHYSICAL_ANALYSIS_FIELDS.get(str(item["field"]), str(item["field"]))
    if item["operator"] == "between":
        return f"{label} between {float(item['lower']):.4g} and {float(item['upper']):.4g}"
    return f"{label} {item['operator']} {float(item['value']):.4g}"


def apply_physical_filters(
    data: pd.DataFrame,
    physical_data: pd.DataFrame,
    filters: tuple[dict[str, object], ...],
) -> pd.DataFrame:
    if not filters:
        return data.copy()
    if data.empty or physical_data.empty:
        return data.iloc[0:0].copy()

    source = data.copy()
    source_ids = normalize_object_ids(source["object_id"])
    clean = analysis_ready_physical_measurements(physical_data)
    fields = list(dict.fromkeys(str(item["field"]) for item in filters))
    lookup = clean[["object_id", *fields]].drop_duplicates("object_id")
    working = pd.DataFrame({"object_id": source_ids}, index=source.index).merge(
        lookup,
        on="object_id",
        how="left",
    )
    working.index = source.index

    mask = pd.Series(True, index=source.index)
    for item in filters:
        values = pd.to_numeric(working[str(item["field"])], errors="coerce")
        if item["operator"] == "between":
            current = values.between(
                float(item["lower"]),
                float(item["upper"]),
                inclusive="both",
            )
        elif item["operator"] == ">=":
            current = values >= float(item["value"])
        else:
            current = values <= float(item["value"])
        mask &= current.fillna(False).to_numpy()

    return source.loc[mask].copy()


def merge_physical_measurements(
    data: pd.DataFrame,
    physical_data: pd.DataFrame,
) -> pd.DataFrame:
    if data.empty or physical_data.empty or "object_id" not in data.columns:
        return data.copy()

    base = data.copy()
    base["object_id"] = normalize_object_ids(base["object_id"])
    physical = clean_physical_measurements(physical_data).drop_duplicates("object_id")
    export_columns = [
        column
        for column in ("object_id", *PHYSICAL_ANALYSIS_FIELDS)
        if column in physical.columns
    ]
    physical = physical[export_columns]
    duplicate_columns = {
        column for column in physical.columns if column in base.columns and column != "object_id"
    }
    physical = physical.rename(
        columns={column: f"physical_{column}" for column in duplicate_columns}
    )
    return base.merge(physical, on="object_id", how="left")
