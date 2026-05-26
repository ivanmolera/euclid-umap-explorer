from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from .catalogs import load_morphology_object
from .config import MORPH_PATH
from .images import morphology_cutout_path

def normalize_search_object_id(object_id: str) -> str:
    value = str(object_id).strip()
    if not value:
        raise ValueError("Enter an object_id.")
    try:
        return str(int(value))
    except ValueError as exc:
        raise ValueError("object_id must be an integer value.") from exc

def is_valid_search_object_id(object_id: str) -> bool:
    try:
        normalize_search_object_id(object_id)
    except ValueError:
        return False
    return True

def serializable_table_value(value: object) -> object:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value

def table_row_to_dict(row: object) -> dict[str, object]:
    columns = getattr(row, "colnames", None)
    if columns is None and hasattr(row, "columns"):
        columns = row.columns
    return {column: serializable_table_value(row[column]) for column in columns}

@st.cache_data(show_spinner=False, ttl=24 * 60 * 60)
def fetch_euclid_object_summary(
    object_id: str,
    search_radius_arcmin: float = 0.10,
    instrument: str = "VIS",
) -> dict[str, object]:
    from astroquery.esa.euclid import Euclid

    normalized_object_id = normalize_search_object_id(object_id)

    object_query = f"""
SELECT
    object_id,
    right_ascension,
    declination,
    segmentation_area,
    ellipticity,
    kron_radius,
    flux_detection_total,
    flux_vis_sersic,
    vis_det,
    det_quality_flag
FROM catalogue.mer_catalogue
WHERE object_id = {normalized_object_id}
"""

    object_job = Euclid.launch_job_async(object_query, verbose=False)
    object_result = object_job.get_results()
    if len(object_result) == 0:
        raise ValueError(f"object_id {normalized_object_id} was not found in catalogue.mer_catalogue.")

    object_row = object_result[0]
    object_summary = table_row_to_dict(object_row)
    ra = float(object_row["right_ascension"])
    dec = float(object_row["declination"])

    search_radius_deg = max(float(search_radius_arcmin) / 60.0, 0.5 / 60.0)
    mosaic_query = f"""
SELECT
    file_name,
    file_path,
    instrument_name,
    filter_name,
    product_type,
    tile_index
FROM q1.mosaic_product
WHERE instrument_name = '{instrument}'
  AND INTERSECTS(CIRCLE({ra}, {dec}, {search_radius_deg}), fov) = 1
ORDER BY file_name
"""

    mosaic_job = Euclid.launch_job_async(mosaic_query, verbose=False)
    mosaic_result = mosaic_job.get_results()
    if len(mosaic_result) == 0:
        raise ValueError(f"No {instrument} mosaic product was found for object_id {normalized_object_id}.")

    mosaic_row = mosaic_result[0]
    mosaic_summary = table_row_to_dict(mosaic_row)
    file_path = f"{mosaic_row['file_path']}/{mosaic_row['file_name']}"
    local_cutout_path = morphology_cutout_path(
        f"{mosaic_summary.get('tile_index')}_{normalized_object_id}",
        normalized_object_id,
    )
    if local_cutout_path is None:
        morphology_df = load_morphology_object(MORPH_PATH, normalized_object_id)
        if not morphology_df.empty and "id_str" in morphology_df.columns:
            local_cutout_path = morphology_cutout_path(
                morphology_df.iloc[0]["id_str"],
                normalized_object_id,
            )

    return {
        "object_id": normalized_object_id,
        "object_summary": object_summary,
        "mosaic_summary": mosaic_summary,
        "file_path": file_path,
        "cutout_path": local_cutout_path,
    }

def selected_point_index(event: object) -> int | None:
    if not event:
        return None

    try:
        points = event["selection"]["points"]
    except (KeyError, TypeError):
        return None

    if not points:
        return None

    customdata = points[0].get("customdata")
    if isinstance(customdata, Iterable) and not isinstance(customdata, str):
        customdata = list(customdata)[0] if customdata else None

    try:
        return int(customdata)
    except (TypeError, ValueError):
        return None
