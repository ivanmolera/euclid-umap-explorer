from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .catalogs import load_morphology_object
from .config import MORPH_PATH
from .images import morphology_cutout_path
from .runtime import log_app_event


class EuclidArchiveError(RuntimeError):
    """The archive failed to provide a usable query result, not an empty table."""


def query_archive(query: str):
    from astroquery.esa.euclid import Euclid

    try:
        job = Euclid.launch_job_async(query, verbose=False)
        if job is None:
            raise EuclidArchiveError("The Euclid archive did not return a query job.")
        result = job.get_results()
        if result is None:
            raise EuclidArchiveError("The Euclid archive did not return query results.")
        return result
    except EuclidArchiveError:
        raise
    except Exception as exc:
        raise EuclidArchiveError("The Euclid archive query failed. Please try again later.") from exc


def morphology_search_fallback(object_id: str) -> dict[str, object] | None:
    morphology_df = load_morphology_object(MORPH_PATH, object_id)
    if morphology_df.empty:
        return None
    row = morphology_df.iloc[0].dropna().to_dict()
    try:
        ra, dec = float(row["right_ascension"]), float(row["declination"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (np.isfinite(ra) and np.isfinite(dec) and 0 <= ra < 360 and -90 <= dec <= 90):
        return None
    id_str = str(row.get("id_str", ""))
    parts = id_str.split("_")
    tile_index = ""
    if len(parts) >= 2 and parts[-1] == object_id and parts[-2].isdigit():
        tile_index = parts[-2]
    object_summary = {
        field: serializable_table_value(row[field])
        for field in (
            "object_id", "id_str", "right_ascension", "declination", "segmentation_area",
            "ellipticity", "kron_radius", "flux_detection_total", "flux_vis_sersic",
            "vis_det", "det_quality_flag",
        )
        if field in row
    }
    object_summary["object_id"] = object_id
    object_summary["data_source"] = "ESCOPE Q1 morphology catalogue"
    return {
        "object_id": object_id,
        "object_summary": object_summary,
        "mosaic_summary": {},
        "tile_index": tile_index,
        "file_path": None,
        "cutout_path": morphology_cutout_path(id_str, object_id) if tile_index else None,
        "warnings": [
            "Live Euclid archive data could not be retrieved. "
            "Showing the existing Q1 morphology catalogue and available cutout instead; "
            "mosaic metadata is unavailable."
        ],
    }

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

def fetch_euclid_object_summary(
    object_id: str,
    search_radius_arcmin: float = 0.10,
    instrument: str = "VIS",
) -> dict[str, object]:
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
FROM q1.mer_catalogue
WHERE object_id = {normalized_object_id}
"""

    archive_error = None
    try:
        object_result = query_archive(object_query)
    except EuclidArchiveError as exc:
        archive_error = exc
        object_result = None
    if object_result is None or len(object_result) == 0:
        try:
            fallback = morphology_search_fallback(normalized_object_id)
        except Exception as exc:
            log_app_event("object_search_fallback_failed", object_id=normalized_object_id,
                          error_type=type(exc).__name__)
            raise EuclidArchiveError(
                "The Euclid archive and the Q1 morphology catalogue could not be read. "
                "Please try again later."
            ) from exc
        if fallback is not None:
            log_app_event("object_search_catalogue_fallback", object_id=normalized_object_id)
            return fallback
        if archive_error is not None:
            raise EuclidArchiveError(
                "The Euclid archive is unavailable and no usable Q1 morphology record "
                "was found for this object_id. Please try again later."
            ) from archive_error
        raise ValueError(f"object_id {normalized_object_id} was not found in q1.mer_catalogue "
                         "or the Q1 morphology catalogue.")

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

    warnings = []
    try:
        mosaic_result = query_archive(mosaic_query)
    except EuclidArchiveError:
        mosaic_result = None
    mosaic_summary = {}
    file_path = None
    if mosaic_result is not None and len(mosaic_result):
        mosaic_summary = table_row_to_dict(mosaic_result[0])
        file_path = f"{mosaic_summary['file_path']}/{mosaic_summary['file_name']}"
    else:
        warnings.append("Mosaic metadata could not be retrieved. Available object data and cutouts are still shown.")
    tile_index = str(mosaic_summary.get("tile_index", ""))
    local_cutout_path = morphology_cutout_path(
        f"{tile_index}_{normalized_object_id}" if tile_index else None,
        normalized_object_id,
    )
    if local_cutout_path is None:
        morphology_df = load_morphology_object(MORPH_PATH, normalized_object_id)
        if not morphology_df.empty and "id_str" in morphology_df.columns:
            morphology_id = str(morphology_df.iloc[0]["id_str"])
            if not tile_index and len(morphology_id.split("_")) >= 2:
                tile_index = morphology_id.split("_")[-2]
            local_cutout_path = morphology_cutout_path(
                morphology_id,
                normalized_object_id,
            )

    return {
        "object_id": normalized_object_id,
        "object_summary": object_summary,
        "mosaic_summary": mosaic_summary,
        "tile_index": tile_index,
        "file_path": file_path,
        "cutout_path": local_cutout_path,
        "warnings": warnings,
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
