from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from math import isfinite

import numpy as np
import pandas as pd

from .analysis import sample_for_display, umap_point_labels
from .catalogs import normalize_object_ids
from .config import UMAP_POINT_COLORS
from .physical import PHYSICAL_ANALYSIS_FIELDS, analysis_ready_physical_measurements


PARALLEL_MAX_OBJECTS = 20_000
PARALLEL_LABELS = {
    "phz_median": "Photo-z", "phz_pp_median_stellarmass": "Mass (log M_sun)",
    "concentration": "CAS C", "asymmetry": "CAS A", "smoothness": "CAS S",
    "gini": "Gini", "moment_20": "M20", "sersic_sersic_vis_index": "Sersic n",
}
CLUSTER_COLOURS = {
    "Enrichment": "enrichment", "Density (%)": "density_percent",
    "Labelled candidates": "n_lenses", "Entropy": "label_entropy_bits",
}


def cluster_parallel_frame(profiles: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    values = profiles.rename(columns={f"{field}_median": field for field in PHYSICAL_ANALYSIS_FIELDS})
    columns = ["cluster", *[column for field in PHYSICAL_ANALYSIS_FIELDS
                             for column in (field, f"{field}_coverage") if column in values]]
    result = summary.merge(values[columns], on="cluster", how="left", validate="one_to_one")
    result["density_percent"] = result["lens_rate"] * 100
    return result


def object_parallel_frame(objects: pd.DataFrame, measurements: pd.DataFrame) -> pd.DataFrame:
    result = objects.drop(columns=[*PHYSICAL_ANALYSIS_FIELDS, "point_role"], errors="ignore").copy()
    result["object_id"] = normalize_object_ids(result.object_id)
    if result.object_id.isna().any() or result.object_id.duplicated().any():
        raise ValueError("Object profiles require unique, non-missing IDs.")
    if "is_lens" in result:
        result["is_lens"] = result.is_lens.fillna(False).astype(bool)
    clean = analysis_ready_physical_measurements(measurements)
    if not clean.empty:
        clean = clean.copy()
        clean["object_id"] = normalize_object_ids(clean.object_id)
        columns = ["object_id", *[field for field in PHYSICAL_ANALYSIS_FIELDS if field in clean]]
        result = result.merge(clean[columns].drop_duplicates("object_id"), on="object_id", how="left")
    result["label"] = umap_point_labels(result)
    return result


def complete_parallel_profiles(
    frame: pd.DataFrame, fields: Sequence[str], colour: str, *, min_coverage: float | None = None,
) -> pd.DataFrame:
    if not fields or len(set(fields)) != len(fields) or any(field not in PHYSICAL_ANALYSIS_FIELDS for field in fields):
        raise ValueError("Select unique physical/morphological properties.")
    if min_coverage is not None and not 0 < min_coverage <= 1:
        raise ValueError("Invalid profile coverage.")
    result = frame.copy()
    valid = pd.Series(True, index=result.index)
    required = list(dict.fromkeys([*fields, *([colour] if colour != "label" else [])]))
    for field in required:
        values = pd.to_numeric(result.get(field, pd.Series(np.nan, index=result.index)), errors="coerce")
        result[field] = values.replace([np.inf, -np.inf], np.nan)
        valid &= result[field].notna()
        if min_coverage is not None and field in PHYSICAL_ANALYSIS_FIELDS:
            coverage = pd.to_numeric(result.get(f"{field}_coverage", pd.Series(0., index=result.index)), errors="coerce")
            valid &= coverage.ge(min_coverage)
    if colour == "label":
        valid &= result.get("label", pd.Series(pd.NA, index=result.index)).notna()
    return result.loc[valid].reset_index(drop=True)


def normalize_parallel_ranges(raw: object, fields: Sequence[str]) -> dict[str, list[list[float]]]:
    if not isinstance(raw, Mapping):
        return {}
    result = {}
    for field in fields:
        ranges = raw.get(field)
        if not isinstance(ranges, (list, tuple)) or not ranges:
            continue
        if len(ranges) == 2 and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in ranges):
            ranges = [ranges]
        clean = []
        for bounds in ranges[:16]:
            if (isinstance(bounds, (list, tuple)) and len(bounds) == 2
                    and all(isinstance(v, (int, float)) and not isinstance(v, bool) and isfinite(v) for v in bounds)):
                clean.append(sorted(float(v) for v in bounds))
        if clean:
            result[field] = clean
    return result


def filter_parallel_profiles(frame: pd.DataFrame, ranges: object, fields: Sequence[str]) -> pd.DataFrame:
    selected = pd.Series(True, index=frame.index)
    for field, intervals in normalize_parallel_ranges(ranges, fields).items():
        matches = pd.Series(False, index=frame.index)
        for low, high in intervals:
            matches |= frame[field].between(low, high)
        selected &= matches
    return frame.loc[selected].copy()


def parallel_signature(frame: pd.DataFrame, fields: Sequence[str], colour: str, identifier: str) -> str:
    columns = list(dict.fromkeys([identifier, *fields, colour]))
    digest = hashlib.sha256(pd.util.hash_pandas_object(frame[columns], index=False).to_numpy().tobytes())
    digest.update(json.dumps([list(fields), colour, identifier]).encode())
    return digest.hexdigest()[:20]


def parallel_chart_sample(frame: pd.DataFrame) -> pd.DataFrame:
    if len(frame) <= PARALLEL_MAX_OBJECTS:
        return frame
    # Stable source order and the existing labelled-priority sampling policy.
    return sample_for_display(frame.sort_values("object_id"), PARALLEL_MAX_OBJECTS)


def build_parallel_figure(
    frame: pd.DataFrame, fields: Sequence[str], colour: str, brush: Mapping | None = None,
    *, reference: pd.DataFrame | None = None,
) -> dict:
    import plotly.graph_objects as go

    brush = brush or {}
    reference = frame if reference is None else reference
    ranges = normalize_parallel_ranges(brush.get("ranges"), fields)
    order = brush.get("order", [])
    if (not isinstance(order, list) or len(order) != len(fields)
            or not all(isinstance(field, str) for field in order) or set(order) != set(fields)):
        order = list(fields)
    dimensions = []
    for field in order:
        low, high = float(reference[field].min()), float(reference[field].max())
        if low == high:
            padding = max(abs(low) * .05, .5)
            low, high = low - padding, high + padding
        dimension = {"name": field, "label": PARALLEL_LABELS[field], "values": frame[field].tolist(), "range": [low, high]}
        if field in ranges:
            dimension["constraintrange"] = ranges[field][0] if len(ranges[field]) == 1 else ranges[field]
        dimensions.append(dimension)
    if colour == "label":
        labels = ["Unknown", "Grade C", "Grade B", "Grade A", "Lens candidate"]
        labels = [label for label in labels if label in set(reference.label)]
        palette = [UMAP_POINT_COLORS.get(label, "#667788") for label in labels]
        scale = [[value, color] for index, color in enumerate(palette)
                 for value in (index / len(palette), (index + 1) / len(palette))]
        line = {"color": frame.label.map(dict(zip(labels, range(len(labels))))).tolist(),
                "colorscale": scale, "cmin": -.5, "cmax": len(labels) - .5,
                "showscale": True, "colorbar": {"title": "Label", "tickvals": list(range(len(labels))), "ticktext": labels}}
    else:
        low, high = float(reference[colour].min()), float(reference[colour].max())
        if low == high:
            low, high = low - .5, high + .5
        title = next((name for name, field in CLUSTER_COLOURS.items() if field == colour), PARALLEL_LABELS.get(colour, colour))
        line = {"color": frame[colour].tolist(), "colorscale": "Viridis", "cmin": low, "cmax": high,
                "showscale": True, "colorbar": {"title": title}}
    line["colorbar"].update(thickness=12, len=.85)
    figure = go.Figure(go.Parcoords(
        dimensions=dimensions, line=line, labelfont={"size": 12}, tickfont={"size": 10},
        unselected={"line": {"color": "#9099a3", "opacity": .12}},
    ))
    figure.update_layout(template="none", height=390, margin={"l": 35, "r": 135, "t": 50, "b": 25})
    return json.loads(figure.to_json())
