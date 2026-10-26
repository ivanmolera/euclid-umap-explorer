from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections.abc import Mapping, Sequence

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from .catalogs import normalize_object_ids
from .physical import PHYSICAL_ANALYSIS_FIELDS, analysis_ready_physical_measurements


MORPHOLOGY_GRAPH_FIELDS = (
    "concentration", "asymmetry", "smoothness", "gini", "moment_20",
    "sersic_sersic_vis_index",
)
GRAPH_MIN_COVERAGE = 0.8
GRAPH_LAYOUT_SEED = 20261026
GRAPH_LAYOUT_SPACING = 2.0
GRAPH_HELP = (
    "Links join the nearest profiles among the displayed previews, using cluster "
    "medians of the selected catalogue properties. Properties have equal weight "
    "after standardization across all complete cluster profiles in this run. "
    "Every selected property needs 80% valid coverage. Missing values are not "
    "imputed. Labels affect colour only; screen distances are not profile distances."
)


def membership_signature(memberships: pd.DataFrame) -> str:
    membership = memberships[["object_id", "cluster"]].copy()
    membership["object_id"] = normalize_object_ids(membership["object_id"])
    return hashlib.sha256(
        pd.util.hash_pandas_object(membership, index=False).to_numpy().tobytes()
    ).hexdigest()


def build_cluster_profiles(
    memberships: pd.DataFrame,
    measurements: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate all members, rather than the labelled or preview image samples."""
    membership = memberships[["object_id", "cluster"]].copy()
    membership["object_id"] = normalize_object_ids(membership["object_id"])
    if membership.object_id.isna().any() or membership.object_id.duplicated().any():
        raise ValueError("Cluster memberships require unique, non-missing object IDs.")
    clean = analysis_ready_physical_measurements(measurements)
    if not clean.empty:
        clean = clean.drop_duplicates("object_id")
    fields = [field for field in PHYSICAL_ANALYSIS_FIELDS if field in clean]
    if fields:
        merged = membership.merge(clean[["object_id", *fields]], on="object_id", how="left")
        for field in fields:
            merged[field] = pd.to_numeric(merged[field], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
    else:
        merged = membership
    grouped = merged.groupby("cluster", sort=True)
    result = grouped.size().rename("n_profile_objects").to_frame()
    for field in fields:
        result[f"{field}_median"] = grouped[field].median()
        result[f"{field}_q25"] = grouped[field].quantile(0.25)
        result[f"{field}_q75"] = grouped[field].quantile(0.75)
        result[f"{field}_coverage"] = grouped[field].count() / result.n_profile_objects
    return result.reset_index()


def build_morphology_graph(
    profiles: pd.DataFrame,
    visible_summary: pd.DataFrame,
    fields: Sequence[str] = MORPHOLOGY_GRAPH_FIELDS,
    *,
    neighbors: int = 2,
    min_coverage: float = GRAPH_MIN_COVERAGE,
) -> dict:
    fields = tuple(fields)
    profiles = profiles.sort_values("cluster").reset_index(drop=True)
    if not fields or len(set(fields)) != len(fields) or any(
        field not in PHYSICAL_ANALYSIS_FIELDS for field in fields
    ):
        raise ValueError("Select unique physical/morphological properties, not outcomes.")
    if not 0 < min_coverage <= 1 or neighbors < 1:
        raise ValueError("Invalid graph coverage or neighbor count.")
    if profiles.cluster.duplicated().any() or visible_summary.cluster.duplicated().any():
        raise ValueError("Cluster profiles and preview rows must be unique.")
    columns = [f"{field}_median" for field in fields]
    values = pd.DataFrame(index=profiles.index)
    for field, column in zip(fields, columns):
        values[column] = pd.to_numeric(profiles.get(column, np.nan), errors="coerce")
        coverage = pd.to_numeric(
            profiles.get(f"{field}_coverage", pd.Series(0.0, index=profiles.index)),
            errors="coerce",
        )
        values[column] = values[column].where(coverage.ge(min_coverage))
    values = values.replace([np.inf, -np.inf], np.nan)
    complete = values.notna().all(axis=1)
    reference = profiles.loc[complete, ["cluster"]].join(values.loc[complete])
    variable = [column for column in columns if reference[column].nunique() > 1]
    visible_ids = visible_summary.cluster.astype(int).tolist()
    diagnostics = pd.DataFrame({"cluster": visible_ids})
    reasons = {
        int(profiles.loc[index, "cluster"]): ", ".join(
            PHYSICAL_ANALYSIS_FIELDS[field]
            for field, column in zip(fields, columns) if pd.isna(values.loc[index, column])
        ) for index in profiles.index
    }
    diagnostics["missing_or_low_coverage"] = diagnostics.cluster.map(reasons).fillna("No profile")
    diagnostics["included"] = diagnostics.missing_or_low_coverage.eq("")
    selected = visible_summary.merge(profiles, on="cluster", how="left", validate="one_to_one")
    selected = selected[selected.cluster.isin(diagnostics.loc[diagnostics.included, "cluster"])].copy()
    selected = selected.sort_values("cluster").reset_index(drop=True)
    meta = {
        "fields": list(fields), "used_fields": [column.removesuffix("_median") for column in variable],
        "min_coverage": min_coverage, "neighbors": int(neighbors),
        "reference_clusters": len(reference), "visible_clusters": len(visible_ids),
        "included_clusters": len(selected), "layout_seed": GRAPH_LAYOUT_SEED,
        "layout_spacing": GRAPH_LAYOUT_SPACING, "layout_iterations": 300,
        "scaling": "StandardScaler across all complete cluster medians in this run",
        "neighbors_scope": "Displayed preview clusters only; undirected union of kNN",
        "labels_used_in_distances": False,
        "status": "ready",
    }
    result = {"nodes": selected, "reference_profiles": profiles.copy(), "edges": pd.DataFrame(columns=[
        "source", "target", "profile_distance", "main_differences"
    ]), "diagnostics": diagnostics, "positions": {}, "meta": meta}
    if len(reference) < 2 or not variable:
        meta["status"] = "insufficient_variable_profiles"
        return result
    scaler = StandardScaler().fit(reference[variable])
    meta.update(scaler_mean=scaler.mean_.tolist(), scaler_scale=scaler.scale_.tolist())
    if len(selected) < 2:
        meta["status"] = "insufficient_visible_profiles"
        return result
    coordinates = scaler.transform(selected[variable])
    distances = pairwise_distances(coordinates)
    ids = selected.cluster.astype(int).to_numpy()
    k = min(neighbors, len(ids) - 1)
    pairs = set()
    for index, cluster_id in enumerate(ids):
        order = np.lexsort((ids, distances[index]))
        nearest = [other for other in order if other != index][:k]
        pairs.update(tuple(sorted((int(cluster_id), int(ids[other])))) for other in nearest)
    index_by_id = {int(cluster_id): index for index, cluster_id in enumerate(ids)}
    rows = []
    for source, target in sorted(pairs):
        left, right = index_by_id[source], index_by_id[target]
        contributions = np.abs(coordinates[left] - coordinates[right])
        order = np.argsort(-contributions, kind="stable")[:3]
        differences = "; ".join(
            f"{PHYSICAL_ANALYSIS_FIELDS[variable[index].removesuffix('_median')]}: "
            f"{contributions[index]:.2f} standardized units" for index in order
        )
        rows.append({"source": source, "target": target,
                     "profile_distance": float(distances[left, right]),
                     "main_differences": differences})
    edges = pd.DataFrame(rows)
    graph = nx.Graph()
    graph.add_nodes_from(int(cluster_id) for cluster_id in ids)
    graph.add_edges_from((int(row.source), int(row.target)) for row in edges.itertuples())
    layout = nx.spring_layout(
        graph, seed=GRAPH_LAYOUT_SEED, weight=None, iterations=300, k=GRAPH_LAYOUT_SPACING,
    )
    result.update(edges=edges, positions={int(key): value.tolist() for key, value in layout.items()})
    return result


def build_morphology_graph_figure(model: dict, thumbnails: Mapping[int, str] | None = None):
    import plotly.graph_objects as go

    nodes, edges, positions = model["nodes"], model["edges"], model["positions"]
    figure = go.Figure()
    x, y, hover_x, hover_y, edge_text = [], [], [], [], []
    for row in edges.itertuples():
        left, right = positions[int(row.source)], positions[int(row.target)]
        x.extend((left[0], right[0], None))
        y.extend((left[1], right[1], None))
        hover_x.append((left[0] + right[0]) / 2)
        hover_y.append((left[1] + right[1]) / 2)
        edge_text.append(f"Clusters {row.source} - {row.target}<br>Profile distance: "
                         f"{row.profile_distance:.3f}<br>{row.main_differences.replace('; ', '<br>')}")
    figure.add_trace(go.Scatter(x=x, y=y, mode="lines", line={"color": "#87929d", "width": 1},
                               hoverinfo="skip", showlegend=False))
    figure.add_trace(go.Scatter(x=hover_x, y=hover_y, mode="markers", marker={"size": 12, "opacity": 0},
                               text=edge_text, hovertemplate="%{text}<extra></extra>", showlegend=False))
    ids = nodes.cluster.astype(int).tolist()
    n_objects = nodes.n_objects.to_numpy(dtype=float)
    sizes = 14 + 6 * np.log1p(n_objects) / max(float(np.log1p(n_objects).max()), 1.0)
    node_text = []
    for row in nodes.to_dict("records"):
        properties = "<br>".join(
            f"{PHYSICAL_ANALYSIS_FIELDS[field]}: {row[f'{field}_median']:.3g} "
            f"({row[f'{field}_coverage']:.0%} valid)" for field in model["meta"]["fields"]
        )
        node_text.append(f"Cluster {int(row['cluster'])}<br>Objects: {int(row['n_objects']):,}"
                         f"<br>Labelled candidates: {int(row['n_lenses']):,}"
                         f"<br>Enrichment: {row['enrichment']:.2f}x<br>{properties}")
    figure.add_trace(go.Scatter(
        x=[positions[key][0] for key in ids], y=[positions[key][1] for key in ids],
        mode="markers+text", text=[str(key) for key in ids], textposition="bottom center",
        textfont={"size": 11}, customdata=[[key] for key in ids], hovertext=node_text,
        hovertemplate="%{hovertext}<extra></extra>",
        marker={"size": sizes, "color": nodes.enrichment.tolist(), "colorscale": "Viridis",
                "showscale": True, "colorbar": {"title": "Enrichment (x)", "thickness": 10,
                                               "orientation": "h", "len": .8,
                                               "x": .5, "xanchor": "center", "y": -.02},
                "line": {"color": "#f1f5f9", "width": 1}}, showlegend=False,
    ))
    for key, source in (thumbnails or {}).items():
        if key in positions:
            figure.add_layout_image(dict(source=source, x=positions[key][0], y=positions[key][1],
                                         xref="x", yref="y", xanchor="center", yanchor="middle",
                                         sizex=.09, sizey=.09, sizing="contain", layer="above"))
    figure.update_layout(
        height=550, margin={"l": 5, "r": 5, "t": 15, "b": 60},
        xaxis={"visible": False, "range": [-1.2, 1.2], "fixedrange": False},
        yaxis={"visible": False, "range": [-1.2, 1.2], "scaleanchor": "x", "scaleratio": 1},
        hovermode="closest", dragmode="pan", uirevision=membership_layout_key(model),
        clickmode="event+select",
    )
    return figure


def membership_layout_key(model: dict) -> str:
    payload = {"clusters": model["nodes"].cluster.astype(int).tolist(), "meta": model["meta"],
               "positions": model["positions"], "edges": model["edges"].to_dict("records")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]


def selected_graph_cluster(event: Mapping, allowed_ids: Sequence[int]) -> int | None:
    allowed = set(int(value) for value in allowed_ids)
    for point in event.get("selection", {}).get("points", []):
        if point.get("curve_number") != 2:
            continue
        data = point.get("customdata") or []
        try:
            value = int(data[0])
        except (ValueError, TypeError, IndexError):
            continue
        if value in allowed:
            return value
    return None


def graph_download_bytes(model: dict, provenance: Mapping) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ("nodes", "edges", "diagnostics", "reference_profiles"):
            archive.writestr(f"{name}.csv", model[name].to_csv(index=False))
        archive.writestr("layout.json", json.dumps(model["positions"], sort_keys=True))
        archive.writestr("manifest.json", json.dumps({**model["meta"], "provenance": dict(provenance)},
                                                    sort_keys=True, indent=2, allow_nan=False))
    return buffer.getvalue()
