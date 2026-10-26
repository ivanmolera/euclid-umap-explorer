from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from .birch_cache import source_signature
from .cluster_graph import GRAPH_MIN_COVERAGE, MORPHOLOGY_GRAPH_FIELDS, membership_signature
from .cluster_graph_ui import load_morphology_profiles
from .components import render_help_label, show_object_details
from .config import PHYSICAL_MEASUREMENTS_PATH
from .multivariate import (
    CLUSTER_COLOURS, build_parallel_figure, cluster_parallel_frame, complete_parallel_profiles,
    filter_parallel_profiles, normalize_parallel_ranges, object_parallel_frame,
    parallel_chart_sample, parallel_signature,
)
from .physical import PHYSICAL_ANALYSIS_FIELDS
from .runtime import log_app_event


CLUSTER_HELP = (
    "Each line is a cluster's median profile. Each selected "
    "property needs 80% valid coverage. Axes retain their original units. "
    "Range selections are exploratory and do not change BIRCH or its metrics."
)
OBJECT_HELP = (
    "Each line is one object with valid selected properties and colour. "
    "Catalogue quality flags apply; missing values are not imputed. "
    "Unknown is unlabelled, not a confirmed non-lens. Ranges affect this view only."
)


@st.cache_resource
def parallel_component():
    from plotly.offline import get_plotlyjs
    from streamlit.components.v2 import component

    javascript = Path(__file__).with_name("parallel_coordinates.js").read_text()
    return component("escope_parallel_coordinates", js=get_plotlyjs() + "\n" + javascript, isolate_styles=False)


def _on_brush_change() -> None:
    pass


def _parallel_view(frame: pd.DataFrame, *, prefix: str, clusters: bool, dataset_key: str = "") -> pd.DataFrame | None:
    with st.container(horizontal=True, vertical_alignment="bottom", gap="small"):
        fields = st.multiselect(
            "Properties", list(PHYSICAL_ANALYSIS_FIELDS), default=list(MORPHOLOGY_GRAPH_FIELDS),
            format_func=PHYSICAL_ANALYSIS_FIELDS.get, key=f"{prefix}_fields",
        )
        colours = CLUSTER_COLOURS if clusters else {"Label": "label", **{v: k for k, v in PHYSICAL_ANALYSIS_FIELDS.items()}}
        colour_label = st.selectbox("Colour", list(colours), key=f"{prefix}_colour", width=220)
        reset = st.button("", icon=":material/restart_alt:", help="Reset ranges", key=f"{prefix}_reset")
    if len(fields) < 2:
        st.info("Select at least two properties.")
        return None
    colour = colours[colour_label]
    complete = complete_parallel_profiles(frame, fields, colour, min_coverage=GRAPH_MIN_COVERAGE if clusters else None)
    if complete.empty:
        st.info("No complete profiles for the selected properties and colour.")
        return None
    identifier = "cluster" if clusters else "object_id"
    signature = parallel_signature(complete, fields, colour, identifier)
    reset_key = f"{prefix}_reset_generation"
    if reset:
        st.session_state[reset_key] = st.session_state.get(reset_key, 0) + 1
    key = f"{prefix}_chart_{dataset_key}_{signature}_{st.session_state.get(reset_key, 0)}"
    previous = st.session_state.get(key, {})
    brush = previous.get("brush", {})
    if not isinstance(brush, dict):
        brush = {}
    chart_frame = complete if clusters else parallel_chart_sample(complete)
    figure = build_parallel_figure(chart_frame, fields, colour, brush, reference=complete)
    result = parallel_component()(
        key=key, data={"figure": figure, "brush": brush,
                       "label": "Cluster median profiles" if clusters else "Object physical measurements"},
        default={"brush": brush}, on_brush_change=_on_brush_change, height=410,
    )
    result_brush = result.get("brush")
    ranges = normalize_parallel_ranges(result_brush.get("ranges") if isinstance(result_brush, dict) else None, fields)
    selected = filter_parallel_profiles(complete, ranges, fields)
    noun = "Clusters" if clusters else "Objects"
    st.caption(f"{noun}: {len(frame):,}. Complete profiles: {len(complete):,}. Selected: {len(selected):,}.")
    if len(chart_frame) < len(complete):
        st.caption(f"Chart: {len(chart_frame):,} objects, labelled-priority sample (seed 42). Selection counts use all complete profiles.")
    columns = (["cluster", "n_objects", "n_lenses", "density_percent", "enrichment", "label_entropy_bits"]
               if clusters else ["object_id", "label"])
    display = selected[[*columns, *fields]].head(5000).reset_index(drop=True)
    selection_key = hashlib.sha256(json.dumps(display[identifier].tolist()).encode()).hexdigest()[:16]
    table_key = f"{key}_table_{selection_key}"
    event = st.dataframe(
        display, hide_index=True, width="stretch", height=230, key=table_key,
        on_select="rerun", selection_mode="single-row",
        column_config={
            **{field: st.column_config.NumberColumn(label, format="%.3g") for field, label in PHYSICAL_ANALYSIS_FIELDS.items()},
            "object_id": st.column_config.TextColumn("Object ID"), "label": "Label",
            "n_objects": "Objects", "n_lenses": "A/B/C", "density_percent": st.column_config.NumberColumn("Density (%)", format="%.2f"),
            "enrichment": st.column_config.NumberColumn("Enrichment (x)", format="%.2f"),
            "label_entropy_bits": st.column_config.NumberColumn("Entropy", format="%.3f"),
        },
    )
    if len(selected) > len(display):
        st.caption(f"Table: first {len(display):,} of {len(selected):,} selected profiles.")
    rows = event.selection.rows
    if not rows or not 0 <= rows[0] < len(display):
        st.session_state.pop(f"{prefix}_handled_selection", None)
        return None
    chosen = display.iloc[rows[0]][identifier]
    if clusters and st.session_state.get(f"{prefix}_handled_selection") == (table_key, int(chosen)):
        return None
    st.session_state[f"{prefix}_handled_selection"] = (table_key, int(chosen) if clusters else str(chosen))
    return selected.loc[selected[identifier] == chosen].iloc[[0]]


def render_cluster_multivariate(clustered_df: pd.DataFrame, view_model: dict) -> int | None:
    if not st.toggle("Show cluster multivariate analysis", key="show_cluster_multivariate", value=False):
        return None
    render_help_label("Cluster medians", CLUSTER_HELP)
    state = st.session_state.get("multivariate_profile_state")
    if state is None or state["data"] is not clustered_df:
        try:
            graph_state = st.session_state.get("morphology_profile_state", {})
            if graph_state.get("data") is clustered_df and "profiles" in graph_state:
                profiles = graph_state["profiles"]
            else:
                with st.spinner("Loading catalogue profiles..."):
                    version = source_signature(PHYSICAL_MEASUREMENTS_PATH)
                    profiles = load_morphology_profiles(
                        PHYSICAL_MEASUREMENTS_PATH, json.dumps(version, sort_keys=True),
                        membership_signature(clustered_df), clustered_df[["object_id", "cluster"]],
                    )
            state = {"data": clustered_df, "profiles": profiles}
            st.session_state["multivariate_profile_state"] = state
        except Exception as exc:
            log_app_event("multivariate_profiles_failed", error_type=type(exc).__name__)
            st.warning("Could not load cluster profiles. Existing clustering results are unchanged.")
            return None
    scope = st.segmented_control(
        "Clusters", ["First 20 previews", "All clusters"], default="First 20 previews",
        key="cluster_multivariate_scope", label_visibility="collapsed",
    )
    summary = view_model["cluster_download_df"]
    if scope != "All clusters":
        visible = [preview["cluster_id"] for preview in view_model["clusters"]]
        summary = summary.loc[summary.cluster.isin(visible)]
    frame = cluster_parallel_frame(state["profiles"], summary)
    selected = _parallel_view(frame, prefix="cluster_multivariate", clusters=True, dataset_key=str(id(clustered_df)))
    return int(selected.cluster.iloc[0]) if selected is not None else None


def render_object_multivariate(
    cluster_df: pd.DataFrame, filtered_cluster_df: pd.DataFrame, physical_df: pd.DataFrame,
    *, dataset_key: str = "",
) -> None:
    if not st.toggle("Show object multivariate analysis", key="show_object_multivariate", value=False):
        return
    render_help_label("Object measurements", OBJECT_HELP)
    scope = st.segmented_control(
        "Objects", ["Full cluster", "After active filters"], default="Full cluster",
        key="object_multivariate_scope", label_visibility="collapsed",
    )
    objects = filtered_cluster_df if scope == "After active filters" else cluster_df
    if objects.empty or physical_df.empty:
        st.info("No physical measurements for this object selection.")
        return
    frame = object_parallel_frame(objects, physical_df)
    selected = _parallel_view(frame, prefix="object_multivariate", clusters=False, dataset_key=dataset_key)
    if selected is not None:
        show_object_details(selected.iloc[0], key="multivariate_object")
