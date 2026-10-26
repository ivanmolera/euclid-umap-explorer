from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from .birch_cache import source_signature
from .catalogs import load_physical_measurements
from .cluster_graph import (
    GRAPH_HELP,
    MORPHOLOGY_GRAPH_FIELDS,
    build_cluster_profiles,
    build_morphology_graph,
    build_morphology_graph_figure,
    membership_layout_key,
    membership_signature,
    selected_graph_cluster,
)
from .components import render_help_label
from .config import PHYSICAL_MEASUREMENTS_PATH
from .images import object_image_path, thumbnail_image_src
from .physical import PHYSICAL_ANALYSIS_FIELDS, PHYSICAL_QUERY_FIELDS
from .runtime import log_app_event


@st.cache_data(show_spinner=False, max_entries=3, ttl=3600)
def load_morphology_profiles(
    path: str, source_version: str, membership_key: str, _memberships: pd.DataFrame,
) -> pd.DataFrame:
    # Explicit full membership hash avoids Streamlit's sampled large-frame hash.
    measurements = load_physical_measurements(
        path, _memberships.object_id, columns=PHYSICAL_QUERY_FIELDS,
    )
    return build_cluster_profiles(_memberships, measurements)


def render_morphology_graph(clustered_df: pd.DataFrame, view_model: dict) -> int | None:
    if not st.toggle("Show morphological graph", key="show_morphology_graph", value=False):
        return None
    state = st.session_state.get("morphology_profile_state")
    if state is None or state["data"] is not clustered_df:
        state = {"data": clustered_df, "membership": membership_signature(clustered_df)}
        st.session_state["morphology_profile_state"] = state
        for key in ("morphology_graph_fields", "morphology_graph_neighbors"):
            st.session_state.pop(key, None)
    visible_ids = [preview["cluster_id"] for preview in view_model["clusters"]]
    visible_summary = view_model["cluster_download_df"].set_index("cluster").loc[visible_ids].reset_index()
    render_help_label("Cluster profiles", GRAPH_HELP)
    with st.container(
        horizontal=True, vertical_alignment="center", width=1000, wrap=False,
        key="morphology_graph_properties",
    ):
        st.markdown("Profile properties", width="content")
        fields = st.multiselect(
            "Profile properties", list(PHYSICAL_ANALYSIS_FIELDS),
            default=list(state.get("fields", MORPHOLOGY_GRAPH_FIELDS)),
            format_func=PHYSICAL_ANALYSIS_FIELDS.get,
            key="morphology_graph_fields", label_visibility="collapsed",
        )
    with st.container(
        horizontal=True, vertical_alignment="center", gap="small",
        key="morphology_graph_controls",
    ):
        with st.container(horizontal=True, vertical_alignment="center", width="content"):
            st.markdown("Nearest profiles per cluster", width="content")
            neighbors = st.number_input(
                "Nearest profiles per cluster", min_value=1, max_value=4,
                value=state.get("neighbors", 2), step=1,
                key="morphology_graph_neighbors", label_visibility="collapsed", width=120,
            )
        refresh_graph = st.button("Build / refresh graph", icon=":material/hub:")
    if refresh_graph or not state.get("initial_build_attempted"):
        state["initial_build_attempted"] = True
        if not fields:
            st.info("Select at least one profile property.")
        else:
            try:
                with st.spinner("Loading catalogue profiles..."):
                    version = source_signature(PHYSICAL_MEASUREMENTS_PATH)
                    profiles = load_morphology_profiles(
                        PHYSICAL_MEASUREMENTS_PATH, json.dumps(version, sort_keys=True),
                        state["membership"], clustered_df[["object_id", "cluster"]],
                    )
                    model = build_morphology_graph(profiles, visible_summary, fields, neighbors=neighbors)
                # Keep the last applied graph and its parameters together until a build succeeds.
                state.update(
                    profiles=profiles, source=version, model=model, visible_ids=visible_ids,
                    fields=tuple(fields), neighbors=neighbors,
                    loaded_at=datetime.now(timezone.utc).isoformat(),
                )
            except Exception as exc:
                log_app_event("morphology_graph_load_failed", error_type=type(exc).__name__)
                st.warning("Could not load catalogue profiles. Existing cluster results are unchanged.")
    if "model" not in state:
        return None
    if state["visible_ids"] != visible_ids:
        state["model"] = build_morphology_graph(
            state["profiles"], visible_summary, state["fields"], neighbors=state["neighbors"],
        )
        state["visible_ids"] = visible_ids
    model = state["model"]
    st.caption(
        f"Profiles: {len(model['nodes'])} / {len(visible_ids)} preview clusters. "
        "Colour: enrichment. Size: number of objects."
    )
    excluded = model["diagnostics"].loc[~model["diagnostics"].included]
    if not excluded.empty:
        st.dataframe(excluded, hide_index=True, use_container_width=True)
    selected = None
    if model["meta"]["status"] == "ready":
        thumbnails = {}
        for preview in view_model["clusters"]:
            row = preview["canonical_row"]
            if row is None or preview["cluster_id"] not in model["positions"]:
                continue
            try:
                path = object_image_path(row)
                if path:
                    thumbnails[preview["cluster_id"]] = thumbnail_image_src(path)
            except OSError:
                pass
        key = f"morphology_graph_{state['membership']}_{membership_layout_key(model)}"
        st.markdown(
            f"<style>.st-key-{key} .imagelayer image {{pointer-events: none;}}</style>",
            unsafe_allow_html=True,
        )
        event = st.plotly_chart(
            build_morphology_graph_figure(model, thumbnails), key=key,
            use_container_width=True, on_select="rerun", selection_mode="points",
        )
        selected = selected_graph_cluster(event, model["nodes"].cluster)
        if selected == st.session_state.get(f"{key}_handled_selection"):
            selected = None
        elif selected is not None:
            st.session_state[f"{key}_handled_selection"] = selected
    else:
        st.info("At least two complete, non-constant cluster profiles are needed for this graph.")
    return selected
