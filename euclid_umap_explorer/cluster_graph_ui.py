from __future__ import annotations

import json
from datetime import datetime, timezone

import networkx as nx
import pandas as pd
import sklearn
import streamlit as st

from .birch_cache import source_signature
from .catalogs import load_physical_measurements
from .cluster_graph import (
    GRAPH_HELP,
    MORPHOLOGY_GRAPH_FIELDS,
    build_cluster_profiles,
    build_morphology_graph,
    build_morphology_graph_figure,
    graph_download_bytes,
    membership_layout_key,
    membership_signature,
    selected_graph_cluster,
)
from .components import render_help_label
from .config import APP_VERSION, PHYSICAL_MEASUREMENTS_PATH
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


def render_morphology_graph(
    clustered_df: pd.DataFrame, view_model: dict, params: dict, priority: str,
) -> int | None:
    if not st.toggle("Morphological graph", key="show_morphology_graph", value=False):
        return None
    render_help_label("Cluster profiles", GRAPH_HELP)
    with st.container(
        horizontal=True, vertical_alignment="center", width=1000, wrap=False,
        key="morphology_graph_properties",
    ):
        st.markdown("Profile properties", width="content")
        fields = st.multiselect(
            "Profile properties", list(PHYSICAL_ANALYSIS_FIELDS),
            default=list(MORPHOLOGY_GRAPH_FIELDS), format_func=PHYSICAL_ANALYSIS_FIELDS.get,
            key="morphology_graph_fields", label_visibility="collapsed",
        )
    with st.container(
        horizontal=True, vertical_alignment="center", gap="small",
        key="morphology_graph_controls",
    ):
        with st.container(horizontal=True, vertical_alignment="center", width="content"):
            st.markdown("Nearest profiles per cluster", width="content")
            neighbors = st.number_input(
                "Nearest profiles per cluster", min_value=1, max_value=4, value=2, step=1,
                key="morphology_graph_neighbors", label_visibility="collapsed", width=120,
            )
        show_cutouts = st.checkbox(
            "Show canonical cutouts", key="morphology_graph_cutouts",
            help="Same PCA representatives as the previews.",
        )
        refresh_graph = st.button("Build / refresh graph", icon=":material/hub:")
    state = st.session_state.get("morphology_profile_state")
    if state is None or state["data"] is not clustered_df:
        state = {"data": clustered_df, "membership": membership_signature(clustered_df)}
        st.session_state["morphology_profile_state"] = state
    if refresh_graph:
        try:
            with st.spinner("Loading catalogue profiles..."):
                version = source_signature(PHYSICAL_MEASUREMENTS_PATH)
                profiles = load_morphology_profiles(
                    PHYSICAL_MEASUREMENTS_PATH, json.dumps(version, sort_keys=True),
                    state["membership"], clustered_df[["object_id", "cluster"]],
                )
            state.update(
                profiles=profiles, source=version,
                loaded_at=datetime.now(timezone.utc).isoformat(),
            )
        except Exception as exc:
            log_app_event("morphology_graph_load_failed", error_type=type(exc).__name__)
            st.warning("Could not load catalogue profiles. Existing cluster results are unchanged.")
    if "profiles" not in state:
        return None
    if not fields:
        st.info("Select at least one profile property.")
        return None
    visible_ids = [model["cluster_id"] for model in view_model["clusters"]]
    visible_summary = view_model["cluster_download_df"].set_index("cluster").loc[visible_ids].reset_index()
    model = build_morphology_graph(state["profiles"], visible_summary, fields, neighbors=neighbors)
    st.caption(
        f"Profiles: {len(model['nodes'])} / {len(visible_ids)} preview clusters. "
        "Colour: enrichment. Size: number of objects."
    )
    excluded = model["diagnostics"].loc[~model["diagnostics"].included]
    if not excluded.empty:
        st.dataframe(excluded, hide_index=True, use_container_width=True)
    provenance = {
        "app_version": APP_VERSION, "membership_sha256": state["membership"],
        "physical_source": state["source"], "loaded_at_utc": state["loaded_at"],
        "birch": params, "priority": priority,
        "sklearn_version": sklearn.__version__, "networkx_version": nx.__version__,
    }
    selected = None
    if model["meta"]["status"] == "ready":
        thumbnails = {}
        if show_cutouts:
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
        selection_col, action_col = st.columns([3, 1], vertical_alignment="bottom")
        chosen = selection_col.selectbox(
            "Graph cluster", model["nodes"].cluster.astype(int).tolist(), index=None,
            format_func=lambda value: f"Cluster {value}", key=f"{key}_cluster",
        )
        if action_col.button(
            "Analyse", icon=":material/arrow_forward:", key=f"{key}_analyse",
            disabled=chosen is None, use_container_width=True,
        ):
            selected = int(chosen)
    else:
        st.info("At least two complete, non-constant cluster profiles are needed for this graph.")
    st.download_button(
        "Download graph data", icon=":material/download:",
        data=graph_download_bytes(model, provenance), file_name="morphological_graph.zip",
        mime="application/zip",
    )
    return selected
