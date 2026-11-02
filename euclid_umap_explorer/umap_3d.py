from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from .config import UMAP_POINT_COLORS, UMAP_POINT_ORDER, UMAP_POINT_SYMBOLS


def build_3d_payload(data: pd.DataFrame, *, label_column="umap_point_label",
                     colour_subclusters=False) -> dict:
    coordinates = data[["umap_1", "umap_2", "umap_3"]].to_numpy(dtype=float)
    if not np.isfinite(coordinates).all():
        raise ValueError("3D UMAP coordinates must be finite.")
    from plotly.colors import qualitative

    subclusters = colour_subclusters and "hierarchical_subcluster_label" in data
    colour_labels = data["hierarchical_subcluster_label"].astype(str) if subclusters else data[label_column].astype(str)
    names = list(dict.fromkeys(colour_labels)) if subclusters else [
        name for name in UMAP_POINT_ORDER if name in set(colour_labels)
    ]
    colours = ({name: qualitative.Plotly[i % len(qualitative.Plotly)] for i, name in enumerate(names)}
               if subclusters else UMAP_POINT_COLORS)
    groups, group_lookup, points = [], {}, []
    for position, (_, row) in enumerate(data.iterrows()):
        role = str(row.get(label_column, "Unknown"))
        colour_label = str(colour_labels.iloc[position])
        group_key = (role, colour_label)
        if group_key not in group_lookup:
            group_lookup[group_key] = len(groups)
            groups.append({"label": colour_label, "role": role,
                           "color": colours.get(colour_label, "#6f9fc5"),
                           "shape": UMAP_POINT_SYMBOLS.get(role, "circle")})
        points.append({"index": int(row.get("point_index", position)),
                       "id": str(row["object_id"]), "label": role,
                       "xyz": coordinates[position].tolist(), "group": group_lookup[group_key]})
    return {"points": points, "groups": groups,
            "legend": [{"label": name, "color": colours.get(name, "#6f9fc5")} for name in names]}


@st.cache_resource
def scatter_3d_component():
    from streamlit.components.v2 import component

    return component("escope_umap_3d", js=Path(__file__).with_name("umap_3d.bundle.js").read_text(),
                     isolate_styles=False)


def _selection_changed():
    pass


def render_umap_3d(data, *, key, signature, height=680, label_column="umap_point_label"):
    token = hashlib.sha256(repr(signature).encode()).hexdigest()[:16]
    chart_key = f"{key}_{token}"
    reset_key = f"{chart_key}_reset"
    with st.container(horizontal=True, gap="small"):
        if st.button("", icon=":material/restart_alt:", help="Reset 3D camera", key=f"{chart_key}_camera"):
            st.session_state[reset_key] = st.session_state.get(reset_key, 0) + 1
        clear = st.button("", icon=":material/deselect:", help="Clear object selection", key=f"{chart_key}_clear")
    generation = st.session_state.get(reset_key, 0)
    if clear:
        st.session_state[f"{chart_key}_selection_generation"] = st.session_state.get(f"{chart_key}_selection_generation", 0) + 1
    selection_generation = st.session_state.get(f"{chart_key}_selection_generation", 0)
    widget_key = f"{chart_key}_{selection_generation}"
    previous = st.session_state.get(widget_key, {}).get("selection", {"points": []})
    payload = build_3d_payload(data, label_column=label_column,
                               colour_subclusters=st.session_state.get("color_by_subcluster", False)
                               and label_column == "umap_point_label")
    result = scatter_3d_component()(
        key=widget_key, data={**payload, "viewKey": chart_key, "signature": f"{token}_{generation}",
                             "height": height, "selection": previous},
        default={"selection": previous}, on_selection_change=_selection_changed, height=height,
    )
    return {"selection": result.get("selection") or {"points": []}}
