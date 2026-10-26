from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
import hashlib
import html
from pathlib import Path
import time
from urllib.parse import quote

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as st_components

from .analysis import (
    add_cluster_extreme_roles,
    format_pca_filter,
    lens_grade_sort_key,
    recommended_numeric_filter,
)
from .catalogs import (
    load_lens_catalog,
    load_morphology_object,
    load_morphology_objects,
    load_physical_measurement_object,
    normalize_object_ids,
)
from .config import (
    ANALYSIS_FLOW_STEPS,
    CLUSTER_EXTREME_HELP,
    CUTOUT_BASE,
    DEFAULT_CLUSTER_FEATURES,
    ENABLE_ARC_LIKE_STRUCTURE_DETECTION,
    ENABLE_SLED_CHECK,
    ENRICHMENT_HELP,
    LABEL_DENSITY_HELP,
    LABEL_ENTROPY_HELP,
    LENS_GRADE_HELP_NOTE,
    LENS_GRADE_OPTIONS,
    LENS_IMG_BASE,
    LENS_PATH,
    MORPH_PATH,
    PARQUET_PATH,
    PHYSICAL_MEASUREMENTS_PATH,
    PCA_FILTER_OPERATORS,
    PCA_FILTER_RECOMMENDATION_MIN_LENSES,
    PCA_FILTER_RECOMMENDATION_MIN_OBJECTS,
    PCA_FILTER_RECOMMENDATION_MIN_RECALL,
    PHYSICAL_FILTER_RECOMMENDATION_MIN_LENSES,
    PHYSICAL_FILTER_RECOMMENDATION_MIN_UNKNOWNS,
    PHYSICAL_FILTER_RECOMMENDATION_MIN_VALID_OBJECTS,
    RECOMMENDED_FILTER_HELP,
    SUMMARY_DISTPLOT_MAX_POINTS_PER_GROUP,
    SUMMARY_HISTOGRAM_BINS,
    SUMMARY_HISTOGRAM_FEATURE_LIMIT,
    SUMMARY_LENS_OBJECTS,
    SUMMARY_RANDOM_OBJECTS,
    SUMMARY_THUMBNAIL_WIDTH,
    SUMMARY_VISUAL_CLUSTER_LIMIT,
)
from .downloads import dataframe_to_csv_bytes, object_search_download_df
from .euclid_search import fetch_euclid_object_summary
from .images import (
    display_image_src,
    lens_image_path,
    morphology_cutout_path,
    object_image_path,
    render_image_caption,
    render_image_src,
    thumbnail_image_src,
)
from .runtime import log_app_event
from .help_tooltips import HELP_TOOLTIP_SCRIPT
from .storage import path_exists
from .sled_ui import render_sled_check
from .service_icons import ALADIN_LINK_CSS, aladin_icon_link
from .physical import (
    PHYSICAL_ANALYSIS_FIELDS,
    PHYSICAL_FILTER_OPERATORS,
    PHYSICAL_MEASUREMENT_HELP,
    analysis_ready_physical_measurements,
    available_physical_analysis_fields,
    build_physical_summary,
    physical_measurement_display_rows,
)

# Streamlit's default row and header height is 35 px, plus the table borders.
OBJECT_DETAIL_TABLE_HEIGHT = 35 * (8 + 1) + 3

COPY_ICON_SRC = "data:image/svg+xml;base64," + base64.b64encode(
    b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
    b'stroke="#cbd5e1" stroke-width="2" stroke-linecap="round" '
    b'stroke-linejoin="round"><rect x="8" y="8" width="13" height="13" rx="2"/>'
    b'<path d="M16 8V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h3"/>'
    b'</svg>'
).decode("ascii")
COPIED_ICON_SRC = "data:image/svg+xml;base64," + base64.b64encode(
    b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
    b'stroke="#86efac" stroke-width="2" stroke-linecap="round" '
    b'stroke-linejoin="round"><path d="m20 6-11 11-5-5"/></svg>'
).decode("ascii")

OVERLAY_BUTTON_MESSAGES = {
    "Run clustering": "Running BIRCH clustering...",
    "Assign excluded artifacts": "Assigning excluded artifacts to BIRCH clusters...",
    "Compute UMAP": "Computing UMAP...",
    "Recompute UMAP": "Computing UMAP...",
    "Compute hierarchical clustering": "Computing hierarchical clustering...",
    "Compute hierarchical subclusters": "Computing hierarchical subclusters...",
    "Compute semi-supervised UMAP": "Computing semi-supervised UMAP...",
    "Search": "Searching Euclid object...",
}

OVERLAY_STYLE = """
    align-items: center;
    background: rgba(3, 7, 18, 0.72);
    backdrop-filter: blur(3px);
    display: flex;
    inset: 0;
    justify-content: center;
    pointer-events: all;
    position: fixed;
    z-index: 100000;
"""

OVERLAY_CARD_STYLE = """
    background: #111923;
    border: 1px solid rgba(248, 250, 252, 0.24);
    border-radius: 10px;
    box-shadow: 0 24px 80px rgba(0, 0, 0, 0.45);
    color: #f8fafc;
    font-size: 1rem;
    font-weight: 700;
    line-height: 1.4;
    max-width: min(520px, calc(100vw - 2rem));
    padding: 1.15rem 1.35rem;
    text-align: center;
"""

OVERLAY_SPINNER_CSS = """
    @keyframes euclid-processing-spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    .euclid-processing-spinner {
        animation: euclid-processing-spin 0.9s linear infinite;
        border: 3px solid rgba(248, 250, 252, 0.25);
        border-top-color: #ff5a52;
        border-radius: 999px;
        height: 2rem;
        margin: 0 auto 0.8rem auto;
        width: 2rem;
    }
"""

def inject_plot_cursor_css() -> None:
    st.markdown(
        "<style>" + ALADIN_LINK_CSS + """
        .block-container {
            padding-top: 2rem;
        }
        div[data-testid="stTooltipContent"] {
            max-width: min(42rem, calc(100vw - 2rem));
        }
        @media (min-width: 768px) {
            section[data-testid="stSidebar"][aria-expanded="true"] {
                max-width: 370px !important;
                min-width: 370px !important;
                width: 370px !important;
            }
            section[data-testid="stSidebar"][aria-expanded="true"]
            > div:first-child {
                width: 370px !important;
            }
        }
        .js-plotly-plot .plotly .draglayer .drag,
        .js-plotly-plot .plotly .draglayer .nsewdrag,
        .js-plotly-plot .plotly .cursor-crosshair,
        .js-plotly-plot .plotly .cursor-move,
        .js-plotly-plot .plotly .cursor-pointer {
            cursor: default !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stButton"] button:disabled {
            background-color: #ff5a52 !important;
            border-color: #ff5a52 !important;
            color: #ffffff !important;
            opacity: 1 !important;
        }
        section[data-testid="stSidebar"]
        .st-key-straight_line_artifacts_filter_enabled {
            transform: translateY(0.5rem);
        }
        .lens-status {
            border-radius: 8px;
            font-weight: 700;
            margin: 0.25rem 0 0.85rem 0;
            padding: 0.7rem 0.85rem;
        }
        .lens-status--yes {
            background: #fff1f2;
            border: 1px solid #e11d48;
            color: #9f1239;
        }
        .lens-status--no {
            background: #f8fafc;
            border: 1px solid #94a3b8;
            color: #334155;
        }
        .lens-status__label {
            display: block;
            font-size: 1rem;
            line-height: 1.25;
        }
        .lens-status__meta {
            display: block;
            font-size: 0.82rem;
            font-weight: 500;
            line-height: 1.25;
            margin-top: 0.2rem;
        }
        :is(.st-key-umap_object_id, .st-key-semisupervised_umap_object_id)
        [data-testid="stCode"] {
            background: transparent;
        }
        :is(.st-key-umap_object_id, .st-key-semisupervised_umap_object_id)
        [data-testid="stCode"] pre {
            font-size: 1rem;
            font-weight: 600;
            padding: 0.5rem 2.5rem 0.5rem 0;
        }
        :is(.st-key-umap_object_id, .st-key-semisupervised_umap_object_id)
        [data-testid="stCode"] > div {
            opacity: 1;
            visibility: visible;
        }
        .st-key-search_physical_measurements {
            min-height: 318px;
        }
        .back-to-top {
            align-items: center;
            background: #ff5a52;
            border: 1px solid rgba(255, 255, 255, 0.35);
            border-radius: 999px;
            bottom: 1.4rem;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.28);
            color: white !important;
            display: flex;
            font-size: 1.25rem;
            font-weight: 800;
            height: 2.65rem;
            justify-content: center;
            position: fixed;
            right: 1.4rem;
            text-decoration: none !important;
            width: 2.65rem;
            z-index: 1000;
        }
        .back-to-top:hover {
            background: #e84840;
            color: white !important;
        }
        .concept-help {
            border-bottom: 1px dotted currentColor;
            cursor: help !important;
            display: inline-block;
            position: relative;
        }
        .concept-popover {
            background: #111923;
            border: 1px solid rgba(148, 163, 184, 0.45);
            border-radius: 10px;
            box-sizing: border-box;
            bottom: calc(100% + 0.75rem);
            box-shadow: 0 14px 34px rgba(15, 23, 42, 0.32);
            color: #f8fafc !important;
            display: none;
            font-size: 0.9rem;
            font-weight: 400;
            left: 0;
            line-height: 1.35;
            max-height: calc(100dvh - 1.5rem);
            max-width: min(360px, calc(100vw - 2rem));
            min-width: min(260px, calc(100vw - 2rem));
            overflow-wrap: anywhere;
            overflow-y: auto;
            overscroll-behavior: contain;
            padding: 0.75rem 0.9rem;
            position: absolute;
            text-align: left;
            white-space: normal;
            z-index: 1000;
        }
        .concept-popover strong {
            font-weight: 700;
        }
        .concept-popover__steps {
            margin: 0.5rem 0 0;
            padding-left: 1.35rem;
        }
        .concept-popover__steps li + li {
            margin-top: 0.5rem;
        }
        .concept-popover--with-thumbnails {
            min-width: min(330px, calc(100vw - 2rem));
        }
        .concept-popover__thumbnail-grid {
            display: grid;
            gap: 0.35rem;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            margin-top: 0.65rem;
        }
        .concept-popover__thumbnail-grid img {
            aspect-ratio: 1;
            background: #05080d;
            border: 1px solid rgba(148, 163, 184, 0.38);
            border-radius: 4px;
            display: block;
            object-fit: contain;
            width: 100%;
        }
        .concept-popover--lens-grades {
            min-width: min(330px, calc(100vw - 2rem));
        }
        .lens-grade-help__section + .lens-grade-help__section {
            margin-top: 0.7rem;
        }
        .lens-grade-help__description {
            font-weight: 400;
        }
        .lens-grade-help__thumbnail-grid {
            display: grid;
            gap: 0.35rem;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            margin-top: 0.4rem;
        }
        .lens-grade-help__thumbnail-grid img {
            aspect-ratio: 1;
            background: #05080d;
            border: 1px solid rgba(148, 163, 184, 0.38);
            border-radius: 4px;
            display: block;
            object-fit: contain;
            width: 100%;
        }
        html:not(.escope-tooltips-ready) .concept-help:hover .concept-popover,
        html:not(.escope-tooltips-ready) .concept-help:focus .concept-popover {
            display: block;
        }
        .concept-popover[data-escope-open] {
            bottom: auto;
            display: block;
            inset: auto;
            margin: 0;
            min-width: 0;
            position: fixed;
            z-index: 100000;
        }
        .concept-popover::-webkit-scrollbar {
            width: 8px;
        }
        .concept-popover::-webkit-scrollbar-thumb {
            background: #94a3b8;
            border: 2px solid #111923;
            border-radius: 4px;
        }
        .sidebar-flow-help {
            line-height: 1;
            text-align: right;
        }
        .sidebar-flow-help .concept-help {
            border: 1px solid rgba(148, 163, 184, 0.65);
            border-radius: 999px;
            color: inherit;
            font-size: 0.8rem;
            font-weight: 800;
            height: 1.35rem;
            line-height: 1.25rem;
            text-align: center;
            width: 1.35rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st_components.html(HELP_TOOLTIP_SCRIPT, height=0, width=0)

def render_back_to_top_control() -> None:
    st.markdown(
        """
        <div id="euclid-page-top"></div>
        <a class="back-to-top" href="#euclid-page-top" title="Back to top">↑</a>
        """,
        unsafe_allow_html=True,
    )

def install_click_processing_overlay() -> None:
    button_messages = {
        html.escape(label): html.escape(message)
        for label, message in OVERLAY_BUTTON_MESSAGES.items()
    }
    st_components.html(
        f"""
        <script>
        (() => {{
            const buttonMessages = {button_messages!r};
            const overlayId = "euclid-client-processing-overlay";

            function ensureOverlay(root) {{
                let overlay = root.getElementById(overlayId);
                if (overlay) {{
                    return overlay;
                }}
                overlay = root.createElement("div");
                overlay.id = overlayId;
                overlay.setAttribute("style", `{OVERLAY_STYLE}`);
                overlay.innerHTML = `
                    <div style="{OVERLAY_CARD_STYLE}">
                        <style>{OVERLAY_SPINNER_CSS}</style>
                        <div class="euclid-processing-spinner"></div>
                        <span id="euclid-client-processing-message"></span>
                    </div>
                `;
                root.body.appendChild(overlay);
                return overlay;
            }}

            function showOverlay(message) {{
                const root = window.parent.document;
                const overlay = ensureOverlay(root);
                const messageNode = root.getElementById("euclid-client-processing-message");
                if (messageNode) {{
                    messageNode.textContent = message;
                }}
                overlay.style.display = "flex";
            }}

            if (!window.parent.__euclidProcessingOverlayInstalled) {{
                window.parent.__euclidProcessingOverlayInstalled = true;
                window.parent.document.addEventListener("click", (event) => {{
                    const button = event.target.closest("button");
                    if (!button || button.disabled) {{
                        return;
                    }}
                    const label = (button.innerText || button.textContent || "").trim();
                    const message = buttonMessages[label];
                    if (message) {{
                        showOverlay(message);
                    }}
                }}, true);
            }}
        }})();
        </script>
        """,
        height=0,
        width=0,
    )

class ProcessingOverlay:
    def __init__(self) -> None:
        self.is_open = False

    def open(self, message: str) -> None:
        escaped_message = html.escape(message)
        nonce = time.time_ns()
        st_components.html(
            f"""
            <script>
            (() => {{
                const nonce = "{nonce}";
                const root = window.parent.document;
                const overlayId = "euclid-client-processing-overlay";
                let overlay = root.getElementById(overlayId);
                if (!overlay) {{
                    overlay = root.createElement("div");
                    overlay.id = overlayId;
                    overlay.setAttribute("style", `{OVERLAY_STYLE}`);
                    overlay.innerHTML = `
                        <div style="{OVERLAY_CARD_STYLE}">
                            <style>{OVERLAY_SPINNER_CSS}</style>
                            <div class="euclid-processing-spinner"></div>
                            <span id="euclid-client-processing-message"></span>
                        </div>
                    `;
                    root.body.appendChild(overlay);
                }}
                const messageNode = root.getElementById("euclid-client-processing-message");
                if (messageNode) {{
                    messageNode.textContent = "{escaped_message}";
                }}
                overlay.style.display = "flex";
            }})();
            </script>
            """,
            height=0,
            width=0,
        )
        self.is_open = True

    def close(self) -> None:
        nonce = time.time_ns()
        st_components.html(
            f"""
            <script>
            (() => {{
                const nonce = "{nonce}";
                const overlay = window.parent.document.getElementById(
                    "euclid-client-processing-overlay"
                );
                if (overlay) {{
                    overlay.style.display = "none";
                }}
            }})();
            </script>
            """,
            height=0,
            width=0,
        )
        self.is_open = False

def close_processing_overlay() -> None:
    nonce = time.time_ns()
    st_components.html(
        f"""
        <script>
        (() => {{
            const nonce = "{nonce}";
            const overlay = window.parent.document.getElementById(
                "euclid-client-processing-overlay"
            );
            if (overlay) {{
                overlay.style.display = "none";
            }}
        }})();
        </script>
        """,
        height=0,
        width=0,
    )

def add_pca_filter() -> None:
    st.session_state["pca_filter_count"] = (
        st.session_state.get("pca_filter_count", 0) + 1
    )

def render_pca_filter_controls(pca_columns: list[str]) -> list[dict]:
    with st.expander("PCA value filters", expanded=False):
        st.caption("Filters are combined with AND before UMAP is computed.")
        if "pca_filter_count" not in st.session_state:
            st.session_state["pca_filter_count"] = 0

        button_col, clear_col = st.columns(2)
        with button_col:
            st.button("Add filter", on_click=add_pca_filter)
        with clear_col:
            if st.session_state.get("pca_filter_count", 0):
                if st.button("Clear filters", key="clear_pca_filters"):
                    st.session_state["pca_filter_count"] = 0
                    st.rerun()

        filter_count = st.number_input(
            "Number of filters",
            min_value=0,
            max_value=12,
            step=1,
            key="pca_filter_count",
        )

        raw_filters = []
        for index in range(int(filter_count)):
            st.markdown(f"**Filter {index + 1}**")
            enabled = st.checkbox(
                "Enabled",
                value=True,
                key=f"pca_filter_{index}_enabled",
            )
            feature = st.selectbox(
                "Component",
                pca_columns,
                index=min(index, len(pca_columns) - 1),
                key=f"pca_filter_{index}_feature",
            )
            operator = st.selectbox(
                "Operator",
                PCA_FILTER_OPERATORS,
                key=f"pca_filter_{index}_operator",
            )

            if operator == "between":
                lower_col, upper_col = st.columns(2)
                with lower_col:
                    lower = st.number_input(
                        "Lower",
                        value=0.0,
                        step=0.1,
                        format="%.6f",
                        key=f"pca_filter_{index}_lower",
                    )
                with upper_col:
                    upper = st.number_input(
                        "Upper",
                        value=1.0,
                        step=0.1,
                        format="%.6f",
                        key=f"pca_filter_{index}_upper",
                    )
                raw_filters.append(
                    {
                        "feature": feature,
                        "operator": operator,
                        "lower": lower,
                        "upper": upper,
                        "enabled": enabled,
                    }
                )
            else:
                value = st.number_input(
                    "Value",
                    value=0.0,
                    step=0.1,
                    format="%.6f",
                    key=f"pca_filter_{index}_value",
                )
                raw_filters.append(
                    {
                        "feature": feature,
                        "operator": operator,
                        "value": value,
                        "enabled": enabled,
                    }
                )

    return raw_filters


def render_physical_filter_controls(
    physical_df: pd.DataFrame,
    selected_cluster: int,
) -> list[dict[str, object]]:
    with st.expander("Physical property filters", expanded=False):
        st.caption(
            "Filters are combined with AND and applied to the selected cluster before "
            "UMAP and subsequent analyses. Changing them does not start a calculation. "
            "Only usable measurements are included; photometric redshift, stellar "
            "mass and Sersic index also require a zero quality flag."
        )
        clean = analysis_ready_physical_measurements(physical_df)
        available_fields = available_physical_analysis_fields(clean)
        if not available_fields:
            st.info("No filterable physical measurements are available for this cluster.")
            return []

        selected_fields = st.multiselect(
            "Properties",
            available_fields,
            format_func=lambda field: PHYSICAL_ANALYSIS_FIELDS.get(field, field),
            key=f"physical_filter_fields_{selected_cluster}",
        )
        raw_filters: list[dict[str, object]] = []
        for field in selected_fields:
            values = pd.to_numeric(clean[field], errors="coerce").dropna()
            if values.empty:
                continue
            minimum = float(values.min())
            maximum = float(values.max())
            st.markdown(f"**{PHYSICAL_ANALYSIS_FIELDS.get(field, field)}**")
            operator = st.selectbox(
                "Operator",
                PHYSICAL_FILTER_OPERATORS,
                key=f"physical_filter_{selected_cluster}_{field}_operator",
                label_visibility="collapsed",
            )
            if operator == "between":
                lower_col, upper_col = st.columns(2)
                lower_key = f"physical_filter_{selected_cluster}_{field}_lower"
                upper_key = f"physical_filter_{selected_cluster}_{field}_upper"
                st.session_state.setdefault(lower_key, minimum)
                st.session_state.setdefault(upper_key, maximum)
                with lower_col:
                    lower = st.number_input(
                        "Minimum",
                        format="%.6f",
                        key=lower_key,
                    )
                with upper_col:
                    upper = st.number_input(
                        "Maximum",
                        format="%.6f",
                        key=upper_key,
                    )
                raw_filters.append(
                    {
                        "field": field,
                        "operator": operator,
                        "lower": lower,
                        "upper": upper,
                        "enabled": True,
                    }
                )
            else:
                default_value = float(values.median())
                value_key = f"physical_filter_{selected_cluster}_{field}_value"
                st.session_state.setdefault(value_key, default_value)
                value = st.number_input(
                    "Value",
                    format="%.6f",
                    key=value_key,
                )
                raw_filters.append(
                    {
                        "field": field,
                        "operator": operator,
                        "value": value,
                        "enabled": True,
                    }
                )

        return raw_filters


def queue_recommended_physical_filter(
    recommendation: dict[str, object],
    cluster_id: int,
) -> None:
    st.session_state["pending_recommended_physical_filter"] = {
        "cluster_id": int(cluster_id),
        "field": str(recommendation["feature"]),
        "operator": str(recommendation["operator"]),
        "value": float(recommendation["value"]),
    }


def apply_pending_recommended_physical_filter(cluster_id: int) -> None:
    recommendation = st.session_state.get("pending_recommended_physical_filter")
    if recommendation is None or int(recommendation["cluster_id"]) != int(cluster_id):
        return
    st.session_state.pop("pending_recommended_physical_filter", None)

    field = str(recommendation["field"])
    fields_key = f"physical_filter_fields_{cluster_id}"
    selected_fields = list(st.session_state.get(fields_key, []))
    if field not in selected_fields:
        selected_fields.append(field)
    st.session_state[fields_key] = selected_fields
    st.session_state[
        f"physical_filter_{cluster_id}_{field}_operator"
    ] = recommendation["operator"]
    st.session_state[f"physical_filter_{cluster_id}_{field}_value"] = float(
        recommendation["value"]
    )


def show_thumbnail(
    row: pd.Series | None,
    caption: str,
    image_src: str | None = None,
    aladin_link: str | None = None,
) -> None:
    if row is None:
        st.caption(caption)
        st.caption("No object")
        return

    if image_src is None:
        path = object_image_path(row)
        if path is None:
            st.caption(caption)
            st.caption("No image")
            return

        try:
            image_src = thumbnail_image_src(path)
        except Exception:
            st.caption(caption)
            st.caption("No image")
            return

    object_id = row.get("object_id", "")
    id_str = row.get("id_str", "")
    modal_key = hashlib.sha1(f"{caption}|{object_id}|{id_str}".encode()).hexdigest()[:12]
    modal_id = f"thumb-{modal_key}"
    escaped_caption = html.escape(str(caption))
    modal_caption = str(caption)
    is_lens = row.get("is_lens", False)
    lens_grade = row.get("lens_grade", "")
    if (
        caption in ("Canonical", "Anomalous")
        and pd.notna(is_lens) and bool(is_lens)
        and pd.notna(lens_grade)
    ):
        grade = str(lens_grade).strip().upper()
        if grade in {"A", "B", "C"}:
            modal_caption = f"{caption} - Grade {grade}"
    escaped_modal_caption = html.escape(modal_caption)
    escaped_object_id = html.escape(str(object_id))
    escaped_id_str = html.escape(str(id_str))
    id_str_html = (
        f"<div class='thumb-modal-meta'>id_str: {escaped_id_str}</div>"
        if not pd.isna(id_str) and str(id_str).strip()
        else ""
    )
    aladin_link_html = aladin_icon_link(aladin_link, css_class="thumb-aladin-link") if aladin_link else ""

    st.html(
        f"""
        <style>
        {ALADIN_LINK_CSS}
        #{modal_id} {{
            display: none;
        }}
        #{modal_id}:target {{
            align-items: center;
            background: rgba(0, 0, 0, 0.82);
            display: flex;
            inset: 0;
            justify-content: center;
            padding: 2rem;
            position: fixed;
            z-index: 10000;
        }}
        #{modal_id} .thumb-modal-panel {{
            background: #0f1117;
            border: 1px solid rgba(255, 255, 255, 0.18);
            border-radius: 8px;
            padding: 1rem;
            position: relative;
            z-index: 1;
        }}
        #{modal_id} .thumb-modal-meta {{
            color: #f8fafc;
            font-size: 0.9rem;
            margin-bottom: 0.45rem;
        }}
        #{modal_id} .thumb-modal-object-id {{
            align-items: center;
            display: flex;
            gap: 0.4rem;
        }}
        #{modal_id} .thumb-copy-id {{
            align-items: center;
            background: transparent;
            border: 0;
            border-radius: 4px;
            color: #cbd5e1;
            cursor: pointer;
            display: inline-flex;
            flex: none;
            height: 24px;
            justify-content: center;
            padding: 0;
            width: 24px;
        }}
        #{modal_id} .thumb-copy-id:hover {{
            background: rgba(255, 255, 255, 0.12);
            color: #fff;
        }}
        #{modal_id} .thumb-copy-id:focus-visible {{
            outline: 2px solid #93c5fd;
            outline-offset: 2px;
        }}
        #{modal_id} .thumb-copy-id img {{
            height: 16px;
            width: 16px;
        }}
        #{modal_id} .thumb-copy-id .thumb-copied-icon,
        #{modal_id} .thumb-copy-id.copied .thumb-copy-icon {{
            display: none;
        }}
        #{modal_id} .thumb-copy-id.copied .thumb-copied-icon {{
            display: block;
        }}
        #{modal_id} .thumb-aladin-link {{
            margin-left: 0.2rem;
        }}
        #{modal_id} .thumb-modal-image {{
            display: block;
            height: 400px;
            object-fit: contain;
            width: 400px;
        }}
        .thumb-link img {{
            cursor: pointer;
            display: block;
            height: {SUMMARY_THUMBNAIL_WIDTH}px;
            object-fit: cover;
            width: {SUMMARY_THUMBNAIL_WIDTH}px;
        }}
        .thumb-caption {{
            color: inherit;
            font-size: 0.82rem;
            margin-top: 0.2rem;
            opacity: 0.72;
            text-align: center;
            width: {SUMMARY_THUMBNAIL_WIDTH}px;
        }}
        </style>
        <a class="thumb-link" href="#{modal_id}" title="Open enlarged image">
            <img src="{image_src}" width="{SUMMARY_THUMBNAIL_WIDTH}" height="{SUMMARY_THUMBNAIL_WIDTH}" />
        </a>
        <div class="thumb-caption">{escaped_caption}</div>
        <div id="{modal_id}">
            <a href="#" style="position: fixed; inset: 0;" aria-label="Close"></a>
            <div class="thumb-modal-panel">
                <div class="thumb-modal-meta"><strong>{escaped_modal_caption}</strong></div>
                <div class="thumb-modal-meta thumb-modal-object-id">
                    <span>object_id: {escaped_object_id}</span>
                    <button type="button" class="thumb-copy-id"
                            data-object-id="{escaped_object_id}"
                            title="Copy object ID" aria-label="Copy object ID">
                        <img class="thumb-copy-icon" src="{COPY_ICON_SRC}" alt="" />
                        <img class="thumb-copied-icon" src="{COPIED_ICON_SRC}" alt="" />
                    </button>
                    {aladin_link_html}
                </div>
                {id_str_html}
                <img class="thumb-modal-image" src="{image_src}" alt="{escaped_caption}" />
            </div>
        </div>
        <script>
        if (!window.__escopeCopyObjectIdBound) {{
            document.addEventListener("click", async (event) => {{
                const button = event.target.closest(".thumb-copy-id");
                if (!button) return;
                try {{
                    await navigator.clipboard.writeText(button.dataset.objectId);
                    button.classList.add("copied");
                    button.title = "Copied";
                    button.setAttribute("aria-label", "Object ID copied");
                    setTimeout(() => {{
                        button.classList.remove("copied");
                        button.title = "Copy object ID";
                        button.setAttribute("aria-label", "Copy object ID");
                    }}, 1800);
                }} catch (error) {{
                    button.title = "Copy failed";
                    button.setAttribute("aria-label", "Could not copy object ID");
                }}
            }});
            window.__escopeCopyObjectIdBound = true;
        }}
        </script>
        """,
        unsafe_allow_javascript=True,
    )

def show_thumbnail_group(
    title: str,
    rows: list[pd.Series],
    captions: list[str],
    skip_missing_images: bool = False,
    aladin_links: Mapping[str, str] | None = None,
) -> None:
    aladin_links = aladin_links or {}
    if skip_missing_images:
        available = []
        for row, caption in zip(rows, captions):
            try:
                path = object_image_path(row)
                if path is None:
                    continue
                available.append((row, caption, thumbnail_image_src(path)))
            except Exception:
                continue
        if not available:
            return
        render_thumbnail_group_title(title)
        for column, (row, caption, image_src) in zip(st.columns(len(available)), available):
            with column:
                show_thumbnail(
                    row,
                    caption,
                    image_src=image_src,
                    aladin_link=aladin_links.get(str(row.get("object_id"))),
                )
        return

    render_thumbnail_group_title(title)
    count = max(len(rows), 1)
    for column, row, caption in zip(st.columns(count), rows or [None], captions or [""]):
        with column:
            aladin_link = (
                aladin_links.get(str(row.get("object_id")))
                if row is not None else None
            )
            show_thumbnail(
                row,
                caption,
                aladin_link=aladin_link,
            )

def cluster_extreme_rows(
    cluster_df: pd.DataFrame,
    summary_features: list[str],
) -> tuple[pd.Series | None, pd.Series | None]:
    marked = add_cluster_extreme_roles(cluster_df, summary_features)
    canonical_rows = marked[marked["is_canonical"]]
    anomaly_rows = marked[marked["is_anomaly"]]
    canonical_row = canonical_rows.iloc[0] if not canonical_rows.empty else None
    anomaly_row = anomaly_rows.iloc[0] if not anomaly_rows.empty else None
    return canonical_row, anomaly_row

def cluster_visual_rows(
    cluster_df: pd.DataFrame,
    summary_features: list[str],
    cluster_id: int,
) -> tuple[pd.Series | None, pd.Series | None, list[pd.Series], list[pd.Series]]:
    canonical_row, anomaly_row = cluster_extreme_rows(cluster_df, summary_features)

    used_object_ids = {
        str(row.get("object_id"))
        for row in (canonical_row, anomaly_row)
        if row is not None and not pd.isna(row.get("object_id"))
    }
    unknown_pool = cluster_df[~cluster_df["is_lens"].fillna(False).astype(bool)]
    random_pool = unknown_pool[
        ~unknown_pool["object_id"].astype("string").isin(used_object_ids)
    ]
    if random_pool.empty:
        random_pool = unknown_pool
    random_rows = random_pool.sample(
        n=min(SUMMARY_RANDOM_OBJECTS, len(random_pool)),
        random_state=int(cluster_id) + 17,
    )

    lens_rows = cluster_df[cluster_df["is_lens"]].copy()
    if not lens_rows.empty:
        if "lens_grade" not in lens_rows.columns:
            lens_rows["lens_grade"] = ""
        lens_rows["_grade_order"] = lens_grade_sort_key(lens_rows["lens_grade"])
        lens_rows = lens_rows.sort_values(
            ["_grade_order", "lens_grade", "object_id"],
            na_position="last",
        ).drop(columns=["_grade_order"])
    lens_rows = lens_rows.head(SUMMARY_LENS_OBJECTS)

    return (
        canonical_row,
        anomaly_row,
        [row for _, row in random_rows.iterrows()],
        [row for _, row in lens_rows.iterrows()],
    )

def sample_distplot_values(values: pd.Series, random_state: int) -> list[float]:
    clean_values = values.dropna()
    if len(clean_values) > SUMMARY_DISTPLOT_MAX_POINTS_PER_GROUP:
        clean_values = clean_values.sample(
            n=SUMMARY_DISTPLOT_MAX_POINTS_PER_GROUP,
            random_state=random_state,
        )
    return clean_values.astype(float).tolist()

def feature_bin_size(values: list[float]) -> float:
    if len(values) < 2:
        return 1.0
    value_range = max(values) - min(values)
    if value_range <= 0:
        return 1.0
    return value_range / SUMMARY_HISTOGRAM_BINS

def can_show_kde(values: list[float]) -> bool:
    return len(values) > 1 and len(set(values)) > 1

def format_decimal_comma(value: float, decimals: int = 2) -> str:
    return f"{value:.{decimals}f}".replace(".", ",")

def format_thousands_dot(value: int | float) -> str:
    return f"{int(value):,}".replace(",", ".")

def recommended_pca_filter(cluster_df: pd.DataFrame, feature: str) -> dict | None:
    return recommended_numeric_filter(
        cluster_df,
        feature,
        min_valid_objects=2,
        min_selected_objects=PCA_FILTER_RECOMMENDATION_MIN_OBJECTS,
        min_lenses=PCA_FILTER_RECOMMENDATION_MIN_LENSES,
        min_unknowns=1,
        min_recall=PCA_FILTER_RECOMMENDATION_MIN_RECALL,
    )

def queue_recommended_pca_filter(recommendation: dict) -> None:
    st.session_state["pending_recommended_pca_filter"] = {
        "feature": recommendation["feature"],
        "operator": recommendation["operator"],
        "value": float(recommendation["value"]),
    }

def apply_pending_recommended_pca_filter() -> None:
    recommendation = st.session_state.pop("pending_recommended_pca_filter", None)
    if recommendation is None:
        return

    target_feature = recommendation["feature"]
    filter_count = int(st.session_state.get("pca_filter_count", 0))
    target_index = filter_count
    for index in range(filter_count):
        if st.session_state.get(f"pca_filter_{index}_feature") == target_feature:
            target_index = index
            break

    if target_index == filter_count:
        st.session_state["pca_filter_count"] = filter_count + 1

    st.session_state[f"pca_filter_{target_index}_enabled"] = True
    st.session_state[f"pca_filter_{target_index}_feature"] = target_feature
    st.session_state[f"pca_filter_{target_index}_operator"] = recommendation["operator"]
    st.session_state[f"pca_filter_{target_index}_value"] = float(recommendation["value"])

def build_cluster_distplot_figure(
    cluster_df: pd.DataFrame,
    feature: str,
    feature_index: int,
    recommendation: dict | None = None,
) -> object | None:
    import plotly.figure_factory as ff

    lens_df = cluster_df[cluster_df["is_lens"]]
    non_lens_df = cluster_df[~cluster_df["is_lens"]]
    non_lens_values = sample_distplot_values(
        non_lens_df[feature],
        random_state=feature_index + 101,
    )
    lens_values = sample_distplot_values(
        lens_df[feature],
        random_state=feature_index + 701,
    )

    hist_data = []
    group_labels = []
    colors = []
    if non_lens_values:
        hist_data.append(non_lens_values)
        group_labels.append("Unknown")
        colors.append("#4c78a8")
    if lens_values:
        hist_data.append(lens_values)
        group_labels.append("Lens candidate")
        colors.append("#d62728")
    if not hist_data:
        return None

    all_values = [value for values in hist_data for value in values]
    show_curve = all(can_show_kde(values) for values in hist_data)
    fig = ff.create_distplot(
        hist_data,
        group_labels,
        bin_size=feature_bin_size(all_values),
        colors=colors,
        curve_type="kde",
        show_curve=show_curve,
        show_hist=True,
        show_rug=True,
        histnorm="probability density",
    )
    fig.update_layout(
        title={"text": feature, "x": 0.5, "xanchor": "center"},
        height=300,
        margin={"l": 28, "r": 12, "t": 46, "b": 58},
        legend={"orientation": "h", "yanchor": "top", "y": -0.18, "x": 0},
        barmode="overlay",
        yaxis={"title": "density", "showgrid": True, "zeroline": False},
        yaxis2={"showgrid": True, "showticklabels": False, "zeroline": False},
    )
    fig.update_traces(opacity=0.72, selector={"type": "histogram"})
    fig.update_traces(line={"width": 2.0}, selector={"mode": "lines"})
    if recommendation is not None:
        threshold = float(recommendation["value"])
        fig.add_vline(
            x=threshold,
            line_dash="dash",
            line_width=2,
            line_color="#ffcc00",
            annotation_text=f"{recommendation['operator']} {threshold:.3g}",
            annotation_position="top right",
            annotation_font_size=10,
            annotation_font_color="#ffcc00",
        )
    fig.update_xaxes(showgrid=False, zeroline=False)
    return fig

def render_selected_cluster_pca_characterization(
    cluster_id: int,
    cluster_df: pd.DataFrame,
    summary_features: list[str],
) -> None:
    st.subheader("PCA characterization")
    n_lenses = int(cluster_df["is_lens"].sum())
    n_non_lenses = len(cluster_df) - n_lenses
    if n_lenses == 0:
        st.info(
            "This cluster has no labelled lens candidates, so lens-versus-unknown "
            "PCA distributions and recommended thresholds cannot be calculated."
        )
        return

    if n_non_lenses == 0:
        st.info("This cluster does not contain Unknown objects for comparison.")
        return

    recommendations = {
        feature: recommended_pca_filter(cluster_df, feature)
        for feature in summary_features
    }
    valid_recommendations = sorted(
        [
            recommendation
            for recommendation in recommendations.values()
            if recommendation is not None
        ],
        key=lambda recommendation: recommendation["score"],
        reverse=True,
    )
    chart_columns = st.columns(2)
    for index, feature in enumerate(summary_features):
        fig = build_cluster_distplot_figure(
            cluster_df,
            feature,
            index,
            recommendations.get(feature),
        )
        if fig is None:
            continue
        with chart_columns[index % 2]:
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displaylogo": False, "responsive": True},
                key=f"cluster_distplot_chart_{cluster_id}_{feature}",
            )

    if valid_recommendations:
        st.markdown(
            f"""
            <div style="font-weight: 700; margin-bottom: 0.35rem;">
                Recommended PCA thresholds
                <span class="concept-help" tabindex="0" style="margin-left: 0.25rem;">?
                    <span class="concept-popover">
                        {html.escape(RECOMMENDED_FILTER_HELP)}
                    </span>
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        header_cols = st.columns([1.05, 1.8, 0.9, 0.8, 1.0, 1.0, 0.9])
        headers = [
            "filter",
            "threshold",
            "objects",
            "lenses",
            "lens_rate_%",
            "enrichment_x",
            "recall_%",
        ]
        for header_col, header in zip(header_cols, headers):
            align = (
                "right"
                if header in {"lens_rate_%", "enrichment_x", "recall_%"}
                else "left"
            )
            header_col.markdown(
                f"""
                <div style="
                    border-bottom: 1px solid rgba(128, 128, 128, 0.35);
                    font-size: 0.82rem;
                    font-weight: 700;
                    padding: 0.25rem 0;
                    text-align: {align};
                ">{html.escape(header)}</div>
                """,
                unsafe_allow_html=True,
            )

        for recommendation_index, recommendation in enumerate(valid_recommendations):
            row_cols = st.columns([1.05, 1.8, 0.9, 0.8, 1.0, 1.0, 0.9])
            button_label = (
                "Apply filter"
            )
            with row_cols[0]:
                if st.button(
                    button_label,
                    key=(
                        f"apply_pca_filter_{cluster_id}_"
                        f"{recommendation['feature']}_{recommendation_index}"
                    ),
                    type="secondary",
                ):
                    queue_recommended_pca_filter(recommendation)
                    st.rerun()
            threshold_text = (
                f"{recommendation['feature']} "
                f"{recommendation['operator']} "
                f"{format_decimal_comma(recommendation['value'], 4)}"
            )
            row_values = [
                threshold_text,
                format_thousands_dot(recommendation["n_selected"]),
                format_thousands_dot(recommendation["n_lenses"]),
                format_decimal_comma(recommendation["lens_rate"] * 100, 2),
                format_decimal_comma(recommendation["enrichment"], 2),
                format_decimal_comma(recommendation["recall"] * 100, 2),
            ]
            for value_col, value, align in zip(
                row_cols[1:],
                row_values,
                ["left", "left", "left", "right", "right", "right"],
            ):
                value_col.markdown(
                    f"""
                    <div style="
                        border-bottom: 1px solid rgba(128, 128, 128, 0.16);
                        font-size: 0.88rem;
                        min-height: 2.25rem;
                        padding: 0.45rem 0;
                        text-align: {align};
                    ">{html.escape(str(value))}</div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.caption("No stable PCA threshold recommendation was found for this cluster.")


def build_compact_physical_histogram_figure(
    physical_df: pd.DataFrame,
    fields: list[str],
) -> object:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    clean = analysis_ready_physical_measurements(physical_df)
    columns = 4
    rows = max(1, (len(fields) + columns - 1) // columns)
    figure = make_subplots(
        rows=rows,
        cols=columns,
        subplot_titles=[PHYSICAL_ANALYSIS_FIELDS[field] for field in fields],
        horizontal_spacing=0.055,
        vertical_spacing=0.22,
    )
    for index, field in enumerate(fields):
        values = pd.to_numeric(clean[field], errors="coerce").dropna()
        figure.add_trace(
            go.Histogram(
                x=values,
                nbinsx=24,
                marker_color="#5b8db8",
                opacity=0.82,
                showlegend=False,
                hovertemplate="Value: %{x}<br>Objects: %{y}<extra></extra>",
            ),
            row=index // columns + 1,
            col=index % columns + 1,
        )
    figure.update_annotations(font_size=11)
    figure.update_xaxes(tickfont={"size": 9}, showgrid=False, zeroline=False)
    figure.update_yaxes(tickfont={"size": 9}, title=None, zeroline=False)
    figure.update_layout(
        height=max(220, rows * 220),
        margin={"l": 20, "r": 10, "t": 38, "b": 25},
        bargap=0.05,
    )
    return figure


def _cluster_physical_analysis_data(
    cluster_df: pd.DataFrame,
    physical_df: pd.DataFrame,
) -> pd.DataFrame:
    memberships = cluster_df[["object_id", "is_lens"]].copy()
    memberships["object_id"] = normalize_object_ids(memberships["object_id"])
    memberships = memberships.drop_duplicates("object_id")
    clean = analysis_ready_physical_measurements(physical_df).drop_duplicates(
        "object_id"
    )
    return memberships.merge(clean, on="object_id", how="left")


def _physical_filter_unavailable_reason(
    analysis_df: pd.DataFrame,
    field: str,
) -> str:
    values = analysis_df[[field, "is_lens"]].copy()
    values[field] = pd.to_numeric(values[field], errors="coerce")
    values = values.dropna(subset=[field, "is_lens"])
    if len(values) < PHYSICAL_FILTER_RECOMMENDATION_MIN_VALID_OBJECTS:
        return (
            "At least "
            f"{PHYSICAL_FILTER_RECOMMENDATION_MIN_VALID_OBJECTS} valid measurements "
            "are required."
        )
    n_lenses = int(values["is_lens"].astype(bool).sum())
    n_unknowns = len(values) - n_lenses
    if n_lenses < PHYSICAL_FILTER_RECOMMENDATION_MIN_LENSES:
        return (
            "At least "
            f"{PHYSICAL_FILTER_RECOMMENDATION_MIN_LENSES} labelled lens candidates "
            "with valid measurements are required."
        )
    if n_unknowns < PHYSICAL_FILTER_RECOMMENDATION_MIN_UNKNOWNS:
        return (
            "At least "
            f"{PHYSICAL_FILTER_RECOMMENDATION_MIN_UNKNOWNS} unknown objects with "
            "valid measurements are required."
        )
    return "No one-sided cut meets the minimum selected-size and candidate-recall requirements."


def render_selected_cluster_physical_characterization(
    cluster_id: int,
    cluster_df: pd.DataFrame,
    physical_df: pd.DataFrame,
) -> None:
    st.subheader("Physical characterization")
    summary = build_physical_summary(physical_df, total_objects=len(cluster_df))
    if summary.empty:
        st.info("No usable physical measurements were found for this cluster.")
        return

    st.caption(
        "Only usable values from the Euclid Q1 physical-measurements table are "
        "included. Photometric redshift, stellar mass and Sersic index also "
        "require a zero quality flag when their flag column is available."
    )
    st.plotly_chart(
        build_compact_physical_histogram_figure(
            physical_df,
            summary["field"].tolist(),
        ),
        use_container_width=True,
        config={"displaylogo": False, "responsive": True},
        key=f"selected_cluster_physical_histograms_{cluster_id}",
    )

    analysis_df = _cluster_physical_analysis_data(cluster_df, physical_df)
    recommendations = {
        field: recommended_numeric_filter(
            analysis_df,
            field,
            min_valid_objects=PHYSICAL_FILTER_RECOMMENDATION_MIN_VALID_OBJECTS,
            min_selected_objects=PCA_FILTER_RECOMMENDATION_MIN_OBJECTS,
            min_lenses=PHYSICAL_FILTER_RECOMMENDATION_MIN_LENSES,
            min_unknowns=PHYSICAL_FILTER_RECOMMENDATION_MIN_UNKNOWNS,
            min_recall=PCA_FILTER_RECOMMENDATION_MIN_RECALL,
        )
        for field in summary["field"]
    }

    st.markdown(
        "**Physical measurements and recommended thresholds**",
        help=RECOMMENDED_FILTER_HELP,
    )
    st.caption(
        "For each property, ESCOPE compares candidates of the selected reference grades with "
        "unlabelled objects in this cluster. It tests upper and lower cuts and "
        "recommends the one that best balances candidate enrichment and recall "
        "among objects with usable measurements. These cuts guide inspection; "
        "they do not confirm lenses."
    )
    column_widths = [1.8, 1.7, 2.2, 1.25, 1.35, 0.95, 0.85]
    header_cols = st.columns(column_widths)
    headers = [
        "filter",
        "measurement",
        "median / IQR",
        "valid / coverage",
        "threshold",
        "enrichment_x",
        "recall_%",
    ]
    for header_col, header in zip(header_cols, headers):
        align = "right" if header in {"enrichment_x", "recall_%"} else "left"
        header_col.markdown(
            f"""
            <div style="
                border-bottom: 1px solid rgba(128, 128, 128, 0.35);
                font-size: 0.78rem;
                font-weight: 700;
                padding: 0.25rem 0;
                text-align: {align};
            ">{html.escape(header)}</div>
            """,
            unsafe_allow_html=True,
        )

    for row_index, row in summary.reset_index(drop=True).iterrows():
        field = str(row["field"])
        recommendation = recommendations[field]
        row_cols = st.columns(column_widths)
        with row_cols[0]:
            if st.button(
                "Apply filter",
                key=f"apply_physical_filter_{cluster_id}_{field}_{row_index}",
                disabled=recommendation is None,
                type="secondary",
                help=(
                    None
                    if recommendation is not None
                    else _physical_filter_unavailable_reason(analysis_df, field)
                ),
            ):
                queue_recommended_physical_filter(recommendation, cluster_id)
                st.rerun()

        threshold = "-"
        enrichment = "-"
        recall = "-"
        if recommendation is not None:
            threshold = (
                f"{recommendation['operator']} "
                f"{format_decimal_comma(float(recommendation['value']), 4)}"
            )
            enrichment = format_decimal_comma(
                float(recommendation["enrichment"]),
                2,
            )
            recall = format_decimal_comma(float(recommendation["recall"]) * 100, 2)

        row_values = [
            row["measurement"],
            (
                f"{format_decimal_comma(float(row['median']), 3)} / "
                f"[{format_decimal_comma(float(row['q25']), 3)}, "
                f"{format_decimal_comma(float(row['q75']), 3)}]"
            ),
            (
                f"{format_thousands_dot(int(row['valid_objects']))} / "
                f"{format_decimal_comma(float(row['coverage_%']), 1)}%"
            ),
            threshold,
            enrichment,
            recall,
        ]
        alignments = ["left", "left", "left", "left", "right", "right"]
        for value_index, (value_col, value, align) in enumerate(
            zip(row_cols[1:], row_values, alignments)
        ):
            value_html = html.escape(str(value))
            if value_index == 0:
                value_html = (
                    '<span class="concept-help" tabindex="0">'
                    f'{value_html}<span class="concept-popover">'
                    f'{html.escape(PHYSICAL_MEASUREMENT_HELP[field])}</span></span>'
                )
            value_col.markdown(
                f"""
                <div style="
                    border-bottom: 1px solid rgba(128, 128, 128, 0.16);
                    font-size: 0.8rem;
                    min-height: 2.5rem;
                    overflow-wrap: anywhere;
                    padding: 0.48rem 0;
                    text-align: {align};
                ">{value_html}</div>
                """,
                unsafe_allow_html=True,
            )


def render_thumbnail_group_title(title: str) -> None:
    if title != "Canonical / anomalous":
        st.caption(title)
        return

    st.markdown(
        f"""
        <div style="color: inherit; font-size: 0.82rem; margin-bottom: 0.35rem;">
            <span class="concept-help" tabindex="0">Canonical
                <span class="concept-popover">
                    {html.escape(CLUSTER_EXTREME_HELP['Canonical'])}
                </span>
            </span>
            <span style="margin: 0 0.2rem;">/</span>
            <span class="concept-help" tabindex="0">Anomalous
                <span class="concept-popover">
                    {html.escape(CLUSTER_EXTREME_HELP['Anomalous'])}
                </span>
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

def _tooltip_thumbnail_grid(
    thumbnail_paths: Sequence[str | Path],
    *,
    css_class: str = "concept-popover__thumbnail-grid",
    alt_text: str = "Tooltip example",
) -> str:
    thumbnails = []
    for thumbnail_path in thumbnail_paths:
        path = Path(thumbnail_path)
        if not path.is_file():
            continue
        encoded_image = base64.b64encode(path.read_bytes()).decode("ascii")
        thumbnails.append(
            '<img src="data:image/jpeg;base64,'
            f'{encoded_image}" alt="{html.escape(alt_text, quote=True)}">'
        )
    if not thumbnails:
        return ""
    return (
        f'<div class="{html.escape(css_class, quote=True)}">'
        + "".join(thumbnails)
        + "</div>"
    )


def _render_help_label_html(
    label: str,
    popover_html: str,
    popover_classes: Sequence[str] = (),
) -> None:
    escaped_label = html.escape(label)
    class_names = " ".join(("concept-popover", *popover_classes))
    st.markdown(
        f"""
        <div style="font-size: 0.88rem; font-weight: 600; margin-bottom: 0.2rem;">
            <span class="concept-help" tabindex="0">
                {escaped_label}
                <span class="{class_names}">{popover_html}</span>
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_help_label(
    label: str,
    help_text: str,
    thumbnail_paths: Sequence[str | Path] = (),
) -> None:
    escaped_help = html.escape(help_text).replace("\n", "<br>")
    thumbnail_grid = _tooltip_thumbnail_grid(
        thumbnail_paths,
        alt_text="Excluded straight-line artifact example",
    )
    popover_classes = ("concept-popover--with-thumbnails",) if thumbnail_grid else ()
    _render_help_label_html(
        label,
        escaped_help + thumbnail_grid,
        popover_classes,
    )


def render_lens_grade_help_label(
    label: str,
    grade_descriptions: Mapping[str, str],
    grade_example_paths: Mapping[str, Sequence[str | Path]],
) -> None:
    sections = [
        '<div class="lens-grade-help__description">'
        f'{html.escape(LENS_GRADE_HELP_NOTE)}</div>'
    ]
    for grade, description in grade_descriptions.items():
        thumbnail_grid = _tooltip_thumbnail_grid(
            grade_example_paths.get(grade, ()),
            css_class="lens-grade-help__thumbnail-grid",
            alt_text=f"Grade {grade} lens candidate example",
        )
        sections.append(
            '<div class="lens-grade-help__section">'
            '<div class="lens-grade-help__description">'
            f"<strong>Grade {html.escape(grade)}:</strong> "
            f"{html.escape(description)}"
            "</div>"
            f"{thumbnail_grid}"
            "</div>"
        )
    _render_help_label_html(
        label,
        "".join(sections),
        ("concept-popover--lens-grades",),
    )

def render_app_flow_help() -> None:
    steps_html = "".join(
        f"<li>{html.escape(step)}</li>"
        for step in ANALYSIS_FLOW_STEPS
    )
    st.markdown(
        f"""
        <div class="sidebar-flow-help">
            <span class="concept-help" tabindex="0">?
                <span class="concept-popover">
                    <strong>Analysis flow</strong>
                    <ol class="concept-popover__steps">{steps_html}</ol>
                </span>
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_cluster_visual_summary(
    clustered_df: pd.DataFrame,
    cluster_summary_df: pd.DataFrame,
    pca_columns: list[str],
    selected_features: list[str],
) -> None:
    view_model = build_cluster_summary_view_model(
        clustered_df,
        cluster_summary_df,
        pca_columns,
        selected_features,
    )
    render_cluster_visual_summary_view_model(view_model)

def cluster_summary_signature(
    clustered_df: pd.DataFrame,
    cluster_summary_df: pd.DataFrame,
    pca_columns: list[str],
    selected_features: list[str],
) -> tuple:
    cluster_values = tuple(
        tuple(row)
        for row in cluster_summary_df[
            ["cluster", "n_objects", "n_lenses", "lens_rate", "enrichment", "label_entropy_bits"]
        ].itertuples(index=False, name=None)
    )
    return (
        id(clustered_df),
        cluster_values,
        tuple(pca_columns),
        tuple(selected_features),
        SUMMARY_VISUAL_CLUSTER_LIMIT,
    )

def summary_features_for_visuals(
    pca_columns: list[str],
    selected_features: list[str],
) -> list[str]:
    summary_features = [
        feature for feature in selected_features if feature in pca_columns
    ] or [feature for feature in DEFAULT_CLUSTER_FEATURES if feature in pca_columns]
    return summary_features or pca_columns[: min(4, len(pca_columns))]


def build_cluster_preview_model(
    cluster_df: pd.DataFrame,
    summary_row: pd.Series,
    summary_features: list[str],
) -> dict[str, object]:
    cluster_id = int(summary_row["cluster"])
    canonical_row, anomaly_row, random_rows, lens_rows = cluster_visual_rows(
        cluster_df,
        summary_features,
        cluster_id,
    )
    lens_captions = []
    for row in lens_rows:
        lens_grade = row.get("lens_grade", "")
        if pd.isna(lens_grade) or not str(lens_grade).strip():
            lens_captions.append("Grade ?")
        else:
            lens_captions.append(f"Grade {str(lens_grade).strip()}")

    return {
        "cluster_id": cluster_id,
        "n_objects": int(summary_row["n_objects"]),
        "n_lenses": int(summary_row["n_lenses"]),
        "lens_rate": float(summary_row["lens_rate"]),
        "enrichment": float(summary_row["enrichment"]),
        "label_entropy_bits": float(summary_row["label_entropy_bits"]),
        "canonical_row": canonical_row,
        "anomaly_row": anomaly_row,
        "random_rows": random_rows,
        "lens_rows": lens_rows,
        "lens_captions": lens_captions,
    }


def build_selected_cluster_view_model(
    clustered_df: pd.DataFrame,
    cluster_summary_df: pd.DataFrame,
    selected_cluster: int,
    pca_columns: list[str],
    selected_features: list[str],
) -> dict[str, object]:
    summary_rows = cluster_summary_df[
        cluster_summary_df["cluster"].astype(int) == int(selected_cluster)
    ]
    if summary_rows.empty:
        raise ValueError(f"Cluster {selected_cluster} is not present in the summary.")
    cluster_df = clustered_df[
        clustered_df["cluster"].astype(int) == int(selected_cluster)
    ].copy()
    return build_cluster_preview_model(
        cluster_df,
        summary_rows.iloc[0],
        summary_features_for_visuals(pca_columns, selected_features),
    )

def build_cluster_summary_view_model(
    clustered_df: pd.DataFrame,
    cluster_summary_df: pd.DataFrame,
    pca_columns: list[str],
    selected_features: list[str],
) -> dict:
    summary_features = summary_features_for_visuals(pca_columns, selected_features)
    histogram_features = summary_features[:SUMMARY_HISTOGRAM_FEATURE_LIMIT]
    cluster_download_rows = []
    visual_clusters = []
    cluster_groups = clustered_df.groupby("cluster", sort=False)

    for position, (_, summary_row) in enumerate(cluster_summary_df.iterrows()):
        cluster_id = int(summary_row["cluster"])
        cluster_df = cluster_groups.get_group(cluster_id).copy()
        include_visual_preview = position < SUMMARY_VISUAL_CLUSTER_LIMIT
        if include_visual_preview:
            cluster_model = build_cluster_preview_model(
                cluster_df,
                summary_row,
                summary_features,
            )
            canonical_row = cluster_model["canonical_row"]
            anomaly_row = cluster_model["anomaly_row"]
        else:
            canonical_row, anomaly_row = cluster_extreme_rows(
                cluster_df,
                summary_features,
            )
        canonical = ""
        anomalous = ""
        if canonical_row is not None:
            canonical = str(canonical_row.get("object_id", ""))
        if anomaly_row is not None:
            anomalous = str(anomaly_row.get("object_id", ""))

        cluster_download_rows.append(
            {
                "cluster": cluster_id,
                "n_objects": int(summary_row["n_objects"]),
                "n_lenses": int(summary_row["n_lenses"]),
                "lens_rate": float(summary_row["lens_rate"]),
                "enrichment": float(summary_row["enrichment"]),
                "label_entropy_bits": float(summary_row["label_entropy_bits"]),
                "canonical": canonical,
                "anomalous": anomalous,
            }
        )

        if not include_visual_preview:
            continue
        visual_clusters.append(cluster_model)

    return {
        "cluster_download_df": pd.DataFrame(cluster_download_rows),
        "histogram_features": histogram_features,
        "clusters": visual_clusters,
        "total_clusters": len(cluster_download_rows),
    }


def cluster_preview_object_ids(cluster_models: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    object_ids = set()
    for model in cluster_models:
        rows = [
            model["canonical_row"],
            model["anomaly_row"],
            *model["random_rows"],
            *model["lens_rows"],
        ]
        for row in rows:
            if row is not None and not pd.isna(row.get("object_id")):
                object_ids.add(str(row["object_id"]))
    return tuple(sorted(object_ids))


@st.cache_data(show_spinner=False)
def cluster_preview_aladin_links(
    morph_path: str,
    object_ids: tuple[str, ...],
) -> dict[str, str]:
    if not object_ids:
        return {}
    try:
        morphology = load_morphology_objects(
            morph_path,
            object_ids,
            columns=("right_ascension", "declination"),
        )
    except OSError:
        return {}
    if not {"right_ascension", "declination"}.issubset(morphology.columns):
        return {}

    links = {}
    for row in morphology.itertuples(index=False):
        try:
            ra = float(row.right_ascension)
            dec = float(row.declination)
        except (TypeError, ValueError):
            continue
        if np.isfinite(ra) and np.isfinite(dec) and -90 <= dec <= 90:
            links[str(row.object_id)] = aladin_url(ra, dec)
    return links


def render_cluster_preview_card(
    cluster_model: Mapping[str, object],
    *,
    show_analysis_button: bool,
    aladin_links: Mapping[str, str] | None = None,
) -> bool:
    cluster_id = int(cluster_model["cluster_id"])
    if aladin_links is None:
        aladin_links = cluster_preview_aladin_links(
            MORPH_PATH,
            cluster_preview_object_ids([cluster_model]),
        )
    requested = False
    with st.container(border=True):
        stats_cols = st.columns([1, 1, 1, 1, 1, 1])
        stats_cols[0].metric("Cluster", cluster_id)
        stats_cols[1].metric(
            "Objects",
            format_thousands_dot(int(cluster_model["n_objects"])),
        )
        stats_cols[2].metric(
            "Lenses",
            format_thousands_dot(int(cluster_model["n_lenses"])),
        )
        stats_cols[3].metric(
            "Density",
            f"{format_decimal_comma(float(cluster_model['lens_rate']) * 100, 2)}%",
            help=LABEL_DENSITY_HELP,
        )
        stats_cols[4].metric(
            "Enrichment",
            f"{format_decimal_comma(float(cluster_model['enrichment']), 2)}x",
            help=ENRICHMENT_HELP,
        )
        stats_cols[5].metric(
            "Entropy",
            format_decimal_comma(float(cluster_model["label_entropy_bits"]), 3),
            help=LABEL_ENTROPY_HELP,
        )

        image_cols = st.columns([2, 3, 5])
        with image_cols[0]:
            show_thumbnail_group(
                "Canonical / anomalous",
                [cluster_model["canonical_row"], cluster_model["anomaly_row"]],
                ["Canonical", "Anomalous"],
                aladin_links=aladin_links,
            )
        with image_cols[1]:
            random_rows = list(cluster_model["random_rows"])
            show_thumbnail_group(
                "Random",
                random_rows,
                [f"Random {index + 1}" for index in range(len(random_rows))],
                aladin_links=aladin_links,
            )
        with image_cols[2]:
            show_thumbnail_group(
                "Labelled lens candidates in the cluster",
                list(cluster_model["lens_rows"]),
                list(cluster_model["lens_captions"]),
                skip_missing_images=True,
                aladin_links=aladin_links,
            )

        if show_analysis_button:
            requested = st.button(
                "Analyse this cluster",
                key=f"analyse_cluster_{cluster_id}",
                type="secondary",
            )
    return requested


def render_cluster_visual_summary_view_model(
    view_model: dict,
) -> int | None:
    total_clusters = int(view_model.get("total_clusters", len(view_model["clusters"])))
    visible_clusters = len(view_model["clusters"])
    if total_clusters > visible_clusters:
        st.caption(
            "Showing visual previews for the first "
            f"{visible_clusters} of {format_thousands_dot(total_clusters)} clusters."
        )
    aladin_links = cluster_preview_aladin_links(
        MORPH_PATH,
        cluster_preview_object_ids(view_model["clusters"]),
    )
    requested_cluster: int | None = None
    for cluster_model in view_model["clusters"]:
        cluster_id = int(cluster_model["cluster_id"])
        if render_cluster_preview_card(
            cluster_model,
            show_analysis_button=True,
            aladin_links=aladin_links,
        ):
            requested_cluster = cluster_id
    return requested_cluster

def show_lens_status(row: pd.Series) -> None:
    is_lens = bool(row.get("is_lens", False))
    lens_grade = row.get("lens_grade", "")
    lens_grade_text = ""
    if not pd.isna(lens_grade) and str(lens_grade).strip():
        lens_grade_text = f"Grade: {lens_grade}"

    if is_lens:
        label = "LENS CANDIDATE"
        css_class = "lens-status--yes"
        meta = lens_grade_text or "Object present in the strong-lensing catalogue."
    else:
        label = "UNKNOWN"
        css_class = "lens-status--no"
        meta = "Object not marked as a lens candidate in the joined catalogue."

    st.markdown(
        f"""
        <div class="lens-status {css_class}">
            <span class="lens-status__label">{label}</span>
            <span class="lens-status__meta">{meta}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_detectable_image(
    path: str,
    caption: str,
    label: str,
    caption_markdown: str | None = None,
) -> None:
    if not ENABLE_ARC_LIKE_STRUCTURE_DETECTION:
        try:
            image_src = display_image_src(path)
        except Exception as exc:
            st.warning(f"Could not open the image: {exc}")
            return
        render_image_src(image_src, caption)
        render_image_caption(caption, caption_markdown)
        return

    key_digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]
    button_key = f"detect_arc_like_structures_{label}_{key_digest}"
    active_key = f"{button_key}_active"
    active = bool(st.session_state.get(active_key, False))

    if active:
        try:
            from .arc_detection_cv import detect_arc_overlay_src

            image_src = detect_arc_overlay_src(path)
        except Exception as exc:
            st.warning(f"Could not run arc-like structure detection: {exc}")
            return
        image_caption = "Arc-like structure candidates"
    else:
        try:
            image_src = display_image_src(path)
        except Exception as exc:
            st.warning(f"Could not open the image: {exc}")
            return
        image_caption = caption

    render_image_src(image_src, image_caption)
    render_image_caption(image_caption, caption_markdown if not active else None)

    button_label = "Show original image" if active else "Detect arc-like structures"
    if st.button(button_label, key=button_key, type="secondary"):
        st.session_state[active_key] = not active
        st.rerun()


def selected_umap_cutout_path(row: pd.Series) -> str | None:
    object_id = str(row.get("object_id", "")).strip()
    if object_id:
        morphology_df = load_morphology_object(MORPH_PATH, object_id)
        if not morphology_df.empty:
            morphology_id_str = morphology_df.iloc[0].get("id_str")
            cutout_path = morphology_cutout_path(morphology_id_str, object_id)
            if cutout_path is not None:
                return cutout_path

    return morphology_cutout_path(row.get("id_str"), row.get("object_id"))


def show_object_details(row: pd.Series, *, key: str) -> None:
    st.subheader("Selected object")
    with st.container(key=f"{key}_object_id"):
        label_col, value_col = st.columns(
            [1, 3], gap="small", vertical_alignment="center"
        )
        with label_col:
            st.markdown("**Object ID**")
        with value_col:
            st.code(str(row.get("object_id", "")), language=None, wrap_lines=True)
    show_lens_status(row)
    if ENABLE_SLED_CHECK and not bool(row.get("is_lens", False)):
        ra, dec = object_sky_coordinates(str(row.get("object_id", "")))
        render_sled_check(str(row.get("object_id", "")), ra, dec, key=key)

    cutout_path = selected_umap_cutout_path(row)
    lens_path = lens_image_path(row.get("lens_id_str"))
    if lens_path is None and bool(row.get("is_lens", False)):
        lens_path = lens_image_path(row.get("id_str"))

    if cutout_path is None and lens_path is None:
        st.info("No associated image was found in the configured paths.")
    else:
        if cutout_path is not None:
            show_detectable_image(
                cutout_path,
                "Morphology cutout",
                "selected_morphology",
                caption_markdown=aladin_caption_for_object(
                    "Morphology cutout",
                    row.get("object_id", ""),
                ),
            )
        if lens_path is not None:
            show_detectable_image(lens_path, "Strong-lens image", "selected_lens")


def show_selected_object_summary(row: pd.Series) -> None:
    st.markdown("**Object summary**")
    details = {
        "id_str": row.get("id_str", ""),
        "cluster": row.get("cluster", ""),
        "is_lens": bool(row.get("is_lens", False)),
        "is_canonical": bool(row.get("is_canonical", False)),
        "is_anomaly": bool(row.get("is_anomaly", False)),
        "lens_grade": row.get("lens_grade", ""),
        "dist_to_cluster_centroid": row.get("dist_to_cluster_centroid", np.nan),
        "umap_1": row.get("umap_1", np.nan),
        "umap_2": row.get("umap_2", np.nan),
    }
    st.dataframe(
        pd.DataFrame([
            {"field": field, "value": str(value) if pd.notna(value) else None}
            for field, value in details.items()
        ]),
        use_container_width=True,
        hide_index=True,
        height=OBJECT_DETAIL_TABLE_HEIGHT,
    )


def show_selected_object_tables(row: pd.Series, selected_features: list[str]) -> None:
    summary_col, morphology_col = st.columns(2)
    with summary_col:
        show_selected_object_summary(row)
    with morphology_col:
        show_morphology_catalogue_row(row)
    pca_col, physical_col = st.columns(2)
    with pca_col:
        show_selected_pca_components(row, selected_features)
    with physical_col:
        show_physical_measurements_row(row)


def show_morphology_catalogue_row(row: pd.Series) -> None:
    morphology_df = load_morphology_object(MORPH_PATH, str(row.get("object_id", "")))
    if not morphology_df.empty:
        st.markdown("**Morphology catalogue features**")
        morph_display = pd.DataFrame(
            table_rows_with_coordinate_formats(morphology_df.iloc[0].dropna().to_dict())
        )
        st.dataframe(
            morph_display,
            use_container_width=True,
            hide_index=True,
            height=OBJECT_DETAIL_TABLE_HEIGHT,
        )


def object_physical_measurements(object_id: object) -> pd.DataFrame:
    """Cache complete single-object rows without widening cluster-level queries."""
    normalized_id = normalize_object_ids(pd.Series([object_id])).iloc[0]
    cache = st.session_state.setdefault("object_physical_measurements", {})
    cache_key = (PHYSICAL_MEASUREMENTS_PATH, normalized_id)
    if cache_key not in cache:
        measurements = load_physical_measurement_object(
            PHYSICAL_MEASUREMENTS_PATH, normalized_id
        )
        if len(cache) >= 64:
            cache.pop(next(iter(cache)))
        cache[cache_key] = measurements
    return cache[cache_key].copy()


def show_physical_measurements_row(row: pd.Series) -> None:
    st.markdown("**Physical measurements**")
    try:
        physical_df = object_physical_measurements(row.get("object_id", ""))
    except Exception as exc:
        log_app_event(
            "object_physical_measurements_failed",
            object_id=str(row.get("object_id", "")),
            error_type=type(exc).__name__,
        )
        st.warning("Physical measurements could not be loaded for this object.")
        return
    if physical_df.empty:
        st.info("No physical measurements were found for this object_id.")
        return

    display = pd.DataFrame(
        physical_measurement_display_rows(
            physical_df.iloc[0],
            excluded_fields=("object_id", "right_ascension", "declination"),
        )
    )
    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        height=OBJECT_DETAIL_TABLE_HEIGHT,
    )

def show_selected_pca_components(row: pd.Series, selected_features: list[str]) -> None:
    st.markdown("**Selected PCA components**")
    st.dataframe(
        pd.DataFrame(
            [{"feature": feature, "value": row.get(feature)} for feature in selected_features]
        ),
        use_container_width=True,
        hide_index=True,
        height=OBJECT_DETAIL_TABLE_HEIGHT,
    )

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

def aladin_target_coordinates(ra_degrees: float, dec_degrees: float) -> str:
    ra_seconds_total = (float(ra_degrees) % 360.0) / 15.0 * 3600.0
    ra_hours = int(ra_seconds_total // 3600)
    ra_minutes = int((ra_seconds_total % 3600) // 60)
    ra_seconds = ra_seconds_total % 60

    dec_sign = "+" if float(dec_degrees) >= 0 else "-"
    dec_seconds_total = abs(float(dec_degrees)) * 3600.0
    dec_degrees_int = int(dec_seconds_total // 3600)
    dec_minutes = int((dec_seconds_total % 3600) // 60)
    dec_seconds = dec_seconds_total % 60

    return (
        f"{ra_hours:02d} {ra_minutes:02d} {ra_seconds:06.3f}"
        f"{dec_sign}{dec_degrees_int:02d} {dec_minutes:02d} {dec_seconds:06.3f}"
    )

def aladin_url(ra_degrees: float, dec_degrees: float) -> str:
    target = quote(aladin_target_coordinates(ra_degrees, dec_degrees), safe="")
    return (
        "https://aladin.unistra.fr/AladinLite/"
        f"?target={target}&fov=0.02&survey=CDS%2FP%2FEuclid%2FQ1%2Fcolor"
    )

def aladin_caption(caption: str, ra_degrees: float, dec_degrees: float) -> str:
    aladin_link = aladin_url(ra_degrees, dec_degrees)
    return (
        '<span style="display:inline-flex;align-items:center;justify-content:center;'
        'gap:0.6rem;flex-wrap:wrap;max-width:100%">'
        f'<span style="opacity:0.65">{html.escape(caption)}</span>'
        + aladin_icon_link(aladin_link)
        + "</span>"
    )

@st.cache_data(ttl=3600, max_entries=256, show_spinner=False)
def object_sky_coordinates(object_id: str) -> tuple[float | None, float | None]:
    try:
        morphology_df = load_morphology_object(MORPH_PATH, object_id)
        if not morphology_df.empty:
            row = morphology_df.iloc[0]
            ra, dec = float(row["right_ascension"]), float(row["declination"])
            if np.isfinite(ra) and np.isfinite(dec):
                return ra, dec
    except Exception:
        log_app_event("object_coordinates_unavailable", object_id=object_id)
    return None, None


def aladin_caption_for_object(caption: str, object_id: object) -> str | None:
    ra, dec = object_sky_coordinates(str(object_id))
    if ra is None or dec is None:
        return None
    return aladin_caption(caption, ra, dec)

def table_rows_with_coordinate_formats(values: dict) -> list[dict]:
    rows = []
    for field, value in values.items():
        rows.append({"field": field, "value": value})
        if field == "right_ascension":
            try:
                rows.append(
                    {
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
                        "field": "declination_dms",
                        "value": format_dec_dms(float(value)),
                    }
                )
            except (TypeError, ValueError):
                pass
    return rows

def object_summary_display_rows(object_summary: dict) -> list[dict]:
    return table_rows_with_coordinate_formats(object_summary)

def object_search_morphology_values(morphology_row: dict) -> dict:
    redundant_fields = {
        "object_id",
        "right_ascension",
        "right_ascension_hms",
        "declination",
        "declination_dms",
        "declination_hms",
    }
    return {
        field: value
        for field, value in morphology_row.items()
        if field not in redundant_fields
    }

def object_lens_candidate_summary(object_id: object) -> dict[str, object]:
    lens_df = load_lens_catalog(LENS_PATH, tuple(LENS_GRADE_OPTIONS))
    normalized_object_id = normalize_object_ids(pd.Series([object_id])).iloc[0]
    matching_rows = lens_df[
        normalize_object_ids(lens_df["object_id"]) == normalized_object_id
    ]
    if matching_rows.empty:
        return {
            "is_lens_candidate": "No",
            "lens_candidate_grade": "",
        }

    grade = matching_rows.iloc[0].get("grade", "")
    return {
        "is_lens_candidate": "Yes",
        "lens_candidate_grade": str(grade).strip().upper(),
    }

def render_euclid_object_search(object_id: str) -> None:
    cached_search = st.session_state.get("euclid_search_result")
    if cached_search and cached_search.get("requested_object_id") == str(object_id):
        result = cached_search["result"]
        morphology_df = cached_search["morphology_df"]
        physical_df = cached_search.get("physical_df", pd.DataFrame())
        lens_candidate_summary = cached_search.get("lens_candidate_summary", {})
    else:
        started_at = time.perf_counter()
        result = fetch_euclid_object_summary(object_id)
        morphology_df = load_morphology_object(MORPH_PATH, str(result["object_id"]))
        try:
            physical_df = object_physical_measurements(result["object_id"])
        except Exception as exc:
            log_app_event(
                "object_physical_measurements_failed",
                object_id=str(result["object_id"]),
                error_type=type(exc).__name__,
            )
            physical_df = pd.DataFrame()
        lens_candidate_summary = object_lens_candidate_summary(result["object_id"])
        mosaic_summary = result["mosaic_summary"]
        log_app_event(
            "object_search_completed",
            duration_seconds=round(time.perf_counter() - started_at, 3),
            has_precomputed_cutout=bool(result.get("cutout_path")),
            has_morphology_row=not morphology_df.empty,
            instrument=str(mosaic_summary.get("instrument_name", "")),
            tile_index=str(mosaic_summary.get("tile_index", "")),
        )
        st.session_state["euclid_search_result"] = {
            "requested_object_id": str(object_id),
            "result": result,
            "morphology_df": morphology_df,
            "physical_df": physical_df,
            "lens_candidate_summary": lens_candidate_summary,
        }

    object_summary = {**result["object_summary"], **lens_candidate_summary}
    mosaic_summary = result["mosaic_summary"]

    with st.container(border=True):
        ra_degrees = float(object_summary["right_ascension"])
        dec_degrees = float(object_summary["declination"])
        metric_cols = st.columns([2.4, 1, 1, 1])
        metric_cols[0].metric("object_id", str(result["object_id"]))
        metric_cols[1].metric("RA", f"{ra_degrees:.6f}")
        metric_cols[2].metric("Dec", f"{dec_degrees:.6f}")
        metric_cols[3].metric("Tile", str(mosaic_summary.get("tile_index", "")))

        image_col, summary_col = st.columns([1, 1])
        with image_col:
            if ENABLE_SLED_CHECK and lens_candidate_summary.get("is_lens_candidate") != "Yes":
                render_sled_check(
                    str(result["object_id"]), ra_degrees, dec_degrees, key="object_search"
                )
            cutout_path = result.get("cutout_path")
            if cutout_path:
                show_detectable_image(
                    str(cutout_path),
                    "Euclid VIS cutout",
                    "search_cutout",
                    caption_markdown=aladin_caption("Euclid VIS cutout", ra_degrees, dec_degrees),
                )
            else:
                st.info(
                    "No precomputed JPEG cutout was found for this object. "
                    "No FITS file was downloaded."
                )

        with summary_col:
            object_summary_title = "**Object summary**"
            lens_candidate_grade = str(
                lens_candidate_summary.get("lens_candidate_grade", "")
            ).strip()
            if (
                lens_candidate_summary.get("is_lens_candidate") == "Yes"
                and lens_candidate_grade
            ):
                object_summary_title += (
                    f" (Lens candidate grade {lens_candidate_grade})"
                )
            st.markdown(object_summary_title)
            object_display = pd.DataFrame(
                object_summary_display_rows(object_summary)
            )
            st.dataframe(object_display, use_container_width=True, hide_index=True)

        catalogue_col, physical_col = st.columns([1, 1])
        with catalogue_col:
            st.markdown("**Mosaic summary**")
            mosaic_display = pd.DataFrame(
                [{"field": field, "value": value} for field, value in mosaic_summary.items()]
            )
            st.dataframe(mosaic_display, use_container_width=True, hide_index=True)

            st.markdown("**Morphology catalogue features**")
            if morphology_df.empty:
                st.info("No morphology catalogue row was found for this object_id.")
            else:
                morphology_display = pd.DataFrame(
                    table_rows_with_coordinate_formats(
                        object_search_morphology_values(
                            morphology_df.iloc[0].dropna().to_dict()
                        )
                    )
                )
                st.dataframe(
                    morphology_display,
                    use_container_width=True,
                    hide_index=True,
                    height=OBJECT_DETAIL_TABLE_HEIGHT,
                )

        with physical_col:
            st.markdown("**Physical measurements**")
            if physical_df.empty:
                st.info("No physical measurements were found for this object_id.")
            else:
                physical_display = pd.DataFrame(
                    physical_measurement_display_rows(
                        physical_df.iloc[0],
                        excluded_fields=(
                            "object_id",
                            "right_ascension",
                            "declination",
                        ),
                    )
                )
                with st.container(key="search_physical_measurements", height="stretch"):
                    st.dataframe(
                        physical_display,
                        use_container_width=True,
                        hide_index=True,
                        height="stretch",
                    )

        object_download_df = object_search_download_df(
            object_summary,
            morphology_df,
            mosaic_summary,
            physical_df,
        )
        st.download_button(
            "Download object search data",
            data=dataframe_to_csv_bytes(object_download_df),
            file_name=f"object_search_{result['object_id']}.csv",
            mime="text/csv",
        )

def validate_paths() -> pd.DataFrame:
    rows = [
        ("MORPH_PATH", MORPH_PATH),
        ("PHYSICAL_MEASUREMENTS_PATH", PHYSICAL_MEASUREMENTS_PATH),
        ("PARQUET_PATH", PARQUET_PATH),
        ("CUTOUT_BASE", CUTOUT_BASE),
        ("LENS_PATH", LENS_PATH),
        ("LENS_IMG_BASE", LENS_IMG_BASE),
    ]
    return pd.DataFrame(
        [
            {
                "name": name,
                "path": path,
                "exists": path_exists(path),
            }
            for name, path in rows
        ]
    )

def request_clustering() -> None:
    st.session_state["cluster_requested"] = True
    st.session_state["cluster_summary_expanded"] = True

def collapse_cluster_summary() -> None:
    st.session_state["cluster_summary_expanded"] = False
