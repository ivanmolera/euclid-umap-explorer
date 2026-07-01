from __future__ import annotations

import base64
import hashlib
import html
import time
from urllib.parse import quote

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as st_components

from .arc_detection_cv import detect_arc_overlay_src
from .analysis import (
    add_cluster_extreme_roles,
    format_pca_filter,
    lens_grade_sort_key,
)
from .catalogs import load_lens_catalog, load_morphology_object, normalize_object_ids
from .config import (
    CUTOUT_BASE,
    DEFAULT_CLUSTER_FEATURES,
    LENS_GRADE_OPTIONS,
    LENS_IMG_BASE,
    LENS_PATH,
    MORPH_PATH,
    PARQUET_PATH,
    PCA_FILTER_OPERATORS,
    PCA_FILTER_RECOMMENDATION_MIN_LENSES,
    PCA_FILTER_RECOMMENDATION_MIN_OBJECTS,
    PCA_FILTER_RECOMMENDATION_MIN_RECALL,
    SUMMARY_DISTPLOT_MAX_POINTS_PER_GROUP,
    SUMMARY_HISTOGRAM_BINS,
    SUMMARY_HISTOGRAM_FEATURE_LIMIT,
    SUMMARY_LENS_OBJECTS,
    SUMMARY_RANDOM_OBJECTS,
    SUMMARY_THUMBNAIL_WIDTH,
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
from .storage import path_exists

OVERLAY_BUTTON_MESSAGES = {
    "Run clustering": "Running BIRCH clustering...",
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
        """
        <style>
        .block-container {
            padding-top: 2rem;
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
            bottom: calc(100% + 0.75rem);
            box-shadow: 0 14px 34px rgba(15, 23, 42, 0.32);
            color: #f8fafc !important;
            display: none;
            font-size: 0.9rem;
            font-weight: 700;
            left: 0;
            line-height: 1.35;
            max-width: min(360px, calc(100vw - 2rem));
            min-width: 260px;
            padding: 0.75rem 0.9rem;
            position: absolute;
            text-align: left;
            z-index: 1000;
        }
        .concept-popover::after {
            border-left: 0.55rem solid transparent;
            border-right: 0.55rem solid transparent;
            border-top: 0.55rem solid #111923;
            bottom: -0.52rem;
            content: "";
            left: 1.8rem;
            position: absolute;
            transform: translateX(-50%);
        }
        .concept-help:hover .concept-popover,
        .concept-help:focus .concept-popover {
            display: block;
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
        .sidebar-flow-help .concept-popover {
            bottom: auto;
            left: 5.75rem;
            max-width: min(360px, calc(100vw - 7rem));
            min-width: 310px;
            position: fixed;
            right: auto;
            top: 4.75rem;
            white-space: normal;
        }
        .sidebar-flow-help .concept-popover::after {
            border-bottom: 0.55rem solid transparent;
            border-right: 0.55rem solid #111923;
            border-top: 0.55rem solid transparent;
            bottom: auto;
            left: -0.52rem;
            right: auto;
            top: 0.3rem;
            transform: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

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

def show_thumbnail(
    row: pd.Series | None,
    caption: str,
    prefer_lens_image: bool = False,
) -> None:
    if row is None:
        st.caption(caption)
        st.caption("No object")
        return

    path = object_image_path(row, prefer_lens_image=prefer_lens_image)
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
    escaped_object_id = html.escape(str(object_id))
    escaped_id_str = html.escape(str(id_str))
    id_str_html = (
        f"<div class='thumb-modal-meta'>id_str: {escaped_id_str}</div>"
        if not pd.isna(id_str) and str(id_str).strip()
        else ""
    )

    st.markdown(
        f"""
        <style>
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
        }}
        #{modal_id} .thumb-modal-meta {{
            color: #f8fafc;
            font-size: 0.9rem;
            margin-bottom: 0.45rem;
        }}
        #{modal_id} img {{
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
                <div class="thumb-modal-meta"><strong>{escaped_caption}</strong></div>
                <div class="thumb-modal-meta">object_id: {escaped_object_id}</div>
                {id_str_html}
                <img src="{image_src}" alt="{escaped_caption}" />
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def show_thumbnail_group(
    title: str,
    rows: list[pd.Series],
    captions: list[str],
    prefer_lens_image: bool = False,
) -> None:
    render_thumbnail_group_title(title)
    count = max(len(rows), 1)
    for column, row, caption in zip(st.columns(count), rows or [None], captions or [""]):
        with column:
            show_thumbnail(row, caption, prefer_lens_image=prefer_lens_image)

def cluster_visual_rows(
    cluster_df: pd.DataFrame,
    summary_features: list[str],
    cluster_id: int,
) -> tuple[pd.Series | None, pd.Series | None, list[pd.Series], list[pd.Series]]:
    marked = add_cluster_extreme_roles(cluster_df, summary_features)

    canonical_rows = marked[marked["is_canonical"]]
    anomaly_rows = marked[marked["is_anomaly"]]
    canonical_row = canonical_rows.iloc[0] if not canonical_rows.empty else None
    anomaly_row = anomaly_rows.iloc[0] if not anomaly_rows.empty else None

    used_object_ids = {
        str(row.get("object_id"))
        for row in (canonical_row, anomaly_row)
        if row is not None and not pd.isna(row.get("object_id"))
    }
    random_pool = cluster_df[
        ~cluster_df["object_id"].astype("string").isin(used_object_ids)
    ]
    if random_pool.empty:
        random_pool = cluster_df
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
    if "is_lens" not in cluster_df.columns or feature not in cluster_df.columns:
        return None

    work_df = cluster_df[[feature, "is_lens"]].dropna().copy()
    if work_df.empty:
        return None

    work_df["is_lens"] = work_df["is_lens"].astype(bool)
    total_objects = len(work_df)
    total_lenses = int(work_df["is_lens"].sum())
    if total_lenses == 0 or total_lenses == total_objects:
        return None

    base_lens_rate = total_lenses / total_objects
    values = work_df[feature].astype(float)
    candidate_thresholds = np.unique(np.percentile(values, np.linspace(5, 95, 91)))
    best: dict | None = None

    min_objects = min(
        max(PCA_FILTER_RECOMMENDATION_MIN_OBJECTS, int(total_objects * 0.005)),
        max(total_objects - 1, 1),
    )
    min_lenses = min(PCA_FILTER_RECOMMENDATION_MIN_LENSES, total_lenses)

    for threshold in candidate_thresholds:
        for operator, selected_mask in (
            (">=", values >= threshold),
            ("<=", values <= threshold),
        ):
            selected = work_df[selected_mask]
            n_selected = len(selected)
            if n_selected < min_objects:
                continue

            n_lenses = int(selected["is_lens"].sum())
            if n_lenses < min_lenses:
                continue

            lens_rate = n_lenses / n_selected
            recall = n_lenses / total_lenses
            if recall < PCA_FILTER_RECOMMENDATION_MIN_RECALL:
                continue

            enrichment = lens_rate / base_lens_rate if base_lens_rate else 0.0
            score = enrichment * np.sqrt(recall)
            recommendation = {
                "feature": feature,
                "operator": operator,
                "value": float(threshold),
                "n_selected": int(n_selected),
                "n_lenses": int(n_lenses),
                "lens_rate": float(lens_rate),
                "base_lens_rate": float(base_lens_rate),
                "recall": float(recall),
                "enrichment": float(enrichment),
                "score": float(score),
            }
            if best is None or recommendation["score"] > best["score"]:
                best = recommendation

    return best

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

@st.fragment
def render_cluster_histograms(
    cluster_id: int,
    cluster_df: pd.DataFrame,
    summary_features: list[str],
) -> None:
    n_lenses = int(cluster_df["is_lens"].sum())
    n_non_lenses = len(cluster_df) - n_lenses
    if n_lenses == 0:
        return

    state_key = f"cluster_histograms_visible_{cluster_id}"
    st.session_state.setdefault(state_key, False)
    button_label = (
        "Update PCA histograms"
        if st.session_state.get(state_key)
        else "Compute PCA histograms"
    )
    st.button(
        button_label,
        key=f"cluster_histograms_button_{cluster_id}",
        on_click=lambda key=state_key: st.session_state.update({key: True}),
    )

    if not st.session_state.get(state_key):
        return

    if n_non_lenses == 0:
        st.info("This cluster does not contain non-lens objects for comparison.")
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
            """
            <div style="font-weight: 700; margin-bottom: 0.35rem;">
                Recommended PCA thresholds
                <span class="concept-help" tabindex="0" style="margin-left: 0.25rem;">?
                    <span class="concept-popover">
                        For each PCA component, the table shows the threshold that best enriches
                        lens candidates against unknown objects in this cluster, balancing lens
                        enrichment and lens recall. These thresholds are exploratory filters,
                        not confirmed classification rules.
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


def render_thumbnail_group_title(title: str) -> None:
    if title != "Canonical / anomalous":
        st.caption(title)
        return

    st.markdown(
        """
        <div style="color: inherit; font-size: 0.82rem; margin-bottom: 0.35rem; opacity: 0.72;">
            <span class="concept-help" tabindex="0">Canonical
                <span class="concept-popover">
                    Object closest to the cluster centroid in the selected PCA feature space.
                </span>
            </span>
            <span style="margin: 0 0.2rem;">/</span>
            <span class="concept-help" tabindex="0">Anomalous
                <span class="concept-popover">
                    Object farthest from the cluster centroid in the selected PCA feature space.
                </span>
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_help_label(label: str, help_text: str) -> None:
    escaped_label = html.escape(label)
    escaped_help = html.escape(help_text).replace("\n", "<br>")
    st.markdown(
        f"""
        <div style="font-size: 0.88rem; font-weight: 600; margin-bottom: 0.2rem;">
            <span class="concept-help" tabindex="0">
                {escaped_label}
                <span class="concept-popover">{escaped_help}</span>
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_app_flow_help() -> None:
    st.markdown(
        """
        <div class="sidebar-flow-help">
            <span class="concept-help" tabindex="0">?
                <span class="concept-popover">
                    <strong>Analysis flow</strong><br>
                    1. Run BIRCH over the full PCA catalogue.<br>
                    2. Rank clusters by lens-candidate density.<br>
                    3. Visually inspect enriched clusters.<br>
                    4. Compute PCA histograms and recommended thresholds.<br>
                    5. Apply PCA filters to isolate enriched regions.<br>
                    6. Generate UMAP for the filtered subset.<br>
                    7. Apply hierarchical subclustering.<br>
                    8. Use A/B/C labels to guide semi-supervised UMAP in promising subclusters.<br>
                    9. Export Unknown objects close to lens-rich regions.<br>
                    10. Visually review cutouts and cross-check them in Aladin.<br>
                    11. Use prioritized candidates for human inspection, active learning, or downstream models.
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
    render_cluster_visual_summary_view_model(view_model, clustered_df)

def cluster_summary_signature(
    clustered_df: pd.DataFrame,
    cluster_summary_df: pd.DataFrame,
    pca_columns: list[str],
    selected_features: list[str],
) -> tuple:
    cluster_values = tuple(
        tuple(row)
        for row in cluster_summary_df[
            ["cluster", "n_objects", "n_lenses", "lens_rate"]
        ].itertuples(index=False, name=None)
    )
    return (
        id(clustered_df),
        cluster_values,
        tuple(pca_columns),
        tuple(selected_features),
    )

def summary_features_for_visuals(
    pca_columns: list[str],
    selected_features: list[str],
) -> list[str]:
    summary_features = [
        feature for feature in selected_features if feature in pca_columns
    ] or [feature for feature in DEFAULT_CLUSTER_FEATURES if feature in pca_columns]
    return summary_features or pca_columns[: min(4, len(pca_columns))]

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

    for _, summary_row in cluster_summary_df.iterrows():
        cluster_id = int(summary_row["cluster"])
        cluster_df = clustered_df[clustered_df["cluster"] == cluster_id].copy()
        canonical_row, anomaly_row, random_rows, lens_rows = cluster_visual_rows(
            cluster_df,
            summary_features,
            cluster_id,
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
                "canonical": canonical,
                "anomalous": anomalous,
            }
        )

        lens_captions = []
        for row in lens_rows:
            lens_grade = row.get("lens_grade", "")
            if pd.isna(lens_grade) or not str(lens_grade).strip():
                lens_captions.append("Grade ?")
            else:
                lens_captions.append(f"Grade {str(lens_grade).strip()}")

        visual_clusters.append(
            {
                "cluster_id": cluster_id,
                "n_objects": int(summary_row["n_objects"]),
                "n_lenses": int(summary_row["n_lenses"]),
                "lens_rate": float(summary_row["lens_rate"]),
                "canonical_row": canonical_row,
                "anomaly_row": anomaly_row,
                "random_rows": random_rows,
                "lens_rows": lens_rows,
                "lens_captions": lens_captions,
            }
        )

    return {
        "cluster_download_df": pd.DataFrame(cluster_download_rows),
        "histogram_features": histogram_features,
        "clusters": visual_clusters,
    }

def render_cluster_visual_summary_view_model(
    view_model: dict,
    clustered_df: pd.DataFrame,
) -> None:
    for cluster_model in view_model["clusters"]:
        cluster_id = int(cluster_model["cluster_id"])

        with st.container(border=True):
            stats_cols = st.columns([1, 1, 1, 1])
            stats_cols[0].metric("Cluster", cluster_id)
            stats_cols[1].metric(
                "Objects",
                format_thousands_dot(cluster_model["n_objects"]),
            )
            stats_cols[2].metric(
                "Lenses",
                format_thousands_dot(cluster_model["n_lenses"]),
            )
            stats_cols[3].metric(
                "Density",
                f"{format_decimal_comma(cluster_model['lens_rate'] * 100, 2)}%",
            )

            image_cols = st.columns([2, 3, 5])
            with image_cols[0]:
                show_thumbnail_group(
                    "Canonical / anomalous",
                    [cluster_model["canonical_row"], cluster_model["anomaly_row"]],
                    ["Canonical", "Anomalous"],
                )
            with image_cols[1]:
                show_thumbnail_group(
                    "Random",
                    cluster_model["random_rows"],
                    [
                        f"Random {index + 1}"
                        for index in range(len(cluster_model["random_rows"]))
                    ],
                )
            with image_cols[2]:
                show_thumbnail_group(
                    "Labelled lens candidates in the cluster",
                    cluster_model["lens_rows"],
                    cluster_model["lens_captions"],
                    prefer_lens_image=True,
                )
            if cluster_model["n_lenses"] > 0:
                cluster_df = clustered_df[clustered_df["cluster"] == cluster_id].copy()
                render_cluster_histograms(
                    cluster_id,
                    cluster_df,
                    view_model["histogram_features"],
                )

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
    key_digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]
    button_key = f"detect_arc_like_structures_{label}_{key_digest}"
    active_key = f"{button_key}_active"
    active = bool(st.session_state.get(active_key, False))

    if active:
        try:
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


def show_object_details(row: pd.Series, selected_features: list[str]) -> None:
    st.subheader("Selected object")
    show_lens_status(row)

    details = {
        "id_str": row.get("id_str", ""),
        "object_id": row.get("object_id", ""),
        "cluster": row.get("cluster", ""),
        "is_lens": bool(row.get("is_lens", False)),
        "is_canonical": bool(row.get("is_canonical", False)),
        "is_anomaly": bool(row.get("is_anomaly", False)),
        "lens_grade": row.get("lens_grade", ""),
        "dist_to_cluster_centroid": row.get("dist_to_cluster_centroid", np.nan),
        "umap_1": row.get("umap_1", np.nan),
        "umap_2": row.get("umap_2", np.nan),
    }
    st.dataframe(pd.DataFrame([details]), use_container_width=True, hide_index=True)

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

def show_morphology_catalogue_row(row: pd.Series) -> None:
    morphology_df = load_morphology_object(MORPH_PATH, str(row.get("object_id", "")))
    if not morphology_df.empty:
        st.markdown("**Morphology catalogue features**")
        morph_display = pd.DataFrame(
            table_rows_with_coordinate_formats(morphology_df.iloc[0].dropna().to_dict())
        )
        st.dataframe(morph_display, use_container_width=True, hide_index=True)

def show_selected_pca_components(row: pd.Series, selected_features: list[str]) -> None:
    st.markdown("**Selected PCA components**")
    st.dataframe(
        pd.DataFrame(
            [{"feature": feature, "value": row.get(feature)} for feature in selected_features]
        ),
        use_container_width=True,
        hide_index=True,
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
        f"<span>{html.escape(caption)} </span>"
        f'<a href="{aladin_link}" target="_blank" '
        'rel="noopener noreferrer">[View in Aladin]</a>'
    )

def aladin_caption_for_object(caption: str, object_id: object) -> str | None:
    morphology_df = load_morphology_object(MORPH_PATH, str(object_id))
    if morphology_df.empty:
        return None
    morphology_row = morphology_df.iloc[0]
    try:
        return aladin_caption(
            caption,
            float(morphology_row["right_ascension"]),
            float(morphology_row["declination"]),
        )
    except (KeyError, TypeError, ValueError):
        return None

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
        lens_candidate_summary = cached_search.get("lens_candidate_summary", {})
    else:
        started_at = time.perf_counter()
        result = fetch_euclid_object_summary(object_id)
        morphology_df = load_morphology_object(MORPH_PATH, str(result["object_id"]))
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
            "lens_candidate_summary": lens_candidate_summary,
        }

    object_summary = {**result["object_summary"], **lens_candidate_summary}
    mosaic_summary = result["mosaic_summary"]

    with st.container(border=True):
        st.subheader("Object search")
        ra_degrees = float(object_summary["right_ascension"])
        dec_degrees = float(object_summary["declination"])
        metric_cols = st.columns([2.4, 1, 1, 1])
        metric_cols[0].metric("object_id", str(result["object_id"]))
        metric_cols[1].metric("RA", f"{ra_degrees:.6f}")
        metric_cols[2].metric("Dec", f"{dec_degrees:.6f}")
        metric_cols[3].metric("Tile", str(mosaic_summary.get("tile_index", "")))

        image_col, summary_col = st.columns([1, 1])
        with image_col:
            cutout_path = result.get("cutout_path")
            if cutout_path:
                aladin_link = aladin_url(ra_degrees, dec_degrees)
                show_detectable_image(
                    str(cutout_path),
                    "Euclid VIS cutout",
                    "search_cutout",
                    caption_markdown=(
                        '<span>Euclid VIS cutout </span>'
                        f'<a href="{aladin_link}" target="_blank" '
                        'rel="noopener noreferrer">[View in Aladin]</a>'
                    ),
                )
            else:
                st.info(
                    "No precomputed JPEG cutout was found for this object. "
                    "No FITS file was downloaded."
                )

            st.markdown("**Mosaic summary**")
            mosaic_display = pd.DataFrame(
                [{"field": field, "value": value} for field, value in mosaic_summary.items()]
            )
            st.dataframe(mosaic_display, use_container_width=True, hide_index=True)
        with summary_col:
            st.markdown("**Object summary**")
            object_display = pd.DataFrame(
                object_summary_display_rows(object_summary)
            )
            st.dataframe(object_display, use_container_width=True, hide_index=True)

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
                st.dataframe(morphology_display, use_container_width=True, hide_index=True)

        object_download_df = object_search_download_df(
            object_summary,
            morphology_df,
            mosaic_summary,
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
    st.session_state["cluster_summary_expanded"] = False

def collapse_cluster_summary() -> None:
    st.session_state["cluster_summary_expanded"] = False
