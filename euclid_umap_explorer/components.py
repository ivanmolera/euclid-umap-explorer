from __future__ import annotations

import base64
import hashlib
import html
import time

import numpy as np
import pandas as pd
import streamlit as st

from .analysis import (
    add_cluster_extreme_roles,
    cluster_lens_grades,
    format_pca_filter,
    lens_grade_sort_key,
)
from .catalogs import load_morphology_object
from .config import (
    CUTOUT_BASE,
    DEFAULT_BIRCH_BATCH_SIZE,
    DEFAULT_BIRCH_BRANCHING_FACTOR,
    DEFAULT_BIRCH_THRESHOLD,
    DEFAULT_CLUSTER_FEATURES,
    DEFAULT_LENS_GRADES,
    LENS_GRADE_OPTIONS,
    LENS_IMG_BASE,
    LENS_PATH,
    MORPH_PATH,
    PARQUET_PATH,
    PCA_FILTER_OPERATORS,
    SUMMARY_DISTPLOT_MAX_POINTS_PER_GROUP,
    SUMMARY_HISTOGRAM_BINS,
    SUMMARY_HISTOGRAM_FEATURE_LIMIT,
    SUMMARY_LENS_OBJECTS,
    SUMMARY_RANDOM_OBJECTS,
    SUMMARY_THUMBNAIL_WIDTH,
)
from .euclid_search import fetch_euclid_object_summary
from .images import (
    lens_image_path,
    morphology_cutout_path,
    object_image_path,
    show_image,
    thumbnail_image_src,
)
from .runtime import log_app_event
from .storage import path_exists

def inject_plot_cursor_css() -> None:
    st.markdown(
        """
        <style>
        .js-plotly-plot .plotly .draglayer .drag,
        .js-plotly-plot .plotly .draglayer .nsewdrag,
        .js-plotly-plot .plotly .cursor-crosshair,
        .js-plotly-plot .plotly .cursor-move,
        .js-plotly-plot .plotly .cursor-pointer {
            cursor: default !important;
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
            border-bottom: 1px dotted rgba(250, 250, 250, 0.72);
            cursor: help !important;
            display: inline-block;
            position: relative;
        }
        .concept-popover {
            background: #171a22;
            border: 1px solid rgba(255, 255, 255, 0.18);
            border-radius: 8px;
            box-shadow: 0 12px 32px rgba(0, 0, 0, 0.34);
            color: #f8fafc;
            display: none;
            font-size: 0.82rem;
            font-weight: 400;
            left: 0;
            line-height: 1.35;
            min-width: 240px;
            padding: 0.65rem 0.75rem;
            position: absolute;
            top: 1.45rem;
            z-index: 1000;
        }
        .concept-help:hover .concept-popover,
        .concept-help:focus .concept-popover {
            display: block;
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

def cluster_visual_image_paths(
    cluster_df: pd.DataFrame,
    summary_features: list[str],
    cluster_id: int,
) -> list[str]:
    canonical_row, anomaly_row, random_rows, lens_rows = cluster_visual_rows(
        cluster_df,
        summary_features,
        cluster_id,
    )

    paths = []
    for row in [canonical_row, anomaly_row, *random_rows]:
        if row is None:
            continue
        path = object_image_path(row)
        if path is not None:
            paths.append(path)

    for row in lens_rows:
        path = object_image_path(row, prefer_lens_image=True)
        if path is not None:
            paths.append(path)

    return list(dict.fromkeys(paths))

def is_default_birch_configuration(params: dict) -> bool:
    return (
        tuple(cluster_lens_grades(params)) == tuple(DEFAULT_LENS_GRADES)
        and float(params["threshold"]) == DEFAULT_BIRCH_THRESHOLD
        and int(params["branching_factor"]) == DEFAULT_BIRCH_BRANCHING_FACTOR
        and int(params["batch_size"]) == DEFAULT_BIRCH_BATCH_SIZE
    )

def warm_cluster_visual_image_cache(
    clustered_df: pd.DataFrame,
    cluster_summary_df: pd.DataFrame,
    pca_columns: list[str],
    selected_features: list[str],
) -> None:
    summary_features = [
        feature for feature in selected_features if feature in pca_columns
    ] or [feature for feature in DEFAULT_CLUSTER_FEATURES if feature in pca_columns]
    summary_features = summary_features or pca_columns[: min(4, len(pca_columns))]

    warmed = 0
    started_at = time.perf_counter()
    for _, summary_row in cluster_summary_df.iterrows():
        cluster_id = int(summary_row["cluster"])
        cluster_df = clustered_df[clustered_df["cluster"] == cluster_id].copy()
        for path in cluster_visual_image_paths(cluster_df, summary_features, cluster_id):
            try:
                thumbnail_image_src(path)
                warmed += 1
            except Exception:
                continue

    log_app_event(
        "cluster_visual_image_cache_warmed",
        duration_seconds=round(time.perf_counter() - started_at, 3),
        n_images=int(warmed),
    )

def default_cluster_visual_cache_key(
    params: dict,
    selected_features: list[str],
    cluster_summary_df: pd.DataFrame,
) -> tuple:
    return (
        PARQUET_PATH,
        LENS_PATH,
        CUTOUT_BASE,
        LENS_IMG_BASE,
        cluster_lens_grades(params),
        float(params["threshold"]),
        int(params["branching_factor"]),
        int(params["batch_size"]),
        tuple(selected_features),
        tuple(cluster_summary_df["cluster"].astype(int).tolist()),
    )

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
            color: rgba(250, 250, 250, 0.72);
            font-size: 0.82rem;
            margin-top: 0.2rem;
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

def build_cluster_distplot_figure(
    cluster_df: pd.DataFrame,
    feature: str,
    feature_index: int,
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
    fig.update_xaxes(showgrid=False, zeroline=False)
    return fig

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
    button_label = (
        "Update PCA histograms"
        if st.session_state.get(state_key)
        else "Compute PCA histograms"
    )
    if st.button(button_label, key=f"cluster_histograms_button_{cluster_id}"):
        st.session_state[state_key] = True

    if not st.session_state.get(state_key):
        return

    if n_non_lenses == 0:
        st.info("This cluster does not contain non-lens objects for comparison.")
        return

    chart_columns = st.columns(2)
    for index, feature in enumerate(summary_features):
        fig = build_cluster_distplot_figure(cluster_df, feature, index)
        if fig is None:
            continue
        with chart_columns[index % 2]:
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displaylogo": False, "responsive": True},
                key=f"cluster_distplot_chart_{cluster_id}_{feature}",
            )

def render_thumbnail_group_title(title: str) -> None:
    if title != "Canonical / anomalous":
        st.caption(title)
        return

    st.markdown(
        """
        <div style="color: rgba(250, 250, 250, 0.72); font-size: 0.82rem; margin-bottom: 0.35rem;">
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

def render_cluster_visual_summary(
    clustered_df: pd.DataFrame,
    cluster_summary_df: pd.DataFrame,
    pca_columns: list[str],
    selected_features: list[str],
) -> None:
    summary_features = [
        feature for feature in selected_features if feature in pca_columns
    ] or [feature for feature in DEFAULT_CLUSTER_FEATURES if feature in pca_columns]
    summary_features = summary_features or pca_columns[: min(4, len(pca_columns))]
    histogram_features = summary_features[:SUMMARY_HISTOGRAM_FEATURE_LIMIT]

    for _, summary_row in cluster_summary_df.iterrows():
        cluster_id = int(summary_row["cluster"])
        cluster_df = clustered_df[clustered_df["cluster"] == cluster_id].copy()
        canonical_row, anomaly_row, random_rows, lens_rows = cluster_visual_rows(
            cluster_df,
            summary_features,
            cluster_id,
        )

        with st.container(border=True):
            stats_cols = st.columns([1, 1, 1, 1])
            stats_cols[0].metric("Cluster", cluster_id)
            stats_cols[1].metric("Objects", f"{int(summary_row['n_objects']):,}")
            stats_cols[2].metric("Lenses", f"{int(summary_row['n_lenses']):,}")
            stats_cols[3].metric("Density", f"{summary_row['lens_rate'] * 100:.3f}%")

            image_cols = st.columns([2, 3, 5])
            with image_cols[0]:
                show_thumbnail_group(
                    "Canonical / anomalous",
                    [canonical_row, anomaly_row],
                    ["Canonical", "Anomalous"],
                )
            with image_cols[1]:
                show_thumbnail_group(
                    "Random",
                    random_rows,
                    [f"Random {index + 1}" for index in range(len(random_rows))],
                )
            with image_cols[2]:
                lens_captions = []
                for row in lens_rows:
                    lens_grade = row.get("lens_grade", "")
                    if pd.isna(lens_grade) or not str(lens_grade).strip():
                        lens_captions.append("Grade ?")
                    else:
                        lens_captions.append(f"Grade {str(lens_grade).strip()}")
                show_thumbnail_group(
                    "Labelled lens candidates in the cluster",
                    lens_rows,
                    lens_captions,
                    prefer_lens_image=True,
                )
            render_cluster_histograms(cluster_id, cluster_df, histogram_features)

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

    cutout_path = morphology_cutout_path(row.get("id_str"), row.get("object_id"))
    lens_path = lens_image_path(row.get("lens_id_str"))
    if lens_path is None and bool(row.get("is_lens", False)):
        lens_path = lens_image_path(row.get("id_str"))

    if cutout_path is None and lens_path is None:
        st.info("No associated image was found in the configured paths.")
    else:
        if cutout_path is not None:
            show_image(cutout_path, "Morphology cutout")
        if lens_path is not None:
            show_image(lens_path, "Strong-lens image")

def show_morphology_catalogue_row(row: pd.Series) -> None:
    morphology_df = load_morphology_object(MORPH_PATH, str(row.get("object_id", "")))
    if not morphology_df.empty:
        st.markdown("**Morphology catalogue features**")
        morph_display = morphology_df.iloc[0].dropna().astype(str).reset_index()
        morph_display = (
            morph_display.rename(columns={"index": "field", morph_display.columns[-1]: "value"})
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

def render_euclid_object_search(object_id: str) -> None:
    started_at = time.perf_counter()
    with st.spinner(f"Searching Euclid object {object_id}..."):
        result = fetch_euclid_object_summary(object_id)

    object_summary = result["object_summary"]
    mosaic_summary = result["mosaic_summary"]
    morphology_df = load_morphology_object(MORPH_PATH, str(result["object_id"]))
    log_app_event(
        "object_search_completed",
        duration_seconds=round(time.perf_counter() - started_at, 3),
        has_precomputed_cutout=bool(result.get("cutout_path")),
        has_morphology_row=not morphology_df.empty,
        instrument=str(mosaic_summary.get("instrument_name", "")),
        tile_index=str(mosaic_summary.get("tile_index", "")),
    )

    with st.container(border=True):
        st.subheader("Object search")
        metric_cols = st.columns([2.4, 1, 1, 1])
        metric_cols[0].metric("object_id", str(result["object_id"]))
        metric_cols[1].metric("RA", f"{float(object_summary['right_ascension']):.6f}")
        metric_cols[2].metric("Dec", f"{float(object_summary['declination']):.6f}")
        metric_cols[3].metric("Tile", str(mosaic_summary.get("tile_index", "")))

        image_col, summary_col = st.columns([1, 1])
        with image_col:
            cutout_path = result.get("cutout_path")
            if cutout_path:
                show_image(
                    str(cutout_path),
                    "Euclid VIS cutout",
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
                [{"field": field, "value": value} for field, value in object_summary.items()]
            )
            st.dataframe(object_display, use_container_width=True, hide_index=True)

            st.markdown("**Morphology catalogue features**")
            if morphology_df.empty:
                st.info("No morphology catalogue row was found for this object_id.")
            else:
                morphology_display = morphology_df.iloc[0].dropna().astype(str).reset_index()
                morphology_display = morphology_display.rename(
                    columns={
                        "index": "field",
                        morphology_display.columns[-1]: "value",
                    }
                )
                st.dataframe(morphology_display, use_container_width=True, hide_index=True)

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
    st.session_state["cluster_summary_expanded"] = True

def collapse_cluster_summary() -> None:
    st.session_state["cluster_summary_expanded"] = False
