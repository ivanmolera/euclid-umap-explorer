from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from .analysis import (
    add_cluster_extreme_roles,
    apply_pca_filters,
    build_cluster_summary,
    cluster_lens_grades,
    default_cluster_option_index,
    format_cluster_option,
    format_pca_filter,
    normalize_pca_filters,
    sample_for_display,
)
from .birch import run_birch_clustering
from .catalogs import normalize_lens_grades
from .components import (
    collapse_cluster_summary,
    default_cluster_visual_cache_key,
    inject_plot_cursor_css,
    is_default_birch_configuration,
    render_back_to_top_control,
    render_cluster_visual_summary,
    render_euclid_object_search,
    render_help_label,
    render_pca_filter_controls,
    request_clustering,
    show_morphology_catalogue_row,
    show_object_details,
    show_selected_pca_components,
    warm_cluster_visual_image_cache,
)
from .config import (
    APP_TITLE,
    APP_VERSION,
    DEFAULT_BIRCH_BATCH_SIZE,
    DEFAULT_BIRCH_BRANCHING_FACTOR,
    DEFAULT_BIRCH_THRESHOLD,
    DEFAULT_CLUSTER_FEATURES,
    DEFAULT_LENS_GRADES,
    EUCLID_FAVICON_PATH,
    EUCLID_LOGO_PATH,
    LENS_GRADE_HELP,
    LENS_GRADE_OPTIONS,
    LENS_PATH,
    MAX_ALGORITHM_SECONDS,
    PARAMETER_HELP,
    PARQUET_PATH,
)
from .euclid_search import is_valid_search_object_id, selected_point_index
from .runtime import AlgorithmTimeoutError, format_duration, log_app_event
from .storage import path_exists, prepare_catalog_cache
from .umap import build_umap_signature, compute_umap_embedding


def request_umap_computation() -> None:
    collapse_cluster_summary()
    st.session_state["umap_requested"] = True
    st.session_state["umap_running"] = True


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon=str(EUCLID_FAVICON_PATH), layout="wide")
    inject_plot_cursor_css()
    render_back_to_top_control()
    st.title(APP_TITLE)
    loading_placeholder = st.empty()
    loading_placeholder.info("Loading application...")

    required = [PARQUET_PATH, LENS_PATH]
    with st.spinner("Loading application..."):
        missing = [path for path in required if not path_exists(path)]
    if missing:
        loading_placeholder.empty()
        st.error(
            "Required files were not found. Check the "
            "MORPH_PATH, PARQUET_PATH, LENS_PATH, "
            "CUTOUT_BASE and LENS_IMG_BASE environment variables."
        )
        st.code("\n".join(missing), language="text")
        st.stop()
    loading_placeholder.empty()

    with st.sidebar:
        logo_left, logo_center, logo_right = st.columns([1, 2, 1])
        with logo_center:
            st.image(str(EUCLID_LOGO_PATH), use_container_width=True)
        st.caption(f"Version {APP_VERSION}")

        st.header("Data")
        st.markdown(
            """
This analysis uses Euclid Q1 catalogue products available at:

- [The Strong Lensing Discovery Engine](https://zenodo.org/records/15025832)
- [First visual morphology catalogue](https://zenodo.org/records/15106473)
            """
        )
        search_input_col, search_button_col = st.columns([3, 1])
        with search_input_col:
            object_id_search_value = st.text_input(
                "object_id",
                placeholder="object_id",
                label_visibility="collapsed",
                key="object_id_search_value",
            )
        search_object_id = object_id_search_value.strip()
        search_submitted = search_button_col.button(
            "Search",
            disabled=not is_valid_search_object_id(search_object_id),
        )
        if search_submitted:
            st.session_state["euclid_search_object_id"] = search_object_id

        st.header("Lens candidates")
        render_help_label("Lens grades", LENS_GRADE_HELP)
        selected_lens_grades = st.multiselect(
            "Lens grades",
            LENS_GRADE_OPTIONS,
            default=DEFAULT_LENS_GRADES,
            label_visibility="collapsed",
        )
        selected_lens_grades = normalize_lens_grades(selected_lens_grades)

        st.header("BIRCH clustering")
        birch_expanded = not st.session_state.get("cluster_ready") and not st.session_state.get(
            "cluster_requested"
        )
        with st.expander("BIRCH parameters", expanded=birch_expanded):
            st.caption("All available PCA components are used for the initial clustering")
            render_help_label("threshold", PARAMETER_HELP["threshold"])
            threshold = st.number_input(
                "threshold",
                min_value=0.1,
                value=DEFAULT_BIRCH_THRESHOLD,
                step=0.1,
                label_visibility="collapsed",
            )
            render_help_label("branching_factor", PARAMETER_HELP["branching_factor"])
            branching_factor = st.number_input(
                "branching_factor",
                min_value=2,
                value=DEFAULT_BIRCH_BRANCHING_FACTOR,
                step=1,
                label_visibility="collapsed",
            )
            render_help_label("batch_size", PARAMETER_HELP["batch_size"])
            batch_size = st.number_input(
                "batch_size",
                min_value=1_000,
                max_value=250_000,
                value=DEFAULT_BIRCH_BATCH_SIZE,
                step=1_000,
                label_visibility="collapsed",
            )
            run_clustering = st.button(
                "Run clustering",
                type="primary",
                on_click=request_clustering,
            )

    searched_object_id = st.session_state.get("euclid_search_object_id")
    if searched_object_id:
        try:
            render_euclid_object_search(searched_object_id)
        except ValueError as exc:
            log_app_event("object_search_failed", error_type=type(exc).__name__)
            st.warning(str(exc))
        except ImportError as exc:
            log_app_event("object_search_failed", error_type=type(exc).__name__)
            st.error("The Euclid object search dependencies are not installed.")
            st.exception(exc)
        except Exception as exc:
            log_app_event("object_search_failed", error_type=type(exc).__name__)
            st.error("Could not retrieve the Euclid cutout for this object_id.")
            st.exception(exc)

    clustering_requested = st.session_state.pop("cluster_requested", False)
    if run_clustering or clustering_requested:
        if not selected_lens_grades:
            st.warning("Select at least one lens grade before clustering.")
            st.stop()

        st.session_state["cluster_ready"] = True
        st.session_state["cluster_params"] = {
            "lens_grades": selected_lens_grades,
            "threshold": threshold,
            "branching_factor": int(branching_factor),
            "batch_size": int(batch_size),
        }
        st.session_state["cluster_summary_expanded"] = True
        log_app_event(
            "birch_clustering_requested",
            selected_grades=list(selected_lens_grades),
            threshold=float(threshold),
            branching_factor=int(branching_factor),
            batch_size=int(batch_size),
        )

    if not st.session_state.get("cluster_ready"):
        st.info("Click **Run clustering** button to clusterize data.")
        st.stop()

    # Only individual catalogues are cached. Image folders are read on demand.
    prepare_catalog_cache([PARQUET_PATH, LENS_PATH])

    params = st.session_state["cluster_params"]
    lens_grades = cluster_lens_grades(params)
    try:
        clustered_df, pca_columns = run_birch_clustering(
            PARQUET_PATH,
            LENS_PATH,
            lens_grades,
            float(params["threshold"]),
            int(params["branching_factor"]),
            int(params["batch_size"]),
        )
    except AlgorithmTimeoutError as exc:
        log_app_event("birch_clustering_timeout", timeout_seconds=MAX_ALGORITHM_SECONDS)
        st.error(
            "BIRCH clustering was cancelled because it exceeded the "
            f"{format_duration(MAX_ALGORITHM_SECONDS)} execution limit. "
            "Try a less expensive configuration before running it again."
        )
        st.exception(exc)
        st.stop()
    except TimeoutError as exc:
        st.error(
            "The data source timed out while reading a catalogue. "
            "If you are using a synchronized drive, make the files available offline or copy them to local cache first."
        )
        st.exception(exc)
        st.stop()
    except OSError as exc:
        st.error(
            "Could not read a catalogue from the configured paths. "
            "If you are using a synchronized drive, check that the files are available offline."
        )
        st.exception(exc)
        st.stop()

    cluster_summary_df = build_cluster_summary(clustered_df)
    cluster_summary_df["option"] = cluster_summary_df.apply(format_cluster_option, axis=1)

    left_metric, middle_metric, right_metric = st.columns(3)
    left_metric.metric("Clustered objects", f"{len(clustered_df):,}")
    middle_metric.metric("Clusters", f"{clustered_df['cluster'].nunique():,}")
    right_metric.metric("Lenses", f"{int(clustered_df['is_lens'].sum()):,}")
    st.caption(f"Lens grades used: {', '.join(lens_grades)}")

    with st.sidebar:
        st.header("PCA components")
        default_features = [
            feature for feature in DEFAULT_CLUSTER_FEATURES if feature in pca_columns
        ] or pca_columns[: min(4, len(pca_columns))]
        selected_features = st.multiselect(
            "PCA components",
            pca_columns,
            default=default_features,
        )
        raw_pca_filters = render_pca_filter_controls(pca_columns)

        st.header("UMAP")
        selected_option = st.selectbox(
            "Cluster",
            cluster_summary_df["option"].tolist(),
            index=default_cluster_option_index(cluster_summary_df),
        )
        selected_cluster = int(
            cluster_summary_df.loc[
                cluster_summary_df["option"] == selected_option,
                "cluster",
            ].iloc[0]
        )

        with st.expander("UMAP parameters", expanded=True):
            render_help_label("n_neighbors", PARAMETER_HELP["n_neighbors"])
            n_neighbors = st.slider(
                "n_neighbors",
                2,
                100,
                25,
                label_visibility="collapsed",
            )
            render_help_label("min_dist", PARAMETER_HELP["min_dist"])
            min_dist = st.slider(
                "min_dist",
                0.0,
                1.0,
                0.15,
                step=0.01,
                label_visibility="collapsed",
            )
            render_help_label("Maximum objects", PARAMETER_HELP["Maximum objects"])
            max_objects = st.slider(
                "Maximum objects",
                100,
                100_000,
                20_000,
                step=100,
                label_visibility="collapsed",
            )

    if not selected_features:
        st.warning("Select at least one PCA component to build UMAP.")
        st.stop()
    pca_filters = normalize_pca_filters(raw_pca_filters, pca_columns)

    with st.expander(
        "Clustering summary",
        expanded=st.session_state.get("cluster_summary_expanded", False),
    ):
        st.markdown(
            f"""
            <div style="display: flex; align-items: baseline; gap: 0.45rem; margin-bottom: 0.75rem;">
                <span style="color: rgba(250, 250, 250, 0.72); font-size: 0.95rem;">
                    Execution time:
                </span>
                <span style="font-size: 1.8rem; font-weight: 600; line-height: 1;">
                    {format_duration(clustered_df.attrs.get("processing_seconds"))}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        summary_display = cluster_summary_df.copy()
        summary_display["lens_rate"] = (summary_display["lens_rate"] * 100).round(3)
        st.dataframe(
            summary_display[["cluster", "n_objects", "n_lenses", "lens_rate"]],
            use_container_width=True,
            hide_index=True,
        )
        if is_default_birch_configuration(params):
            cache_key = default_cluster_visual_cache_key(
                params,
                selected_features,
                cluster_summary_df,
            )
            if st.session_state.get("cluster_visual_image_cache_key") != cache_key:
                with st.spinner("Preparing cluster summary images..."):
                    warm_cluster_visual_image_cache(
                        clustered_df,
                        cluster_summary_df,
                        pca_columns,
                        selected_features,
                    )
                st.session_state["cluster_visual_image_cache_key"] = cache_key
        render_cluster_visual_summary(
            clustered_df,
            cluster_summary_df,
            pca_columns,
            selected_features,
        )

    cluster_df = clustered_df[clustered_df["cluster"] == selected_cluster].copy()
    filtered_cluster_df = apply_pca_filters(cluster_df, pca_filters)
    if pca_filters:
        st.caption(
            "Active PCA filters: "
            + "; ".join(format_pca_filter(pca_filter) for pca_filter in pca_filters)
        )

    umap_signature = build_umap_signature(
        selected_cluster=selected_cluster,
        selected_features=selected_features,
        pca_filters=pca_filters,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        max_objects=int(max_objects),
        cluster_params=params,
    )
    stored_signature = st.session_state.get("umap_signature")
    needs_recalculation = stored_signature != umap_signature
    umap_running = bool(st.session_state.get("umap_running", False))

    button_label = "Compute UMAP" if stored_signature is None else "Recompute UMAP"
    umap_button_disabled = (
        umap_running
        or (not selected_features)
        or len(filtered_cluster_df) < 3
        or (not needs_recalculation and "umap_embedding_df" in st.session_state)
    )
    st.sidebar.button(
        button_label,
        type="primary" if needs_recalculation else "secondary",
        disabled=umap_button_disabled,
        on_click=request_umap_computation,
    )
    recalculate_umap = bool(st.session_state.get("umap_requested", False))

    if len(filtered_cluster_df) < 3:
        st.session_state["umap_running"] = False
        st.session_state["umap_requested"] = False
        st.warning("At least 3 objects must remain after PCA filters to compute UMAP.")
        st.stop()

    if recalculate_umap:
        try:
            filtered_cluster_df = add_cluster_extreme_roles(filtered_cluster_df, selected_features)
            display_df = sample_for_display(filtered_cluster_df, int(max_objects))
            log_app_event(
                "umap_requested",
                cluster=int(selected_cluster),
                cluster_objects=int(len(cluster_df)),
                filtered_objects=int(len(filtered_cluster_df)),
                display_objects=int(len(display_df)),
                n_features=int(len(selected_features)),
                n_pca_filters=int(len(pca_filters)),
                n_neighbors=int(n_neighbors),
                min_dist=float(min_dist),
                max_objects=int(max_objects),
            )

            if len(display_df) < 3:
                st.warning("At least 3 objects are required to compute UMAP.")
                st.stop()

            embedding_df = compute_umap_embedding(
                display_df,
                selected_features,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
            )
        except AlgorithmTimeoutError as exc:
            log_app_event(
                "umap_timeout",
                timeout_seconds=MAX_ALGORITHM_SECONDS,
                cluster=int(selected_cluster),
                display_objects=int(len(display_df)),
                n_features=int(len(selected_features)),
            )
            st.error(
                "UMAP was cancelled because it exceeded the "
                f"{format_duration(MAX_ALGORITHM_SECONDS)} execution limit. "
                "Try reducing the maximum number of objects, using fewer PCA components, "
                "or adjusting the UMAP parameters before running it again."
            )
            st.exception(exc)
            st.stop()
        finally:
            st.session_state["umap_running"] = False
            st.session_state["umap_requested"] = False

        if embedding_df.empty:
            st.warning("No objects remain with complete values for the selected components.")
            st.stop()

        embedding_df = embedding_df.reset_index(drop=True)
        embedding_df["point_index"] = embedding_df.index
        st.session_state["umap_embedding_df"] = embedding_df
        st.session_state["umap_signature"] = umap_signature
        needs_recalculation = False

    if needs_recalculation or "umap_embedding_df" not in st.session_state:
        st.info("Click **Compute UMAP** or **Recompute UMAP** to update the visualization.")
        st.stop()

    embedding_df = st.session_state["umap_embedding_df"]

    (
        cluster_left,
        cluster_filtered,
        cluster_middle,
        cluster_right,
        cluster_fourth,
    ) = st.columns(5)
    cluster_left.metric("Cluster objects", f"{len(cluster_df):,}")
    cluster_filtered.metric("After filters", f"{len(filtered_cluster_df):,}")
    cluster_middle.metric("Objects in UMAP", f"{len(embedding_df):,}")
    cluster_right.metric("Lenses in UMAP", f"{int(embedding_df['is_lens'].sum()):,}")
    cluster_fourth.metric("Extremes", "2")

    embedding_df = embedding_df.copy()
    if "lens_grade" in embedding_df.columns:
        lens_grade_marker = (
            embedding_df["lens_grade"]
            .astype("string")
            .str.strip()
            .str.upper()
            .str[:1]
        )
        lens_grade_marker = lens_grade_marker.where(
            lens_grade_marker.isin(LENS_GRADE_OPTIONS),
            "?",
        )
    else:
        lens_grade_marker = pd.Series("?", index=embedding_df.index)
    embedding_df["lens_grade_marker"] = np.where(
        embedding_df["is_lens"],
        lens_grade_marker.fillna("?"),
        "",
    )

    hover_columns = [
        column
        for column in (
            "id_str",
            "object_id",
            "cluster",
            "point_role",
            "is_lens",
            "lens_grade",
            "dist_to_cluster_centroid",
        )
        if column in embedding_df.columns
    ]

    import plotly.express as px

    fig = px.scatter(
        embedding_df,
        x="umap_1",
        y="umap_2",
        color="point_role",
        symbol="point_role",
        custom_data=["point_index"],
        hover_data=hover_columns,
        color_discrete_map={
            "Unknown": "#4c78a8",
            "Lens candidate": "#d62728",
            "Canonical": "#2ca02c",
            "Anomaly": "#111111",
        },
        symbol_map={
            "Unknown": "circle",
            "Lens candidate": "circle",
            "Canonical": "diamond",
            "Anomaly": "x",
        },
        category_orders={
            "point_role": ["Unknown", "Lens candidate", "Canonical", "Anomaly"],
        },
        labels={"umap_1": "UMAP 1", "umap_2": "UMAP 2", "point_role": "Type"},
        height=680,
    )
    fig.update_traces(marker={"size": 7, "opacity": 0.72})
    fig.update_traces(
        marker={"size": 17, "opacity": 0.98, "line": {"width": 1.5, "color": "white"}},
        selector={"name": "Lens candidate"},
    )
    fig.update_traces(
        marker={"size": 14, "opacity": 1.0, "line": {"width": 2, "color": "white"}},
        selector={"name": "Canonical"},
    )
    fig.update_traces(
        marker={"size": 14, "opacity": 1.0, "line": {"width": 2, "color": "#ffcc00"}},
        selector={"name": "Anomaly"},
    )
    for trace in fig.data:
        opacity = getattr(trace.marker, "opacity", None) or 1.0
        trace.selected = {"marker": {"opacity": opacity}}
        trace.unselected = {"marker": {"opacity": opacity}}

    for lens_row in embedding_df[embedding_df["lens_grade_marker"] != ""].itertuples():
        fig.add_annotation(
            x=lens_row.umap_1,
            y=lens_row.umap_2,
            text=lens_row.lens_grade_marker,
            showarrow=False,
            font={"size": 10, "color": "white", "family": "Arial Black"},
            xanchor="center",
            yanchor="middle",
            captureevents=False,
        )

    fig.update_layout(
        title=f"Cluster {selected_cluster} | UMAP",
        legend_title_text="Object",
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        clickmode="event+select",
        dragmode="zoom",
        uirevision=st.session_state.get("umap_signature"),
    )

    with st.expander("UMAP summary", expanded=True):
        st.markdown(
            f"""
            <div style="display: flex; align-items: baseline; gap: 0.45rem; margin-bottom: 0.75rem;">
                <span style="color: rgba(250, 250, 250, 0.72); font-size: 0.95rem;">
                    Execution time:
                </span>
                <span style="font-size: 1.8rem; font-weight: 600; line-height: 1;">
                    {format_duration(embedding_df.attrs.get("processing_seconds"))}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        plot_col, detail_col = st.columns([2, 1])
        with plot_col:
            event = st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "displaylogo": False,
                    "scrollZoom": True,
                    "doubleClick": "reset",
                },
                on_select="rerun",
                selection_mode="points",
                key="umap_selection",
            )

        selected_index = selected_point_index(event)
        selected_row = (
            embedding_df.loc[selected_index]
            if selected_index is not None
            else None
        )
        with detail_col:
            if selected_row is None:
                st.info("Select a point on the map to view its details and image.")
            else:
                show_object_details(selected_row, selected_features)

        if selected_row is not None:
            morphology_col, pca_col = st.columns([1, 1])
            with morphology_col:
                show_morphology_catalogue_row(selected_row)
            with pca_col:
                show_selected_pca_components(selected_row, selected_features)
