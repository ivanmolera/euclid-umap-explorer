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
    PCA_SELECTION_PRESETS,
    pca_filter_signature,
    pca_features_for_preset,
    sample_for_display,
)
from .birch import run_birch_clustering
from .catalogs import normalize_lens_grades
from .components import (
    ProcessingOverlay,
    close_processing_overlay,
    inject_plot_cursor_css,
    install_click_processing_overlay,
    render_back_to_top_control,
    render_cluster_visual_summary,
    render_app_flow_help,
    render_euclid_object_search,
    render_help_label,
    render_pca_filter_controls,
    request_clustering,
    show_morphology_catalogue_row,
    show_object_details,
    show_selected_pca_components,
)
from .config import (
    APP_TITLE,
    APP_VERSION,
    DENDROGRAM_MAX_OBJECTS,
    DENDROGRAM_TRUNCATE_CLUSTERS,
    DEFAULT_BIRCH_BATCH_SIZE,
    DEFAULT_BIRCH_BRANCHING_FACTOR,
    DEFAULT_BIRCH_THRESHOLD,
    DEFAULT_CLUSTER_FEATURES,
    DEFAULT_LENS_GRADES,
    DOWNLOAD_MAX_UMAP_ROWS,
    EUCLID_FAVICON_PATH,
    EUCLID_LOGO_PATH,
    LENS_GRADE_HELP,
    LENS_GRADE_OPTIONS,
    LENS_PATH,
    MAX_ALGORITHM_SECONDS,
    PARAMETER_HELP,
    PARQUET_PATH,
)
from .downloads import (
    cluster_summary_download_df,
    dataframe_to_csv_bytes,
    selected_point_indices,
    umap_download_df,
)
from .euclid_search import is_valid_search_object_id, selected_point_index
from .runtime import AlgorithmTimeoutError, format_duration, log_app_event
from .storage import path_exists, prepare_catalog_cache
from .subclustering import (
    build_subcluster_summary,
    build_subclustering_signature,
    build_dendrogram_figure,
    compute_hierarchical_subclusters,
)
from .umap import (
    build_umap_signature,
    compute_semisupervised_umap_embedding,
    compute_umap_embedding,
)


def request_umap_computation() -> None:
    st.session_state["umap_requested"] = True
    st.session_state["umap_running"] = True


def request_subclustering() -> None:
    st.session_state["subclustering_requested"] = True


def request_semisupervised_umap() -> None:
    st.session_state["semisupervised_umap_requested"] = True


def render_execution_time(seconds: object) -> None:
    st.markdown(
        f"""
        <div style="display: flex; align-items: baseline; gap: 0.45rem; margin-bottom: 0.75rem;">
            <span style="color: var(--text-color); font-size: 0.95rem; opacity: 0.65;">
                Execution time:
            </span>
            <span style="font-size: 1.8rem; font-weight: 600; line-height: 1;">
                {format_duration(seconds)}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.fragment
def render_cluster_umap_interaction(
    fig: object,
    embedding_df: pd.DataFrame,
    selected_features: list[str],
    selected_cluster: int,
) -> None:
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
            selection_mode=("points", "box", "lasso"),
            key="umap_selection",
        )

        selected_indices = selected_point_indices(event)
        umap_source_df = (
            embedding_df.loc[embedding_df.index.intersection(selected_indices)]
            if selected_indices
            else embedding_df
        )
        prepared_df = umap_download_df(
            embedding_df,
            selected_features,
            selected_indices,
        )
        download_col, help_col = st.columns([1, 2])
        with download_col:
            st.download_button(
                "Download object selection",
                data=dataframe_to_csv_bytes(prepared_df),
                file_name=f"cluster_{selected_cluster}_umap_objects.csv",
                mime="text/csv",
            )
        with help_col:
            st.caption("Use box or lasso selection to restrict the export")
        st.caption(
            f"Downloads {min(len(umap_source_df), DOWNLOAD_MAX_UMAP_ROWS):,} "
            f"of {len(umap_source_df):,} selected or loaded UMAP objects."
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


@st.fragment
def render_semisupervised_umap_interaction(
    semi_fig: object,
    semi_display_df: pd.DataFrame,
    selected_features: list[str],
    semi_signature: tuple,
    selected_semi_subcluster: int,
) -> None:
    semi_plot_col, semi_detail_col = st.columns([2, 1])
    with semi_plot_col:
        semi_event = st.plotly_chart(
            semi_fig,
            use_container_width=True,
            config={
                "displaylogo": False,
                "scrollZoom": True,
                "doubleClick": "reset",
            },
            on_select="rerun",
            selection_mode=("points", "box", "lasso"),
            key="semisupervised_umap_chart",
        )

        semi_selected_indices = selected_point_indices(semi_event)
        semi_source_df = (
            semi_display_df.loc[
                semi_display_df.index.intersection(semi_selected_indices)
            ]
            if semi_selected_indices
            else semi_display_df
        )
        prepared_semi_df = umap_download_df(
            semi_display_df,
            selected_features,
            semi_selected_indices,
        )
        semi_download_col, semi_help_col = st.columns([1, 2])
        with semi_download_col:
            st.download_button(
                "Download object selection",
                data=dataframe_to_csv_bytes(prepared_semi_df),
                file_name=(
                    f"subcluster_{selected_semi_subcluster}_"
                    "semisupervised_umap_objects.csv"
                ),
                mime="text/csv",
            )
        with semi_help_col:
            st.caption("Use box or lasso selection to restrict the export")
        st.caption(
            f"Downloads {min(len(semi_source_df), DOWNLOAD_MAX_UMAP_ROWS):,} "
            f"of {len(semi_source_df):,} selected or loaded UMAP objects."
        )

    semi_selected_index = selected_point_index(semi_event)
    semi_selected_row = (
        semi_display_df.loc[semi_selected_index]
        if semi_selected_index is not None
        else None
    )
    with semi_detail_col:
        if semi_selected_row is None:
            st.info("Select a point on the map to view its details and image.")
        else:
            show_object_details(semi_selected_row, selected_features)

    if semi_selected_row is not None:
        semi_morphology_col, semi_pca_col = st.columns([1, 1])
        with semi_morphology_col:
            show_morphology_catalogue_row(semi_selected_row)
        with semi_pca_col:
            show_selected_pca_components(semi_selected_row, selected_features)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon=str(EUCLID_FAVICON_PATH), layout="wide")
    inject_plot_cursor_css()
    install_click_processing_overlay()
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
        version_col, flow_help_col = st.columns([4, 1])
        with version_col:
            st.caption(f"Version {APP_VERSION}")
        with flow_help_col:
            render_app_flow_help()

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
            type="primary",
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
        finally:
            close_processing_overlay()

    clustering_requested = st.session_state.pop("cluster_requested", False)
    birch_requested = run_clustering or clustering_requested
    if birch_requested:
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
        st.session_state["cluster_summary_expanded"] = False
        log_app_event(
            "birch_clustering_requested",
            selected_grades=list(selected_lens_grades),
            threshold=float(threshold),
            branching_factor=int(branching_factor),
            batch_size=int(batch_size),
        )

    if not st.session_state.get("cluster_ready"):
        close_processing_overlay()
        st.info("Click **Run clustering** button to clusterize data.")
        st.stop()

    params = st.session_state["cluster_params"]
    lens_grades = cluster_lens_grades(params)
    cached_cluster = st.session_state.get("cluster_result")
    should_run_clustering = birch_requested or cached_cluster is None

    if should_run_clustering:
        overlay = ProcessingOverlay()
        try:
            overlay.open("Running BIRCH clustering...")
            # Only individual catalogues are cached. Image folders are read on demand.
            prepare_catalog_cache([PARQUET_PATH, LENS_PATH])
            clustered_df, pca_columns = run_birch_clustering(
                PARQUET_PATH,
                LENS_PATH,
                lens_grades,
                float(params["threshold"]),
                int(params["branching_factor"]),
                int(params["batch_size"]),
            )
            st.session_state["cluster_result"] = (clustered_df, pca_columns)
            st.session_state["cluster_summary_df"] = build_cluster_summary(clustered_df)
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
        finally:
            overlay.close()
    else:
        clustered_df, pca_columns = cached_cluster

    cluster_summary_df = st.session_state.get("cluster_summary_df")
    if cluster_summary_df is None:
        cluster_summary_df = build_cluster_summary(clustered_df)
        st.session_state["cluster_summary_df"] = cluster_summary_df
    cluster_summary_df = cluster_summary_df.copy()
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
        selected_pca_preset = st.selectbox(
            "PCA selection preset",
            PCA_SELECTION_PRESETS,
            index=0,
        )
        if selected_pca_preset != st.session_state.get("last_pca_selection_preset"):
            st.session_state["last_pca_selection_preset"] = selected_pca_preset
            if selected_pca_preset == "Manual selection":
                st.session_state.setdefault("selected_pca_components", default_features)
            else:
                st.session_state["selected_pca_components"] = pca_features_for_preset(
                    pca_columns,
                    selected_pca_preset,
                )
        elif "selected_pca_components" not in st.session_state:
            st.session_state["selected_pca_components"] = default_features
        else:
            selected_pca_components = [
                feature
                for feature in st.session_state["selected_pca_components"]
                if feature in pca_columns
            ]
            st.session_state["selected_pca_components"] = (
                selected_pca_components or default_features
            )

        selected_features = st.multiselect(
            "PCA components",
            pca_columns,
            key="selected_pca_components",
        )
        raw_pca_filters = render_pca_filter_controls(pca_columns)
        pca_filters = normalize_pca_filters(raw_pca_filters, pca_columns)

    if not selected_features:
        st.warning("Select at least one PCA component to build UMAP.")
        st.stop()

    with st.expander(
        "Clustering summary",
        expanded=st.session_state.get("cluster_summary_expanded", False),
    ):
        render_execution_time(clustered_df.attrs.get("processing_seconds"))
        cluster_download_df = cluster_summary_download_df(
            clustered_df,
            cluster_summary_df,
            selected_features,
        )
        summary_display = cluster_download_df.copy()
        summary_display["lens_rate"] = (summary_display["lens_rate"] * 100).round(3)
        st.dataframe(
            summary_display[
                [
                    "cluster",
                    "n_objects",
                    "n_lenses",
                    "lens_rate",
                    "canonical",
                    "anomalous",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download clustering table",
            data=dataframe_to_csv_bytes(cluster_download_df),
            file_name="clustering_summary.csv",
            mime="text/csv",
        )
        render_cluster_visual_summary(
            clustered_df,
            cluster_summary_df,
            pca_columns,
            selected_features,
        )

    selected_option = st.selectbox(
        "Cluster selection",
        cluster_summary_df["option"].tolist(),
        index=default_cluster_option_index(cluster_summary_df),
    )
    selected_cluster = int(
        cluster_summary_df.loc[
            cluster_summary_df["option"] == selected_option,
            "cluster",
        ].iloc[0]
    )

    cluster_df = clustered_df[clustered_df["cluster"] == selected_cluster].copy()
    filtered_cluster_df = apply_pca_filters(cluster_df, pca_filters)

    with st.expander(
        "UMAP",
        expanded=st.session_state.get("umap_parameters_expanded", True),
    ):
        with st.form("umap_parameters_form"):
            umap_param_cols = st.columns([1, 1, 1])
            with umap_param_cols[0]:
                render_help_label("n_neighbors", PARAMETER_HELP["n_neighbors"])
                n_neighbors = st.slider(
                    "n_neighbors",
                    2,
                    50,
                    10,
                    label_visibility="collapsed",
                )
            with umap_param_cols[1]:
                render_help_label("min_dist", PARAMETER_HELP["min_dist"])
                min_dist = st.slider(
                    "min_dist",
                    0.0,
                    1.0,
                    0.15,
                    step=0.01,
                    label_visibility="collapsed",
                )
            with umap_param_cols[2]:
                render_help_label("Maximum objects", PARAMETER_HELP["Maximum objects"])
                max_objects = st.slider(
                    "Maximum objects",
                    100,
                    100_000,
                    20_000,
                    step=100,
                    label_visibility="collapsed",
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
                or len(filtered_cluster_df) < 3
            )
            umap_submitted = st.form_submit_button(
                button_label,
                type="primary" if needs_recalculation else "secondary",
                disabled=umap_button_disabled,
            )
            if umap_submitted:
                request_umap_computation()

    with st.expander("Hierarchical clustering", expanded=False):
        dendrogram_signature = (
            int(selected_cluster),
            tuple(selected_features),
            pca_filter_signature(pca_filters),
            int(DENDROGRAM_MAX_OBJECTS),
            int(DENDROGRAM_TRUNCATE_CLUSTERS),
        )
        if st.button("Compute dendrogram preview"):
            try:
                st.session_state["hierarchical_dendrogram_fig"] = build_dendrogram_figure(
                    filtered_cluster_df,
                    selected_features,
                    DENDROGRAM_MAX_OBJECTS,
                    DENDROGRAM_TRUNCATE_CLUSTERS,
                )
                st.session_state["hierarchical_dendrogram_signature"] = dendrogram_signature
            except ValueError as exc:
                st.info(str(exc))

        if st.session_state.get("hierarchical_dendrogram_signature") == dendrogram_signature:
            st.plotly_chart(
                st.session_state["hierarchical_dendrogram_fig"],
                use_container_width=True,
                config={"displaylogo": False},
            )
            st.caption(
                "Dendrogram preview uses a sampled, truncated view of the selected cluster."
            )
        with st.form("hierarchical_subclustering_form"):
            subclustering_cols = st.columns([1, 1, 1])
            with subclustering_cols[0]:
                n_subclusters = st.slider(
                    "Subclusters",
                    2,
                    20,
                    2,
                )
            with subclustering_cols[1]:
                max_subcluster_objects = st.slider(
                    "Maximum objects for subclustering",
                    100,
                    20_000,
                    min(15_000, max(100, len(cluster_df))),
                    step=100,
                )
            with subclustering_cols[2]:
                color_by_subcluster = st.checkbox(
                    "Color UMAP by hierarchical subcluster",
                    value=bool(st.session_state.get("color_by_subcluster", False)),
                    key="color_by_subcluster",
            )
            subclustering_ready = "umap_embedding_df" in st.session_state
            subclustering_submitted = st.form_submit_button(
                "Compute hierarchical clustering",
                type="primary",
                disabled=(not subclustering_ready)
                or len(filtered_cluster_df) < 3,
            )
            if subclustering_submitted:
                request_subclustering()

    recalculate_umap = bool(st.session_state.get("umap_requested", False))

    if len(filtered_cluster_df) < 3:
        st.session_state["umap_running"] = False
        st.session_state["umap_requested"] = False
        st.warning("At least 3 objects must remain after PCA filters to compute UMAP.")
        st.stop()

    if recalculate_umap:
        overlay = ProcessingOverlay()
        try:
            overlay.open("Computing UMAP...")
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
            overlay.close()
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
        st.rerun()

    if pca_filters:
        st.caption(
            "Active PCA filters: "
            + "; ".join(format_pca_filter(pca_filter) for pca_filter in pca_filters)
        )

    if needs_recalculation or "umap_embedding_df" not in st.session_state:
        st.info("Click **Compute UMAP** or **Recompute UMAP** to update the visualization.")
        st.stop()

    embedding_df = st.session_state["umap_embedding_df"]
    umap_processing_seconds = embedding_df.attrs.get("processing_seconds")

    subclustering_signature = build_subclustering_signature(
        umap_signature,
        selected_features,
        n_subclusters,
        max_subcluster_objects,
    )
    subclustered_df = st.session_state.get("hierarchical_subcluster_df")
    subcluster_signature = st.session_state.get("hierarchical_subcluster_signature")
    if subcluster_signature != subclustering_signature:
        subclustered_df = None

    if st.session_state.get("subclustering_requested"):
        overlay = ProcessingOverlay()
        try:
            overlay.open("Computing hierarchical clustering...")
            subclustered_df = compute_hierarchical_subclusters(
                cluster_df,
                selected_features,
                n_subclusters,
                max_subcluster_objects,
            )
        except AlgorithmTimeoutError as exc:
            log_app_event(
                "hierarchical_subclustering_timeout",
                timeout_seconds=MAX_ALGORITHM_SECONDS,
                cluster=int(selected_cluster),
                n_features=int(len(selected_features)),
                max_objects=int(max_subcluster_objects),
            )
            st.error(
                "Hierarchical clustering was cancelled because it exceeded the "
                f"{format_duration(MAX_ALGORITHM_SECONDS)} execution limit. "
                "Try reducing the maximum number of objects before running it again."
            )
            st.exception(exc)
            st.stop()
        finally:
            overlay.close()
            st.session_state["subclustering_requested"] = False

        st.session_state["hierarchical_subcluster_df"] = subclustered_df
        st.session_state["hierarchical_subcluster_signature"] = subclustering_signature

    if subclustered_df is not None and not subclustered_df.empty:
        subcluster_lookup = subclustered_df[
            ["object_id", "hierarchical_subcluster"]
        ].drop_duplicates("object_id")
        embedding_df = embedding_df.merge(
            subcluster_lookup,
            on="object_id",
            how="left",
        )
        embedding_df.attrs["processing_seconds"] = umap_processing_seconds
        embedding_df["hierarchical_subcluster_label"] = embedding_df[
            "hierarchical_subcluster"
        ].map(lambda value: f"Subcluster {int(value)}" if pd.notna(value) else "Not sampled")

    (
        cluster_left,
        cluster_filtered,
        cluster_middle,
        cluster_right,
        cluster_fourth,
        cluster_fifth,
    ) = st.columns(6)
    cluster_left.metric("Cluster objects", f"{len(cluster_df):,}")
    cluster_filtered.metric("After filters", f"{len(filtered_cluster_df):,}")
    cluster_middle.metric("Objects in UMAP", f"{len(embedding_df):,}")
    cluster_right.metric("Lenses in UMAP", f"{int(embedding_df['is_lens'].sum()):,}")
    cluster_fourth.metric("Extremes", "2")
    if subclustered_df is not None and not subclustered_df.empty:
        cluster_fifth.metric(
            "Subclusters",
            f"{subclustered_df['hierarchical_subcluster'].nunique():,}",
        )
    else:
        cluster_fifth.metric("Subclusters", "-")

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
            "hierarchical_subcluster_label",
        )
        if column in embedding_df.columns
    ]

    import plotly.express as px
    color_column = (
        "hierarchical_subcluster_label"
        if color_by_subcluster and "hierarchical_subcluster_label" in embedding_df.columns
        else "point_role"
    )
    color_map = (
        None
        if color_column == "hierarchical_subcluster_label"
        else {
            "Unknown": "#4c78a8",
            "Lens candidate": "#d62728",
            "Canonical": "#2ca02c",
            "Anomaly": "#111111",
        }
    )

    fig = px.scatter(
        embedding_df,
        x="umap_1",
        y="umap_2",
        color=color_column,
        symbol="point_role",
        custom_data=["point_index"],
        hover_data=hover_columns,
        color_discrete_map=color_map,
        symbol_map={
            "Unknown": "circle",
            "Lens candidate": "circle",
            "Canonical": "diamond",
            "Anomaly": "x",
        },
        category_orders={
            "point_role": ["Unknown", "Lens candidate", "Canonical", "Anomaly"],
        },
        labels={
            "umap_1": "UMAP 1",
            "umap_2": "UMAP 2",
            "point_role": "Type",
            "hierarchical_subcluster_label": "Hierarchical subcluster",
        },
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
        legend_title_text=(
            "Hierarchical subcluster"
            if color_column == "hierarchical_subcluster_label"
            else "Object"
        ),
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        clickmode="event+select",
        dragmode="zoom",
        uirevision=st.session_state.get("umap_signature"),
    )

    with st.expander("UMAP summary", expanded=True):
        render_execution_time(embedding_df.attrs.get("processing_seconds"))
        if subclustered_df is not None and not subclustered_df.empty:
            st.markdown("**Hierarchical clustering summary**")
            subcluster_summary_df = build_subcluster_summary(subclustered_df).copy()
            if not subcluster_summary_df.empty:
                subcluster_summary_df["lens_rate"] = (
                    subcluster_summary_df["lens_rate"] * 100
                ).round(3)
                st.dataframe(
                    subcluster_summary_df[
                        ["hierarchical_subcluster", "n_objects", "n_lenses", "lens_rate"]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

        render_cluster_umap_interaction(
            fig,
            embedding_df,
            selected_features,
            selected_cluster,
        )

    if subclustered_df is not None and not subclustered_df.empty:
        with st.expander("Semi-supervised UMAP", expanded=False):
            st.caption(
                "Supervised labels guide the projection as A=2, B=1, C=0, "
                "and unknown objects=-1."
            )
            semi_subcluster_summary = build_subcluster_summary(subclustered_df)
            if semi_subcluster_summary.empty:
                subcluster_options = sorted(
                    subclustered_df["hierarchical_subcluster"].dropna().astype(int).unique()
                )
            else:
                subcluster_options = (
                    semi_subcluster_summary.sort_values(
                        ["lens_rate", "n_lenses", "n_objects", "hierarchical_subcluster"],
                        ascending=[False, False, False, True],
                    )["hierarchical_subcluster"]
                    .astype(int)
                    .tolist()
                )
            with st.form("semisupervised_umap_form"):
                semi_control_cols = st.columns([1, 1, 1, 1])
                with semi_control_cols[0]:
                    selected_semi_subcluster = st.selectbox(
                        "Subcluster",
                        subcluster_options,
                        format_func=lambda value: f"Subcluster {value}",
                        key="semisupervised_subcluster",
                    )
                with semi_control_cols[1]:
                    render_help_label(
                        "n_neighbors",
                        PARAMETER_HELP["n_neighbors"],
                    )
                    semi_n_neighbors = st.slider(
                        "n_neighbors",
                        2,
                        50,
                        10,
                        key="semisupervised_n_neighbors",
                        label_visibility="collapsed",
                    )
                with semi_control_cols[2]:
                    render_help_label(
                        "min_dist",
                        PARAMETER_HELP["min_dist"],
                    )
                    semi_min_dist = st.slider(
                        "min_dist",
                        0.0,
                        1.0,
                        0.15,
                        step=0.01,
                        key="semisupervised_min_dist",
                        label_visibility="collapsed",
                    )
                with semi_control_cols[3]:
                    semi_submitted = st.form_submit_button(
                        "Compute semi-supervised UMAP",
                        type="primary",
                    )
                    if semi_submitted:
                        request_semisupervised_umap()

            semi_signature = (
                subclustering_signature,
                int(selected_semi_subcluster),
                tuple(selected_features),
                int(semi_n_neighbors),
                round(float(semi_min_dist), 4),
            )
            semi_df = st.session_state.get("semisupervised_umap_df")
            if st.session_state.get("semisupervised_umap_signature") != semi_signature:
                semi_df = None

            if st.session_state.get("semisupervised_umap_requested"):
                overlay = ProcessingOverlay()
                semi_source_df = subclustered_df[
                    subclustered_df["hierarchical_subcluster"].astype(int)
                    == int(selected_semi_subcluster)
                ].copy()
                if len(semi_source_df) < 3:
                    st.session_state["semisupervised_umap_requested"] = False
                    st.warning(
                        "At least 3 objects are required to compute semi-supervised UMAP."
                    )
                    st.stop()
                try:
                    overlay.open("Computing semi-supervised UMAP...")
                    semi_df = compute_semisupervised_umap_embedding(
                        semi_source_df,
                        selected_features,
                        semi_n_neighbors,
                        semi_min_dist,
                    )
                except AlgorithmTimeoutError as exc:
                    log_app_event(
                        "semisupervised_umap_timeout",
                        timeout_seconds=MAX_ALGORITHM_SECONDS,
                        cluster=int(selected_cluster),
                        hierarchical_subcluster=int(selected_semi_subcluster),
                        n_features=int(len(selected_features)),
                    )
                    st.error(
                        "Semi-supervised UMAP was cancelled because it exceeded the "
                        f"{format_duration(MAX_ALGORITHM_SECONDS)} execution limit."
                    )
                    st.exception(exc)
                    st.stop()
                finally:
                    overlay.close()
                    st.session_state["semisupervised_umap_requested"] = False

                st.session_state["semisupervised_umap_df"] = semi_df
                st.session_state["semisupervised_umap_signature"] = semi_signature

            if semi_df is not None and not semi_df.empty:
                render_execution_time(semi_df.attrs.get("processing_seconds"))
                semi_metric_cols = st.columns(4)
                semi_metric_cols[0].metric("Subcluster objects", f"{len(semi_df):,}")
                semi_metric_cols[1].metric(
                    "Labelled candidates",
                    f"{int((semi_df['semi_supervised_target'] >= 0).sum()):,}",
                )
                semi_metric_cols[2].metric(
                    "Unknown",
                    f"{int((semi_df['semi_supervised_target'] < 0).sum()):,}",
                )
                semi_metric_cols[3].metric("PCA components", f"{len(selected_features):,}")

                semi_display_df = semi_df.reset_index(drop=True).copy()
                semi_display_df["point_index"] = semi_display_df.index
                semi_display_df["umap_1"] = semi_display_df["semi_umap_1"]
                semi_display_df["umap_2"] = semi_display_df["semi_umap_2"]
                semi_hover_columns = [
                    column
                    for column in (
                        "id_str",
                        "object_id",
                        "lens_grade",
                        "semi_supervised_label",
                        "hierarchical_subcluster",
                    )
                    if column in semi_display_df.columns
                ]
                semi_fig = px.scatter(
                    semi_display_df,
                    x="semi_umap_1",
                    y="semi_umap_2",
                    color="semi_supervised_label",
                    symbol="semi_supervised_label",
                    custom_data=["point_index"],
                    hover_data=semi_hover_columns,
                    color_discrete_map={
                        "Grade A": "#d62728",
                        "Grade B": "#ff7f0e",
                        "Grade C": "#f2c94c",
                        "Unknown": "#4c78a8",
                    },
                    labels={
                        "semi_umap_1": "Semi-supervised UMAP 1",
                        "semi_umap_2": "Semi-supervised UMAP 2",
                        "semi_supervised_label": "Label",
                    },
                    height=520,
                )
                semi_fig.update_traces(marker={"size": 7, "opacity": 0.78})
                for trace in semi_fig.data:
                    opacity = getattr(trace.marker, "opacity", None) or 1.0
                    trace.selected = {"marker": {"opacity": opacity}}
                    trace.unselected = {"marker": {"opacity": opacity}}
                semi_fig.update_layout(
                    title=(
                        f"Subcluster {selected_semi_subcluster} | "
                        "Semi-supervised UMAP"
                    ),
                    legend_title_text="Semi-supervised label",
                    margin={"l": 10, "r": 10, "t": 50, "b": 10},
                    dragmode="zoom",
                    clickmode="event+select",
                )
                render_semisupervised_umap_interaction(
                    semi_fig,
                    semi_display_df,
                    selected_features,
                    semi_signature,
                    selected_semi_subcluster,
                )

    close_processing_overlay()
