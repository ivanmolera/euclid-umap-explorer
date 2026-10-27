from __future__ import annotations

import hashlib
import json
import platform
import time
import zipfile
from datetime import datetime, timezone
from io import BytesIO

import pandas as pd
import streamlit as st

from .cluster_diagnostics import (
    DIAGNOSTIC_SEED, add_canonical_scores, add_reference_comparison, attach_evidence,
    build_diagnostic_figure, build_diagnostic_profiles, classify_diagnostic_regions,
    diagnostic_signature, diagnostic_statistics, independent_validation_auc,
    read_validation_labels, select_diagnostic_cohort,
)
from .components import render_help_label
from .images import load_image_bytes, object_image_path
from .lens_plausibility import DETECTOR_CONFIG, DETECTOR_VERSION, score_image_bytes
from .runtime import log_app_event


DIAGNOSTIC_HELP = (
    "Automatic canonical-image evidence, not P(lens). Catalogue enrichment counts "
    "reference candidates, not confirmed lenses. Low scores do not exclude lenses. "
    "Correlations are exploratory; they do not validate PCA or prove discoveries."
)
SCORE_HELP = (
    "Fixed notebook detector: up to 65 points for the best region, 20 for a second, "
    "and 15 for their geometry. No catalogue grades enter the score. "
    "JPEG contrast, centring, neighbours and artifacts can change it. "
    "The cyan circle is a median marked radius, not an Einstein radius."
)
DISPERSION_HELP = (
    "RMS distance per PCA coordinate from each group's mean, after standardizing "
    "with the complete catalogue reference population. Lower means more compact. "
    "Bubble area scales with Unknown count, with a minimum visible size; crosses have zero Unknown."
)


@st.cache_data(show_spinner=False, max_entries=256)
def cached_visual_evidence(image_bytes: bytes, detector_version: str) -> dict:
    if detector_version != DETECTOR_VERSION:
        raise ValueError("Detector version mismatch.")
    return score_image_bytes(image_bytes)


def _load_canonical_evidence(row):
    path = object_image_path(row)
    if path is None:
        return {"status": "missing_cutout", "score": None}
    image_bytes = load_image_bytes(path)
    evidence = cached_visual_evidence(image_bytes, DETECTOR_VERSION)
    return {**evidence, "image_path": path,
            "image_sha256": hashlib.sha256(image_bytes).hexdigest()}


def _run_scores(cohort, canonicals):
    evidence = {}
    started = time.monotonic()
    progress = st.progress(0., text="Scoring canonical cutouts...")
    for index, row in enumerate(cohort.itertuples()):
        if row.canonical:
            if time.monotonic() - started > 120:
                evidence[str(row.canonical)] = {"status": "time_limit", "score": None}
            else:
                try:
                    evidence[str(row.canonical)] = _load_canonical_evidence(canonicals[int(row.cluster)])
                except Exception as exc:
                    log_app_event("canonical_diagnostic_failed", error_type=type(exc).__name__)
                    evidence[str(row.canonical)] = {"status": "image_error", "score": None}
        progress.progress((index + 1) / len(cohort), text=f"Canonical cutouts: {index + 1} / {len(cohort)}")
    progress.empty()
    return evidence


def _validation_result(data, frame, *, group_column, exclude_canonical, prefix):
    result = {"auc": None}
    with st.expander("Independent validation labels", expanded=False):
        upload = st.file_uploader(
            "Validation CSV", type="csv", key=f"{prefix}_validation_csv",
            help="Columns: object_id and is_lens (0/1). Explicit independently reviewed labels only; Unknown is not 0.",
        )
        independent = st.checkbox(
            "Labels assessed independently of these scores", value=False,
            key=f"{prefix}_independent_labels",
        )
        if upload is not None and independent:
            try:
                if upload.size > 2_000_000:
                    raise ValueError("Validation CSV exceeds 2 MB.")
                labels = read_validation_labels(upload.getvalue())
                result = independent_validation_auc(
                    data, frame, labels, group_column=group_column,
                    exclude_canonical=exclude_canonical,
                )
                result.update(labels_sha256=hashlib.sha256(upload.getvalue()).hexdigest(),
                              independence_acknowledged=True)
                st.caption(
                    f"Matched validation objects: {result['n_validated']:,} / {result['n_submitted']:,}. "
                    f"Positive: {result['n_positive']:,}. Negative: {result['n_negative']:,}."
                )
                if result["auc"] is None:
                    st.info("ROC AUC requires both positive and negative validation objects in scored groups.")
            except (ValueError, KeyError, pd.errors.ParserError, UnicodeError) as exc:
                st.warning(str(exc))
    return result


def _export_diagnostics(frame, state, *, grades, exclude_canonical, score_threshold,
                        enrichment_threshold, group_column, scope_label, validation):
    import cv2
    import numpy
    import scipy
    import sklearn

    cohort_ids = set(frame.canonical.astype(str))
    evidence = {key: item for key, item in state["evidence"].items() if key in cohort_ids}
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "detector_version": DETECTOR_VERSION, "detector_config": DETECTOR_CONFIG,
        "score_semantics": "uncalibrated geometric evidence in a morphology JPEG, not P(lens)",
        "scope": scope_label, "group_column": group_column,
        "membership_features_labels_sha256": state["signature"],
        "reference_features_labels_sha256": state["reference_signature"],
        "features": state["features"], "selection": state["settings"],
        "canonical_features": state["canonical_features"],
        "reference_grades": list(grades) if grades is not None else "selected run grades",
        "exclude_canonical": exclude_canonical,
        "reference_definition": "Current clustering catalogue; if excluded, omit this group's canonical from group and catalogue counts",
        "pca_dispersion": "RMS per feature after StandardScaler fitted to complete reference catalogue vectors",
        "thresholds": {"visual_score": score_threshold, "enrichment": enrichment_threshold},
        "independent_validation": validation,
        "versions": {"python": platform.python_version(), "opencv": cv2.__version__,
                     "numpy": numpy.__version__, "pandas": pd.__version__,
                     "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
                     "streamlit": st.__version__},
        "images": [{"object_id": object_id, **{
            key: item[key] for key in ("image_path", "image_sha256", "status", "score", "center",
                                      "contributions", "quality_flag", "artifact_lines") if key in item
        }} for object_id, item in evidence.items()],
        "caveats": ["Unknown is not a negative", "Catalogue grades are not physical confirmation",
                    "No region confirms or rejects a group", "No equivalence to Cheng et al. AUC is assumed",
                    "Review-label AUC is limited to uploaded objects, not catalogue-wide",
                    "The canonical represents one member, not group purity"],
    }
    regions = [{"object_id": object_id, "region": index, **{
        key: value for key, value in region.items() if key != "features"
    }} for object_id, item in evidence.items()
        for index, region in enumerate(item.get("regions", []), start=1)]
    output = BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("cluster_diagnostics.csv", frame.to_csv(index=False))
        archive.writestr("marked_regions.csv", pd.DataFrame(regions).to_csv(index=False))
        archive.writestr("manifest.json", json.dumps(manifest, indent=2, allow_nan=False))
    return output.getvalue()


def _render_detail(frame, evidence, prefix):
    if frame.empty:
        return
    options = frame.cluster.astype(int).tolist()
    detail_key = f"{prefix}_detail"
    if st.session_state.get(detail_key) not in options:
        st.session_state[detail_key] = options[0]
    selected = st.selectbox("Cluster", options, key=detail_key, width=220)
    row = frame.loc[frame.cluster.eq(selected)].iloc[0]
    item = evidence.get(str(row.canonical), {})
    st.code(str(row.canonical) if row.canonical else "No canonical", language=None)
    if "image_path" not in item:
        st.info(f"Canonical image: {item.get('status', row.score_status).replace('_', ' ')}.")
        return
    try:
        image_bytes = load_image_bytes(item["image_path"])
    except Exception:
        st.warning("Could not reload the canonical cutout.")
        return
    original, annotated = st.columns(2, gap="small")
    with original:
        render_help_label("Original cutout", "Unaltered full-resolution morphology cutout.")
        st.image(image_bytes, width="stretch")
    with annotated:
        score_label = f"Visual evidence: {item['score']:.0f} / 100" if item.get("score") is not None else "Visual evidence"
        render_help_label(score_label, SCORE_HELP)
        st.image(item.get("overlay_png", image_bytes), width="stretch")
    st.caption("Red: contributing regions. Yellow: assumed centre. Cyan: median marked radius. Orange: excluded border lines.")
    if row.score_status != "exploratory_visual_only":
        st.info(f"Not scored: {row.score_status.replace('_', ' ')}.")
    if item.get("quality_flag"):
        st.warning("Possible border-line artifact: inspect the original cutout.")
    if item.get("regions"):
        measurements = pd.DataFrame([
            {"region": index, **{key: region[key] for key in (
                "angular_span_deg", "tangential_alignment", "radius_px", "radial_cv", "component_score"
            )}} for index, region in enumerate(item["regions"], start=1)
        ])
        st.dataframe(measurements, hide_index=True, width="stretch")
    contributions = item.get("contributions", {})
    if contributions:
        st.caption(
            f"Best region: {contributions['best_region']:.1f}. "
            f"Second region: {contributions['second_region']:.1f}. "
            f"Multi-region geometry: {contributions['multi_region_geometry']:.1f}."
        )


def summary_with_canonical_scores(summary, data, features, canonical_features=None):
    state = st.session_state.get("cluster_diagnostics_state", {})
    canonical_features = tuple(features if canonical_features is None else canonical_features)
    matching = (state.get("_data") is data and state.get("features") == tuple(features)
                and state.get("canonical_features") == canonical_features)
    return add_canonical_scores(summary, state.get("evidence", {}) if matching else {})


def render_cluster_diagnostics(data, features, *, reference=None, group_column="cluster",
                               prefix="cluster_diagnostics", scope_label="BIRCH clusters", preview_ids=None,
                               canonical_features=None):
    if not st.toggle("Show cluster diagnostics", key=f"{prefix}_show", value=False, persist_state="session"):
        return
    reference = data if reference is None else reference
    features = tuple(feature for feature in features if feature.startswith("feat_pca_"))
    canonical_features = tuple(features if canonical_features is None else canonical_features)
    render_help_label("Cluster diagnostic matrix", DIAGNOSTIC_HELP)
    state_key = f"{prefix}_state"
    state = st.session_state.get(state_key)
    if (state is None or state["_data"] is not data or state["_reference"] is not reference
            or state["features"] != features or state.get("canonical_features") != canonical_features):
        try:
            with st.spinner("Preparing canonical profiles..."):
                profiles, canonicals = build_diagnostic_profiles(
                    data, features, reference=reference, group_column=group_column,
                    canonical_features=canonical_features,
                )
            state = {"_data": data, "_reference": reference, "features": features,
                     "canonical_features": canonical_features,
                     "profiles": profiles, "canonicals": canonicals,
                     "signature": diagnostic_signature(data, features, group_column),
                     "reference_signature": diagnostic_signature(reference, features),
                     "evidence": {}, "cohort": None}
            st.session_state[state_key] = state
        except (ValueError, KeyError) as exc:
            st.warning(f"Diagnostics unavailable: {exc}")
            return
    with st.container(horizontal=True, vertical_alignment="bottom", gap="small"):
        min_objects = st.number_input("Minimum objects", min_value=1, value=100, step=10, key=f"{prefix}_min", width=160)
        maximum = st.number_input("Maximum clusters", min_value=1, max_value=10000, value=200, step=20,
                                  key=f"{prefix}_max", width=160)
        scope_options = ["Label-blind sample"] + (["First 20 previews"] if preview_ids is not None else [])
        scope = st.selectbox("Population", scope_options, key=f"{prefix}_population", width=220,
                            help="The capped sample is drawn by cluster ID with seed 42. Previews are label-prioritized.")
        run = st.button("Run diagnostics", icon=":material/science:", key=f"{prefix}_run")
    settings = {"min_objects": int(min_objects), "max_clusters": int(maximum),
                "seed": DIAGNOSTIC_SEED, "scope": scope,
                "preview_ids": list(preview_ids) if scope == "First 20 previews" else None}
    if run:
        cohort = select_diagnostic_cohort(
            state["profiles"], min_objects=min_objects, max_clusters=maximum,
            preview_ids=settings["preview_ids"],
        )
        if cohort.empty:
            state.update(cohort=None)
            st.info("No groups meet the sample criteria.")
            return
        evidence = _run_scores(cohort, state["canonicals"])
        state.update(cohort=attach_evidence(cohort, evidence),
                     evidence={**state["evidence"], **evidence}, settings=settings)
        st.session_state.pop(f"{prefix}_detail", None)
        st.session_state.pop(f"{prefix}_handled_chart", None)
        st.rerun()
    if state["cohort"] is None:
        st.caption(f"{scope_label}: {len(state['profiles']):,}. No diagnostic run yet.")
        return
    if state["settings"] != settings:
        st.warning("Sample settings changed. Results below belong to the last diagnostic run.")
    if state["settings"]["scope"] == "First 20 previews":
        st.warning("Preview-only selection is label-prioritized; this is not a catalogue-wide comparison.")
    st.caption(
        f"{scope_label}: {len(state['profiles']):,} groups. "
        f"Diagnostic sample: {len(state['cohort']):,}. Seed: {DIAGNOSTIC_SEED}."
    )
    with st.container(horizontal=True, vertical_alignment="bottom", gap="small"):
        reference_choice = st.selectbox("Reference labels", ["Selected run grades", "A+B only", "A only"],
                                        key=f"{prefix}_grades", width=200)
        score_threshold = st.number_input("High visual score", min_value=0., max_value=100., value=70.,
                                          key=f"{prefix}_score_threshold", width=160)
        enrichment_threshold = st.number_input("High enrichment (x)", min_value=0., value=10.,
                                               key=f"{prefix}_enrichment_threshold", width=180)
    exclude = st.checkbox("Exclude canonical from reference counts", value=True, key=f"{prefix}_exclude",
                          help="Omit each group's canonical from both group and catalogue counts for that comparison.")
    grades = {"Selected run grades": None, "A+B only": ("A", "B"), "A only": ("A",)}[reference_choice]
    frame = classify_diagnostic_regions(
        add_reference_comparison(state["cohort"], reference, grades=grades, exclude_canonical=exclude),
        score_threshold, enrichment_threshold,
    )
    validation = _validation_result(data, frame, group_column=group_column, exclude_canonical=exclude, prefix=prefix)
    stats = diagnostic_statistics(frame)
    metrics = st.columns(4)
    metrics[0].metric("Compared groups", f"{stats['n_compared']} / {len(frame)}")
    metrics[1].metric("Spearman rho", "-" if stats["spearman"] is None else f"{stats['spearman']:.3f}")
    metrics[2].metric("Pearson r", "-" if stats["pearson"] is None else f"{stats['pearson']:.3f}")
    metrics[3].metric("Validation ROC AUC", "-" if validation["auc"] is None else f"{validation['auc']:.3f}",
                      help="Only uploaded independent positive/negative labels. Never treats Unknown as negative; not catalogue-wide AUC.")
    render_help_label("PCA dispersion", DISPERSION_HELP)
    chart_key = f"{prefix}_chart_{state['signature']}_{state['settings']}"
    chart = st.plotly_chart(
        build_diagnostic_figure(frame, score_threshold=score_threshold, enrichment_threshold=enrichment_threshold,
                                title=scope_label),
        width="stretch", key=chart_key, on_select="rerun", selection_mode="points",
    )
    points = chart.selection.points
    selection = str(points[0].get("customdata", [""])[0]) if points else ""
    if selection and selection != st.session_state.get(f"{prefix}_handled_chart"):
        st.session_state[f"{prefix}_handled_chart"] = selection
        st.session_state[f"{prefix}_detail"] = int(selection)
    elif not selection:
        st.session_state.pop(f"{prefix}_handled_chart", None)
    st.caption("Diagnostic regions are review categories. No group is confirmed, discovered or discarded automatically.")
    columns = ["cluster", "canonical", "visual_score", "score_status", "n_objects", "n_unknown",
               "n_grade_a", "n_grade_b", "n_grade_c", "n_reference_compared", "enrichment", "pca_rms", "diagnostic_region"]
    st.dataframe(frame[columns], hide_index=True, width="stretch",
                 column_config={"canonical": st.column_config.TextColumn("Canonical object ID"),
                                "visual_score": st.column_config.NumberColumn("Visual evidence / 100", format="%.0f"),
                                "enrichment": st.column_config.NumberColumn("Enrichment (x)", format="%.2f"),
                                "pca_rms": st.column_config.NumberColumn("PCA RMS dispersion", format="%.3f")})
    st.download_button(
        "Download diagnostics", data=_export_diagnostics(
            frame, state, grades=grades, exclude_canonical=exclude, score_threshold=score_threshold,
            enrichment_threshold=enrichment_threshold, group_column=group_column, scope_label=scope_label,
            validation=validation,
        ), file_name=f"{prefix}.zip", mime="application/zip", icon=":material/download:", key=f"{prefix}_download",
    )
    _render_detail(frame, state["evidence"], prefix)
