"""Scores for every cluster, with bounded image details for visible previews."""
from __future__ import annotations

import base64
import hashlib
import html
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

import pandas as pd
import streamlit as st
from PIL import Image

from .cluster_diagnostics import add_canonical_scores
from .canonical_cache import CanonicalScoreStore, detector_fingerprint
from .config import MAX_ALGORITHM_SECONDS
from .images import load_image_bytes, morphology_cutout_candidates, object_image_path
from .lens_plausibility import DETECTOR_VERSION
from .morphology_scoring import (
    MODEL_NAME, MODEL_SHA256, SCORING_KEY, SCORING_VERSION, score_image_bytes, scoring_fingerprint,
)
from .runtime import log_app_event
from .storage import is_gcs_path


_SCORE_WEIGHT_LABELS = {
    "tangential": "tangential alignment",
    "angular_span": "angular extent",
    "coverage": "occupied angular sectors",
    "coherence": "radial coherence",
    "continuity": "continuous angular coverage",
    "multi_region_geometry": "arc-pair geometry",
}


def _weight_percent(weight):
    return f"{100 * weight:.2f}".rstrip("0").rstrip(".") + "%"


def _score_help_text() -> str:
    model = scoring_fingerprint()
    radial = model["radial_filter"]

    def weighted_terms(weights):
        def term(key, weight):
            label = _SCORE_WEIGHT_LABELS.get(key, key.replace("_", " "))
            return f"{_weight_percent(weight)} {label}"

        return ", ".join(
            term(key, weight) for key, weight in sorted(weights.items(), key=lambda item: -item[1])
        )

    return "\n\n".join((
        "Experimental visual index (0-100), not P(lens).",
        f"Arc-like and other regions (learned weights): {weighted_terms(model['arc_feature_model']['region_weights'])}.",
        f"Ring-like regions: {weighted_terms(model['ring_weights'])}. No elongation requirement.",
        "Distance filter: the innermost arc or ring scoring "
        f"at least {radial['anchor_min_score']:g} with {radial['anchor_min_contiguous_deg']:g} degrees "
        "of continuous coverage sets the reference radius. Only regions marked Other beyond "
        f"{radial['max_other_radius_ratio']:g}x that radius are excluded. "
        "Recognized arcs and rings are never excluded by this distance filter.",
        f"Final score: {weighted_terms(model['aggregation_weights'])}.",
        "A low score does not exclude lensing. Cyan marks the median pixel radius of the top region, "
        "not an Einstein radius.",
    ))


CANONICAL_SCORE_HELP = _score_help_text()
STATE_KEY = "canonical_preview_scores"
DETAIL_STATE_KEY = "canonical_image_details"

SCORE_DETAIL_CSS = """
.canonical-analysis .score-detail-label {
    font-size: 0.9rem;
    margin-bottom: 0.45rem;
}
.canonical-image-pair {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.75rem;
}
.canonical-image-pair figure { margin: 0; min-width: 0; }
.canonical-image-pair figcaption {
    font-size: 0.85rem;
    margin-bottom: 0.35rem;
}
.canonical-image-pair img { display: block; width: 100%; height: auto; }
.canonical-note {
    font-size: 0.78rem;
    line-height: 1.4;
    margin: 0.75rem 0;
    opacity: 0.82;
}
.canonical-region-table { overflow-x: auto; max-width: 100%; }
.canonical-region-table table {
    border-collapse: collapse;
    font-size: 0.78rem;
    width: 100%;
}
.canonical-region-table th, .canonical-region-table td {
    border: 1px solid var(--score-table-border, var(--border-color, #7f88936b));
    padding: 0.35rem 0.5rem;
    text-align: right;
    white-space: nowrap;
}
@media (max-width: 640px) {
    .canonical-image-pair { grid-template-columns: minmax(0, 1fr); }
}
"""


@st.cache_resource(show_spinner=False)
def score_filesystem():
    import gcsfs

    return gcsfs.GCSFileSystem(requests_timeout=30, timeout=30, skip_instance_cache=True)


def score_canonical_row(store, row):
    return store.score_candidates(morphology_cutout_candidates(row.get("id_str"), row.object_id))


@st.cache_data(show_spinner=False, max_entries=256)
def cached_visual_evidence(image_bytes: bytes, detector_version: str, scoring_key: str) -> dict:
    if detector_version != DETECTOR_VERSION:
        raise ValueError("Detector version mismatch.")
    if scoring_key != SCORING_KEY:
        raise ValueError("Canonical scoring model mismatch.")
    return score_image_bytes(image_bytes)


def load_canonical_evidence(row):
    path = object_image_path(row)
    if path is None:
        return {"status": "missing_cutout", "score": None}
    return load_cutout_evidence(path)


def load_cutout_evidence(path: str) -> dict:
    image_bytes = load_image_bytes(path)
    with Image.open(BytesIO(image_bytes)) as image:
        mime = Image.MIME.get(image.format, "image/jpeg")
    return {**cached_visual_evidence(image_bytes, DETECTOR_VERSION, SCORING_KEY),
            "image_path": path, "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
            "original_bytes": image_bytes, "original_mime": mime}


def ensure_preview_canonical_scores(view_model, data, canonical_features):
    """Score all summary representatives; workers never access Streamlit state."""
    ids = tuple(dict.fromkeys(str(value) for value in view_model["cluster_download_df"].canonical if str(value)))
    features = tuple(canonical_features)
    state = st.session_state.get(STATE_KEY)
    fingerprint = detector_fingerprint()
    if (state is not None and state["_data"] is data and state["canonical_features"] == features
            and state["ids"] == ids and state.get("detector_fingerprint") == fingerprint):
        return
    rows_by_id = data.set_index(data.object_id.astype(str), drop=False)
    rows = [rows_by_id.loc[object_id] for object_id in ids]
    candidates = [path for row in rows for path in morphology_cutout_candidates(row.get("id_str"), row.object_id)]
    store = CanonicalScoreStore(filesystem=score_filesystem() if any(is_gcs_path(p) for p in candidates) else None)
    evidence = {}
    state = {"_data": data, "canonical_features": features, "ids": ids,
             "detector_version": DETECTOR_VERSION, "detector_fingerprint": fingerprint, "evidence": evidence}
    st.session_state[STATE_KEY] = state
    st.session_state[DETAIL_STATE_KEY] = {}
    progress = st.progress(0., text=f"Canonical scores: 0 / {len(rows)}")
    started = updated = time.monotonic()
    hits = 0
    pool = ThreadPoolExecutor(max_workers=8)
    futures = {pool.submit(score_canonical_row, store, row): str(row.object_id) for row in rows}
    try:
        for future in as_completed(futures, timeout=MAX_ALGORITHM_SECONDS):
            object_id = futures[future]
            try:
                item, cached = future.result()
                # Persist only compact metadata, never thousands of original/annotated images.
                evidence[object_id] = {k: v for k, v in item.items() if k not in ("original_bytes", "overlay_png")}
                hits += int(cached)
            except Exception as exc:
                log_app_event("canonical_score_failed", error_type=type(exc).__name__)
                evidence[object_id] = {"status": "image_error", "score": None}
            now = time.monotonic()
            if len(evidence) == len(rows) or now - updated > .5:
                progress.progress(len(evidence) / max(len(rows), 1), text=f"Canonical scores: {len(evidence)} / {len(rows)}")
                updated = now
    except TimeoutError:
        st.warning("Canonical scoring timed out. Completed scores are cached; retry to finish the remaining cutouts.")
    finally:
        for future in futures:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
        progress.empty()
    for object_id in ids:
        evidence.setdefault(object_id, {"status": "timeout", "score": None})
    state["cache_hits"] = hits
    state["elapsed_seconds"] = time.monotonic() - started
    state["cache_issue"] = store.cache_issue
    log_app_event("canonical_scores_complete", count=len(evidence), cache_hits=hits,
                  elapsed_seconds=round(state["elapsed_seconds"], 2), cache_issue=store.cache_issue)


def preview_canonical_evidence(row):
    state = st.session_state.get(STATE_KEY, {})
    if state.get("detector_fingerprint") != detector_fingerprint():
        return None
    item = state.get("evidence", {}).get(str(row.object_id))
    if item is None:
        return None
    details = st.session_state.setdefault(DETAIL_STATE_KEY, {})
    key = (str(row.object_id), item.get("image_sha256"), SCORING_KEY)
    if key not in details:
        try:
            detail = load_canonical_evidence(row)
            if item.get("image_sha256") and detail.get("image_sha256") != item["image_sha256"]:
                return {**item, "regions": []}
            details[key] = detail
            while len(details) > 32:
                details.pop(next(iter(details)))
        except Exception:
            return item
    return details[key]


def summary_with_preview_scores(summary, data, canonical_features):
    state = st.session_state.get(STATE_KEY, {})
    matches = (state.get("_data") is data and state.get("canonical_features") == tuple(canonical_features)
               and state.get("detector_fingerprint") == detector_fingerprint())
    result = add_canonical_scores(summary, state.get("evidence", {}) if matches else {})
    evidence = state.get("evidence", {}) if matches else {}
    for field in ("image_path", "image_sha256"):
        result[f"canonical_{field}"] = [evidence.get(str(value), {}).get(field, "") for value in result.canonical]
    result["canonical_detector_version"] = DETECTOR_VERSION
    result["canonical_scoring_version"] = SCORING_VERSION
    result["canonical_scoring_model"] = MODEL_NAME
    result["canonical_weights_sha256"] = MODEL_SHA256
    result["canonical_detector_fingerprint"] = json.dumps(detector_fingerprint(), sort_keys=True)
    result["canonical_source_signature"] = [json.dumps(evidence.get(str(value), {}).get("source_signature", {}), sort_keys=True)
                                             for value in result.canonical]
    result["canonical_pca_features"] = json.dumps(list(canonical_features))
    labelled = data.loc[data.is_lens.fillna(False).astype(bool)]
    grades = labelled.get("lens_grade", pd.Series("", index=labelled.index))
    for grade in "ABC":
        counts = labelled.loc[grades.eq(grade)].groupby("cluster").size()
        result[f"n_grade_{grade.lower()}"] = result.cluster.map(counts).fillna(0).astype(int)
    return result


def canonical_score_label(evidence):
    score = evidence.get("score")
    return f"{score:.1f} / 100" if score is not None else "N/A"


def score_help_html(label):
    help_html = html.escape(CANONICAL_SCORE_HELP).replace("\n", "<br>")
    return (f'<span class="concept-help" tabindex="0">{html.escape(label)}'
            f'<span class="concept-popover">{help_html}</span></span>')


def score_overlay_legend(evidence):
    parts = ["Yellow: estimated centre."]
    if evidence.get("regions"):
        parts = ["Red: scored regions.", *parts, "Cyan: median radius of the top-scoring region."]
    if evidence.get("artifact_lines"):
        parts.append("Orange: excluded border lines.")
    return " ".join(parts)


def canonical_detail_html(evidence, fallback_src, *, score_title="Canonical score"):
    def source(content, mime):
        return f"data:{mime};base64,{base64.b64encode(content).decode()}"

    original = (source(evidence["original_bytes"], evidence.get("original_mime", "image/jpeg"))
                if evidence.get("original_bytes") else fallback_src)
    annotated = source(evidence["overlay_png"], "image/png") if evidence.get("overlay_png") else None
    label = score_help_html(f"{score_title}: {canonical_score_label(evidence)}")
    image_pair = "".join(
        f'<figure><figcaption>{title}</figcaption><img src="{html.escape(src or "")}" alt="{title}" /></figure>'
        for title, src in (("Original cutout", original), ("Annotated cutout", annotated)) if src
    )
    fields = [("angular_span_deg", "Angular extent (deg)"), ("tangential_alignment", "Tangential alignment"),
              ("elongation", "Elongation"), ("radius_px", "Radius (px)"), ("radial_cv", "Radial CV"),
              ("contrast_ratio", "Contrast"), ("component_score", "Region score (0-1)")]
    regions = evidence.get("regions", [])
    if regions and "learned_rank" in regions[0]:
        fields = [("learned_rank", "Learned rank"), *fields,
                  ("region_contribution_points", "Contribution (points)")]
    elif regions and "shape" in regions[0]:
        fields = [("shape", "Shape"), ("angular_coverage_deg", "Coverage (deg)"), *fields,
                  ("region_contribution_points", "Contribution (points)")]
        fields = [(key, "Median radius (px)" if key == "radius_px" else title) for key, title in fields]

    def format_region_value(region, key):
        if key == "shape":
            return html.escape(region[key])
        return format(float(region[key]), ".0f" if key in ("learned_rank", "angular_coverage_deg") else ".3f")

    table = ""
    if regions:
        headers = "<th>Region</th>" + "".join(f"<th>{title}</th>" for _, title in fields)
        body = "".join(
            f"<tr><td>{index}</td>" + "".join(
                f"<td>{format_region_value(region, key)}</td>"
                for key, _ in fields) + "</tr>"
            for index, region in enumerate(regions, 1)
        )
        table = f'<div class="canonical-region-table"><table><thead><tr>{headers}</tr></thead><tbody>{body}</tbody></table></div>'
    else:
        table = '<p class="canonical-note">No contributing regions detected.</p>'
    parts = evidence.get("contributions", {})
    reference_radius = evidence.get("radial_reference_radius_px")
    reference_note = f" | Radial reference: {reference_radius:.2f} px" if reference_radius is not None else ""
    excluded = evidence.get("radial_excluded_count", 0)
    excluded_note = f" | Distant regions excluded: {excluded}" if excluded else ""
    contributions = (
        '<p class="canonical-note">' + " | ".join(
            f"{title}: {parts[key]:.1f}" for key, title in (
                ("best_region", "Best region"), ("second_region", "Second region"),
                ("multi_region_geometry", "Multi-region geometry"))) + reference_note + excluded_note + "</p>"
        if parts else ""
    )
    status = ""
    if evidence.get("score") is None:
        status = f'<p class="canonical-note">Not evaluable: {html.escape(evidence.get("status", "unavailable").replace("_", " "))}.</p>'
    legend = (
        f'<p class="canonical-note">{score_overlay_legend(evidence)}</p>'
        if annotated else ""
    )
    return (
        f'<style>{SCORE_DETAIL_CSS}</style>'
        f'<div class="canonical-analysis"><div class="score-detail-label">{label}</div>{status}'
        f'<div class="canonical-image-pair">{image_pair}</div>{legend}{table}{contributions}</div>'
    )
