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
from .lens_plausibility import DETECTOR_VERSION, score_image_bytes
from .runtime import log_app_event
from .storage import is_gcs_path


CANONICAL_SCORE_HELP = (
    "Uncalibrated geometric evidence (0-100), not P(lens). Bright residual regions "
    "are scored by tangential alignment (30%), angular extent (25%), elongation (20%), "
    "radial coherence (15%) and contrast (10%). Total: up to 65 points for the best "
    "region, 20 for a second and 15 for their geometry. No catalogue labels enter "
    "the score. Spirals, neighbours, centring, noise and artifacts can affect it. "
    "A low score does not exclude a lens. The cyan circle is not an Einstein radius."
)
STATE_KEY = "canonical_preview_scores"
DETAIL_STATE_KEY = "canonical_image_details"


@st.cache_resource(show_spinner=False)
def score_filesystem():
    import gcsfs

    return gcsfs.GCSFileSystem(requests_timeout=30, timeout=30, skip_instance_cache=True)


def score_canonical_row(store, row):
    return store.score_candidates(morphology_cutout_candidates(row.get("id_str"), row.object_id))


@st.cache_data(show_spinner=False, max_entries=256)
def cached_visual_evidence(image_bytes: bytes, detector_version: str) -> dict:
    if detector_version != DETECTOR_VERSION:
        raise ValueError("Detector version mismatch.")
    return score_image_bytes(image_bytes)


def load_canonical_evidence(row):
    path = object_image_path(row)
    if path is None:
        return {"status": "missing_cutout", "score": None}
    image_bytes = load_image_bytes(path)
    with Image.open(BytesIO(image_bytes)) as image:
        mime = Image.MIME.get(image.format, "image/jpeg")
    return {**cached_visual_evidence(image_bytes, DETECTOR_VERSION),
            "image_path": path, "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
            "original_bytes": image_bytes, "original_mime": mime}


def ensure_preview_canonical_scores(view_model, data, canonical_features):
    """Score all summary representatives; workers never access Streamlit state."""
    ids = tuple(dict.fromkeys(str(value) for value in view_model["cluster_download_df"].canonical if str(value)))
    features = tuple(canonical_features)
    state = st.session_state.get(STATE_KEY)
    if (state is not None and state["_data"] is data and state["canonical_features"] == features
            and state["ids"] == ids and state["detector_version"] == DETECTOR_VERSION):
        return
    rows_by_id = data.set_index(data.object_id.astype(str), drop=False)
    rows = [rows_by_id.loc[object_id] for object_id in ids]
    candidates = [path for row in rows for path in morphology_cutout_candidates(row.get("id_str"), row.object_id)]
    store = CanonicalScoreStore(filesystem=score_filesystem() if any(is_gcs_path(p) for p in candidates) else None)
    evidence = {}
    state = {"_data": data, "canonical_features": features, "ids": ids,
             "detector_version": DETECTOR_VERSION, "evidence": evidence}
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
    item = st.session_state.get(STATE_KEY, {}).get("evidence", {}).get(str(row.object_id))
    if item is None:
        return None
    details = st.session_state.setdefault(DETAIL_STATE_KEY, {})
    key = (str(row.object_id), item.get("image_sha256"))
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
    matches = state.get("_data") is data and state.get("canonical_features") == tuple(canonical_features)
    result = add_canonical_scores(summary, state.get("evidence", {}) if matches else {})
    evidence = state.get("evidence", {}) if matches else {}
    for field in ("image_path", "image_sha256"):
        result[f"canonical_{field}"] = [evidence.get(str(value), {}).get(field, "") for value in result.canonical]
    result["canonical_detector_version"] = DETECTOR_VERSION
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
    return f"{score:.0f} / 100" if score is not None else "N/A"


def score_help_html(label):
    return (f'<span class="concept-help" tabindex="0">{html.escape(label)}'
            f'<span class="concept-popover">{html.escape(CANONICAL_SCORE_HELP)}</span></span>')


def canonical_detail_html(evidence, fallback_src):
    def source(content, mime):
        return f"data:{mime};base64,{base64.b64encode(content).decode()}"

    original = (source(evidence["original_bytes"], evidence.get("original_mime", "image/jpeg"))
                if evidence.get("original_bytes") else fallback_src)
    annotated = source(evidence["overlay_png"], "image/png") if evidence.get("overlay_png") else original
    label = score_help_html(f"Canonical score: {canonical_score_label(evidence)}")
    image_pair = "".join(
        f'<figure><figcaption>{title}</figcaption><img src="{html.escape(src or "")}" alt="{title}" /></figure>'
        for title, src in (("Original cutout", original), ("Annotated cutout", annotated)) if src
    )
    fields = [("angular_span_deg", "Angular extent (deg)"), ("tangential_alignment", "Tangential alignment"),
              ("elongation", "Elongation"), ("radius_px", "Radius (px)"), ("radial_cv", "Radial CV"),
              ("contrast_ratio", "Contrast"), ("component_score", "Region score (0-1)")]
    regions = evidence.get("regions", [])
    table = ""
    if regions:
        headers = "<th>Region</th>" + "".join(f"<th>{title}</th>" for _, title in fields)
        body = "".join(
            f"<tr><td>{index}</td>" + "".join(f"<td>{float(region[key]):.3f}</td>" for key, _ in fields) + "</tr>"
            for index, region in enumerate(regions, 1)
        )
        table = f'<div class="canonical-region-table"><table><thead><tr>{headers}</tr></thead><tbody>{body}</tbody></table></div>'
    else:
        table = '<p class="canonical-note">No contributing regions detected.</p>'
    parts = evidence.get("contributions", {})
    contributions = (
        '<p class="canonical-note">' + " | ".join(
            f"{title}: {parts[key]:.1f}" for key, title in (
                ("best_region", "Best region"), ("second_region", "Second region"),
                ("multi_region_geometry", "Multi-region geometry"))) + "</p>"
        if parts else ""
    )
    status = ""
    if evidence.get("score") is None:
        status = f'<p class="canonical-note">Not evaluable: {html.escape(evidence.get("status", "unavailable").replace("_", " "))}.</p>'
    return (
        f'<div class="canonical-analysis"><div class="thumb-modal-meta">{label}</div>{status}'
        f'<div class="canonical-image-pair">{image_pair}</div>'
        '<p class="canonical-note">Red: contributing regions. Yellow: assumed centre. '
        'Cyan: median marked radius. Orange: excluded border lines.</p>'
        f'{table}{contributions}</div>'
    )
