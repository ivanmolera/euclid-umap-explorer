"""Automatic preview scores and the canonical cutout's modal diagnostics."""
from __future__ import annotations

import base64
import hashlib
import html
from io import BytesIO

import pandas as pd
import streamlit as st
from PIL import Image

from .cluster_diagnostics import add_canonical_scores
from .config import SUMMARY_VISUAL_CLUSTER_LIMIT
from .images import load_image_bytes, object_image_path
from .lens_plausibility import DETECTOR_VERSION, score_image_bytes
from .runtime import log_app_event


CANONICAL_SCORE_HELP = (
    "Uncalibrated geometric evidence (0-100), not P(lens). Bright residual regions "
    "are scored by tangential alignment (30%), angular extent (25%), elongation (20%), "
    "radial coherence (15%) and contrast (10%). Total: up to 65 points for the best "
    "region, 20 for a second and 15 for their geometry. No catalogue labels enter "
    "the score. Spirals, neighbours, centring, noise and artifacts can affect it. "
    "A low score does not exclude a lens. The cyan circle is not an Einstein radius."
)
STATE_KEY = "canonical_preview_scores"


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
    rows = [item["canonical_row"] for item in view_model["clusters"][:SUMMARY_VISUAL_CLUSTER_LIMIT]
            if item["canonical_row"] is not None]
    ids = tuple(str(row.object_id) for row in rows)
    features = tuple(canonical_features)
    state = st.session_state.get(STATE_KEY)
    if (state is not None and state["_data"] is data and state["canonical_features"] == features
            and state["ids"] == ids and state["detector_version"] == DETECTOR_VERSION):
        return
    evidence = {}
    with st.spinner("Scoring canonical cutouts..."):
        for row in rows:
            try:
                evidence[str(row.object_id)] = load_canonical_evidence(row)
            except Exception as exc:
                log_app_event("canonical_preview_failed", error_type=type(exc).__name__)
                evidence[str(row.object_id)] = {"status": "image_error", "score": None}
    st.session_state[STATE_KEY] = {"_data": data, "canonical_features": features,
                                  "ids": ids, "detector_version": DETECTOR_VERSION,
                                  "evidence": evidence}


def preview_canonical_evidence(row):
    return st.session_state.get(STATE_KEY, {}).get("evidence", {}).get(str(row.object_id))


def summary_with_preview_scores(summary, data, canonical_features):
    state = st.session_state.get(STATE_KEY, {})
    matches = state.get("_data") is data and state.get("canonical_features") == tuple(canonical_features)
    result = add_canonical_scores(summary, state.get("evidence", {}) if matches else {})
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
