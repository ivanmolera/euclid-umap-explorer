"""Curve-aware visual scoring over the frozen detector's segmented regions."""
from __future__ import annotations

import hashlib
from io import BytesIO
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import canonical_scoring as legacy
from . import lens_plausibility as detector


SCORING_VERSION = "curve-aware-visual-evidence-2"
MODEL_NAME = "curve-aware-heuristic"
MODEL_SHA256 = legacy.MODEL_SHA256
_RULES_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
SCORING_KEY = f"{SCORING_VERSION}:{MODEL_SHA256}:{_RULES_SHA256}"
POOL_WEIGHTS = {"best_region": 0.80, "second_region": 0.15, "multi_region_geometry": 0.05}
RING_WEIGHTS = {"coverage": 0.35, "coherence": 0.30, "contrast": 0.20, "continuity": 0.15}
RADIAL_WEIGHTING = {
    "anchor_min_score": 0.60,
    "anchor_min_contiguous_deg": 60.0,
    "full_weight_radius_ratio": 1.6,
    "min_weight_radius_ratio": 2.5,
    "min_weight": 0.25,
}
SECTORS = 36
REGION_CAP = 6


def scoring_fingerprint():
    return {
        "version": SCORING_VERSION,
        "selected_model": MODEL_NAME,
        "rules_sha256": _RULES_SHA256,
        "arc_feature_model": legacy.scoring_fingerprint(),
        "aggregation_weights": dict(POOL_WEIGHTS),
        "ring_weights": dict(RING_WEIGHTS),
        "radial_weighting": dict(RADIAL_WEIGHTING),
        "angular_sectors": SECTORS,
        "region_cap": REGION_CAP,
    }


def _angular_support(angles):
    indices = (np.mod(angles, 2 * np.pi) * SECTORS / (2 * np.pi)).astype(int) % SECTORS
    occupied = np.bincount(indices, minlength=SECTORS) > 0
    # Repeat the circular histogram so a contiguous arc can cross zero degrees.
    run = longest = 0
    for present in np.tile(occupied, 2):
        run = run + 1 if present else 0
        longest = max(longest, run)
    return int(occupied.sum()), min(longest, SECTORS)


def _score_region(component, center, shape):
    h, w = shape
    x, y = component["pixels"].T
    # Preserve edge and area vetoes even when rescuing a curved component.
    if min(x.min(), y.min(), w - 1 - x.max(), h - 1 - y.max()) <= 1 or len(x) > .08 * h * w:
        return None
    dx, dy = x - center[0], y - center[1]
    radius = float(np.median(np.hypot(dx, dy)))
    covered, contiguous = _angular_support(np.arctan2(dy, dx))
    cfg = detector.DETECTOR_CONFIG
    scale = min(h, w)
    inner = cfg["inner_radius_fraction"] * scale
    outer = cfg["outer_radius_fraction"] * scale
    if not inner <= radius <= outer:
        return None

    radial_cv = component["radial_cv"]
    features = component["features"]
    arc_score = sum(features[key] * legacy.REGION_WEIGHTS[key] for key in legacy.FEATURE_NAMES)
    ring_score = None
    if (len(x) >= 24 and radius >= 1.12 * inner and covered >= 20 and contiguous >= 14
            and radial_cv <= .18 and component["contrast_ratio"] >= 1.25):
        ring_features = {
            "coverage": detector.unit_interval(covered / (.80 * SECTORS)),
            "continuity": detector.unit_interval(contiguous / (.75 * SECTORS)),
            "coherence": features["radial_coherence"],
            "contrast": features["contrast"],
        }
        ring_score = sum(ring_features[key] * weight for key, weight in RING_WEIGHTS.items())
    recovered_arc = (
        component["radius_px"] < inner and covered >= 6 and contiguous >= 5
        and len(x) >= 12 and radial_cv <= .18 and component["contrast_ratio"] >= 1.25
        and component["tangential_alignment"] >= .75 and component["elongation"] >= cfg["min_elongation"]
    )
    # Wide tangential streaks can pass the legacy shape gate without being curved.
    arc_allowed = recovered_arc or (component["accepted"] and (covered < 6 or radial_cv <= .18))
    if ring_score is not None and (not arc_allowed or ring_score > arc_score):
        kind, score = "Ring-like", ring_score
    elif arc_allowed:
        kind, score = "Arc-like", arc_score
    else:
        return None
    return {
        **{k: component[k] for k in (
            "centroid_x", "centroid_y", "angular_span_deg", "tangential_alignment",
            "elongation", "radial_cv", "contrast_ratio", "pixels",
        )},
        "shape": kind,
        "radius_px": radius,
        "angular_coverage_deg": covered * 360.0 / SECTORS,
        "contiguous_coverage_deg": contiguous * 360.0 / SECTORS,
        "component_score": score,
    }


def _overlay_png(rgb, regions, center, artifact_lines):
    overlay = rgb.copy()
    h, w = rgb.shape[:2]
    for index, region in enumerate(regions, 1):
        x, y = region["pixels"].T
        overlay[y, x] = .45 * overlay[y, x] + .55 * np.array([255, 55, 55])
        text_x = min(w - 10, max(1, int(region["centroid_x"]) + 3))
        text_y = min(h - 2, max(9, int(region["centroid_y"]) - 3))
        cv2.putText(overlay, str(index), (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1, cv2.LINE_AA)
    centre_px = tuple(round(value) for value in center)
    cv2.drawMarker(overlay, centre_px, (255, 230, 0),
                   markerType=cv2.MARKER_CROSS, markerSize=11, thickness=1)
    if regions:
        radius = round(regions[0]["radius_px"])
        cv2.circle(overlay, centre_px, radius, (0, 220, 220), 1, cv2.LINE_AA)
    stroke = max(2, round(.018 * min(h, w)))
    for x1, y1, x2, y2 in artifact_lines:
        cv2.line(overlay, (x1, y1), (x2, y2), (255, 165, 0), stroke, cv2.LINE_AA)
    buffer = BytesIO()
    Image.fromarray(overlay).save(buffer, format="PNG")
    return buffer.getvalue()


def _apply_radial_weights(regions):
    rule = RADIAL_WEIGHTING
    anchor_radius = min((
        r["radius_px"] for r in regions
        if r["component_score"] >= rule["anchor_min_score"]
        and r["contiguous_coverage_deg"] >= rule["anchor_min_contiguous_deg"]
    ), default=None)
    for region in regions:
        weight = 1.
        if anchor_radius is not None:
            ratio = region["radius_px"] / anchor_radius
            progress = detector.unit_interval(
                (ratio - rule["full_weight_radius_ratio"])
                / (rule["min_weight_radius_ratio"] - rule["full_weight_radius_ratio"])
            )
            weight -= (1. - rule["min_weight"]) * progress
        region["raw_component_score"] = region["component_score"]
        region["radial_weight"] = weight
        region["component_score"] *= weight
    return anchor_radius


def score_image_bytes(image_bytes: bytes, *, include_overlay: bool = True) -> dict:
    with Image.open(BytesIO(image_bytes)) as image:
        if image.width * image.height > 10_000_000:
            return {"status": "image_too_large", "score": None}
        rgb = np.asarray(image.convert("RGB"))
    baseline = detector.analyze_image(rgb)
    result = {"status": baseline["status"], "score": None, "scoring_model": scoring_fingerprint()}
    if baseline["score"] is None:
        return result

    center = baseline["center"]
    regions = []
    for component in baseline["components"]:
        region = _score_region(component, center, rgb.shape[:2])
        if region is not None:
            regions.append(region)
    anchor_radius = _apply_radial_weights(regions)
    regions.sort(key=lambda r: -r["component_score"])
    del regions[REGION_CAP:]
    arcs = [r for r in regions if r["shape"] == "Arc-like"]
    geometry = max((
        detector.pair_agreement((left, right), center) * min(left["radial_weight"], right["radial_weight"])
        for left, right in combinations(arcs, 2)
    ), default=0.)
    contributions = {
        "best_region": 100 * POOL_WEIGHTS["best_region"] * regions[0]["component_score"] if regions else 0.,
        "second_region": 100 * POOL_WEIGHTS["second_region"] * regions[1]["component_score"] if len(regions) > 1 else 0.,
        "multi_region_geometry": 100 * POOL_WEIGHTS["multi_region_geometry"] * geometry,
    }
    artifact_lines = baseline["artifact_lines"]
    for rank, region in enumerate(regions):
        region["region_contribution_points"] = (
            contributions[("best_region", "second_region")[rank]] if rank < 2 else 0.
        )
    result.update(
        score=min(100., sum(contributions.values())),
        center=center,
        contributions=contributions,
        pair_agreement=geometry,
        radial_reference_radius_px=anchor_radius,
        quality_flag=baseline["quality_flag"],
        artifact_lines=[list(line) for line in artifact_lines],
        image_width=int(rgb.shape[1]),
        image_height=int(rgb.shape[0]),
        regions=[{k: v for k, v in r.items() if k != "pixels"} for r in regions],
    )
    if include_overlay:
        result["overlay_png"] = _overlay_png(rgb, regions, center, artifact_lines)
    return result
