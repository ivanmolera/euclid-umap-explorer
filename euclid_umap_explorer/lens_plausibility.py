"""Label-blind, uncalibrated visual evidence from morphology cutouts.

Port of the detector in escope_lens_plausibility_review_colab.ipynb. Pixel
measurements describe the displayed JPEG, not scientific VIS photometry.
"""
from __future__ import annotations

import math
from io import BytesIO

import cv2
import numpy as np
from PIL import Image


DETECTOR_VERSION = "visual-evidence-1"
DETECTOR_CONFIG = {
    "inner_radius_fraction": 0.09,
    "outer_radius_fraction": 0.46,
    "residual_percentile": 91.0,
    "min_tangential_alignment": 0.55,
    "min_elongation": 1.25,
    "min_angular_span_deg": 10.0,
    "max_radial_cv": 0.35,
    "long_line_fraction": 0.50,
}


def unit_interval(value):
    return float(np.clip(value, 0.0, 1.0))


def angular_span_deg(angles):
    if len(angles) < 2:
        return 0.0
    ordered = np.sort(np.mod(angles, 2.0 * np.pi))
    gaps = np.diff(np.r_[ordered, ordered[0] + 2.0 * np.pi])
    return float(np.degrees(2.0 * np.pi - gaps.max()))


def normalize_gray(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    low, high = np.percentile(gray, [1.0, 99.5])
    if high - low < 4.0:
        return None
    return np.clip((gray - low) / (high - low), 0.0, 1.0)


def estimate_center(gray):
    height, width = gray.shape
    scale = min(height, width)
    ys, xs = np.mgrid[:height, :width]
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    prior = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * (0.18 * scale) ** 2))
    smooth = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(1.5, 0.022 * scale))
    peak_y, peak_x = np.unravel_index(np.argmax(smooth * prior), gray.shape)
    local = (xs - peak_x) ** 2 + (ys - peak_y) ** 2 < (0.08 * scale) ** 2
    weights = np.maximum(smooth - np.percentile(smooth[local], 35), 0.0) * local
    total = float(weights.sum())
    if total <= 0:
        return float(peak_x), float(peak_y)
    return float((weights * xs).sum() / total), float((weights * ys).sum() / total)


def long_border_lines(gray, cfg):
    height, width = gray.shape
    scale = min(height, width)
    edges = cv2.Canny((gray * 255).astype(np.uint8), 70, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=max(12, int(0.13 * scale)),
        minLineLength=int(cfg["long_line_fraction"] * scale),
        maxLineGap=max(2, int(0.06 * scale)),
    )
    output = []
    if lines is None:
        return output
    margin = 0.07 * scale
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        near_border = lambda x, y: min(x, y, width - 1 - x, height - 1 - y) < margin
        length = math.hypot(x2 - x1, y2 - y1)
        if length == 0:
            continue
        border_supported = (near_border(x1, y1) and near_border(x2, y2)) or (
            length >= 0.68 * scale and (near_border(x1, y1) or near_border(x2, y2))
        )
        if not border_supported:
            continue
        normal_x, normal_y = -(y2 - y1) / length, (x2 - x1) / length
        t = np.linspace(0.1, 0.9, max(24, int(length)))
        sample_x, sample_y = x1 + t * (x2 - x1), y1 + t * (y2 - y1)

        def sample(offset):
            xi = np.clip(np.rint(sample_x + offset * normal_x).astype(int), 0, width - 1)
            yi = np.clip(np.rint(sample_y + offset * normal_y).astype(int), 0, height - 1)
            return gray[yi, xi]

        ridge = np.maximum.reduce([sample(-1), sample(0), sample(1)])
        contrast = ridge - 0.5 * (sample(-5) + sample(5))
        if np.median(contrast) > 0.18 and np.mean(contrast > 0.08) > 0.75:
            output.append((int(x1), int(y1), int(x2), int(y2)))
    return sorted(output, key=lambda row: math.hypot(row[2] - row[0], row[3] - row[1]),
                  reverse=True)[:12]


def measure_component(points_xy, residual, center, threshold, cfg, shape):
    height, width = shape
    scale = min(height, width)
    pts = points_xy.astype(np.float64)
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(centered, rowvar=False))
    eigenvalues = np.maximum(eigenvalues, 1e-6)
    major = eigenvectors[:, np.argmax(eigenvalues)]
    elongation = float(np.sqrt(eigenvalues.max() / eigenvalues.min()))
    radial = centroid - np.asarray(center)
    radius = float(np.linalg.norm(radial))
    tangent = np.array([-radial[1], radial[0]]) / max(radius, 1e-6)
    alignment = float(abs(np.dot(major, tangent)))
    offsets = pts - np.asarray(center)
    radii = np.linalg.norm(offsets, axis=1)
    span = angular_span_deg(np.arctan2(offsets[:, 1], offsets[:, 0]))
    radial_cv = float(np.std(radii) / max(float(np.mean(radii)), 1e-6))
    values = residual[points_xy[:, 1], points_xy[:, 0]]
    strength = float(np.mean(values) / max(threshold, 0.01))
    touches_border = bool(np.any(
        (pts[:, 0] <= 1) | (pts[:, 1] <= 1)
        | (pts[:, 0] >= width - 2) | (pts[:, 1] >= height - 2)
    ))
    features = {
        "tangential": unit_interval((alignment - 0.45) / 0.55),
        "elongation": unit_interval((elongation - 1.0) / 2.5),
        "angular_span": unit_interval((span - 8.0) / 52.0),
        "radial_coherence": unit_interval(1.0 - radial_cv / 0.30),
        "contrast": unit_interval((strength - 1.0) / 2.0),
    }
    component_score = sum(features[key] * weight for key, weight in (
        ("tangential", 0.30), ("elongation", 0.20), ("angular_span", 0.25),
        ("radial_coherence", 0.15), ("contrast", 0.10),
    ))
    accepted = (
        cfg["inner_radius_fraction"] * scale <= radius <= cfg["outer_radius_fraction"] * scale
        and alignment >= cfg["min_tangential_alignment"]
        and elongation >= cfg["min_elongation"]
        and span >= cfg["min_angular_span_deg"]
        and radial_cv <= cfg["max_radial_cv"]
        and not touches_border and len(pts) <= 0.08 * height * width
    )
    return {
        "centroid_x": float(centroid[0]), "centroid_y": float(centroid[1]),
        "radius_px": radius, "angular_span_deg": span,
        "tangential_alignment": alignment, "elongation": elongation,
        "radial_cv": radial_cv, "contrast_ratio": strength,
        "component_score": float(component_score), "accepted": bool(accepted),
        "features": features, "pixels": points_xy,
    }


def pair_agreement(components, center):
    if len(components) < 2:
        return 0.0
    best = 0.0
    for i, left in enumerate(components):
        for right in components[i + 1:]:
            left_angle = math.atan2(left["centroid_y"] - center[1], left["centroid_x"] - center[0])
            right_angle = math.atan2(right["centroid_y"] - center[1], right["centroid_x"] - center[0])
            separation = abs((left_angle - right_angle + math.pi) % (2 * math.pi) - math.pi)
            radius_gap = abs(left["radius_px"] - right["radius_px"]) / max(
                left["radius_px"], right["radius_px"], 1.0)
            best = max(best, unit_interval((math.degrees(separation) - 25) / 100)
                       * unit_interval(1.0 - radius_gap / 0.4))
    return best


def analyze_image(rgb, center_override=None, cfg=None):
    cfg = DETECTOR_CONFIG if cfg is None else cfg
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Expected an RGB uint8 image.")
    height, width = rgb.shape[:2]
    if min(height, width) < 48:
        return {"status": "insufficient_resolution", "score": None, "original": rgb}
    if height * width > 10_000_000:
        return {"status": "image_too_large", "score": None, "original": rgb}
    gray = normalize_gray(rgb)
    if gray is None:
        return {"status": "insufficient_contrast", "score": None, "original": rgb}
    center = center_override if center_override is not None else estimate_center(gray)
    center = (float(center[0]), float(center[1]))
    if not all(np.isfinite(center)) or not (0 <= center[0] < width and 0 <= center[1] < height):
        raise ValueError("Center outside the image.")
    scale = min(height, width)
    yy, xx = np.mgrid[:height, :width]
    distance = np.hypot(xx - center[0], yy - center[1])
    annulus = ((distance >= cfg["inner_radius_fraction"] * scale)
               & (distance <= cfg["outer_radius_fraction"] * scale))
    smooth = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(4.0, 0.055 * scale))
    residual = np.maximum(gray - smooth, 0.0)
    reference = residual[annulus]
    if not len(reference):
        return {"status": "empty_annulus", "score": None, "original": rgb}
    median = float(np.median(reference))
    noise = float(1.4826 * np.median(np.abs(reference - median)))
    threshold = max(float(np.percentile(reference, cfg["residual_percentile"])),
                    median + 2.5 * noise, 0.04)
    binary = ((residual > threshold) & annulus).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    artifact_lines = long_border_lines(gray, cfg)
    artifact_mask = np.zeros((height, width), dtype=np.uint8)
    for x1, y1, x2, y2 in artifact_lines:
        cv2.line(artifact_mask, (x1, y1), (x2, y2), 255,
                 max(2, round(0.018 * scale)), cv2.LINE_AA)
    binary[artifact_mask > 0] = 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    min_area = max(4, round(0.0002 * height * width))
    components = []
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] < min_area:
            continue
        ys, xs = np.where(labels == index)
        points = np.column_stack([xs, ys]).astype(np.int32)
        if len(points) >= 3:
            components.append(measure_component(points, residual, center, threshold, cfg, (height, width)))
    accepted = sorted((c for c in components if c["accepted"]),
                      key=lambda c: c["component_score"], reverse=True)
    pair = pair_agreement(accepted[:6], center)
    contributions = {
        "best_region": 65.0 * accepted[0]["component_score"] if accepted else 0.0,
        "second_region": 20.0 * accepted[1]["component_score"] if len(accepted) > 1 else 0.0,
        "multi_region_geometry": 15.0 * pair,
    }
    score = int(round(np.clip(sum(contributions.values()), 0.0, 100.0)))
    overlay = rgb.copy().astype(np.float32)
    for component in accepted[:6]:
        pixels = component["pixels"]
        original = overlay[pixels[:, 1], pixels[:, 0]]
        overlay[pixels[:, 1], pixels[:, 0]] = 0.45 * original + 0.55 * np.array([255, 55, 55])
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    for rank, component in enumerate(accepted[:6], start=1):
        cv2.putText(overlay, str(rank),
                    (int(component["centroid_x"]) + 3, int(component["centroid_y"]) - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    for x1, y1, x2, y2 in artifact_lines:
        cv2.line(overlay, (x1, y1), (x2, y2), (255, 165, 0), 1, cv2.LINE_AA)
    cv2.drawMarker(overlay, tuple(round(value) for value in center), (255, 230, 0),
                   markerType=cv2.MARKER_CROSS, markerSize=11, thickness=1)
    if accepted:
        radius = round(np.median([item["radius_px"] for item in accepted[:6]]))
        cv2.circle(overlay, tuple(round(value) for value in center), radius,
                   (0, 220, 220), 1, cv2.LINE_AA)
    return {
        "status": "exploratory_visual_only", "score": score,
        "original": rgb, "overlay": overlay, "center": center,
        "components": components, "accepted": accepted, "artifact_lines": artifact_lines,
        "threshold": threshold, "contributions": contributions, "pair_agreement": pair,
        "quality_flag": "possible_border_line" if artifact_lines else "",
    }


def score_image_bytes(image_bytes: bytes, *, include_overlay: bool = True) -> dict:
    """Return compact diagnostics; never replace a failed image with score zero."""
    with Image.open(BytesIO(image_bytes)) as image:
        if image.width * image.height > 10_000_000:
            return {"status": "image_too_large", "score": None}
        rgb = np.asarray(image.convert("RGB"))
    result = analyze_image(rgb)
    output = {key: result[key] for key in (
        "status", "score", "center", "contributions", "pair_agreement", "quality_flag"
    ) if key in result}
    output["regions"] = [
        {key: value for key, value in component.items() if key not in {"pixels", "accepted"}}
        for component in result.get("accepted", [])[:6]
    ]
    output["artifact_lines"] = result.get("artifact_lines", [])
    output["image_width"], output["image_height"] = rgb.shape[1], rgb.shape[0]
    if include_overlay and "overlay" in result:
        buffer = BytesIO()
        Image.fromarray(result["overlay"]).save(buffer, format="PNG")
        output["overlay_png"] = buffer.getvalue()
    return output
