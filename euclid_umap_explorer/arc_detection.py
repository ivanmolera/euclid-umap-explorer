from __future__ import annotations

import base64
import hashlib
from io import BytesIO

import numpy as np
from PIL import Image
from scipy import ndimage

from .config import ARC_DETECTION_CACHE_MAX_ITEMS


def _robust_normalize(gray: np.ndarray) -> np.ndarray:
    low, high = np.percentile(gray, [1, 99.5])
    if high <= low:
        return np.zeros_like(gray, dtype=np.float32)
    normalized = (gray - low) / (high - low)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def _component_geometry(yx: np.ndarray) -> tuple[float, float, float]:
    if len(yx) < 3:
        return 0.0, 0.0, 0.0

    centered = yx - yx.mean(axis=0, keepdims=True)
    covariance = np.cov(centered, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(covariance)
    major = float(np.sqrt(max(eigenvalues[-1], 0.0)))
    minor = float(np.sqrt(max(eigenvalues[0], 1e-6)))
    elongation = major / minor
    eccentricity = float(np.sqrt(max(0.0, 1.0 - (minor * minor) / max(major * major, 1e-6))))

    y_min, x_min = yx.min(axis=0)
    y_max, x_max = yx.max(axis=0)
    box_area = max(float((y_max - y_min + 1) * (x_max - x_min + 1)), 1.0)
    fill_fraction = float(len(yx) / box_area)
    return elongation, eccentricity, fill_fraction


def _positive_percentile(values: np.ndarray, percentile: float) -> float:
    positive = values[values > 1e-6]
    if len(positive) == 0:
        return float(np.percentile(values, percentile))
    return float(np.percentile(positive, percentile))


def detect_arc_mask(image: Image.Image) -> np.ndarray:
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    normalized = _robust_normalize(gray)

    smooth = ndimage.gaussian_filter(normalized, sigma=1.2)
    background = ndimage.gaussian_filter(normalized, sigma=8.0)
    residual = np.clip(smooth - background, 0.0, 1.0)

    sobel_y = ndimage.sobel(smooth, axis=0)
    sobel_x = ndimage.sobel(smooth, axis=1)
    gradient = np.hypot(sobel_y, sobel_x)
    gradient = _robust_normalize(gradient)

    bright_threshold = _positive_percentile(residual, 74.0)
    soft_bright_threshold = _positive_percentile(residual, 55.0)
    edge_threshold = _positive_percentile(gradient, 70.0)
    candidate = (residual >= bright_threshold) | (
        (residual >= soft_bright_threshold) & (gradient >= edge_threshold)
    )

    candidate = ndimage.binary_closing(candidate, structure=np.ones((3, 3)))

    labels, n_labels = ndimage.label(candidate)
    mask = np.zeros_like(candidate, dtype=bool)
    image_area = candidate.size
    min_area = max(12, int(image_area * 0.0002))
    max_area = max(min_area + 1, int(image_area * 0.08))

    for label_id in range(1, n_labels + 1):
        yx = np.argwhere(labels == label_id)
        area = len(yx)
        if area < min_area or area > max_area:
            continue

        elongation, eccentricity, fill_fraction = _component_geometry(yx)
        if elongation < 1.8 or eccentricity < 0.72:
            continue
        if fill_fraction > 0.88:
            continue

        mask[labels == label_id] = True

    return ndimage.binary_dilation(mask, structure=np.ones((3, 3)))


def arc_overlay_image(image: Image.Image, mask: np.ndarray) -> Image.Image:
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (255, 0, 0, 0))
    alpha = (mask.astype(np.uint8) * 150)
    overlay.putalpha(Image.fromarray(alpha, mode="L"))
    return Image.alpha_composite(base, overlay).convert("RGB")


def detect_arc_overlay_src(path: str, max_size: int = 900) -> str:
    cache_key = f"{path}|{max_size}|arc_detection_v1"
    digest = hashlib.sha1(cache_key.encode("utf-8")).hexdigest()

    from streamlit import session_state

    cache = session_state.setdefault("arc_detection_overlay_cache", {})
    if digest in cache:
        return cache[digest]

    from .images import load_image_bytes

    image_bytes = load_image_bytes(path)
    with Image.open(BytesIO(image_bytes)) as image:
        image = image.convert("RGB")
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        mask = detect_arc_mask(image)
        overlay_image = arc_overlay_image(image, mask)
        output = BytesIO()
        overlay_image.save(output, format="JPEG", quality=90, optimize=True)

    image_src = "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode()
    cache[digest] = image_src
    while len(cache) > ARC_DETECTION_CACHE_MAX_ITEMS:
        cache.pop(next(iter(cache)))
    return image_src
