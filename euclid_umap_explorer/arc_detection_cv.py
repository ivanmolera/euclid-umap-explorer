from __future__ import annotations

import base64
import hashlib
from io import BytesIO

import numpy as np
from PIL import Image

from .config import ARC_DETECTION_CACHE_MAX_ITEMS


LOW_PERCENTILE_THRESHOLD = 88.0
HIGH_PERCENTILE_THRESHOLD = 95.0
MIN_COMPONENT_AREA = 8
MIN_CONTOUR_POINTS = 5
MIN_AREA = 3.0
MIN_LENGTH = 5.0
MIN_ASPECT_RATIO = 1.2
MAX_FILL_RATIO = 0.9
MAX_CENTER_DISTANCE_FRAC = 0.50
BACKGROUND_KERNEL_SIZE = (31, 31)
MORPH_CONNECTIVITY = 4
CONTOUR_MASK_THICKNESS = 2
CONTOUR_COLOR_RGB = (255, 0, 0)


def _cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required to detect arc-like structures. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc
    return cv2


def _pil_to_rgb_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _robust_gray_normalize(gray: np.ndarray) -> np.ndarray:
    p1, p99 = np.percentile(gray, (1, 99))
    if p99 <= p1:
        return np.zeros_like(gray, dtype=np.uint8)
    gray_clip = np.clip(gray, p1, p99)
    return ((gray_clip - p1) / (p99 - p1 + 1e-6) * 255).astype(np.uint8)


def _background_subtracted_residual(gray: np.ndarray) -> np.ndarray:
    cv2 = _cv2()
    gray_norm = _robust_gray_normalize(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray_norm)
    background = cv2.GaussianBlur(gray_eq, BACKGROUND_KERNEL_SIZE, 0)
    residual = cv2.subtract(gray_eq, background)
    return cv2.normalize(residual, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _clean_binary_mask(residual: np.ndarray) -> np.ndarray:
    cv2 = _cv2()
    low_threshold = np.percentile(residual, LOW_PERCENTILE_THRESHOLD)
    high_threshold = np.percentile(residual, HIGH_PERCENTILE_THRESHOLD)
    binary_low = (residual >= low_threshold).astype(np.uint8) * 255
    binary_high = (residual >= high_threshold).astype(np.uint8) * 255

    # Keep the notebook's low/high hysteresis step explicit so the production
    # detector follows the same preprocessing path, even though the permissive
    # low-threshold mask below is the one used for final contours.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_low,
        connectivity=8,
    )
    binary_hysteresis = np.zeros_like(binary_low)
    for label in range(1, num_labels):
        component_mask = labels == label
        if not np.any(binary_high[component_mask]):
            continue
        if stats[label, cv2.CC_STAT_AREA] < MIN_COMPONENT_AREA:
            continue
        binary_hysteresis[component_mask] = 255

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_low,
        connectivity=MORPH_CONNECTIVITY,
    )
    binary_clean = np.zeros_like(binary_low)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= MIN_COMPONENT_AREA:
            binary_clean[labels == label] = 255
    return binary_clean


def _candidate_contours(binary_clean: np.ndarray) -> list[np.ndarray]:
    cv2 = _cv2()
    contours, _ = cv2.findContours(
        binary_clean,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE,
    )

    image_height, image_width = binary_clean.shape[:2]
    image_center = np.array([image_width / 2, image_height / 2])
    max_center_distance = MAX_CENTER_DISTANCE_FRAC * min(image_width, image_height)
    candidates = []

    for contour in contours:
        if len(contour) < MIN_CONTOUR_POINTS:
            continue

        area = float(cv2.contourArea(contour))
        length = float(cv2.arcLength(contour, closed=False))
        x, y, width, height = cv2.boundingRect(contour)
        bbox_area = width * height
        if bbox_area == 0:
            continue

        contour_center = np.array([x + width / 2, y + height / 2])
        center_distance = float(np.linalg.norm(contour_center - image_center))
        aspect_ratio = max(width, height) / max(1, min(width, height))
        fill_ratio = area / bbox_area

        if area < MIN_AREA:
            continue
        if length < MIN_LENGTH:
            continue
        if aspect_ratio < MIN_ASPECT_RATIO:
            continue
        if fill_ratio > MAX_FILL_RATIO:
            continue
        if center_distance > max_center_distance:
            continue

        candidates.append((length, contour))

    return [
        contour
        for _, contour in sorted(candidates, key=lambda item: item[0], reverse=True)
    ]


def _candidate_contours_for_image(image: Image.Image) -> tuple[np.ndarray, list[np.ndarray]]:
    cv2 = _cv2()
    rgb = _pil_to_rgb_array(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    residual = _background_subtracted_residual(gray)
    binary_clean = _clean_binary_mask(residual)
    return rgb, _candidate_contours(binary_clean)


def detect_arc_mask(image: Image.Image) -> np.ndarray:
    cv2 = _cv2()
    rgb, contours = _candidate_contours_for_image(image)
    height, width = rgb.shape[:2]

    mask = np.zeros((height, width), dtype=np.uint8)
    for contour in contours:
        cv2.drawContours(mask, [contour], -1, 255, thickness=CONTOUR_MASK_THICKNESS)

    return mask.astype(bool)


def arc_overlay_image(image: Image.Image) -> Image.Image:
    cv2 = _cv2()
    rgb, contours = _candidate_contours_for_image(image)
    output_rgb = rgb.copy()
    for contour in contours:
        cv2.drawContours(
            output_rgb,
            [contour],
            -1,
            CONTOUR_COLOR_RGB,
            thickness=CONTOUR_MASK_THICKNESS,
        )
    return Image.fromarray(output_rgb, mode="RGB")


def detect_arc_overlay_src(path: str, max_size: int = 900) -> str:
    cache_key = f"{path}|{max_size}|arc_detection_cv_contours_v1"
    digest = hashlib.sha1(cache_key.encode("utf-8")).hexdigest()

    from streamlit import session_state

    cache = session_state.setdefault("arc_detection_overlay_cache", {})
    if digest in cache:
        return cache[digest]

    from .images import load_image_bytes

    image_bytes = load_image_bytes(path)
    with Image.open(BytesIO(image_bytes)) as image:
        image = image.convert("RGB")
        overlay_image = arc_overlay_image(image)
        overlay_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        output = BytesIO()
        overlay_image.save(output, format="JPEG", quality=90, optimize=True)

    image_src = "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode()
    cache[digest] = image_src
    while len(cache) > ARC_DETECTION_CACHE_MAX_ITEMS:
        cache.pop(next(iter(cache)))
    return image_src
