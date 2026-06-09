from __future__ import annotations

import base64
import hashlib
from io import BytesIO

import numpy as np
from PIL import Image

from .config import ARC_DETECTION_CACHE_MAX_ITEMS


PERCENTILE_THRESHOLD = 94.0
MIN_CONTOUR_POINTS = 5
BACKGROUND_KERNEL_SIZE = (31, 31)
MORPH_KERNEL_SIZE = (3, 3)
CONTOUR_MASK_THICKNESS = 2


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


def _enhanced_residual(gray: np.ndarray) -> np.ndarray:
    cv2 = _cv2()
    gray_norm = _robust_gray_normalize(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray_norm)
    background = cv2.GaussianBlur(gray_eq, BACKGROUND_KERNEL_SIZE, 0)
    residual = cv2.subtract(gray_eq, background)
    return cv2.normalize(residual, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _candidate_binary_mask(residual: np.ndarray) -> np.ndarray:
    cv2 = _cv2()
    threshold = np.percentile(residual, PERCENTILE_THRESHOLD)
    binary_raw = (residual >= threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL_SIZE)
    binary_clean = cv2.morphologyEx(binary_raw, cv2.MORPH_OPEN, kernel, iterations=1)
    binary_clean = cv2.morphologyEx(binary_clean, cv2.MORPH_CLOSE, kernel, iterations=2)
    return cv2.dilate(binary_clean, kernel, iterations=1)


def _is_drawable_contour(contour: np.ndarray) -> bool:
    cv2 = _cv2()
    if len(contour) < MIN_CONTOUR_POINTS:
        return False

    _x, _y, width, height = cv2.boundingRect(contour)
    return width * height > 0


def detect_arc_mask(image: Image.Image) -> np.ndarray:
    cv2 = _cv2()
    rgb = _pil_to_rgb_array(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    residual = _enhanced_residual(gray)
    binary_clean = _candidate_binary_mask(residual)
    contours, _ = cv2.findContours(binary_clean, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    mask = np.zeros(gray.shape, dtype=np.uint8)
    for contour in contours:
        if _is_drawable_contour(contour):
            cv2.drawContours(mask, [contour], -1, 255, thickness=CONTOUR_MASK_THICKNESS)

    if np.any(mask):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL_SIZE)
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask.astype(bool)


def arc_overlay_image(image: Image.Image, mask: np.ndarray) -> Image.Image:
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (255, 0, 0, 0))
    alpha = (mask.astype(np.uint8) * 150)
    overlay.putalpha(Image.fromarray(alpha, mode="L"))
    return Image.alpha_composite(base, overlay).convert("RGB")


def detect_arc_overlay_src(path: str, max_size: int = 900) -> str:
    cache_key = f"{path}|{max_size}|arc_detection_opencv_all_contours_v1"
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
