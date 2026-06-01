from __future__ import annotations

import base64
import time
from collections import OrderedDict
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from .config import (
    CUTOUT_BASE,
    IMAGE_BYTES_CACHE_MAX_ITEMS,
    LENS_IMG_BASE,
    SUMMARY_THUMBNAIL_PIXEL_SIZE,
    THUMBNAIL_CACHE_MAX_ITEMS,
)
from .runtime import log_app_event
from .storage import gcs_filesystem, is_gcs_path, join_data_path, path_exists

def image_path_exists(path: str) -> bool:
    return path_exists(path)

def normalize_filename_object_id(object_id: object) -> str:
    value = str(object_id).strip()
    if value.endswith(".0"):
        value = value[:-2]
    if value.startswith("-"):
        return f"NEG{value[1:]}"
    return value.replace("-", "NEG")

def morphology_cutout_path(id_str: object, object_id: object = None) -> str | None:
    if pd.isna(id_str):
        return None

    parts = str(id_str).strip().split("_")
    if len(parts) == 2:
        tile_index = parts[0]
        object_id_part = parts[1]
    elif len(parts) >= 3:
        tile_index = parts[-2]
        object_id_part = parts[-1]
    else:
        return None

    candidate_object_ids = [object_id_part]
    if object_id is not None and not pd.isna(object_id):
        candidate_object_ids.append(object_id)

    for candidate_object_id in dict.fromkeys(
        normalize_filename_object_id(candidate) for candidate in candidate_object_ids
    ):
        filename = f"{tile_index}_{candidate_object_id}_gz_arcsinh_vis_only.jpg"
        path = join_data_path(CUTOUT_BASE, tile_index, filename)
        if image_path_exists(path):
            return path

    return None

def lens_image_path(lens_id_str: object) -> str | None:
    if pd.isna(lens_id_str):
        return None
    path = join_data_path(LENS_IMG_BASE, str(lens_id_str), "rgb_1.png")
    return path if image_path_exists(path) else None

def session_lru_cache(key: str) -> OrderedDict:
    cache = st.session_state.setdefault(key, OrderedDict())
    if not isinstance(cache, OrderedDict):
        cache = OrderedDict(cache)
        st.session_state[key] = cache
    return cache

def load_image_bytes(path: str) -> bytes:
    cache = session_lru_cache("image_bytes_cache")
    if path in cache:
        image_bytes = cache.pop(path)
        cache[path] = image_bytes
        return image_bytes

    started_at = time.perf_counter()
    if is_gcs_path(path):
        with gcs_filesystem().open(path, "rb") as image_file:
            image_bytes = image_file.read()
        source = "gcs"
    else:
        image_bytes = Path(path).read_bytes()
        source = "local"

    log_app_event(
        "image_loaded",
        duration_seconds=round(time.perf_counter() - started_at, 3),
        source=source,
        bytes=int(len(image_bytes)),
        suffix=Path(str(path)).suffix.lower(),
    )
    cache[path] = image_bytes
    while len(cache) > IMAGE_BYTES_CACHE_MAX_ITEMS:
        cache.popitem(last=False)
    return image_bytes

def thumbnail_image_src(path: str) -> str:
    cache = session_lru_cache("thumbnail_image_src_cache")
    if path in cache:
        image_src = cache.pop(path)
        cache[path] = image_src
        return image_src

    image_bytes = load_image_bytes(path)
    with Image.open(BytesIO(image_bytes)) as image:
        image = image.convert("RGB")
        image.thumbnail(
            (SUMMARY_THUMBNAIL_PIXEL_SIZE, SUMMARY_THUMBNAIL_PIXEL_SIZE),
            Image.Resampling.LANCZOS,
        )
        thumbnail_buffer = BytesIO()
        image.save(thumbnail_buffer, format="JPEG", quality=82, optimize=True)

    image_src = (
        "data:image/jpeg;base64,"
        f"{base64.b64encode(thumbnail_buffer.getvalue()).decode()}"
    )
    cache[path] = image_src
    while len(cache) > THUMBNAIL_CACHE_MAX_ITEMS:
        cache.popitem(last=False)
    return image_src

def show_image(path: str, caption: str, caption_markdown: str | None = None) -> None:
    try:
        image = Image.open(BytesIO(load_image_bytes(path)))
    except Exception as exc:
        st.warning(f"Could not open the image: {exc}")
        return
    if caption_markdown:
        st.image(image, use_container_width=True)
        st.markdown(
            f"""
            <div style="
                color: var(--text-color);
                font-family: inherit;
                font-size: 0.875rem;
                line-height: 1.25;
                margin-top: -0.35rem;
                opacity: 0.65;
                text-align: center;
                width: 100%;
            ">
                {caption_markdown}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.image(image, caption=caption, use_container_width=True)

def object_image_path(row: pd.Series, prefer_lens_image: bool = False) -> str | None:
    if prefer_lens_image:
        lens_path = lens_image_path(row.get("lens_id_str"))
        if lens_path is None:
            lens_path = lens_image_path(row.get("id_str"))
        if lens_path is not None:
            return lens_path

    return morphology_cutout_path(row.get("id_str"), row.get("object_id"))
