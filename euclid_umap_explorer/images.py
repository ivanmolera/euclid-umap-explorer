from __future__ import annotations

import base64
import time
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from .config import CUTOUT_BASE, LENS_IMG_BASE
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

def load_image_bytes(path: str) -> bytes:
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
    return image_bytes

def thumbnail_image_src(path: str) -> str:
    image_bytes = load_image_bytes(path)
    Image.open(BytesIO(image_bytes)).verify()
    image_type = "png" if str(path).lower().endswith(".png") else "jpeg"
    return f"data:image/{image_type};base64,{base64.b64encode(image_bytes).decode()}"

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
                color: rgba(49, 51, 63, 0.6);
                font-family: inherit;
                font-size: 0.875rem;
                line-height: 1.25;
                margin-top: -0.35rem;
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
