from __future__ import annotations

import shutil
from pathlib import Path

import streamlit as st

from .config import CACHE_DIR, USE_LOCAL_CACHE
from .runtime import log_app_event

def is_gcs_path(path: str) -> bool:
    return str(path).startswith("gs://")

@st.cache_resource(show_spinner=False)
def gcs_filesystem():
    import gcsfs

    return gcsfs.GCSFileSystem()

def path_exists(path: str) -> bool:
    if is_gcs_path(path):
        return gcs_filesystem().exists(path)
    return Path(path).exists()

def join_data_path(base: str, *parts: str) -> str:
    clean_base = str(base).rstrip("/")
    clean_parts = [str(part).strip("/") for part in parts if str(part).strip("/")]
    if not clean_parts:
        return clean_base
    return f"{clean_base}/{'/'.join(clean_parts)}"

def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"

def cache_target_for_path(path: str) -> Path:
    source = Path(path)
    stat = source.stat()
    cache_name = f"{source.stem}-{stat.st_size}-{int(stat.st_mtime)}{source.suffix}"
    return CACHE_DIR / cache_name

def copy_file_to_cache(source: Path, target: Path, progress=None) -> None:
    chunk_size = 8 * 1024 * 1024
    total_size = source.stat().st_size
    copied = 0
    tmp_target = target.with_name(f"{target.name}.tmp")

    try:
        with source.open("rb") as src, tmp_target.open("wb") as dst:
            while True:
                chunk = src.read(chunk_size)
                if not chunk:
                    break
                dst.write(chunk)
                copied += len(chunk)
                if progress is not None and total_size > 0:
                    progress.progress(
                        min(copied / total_size, 1.0),
                        text=f"{format_bytes(copied)} / {format_bytes(total_size)}",
                    )
        tmp_target.replace(target)
    except Exception:
        tmp_target.unlink(missing_ok=True)
        raise

def cached_input_path(path: str, progress=None):
    if is_gcs_path(path):
        return path

    source = Path(path)
    if not USE_LOCAL_CACHE or not source.exists() or source.is_dir():
        return source

    target = cache_target_for_path(path)
    if target.exists() and target.stat().st_size == source.stat().st_size:
        return target

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    copy_file_to_cache(source, target, progress=progress)
    return target

def prepare_catalog_cache(paths: list[str]) -> None:
    if not USE_LOCAL_CACHE:
        return

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    copied_any = False

    with st.status("Preparing catalogues in local cache", expanded=True) as status:
        st.write(f"Cache: `{CACHE_DIR}`")
        for path in paths:
            if is_gcs_path(path):
                st.write(f"Using catalogue in Cloud Storage: `{path}`")
                continue

            source = Path(path)
            if not source.exists() or source.is_dir():
                continue

            size = source.stat().st_size
            target = cache_target_for_path(path)
            if target.exists() and target.stat().st_size == size:
                st.write(f"Already cached: `{source.name}` ({format_bytes(size)})")
                continue

            copied_any = True
            st.write(
                f"Copying `{source.name}` ({format_bytes(size)}) from the configured source..."
            )
            progress = st.progress(0.0, text=f"0 B / {format_bytes(size)}")
            try:
                cached_input_path(path, progress=progress)
            except TimeoutError as exc:
                progress.empty()
                st.error(
                    "The data source timed out while serving this file. "
                    "If you are using a synchronized drive, make it available offline and try again."
                )
                status.update(label="Could not copy a catalogue", state="error")
                raise exc
            progress.empty()
            st.write(f"Copied: `{source.name}` ({format_bytes(size)})")

        if copied_any:
            status.update(label="Catalogues copied to local cache", state="complete")
        else:
            status.update(label="Catalogues already available in local cache", state="complete")
