from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from .config import (
    CACHE_DIR,
    DEFAULT_BIRCH_BATCH_SIZE,
    DEFAULT_BIRCH_BRANCHING_FACTOR,
    DEFAULT_BIRCH_FEATURES,
    DEFAULT_BIRCH_SCALING,
    DEFAULT_BIRCH_THRESHOLD,
)
from .storage import is_gcs_path


CACHE_SCHEMA = 2
PREPARATION_SCHEMA = 1
_LEGACY_CONFIGURATIONS = (
    (5.0, DEFAULT_BIRCH_BRANCHING_FACTOR),
    (5.5, DEFAULT_BIRCH_BRANCHING_FACTOR),
    (DEFAULT_BIRCH_THRESHOLD, DEFAULT_BIRCH_BRANCHING_FACTOR),
    (5.5, 25),
)


def _gcs_filesystem():
    import gcsfs

    return gcsfs.GCSFileSystem(skip_instance_cache=True)


def _digest(value: dict) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:24]


def _cache_base(parquet_path: str) -> str:
    base = os.getenv("EUCLID_BIRCH_CACHE_BASE")
    if not base:
        if is_gcs_path(parquet_path):
            bucket = parquet_path.removeprefix("gs://").split("/", 1)[0]
            base = f"gs://{bucket}/analysis_cache/birch"
        else:
            base = str(CACHE_DIR / "birch")
    return base.rstrip("/")


def cache_path(
    parquet_path: str,
    threshold: float = DEFAULT_BIRCH_THRESHOLD,
    branching_factor: int = DEFAULT_BIRCH_BRANCHING_FACTOR,
    batch_size: int = DEFAULT_BIRCH_BATCH_SIZE,
    selected_features: tuple[str, ...] = DEFAULT_BIRCH_FEATURES,
    scaling: str = DEFAULT_BIRCH_SCALING,
) -> str:
    manifest = _expected_manifest(
        {"path": parquet_path}, threshold, branching_factor,
        batch_size, selected_features, scaling,
    )
    return f"{_cache_base(parquet_path)}/v{CACHE_SCHEMA}/{_digest(manifest)}.npz"


def legacy_cache_path(
    parquet_path: str,
    threshold: float,
    branching_factor: int,
    batch_size: int,
    selected_features: tuple[str, ...],
    scaling: str,
) -> str | None:
    if (
        (threshold, branching_factor) not in _LEGACY_CONFIGURATIONS
        or batch_size != DEFAULT_BIRCH_BATCH_SIZE
        or tuple(selected_features) != DEFAULT_BIRCH_FEATURES
        or scaling != DEFAULT_BIRCH_SCALING
    ):
        return None
    source_key = hashlib.sha256(parquet_path.encode("utf-8")).hexdigest()[:20]
    name = (
        "default"
        if threshold == DEFAULT_BIRCH_THRESHOLD
        else f"threshold-{threshold:.1f}".replace(".", "p")
    )
    if branching_factor != DEFAULT_BIRCH_BRANCHING_FACTOR:
        name += f"-branching-{branching_factor}"
    return f"{_cache_base(parquet_path)}/{name}-v1-{source_key}.npz"


def source_signature(parquet_path: str) -> dict[str, str | int]:
    if is_gcs_path(parquet_path):
        info = _gcs_filesystem().info(parquet_path)
        generation = info.get("generation")
        if generation is None:
            raise ValueError("The PCA object has no Cloud Storage generation.")
        return {
            "path": parquet_path,
            "generation": str(generation),
            "size": int(info["size"]),
        }
    stat = Path(parquet_path).stat()
    return {
        "path": str(Path(parquet_path).resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def ordered_id_digest(ids: pd.Series) -> str:
    digest = hashlib.sha256()
    for object_id in ids:
        encoded = str(object_id).encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _expected_manifest(
    source: dict[str, str | int], threshold: float, branching_factor: int,
    batch_size: int = DEFAULT_BIRCH_BATCH_SIZE,
    selected_features: tuple[str, ...] = DEFAULT_BIRCH_FEATURES,
    scaling: str = DEFAULT_BIRCH_SCALING,
) -> dict:
    return {
        "schema": CACHE_SCHEMA,
        "source": source,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "preparation_schema": PREPARATION_SCHEMA,
        "threshold": float(threshold),
        "branching_factor": int(branching_factor),
        "batch_size": int(batch_size),
        "features": list(selected_features),
        "scaling": scaling,
    }


def _read_bytes(path: str) -> bytes | None:
    if is_gcs_path(path):
        filesystem = _gcs_filesystem()
        if not filesystem.exists(path):
            return None
        with filesystem.open(path, "rb") as file:
            return file.read()
    source = Path(path)
    return source.read_bytes() if source.is_file() else None


def _write_bytes(path: str, payload: bytes) -> None:
    if is_gcs_path(path):
        with _gcs_filesystem().open(path, "wb") as file:
            file.write(payload)
        return

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as file:
        temporary = Path(file.name)
    try:
        temporary.write_bytes(payload)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def load_cached_projection(
    path: str,
    source: dict[str, str | int],
    threshold: float = DEFAULT_BIRCH_THRESHOLD,
    branching_factor: int = DEFAULT_BIRCH_BRANCHING_FACTOR,
    batch_size: int = DEFAULT_BIRCH_BATCH_SIZE,
    selected_features: tuple[str, ...] = DEFAULT_BIRCH_FEATURES,
    scaling: str = DEFAULT_BIRCH_SCALING,
):
    from .birch import BirchProjection

    payload = _read_bytes(path)
    if payload is None:
        return None
    with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
        manifest = json.loads(arrays["manifest"].tobytes().decode("utf-8"))
        expected = _expected_manifest(
            source, threshold, branching_factor, batch_size, selected_features, scaling
        )
        if manifest.get("schema") == 1:
            expected["schema"] = 1
            for key in ("numpy_version", "preparation_schema"):
                expected.pop(key)
        if any(manifest.get(key) != value for key, value in expected.items()):
            return None

        labels = arrays["labels"].astype(np.int32, copy=True)
        centers = arrays["centers"].astype(np.float32, copy=True)
        center_labels = arrays["center_labels"].astype(np.int32, copy=True)
        cutoff_clusters = arrays["cutoff_clusters"].astype(np.int32, copy=True)
        cutoff_values = arrays["cutoff_values"].astype(np.float64, copy=True)
        scaler_mean = arrays["scaler_mean"].copy()
        scaler_scale = arrays["scaler_scale"].copy()
        global_cutoff = float(arrays["global_cutoff"])

    if (
        labels.ndim != 1
        or len(labels) != manifest.get("n_rows")
        or centers.ndim != 2
        or centers.shape[1] != len(selected_features)
        or len(center_labels) != len(centers)
        or not np.array_equal(center_labels, np.arange(len(centers)))
        or (len(labels) > 0 and (labels.min() < 0 or labels.max() >= len(centers)))
        or len(cutoff_clusters) != len(cutoff_values)
        or scaler_mean.shape != ((len(selected_features),) if scaling == "standard" else (0,))
        or scaler_scale.shape != scaler_mean.shape
        or not isinstance(manifest.get("ordered_id_sha256"), str)
    ):
        raise ValueError("The precomputed BIRCH result is incomplete.")

    projection = BirchProjection(
        feature_cols=tuple(selected_features),
        centers=centers,
        center_labels=center_labels,
        scaler_mean=scaler_mean if len(scaler_mean) else None,
        scaler_scale=scaler_scale if len(scaler_scale) else None,
        distance_cutoffs=dict(zip(cutoff_clusters.tolist(), cutoff_values.tolist())),
        global_distance_cutoff=global_cutoff,
    )
    return labels, projection, manifest["ordered_id_sha256"]


def save_cached_projection(
    path: str,
    source: dict[str, str | int],
    ids: pd.Series,
    labels: pd.Series,
    projection,
    threshold: float = DEFAULT_BIRCH_THRESHOLD,
    branching_factor: int = DEFAULT_BIRCH_BRANCHING_FACTOR,
    batch_size: int = DEFAULT_BIRCH_BATCH_SIZE,
    selected_features: tuple[str, ...] = DEFAULT_BIRCH_FEATURES,
    scaling: str = DEFAULT_BIRCH_SCALING,
) -> None:
    manifest = _expected_manifest(
        source, threshold, branching_factor, batch_size, selected_features, scaling
    )
    manifest["n_rows"] = len(ids)
    manifest["ordered_id_sha256"] = ordered_id_digest(ids)
    cutoffs = sorted(projection.distance_cutoffs.items())
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        manifest=np.frombuffer(json.dumps(manifest, sort_keys=True).encode(), dtype=np.uint8),
        labels=np.asarray(labels, dtype=np.int32),
        centers=projection.centers,
        center_labels=projection.center_labels,
        scaler_mean=(projection.scaler_mean if projection.scaler_mean is not None else np.array([])),
        scaler_scale=(projection.scaler_scale if projection.scaler_scale is not None else np.array([])),
        cutoff_clusters=np.array([cluster for cluster, _ in cutoffs], dtype=np.int32),
        cutoff_values=np.array([cutoff for _, cutoff in cutoffs], dtype=np.float64),
        global_cutoff=np.array(projection.global_distance_cutoff, dtype=np.float64),
    )
    _write_bytes(path, buffer.getvalue())


def _preparation_manifest(source, batch_size, selected_features, scaling) -> dict:
    return {
        "schema": PREPARATION_SCHEMA,
        "source": source,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "dtype": "float32",
        "batch_size": int(batch_size),
        "features": list(selected_features),
        "scaling": scaling,
    }


def prepared_cache_path(source, batch_size, selected_features, scaling) -> str:
    base = Path(os.getenv(
        "EUCLID_BIRCH_PREPARED_CACHE_DIR", CACHE_DIR / "birch" / "prepared"
    ))
    manifest = _preparation_manifest(source, batch_size, selected_features, scaling)
    return str(base / f"{_digest(manifest)}.npz")


def load_prepared_features(path, source, ids, batch_size, selected_features, scaling):
    if not Path(path).is_file():
        return None
    expected = _preparation_manifest(source, batch_size, selected_features, scaling)
    expected.update(n_rows=len(ids), ordered_id_sha256=ordered_id_digest(ids))
    with np.load(path, allow_pickle=False) as arrays:
        manifest = json.loads(arrays["manifest"].tobytes().decode("utf-8"))
        if manifest != expected:
            return None
        values = arrays["values"]
        mean = arrays["scaler_mean"]
        scale = arrays["scaler_scale"]
    if (
        values.shape != (len(ids), len(selected_features))
        or values.dtype != np.float32
        or mean.shape != ((len(selected_features),) if scaling == "standard" else (0,))
        or scale.shape != mean.shape
    ):
        raise ValueError("The prepared BIRCH features are incomplete.")
    return values, (mean if mean.size else None), (scale if scale.size else None)


def save_prepared_features(
    path, source, ids, batch_size, selected_features, scaling, values, mean, scale
) -> None:
    manifest = _preparation_manifest(source, batch_size, selected_features, scaling)
    manifest.update(n_rows=len(ids), ordered_id_sha256=ordered_id_digest(ids))
    buffer = io.BytesIO()
    np.savez(
        buffer,
        manifest=np.frombuffer(json.dumps(manifest, sort_keys=True).encode(), dtype=np.uint8),
        values=values,
        scaler_mean=mean if mean is not None else np.array([]),
        scaler_scale=scale if scale is not None else np.array([]),
    )
    _write_bytes(path, buffer.getvalue())
