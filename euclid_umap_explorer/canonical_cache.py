"""Compact, source-versioned image scores, independent of clusters and labels."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import PIL

from .config import CACHE_DIR
from .lens_plausibility import DETECTOR_CONFIG, DETECTOR_VERSION, score_image_bytes
from .storage import is_gcs_path


CACHE_SCHEMA = 1
DETERMINISTIC_STATUSES = {
    "exploratory_visual_only", "insufficient_resolution", "insufficient_contrast",
    "image_too_large", "empty_annulus",
}


def detector_fingerprint():
    return {"version": DETECTOR_VERSION, "config": DETECTOR_CONFIG,
            "opencv": cv2.__version__, "numpy": np.__version__, "pillow": PIL.__version__}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class CanonicalScoreStore:
    def __init__(self, *, filesystem=None, cache_base=None):
        self.filesystem = filesystem
        self.cache_base = cache_base or os.getenv("EUCLID_CANONICAL_CACHE_BASE")
        self.detector = detector_fingerprint()
        self.cache_issue = None
        self._remote_writable = True
        self._remote_readable = True
        self._lock = Lock()

    def source_signature(self, path):
        if is_gcs_path(path):
            info = self.filesystem.info(path)
            if info.get("generation") is None:
                raise ValueError("Missing cutout generation.")
            return {"path": path, "generation": str(info["generation"]), "size": int(info["size"])}
        stat = Path(path).stat()
        return {"path": str(Path(path).resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}

    def _cache_path(self, source):
        base = self.cache_base
        if not base:
            if is_gcs_path(source["path"]):
                bucket = source["path"].removeprefix("gs://").split("/", 1)[0]
                base = f"gs://{bucket}/analysis_cache/canonical_scores"
            else:
                base = str(CACHE_DIR / "canonical_scores")
        key = _digest({"schema": CACHE_SCHEMA, "source": source, "detector": self.detector})
        return f"{str(base).rstrip('/')}/v{CACHE_SCHEMA}/{key}.json"

    def _issue(self, exc, *, write=False):
        with self._lock:
            self.cache_issue = type(exc).__name__
            if isinstance(exc, PermissionError) or getattr(exc, "code", None) in (401, 403):
                if write:
                    self._remote_writable = False
                else:
                    self._remote_readable = False

    def _read(self, path):
        if is_gcs_path(path):
            if not self._remote_readable:
                return None
            with self.filesystem.open(path, "rb") as source:
                return source.read()
        target = Path(path)
        return target.read_bytes() if target.is_file() else None

    def _load(self, path, source):
        try:
            payload = self._read(path)
            if payload is None:
                return None
            entry = json.loads(payload)
            evidence = entry["evidence"]
            if (entry["schema"] != CACHE_SCHEMA or entry["source"] != source
                    or entry["detector"] != self.detector
                    or evidence["status"] not in DETERMINISTIC_STATUSES):
                return None
            score = evidence["score"]
            if evidence["status"] == "exploratory_visual_only":
                if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 100:
                    return None
            elif score is not None:
                return None
            if not isinstance(evidence.get("image_sha256"), str) or len(evidence["image_sha256"]) != 64:
                return None
            if any(key in evidence for key in ("original_bytes", "overlay_png", "pixels")):
                return None
            return evidence
        except FileNotFoundError:
            return None
        except (ValueError, KeyError, TypeError):
            return None
        except Exception as exc:
            self._issue(exc)
            return None

    def _save(self, path, source, evidence):
        payload = json.dumps({"schema": CACHE_SCHEMA, "source": source,
                              "detector": self.detector, "evidence": evidence}, allow_nan=False).encode()
        try:
            if is_gcs_path(path):
                if self._remote_writable:
                    with self.filesystem.open(path, "wb") as target:
                        target.write(payload)
                return
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as temporary:
                temp_path = Path(temporary.name)
                temporary.write(payload)
            try:
                temp_path.replace(target)
            finally:
                temp_path.unlink(missing_ok=True)
        except Exception as exc:
            self._issue(exc, write=True)

    def score_candidates(self, candidates):
        for path in candidates:
            try:
                source = self.source_signature(path)
            except FileNotFoundError:
                continue
            cache_path = self._cache_path(source)
            cached = self._load(cache_path, source)
            if cached is not None:
                return cached, True
            if is_gcs_path(path):
                # Pin bytes to the generation used in the cache key.
                with self.filesystem.open(path, "rb", generation=source["generation"], timeout=30) as image:
                    image_bytes = image.read()
            else:
                image_bytes = Path(path).read_bytes()
                if self.source_signature(path) != source:
                    raise OSError("Cutout changed while reading.")
            evidence = {**score_image_bytes(image_bytes, include_overlay=False), "image_path": path,
                        "source_signature": source, "image_sha256": hashlib.sha256(image_bytes).hexdigest()}
            if evidence["status"] in DETERMINISTIC_STATUSES:
                self._save(cache_path, source, evidence)
            return evidence, False
        return {"status": "missing_cutout", "score": None}, False
