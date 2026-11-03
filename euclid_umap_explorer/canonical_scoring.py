"""Apply learned weights to the frozen detector's original six-region shortlist."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import lens_plausibility


FEATURE_NAMES = ("tangential", "elongation", "angular_span", "radial_coherence", "contrast")
POOL_NAMES = ("best_region", "second_region", "geometry")
SCORING_VERSION = "learned-visual-evidence-1"
MODEL_PATH = Path(__file__).with_name("canonical_score_weights.json")
_MODEL_BYTES = MODEL_PATH.read_bytes()
_MODEL = json.loads(_MODEL_BYTES)
MODEL_SHA256 = hashlib.sha256(_MODEL_BYTES).hexdigest()
MODEL_NAME = _MODEL["selected_model"]
REGION_WEIGHTS = _MODEL["weights"]["region"]
POOL_WEIGHTS = _MODEL["weights"]["aggregation"]

for names, weights in ((FEATURE_NAMES, REGION_WEIGHTS), (POOL_NAMES, POOL_WEIGHTS)):
    if (set(weights) != set(names)
            or any(not math.isfinite(value) or value < 0 for value in weights.values())
            or not math.isclose(sum(weights.values()), 1., abs_tol=1e-6)):
        raise ValueError("Invalid learned canonical-score weights.")
if hashlib.sha256(Path(lens_plausibility.__file__).read_bytes()).hexdigest() != _MODEL["detector_sha256"]:
    raise ValueError("Learned canonical weights require the frozen training detector.")

SCORING_KEY = f"{SCORING_VERSION}:{MODEL_SHA256}"


def scoring_fingerprint():
    return {"version": SCORING_VERSION, "selected_model": MODEL_NAME, "weights_sha256": MODEL_SHA256,
            "detector_sha256": _MODEL["detector_sha256"], "region_cap": 6,
            "region_weights": dict(REGION_WEIGHTS), "aggregation_weights": dict(POOL_WEIGHTS)}


def score_image_bytes(image_bytes: bytes, *, include_overlay: bool = True) -> dict:
    result = dict(lens_plausibility.score_image_bytes(image_bytes, include_overlay=include_overlay))
    result["manual_score"] = result.get("score")
    result["scoring_model"] = scoring_fingerprint()
    if result.get("score") is None:
        return result

    # Selection, geometry and overlay numbering remain exactly as in training.
    regions = [dict(region) for region in result["regions"][:6]]
    features = np.zeros((6, len(FEATURE_NAMES)), dtype=np.float32)
    for index, region in enumerate(regions):
        features[index] = [region["features"][name] for name in FEATURE_NAMES]
    weights = np.array([REGION_WEIGHTS[name] for name in FEATURE_NAMES], dtype=np.float32)
    region_scores = (features * weights).sum(axis=1)
    order = sorted(range(len(regions)), key=lambda index: -float(region_scores[index]))
    best = sorted(region_scores, reverse=True)[:2]
    branches = np.array([best[0], best[1], result["pair_agreement"]], dtype=np.float32)
    pool = np.array([POOL_WEIGHTS[name] for name in POOL_NAMES], dtype=np.float32)
    parts = np.float32(100) * branches * pool
    result["contributions"] = dict(zip(("best_region", "second_region", "multi_region_geometry"),
                                       (float(value) for value in parts)))
    # Keep the unrounded index for sorting, matching learned_score in the notebook.
    result["score"] = float(np.clip(np.float32(100) * (branches * pool).sum(), 0., 100.))
    for rank, index in enumerate(order):
        region = regions[index]
        region["manual_component_score"] = region["component_score"]
        region["component_score"] = float(region_scores[index])
        region["learned_rank"] = rank + 1
        region["region_contribution_points"] = float(np.float32(100) * pool[rank] * region_scores[index]) if rank < 2 else 0.
    result["regions"] = regions
    return result
