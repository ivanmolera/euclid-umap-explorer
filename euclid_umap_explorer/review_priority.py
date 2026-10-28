"""Operational inspection priorities, not calibrated lens probabilities."""
from __future__ import annotations

import numpy as np
import pandas as pd


REVIEW_PRIORITY_HELP = (
    "Reference-supported: minimum objects and selected ABC labels, with at least one A/B. "
    "Ordered by A+B enrichment, A count, then canonical score. Morphology exploration: "
    "minimum objects, fewer ABC labels and score at or above the exploration threshold; "
    "ordered by score. Other clusters remain listed. These are inspection priorities, "
    "not confirmations. Unknown does not mean non-lens; N/A is not a zero score."
)
AB_ENRICHMENT_HELP = (
    "(A+B in cluster / cluster objects) divided by (selected A+B in catalogue / catalogue objects). "
    "Uses only grades selected for this BIRCH run. N/A if no A/B references are selected."
)


def prioritize_for_review(summary, *, min_objects=100, min_labels=25, exploration_score=70.):
    if min_objects < 1 or min_labels < 1 or not 0 <= exploration_score <= 100:
        raise ValueError("Invalid review thresholds.")
    result = summary.copy()
    result["canonical_score"] = pd.to_numeric(result.canonical_score, errors="coerce").astype(float)
    ab = result.n_grade_a + result.n_grade_b
    baseline = ab.sum() / result.n_objects.sum() if result.n_objects.sum() else 0.
    result["enrichment_ab"] = (ab / result.n_objects / baseline) if baseline else np.nan
    reference = result.n_objects.ge(min_objects) & result.n_lenses.ge(min_labels) & ab.gt(0)
    exploration = (result.n_objects.ge(min_objects) & result.n_lenses.lt(min_labels)
                   & result.canonical_score.ge(exploration_score))
    result["review_priority"] = np.select([reference, exploration],
                                           ["Reference-supported", "Morphology exploration"], default="Other")
    result["priority_min_objects"] = min_objects
    result["priority_min_abc"] = min_labels
    result["priority_exploration_score"] = exploration_score
    result["_group"] = np.select([reference, exploration], [0, 1], default=2)
    result["_first"] = result.enrichment_ab.where(~exploration, result.canonical_score)
    result["_second"] = result.n_grade_a.astype(float).where(~exploration, result.enrichment_ab)
    result["_third"] = result.canonical_score.where(~exploration, result.n_grade_a)
    return result.sort_values(["_group", "_first", "_second", "_third", "n_objects", "cluster"],
                              ascending=[True, False, False, False, False, True], na_position="last",
                              kind="stable").drop(columns=["_group", "_first", "_second", "_third"]).reset_index(drop=True)
