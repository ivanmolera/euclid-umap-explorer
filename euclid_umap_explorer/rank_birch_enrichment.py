"""Rank supported BIRCH configurations by enriched-cluster count and A/B/C ties."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .analyze_birch_enrichment import ENRICHMENT_CUTOFFS
from .birch_grid_benchmark import CONFIGURATIONS


CATALOGUE_OBJECTS = {"filtered": 374_982, "full": 380_111}
REFERENCE_COUNT = 1_416
MIN_OBJECTS = 100
MIN_REFERENCES = 50


def rank_configurations(clusters: pd.DataFrame) -> pd.DataFrame:
    required = {
        "catalogue", "threshold", "branching_factor", "cluster", "n_objects",
        "n_labelled_abc", "n_grade_a", "n_grade_b", "n_grade_c",
    }
    if not required.issubset(clusters.columns):
        raise ValueError(f"Missing cluster columns: {sorted(required - set(clusters.columns))}")
    if clusters.duplicated(["catalogue", "threshold", "branching_factor", "cluster"]).any():
        raise ValueError("Repeated cluster within a BIRCH configuration")
    if not set(clusters.catalogue).issubset(CATALOGUE_OBJECTS):
        raise ValueError("Unknown catalogue")
    expected = set(CONFIGURATIONS)
    if not set(zip(clusters.threshold, clusters.branching_factor)).issubset(expected):
        raise ValueError("Unexpected BIRCH configuration")
    counts = clusters[["n_objects", "n_labelled_abc", "n_grade_a", "n_grade_b", "n_grade_c"]]
    if (not np.isfinite(counts.to_numpy()).all() or (counts < 0).any().any()
            or not (counts == np.floor(counts)).all().all()
            or clusters.n_objects.le(0).any()
            or clusters.n_labelled_abc.gt(clusters.n_objects).any()
            or not clusters[["n_grade_a", "n_grade_b", "n_grade_c"]].sum(axis=1).eq(
                clusters.n_labelled_abc
            ).all()):
        raise ValueError("Invalid object or A/B/C counts")

    supported = clusters.loc[
        clusters.n_objects.ge(MIN_OBJECTS) & clusters.n_labelled_abc.ge(MIN_REFERENCES)
    ].copy()
    supported["exact_enrichment"] = (
        supported.n_labelled_abc / supported.n_objects
        / (REFERENCE_COUNT / supported.catalogue.map(CATALOGUE_OBJECTS))
    )
    groups = {
        key: group for key, group in supported.groupby(
            ["catalogue", "threshold", "branching_factor"], sort=False
        )
    }
    empty = supported.iloc[:0]
    rows = []
    for catalogue in CATALOGUE_OBJECTS:
        for threshold, branching in CONFIGURATIONS:
            group = groups.get((catalogue, threshold, branching), empty)
            concentrated = group.loc[group.exact_enrichment.ge(50)]
            rows.append({
                "catalogue": catalogue, "threshold": threshold,
                "branching_factor": branching, "batch_size": 25_000,
                "min_objects": MIN_OBJECTS, "min_references": MIN_REFERENCES,
                **{f"clusters_ge_{cutoff}x": int(group.exact_enrichment.ge(cutoff).sum())
                   for cutoff in ENRICHMENT_CUTOFFS},
                **{f"grade_{grade}_ge_50x": int(concentrated[f"n_grade_{grade}"].sum())
                   for grade in ("a", "b", "c")},
                "references_ge_50x": int(concentrated.n_labelled_abc.sum()),
                "objects_ge_50x": int(concentrated.n_objects.sum()),
                "unknown_ge_50x": int(
                    concentrated.n_objects.sum() - concentrated.n_labelled_abc.sum()
                ),
            })
    result = pd.DataFrame(rows).sort_values(
        ["catalogue", "clusters_ge_50x", "grade_a_ge_50x", "grade_b_ge_50x",
         "grade_c_ge_50x", "threshold", "branching_factor"],
        ascending=[True, False, False, False, False, True, True],
        kind="stable",
    ).reset_index(drop=True)
    result.insert(1, "rank", result.groupby("catalogue").cumcount() + 1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cluster_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    result = rank_configurations(pd.read_csv(args.cluster_csv))
    outputs = {"configuration_ranking.csv": result}
    for catalogue in CATALOGUE_OBJECTS:
        outputs[f"top10_{catalogue}.csv"] = result.loc[result.catalogue.eq(catalogue)].head(10)
    for filename in outputs:
        if (args.output_dir / filename).exists():
            raise FileExistsError(args.output_dir / filename)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, frame in outputs.items():
        frame.to_csv(args.output_dir / filename, index=False)
    print(result.loc[result["rank"].le(5)].to_string(index=False))


if __name__ == "__main__":
    main()
