"""Count well-supported, enriched BIRCH clusters without refitting."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .birch_grid_benchmark import CONFIGURATIONS


REFERENCE_COUNT = 1_416
OBJECT_COUNT = 374_982
ENRICHMENT_CUTOFFS = (2, 5, 10, 15, 25, 30, 40, 50)
SUPPORT_RULES = ((10, 100), (20, 100), (50, 100), (10, 0))
MAIN_SUPPORT = (50, 100)
CONFIG_NAME = re.compile(r"t(\d+\.\d)_b(\d+)\.parquet$")


def analyze(input_dir: Path, output_dir: Path, expected_objects: int = OBJECT_COUNT,
            expected_references: int = REFERENCE_COUNT) -> None:
    files = sorted(input_dir.glob("t*_b*.parquet"))
    expected = set(CONFIGURATIONS)
    if len(files) != len(expected):
        raise ValueError(f"Expected {len(expected)} cluster summaries, found {len(files)}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    notable = []
    seen = set()
    population_rate = expected_references / expected_objects
    for file in files:
        match = CONFIG_NAME.fullmatch(file.name)
        if match is None:
            raise ValueError(f"Unexpected cluster-summary filename: {file.name}")
        threshold, branching = float(match[1]), int(match[2])
        if (threshold, branching) not in expected or (threshold, branching) in seen:
            raise ValueError(f"Unexpected or repeated BIRCH configuration: {file.name}")
        seen.add((threshold, branching))

        clusters = pd.read_parquet(file)
        if (int(clusters["n_objects"].sum()) != expected_objects
                or int(clusters["n_references"].sum()) != expected_references
                or (clusters["n_objects"] <= 0).any()
                or (clusters["n_references"] > clusters["n_objects"]).any()):
            raise ValueError(f"Invalid object/reference totals: {file.name}")
        rate = clusters["n_references"] / clusters["n_objects"]
        enrichment = rate / population_rate
        if not np.allclose(rate, clusters["lens_rate"]) or not np.allclose(
            enrichment, clusters["enrichment_x"]
        ):
            raise ValueError(f"Inconsistent prevalence or enrichment: {file.name}")

        if (threshold, branching) in ((6.2, 50), (5.7, 49)):
            supported = clusters.loc[
                (clusters["n_references"] >= MAIN_SUPPORT[0])
                & (clusters["n_objects"] >= MAIN_SUPPORT[1])
            ].copy()
            supported.insert(0, "branching_factor", branching)
            supported.insert(0, "threshold", threshold)
            notable.append(supported)

        for min_references, min_objects in SUPPORT_RULES:
            base = (
                (clusters["n_references"] >= min_references)
                & (clusters["n_objects"] >= min_objects)
            )
            for cutoff in ENRICHMENT_CUTOFFS:
                chosen = clusters.loc[base & (enrichment >= cutoff)]
                rows.append({
                    "threshold": threshold,
                    "branching_factor": branching,
                    "n_clusters_total": len(clusters),
                    "min_enrichment_x": cutoff,
                    "min_references": min_references,
                    "min_objects": min_objects,
                    "max_labelled_fraction": None,
                    "qualifying_clusters": len(chosen),
                    "objects_in_qualifying_clusters": int(chosen["n_objects"].sum()),
                    "references_in_qualifying_clusters": int(chosen["n_references"].sum()),
                })

    if seen != expected:
        raise ValueError(f"Missing {len(expected - seen)} BIRCH configurations")
    counts = pd.DataFrame(rows)
    counts.to_csv(output_dir / "counts_by_configuration_and_support.csv", index=False)

    for min_references in (10, 20, 50):
        supported = counts.loc[
            (counts["min_references"] == min_references)
            & (counts["min_objects"] == 100)
        ]
        wide = supported.pivot(
            index=["threshold", "branching_factor", "n_clusters_total"],
            columns="min_enrichment_x", values="qualifying_clusters",
        ).reset_index()
        wide.columns = [f"ge_{column}x" if isinstance(column, int) else column for column in wide.columns]
        wide.to_csv(output_dir / f"cluster_counts_{min_references}refs_100objects.csv", index=False)

    summaries = []
    for (min_references, min_objects, cutoff), group in counts.groupby(
        ["min_references", "min_objects", "min_enrichment_x"], sort=True
    ):
        values = group["qualifying_clusters"]
        baseline = group.loc[(group["threshold"] == 6.2) & (group["branching_factor"] == 50)].iloc[0]
        selected = group.loc[(group["threshold"] == 5.7) & (group["branching_factor"] == 49)].iloc[0]
        summaries.append({
            "min_enrichment_x": cutoff, "min_references": min_references,
            "min_objects": min_objects, "max_labelled_fraction": None,
            "configurations_with_any": int((values > 0).sum()),
            "min_clusters": int(values.min()), "median_clusters": float(values.median()),
            "max_clusters": int(values.max()),
            "default_6p2_b50_clusters": int(baseline["qualifying_clusters"]),
            "selected_5p7_b49_clusters": int(selected["qualifying_clusters"]),
            "default_references_in_qualifying_clusters": int(baseline["references_in_qualifying_clusters"]),
            "selected_references_in_qualifying_clusters": int(selected["references_in_qualifying_clusters"]),
        })
    summary = pd.DataFrame(summaries).sort_values(
        ["min_references", "min_objects", "min_enrichment_x"]
    )
    summary.to_csv(output_dir / "cutoff_summary.csv", index=False)
    pd.concat(notable, ignore_index=True).sort_values(
        ["threshold", "branching_factor", "n_references", "enrichment_x"],
        ascending=[True, True, False, False],
    ).to_csv(output_dir / "notable_clusters_default_and_selected.csv", index=False)
    print(summary.loc[
        (summary["min_references"] == MAIN_SUPPORT[0])
        & (summary["min_objects"] == MAIN_SUPPORT[1])
    ].to_string(index=False))
    print(f"Saved analysis to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cluster_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--expected-objects", type=int, default=OBJECT_COUNT)
    parser.add_argument("--expected-references", type=int, default=REFERENCE_COUNT)
    args = parser.parse_args()
    analyze(args.cluster_dir, args.output_dir, args.expected_objects, args.expected_references)


if __name__ == "__main__":
    main()
