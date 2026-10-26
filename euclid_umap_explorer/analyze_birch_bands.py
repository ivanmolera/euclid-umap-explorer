"""Summarize supported BIRCH clusters in disjoint enrichment bands."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .birch_grid_benchmark import CONFIGURATIONS


NAME = re.compile(r"t(\d+\.\d)_b(\d+)\.parquet$")
BAND_EDGES = [*range(0, 55, 5), np.inf]
BAND_NAMES = [f"{low}-{low + 5}x" for low in range(0, 50, 5)] + ["ge_50x"]


def summarize_bands(
    cluster_dir: Path,
    object_count: int,
    reference_count: int,
    min_objects: int = 100,
    min_references: int = 50,
) -> pd.DataFrame:
    files = sorted(cluster_dir.glob("t*_b*.parquet"))
    if len(files) != len(CONFIGURATIONS):
        raise ValueError(f"Expected {len(CONFIGURATIONS)} cluster summaries, found {len(files)}")

    expected = set(CONFIGURATIONS)
    seen = set()
    rows = []
    population_rate = reference_count / object_count
    for file in files:
        match = NAME.fullmatch(file.name)
        if match is None:
            raise ValueError(f"Unexpected cluster filename: {file.name}")
        config = (float(match[1]), int(match[2]))
        if config not in expected or config in seen:
            raise ValueError(f"Unexpected or repeated configuration: {file.name}")
        seen.add(config)

        clusters = pd.read_parquet(file, columns=["n_objects", "n_references", "enrichment_x"])
        if (int(clusters["n_objects"].sum()) != object_count
                or int(clusters["n_references"].sum()) != reference_count
                or (clusters["n_objects"] <= 0).any()
                or (clusters["n_references"] > clusters["n_objects"]).any()):
            raise ValueError(f"Invalid catalogue totals: {file.name}")
        rate = clusters["n_references"] / clusters["n_objects"]
        enrichment = rate / population_rate
        if not np.allclose(enrichment, clusters["enrichment_x"]):
            raise ValueError(f"Inconsistent enrichment: {file.name}")

        supported = clusters.loc[
            (clusters["n_objects"] >= min_objects)
            & (clusters["n_references"] >= min_references)
        ].copy()
        supported["enrichment_x"] = enrichment.loc[supported.index]
        bands = pd.cut(
            supported["enrichment_x"], BAND_EDGES, labels=BAND_NAMES,
            right=False, include_lowest=True,
        ).value_counts(sort=False)
        row = {
            "threshold": config[0],
            "branching_factor": config[1],
            "n_clusters_total": len(clusters),
            "n_supported_clusters": len(supported),
            **{name: int(bands[name]) for name in BAND_NAMES},
            "clusters_ge_10x": int((supported["enrichment_x"] >= 10).sum()),
            "clusters_ge_25x": int((supported["enrichment_x"] >= 25).sum()),
            "references_ge_10x": int(supported.loc[
                supported["enrichment_x"] >= 10, "n_references"
            ].sum()),
            "references_ge_50x": int(supported.loc[
                supported["enrichment_x"] >= 50, "n_references"
            ].sum()),
            "objects_ge_50x": int(supported.loc[
                supported["enrichment_x"] >= 50, "n_objects"
            ].sum()),
        }
        if sum(row[name] for name in BAND_NAMES) != row["n_supported_clusters"]:
            raise RuntimeError(f"Band counts do not add up: {file.name}")
        rows.append(row)

    if seen != expected:
        raise ValueError(f"Missing {len(expected - seen)} configurations")
    return pd.DataFrame(rows).sort_values(
        ["ge_50x", "references_ge_50x", "clusters_ge_10x",
         "threshold", "branching_factor"],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cluster_dir", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--objects", type=int, required=True)
    parser.add_argument("--references", type=int, default=1_416)
    parser.add_argument("--min-objects", type=int, default=100)
    parser.add_argument("--min-references", type=int, default=50)
    args = parser.parse_args()
    if min(args.objects, args.references, args.min_objects, args.min_references) < 1:
        parser.error("Counts must be positive")
    result = summarize_bands(
        args.cluster_dir, args.objects, args.references,
        args.min_objects, args.min_references,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_csv, index=False)
    print(result.head(5).to_string(index=False))
    print(f"Saved {len(result)} configurations to {args.output_csv}")


if __name__ == "__main__":
    main()
