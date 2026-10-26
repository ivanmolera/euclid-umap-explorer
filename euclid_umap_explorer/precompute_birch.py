from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import storage
from .birch import _run_birch_clustering_impl
from .birch_cache import (
    cache_path,
    load_cached_projection,
    save_cached_projection,
)
from .config import (
    DEFAULT_BIRCH_BATCH_SIZE,
    DEFAULT_BIRCH_BRANCHING_FACTOR,
    DEFAULT_BIRCH_FEATURES,
    DEFAULT_BIRCH_SCALING,
    DEFAULT_BIRCH_THRESHOLD,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute an ESCOPE BIRCH cache result."
    )
    parser.add_argument("--local-parquet", required=True)
    parser.add_argument("--source-uri", required=True)
    parser.add_argument("--generation", required=True)
    parser.add_argument("--size", required=True, type=int)
    parser.add_argument("--lens-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_BIRCH_THRESHOLD,
    )
    parser.add_argument(
        "--branching-factor", type=int, default=DEFAULT_BIRCH_BRANCHING_FACTOR
    )
    parser.add_argument("--expected-objects", type=int)
    parser.add_argument("--expected-clusters", type=int)
    args = parser.parse_args()
    local_parquet = Path(args.local_parquet)
    if local_parquet.stat().st_size != args.size:
        raise ValueError("Local Parquet size differs from the Cloud Storage source.")
    storage.USE_LOCAL_CACHE = False
    clustered, features, projection = _run_birch_clustering_impl(
        str(local_parquet),
        args.lens_csv,
        ("A", "B", "C"),
        args.threshold,
        args.branching_factor,
        DEFAULT_BIRCH_BATCH_SIZE,
        DEFAULT_BIRCH_FEATURES,
        DEFAULT_BIRCH_SCALING,
    )
    n_clusters = int(clustered["cluster"].nunique())
    if args.expected_objects is not None and len(clustered) != args.expected_objects:
        raise ValueError(f"Expected {args.expected_objects} objects, found {len(clustered)}.")
    if args.expected_clusters is not None and n_clusters != args.expected_clusters:
        raise ValueError(f"Expected {args.expected_clusters} clusters, found {n_clusters}.")
    if features != list(DEFAULT_BIRCH_FEATURES):
        raise ValueError("The PCA columns differ from the configured 40-feature baseline.")

    signature = {
        "path": args.source_uri,
        "generation": args.generation,
        "size": args.size,
    }
    save_cached_projection(
        args.output,
        signature,
        clustered["id_str"],
        clustered["cluster"],
        projection,
        args.threshold,
        args.branching_factor,
    )
    loaded = load_cached_projection(
        args.output, signature, args.threshold, args.branching_factor
    )
    if loaded is None or not np.array_equal(loaded[0], clustered["cluster"].to_numpy()):
        raise RuntimeError("The saved BIRCH artifact failed its round-trip check.")

    print(json.dumps({
        "objects": len(clustered),
        "threshold": args.threshold,
        "branching_factor": args.branching_factor,
        "clusters": n_clusters,
        "labelled_candidates": int(clustered["is_lens"].sum()),
        "artifact": args.output,
        "upload_to": cache_path(
            args.source_uri, args.threshold, args.branching_factor
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
