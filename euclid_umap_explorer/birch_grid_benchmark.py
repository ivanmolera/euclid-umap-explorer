"""Full-catalogue BIRCH grid benchmark for a resumable Cloud Run Job."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import io
import json
import os
import tempfile
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import Birch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .birch_cache import source_signature
from .config import DEFAULT_BIRCH_BATCH_SIZE, DEFAULT_BIRCH_FEATURES, STRAIGHT_LINE_FILTERED_PARQUET_PATH


DEFAULT_LENS_CSV = (
    "gs://euclid-umap-ivan-0424-data/catalogues/strong_lensing_catalogue/"
    "q1_discovery_engine_lens_catalog.csv"
)
THRESHOLDS = tuple(round(value / 10, 1) for value in range(50, 71))
BRANCHING_FACTORS = tuple(range(25, 56))
CONFIGURATIONS = tuple((threshold, branching) for threshold in THRESHOLDS for branching in BRANCHING_FACTORS)
REVIEW_BUDGET = 5_000
SPLIT_SEED = 20261005


@lru_cache(maxsize=1)
def _gcs_filesystem():
    import gcsfs

    return gcsfs.GCSFileSystem(skip_instance_cache=True)


def _filesystem(path: str):
    return _gcs_filesystem() if path.startswith("gs://") else None


def _read(path: str) -> bytes | None:
    fs = _filesystem(path)
    if fs is not None:
        if not fs.exists(path):
            return None
        with fs.open(path, "rb") as stream:
            return stream.read()
    target = Path(path)
    return target.read_bytes() if target.is_file() else None


def _exists(path: str) -> bool:
    fs = _filesystem(path)
    return fs.exists(path) if fs is not None else Path(path).is_file()


def _write(path: str, content: bytes) -> None:
    fs = _filesystem(path)
    if fs is not None:
        with fs.open(path, "wb") as stream:
            stream.write(content)
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
    temporary.replace(target)


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _result_path(output: str, threshold: float, branching: int) -> str:
    return f"{output}/results/t{threshold:.1f}_b{branching}.json"


def _fit_path(output: str, threshold: float, branching: int) -> str:
    return f"{output}/fits/t{threshold:.1f}_b{branching}.npz"


def _cluster_path(output: str, threshold: float, branching: int) -> str:
    return f"{output}/clusters/t{threshold:.1f}_b{branching}.parquet"


def _parquet_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()


def protocol(parquet: str, lens_csv: str, reference_index: str | None = None,
             reference_fit: str | None = None) -> dict:
    if (reference_index is None) != (reference_fit is None):
        raise ValueError("Reference index and fit must be supplied together")
    specification = {
        "schema": 1,
        "parquet": source_signature(parquet),
        "lens_csv": source_signature(lens_csv),
        "features": list(DEFAULT_BIRCH_FEATURES),
        "scaling": "standard",
        "batch_size": DEFAULT_BIRCH_BATCH_SIZE,
        "input_order": "parquet row order",
        "split_seed": SPLIT_SEED,
        "split_fractions": {"train": 0.6, "validation": 0.2, "test": 0.2},
        "review_budget": REVIEW_BUDGET,
        "thresholds": list(THRESHOLDS),
        "branching_factors": list(BRANCHING_FACTORS),
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
    }
    if reference_index is not None:
        specification.update({
            "scaling": "filtered v3.2 reference StandardScaler",
            "reference_index": source_signature(reference_index),
            "reference_fit": source_signature(reference_fit),
            "split_mode": "reuse filtered v3.2 object-ID splits",
        })
    return specification


def protocol_key(value: dict) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def load_input(parquet: str, lens_csv: str, expected_objects: int, expected_labels: int,
               reference_index: str | None = None, reference_fit: str | None = None) -> dict:
    if (reference_index is None) != (reference_fit is None):
        raise ValueError("Reference index and fit must be supplied together")
    started = time.perf_counter()
    frame = pd.read_parquet(parquet, columns=["id_str", *DEFAULT_BIRCH_FEATURES])
    if expected_objects and len(frame) != expected_objects:
        raise ValueError(f"Expected {expected_objects} objects, found {len(frame)}")
    ids = frame["id_str"].astype("string").str.strip()
    object_ids = ids.str.rsplit("_", n=1).str[-1]
    if (ids.isna().any() or ids.eq("").any() or object_ids.isna().any()
            or object_ids.eq("").any() or object_ids.duplicated().any()):
        raise ValueError("PCA catalogue has missing or duplicate object IDs")
    x = frame[list(DEFAULT_BIRCH_FEATURES)].to_numpy(dtype=np.float32, copy=True)
    del frame
    if not np.isfinite(x).all():
        raise ValueError("PCA catalogue contains non-finite values")

    lens = pd.read_csv(lens_csv, dtype={"object_id": "string"})
    lens["object_id"] = lens["object_id"].astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    lens["grade"] = lens["grade"].astype("string").str.strip().str.upper()
    lens = lens[lens["grade"].isin(("A", "B", "C"))]
    if lens.groupby("object_id")["grade"].nunique().gt(1).any():
        raise ValueError("Lens catalogue contains conflicting grades")
    grades = object_ids.map(lens.drop_duplicates("object_id").set_index("object_id")["grade"])
    labelled = np.flatnonzero(grades.notna().to_numpy())
    if expected_labels and len(labelled) != expected_labels:
        raise ValueError(f"Expected {expected_labels} matched A/B/C candidates, found {len(labelled)}")

    grade = grades.fillna("Unknown").to_numpy(dtype=str)
    if reference_index is None:
        train_idx, rest = train_test_split(
            labelled, test_size=0.4, stratify=grade[labelled], random_state=SPLIT_SEED
        )
        val_idx, test_idx = train_test_split(
            rest, test_size=0.5, stratify=grade[rest], random_state=SPLIT_SEED
        )
        train = np.zeros(len(x), dtype=bool)
        val = np.zeros(len(x), dtype=bool)
        test = np.zeros(len(x), dtype=bool)
        train[train_idx], val[val_idx], test[test_idx] = True, True, True

        scaler = StandardScaler()
        for start in range(0, len(x), DEFAULT_BIRCH_BATCH_SIZE):
            scaler.partial_fit(x[start : start + DEFAULT_BIRCH_BATCH_SIZE])
        scaler_mean, scaler_scale = scaler.mean_, scaler.scale_
        z = np.ascontiguousarray(scaler.transform(x), dtype=np.float32)
    else:
        reference = pd.read_parquet(
            reference_index, columns=["id_str", "object_id", "grade", "label_split"]
        )
        if reference["id_str"].duplicated().any() or reference["object_id"].duplicated().any():
            raise ValueError("Reference index contains duplicate IDs")
        reference_ids = set(reference["id_str"].astype(str))
        if not np.array_equal(ids[ids.isin(reference_ids)].to_numpy(dtype=str),
                              reference["id_str"].to_numpy(dtype=str)):
            raise ValueError("Reference objects are not an ordered subset of the input")
        selected = reference.loc[reference["label_split"] != "unknown"].copy()
        if (len(selected) != len(labelled)
                or not selected["label_split"].isin(("train", "validation", "test")).all()):
            raise ValueError("Reference index does not cover exactly the labelled objects")
        reference_grades = selected.set_index("object_id")["grade"]
        if not np.array_equal(reference_grades.reindex(object_ids.iloc[labelled]).to_numpy(dtype=str),
                              grade[labelled]):
            raise ValueError("Reference grades do not match the full catalogue")
        splits = object_ids.map(selected.set_index("object_id")["label_split"])
        if int(splits.notna().sum()) != len(labelled):
            raise ValueError("Reference label splits do not match the full catalogue")
        train = splits.eq("train").fillna(False).to_numpy(dtype=bool)
        val = splits.eq("validation").fillna(False).to_numpy(dtype=bool)
        test = splits.eq("test").fillna(False).to_numpy(dtype=bool)

        fit_bytes = _read(reference_fit)
        if fit_bytes is None:
            raise FileNotFoundError(reference_fit)
        with np.load(io.BytesIO(fit_bytes)) as fit:
            scaler_mean = fit["scaler_mean"].copy()
            scaler_scale = fit["scaler_scale"].copy()
        if (scaler_mean.shape != (x.shape[1],) or scaler_scale.shape != (x.shape[1],)
                or not np.isfinite(scaler_mean).all() or not np.isfinite(scaler_scale).all()
                or (scaler_scale <= 0).any()):
            raise ValueError("Invalid reference scaler")
        z = x.copy()
        z -= scaler_mean
        z /= scaler_scale
        z = np.ascontiguousarray(z, dtype=np.float32)
    print(json.dumps({
        "event": "input_ready", "objects": len(x), "labels": len(labelled),
        "train": int(train.sum()), "validation": int(val.sum()), "test": int(test.sum()),
        "load_scale_seconds": round(time.perf_counter() - started, 2),
    }), flush=True)
    return {
        "z": z, "ids": ids.to_numpy(dtype=str), "object_ids": object_ids.to_numpy(dtype=str),
        "grade": grade, "train": train, "val": val, "test": test,
        "labelled": grade != "Unknown", "scaler_mean": scaler_mean, "scaler_scale": scaler_scale,
    }


def fit_labels(z: np.ndarray, threshold: float, branching: int) -> dict:
    model = Birch(
        threshold=threshold, branching_factor=branching,
        n_clusters=None, compute_labels=False,
    )
    started = time.perf_counter()
    for start in range(0, len(z), DEFAULT_BIRCH_BATCH_SIZE):
        model.partial_fit(z[start : start + DEFAULT_BIRCH_BATCH_SIZE])
    model.partial_fit()
    fit_seconds = time.perf_counter() - started
    labels = np.empty(len(z), dtype=np.int32)
    distances = np.empty(len(z), dtype=np.float32)
    centers = model.subcluster_centers_.astype(np.float32, copy=True)
    center_labels = model.subcluster_labels_.astype(np.int32, copy=True)
    if not np.array_equal(center_labels, np.arange(len(centers))):
        raise RuntimeError("BIRCH centers do not match their labels")
    started = time.perf_counter()
    for start in range(0, len(z), DEFAULT_BIRCH_BATCH_SIZE):
        end = min(start + DEFAULT_BIRCH_BATCH_SIZE, len(z))
        batch = z[start:end]
        labels[start:end] = model.predict(batch)
        distances[start:end] = np.linalg.norm(batch - centers[labels[start:end]], axis=1)
    predict_seconds = time.perf_counter() - started
    return {
        "labels": labels, "distances": distances, "centers": centers,
        "center_labels": center_labels, "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
    }


def cluster_table(fit: dict, data: dict) -> pd.DataFrame:
    labels = fit["labels"]
    n = len(fit["centers"])
    sizes = np.bincount(labels, minlength=n)
    table = pd.DataFrame({
        "cluster": np.arange(n, dtype=np.int32),
        "n_objects": sizes,
        "n_grade_a": np.bincount(labels, weights=(data["grade"] == "A"), minlength=n).astype(int),
        "n_grade_b": np.bincount(labels, weights=(data["grade"] == "B"), minlength=n).astype(int),
        "n_grade_c": np.bincount(labels, weights=(data["grade"] == "C"), minlength=n).astype(int),
        "n_train": np.bincount(labels, weights=data["train"], minlength=n).astype(int),
        "n_validation": np.bincount(labels, weights=data["val"], minlength=n).astype(int),
        "n_test": np.bincount(labels, weights=data["test"], minlength=n).astype(int),
    })
    table["n_references"] = table[["n_grade_a", "n_grade_b", "n_grade_c"]].sum(axis=1)
    table = table.loc[table["n_objects"] > 0].copy()
    table["lens_rate"] = table["n_references"] / table["n_objects"]
    table["enrichment_x"] = table["lens_rate"] / data["labelled"].mean()
    prevalence = data["train"].mean()
    table["train_lens_rate"] = table["n_train"] / table["n_objects"]
    table["train_enrichment_x"] = table["train_lens_rate"] / prevalence
    table["train_priority"] = (
        (table["n_train"] + 50 * prevalence) / (table["n_objects"] + 50) / prevalence
    ) * np.log1p(table["n_train"])
    distances = pd.Series(fit["distances"]).groupby(labels)
    table["median_center_distance"] = distances.median().reindex(table["cluster"]).to_numpy()
    table["p95_center_distance"] = distances.quantile(0.95).reindex(table["cluster"]).to_numpy()
    return table


def ranked_objects(labels: np.ndarray, data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sizes = np.bincount(labels)
    train_counts = np.bincount(labels, weights=data["train"].astype(int), minlength=len(sizes))
    prevalence = data["train"].mean()
    priority = ((train_counts + 50 * prevalence) / (sizes + 50) / prevalence) * np.log1p(train_counts)
    ranked = np.lexsort((data["ids"], -priority[labels]))
    return ranked[~data["train"][ranked]], sizes, train_counts


def score(labels: np.ndarray, data: dict, reserved: str = "val") -> dict:
    ranked, sizes, train_counts = ranked_objects(labels, data)
    all_counts = np.bincount(labels, weights=data["labelled"].astype(int), minlength=len(sizes))
    population_rate = data["labelled"].mean()
    supported = np.flatnonzero(all_counts >= 10)
    best = int(supported[np.argmax(all_counts[supported] / sizes[supported])]) if len(supported) else None
    budget = min(REVIEW_BUDGET, len(ranked))
    held_out = data[reserved]
    hits = int(held_out[ranked[:budget]].sum())
    expected_hits = float(held_out.sum() * budget / len(ranked))
    return {
        "n_clusters": int((sizes > 0).sum()),
        "n_subclusters": int(len(sizes)),
        "n_singleton_clusters": int((sizes == 1).sum()),
        "median_cluster_size": float(np.median(sizes[sizes > 0])),
        "largest_cluster_size": int(sizes.max()),
        "n_clusters_with_10_references": int(len(supported)),
        "n_clusters_with_10_training_references": int((train_counts >= 10).sum()),
        "best_supported_cluster": best,
        "best_supported_cluster_size": int(sizes[best]) if best is not None else None,
        "best_supported_cluster_references": int(all_counts[best]) if best is not None else None,
        "best_supported_cluster_enrichment_x": (
            float(all_counts[best] / sizes[best] / population_rate) if best is not None else None
        ),
        f"{reserved}_hits_at_5000": hits,
        f"{reserved}_recall_at_5000": float(hits / held_out.sum()),
        f"{reserved}_expected_hits_random": expected_hits,
        f"{reserved}_enrichment_vs_random": float(hits / expected_hits) if expected_hits else None,
    }


def run(args: argparse.Namespace) -> None:
    shard_index = int(os.getenv("CLOUD_RUN_TASK_INDEX", "0"))
    shard_count = int(os.getenv("CLOUD_RUN_TASK_COUNT", "1"))
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("Invalid Cloud Run task index/count")
    spec = protocol(args.parquet, args.lens_csv, args.reference_index, args.reference_fit)
    key = protocol_key(spec)
    assigned = CONFIGURATIONS[shard_index::shard_count]
    pending = []
    for threshold, branching in assigned:
        path = _result_path(args.output, threshold, branching)
        existing = _read(path)
        if existing is not None:
            if json.loads(existing)["protocol_key"] != key:
                raise ValueError(f"Existing result uses a different input/protocol: {path}")
            if _exists(_fit_path(args.output, threshold, branching)) and _exists(
                _cluster_path(args.output, threshold, branching)
            ):
                continue
        pending.append((threshold, branching, path))
    index_path = f"{args.output}/object_index.parquet"
    index_manifest_path = f"{args.output}/object_index.json"
    existing_index = _read(index_manifest_path) if shard_index == 0 else None
    if existing_index is not None and json.loads(existing_index)["protocol_key"] != key:
        raise ValueError("The existing object index uses a different input/protocol")
    needs_index = shard_index == 0 and (existing_index is None or not _exists(index_path))
    print(json.dumps({"event": "shard_start", "shard": shard_index, "shards": shard_count,
                      "assigned": len(assigned), "pending": len(pending), "protocol_key": key}), flush=True)
    if not pending and not needs_index:
        return
    data = load_input(args.parquet, args.lens_csv, args.expected_objects, args.expected_labels,
                      args.reference_index, args.reference_fit)
    if needs_index:
        split = np.full(len(data["ids"]), "unknown", dtype="<U10")
        split[data["train"]] = "train"
        split[data["val"]] = "validation"
        split[data["test"]] = "test"
        index = pd.DataFrame({
            "row_index": np.arange(len(split), dtype=np.int32),
            "id_str": data["ids"], "object_id": data["object_ids"],
            "grade": data["grade"], "label_split": split,
        })
        _write(index_path, _parquet_bytes(index))
        _write(index_manifest_path, _json_bytes({
            "protocol_key": key, "rows": len(index),
            "ordered_id_sha256": hashlib.sha256("\n".join(data["ids"]).encode()).hexdigest(),
        }))
    for threshold, branching, path in pending:
        fit = fit_labels(data["z"], threshold, branching)
        metrics = score(fit["labels"], data)
        clusters = cluster_table(fit, data)
        if len(clusters) != metrics["n_clusters"]:
            raise RuntimeError("Cluster summary does not match fitted assignments")
        fit_path = _fit_path(args.output, threshold, branching)
        cluster_path = _cluster_path(args.output, threshold, branching)
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            manifest=np.frombuffer(_json_bytes({
                "protocol_key": key, "threshold": threshold, "branching_factor": branching,
                "rows": len(data["ids"]),
            }), dtype=np.uint8),
            labels=fit["labels"], distances=fit["distances"],
            centers=fit["centers"], center_labels=fit["center_labels"],
            scaler_mean=data["scaler_mean"], scaler_scale=data["scaler_scale"],
        )
        _write(fit_path, buffer.getvalue())
        _write(cluster_path, _parquet_bytes(clusters))
        result = {
            "protocol_key": key, "threshold": threshold, "branching_factor": branching,
            "fit_seconds": fit["fit_seconds"], "predict_seconds": fit["predict_seconds"],
            "fit_path": fit_path, "cluster_path": cluster_path, **metrics,
        }
        _write(path, _json_bytes(result))
        print(json.dumps({"event": "configuration_complete", "threshold": threshold,
                          "branching_factor": branching, "n_clusters": metrics["n_clusters"],
                          "validation_hits_at_5000": metrics["val_hits_at_5000"],
                          "seconds": round(fit["fit_seconds"] + fit["predict_seconds"], 2)}), flush=True)
        del fit, clusters, buffer
        gc.collect()


def collect(args: argparse.Namespace) -> tuple[dict, list[dict]]:
    spec = protocol(args.parquet, args.lens_csv, args.reference_index, args.reference_fit)
    key = protocol_key(spec)
    rows = []
    missing = []
    for threshold, branching in CONFIGURATIONS:
        path = _result_path(args.output, threshold, branching)
        content = _read(path)
        if content is None:
            missing.append(path)
            continue
        result = json.loads(content)
        if result["protocol_key"] != key:
            raise ValueError(f"Mismatched result protocol: {path}")
        if not _exists(result["fit_path"]) or not _exists(result["cluster_path"]):
            raise RuntimeError(f"Missing full clustering artifacts for {path}")
        rows.append(result)
    if missing:
        raise RuntimeError(f"Only {len(rows)}/{len(CONFIGURATIONS)} results are present; first missing: {missing[0]}")
    index_manifest = _read(f"{args.output}/object_index.json")
    if index_manifest is None or not _exists(f"{args.output}/object_index.parquet"):
        raise RuntimeError("Shared object index is missing")
    if json.loads(index_manifest)["protocol_key"] != key:
        raise ValueError("Shared object index uses a different input/protocol")
    return spec, rows


def summarize(args: argparse.Namespace) -> dict:
    spec, rows = collect(args)
    eligible = [row for row in rows if 20 <= row["n_clusters"] <= 2500]
    if not eligible:
        raise RuntimeError("No configuration has 20-2500 clusters")
    ordered = sorted(eligible, key=lambda row: (
        -row["val_hits_at_5000"], row["n_clusters"],
        row["fit_seconds"] + row["predict_seconds"],
        row["threshold"], row["branching_factor"],
    ))
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(sorted(rows, key=lambda row: (row["threshold"], row["branching_factor"])))
    _write(f"{args.output}/grid_results.csv", out.getvalue().encode("utf-8"))
    summary = {
        "protocol": spec,
        "protocol_key": protocol_key(spec),
        "n_configurations": len(rows),
        "selection_rule": "Maximum validation hits at 5000; ties: fewer clusters, shorter runtime, parameter order. Eligible: 20-2500 clusters.",
        "selected": ordered[0],
        "caveat": "One parquet order only. Validation labels select the configuration; reference labels are not confirmed negatives/true purity. The v3.2 input protected known candidates.",
    }
    _write(f"{args.output}/summary.json", _json_bytes(summary))
    print(json.dumps({"event": "summary_complete", "configurations": len(rows),
                      "selected": ordered[0]}), flush=True)
    return summary


def count_enriched(args: argparse.Namespace) -> None:
    _, rows = collect(args)
    reference_col = "n_train" if args.label_set == "train" else "n_references"
    enrichment_col = "train_enrichment_x" if args.label_set == "train" else "enrichment_x"
    counts = []
    for row in rows:
        content = _read(row["cluster_path"])
        clusters = pd.read_parquet(io.BytesIO(content))
        selected = clusters.loc[
            (clusters[enrichment_col] >= args.min_enrichment)
            & (clusters[reference_col] >= args.min_references)
            & (clusters["n_objects"] >= args.min_objects)
        ]
        counts.append({
            "threshold": row["threshold"], "branching_factor": row["branching_factor"],
            "n_clusters": row["n_clusters"], "label_set": args.label_set,
            "min_enrichment_x": args.min_enrichment,
            "min_references": args.min_references, "min_objects": args.min_objects,
            "qualifying_clusters": len(selected),
            "objects_in_qualifying_clusters": int(selected["n_objects"].sum()),
            "references_in_qualifying_clusters": int(selected[reference_col].sum()),
        })
    result = pd.DataFrame(counts).sort_values(
        ["qualifying_clusters", "references_in_qualifying_clusters", "threshold", "branching_factor"],
        ascending=[False, False, True, True],
    )
    name = f"enrichment_counts_{args.label_set}_ge_{args.min_enrichment:g}_refs_{args.min_references}_objects_{args.min_objects}.csv"
    _write(f"{args.output}/{name}", result.to_csv(index=False).encode("utf-8"))
    print(json.dumps({"event": "enrichment_counts_complete", "file": f"{args.output}/{name}",
                      "top_10": result.head(10).to_dict(orient="records")}), flush=True)


def finalize(args: argparse.Namespace) -> None:
    summary = summarize(args)
    winner = summary["selected"]
    data = load_input(args.parquet, args.lens_csv, args.expected_objects, args.expected_labels,
                      args.reference_index, args.reference_fit)
    fit = fit_labels(
        data["z"], winner["threshold"], winner["branching_factor"]
    )
    metrics = score(fit["labels"], data, reserved="test")
    result = {
        "protocol_key": summary["protocol_key"],
        "selected_threshold": winner["threshold"],
        "selected_branching_factor": winner["branching_factor"],
        "fit_seconds": fit["fit_seconds"],
        "predict_seconds": fit["predict_seconds"],
        **metrics,
        "caveat": "Internal held-out labels only; not independent discovery validation or lens purity.",
    }
    _write(f"{args.output}/final_test.json", _json_bytes(result))
    print(json.dumps({"event": "final_test_complete", **result}), flush=True)


def main(*, default_parquet: str = STRAIGHT_LINE_FILTERED_PARQUET_PATH,
         default_objects: int = 374_982, default_reference_index: str | None = None,
         default_reference_fit: str | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "summarize", "counts", "finalize"))
    parser.add_argument("--parquet", default=default_parquet)
    parser.add_argument("--lens-csv", default=DEFAULT_LENS_CSV)
    parser.add_argument("--reference-index", default=default_reference_index)
    parser.add_argument("--reference-fit", default=default_reference_fit)
    parser.add_argument("--output", required=True, help="Dedicated local or gs:// prefix for this run")
    parser.add_argument("--expected-objects", type=int, default=default_objects)
    parser.add_argument("--expected-labels", type=int, default=1_416)
    parser.add_argument("--min-enrichment", type=float, default=10.0)
    parser.add_argument("--min-references", type=int, default=10)
    parser.add_argument("--min-objects", type=int, default=0)
    parser.add_argument("--label-set", choices=("all", "train"), default="all")
    args = parser.parse_args()
    args.output = args.output.rstrip("/")
    if args.min_enrichment < 0 or args.min_references < 0 or args.min_objects < 0:
        parser.error("Enrichment and support minima must be non-negative")
    if args.mode == "run":
        run(args)
    elif args.mode == "summarize":
        summarize(args)
    elif args.mode == "counts":
        count_enriched(args)
    else:
        finalize(args)


if __name__ == "__main__":
    main()
