import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.cluster import Birch
from sklearn.preprocessing import StandardScaler

if "streamlit" not in sys.modules and importlib.util.find_spec("streamlit") is None:
    streamlit = types.ModuleType("streamlit")
    streamlit.cache_resource = lambda **kwargs: lambda function: function
    sys.modules["streamlit"] = streamlit

from euclid_umap_explorer import birch, birch_cache, storage
from euclid_umap_explorer.config import DEFAULT_BIRCH_FEATURES


def legacy_projection(frame, features, scaling, batch_size, threshold, branching):
    """Reference for the previous per-batch DataFrame/scaler implementation."""
    def batches():
        for start in range(0, len(frame), batch_size):
            yield frame.iloc[start : start + batch_size][list(features)].to_numpy(
                dtype=np.float32, copy=True
            )

    scaler = StandardScaler() if scaling == "standard" else None
    if scaler is not None:
        for batch in batches():
            scaler.partial_fit(batch)
    model = Birch(
        threshold=threshold, branching_factor=branching,
        n_clusters=None, compute_labels=False,
    )
    for batch in batches():
        if scaler is not None:
            batch = scaler.transform(batch).astype(np.float32, copy=False)
        model.partial_fit(batch)
    model.partial_fit()
    labels, distances = [], []
    centers = model.subcluster_centers_.astype(np.float32, copy=True)
    for batch in batches():
        if scaler is not None:
            batch = scaler.transform(batch).astype(np.float32, copy=False)
        assigned = model.predict(batch)
        labels.append(assigned)
        distances.append(np.linalg.norm(batch - centers[assigned], axis=1))
    return model, scaler, np.concatenate(labels), np.concatenate(distances)


class BirchFeaturePreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        env = patch.dict(os.environ, {
            "EUCLID_BIRCH_PREPARED_CACHE_DIR": self.temp_dir.name,
        })
        env.start()
        self.addCleanup(env.stop)
        cache_setting = storage.USE_LOCAL_CACHE
        storage.USE_LOCAL_CACHE = False
        self.addCleanup(setattr, storage, "USE_LOCAL_CACHE", cache_setting)
        rng = np.random.default_rng(42)
        values = rng.normal(size=(173, 40)) * np.geomspace(0.2, 20, 40)
        values[55:110] += np.linspace(0, 30, 40)
        self.frame = pd.DataFrame(values, columns=DEFAULT_BIRCH_FEATURES)
        self.frame.insert(0, "id_str", [f"Q1_R1_1_{i}" for i in range(len(values))])
        self.path = str(Path(self.temp_dir.name) / "pca.parquet")
        self.lenses = str(Path(self.temp_dir.name) / "lenses.csv")
        self.frame.to_parquet(self.path, index=False)
        pd.DataFrame({"object_id": ["2", "75"], "grade": ["A", "B"]}).to_csv(
            self.lenses, index=False
        )

    def test_results_match_the_previous_implementation_exactly(self):
        for features, scaling, batch_size, threshold, branching in (
            (DEFAULT_BIRCH_FEATURES, "standard", 13, 5.5, 42),
            (DEFAULT_BIRCH_FEATURES, "standard", 25_000, 6.2, 50),
            (DEFAULT_BIRCH_FEATURES[:2], "none", 19, 0.5, 10),
            (DEFAULT_BIRCH_FEATURES[1::-1], "standard", 19, 0.5, 10),
        ):
            with self.subTest(scaling=scaling, batch_size=batch_size, features=features):
                frame, _ = birch.load_pca_catalog(self.path)
                model, scaler, labels, distances = legacy_projection(
                    frame, features, scaling, batch_size, threshold, branching
                )
                result, _, projection = birch._run_birch_clustering_impl(
                    self.path, self.lenses, ("A", "B", "C"), threshold,
                    branching, batch_size, features, scaling,
                )
                np.testing.assert_array_equal(result["cluster"], labels)
                np.testing.assert_array_equal(projection.centers, model.subcluster_centers_)
                if scaler is not None:
                    np.testing.assert_array_equal(projection.scaler_mean, scaler.mean_)
                    np.testing.assert_array_equal(projection.scaler_scale, scaler.scale_)
                else:
                    self.assertIsNone(projection.scaler_mean)
                    self.assertIsNone(projection.scaler_scale)
                global_cutoff = float(np.percentile(distances, 95))
                self.assertEqual(projection.global_distance_cutoff, global_cutoff)
                for label, cutoff in projection.distance_cutoffs.items():
                    cluster_distances = distances[labels == label]
                    expected = (
                        float(pd.Series(cluster_distances).quantile(0.95))
                        if len(cluster_distances) >= 20 else global_cutoff
                    )
                    self.assertEqual(cutoff, expected)

    def test_each_batch_is_transformed_once_and_then_reused_from_disk(self):
        original = StandardScaler.transform
        calls = []

        def counted(scaler, values, **kwargs):
            calls.append(len(values))
            return original(scaler, values, **kwargs)

        with patch.object(StandardScaler, "transform", counted):
            result, _, _ = birch._run_birch_clustering_impl(
                self.path, self.lenses, (), 5.5, 42, 13,
                DEFAULT_BIRCH_FEATURES, "standard",
            )
        self.assertEqual(len(calls), (len(result) + 12) // 13)
        frame, _ = birch.load_pca_catalog(self.path)
        with patch.object(StandardScaler, "partial_fit", side_effect=AssertionError("refit")):
            first = birch._prepare_birch_features(
                frame, self.path, DEFAULT_BIRCH_FEATURES, "standard", 13
            )
            second = birch._prepare_birch_features(
                frame, self.path, DEFAULT_BIRCH_FEATURES, "standard", 13
            )
        for expected, actual in zip(first, second):
            np.testing.assert_array_equal(actual, expected)

    def test_prepared_cache_rejects_row_order_and_corruption(self):
        frame, _ = birch.load_pca_catalog(self.path)
        source = birch_cache.source_signature(self.path)
        prepared = birch._prepare_birch_features(
            frame, self.path, DEFAULT_BIRCH_FEATURES, "standard", 13
        )
        cache = birch_cache.prepared_cache_path(source, 13, DEFAULT_BIRCH_FEATURES, "standard")
        self.assertIsNone(birch_cache.load_prepared_features(
            cache, source, frame["id_str"].iloc[::-1], 13, DEFAULT_BIRCH_FEATURES, "standard"
        ))
        Path(cache).write_bytes(b"corrupt matrix")
        recomputed = birch._prepare_birch_features(
            frame, self.path, DEFAULT_BIRCH_FEATURES, "standard", 13
        )
        for expected, actual in zip(prepared, recomputed):
            np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
