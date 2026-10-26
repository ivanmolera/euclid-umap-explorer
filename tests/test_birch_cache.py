import importlib.util
import io
import json
import os
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

if "streamlit" not in sys.modules and importlib.util.find_spec("streamlit") is None:
    streamlit = types.ModuleType("streamlit")
    streamlit.cache_resource = lambda **kwargs: lambda function: function
    sys.modules["streamlit"] = streamlit

from euclid_umap_explorer import birch, birch_cache, storage
from euclid_umap_explorer.birch import _run_birch_clustering_with_cache_impl
from euclid_umap_explorer.birch_cache import cache_path, source_signature
from euclid_umap_explorer.config import DEFAULT_BIRCH_FEATURES


class BirchCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        cache_dir = str(Path(self.temp_dir.name) / "cache")
        env_patch = patch.dict(os.environ, {
            "EUCLID_BIRCH_CACHE_BASE": cache_dir,
            "EUCLID_BIRCH_PREPARED_CACHE_DIR": str(Path(cache_dir) / "prepared"),
        })
        env_patch.start()
        self.addCleanup(env_patch.stop)
        original_cache_setting = storage.USE_LOCAL_CACHE
        storage.USE_LOCAL_CACHE = False
        self.addCleanup(setattr, storage, "USE_LOCAL_CACHE", original_cache_setting)

        self.parquet_path = str(Path(self.temp_dir.name) / "pca.parquet")
        self.lens_path = str(Path(self.temp_dir.name) / "lenses.csv")
        values = np.r_[np.zeros(20), np.full(20, 20.0)]
        catalogue = pd.DataFrame(
            {"id_str": [f"Q1_R1_1_{index}" for index in range(40)]}
        )
        for feature in DEFAULT_BIRCH_FEATURES:
            catalogue[feature] = values
        catalogue.to_parquet(self.parquet_path, index=False)
        pd.DataFrame(
            {"object_id": ["0", "20"], "grade": ["A", "B"]}
        ).to_csv(self.lens_path, index=False)

    def run_cached(
        self, threshold=6.2, grades=("A", "B", "C"), branching_factor=50,
        batch_size=25_000, features=DEFAULT_BIRCH_FEATURES, scaling="standard",
    ):
        return _run_birch_clustering_with_cache_impl(
            self.parquet_path,
            self.lens_path,
            grades,
            threshold,
            branching_factor,
            batch_size,
            features,
            scaling,
        )

    def run_default(self, grades=("A", "B", "C")):
        return self.run_cached(6.2, grades)

    def test_all_supported_configurations_have_distinct_reusable_caches(self):
        configurations = (
            (5.0, 50), (5.5, 50), (6.2, 50), (5.5, 25),
            (5.5, 42), (5.8, 55), (5.51, 42), (5.52, 42),
        )
        paths = {
            configuration: cache_path(self.parquet_path, *configuration)
            for configuration in configurations
        }
        self.assertEqual(len(set(paths.values())), len(configurations))
        for (threshold, branching_factor), path in paths.items():
            first, _, _ = self.run_cached(
                threshold, branching_factor=branching_factor
            )
            self.assertTrue(Path(path).is_file())
            with patch(
                "euclid_umap_explorer.birch._run_birch_clustering_impl",
                side_effect=AssertionError("BIRCH must not run again"),
            ):
                second, _, _ = self.run_cached(
                    threshold, branching_factor=branching_factor
                )
            self.assertEqual(first["cluster"].tolist(), second["cluster"].tolist())

    def test_cache_rejects_another_branching_factors_artifact(self):
        self.run_cached(5.5)
        Path(cache_path(self.parquet_path, 5.5, 25)).write_bytes(
            Path(cache_path(self.parquet_path, 5.5, 50)).read_bytes()
        )
        with patch(
            "euclid_umap_explorer.birch._run_birch_clustering_impl",
            wraps=birch._run_birch_clustering_impl,
        ) as compute:
            self.run_cached(5.5, branching_factor=25)
        self.assertEqual(compute.call_count, 1)

    def test_cache_rejects_another_thresholds_artifact(self):
        self.run_cached(5.0)
        Path(cache_path(self.parquet_path, 5.5)).write_bytes(
            Path(cache_path(self.parquet_path, 5.0)).read_bytes()
        )
        with patch(
            "euclid_umap_explorer.birch._run_birch_clustering_impl",
            wraps=birch._run_birch_clustering_impl,
        ) as compute:
            self.run_cached(5.5)
        self.assertEqual(compute.call_count, 1)

    def test_cache_reuses_clusters_but_refreshes_selected_grades(self):
        first, features, projection = self.run_default(("A",))
        self.assertTrue(Path(cache_path(self.parquet_path)).is_file())
        self.assertEqual(int(first["is_lens"].sum()), 1)

        with patch(
            "euclid_umap_explorer.birch._run_birch_clustering_impl",
            side_effect=AssertionError("BIRCH must not run again"),
        ):
            second, cached_features, cached_projection = self.run_default(("B",))

        self.assertEqual(features, cached_features)
        self.assertEqual(first["cluster"].tolist(), second["cluster"].tolist())
        self.assertEqual(int(second["is_lens"].sum()), 1)
        self.assertEqual(second.loc[20, "lens_grade"], "B")
        self.assertTrue(pd.isna(second.loc[0, "lens_grade"]))
        np.testing.assert_array_equal(projection.centers, cached_projection.centers)
        self.assertEqual(
            projection.distance_cutoffs, cached_projection.distance_cutoffs
        )

    def test_changed_catalogue_recomputes_default(self):
        self.run_default()
        catalogue = pd.read_parquet(self.parquet_path)
        catalogue.loc[0, "feat_pca_0"] = 100.0
        catalogue.to_parquet(self.parquet_path, index=False)

        with patch(
            "euclid_umap_explorer.birch._run_birch_clustering_impl",
            wraps=birch._run_birch_clustering_impl,
        ) as compute:
            self.run_default()
        self.assertEqual(compute.call_count, 1)

    def test_full_configuration_has_separate_reusable_caches(self):
        parameters = (
            (25_000, DEFAULT_BIRCH_FEATURES, "standard"),
            (13, DEFAULT_BIRCH_FEATURES, "standard"),
            (13, DEFAULT_BIRCH_FEATURES, "none"),
            (13, DEFAULT_BIRCH_FEATURES[:2], "standard"),
            (13, DEFAULT_BIRCH_FEATURES[1::-1], "standard"),
        )
        paths = set()
        for batch_size, features, scaling in parameters:
            with self.subTest(batch_size=batch_size, features=features, scaling=scaling):
                paths.add(cache_path(
                    self.parquet_path, 5.7, 49, batch_size, features, scaling
                ))
                first, _, projection = self.run_cached(
                    5.7, branching_factor=49, batch_size=batch_size,
                    features=features, scaling=scaling,
                )
                with patch.object(
                    birch, "_run_birch_clustering_impl",
                    side_effect=AssertionError("BIRCH must not run again"),
                ):
                    second, _, cached_projection = self.run_cached(
                        5.7, branching_factor=49, batch_size=batch_size,
                        features=features, scaling=scaling,
                    )
                self.assertEqual(first["cluster"].tolist(), second["cluster"].tolist())
                self.assertEqual(cached_projection.feature_cols, features)
                np.testing.assert_array_equal(projection.centers, cached_projection.centers)
        self.assertEqual(len(paths), len(parameters))

    def test_other_parameters_do_not_reuse_defaults(self):
        self.run_default()
        with patch(
            "euclid_umap_explorer.birch._run_birch_clustering_impl",
            side_effect=RuntimeError("custom calculation reached"),
        ):
            for threshold, branching_factor in ((6.3, 50), (5.0, 25), (5.5, 30)):
                with self.subTest(threshold=threshold, branching_factor=branching_factor):
                    with self.assertRaisesRegex(RuntimeError, "custom calculation reached"):
                        self.run_cached(threshold, branching_factor=branching_factor)

    def test_prepared_matrix_is_reused_for_another_configuration(self):
        self.run_default()
        with patch(
            "sklearn.preprocessing.StandardScaler.partial_fit",
            side_effect=AssertionError("The PCA matrix must not be scaled again"),
        ):
            self.run_cached(5.7, branching_factor=49)

    def test_cache_failures_do_not_block_computation(self):
        with patch.object(birch_cache, "_read_bytes", side_effect=OSError("unavailable")), patch.object(
            birch_cache, "_write_bytes", side_effect=OSError("read-only")
        ):
            result, _, _ = self.run_cached(5.7, branching_factor=49)
        self.assertEqual(len(result), 40)

    def test_old_precomputed_artifacts_remain_readable(self):
        first, _, _ = self.run_default()
        current_path = Path(cache_path(self.parquet_path))
        with np.load(current_path, allow_pickle=False) as artifact:
            arrays = {name: artifact[name] for name in artifact.files}
        manifest = json.loads(arrays["manifest"].tobytes().decode())
        manifest["schema"] = 1
        for key in ("numpy_version", "preparation_schema"):
            manifest.pop(key)
        arrays["manifest"] = np.frombuffer(json.dumps(manifest).encode(), dtype=np.uint8)
        legacy = birch_cache.legacy_cache_path(
            self.parquet_path, 6.2, 50, 25_000, DEFAULT_BIRCH_FEATURES, "standard"
        )
        np.savez_compressed(legacy, **arrays)
        current_path.unlink()
        with patch.object(
            birch, "_run_birch_clustering_impl",
            side_effect=AssertionError("The old cache must be reused"),
        ):
            second, _, _ = self.run_default()
        self.assertEqual(first["cluster"].tolist(), second["cluster"].tolist())
        self.assertTrue(Path(legacy).is_file())

    def test_library_version_change_recomputes(self):
        self.run_default()
        with patch.object(birch_cache.sklearn, "__version__", "future-test-version"), patch.object(
            birch, "_run_birch_clustering_impl", wraps=birch._run_birch_clustering_impl
        ) as compute:
            self.run_default()
        self.assertEqual(compute.call_count, 1)

    def test_precompute_accepts_a_new_parameter_pair(self):
        from euclid_umap_explorer.precompute_birch import main

        output = str(Path(self.temp_dir.name) / "precomputed.npz")
        arguments = [
            "precompute_birch", "--local-parquet", self.parquet_path,
            "--source-uri", "gs://test/pca.parquet", "--generation", "1",
            "--size", str(Path(self.parquet_path).stat().st_size),
            "--lens-csv", self.lens_path, "--output", output,
            "--threshold", "5.51", "--branching-factor", "42",
            "--expected-objects", "40", "--expected-clusters", "2",
        ]
        printed = io.StringIO()
        with patch.object(sys, "argv", arguments), redirect_stdout(printed):
            main()
        result = json.loads(printed.getvalue())
        self.assertEqual((result["threshold"], result["branching_factor"]), (5.51, 42))
        self.assertTrue(Path(output).is_file())

    def test_gcs_result_reuse_and_generation_invalidation(self):
        class FakeGCS:
            generation = "1"
            objects = {}

            def info(self, path):
                return {"generation": self.generation, "size": 100}

            def exists(self, path):
                return path in self.objects

            def open(self, path, mode):
                if mode == "rb":
                    return io.BytesIO(self.objects[path])
                filesystem = self

                class Upload(io.BytesIO):
                    def close(self):
                        filesystem.objects[path] = self.getvalue()
                        super().close()

                return Upload()

        filesystem = FakeGCS()
        catalogue, features = birch.load_pca_catalog(self.parquet_path)
        parameters = (
            "gs://test/pca.parquet", self.lens_path, ("A", "B", "C"),
            5.7, 49, 25_000, DEFAULT_BIRCH_FEATURES, "standard",
        )
        with patch.dict(os.environ, {"EUCLID_BIRCH_CACHE_BASE": "gs://test/cache"}), patch.object(
            birch_cache, "_gcs_filesystem", return_value=filesystem
        ), patch.object(birch, "load_pca_catalog", return_value=(catalogue, features)):
            first, _, _ = _run_birch_clustering_with_cache_impl(*parameters)
            self.assertEqual(len(filesystem.objects), 1)
            with patch.object(
                birch, "_run_birch_clustering_impl",
                side_effect=AssertionError("GCS result must be reused"),
            ):
                second, _, _ = _run_birch_clustering_with_cache_impl(*parameters)
            self.assertEqual(first["cluster"].tolist(), second["cluster"].tolist())
            filesystem.generation = "2"
            with patch.object(
                birch, "_run_birch_clustering_impl", wraps=birch._run_birch_clustering_impl
            ) as compute:
                _run_birch_clustering_with_cache_impl(*parameters)
            self.assertEqual(compute.call_count, 1)

    def test_corrupt_cache_is_recomputed(self):
        self.run_default()
        Path(cache_path(self.parquet_path)).write_bytes(b"not a valid cache")
        with patch(
            "euclid_umap_explorer.birch._run_birch_clustering_impl",
            wraps=birch._run_birch_clustering_impl,
        ) as compute:
            self.run_default()
        self.assertEqual(compute.call_count, 1)

    def test_row_order_mismatch_is_recomputed(self):
        self.run_default()
        original_signature = source_signature(self.parquet_path)
        catalogue = pd.read_parquet(self.parquet_path)
        catalogue.iloc[::-1].to_parquet(self.parquet_path, index=False)
        with patch(
            "euclid_umap_explorer.birch_cache.source_signature",
            return_value=original_signature,
        ), patch(
            "euclid_umap_explorer.birch._run_birch_clustering_impl",
            wraps=birch._run_birch_clustering_impl,
        ) as compute:
            self.run_default()
        self.assertEqual(compute.call_count, 1)


if __name__ == "__main__":
    unittest.main()
