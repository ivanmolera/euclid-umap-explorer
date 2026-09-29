import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

if importlib.util.find_spec("streamlit") is None:
    streamlit = types.ModuleType("streamlit")
    streamlit.cache_resource = lambda **kwargs: lambda function: function
    sys.modules["streamlit"] = streamlit

from euclid_umap_explorer import storage
from euclid_umap_explorer.birch import (
    _assign_excluded_artifacts_impl,
    _run_birch_clustering_impl,
)


class BirchArtifactAssignmentTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.original_cache_setting = storage.USE_LOCAL_CACHE
        storage.USE_LOCAL_CACHE = False
        self.addCleanup(setattr, storage, "USE_LOCAL_CACHE", self.original_cache_setting)

        offsets = np.linspace(-0.2, 0.2, 40)
        clean = pd.DataFrame(
            {
                "id_str": [f"Q1_R1_1_{index}" for index in range(80)],
                "feat_pca_0": np.r_[offsets, 10 + offsets],
                "feat_pca_1": np.r_[offsets * 0.5, 10 + offsets * 0.5],
            }
        )
        excluded = pd.DataFrame(
            {
                "id_str": ["Q1_R1_1_80", "Q1_R1_1_81", "Q1_R1_1_82"],
                "feat_pca_0": [0.02, 10.02, 40.0],
                "feat_pca_1": [0.01, 10.01, 40.0],
            }
        )
        self.clean_path = str(Path(self.temp_dir.name) / "clean.parquet")
        self.full_path = str(Path(self.temp_dir.name) / "full.parquet")
        self.lens_path = str(Path(self.temp_dir.name) / "lenses.csv")
        clean.to_parquet(self.clean_path, index=False)
        pd.concat([clean, excluded], ignore_index=True).to_parquet(
            self.full_path, index=False
        )
        pd.DataFrame(columns=["object_id", "grade"]).to_csv(
            self.lens_path, index=False
        )

    def test_excluded_objects_are_assigned_without_changing_clean_clusters(self):
        for scaling in ("none", "standard"):
            with self.subTest(scaling=scaling):
                clean, _, projection = _run_birch_clustering_impl(
                    self.clean_path,
                    self.lens_path,
                    ("A", "B", "C"),
                    threshold=0.4,
                    branching_factor=10,
                    batch_size=13,
                    selected_features=("feat_pca_0", "feat_pca_1"),
                    scaling=scaling,
                )
                assignments = _assign_excluded_artifacts_impl(
                    self.full_path,
                    set(clean["id_str"]),
                    projection,
                    batch_size=2,
                ).set_index("object_id")

                self.assertEqual(len(clean), 80)
                self.assertEqual(set(assignments.index), {"80", "81", "82"})
                self.assertEqual(
                    int(assignments.loc["80", "cluster"]), int(clean.iloc[0]["cluster"])
                )
                self.assertEqual(
                    int(assignments.loc["81", "cluster"]), int(clean.iloc[40]["cluster"])
                )
                self.assertTrue(bool(assignments.loc["82", "is_far"]))
                self.assertFalse(bool(assignments.loc["80", "is_far"]))

    def test_filtered_catalogue_must_be_subset_of_full_catalogue(self):
        clean, _, projection = _run_birch_clustering_impl(
            self.clean_path,
            self.lens_path,
            (),
            threshold=0.4,
            branching_factor=10,
            batch_size=13,
            selected_features=("feat_pca_0", "feat_pca_1"),
            scaling="none",
        )
        with self.assertRaisesRegex(ValueError, "not a subset"):
            _assign_excluded_artifacts_impl(
                self.full_path,
                set(clean["id_str"]) | {"missing_id"},
                projection,
                batch_size=2,
            )


if __name__ == "__main__":
    unittest.main()
