import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from euclid_umap_explorer import storage
from euclid_umap_explorer.analysis import recommended_numeric_filter
from euclid_umap_explorer.catalogs import (
    load_physical_measurement_object,
    load_physical_measurements,
)
from euclid_umap_explorer.physical import (
    analysis_ready_physical_measurements,
    apply_physical_filters,
    build_grouped_physical_summary,
    build_physical_summary,
    merge_physical_measurements,
    normalize_physical_filters,
    physical_filter_signature,
    physical_measurement_display_rows,
)


class PhysicalMeasurementsTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.original_cache_setting = storage.USE_LOCAL_CACHE
        storage.USE_LOCAL_CACHE = False
        self.addCleanup(setattr, storage, "USE_LOCAL_CACHE", self.original_cache_setting)

        self.physical_path = str(Path(self.temp_dir.name) / "physical.parquet")
        self.physical_df = pd.DataFrame(
            {
                "object_id": [1, -2, 3, 4],
                "phz_median": [0.2, 0.4, 0.6, 0.8],
                "phz_flags": [0.0, 11.0, 0.0, 0.0],
                "phz_pp_median_stellarmass": [10.0, 11.0, 12.0, np.inf],
                "phys_param_flags": [0.0, 0.0, 1.0, 0.0],
                "flux_detection_total": [100.0, 200.0, 300.0, 400.0],
                "concentration": [2.0, 3.0, 4.0, 5.0],
                "sersic_sersic_vis_index": [1.0, 2.0, 3.0, 4.0],
                "sersic_visnir_flags": [0, 0, 8192, 0],
            }
        )
        self.physical_df.to_parquet(self.physical_path, index=False)

    def test_subset_reader_normalizes_signed_object_ids_and_projects_columns(self):
        result = load_physical_measurements(
            self.physical_path,
            ["-2", "3"],
            columns=["phz_median"],
        )
        self.assertEqual(list(result.columns), ["object_id", "phz_median"])
        self.assertEqual(set(result["object_id"]), {"-2", "3"})

        single = load_physical_measurement_object(self.physical_path, -2)
        self.assertEqual(single.iloc[0]["object_id"], "-2")

    def test_quality_flags_and_non_finite_values_are_excluded(self):
        clean = analysis_ready_physical_measurements(self.physical_df)
        self.assertTrue(pd.isna(clean.loc[1, "phz_median"]))
        self.assertTrue(pd.isna(clean.loc[2, "phz_pp_median_stellarmass"]))
        self.assertTrue(pd.isna(clean.loc[3, "phz_pp_median_stellarmass"]))
        self.assertTrue(pd.isna(clean.loc[2, "sersic_sersic_vis_index"]))

        summary = build_physical_summary(self.physical_df, total_objects=4).set_index(
            "field"
        )
        self.assertEqual(int(summary.loc["phz_median", "valid_objects"]), 3)
        self.assertEqual(
            int(summary.loc["phz_pp_median_stellarmass", "valid_objects"]),
            2,
        )

        display_source = pd.Series(
            {
                "object_id": "1",
                "phz_pp_median_sfr": -np.inf,
                "gini": 0.6,
            }
        )
        displayed_fields = {
            row["field"] for row in physical_measurement_display_rows(display_source)
        }
        self.assertNotIn("phz_pp_median_sfr", displayed_fields)
        self.assertIn("gini", displayed_fields)

        curated_rows = physical_measurement_display_rows(self.physical_df.iloc[0])
        curated_fields = {row["field"] for row in curated_rows}
        self.assertNotIn("flux_detection_total", curated_fields)
        self.assertIn("phz_median", curated_fields)

    def test_physical_filters_are_normalized_signed_and_combined_with_and(self):
        source = pd.DataFrame(
            {
                "object_id": ["1", "-2", "3", "4"],
                "feat_pca_0": [0.0, 1.0, 2.0, 3.0],
            }
        )
        filters = normalize_physical_filters(
            [
                {
                    "field": "phz_median",
                    "operator": "between",
                    "lower": 0.7,
                    "upper": 0.1,
                },
                {
                    "field": "concentration",
                    "operator": ">=",
                    "value": 4.0,
                },
            ],
            ["phz_median", "concentration"],
        )
        self.assertEqual(filters[0]["lower"], 0.1)
        self.assertTrue(physical_filter_signature(filters))

        filtered = apply_physical_filters(source, self.physical_df, filters)
        self.assertEqual(filtered["object_id"].tolist(), ["3"])

    def test_grouped_summary_and_export_merge_keep_one_row_per_object(self):
        membership = pd.DataFrame(
            {
                "object_id": ["1", "-2", "3", "4"],
                "hierarchical_subcluster": [0, 0, 1, 1],
            }
        )
        grouped = build_grouped_physical_summary(
            self.physical_df,
            membership,
            "hierarchical_subcluster",
        )
        self.assertEqual(set(grouped["hierarchical_subcluster"]), {0, 1})

        export = merge_physical_measurements(
            membership[["object_id"]],
            pd.concat([self.physical_df, self.physical_df.iloc[:1]], ignore_index=True),
        )
        self.assertEqual(len(export), 4)
        self.assertIn("phz_median", export.columns)
        self.assertNotIn("flux_detection_total", export.columns)
        self.assertNotIn("phys_param_flags", export.columns)

    def test_recommended_numeric_filter_requires_support_and_finds_enrichment(self):
        supported = pd.DataFrame(
            {
                "measurement": np.linspace(0.0, 1.0, 40),
                "is_lens": [False] * 34 + [True] * 6,
            }
        )
        recommendation = recommended_numeric_filter(
            supported,
            "measurement",
            min_valid_objects=25,
            min_selected_objects=5,
            min_lenses=2,
            min_unknowns=2,
            min_recall=0.05,
        )
        self.assertIsNotNone(recommendation)
        self.assertEqual(recommendation["operator"], ">=")
        self.assertGreater(float(recommendation["enrichment"]), 1.0)

        unsupported = supported.copy()
        unsupported["is_lens"] = [False] * 39 + [True]
        self.assertIsNone(
            recommended_numeric_filter(
                unsupported,
                "measurement",
                min_valid_objects=25,
                min_selected_objects=5,
                min_lenses=2,
                min_unknowns=2,
                min_recall=0.05,
            )
        )


if __name__ == "__main__":
    unittest.main()
