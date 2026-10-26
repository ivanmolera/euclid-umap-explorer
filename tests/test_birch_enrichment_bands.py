import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from euclid_umap_explorer import analyze_birch_bands


class BirchEnrichmentBandsTests(unittest.TestCase):
    def test_keeps_fully_and_mostly_labelled_clusters_with_both_minima(self):
        with tempfile.TemporaryDirectory() as directory:
            clusters = pd.DataFrame({
                "n_objects": [100, 100, 99, 200, 99_501],
                "n_references": [100, 75, 99, 49, 0],
            })
            clusters["enrichment_x"] = (
                clusters.n_references / clusters.n_objects / (323 / 100_000)
            )
            clusters.to_parquet(Path(directory) / "t5.0_b25.parquet", index=False)
            with patch.object(analyze_birch_bands, "CONFIGURATIONS", ((5.0, 25),)):
                result = analyze_birch_bands.summarize_bands(
                    Path(directory), object_count=100_000, reference_count=323,
                )

        row = result.iloc[0]
        self.assertEqual(row["n_supported_clusters"], 2)
        self.assertEqual(row["ge_50x"], 2)
        self.assertEqual(row["references_ge_50x"], 175)
        self.assertEqual(row["objects_ge_50x"], 200)

    def test_disjoint_bands_include_lower_boundary_and_sum_to_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            clusters = pd.DataFrame({
                "n_objects": [100, 200, 200, 9_500],
                "n_references": [50, 10, 20, 20],
            })
            clusters["enrichment_x"] = (
                clusters["n_references"] / clusters["n_objects"] / 0.01
            )
            clusters.to_parquet(Path(directory) / "t5.0_b25.parquet", index=False)
            with patch.object(analyze_birch_bands, "CONFIGURATIONS", ((5.0, 25),)):
                result = analyze_birch_bands.summarize_bands(
                    Path(directory), object_count=10_000, reference_count=100,
                    min_references=10,
                )

        row = result.iloc[0]
        self.assertEqual(row["n_supported_clusters"], 4)
        self.assertEqual(row["0-5x"], 1)
        self.assertEqual(row["5-10x"], 1)
        self.assertEqual(row["10-15x"], 1)
        self.assertEqual(row["ge_50x"], 1)
        self.assertEqual(row["clusters_ge_10x"], 2)
        self.assertEqual(row["references_ge_10x"], 70)
        self.assertEqual(row["references_ge_50x"], 50)
        self.assertEqual(row["objects_ge_50x"], 100)


if __name__ == "__main__":
    unittest.main()
