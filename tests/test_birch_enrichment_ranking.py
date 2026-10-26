import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from euclid_umap_explorer import analyze_birch_enrichment, rank_birch_enrichment


def cluster_row(threshold, branching, cluster, n_objects, a, b, c):
    return dict(
        catalogue="filtered", threshold=threshold, branching_factor=branching,
        cluster=cluster, n_objects=n_objects, n_labelled_abc=a + b + c,
        n_grade_a=a, n_grade_b=b, n_grade_c=c,
    )


class BirchEnrichmentRankingTests(unittest.TestCase):
    def test_uncapped_ranking_keeps_minima_and_zero_configurations(self):
        data = pd.DataFrame([
            cluster_row(5.0, 25, 0, 100, 100, 0, 0),
            cluster_row(5.0, 25, 1, 100, 50, 25, 0),
            cluster_row(5.0, 25, 2, 99, 99, 0, 0),
            cluster_row(5.0, 25, 3, 100, 49, 0, 0),
            cluster_row(5.0, 26, 0, 100, 100, 0, 0),
        ])
        result = rank_birch_enrichment.rank_configurations(data)
        filtered = result.loc[result.catalogue.eq("filtered")]
        winner = filtered.iloc[0]
        self.assertEqual(len(filtered), 651)
        self.assertEqual(winner.branching_factor, 25)
        self.assertEqual(winner.clusters_ge_50x, 2)
        self.assertEqual(winner.references_ge_50x, 175)
        self.assertEqual(winner.grade_a_ge_50x, 150)
        self.assertEqual(winner.unknown_ge_50x, 25)
        zero = filtered.loc[filtered.threshold.eq(7.0) & filtered.branching_factor.eq(55)].iloc[0]
        self.assertEqual(zero.clusters_ge_50x, 0)

    def test_ties_prefer_a_then_b_then_c_not_total_labels(self):
        data = pd.DataFrame([
            cluster_row(5.0, 25, 0, 100, 40, 1, 9),
            cluster_row(5.0, 26, 0, 100, 39, 30, 30),
            cluster_row(5.0, 27, 0, 100, 40, 2, 8),
            cluster_row(5.0, 28, 0, 100, 40, 2, 9),
        ])
        result = rank_birch_enrichment.rank_configurations(data)
        leading = result.loc[result.catalogue.eq("filtered")].head(4)
        self.assertEqual(leading.branching_factor.tolist(), [28, 27, 25, 26])

    def test_rejects_inconsistent_grades(self):
        data = pd.DataFrame([cluster_row(5.0, 25, 0, 100, 50, 0, 0)])
        data.loc[0, "n_labelled_abc"] = 51
        with self.assertRaisesRegex(ValueError, "Invalid object or A/B/C counts"):
            rank_birch_enrichment.rank_configurations(data)

    def test_analyzer_includes_over_half_labels_and_marks_cap_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clusters = pd.DataFrame({
                "n_objects": [100, 100, 99, 200, 99_501],
                "n_references": [100, 75, 99, 49, 0],
            })
            clusters["lens_rate"] = clusters.n_references / clusters.n_objects
            clusters["enrichment_x"] = clusters.lens_rate / (323 / 100_000)
            configurations = ((6.2, 50), (5.7, 49))
            for threshold, branching in configurations:
                clusters.to_parquet(root / f"t{threshold:.1f}_b{branching}.parquet", index=False)
            with patch.object(analyze_birch_enrichment, "CONFIGURATIONS", configurations):
                with contextlib.redirect_stdout(io.StringIO()):
                    analyze_birch_enrichment.analyze(root, root / "output", 100_000, 323)
            counts = pd.read_csv(root / "output/counts_by_configuration_and_support.csv")
            main = counts.loc[counts.min_references.eq(50) & counts.min_objects.eq(100)]
            self.assertTrue(main.qualifying_clusters.eq(2).all())
            self.assertTrue(main.max_labelled_fraction.isna().all())
            notable = pd.read_csv(root / "output/notable_clusters_default_and_selected.csv")
            self.assertEqual(len(notable), 4)


if __name__ == "__main__":
    unittest.main()
