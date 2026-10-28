import unittest

import numpy as np
import pandas as pd

from euclid_umap_explorer.analysis import build_cluster_summary
from euclid_umap_explorer.components import build_cluster_summary_view_model, ranked_cluster_summary_view_model
from euclid_umap_explorer.review_priority import prioritize_for_review


class ReviewPriorityTests(unittest.TestCase):
    def summary(self):
        return pd.DataFrame({
            "cluster": [1, 2, 3, 4, 5, 6, 7], "n_objects": [200, 100, 100, 100, 100, 99, 100],
            "n_lenses": [50, 25, 25, 0, 24, 30, 25], "n_grade_a": [30, 10, 15, 0, 3, 20, 0],
            "n_grade_b": [10, 10, 5, 0, 0, 5, 0], "n_grade_c": [10, 5, 5, 0, 21, 5, 25],
            "canonical_score": [99., 99., 80., 70., 69., 100., np.nan],
        })

    def test_reference_ab_enrichment_then_a_then_score_and_exploration(self):
        frame = self.summary()
        ranked = prioritize_for_review(frame)
        self.assertEqual(ranked.cluster.tolist(), [1, 3, 2, 4, 6, 5, 7])
        self.assertEqual(ranked.review_priority.tolist(), ["Reference-supported"] * 3 + ["Morphology exploration"] + ["Other"] * 3)
        self.assertEqual(len(ranked), len(frame))
        baseline = (frame.n_grade_a.sum() + frame.n_grade_b.sum()) / frame.n_objects.sum()
        self.assertAlmostEqual(ranked.enrichment_ab.iloc[0], .2 / baseline)
        changed = frame.copy()
        changed.loc[changed.cluster.eq(2), "n_grade_a"] = 15
        changed.loc[changed.cluster.eq(2), "n_grade_b"] = 5
        self.assertEqual(prioritize_for_review(changed).cluster.tolist()[:3], [1, 2, 3])

    def test_no_ab_is_undefined_and_unknowns_can_be_explored(self):
        frame = self.summary().assign(n_grade_a=0, n_grade_b=0)
        ranked = prioritize_for_review(frame)
        self.assertTrue(ranked.enrichment_ab.isna().all())
        self.assertEqual(ranked.cluster.iloc[0], 4)
        self.assertEqual(ranked.review_priority.iloc[0], "Morphology exploration")
        self.assertEqual(prioritize_for_review(frame, exploration_score=71).review_priority.eq("Morphology exploration").sum(), 0)

    def test_singletons_have_canonicals_and_ranked_previews_match_table(self):
        data = pd.DataFrame({"cluster": [0, 1, 1], "object_id": ["a", "b", "c"],
                             "is_lens": [False, True, False], "feat_pca_0": [0., 1., 2.]})
        summary = build_cluster_summary(data)
        base = build_cluster_summary_view_model(data, summary, ["feat_pca_0"], ["feat_pca_0"])
        self.assertEqual(base["cluster_download_df"].set_index("cluster").loc[0, "canonical"], "a")
        reverse = base["cluster_download_df"].iloc[::-1]
        ranked = ranked_cluster_summary_view_model(base, reverse, data, ["feat_pca_0"])
        self.assertEqual([item["cluster_id"] for item in ranked["clusters"]], reverse.cluster.tolist())
        self.assertIsNone(next(item for item in ranked["clusters"] if item["cluster_id"] == 0)["anomaly_row"])


if __name__ == "__main__":
    unittest.main()
