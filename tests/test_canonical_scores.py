import unittest
from unittest.mock import patch

import pandas as pd

from euclid_umap_explorer import canonical_scores as scores
from tests.test_cluster_diagnostics import FEATURES, fixture, png, test_images
from euclid_umap_explorer.cluster_diagnostics import build_diagnostic_profiles
from euclid_umap_explorer.lens_plausibility import score_image_bytes


class CanonicalPreviewScoresTests(unittest.TestCase):
    def test_scores_all_clusters_once_and_invalidates_changed_catalogue(self):
        data = pd.concat([fixture(), fixture().assign(object_id=lambda frame: frame.object_id + "1")], ignore_index=True)
        model = {"cluster_download_df": pd.DataFrame({"canonical": data.object_id.iloc[:22]})}
        state = {}
        with patch.object(scores.st, "session_state", state), \
                patch.object(scores, "score_filesystem"), \
                patch.object(scores, "score_canonical_row", return_value=({"score": 86, "status": "exploratory_visual_only", "overlay_png": b"large"}, False)) as load:
            scores.ensure_preview_canonical_scores(model, data, FEATURES)
            self.assertEqual(load.call_count, 22)
            self.assertEqual(len(state[scores.STATE_KEY]["evidence"]), 22)
            self.assertTrue(all("overlay_png" not in value for value in state[scores.STATE_KEY]["evidence"].values()))
            scores.ensure_preview_canonical_scores(model, data, FEATURES)
            self.assertEqual(load.call_count, 22)
            scores.ensure_preview_canonical_scores(model, data.copy(), FEATURES)
            self.assertEqual(load.call_count, 44)

    def test_failed_images_and_zero_scores_are_distinct_in_summary(self):
        data = fixture()
        profiles, canonical = build_diagnostic_profiles(data, FEATURES)
        model = {"cluster_download_df": profiles}
        entries = {str(row.object_id): value for row, value in zip(canonical.values(), [
            dict(score=0, status="exploratory_visual_only"), OSError("missing"),
            dict(score=86, status="exploratory_visual_only"), dict(score=50, status="exploratory_visual_only")])}
        def score_row(store, row):
            result = entries[str(row.object_id)]
            if isinstance(result, Exception):
                raise result
            return result, False
        with patch.object(scores.st, "session_state", {}), patch.object(scores, "score_filesystem"), \
                patch.object(scores, "score_canonical_row", side_effect=score_row):
            scores.ensure_preview_canonical_scores(model, data, FEATURES)
            summary = scores.summary_with_preview_scores(profiles, data, FEATURES)
            self.assertEqual(summary.canonical_score.iloc[0], 0)
            self.assertTrue(pd.isna(summary.canonical_score.iloc[1]))
            self.assertEqual(summary.canonical_score_status.iloc[1], "image_error")
            self.assertEqual(summary.n_grade_a.tolist(), [0, 1, 1, 1])
            self.assertTrue(scores.summary_with_preview_scores(profiles, data.copy(), FEATURES).canonical_score.isna().all())

    def test_modal_contains_native_original_annotation_measurements_and_help(self):
        image = png(test_images()[0])
        evidence = {**score_image_bytes(image), "original_bytes": image, "original_mime": "image/png"}
        output = scores.canonical_detail_html(evidence, "thumbnail.jpg")
        self.assertEqual(output.count("<img "), 2)
        for expected in ("Original cutout", "Annotated cutout", "Canonical score:", "Tangential alignment",
                         "Region score (0-1)", "Best region", "concept-popover", "not P(lens)"):
            self.assertIn(expected, output)
        self.assertNotIn("thumbnail.jpg", output)
        missing = scores.canonical_detail_html({"score": None, "status": "insufficient_contrast"}, "original.jpg")
        self.assertIn("Canonical score: N/A", missing)
        self.assertIn("insufficient contrast", missing)

    def test_thumbnail_adds_diagnostics_only_for_the_canonical_and_preserves_controls(self):
        from euclid_umap_explorer import components

        image = png(test_images()[0])
        evidence = {**score_image_bytes(image), "original_bytes": image, "original_mime": "image/png"}
        row = fixture().iloc[0]
        with patch.object(components, "preview_canonical_evidence", return_value=evidence), \
                patch.object(components.st, "html") as render:
            components.show_thumbnail(row, "Canonical", image_src="thumbnail.jpg", aladin_link="https://aladin.cds.unistra.fr/")
            output = render.call_args.args[0]
            for expected in ('<div class="canonical-analysis">', 'class="thumb-caption canonical-score"',
                             "modal.showPopover()", "thumb-copy-id", "thumb-aladin-link", "Region score (0-1)"):
                self.assertIn(expected, output)
            components.show_thumbnail(row, "Anomalous", image_src="thumbnail.jpg")
            output = render.call_args.args[0]
            self.assertNotIn('<div class="canonical-analysis">', output)
            self.assertNotIn('class="thumb-caption canonical-score"', output)


if __name__ == "__main__":
    unittest.main()
