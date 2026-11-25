import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from euclid_umap_explorer import canonical_scores as scores
from euclid_umap_explorer import components
from euclid_umap_explorer import morphology_scoring
from euclid_umap_explorer.morphology_scoring import score_image_bytes as active_score
from tests.test_cluster_diagnostics import FEATURES, fixture, png, test_images
from euclid_umap_explorer.cluster_diagnostics import build_diagnostic_profiles
from euclid_umap_explorer.lens_plausibility import score_image_bytes


class CanonicalPreviewScoresTests(unittest.TestCase):
    def test_scoring_fingerprint_invalidates_session_and_summary(self):
        data = fixture()
        profiles, _ = build_diagnostic_profiles(data, FEATURES)
        model = {"cluster_download_df": profiles}
        state = {}
        with patch.object(scores.st, "session_state", state), patch.object(scores, "score_filesystem"), \
                patch.object(scores, "score_canonical_row", return_value=({"score": 79.7, "status": "exploratory_visual_only"}, False)) as load:
            scores.ensure_preview_canonical_scores(model, data, FEATURES)
            summary = scores.summary_with_preview_scores(profiles, data, FEATURES)
            self.assertEqual(summary.canonical_scoring_model.unique().tolist(), [scores.MODEL_NAME])
            self.assertEqual(summary.canonical_weights_sha256.unique().tolist(), [scores.MODEL_SHA256])
            self.assertEqual(summary.canonical_score.tolist(), [79.7] * 4)
            fingerprint = {**scores.detector_fingerprint(), "scoring": {"version": "changed"}}
            with patch.object(scores, "detector_fingerprint", return_value=fingerprint):
                self.assertTrue(scores.summary_with_preview_scores(profiles, data, FEATURES).canonical_score.isna().all())
                self.assertIsNone(scores.preview_canonical_evidence(data.iloc[0]))
                scores.ensure_preview_canonical_scores(model, data, FEATURES)
                self.assertEqual(load.call_count, 8)

    def test_help_and_modal_describe_the_active_curve_aware_model(self):
        for phrase in (
            "38.11% contrast", "19.05% radial coherence", "16.85% tangential alignment",
            "16.62% angular extent", "9.37% elongation", "35% occupied angular sectors",
            "30% radial coherence", "20% contrast", "15% continuous angular coverage",
            "1.6x that radius", "25% at 2.5x", "Without a qualifying region",
            "80% best region", "15% second region", "5% arc-pair geometry", "not P(lens)",
        ):
            self.assertIn(phrase, scores.CANONICAL_SCORE_HELP)
            for label in ("Score", "Canonical score"):
                self.assertIn(phrase, scores.score_help_html(label))
        evidence = active_score(png(test_images()[0]))
        markup = scores.canonical_detail_html(evidence, "original.jpg")
        self.assertIn("Shape", markup)
        self.assertIn("Median radius (px)", markup)
        self.assertIn("Radial weight", markup)
        self.assertIn("Contribution (points)", markup)
        self.assertNotIn("Orange: excluded border lines.", markup)
        self.assertEqual(scores.canonical_score_label({"score": 79.7}), "79.7 / 100")

    def test_score_help_uses_active_weights_without_duplicating_values(self):
        with patch.dict(morphology_scoring.POOL_WEIGHTS, best_region=.775, second_region=.175), \
                patch.dict(morphology_scoring.RING_WEIGHTS, coverage=.325, continuity=.175), \
                patch.dict(morphology_scoring.legacy.REGION_WEIGHTS, contrast=.4), \
                patch.dict(morphology_scoring.RADIAL_WEIGHTING, min_weight=.275, min_weight_radius_ratio=2.75):
            help_text = scores._score_help_text()
        for value in ("77.5% best region", "17.5% second region", "32.5% occupied angular sectors",
                      "17.5% continuous angular coverage", "40% contrast", "27.5% at 2.75x"):
            self.assertIn(value, help_text)

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
        self.assertNotIn("Annotated cutout", missing)

    def test_selected_cutout_opens_score_dialog_only_on_request(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "original.png"
            image = png(test_images()[0])
            path.write_bytes(image)
            app = AppTest.from_string(f"""
from euclid_umap_explorer.components import aladin_caption, show_detectable_image
show_detectable_image(
    {str(path)!r}, "Morphology cutout", "test_cutout",
    caption_markdown=aladin_caption("Morphology cutout", 12.5, -4.0),
    score_object_id="-604185454503177752",
)
""")
            with patch.object(components, "ENABLE_ARC_LIKE_STRUCTURE_DETECTION", False), \
                    patch.object(components, "load_cutout_evidence", wraps=scores.load_cutout_evidence) as load:
                app.run()
                self.assertFalse(app.exception)
                load.assert_not_called()
                self.assertTrue(any("View in Aladin" in element.proto.body for element in app.get("html")))
                app.button(key="test_cutout_morphology_score_-604185454503177752").click().run()
                self.assertFalse(app.exception)
                load.assert_called_once_with(str(path))
            detail = next(element.proto.body for element in app.get("html") if 'class="canonical-analysis"' in element.proto.body)
            value = active_score(image)["score"]
            self.assertIn(f"Score: {value:.1f} / 100", detail)
            self.assertNotIn("Canonical score:", detail)
            for expected in ("Original cutout", "Annotated cutout", "Arc-like",
                             "Contribution (points)", "Multi-region geometry", "not P(lens)"):
                self.assertIn(expected, detail)
            self.assertNotIn(scores.STATE_KEY, app.session_state)

            with patch.object(components, "load_cutout_evidence", side_effect=OSError("unreadable")):
                app.button(key="test_cutout_morphology_score_-604185454503177752").click().run()
            self.assertFalse(app.exception)
            detail = next(element.proto.body for element in app.get("html") if 'class="canonical-analysis"' in element.proto.body)
            self.assertIn("Score: N/A", detail)
            self.assertIn("image analysis failed", detail)
            self.assertNotIn("Annotated cutout", detail)

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
