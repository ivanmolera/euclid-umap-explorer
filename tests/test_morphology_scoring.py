import json
import unittest
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

from euclid_umap_explorer import canonical_scoring, lens_plausibility, morphology_scoring as scoring
from euclid_umap_explorer.canonical_scores import canonical_detail_html
from tests.test_cluster_diagnostics import png


def galaxy():
    yy, xx = np.mgrid[:160, :160]
    gray = 10 + 220 * np.exp(-((xx - 80)**2 + (yy - 80)**2) / (2 * 5**2))
    return cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2RGB)


class CurveAwareScoringTests(unittest.TestCase):
    def test_recovers_inner_centroid_arc_and_scores_single_region(self):
        rgb = galaxy()
        cv2.ellipse(rgb, (80, 80), (17, 17), 0, 20, 181, (125,) * 3, 2)
        old = lens_plausibility.analyze_image(rgb)
        self.assertFalse(old["accepted"])
        self.assertLess(old["components"][0]["radius_px"], 14.4)

        scored = scoring.score_image_bytes(png(rgb))
        self.assertEqual(len(scored["regions"]), 1)
        arc = scored["regions"][0]
        self.assertEqual(arc["shape"], "Arc-like")
        self.assertGreater(arc["radius_px"], 14.4)
        self.assertAlmostEqual(scored["score"], 80 * arc["component_score"])
        self.assertGreater(scored["score"], 55.)

    def test_recovers_round_ring_without_global_elongation(self):
        rgb = galaxy()
        cv2.circle(rgb, (80, 80), 26, (125,) * 3, 2)
        self.assertFalse(lens_plausibility.analyze_image(rgb)["accepted"])
        image = png(rgb)
        scored = scoring.score_image_bytes(image)
        ring = scored["regions"][0]
        self.assertEqual(ring["shape"], "Ring-like")
        self.assertLess(ring["elongation"], 1.25)
        self.assertAlmostEqual(ring["radius_px"], 26., delta=1.)
        self.assertEqual(ring["angular_coverage_deg"], 360.)
        self.assertEqual(ring["contiguous_coverage_deg"], 360.)
        self.assertGreater(scored["score"], canonical_scoring.score_image_bytes(image)["score"])
        compact = scoring.score_image_bytes(image, include_overlay=False)
        self.assertEqual(compact, {key: value for key, value in scored.items() if key != "overlay_png"})
        json.dumps(compact, allow_nan=False)

    def test_angular_coverage_distinguishes_fragments_and_wraps(self):
        count, run = scoring._angular_support(np.deg2rad([342, 352, 2, 12, 172, 182]))
        self.assertEqual((count, run), (6, 4))

    def test_cyan_circle_uses_top_ranked_region_not_the_neighbours(self):
        rgb = np.zeros((120, 120, 3), dtype=np.uint8)
        regions = [
            {"radius_px": radius, "pixels": np.array([[x, 60]]), "centroid_x": x, "centroid_y": 60}
            for x, radius in ((80, 20), (105, 45))
        ]
        with Image.open(BytesIO(scoring._overlay_png(rgb, regions, (60, 60), []))) as image:
            overlay = np.asarray(image)
        self.assertGreater(int(overlay[40, 60, 2]), 100)
        self.assertEqual(int(overlay[28, 60, 2]), 0)

    def test_does_not_boost_wide_straight_tangential_streak(self):
        rgb = galaxy()
        cv2.line(rgb, (38, 54), (119, 54), (220,) * 3, 2)
        self.assertTrue(lens_plausibility.analyze_image(rgb)["accepted"])
        self.assertEqual(scoring.score_image_bytes(png(rgb))["score"], 0.)

    def test_excluded_border_line_is_visible_and_never_scores(self):
        rgb = galaxy()
        plain = scoring.score_image_bytes(png(rgb))
        self.assertFalse(plain["artifact_lines"])
        self.assertNotIn("Orange: excluded border lines.", canonical_detail_html(plain, "original.jpg"))

        cv2.line(rgb, (3, 5), (157, 43), (220,) * 3, 2)
        scored = scoring.score_image_bytes(png(rgb))
        self.assertTrue(scored["artifact_lines"])
        self.assertEqual(scored["score"], 0.)
        self.assertIn("Orange: excluded border lines.", canonical_detail_html(scored, "original.jpg"))
        with Image.open(BytesIO(scored["overlay_png"])) as image:
            overlay = np.asarray(image)
        orange = np.all(overlay == [255, 165, 0], axis=2)
        self.assertGreater(int(orange.sum()), 100)
        for x1, y1, x2, y2 in scored["artifact_lines"]:
            self.assertTrue(orange[round((y1 + y2) / 2), round((x1 + x2) / 2)])
