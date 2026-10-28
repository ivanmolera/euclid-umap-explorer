import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, patch

from euclid_umap_explorer.canonical_cache import CanonicalScoreStore
from tests.test_cluster_diagnostics import png, test_images


class CanonicalCacheTests(unittest.TestCase):
    def test_persistent_cache_source_and_detector_invalidation_and_corruption(self):
        with tempfile.TemporaryDirectory() as folder:
            image = Path(folder) / "cutout.png"
            image.write_bytes(png(test_images()[0]))
            base = str(Path(folder) / "cache")
            first = CanonicalScoreStore(cache_base=base)
            evidence, cached = first.score_candidates([str(image)])
            self.assertFalse(cached)
            self.assertIsNotNone(evidence["score"])
            self.assertNotIn("overlay_png", evidence)
            second = CanonicalScoreStore(cache_base=base)
            with patch("euclid_umap_explorer.canonical_cache.score_image_bytes") as score:
                self.assertEqual(second.score_candidates([str(image)]), (json.loads(json.dumps(evidence)), True))
                score.assert_not_called()
            cache_file = next((Path(folder) / "cache").rglob("*.json"))
            self.assertNotIn("original_bytes", json.loads(cache_file.read_text())["evidence"])
            cache_file.write_text("corrupted")
            self.assertFalse(second.score_candidates([str(image)])[1])
            image.write_bytes(png(test_images()[1]))
            self.assertFalse(second.score_candidates([str(image)])[1])
            second.detector = {**second.detector, "version": "changed"}
            self.assertFalse(second.score_candidates([str(image)])[1])

    def test_quality_na_is_cached_but_missing_files_are_not_zero(self):
        with tempfile.TemporaryDirectory() as folder:
            image = Path(folder) / "small.png"
            image.write_bytes(png(test_images()[0][:20, :20]))
            store = CanonicalScoreStore(cache_base=str(Path(folder) / "cache"))
            result, hit = store.score_candidates([str(image)])
            self.assertIsNone(result["score"])
            self.assertEqual(result["status"], "insufficient_resolution")
            self.assertFalse(hit)
            self.assertTrue(store.score_candidates([str(image)])[1])
            self.assertEqual(store.score_candidates([str(image) + "missing"])[0]["status"], "missing_cutout")

    def test_gcs_generation_is_pinned_and_failed_cache_writes_do_not_hide_scores(self):
        filesystem = Mock()
        filesystem.info.return_value = {"generation": "42", "size": 123}
        payload = png(test_images()[0])
        opened = []
        def open_file(path, mode, **kwargs):
            opened.append((path, mode, kwargs))
            if "analysis_cache" in path:
                raise FileNotFoundError() if mode == "rb" else PermissionError()
            return BytesIO(payload)
        filesystem.open.side_effect = open_file
        store = CanonicalScoreStore(filesystem=filesystem, cache_base="gs://bucket/analysis_cache/canonical_scores")
        result, cached = store.score_candidates(["gs://bucket/image.jpg"])
        self.assertIsNotNone(result["score"])
        self.assertFalse(cached)
        self.assertEqual(store.cache_issue, "PermissionError")
        self.assertIn(("gs://bucket/image.jpg", "rb", {"generation": "42", "timeout": 30}), opened)


if __name__ == "__main__":
    unittest.main()
