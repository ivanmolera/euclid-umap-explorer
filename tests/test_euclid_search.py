import sys
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from astropy.table import Table

from euclid_umap_explorer import euclid_search as search


OBJECT_ID = "-608794490472715426"
ID_STR = f"Q1_R1_102021984_{OBJECT_ID}"
MORPHOLOGY = pd.DataFrame([{
    "object_id": OBJECT_ID, "id_str": ID_STR,
    "right_ascension": 60.879449, "declination": -47.271543,
    "segmentation_area": 900, "gini": 0.6,
}])


class EuclidSearchTests(unittest.TestCase):
    def test_query_validates_missing_job_or_results(self):
        for job in (None, Mock(get_results=Mock(return_value=None))):
            with self.subTest(job=job):
                euclid = Mock(launch_job_async=Mock(return_value=job))
                with patch.dict(sys.modules, {
                    "astroquery.esa.euclid": SimpleNamespace(Euclid=euclid),
                }):
                    with self.assertRaises(search.EuclidArchiveError):
                        search.query_archive("SELECT object_id FROM q1.mer_catalogue")

    def test_query_wraps_launch_and_result_failures(self):
        for euclid in (
            Mock(launch_job_async=Mock(side_effect=TimeoutError("timeout"))),
            Mock(launch_job_async=Mock(return_value=Mock(
                get_results=Mock(side_effect=RuntimeError("HTTP 500"))))),
        ):
            with self.subTest(euclid=euclid):
                with patch.dict(sys.modules, {
                    "astroquery.esa.euclid": SimpleNamespace(Euclid=euclid),
                }):
                    with self.assertRaises(search.EuclidArchiveError) as caught:
                        search.query_archive("SELECT 1")
                    self.assertIsNotNone(caught.exception.__cause__)

    def test_successful_empty_result_is_not_an_archive_error(self):
        table = Table(names=["object_id"])
        euclid = Mock(launch_job_async=Mock(return_value=Mock(
            get_results=Mock(return_value=table))))
        with patch.dict(sys.modules, {
            "astroquery.esa.euclid": SimpleNamespace(Euclid=euclid),
        }):
            self.assertIs(search.query_archive("SELECT 1"), table)

    def test_object_query_uses_current_q1_schema_and_exact_identifier(self):
        with patch.object(search, "query_archive", side_effect=search.EuclidArchiveError("unavailable")) as query:
            with patch.object(search, "morphology_search_fallback", return_value=None):
                with self.assertRaises(search.EuclidArchiveError):
                    search.fetch_euclid_object_summary(OBJECT_ID)
        query.assert_called_once()
        self.assertIn("FROM q1.mer_catalogue", query.call_args.args[0])
        self.assertNotIn("catalogue.mer_catalogue", query.call_args.args[0])
        self.assertIn(f"WHERE object_id = {OBJECT_ID}", query.call_args.args[0])

    def test_failed_archive_uses_exact_catalogue_record_and_original_cutout(self):
        with patch.object(search, "query_archive", side_effect=search.EuclidArchiveError("HTTP 500")) as query:
            with patch.object(search, "load_morphology_object", return_value=MORPHOLOGY) as load:
                with patch.object(search, "morphology_cutout_path", return_value="/original.jpg") as image:
                    result = search.fetch_euclid_object_summary(OBJECT_ID)
        self.assertEqual(result["object_id"], OBJECT_ID)
        self.assertEqual(result["cutout_path"], "/original.jpg")
        self.assertEqual(result["tile_index"], "102021984")
        self.assertEqual(result["mosaic_summary"], {})
        self.assertIsNone(result["file_path"])
        self.assertIn("Q1 morphology", result["object_summary"]["data_source"])
        self.assertIn("instead", result["warnings"][0])
        query.assert_called_once()
        self.assertIn(f"WHERE object_id = {OBJECT_ID}", query.call_args.args[0])
        load.assert_called_once_with(search.MORPH_PATH, OBJECT_ID)
        image.assert_called_once_with(ID_STR, OBJECT_ID)

    def test_failed_archive_without_catalogue_does_not_claim_object_is_absent(self):
        with patch.object(search, "query_archive", side_effect=search.EuclidArchiveError("HTTP 500")):
            with patch.object(search, "load_morphology_object", return_value=pd.DataFrame()):
                with self.assertRaises(search.EuclidArchiveError) as caught:
                    search.fetch_euclid_object_summary(OBJECT_ID)
        self.assertIn("unavailable", str(caught.exception))
        self.assertIn("try again", str(caught.exception))

    def test_failed_archive_and_unreadable_catalogue_have_controlled_error(self):
        with patch.object(search, "query_archive", side_effect=search.EuclidArchiveError("HTTP 500")):
            with patch.object(search, "load_morphology_object", side_effect=OSError("storage")):
                with self.assertRaises(search.EuclidArchiveError):
                    search.fetch_euclid_object_summary(OBJECT_ID)

    def test_empty_archive_result_can_still_use_hosted_q1_catalogue(self):
        with patch.object(search, "query_archive", return_value=Table(names=["object_id"])):
            with patch.object(search, "load_morphology_object", return_value=MORPHOLOGY):
                with patch.object(search, "morphology_cutout_path", return_value=None):
                    result = search.fetch_euclid_object_summary(OBJECT_ID)
        self.assertEqual(result["object_id"], OBJECT_ID)
        self.assertIsNone(result["cutout_path"])
        self.assertIsNotNone(result["object_summary"]["right_ascension"])

    def test_absent_object_with_successful_empty_query_is_reported_as_not_found(self):
        with patch.object(search, "query_archive", return_value=Table(names=["object_id"])):
            with patch.object(search, "load_morphology_object", return_value=pd.DataFrame()):
                with self.assertRaisesRegex(ValueError, "was not found in q1.mer_catalogue"):
                    search.fetch_euclid_object_summary(OBJECT_ID)

    def test_mosaic_failures_do_not_hide_object_or_existing_cutout(self):
        objects = Table({
            "object_id": [int(OBJECT_ID)],
            "right_ascension": [60.879449], "declination": [-47.271543],
        })
        for mosaic in (search.EuclidArchiveError("HTTP 500"), Table(names=["file_name"])):
            with self.subTest(mosaic=mosaic):
                with ExitStack() as stack:
                    stack.enter_context(patch.object(search, "query_archive", side_effect=[objects, mosaic]))
                    stack.enter_context(patch.object(search, "load_morphology_object", return_value=MORPHOLOGY))
                    stack.enter_context(patch.object(search, "morphology_cutout_path", side_effect=[None, "/original.jpg"]))
                    result = search.fetch_euclid_object_summary(OBJECT_ID)
                self.assertEqual(result["object_summary"]["object_id"], int(OBJECT_ID))
                self.assertEqual(result["cutout_path"], "/original.jpg")
                self.assertEqual(result["tile_index"], "102021984")
                self.assertEqual(result["mosaic_summary"], {})
                self.assertTrue(result["warnings"])

    def test_healthy_archive_keeps_complete_metadata(self):
        objects = Table({
            "object_id": [int(OBJECT_ID)],
            "right_ascension": [60.879449], "declination": [-47.271543],
        })
        mosaics = Table({
            "file_name": ["vis.fits"], "file_path": ["/q1"],
            "tile_index": [102021984], "instrument_name": ["VIS"],
        })
        with patch.object(search, "query_archive", side_effect=[objects, mosaics]) as query:
            with patch.object(search, "morphology_cutout_path", return_value="/original.jpg"):
                with patch.object(search, "load_morphology_object") as load:
                    result = search.fetch_euclid_object_summary(OBJECT_ID)
        self.assertEqual(result["file_path"], "/q1/vis.fits")
        self.assertEqual(result["warnings"], [])
        self.assertEqual(result["tile_index"], "102021984")
        self.assertIn("FROM q1.mer_catalogue", query.call_args_list[0].args[0])
        self.assertIn("FROM q1.mosaic_product", query.call_args_list[1].args[0])
        load.assert_not_called()

    def test_fallback_rejects_missing_or_invalid_coordinates(self):
        for value in (np.nan, np.inf, 361., None):
            with self.subTest(value=value):
                data = MORPHOLOGY.copy()
                data["right_ascension"] = value
                with patch.object(search, "load_morphology_object", return_value=data):
                    self.assertIsNone(search.morphology_search_fallback(OBJECT_ID))

    def test_invalid_identifier_never_calls_archive(self):
        with patch.object(search, "query_archive") as query:
            with self.assertRaises(ValueError):
                search.fetch_euclid_object_summary("1 OR 1=1")
        query.assert_not_called()


if __name__ == "__main__":
    unittest.main()
