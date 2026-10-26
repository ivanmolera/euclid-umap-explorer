import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import pandas as pd

from euclid_umap_explorer import catalogs, components, images


class ClusterPreviewImageTests(unittest.TestCase):
    def test_labelled_lens_thumbnail_uses_morphology_cutout(self):
        row = pd.Series({
            "id_str": "Q1_R1_102041656_-520953773300324216",
            "object_id": "-520953773300324216",
            "lens_id_str": "lens_catalogue_identifier",
        })
        expected = (
            "/cutouts/102041656/"
            "102041656_NEG520953773300324216_gz_arcsinh_vis_only.jpg"
        )

        with patch.object(images, "CUTOUT_BASE", "/cutouts"), patch.object(
            images, "image_path_exists", return_value=True
        ) as exists, patch.object(images, "lens_image_path") as lens_image:
            self.assertEqual(images.object_image_path(row), expected)

        exists.assert_called_once_with(expected)
        lens_image.assert_not_called()

    def test_unavailable_lens_thumbnails_do_not_occupy_columns(self):
        rows = [pd.Series({"object_id": value}) for value in ("missing", "broken", "valid")]

        def image_path(row):
            return None if row["object_id"] == "missing" else row["object_id"]

        def thumbnail_src(path):
            if path == "broken":
                raise OSError("Could not load cutout")
            return "data:image/jpeg;base64,valid"

        with patch.object(components, "object_image_path", side_effect=image_path), patch.object(
            components, "thumbnail_image_src", side_effect=thumbnail_src
        ), patch.object(components, "render_thumbnail_group_title") as title, patch.object(
            components.st, "columns", return_value=[MagicMock()]
        ) as columns, patch.object(components, "show_thumbnail") as thumbnail:
            components.show_thumbnail_group(
                "Labelled lenses", rows, ["Missing", "Broken", "Valid"],
                skip_missing_images=True,
            )

        title.assert_called_once_with("Labelled lenses")
        columns.assert_called_once_with(1)
        thumbnail.assert_called_once()
        args, kwargs = thumbnail.call_args
        self.assertIs(args[0], rows[2])
        self.assertEqual(args[1], "Valid")
        self.assertEqual(kwargs["image_src"], "data:image/jpeg;base64,valid")

    def test_no_lens_images_hides_thumbnail_group(self):
        row = pd.Series({"object_id": "missing"})
        with patch.object(components, "object_image_path", return_value=None), patch.object(
            components, "render_thumbnail_group_title"
        ) as title, patch.object(components.st, "columns") as columns:
            components.show_thumbnail_group(
                "Labelled lenses", [row], ["Missing"], skip_missing_images=True
            )

        title.assert_not_called()
        columns.assert_not_called()

    def test_modal_offers_copy_for_its_object_id(self):
        row = pd.Series({
            "object_id": "-540908760267374976",
            "id_str": "Q1_R1_102046114_-540908760267374976",
        })
        with patch.object(components, "object_image_path", return_value="/cutout.jpg"), patch.object(
            components, "thumbnail_image_src", return_value="data:image/jpeg;base64,test"
        ), patch.object(components.st, "html") as rendered:
            components.show_thumbnail(row, "Grade B")

        rendered.assert_called_once()
        markup = rendered.call_args.args[0]
        self.assertIn('data-object-id="-540908760267374976"', markup)
        self.assertIn('aria-label="Copy object ID"', markup)
        self.assertIn('navigator.clipboard.writeText(button.dataset.objectId)', markup)
        self.assertTrue(rendered.call_args.kwargs["unsafe_allow_javascript"])

    def test_modal_shows_aladin_link_beside_copy_button(self):
        row = pd.Series({"object_id": "123", "id_str": "Q1_R1_123"})
        aladin_link = components.aladin_url(12.5, -4.0)
        with patch.object(components.st, "html") as rendered:
            components.show_thumbnail(
                row,
                "Canonical",
                image_src="data:image/jpeg;base64,test",
                aladin_link=aladin_link,
            )

        markup = rendered.call_args.args[0]
        self.assertIn('class="aladin-icon-link thumb-aladin-link"', markup)
        self.assertIn("View in Aladin", markup)
        self.assertIn('target="_blank" rel="noopener noreferrer"', markup)
        self.assertIn(aladin_link.replace("&", "&amp;"), markup)
        self.assertIn("background:#e5e7eb", markup)
        self.assertIn("filter: brightness(1.08)", markup)
        self.assertLess(markup.index('class="thumb-copy-id"'), markup.index('class="aladin-icon-link thumb-aladin-link"'))

    def test_canonical_and_anomalous_lens_modals_show_grade(self):
        for caption, grade in (
            ("Canonical", "B"),
            ("Anomalous", "A"),
            ("Anomalous", "C"),
        ):
            with self.subTest(caption=caption):
                row = pd.Series({
                    "object_id": "123",
                    "id_str": "Q1_R1_123",
                    "is_lens": True,
                    "lens_grade": grade,
                })
                with patch.object(components.st, "html") as rendered:
                    components.show_thumbnail(
                        row, caption, image_src="data:image/jpeg;base64,test"
                    )

                markup = rendered.call_args.args[0]
                self.assertIn(f"<strong>{caption} - Grade {grade}</strong>", markup)
                self.assertIn(f'<div class="thumb-caption">{caption}</div>', markup)

    def test_unlabelled_extreme_modal_keeps_original_title(self):
        row = pd.Series({
            "object_id": "123",
            "is_lens": False,
            "lens_grade": "A",
        })
        with patch.object(components.st, "html") as rendered:
            components.show_thumbnail(
                row, "Anomalous", image_src="data:image/jpeg;base64,test"
            )

        self.assertIn("<strong>Anomalous</strong>", rendered.call_args.args[0])

    def test_preview_aladin_links_use_catalogue_coordinates(self):
        morphology = pd.DataFrame({
            "object_id": ["123", "456", "789"],
            "right_ascension": [12.5, float("nan"), 30.0],
            "declination": [-4.0, 2.0, 100.0],
        })
        with patch.object(components, "load_morphology_objects", return_value=morphology) as load:
            links = components.cluster_preview_aladin_links(
                "/tmp/preview-aladin-test.parquet", ("123", "456", "789")
            )

        load.assert_called_once_with(
            "/tmp/preview-aladin-test.parquet",
            ("123", "456", "789"),
            columns=("right_ascension", "declination"),
        )
        self.assertEqual(links, {"123": components.aladin_url(12.5, -4.0)})

    def test_morphology_batch_read_selects_only_requested_columns(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "morphology.parquet"
            pd.DataFrame({
                "object_id": [-123, 456],
                "right_ascension": [12.5, 20.0],
                "declination": [-4.0, 5.0],
                "feat_pca_0": [1.0, 2.0],
            }).to_parquet(path)
            with patch.object(catalogs, "cached_input_path", return_value=str(path)):
                rows = catalogs.load_morphology_objects(
                    str(path), ["-123"], columns=("right_ascension", "declination")
                )

        self.assertEqual(len(rows), 1)
        self.assertEqual(set(rows.columns), {"object_id", "right_ascension", "declination"})
        self.assertEqual(rows.iloc[0]["object_id"], "-123")


if __name__ == "__main__":
    unittest.main()
