import base64
import unittest
from unittest.mock import patch

from PIL import Image
from streamlit.testing.v1 import AppTest

from euclid_umap_explorer import ui
from euclid_umap_explorer.config import (
    APP_TITLE,
    APP_VERSION,
    EUCLID_FAVICON_PATH,
    EUCLID_LOGO_PATH,
)


class SidebarBrandingTests(unittest.TestCase):
    def test_original_euclid_logo_is_available(self):
        self.assertTrue(EUCLID_LOGO_PATH.is_file())
        with Image.open(EUCLID_LOGO_PATH) as logo:
            self.assertEqual(logo.format, "PNG")

    def test_sidebar_restores_text_title_and_original_logo_preserving_favicon(self):
        app = AppTest.from_string(
            "from euclid_umap_explorer.ui import main\nmain()",
            default_timeout=30,
        )
        with patch.object(ui, "path_exists", return_value=True), patch.object(
            ui.st, "image", wraps=ui.st.image
        ) as render_image, patch.object(
            ui.st, "set_page_config", wraps=ui.st.set_page_config
        ) as page_config:
            app.run()

        self.assertEqual(len(app.exception), 0)
        render_image.assert_called_once_with(
            str(EUCLID_LOGO_PATH), use_container_width=True
        )
        self.assertEqual(len(app.sidebar.get("image")), 1)
        self.assertTrue(any(
            ">ESCOPE</div>" in element.value for element in app.sidebar.markdown
        ))
        self.assertIn(
            f"Version {APP_VERSION}", [element.value for element in app.sidebar.caption]
        )
        self.assertEqual(page_config.call_args.kwargs["page_title"], APP_TITLE)
        self.assertEqual(
            page_config.call_args.kwargs["page_icon"],
            "data:image/png;base64,"
            + base64.b64encode(EUCLID_FAVICON_PATH.read_bytes()).decode("ascii"),
        )


if __name__ == "__main__":
    unittest.main()
