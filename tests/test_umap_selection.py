import io
import json
import unittest
from contextlib import ExitStack
from html.parser import HTMLParser
from unittest.mock import Mock, patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from euclid_umap_explorer import ui, umap_3d, umap_selection_highlight


class HighlightPayloadParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reading_payload = False
        self.payload = None

    def handle_starttag(self, tag, attrs):
        self.reading_payload = tag == "script" and dict(attrs).get("id") == "umap-focus-data"

    def handle_endtag(self, tag):
        self.reading_payload = False

    def handle_data(self, data):
        if self.reading_payload:
            self.payload = json.loads(data)


class UmapSelectionTests(unittest.TestCase):
    def test_2d_highlight_uses_exact_object_and_coordinates_without_html_injection(self):
        row = pd.Series({"object_id": "</script><script>unsafe</script>", "umap_1": 1.25, "umap_2": -3.5})
        with patch.object(umap_selection_highlight.components, "html") as render:
            umap_selection_highlight.render_umap_2d_selection_highlight("chart_0", row, namespace="umap")
        parser = HighlightPayloadParser()
        parser.feed(render.call_args.args[0])
        self.assertEqual(parser.payload, {
            "chart_key": "chart_0", "namespace": "umap",
            "focus": {"x": 1.25, "y": -3.5, "object_id": row.object_id},
        })
        self.assertNotIn("</script><script>unsafe", render.call_args.args[0])
        self.assertEqual(render.call_args.kwargs["height"], 0)

    def test_empty_or_nonfinite_selection_removes_the_2d_highlight(self):
        for row in (None, pd.Series({"object_id": "1", "umap_1": float("nan"), "umap_2": 0.})):
            with self.subTest(row=row):
                with patch.object(umap_selection_highlight.components, "html") as render:
                    umap_selection_highlight.render_umap_2d_selection_highlight("chart_0", row, namespace="umap")
                parser = HighlightPayloadParser()
                parser.feed(render.call_args.args[0])
                self.assertIsNone(parser.payload["focus"])

    def test_clear_is_independent_and_preserves_computed_analysis(self):
        embedding = pd.DataFrame({"object_id": ["2700244936644435563"]})
        state = {
            "umap_embedding_df": embedding,
            "umap_signature": (380, 2),
            "semisupervised_umap_df": embedding.copy(),
            "semisupervised_umap_signature": (380, 1, 2),
            "hierarchical_subcluster_signature": (380, 3),
        }
        before = state.copy()
        with patch.object(ui.st, "session_state", state):
            main_key = ui.umap_selection_widget_key("umap_selection", (380, 2))
            semi_key = ui.umap_selection_widget_key("semisupervised_umap_chart", (380, 1, 2))
            ui.clear_umap_selection("umap_selection")
            self.assertNotEqual(ui.umap_selection_widget_key("umap_selection", (380, 2)), main_key)
            self.assertEqual(ui.umap_selection_widget_key("semisupervised_umap_chart", (380, 1, 2)), semi_key)
            ui.clear_umap_selection("semisupervised_umap_chart")
            self.assertNotEqual(ui.umap_selection_widget_key("semisupervised_umap_chart", (380, 1, 2)), semi_key)
        self.assertEqual(set(state) - set(before), {
            "umap_selection_generation", "semisupervised_umap_chart_generation",
        })
        for key, value in before.items():
            self.assertIs(state[key], value)
        self.assertEqual(state["umap_selection_generation"], 1)
        self.assertEqual(state["semisupervised_umap_chart_generation"], 1)

    def test_widget_identity_changes_with_projection_and_reset(self):
        with patch.object(ui.st, "session_state", {}):
            initial = ui.umap_selection_widget_key("umap_selection", (380, 2))
            self.assertEqual(initial, ui.umap_selection_widget_key("umap_selection", (380, 2)))
            self.assertNotEqual(initial, ui.umap_selection_widget_key("umap_selection", (380, 3)))
            self.assertNotEqual(initial, ui.umap_selection_widget_key("umap_selection", (356, 2)))
            ui.clear_umap_selection("umap_selection")
            self.assertNotEqual(initial, ui.umap_selection_widget_key("umap_selection", (380, 2)))

    def test_new_cluster_resets_selection_generations(self):
        state = {"umap_selection_generation": 4, "semisupervised_umap_chart_generation": 2}
        with patch.object(ui.st, "session_state", state):
            ui.clear_selected_cluster_analysis(clear_selection=False)
        self.assertNotIn("umap_selection_generation", state)
        self.assertNotIn("semisupervised_umap_chart_generation", state)

    def test_clear_and_reselect_update_downloads_and_details_in_all_maps(self):
        for mode in ("unsupervised", "semisupervised"):
            for dimensions in (2, 3):
                with self.subTest(mode=mode, dimensions=dimensions):
                    self.check_clear_and_reselect(mode, dimensions)

    def test_clearing_one_visible_map_does_not_clear_the_other(self):
        app = AppTest.from_string("""
import pandas as pd
from euclid_umap_explorer.ui import render_cluster_umap_interaction, render_semisupervised_umap_interaction
data = pd.DataFrame({'object_id': ['1', '2', '3'], 'umap_1': [1., 2., 3.], 'umap_2': [3., 1., 2.]})
render_cluster_umap_interaction(None, data, [], 380, pd.DataFrame())
render_semisupervised_umap_interaction(None, data, [], ('semi',), 1, pd.DataFrame())
""", default_timeout=30)
        app.session_state["umap_signature"] = ("main",)
        keys = []

        def fake_plot(_figure, *, key, **kwargs):
            keys.append(key)
            return ui.st.session_state.get(key, {"selection": {"points": []}})

        with ExitStack() as stack:
            stack.enter_context(patch.object(ui.st, "plotly_chart", side_effect=fake_plot))
            stack.enter_context(patch("euclid_umap_explorer.downloads.load_morphology_objects", return_value=pd.DataFrame()))
            details = stack.enter_context(patch.object(ui, "show_object_details"))
            stack.enter_context(patch.object(ui, "show_selected_object_tables"))
            stack.enter_context(patch.object(ui, "render_umap_2d_selection_highlight"))
            app.run()
            self.assertFalse(app.exception)
            main_key, semi_key = keys[-2:]
            app.session_state[main_key] = {"selection": {"points": [{"customdata": [0]}]}}
            app.session_state[semi_key] = {"selection": {"points": [{"customdata": [2]}]}}
            app.run()
            self.assertFalse(app.exception)
            self.assertFalse(app.button(key="clear_umap_selection").disabled)
            self.assertFalse(app.button(key="clear_semisupervised_umap_selection").disabled)
            details.reset_mock()
            app.button(key="clear_umap_selection").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(app.button(key="clear_umap_selection").disabled)
            self.assertFalse(app.button(key="clear_semisupervised_umap_selection").disabled)
            self.assertEqual(keys[-1], semi_key)
            details.assert_called_once()
            self.assertEqual(details.call_args.kwargs["key"], "semisupervised_umap")
            self.assertEqual(details.call_args.args[0]["object_id"], "3")

    def check_clear_and_reselect(self, mode, dimensions):
        signature = (380, dimensions)
        data = pd.DataFrame({
            "object_id": ["-608794490472715426", "2700244936644435563", "-625302350493849139"],
            "id_str": ["Q1_R1_1_a", "Q1_R1_1_b", "Q1_R1_1_c"],
            "point_index": [10, 20, 30], "feat_pca_0": [1., 2., 3.],
            "umap_1": [1., 2., 3.], "umap_2": [3., 1., 2.],
            "umap_point_label": ["Grade A", "Unknown", "Unknown"],
            "semi_supervised_label": ["Grade A", "Unknown", "Unknown"],
        }, index=[10, 20, 30])
        if dimensions == 3:
            data["umap_3"] = [2., 3., 1.]
        app = AppTest.from_string(f"""
import pandas as pd
import streamlit as st
from euclid_umap_explorer.ui import render_cluster_umap_interaction, render_semisupervised_umap_interaction
data = st.session_state['fixture_data']
if {mode!r} == 'unsupervised':
    render_cluster_umap_interaction(None, data, ['feat_pca_0'], 380, pd.DataFrame())
else:
    render_semisupervised_umap_interaction(None, data, ['feat_pca_0'], {signature!r}, 1, pd.DataFrame())
""", default_timeout=30)
        for key, value in {
            "fixture_data": data, "umap_embedding_df": data,
            "semisupervised_umap_df": data.copy(),
            "umap_signature": signature, "semisupervised_umap_signature": signature,
        }.items():
            app.session_state[key] = value
        widget_keys = []

        def fake_plot(_figure, *, key, **kwargs):
            widget_keys.append(key)
            return ui.st.session_state.get(key, {"selection": {"points": []}})

        def fake_component(*, key, default, **kwargs):
            widget_keys.append(key)
            return ui.st.session_state.get(key, default)

        component = Mock(side_effect=fake_component)
        with ExitStack() as stack:
            stack.enter_context(patch.object(ui.st, "plotly_chart", side_effect=fake_plot))
            stack.enter_context(patch.object(umap_3d, "scatter_3d_component", return_value=component))
            stack.enter_context(patch("euclid_umap_explorer.downloads.load_morphology_objects", return_value=pd.DataFrame()))
            details = stack.enter_context(patch.object(ui, "show_object_details"))
            tables = stack.enter_context(patch.object(ui, "show_selected_object_tables"))
            highlight = stack.enter_context(patch.object(ui, "render_umap_2d_selection_highlight"))
            download = stack.enter_context(patch.object(ui.st, "download_button", wraps=ui.st.download_button))
            fit = stack.enter_context(patch.object(ui, "compute_umap_embedding"))
            semi_fit = stack.enter_context(patch.object(ui, "compute_semisupervised_umap_embedding"))
            app.run()
            self.assertFalse(app.exception)
            clear_key = "clear_umap_selection" if mode == "unsupervised" else "clear_semisupervised_umap_selection"
            self.assertTrue(app.button(key=clear_key).disabled)
            selected_key = widget_keys[-1]
            app.session_state[selected_key] = {"selection": {"points": [
                {"customdata": [10]}, {"customdata": [30]},
            ]}}
            app.run()
            self.assertFalse(app.exception)
            self.assertFalse(app.button(key=clear_key).disabled)
            self.assertEqual(self.download_ids(download), data.loc[[10, 30], "object_id"].tolist())
            self.assertEqual(details.call_args.args[0]["object_id"], data.loc[10, "object_id"])
            if dimensions == 2:
                pd.testing.assert_series_equal(highlight.call_args.args[1], details.call_args.args[0])
            camera_data = component.call_args.kwargs["data"] if dimensions == 3 else None
            details.reset_mock()
            tables.reset_mock()
            app.button(key=clear_key).click().run()
            self.assertFalse(app.exception)
            self.assertTrue(app.button(key=clear_key).disabled)
            self.assertNotEqual(widget_keys[-1], selected_key)
            self.assertEqual(self.download_ids(download), data["object_id"].tolist())
            details.assert_not_called()
            tables.assert_not_called()
            if dimensions == 2:
                self.assertIsNone(highlight.call_args.args[1])
            self.assertTrue(any("Downloads 3 of 3" in caption.value for caption in app.caption))
            if dimensions == 3:
                after_clear = component.call_args.kwargs["data"]
                self.assertEqual(after_clear["selection"], {"points": []})
                self.assertEqual(after_clear["viewKey"], camera_data["viewKey"])
                self.assertEqual(after_clear["signature"], camera_data["signature"])
            app.session_state[widget_keys[-1]] = {"selection": {"points": [{"customdata": [20]}]}}
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(self.download_ids(download), [data.loc[20, "object_id"]])
            self.assertEqual(details.call_args.args[0]["object_id"], data.loc[20, "object_id"])
            if dimensions == 2:
                pd.testing.assert_series_equal(highlight.call_args.args[1], details.call_args.args[0])
            else:
                highlight.assert_not_called()
            pd.testing.assert_frame_equal(app.session_state["umap_embedding_df"], data)
            pd.testing.assert_frame_equal(app.session_state["semisupervised_umap_df"], data)
            fit.assert_not_called()
            semi_fit.assert_not_called()

    @staticmethod
    def download_ids(download):
        return pd.read_csv(io.BytesIO(download.call_args.kwargs["data"]), dtype={"object_id": str})["object_id"].tolist()


if __name__ == "__main__":
    unittest.main()
