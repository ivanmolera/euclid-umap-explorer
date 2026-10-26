import io
import json
import unittest
import zipfile
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from streamlit.testing.v1 import AppTest

from euclid_umap_explorer import cluster_graph_ui
from euclid_umap_explorer.cluster_graph import (
    build_cluster_profiles, build_morphology_graph, build_morphology_graph_figure,
    graph_download_bytes, membership_signature, selected_graph_cluster,
)


GRAPH_UI_APP = """
import pandas as pd
import streamlit as st
from euclid_umap_explorer.cluster_graph_ui import render_morphology_graph
if 'fixture' not in st.session_state:
    st.session_state.fixture = pd.DataFrame({'object_id': ['1', '2', '3', '4'], 'cluster': [0, 1, 2, 3]})
visible_ids = st.session_state.get('visible_ids', [0, 1, 2])
summary = pd.DataFrame({'cluster': [0, 1, 2, 3], 'n_objects': [1, 1, 1, 1],
                        'n_lenses': [1, 0, 0, 1], 'enrichment': [2., 0., 0., 2.]})
st.session_state['graph_requested'] = render_morphology_graph(st.session_state.fixture, {
    'cluster_download_df': summary,
    'clusters': [{'cluster_id': i, 'canonical_row': pd.Series({'object_id': str(i + 1)})
                  if st.session_state.get('canonical_rows') else None} for i in visible_ids],
})
"""


def graph_ui_measurements():
    return pd.DataFrame({
        "object_id": ["1", "2", "3", "4"],
        **{field: np.arange(1., 5.) / (index + 1)
           for index, field in enumerate(cluster_graph_ui.PHYSICAL_ANALYSIS_FIELDS)},
    })


def graph_fixture():
    profiles = pd.DataFrame({
        "cluster": [0, 1, 2, 3], "n_profile_objects": 10,
        "concentration_median": [0., 2., 4., 100.], "concentration_coverage": 1.,
        "asymmetry_median": [0., 3., 1., 20.], "asymmetry_coverage": 1.,
    })
    summary = pd.DataFrame({
        "cluster": [0, 1, 2, 3], "n_objects": [100, 40, 200, 30],
        "n_lenses": [10, 20, 15, 3], "enrichment": [1., 5., 2., 1.],
    })
    return profiles, summary


class ClusterGraphTests(unittest.TestCase):
    def model(self, profiles=None, summary=None, **kwargs):
        default_profiles, default_summary = graph_fixture()
        return build_morphology_graph(
            default_profiles if profiles is None else profiles,
            default_summary if summary is None else summary,
            ["concentration", "asymmetry"], **kwargs,
        )

    def test_all_members_and_flagged_values_in_coverage_denominator(self):
        membership = pd.DataFrame({"object_id": ["1", "2", "3", "4"], "cluster": [0, 0, 0, 1]})
        measurements = pd.DataFrame({
            "object_id": [1, 2, 4], "concentration": [1., 9., np.inf],
            "sersic_sersic_vis_index": [2., 8., 3.], "sersic_visnir_flags": [0, 1, 0],
        })
        result = build_cluster_profiles(membership, measurements).set_index("cluster")
        self.assertEqual(result.loc[0, "concentration_median"], 5.)
        self.assertEqual(result.loc[0, "concentration_q25"], 3.)
        self.assertEqual(result.loc[0, "concentration_coverage"], 2 / 3)
        self.assertEqual(result.loc[0, "sersic_sersic_vis_index_median"], 2.)
        self.assertEqual(result.loc[0, "sersic_sersic_vis_index_coverage"], 1 / 3)
        self.assertEqual(result.loc[1, "concentration_coverage"], 0.)

    def test_empty_measurements_and_duplicate_members(self):
        membership = pd.DataFrame({"object_id": ["1", "2"], "cluster": [0, 1]})
        profiles = build_cluster_profiles(membership, pd.DataFrame())
        self.assertEqual(profiles.n_profile_objects.tolist(), [1, 1])
        self.assertEqual(self.model(profiles=profiles)["meta"]["included_clusters"], 0)
        with self.assertRaises(ValueError):
            build_cluster_profiles(pd.concat([membership, membership]), pd.DataFrame())

    def test_membership_signature_checks_every_id_and_cluster(self):
        first = pd.DataFrame({"object_id": ["1", "2"], "cluster": [0, 0]})
        changed = first.assign(cluster=[0, 1])
        self.assertNotEqual(membership_signature(first), membership_signature(changed))
        self.assertEqual(membership_signature(first), membership_signature(first.assign(is_lens=True)))

    def test_scaling_uses_all_profiles_not_just_visible_previews(self):
        profiles, summary = graph_fixture()
        first = self.model(profiles, summary.iloc[:3], neighbors=2)
        second = self.model(profiles, summary.iloc[1:], neighbors=2)
        self.assertEqual(first["meta"]["scaler_mean"], second["meta"]["scaler_mean"])
        self.assertEqual(first["meta"]["reference_clusters"], 4)
        self.assertEqual(first["meta"]["scaler_mean"][0], 26.5)
        edge1 = first["edges"].query("source == 1 and target == 2").profile_distance.iloc[0]
        edge2 = second["edges"].query("source == 1 and target == 2").profile_distance.iloc[0]
        self.assertAlmostEqual(edge1, edge2)
        self.assertEqual(set(first["nodes"].cluster), {0, 1, 2})

    def test_labels_and_visible_order_never_change_geometry(self):
        profiles, summary = graph_fixture()
        first = self.model(profiles, summary)
        altered = summary.assign(n_lenses=0, enrichment=999., n_objects=500).iloc[::-1]
        second = self.model(profiles, altered)
        assert_frame_equal(first["edges"], second["edges"])
        self.assertEqual(first["positions"], second["positions"])
        self.assertFalse(first["meta"]["labels_used_in_distances"])

    def test_unit_rescaling_does_not_change_distances(self):
        profiles, summary = graph_fixture()
        first = self.model(profiles, summary)
        profiles["concentration_median"] *= 1000
        second = self.model(profiles, summary)
        np.testing.assert_allclose(first["edges"].profile_distance, second["edges"].profile_distance)

    def test_missing_coverage_is_not_imputed(self):
        profiles, summary = graph_fixture()
        profiles.loc[1, "concentration_coverage"] = .79
        profiles.loc[2, "asymmetry_median"] = np.nan
        model = self.model(profiles, summary)
        self.assertEqual(model["nodes"].cluster.tolist(), [0, 3])
        self.assertEqual(model["meta"]["reference_clusters"], 2)
        self.assertEqual(model["diagnostics"].included.tolist(), [True, False, False, True])
        self.assertIn("Concentration", model["diagnostics"].iloc[1].missing_or_low_coverage)

    def test_neighbors_are_deterministic_without_self_loops(self):
        profiles, summary = graph_fixture()
        profiles.loc[:, "concentration_median"] = [0, 0, 2, 100]
        profiles.loc[:, "asymmetry_median"] = 1.
        first = self.model(profiles, summary, neighbors=1)
        second = self.model(profiles.iloc[::-1], summary.iloc[::-1], neighbors=1)
        assert_frame_equal(first["edges"], second["edges"])
        self.assertEqual(first["positions"], second["positions"])
        self.assertTrue((first["edges"].source < first["edges"].target).all())
        self.assertEqual(first["meta"]["used_fields"], ["concentration"])

    def test_constant_and_single_visible_profiles_have_explanatory_state(self):
        profiles, summary = graph_fixture()
        self.assertEqual(self.model(profiles, summary.iloc[:1])["meta"]["status"], "insufficient_visible_profiles")
        profiles[["concentration_median", "asymmetry_median"]] = 1.
        self.assertEqual(self.model(profiles, summary)["meta"]["status"], "insufficient_variable_profiles")

    def test_outcomes_cannot_be_used_as_properties(self):
        profiles, summary = graph_fixture()
        with self.assertRaises(ValueError):
            build_morphology_graph(profiles, summary, ["enrichment"])

    def test_figure_selection_only_opens_node_ids(self):
        model = self.model()
        figure = build_morphology_graph_figure(model)
        self.assertEqual([data[0] for data in figure.data[2].customdata], [0, 1, 2, 3])
        self.assertEqual(figure.layout.height, 550)
        event = {"selection": {"points": [{"curve_number": 2, "customdata": [2]}]}}
        self.assertEqual(selected_graph_cluster(event, [0, 1, 2, 3]), 2)
        self.assertIsNone(selected_graph_cluster(event, [0, 1]))
        event["selection"]["points"][0]["curve_number"] = 1
        self.assertIsNone(selected_graph_cluster(event, [0, 1, 2, 3]))

    def test_cutout_borders_match_enrichment_scale_and_hover_omits_coverage(self):
        from plotly.colors import sample_colorscale

        model = self.model()
        figure = build_morphology_graph_figure(model, {0: "first-cutout", 1: "second-cutout", 99: "ignored"})
        self.assertEqual(len(figure.data), 3)
        self.assertEqual(len(figure.layout.images), 2)
        self.assertEqual(len(figure.layout.shapes), 2)
        marker = figure.data[2].marker
        for cluster, shape, thumbnail in zip((0, 1), figure.layout.shapes, figure.layout.images):
            enrichment = model["nodes"].set_index("cluster").loc[cluster, "enrichment"]
            colour = sample_colorscale("Viridis", [(enrichment - marker.cmin) / (marker.cmax - marker.cmin)])[0]
            self.assertEqual(shape.line.color, colour)
            self.assertEqual(shape.line.width, 3)
            self.assertEqual(shape.layer, "above")
            self.assertAlmostEqual(shape.x1 - shape.x0, thumbnail.sizex)
            self.assertAlmostEqual(shape.y1 - shape.y0, thumbnail.sizey)
            self.assertAlmostEqual((shape.x0 + shape.x1) / 2, thumbnail.x)
            self.assertAlmostEqual((shape.y0 + shape.y1) / 2, thumbnail.y)
        self.assertTrue(all("valid" not in text and "Enrichment:" in text for text in figure.data[2].hovertext))
        self.assertEqual(len(build_morphology_graph_figure(model).layout.shapes), 0)
        self.assertIn("concentration_coverage", model["nodes"])

    def test_equal_enrichment_still_has_consistent_cutout_colours(self):
        profiles, summary = graph_fixture()
        model = self.model(profiles, summary.assign(enrichment=5.))
        figure = build_morphology_graph_figure(model, {0: "first", 1: "second"})
        self.assertEqual(figure.layout.shapes[0].line.color, figure.layout.shapes[1].line.color)
        self.assertEqual((figure.data[2].marker.cmin + figure.data[2].marker.cmax) / 2, 5.)

    def test_labels_follow_cutout_edges_when_zooming_and_keep_missing_image_fallback(self):
        model = self.model()
        figure = build_morphology_graph_figure(model, {0: "first-cutout", 99: "ignored"})
        self.assertEqual(figure.data[2].mode, "markers")
        self.assertEqual([label.text for label in figure.layout.annotations], ["0", "1", "2", "3"])
        cutout = figure.layout.images[0]
        label = figure.layout.annotations[0]
        self.assertEqual((label.xref, label.yref), ("x", "y"))
        self.assertEqual(label.x, cutout.x)
        self.assertAlmostEqual(label.y, cutout.y - cutout.sizey / 2)
        self.assertEqual((label.yanchor, label.yshift), ("top", -5))
        self.assertFalse(label.captureevents)
        for index, fallback in enumerate(figure.layout.annotations[1:], start=1):
            self.assertEqual(fallback.y, model["positions"][index][1])
            self.assertAlmostEqual(fallback.yshift, -(figure.data[2].marker.size[index] / 2 + 5))
        for zoom in (1, 2, 4, 8):
            figure.update_yaxes(range=[cutout.y - 1 / zoom, cutout.y + 1 / zoom])
            self.assertAlmostEqual(label.y, cutout.y - cutout.sizey / 2)
            self.assertEqual(label.yshift, -5)

    def test_export_includes_full_reference_profiles_layout_and_provenance(self):
        model = self.model()
        with zipfile.ZipFile(io.BytesIO(graph_download_bytes(model, {"fit": "5.5/42"}))) as archive:
            self.assertEqual(set(archive.namelist()), {
                "nodes.csv", "edges.csv", "diagnostics.csv", "reference_profiles.csv",
                "layout.json", "manifest.json",
            })
            manifest = json.loads(archive.read("manifest.json"))
            self.assertEqual(manifest["provenance"]["fit"], "5.5/42")
            self.assertEqual(manifest["min_coverage"], .8)
            self.assertEqual(len(pd.read_csv(archive.open("nodes.csv"))), 4)


class GraphUITests(unittest.TestCase):
    def setUp(self):
        cluster_graph_ui.load_morphology_profiles.clear()

    def test_compact_controls_keep_accessible_labels_without_cutout_checkbox(self):
        app = AppTest.from_string(GRAPH_UI_APP)
        with patch.object(cluster_graph_ui, "source_signature", return_value={"generation": 1}), \
                patch.object(cluster_graph_ui, "load_physical_measurements", return_value=graph_ui_measurements()) as load:
            app.run()
            self.assertEqual(app.toggle(key="show_morphology_graph").label, "Show morphological graph")
            app.toggle(key="show_morphology_graph").set_value(True).run()
            self.assertFalse(app.exception)
            properties = app.multiselect(key="morphology_graph_fields")
            self.assertEqual(properties.label, "Profile properties")
            self.assertEqual(properties.value, list(cluster_graph_ui.MORPHOLOGY_GRAPH_FIELDS))
            self.assertEqual(properties.proto.label_visibility.value, 2)
            neighbors = app.number_input(key="morphology_graph_neighbors")
            self.assertEqual(neighbors.label, "Nearest profiles per cluster")
            self.assertEqual(neighbors.value, 2)
            self.assertEqual(neighbors.proto.label_visibility.value, 2)
            self.assertEqual(len(app.checkbox), 0)
            self.assertEqual(len(app.selectbox), 0)
            self.assertFalse(any(button.label == "Analyse" for button in app.button))
            self.assertEqual(len(app.get("download_button")), 0)
            self.assertFalse(app.exception)
            self.assertEqual(load.call_count, 1)

    def test_initial_build_is_automatic_and_parameter_edits_wait_for_refresh(self):
        app = AppTest.from_string(GRAPH_UI_APP)
        with patch.object(cluster_graph_ui, "source_signature", return_value={"generation": 1}), \
                patch.object(cluster_graph_ui, "load_physical_measurements", return_value=graph_ui_measurements()) as load, \
                patch.object(cluster_graph_ui, "build_morphology_graph", wraps=build_morphology_graph) as build:
            app.run()
            load.assert_not_called()
            app.toggle(key="show_morphology_graph").set_value(True).run()
            self.assertFalse(app.exception)
            self.assertEqual(load.call_count, 1)
            self.assertEqual(build.call_count, 1)
            original = app.session_state["morphology_profile_state"]["model"]
            self.assertEqual(original["meta"]["fields"], list(cluster_graph_ui.MORPHOLOGY_GRAPH_FIELDS))
            self.assertEqual(original["meta"]["neighbors"], 2)
            app.multiselect(key="morphology_graph_fields").set_value(["concentration"]).run()
            app.number_input(key="morphology_graph_neighbors").set_value(1).run()
            app.run()
            self.assertEqual(build.call_count, 1)
            self.assertIs(app.session_state["morphology_profile_state"]["model"], original)
            next(b for b in app.button if b.label == "Build / refresh graph").click().run()
            self.assertFalse(app.exception)
            applied = app.session_state["morphology_profile_state"]["model"]
            self.assertEqual(applied["meta"]["fields"], ["concentration"])
            self.assertEqual(applied["meta"]["neighbors"], 1)
            self.assertEqual(build.call_count, 2)
            self.assertEqual(load.call_count, 1)
            app.toggle(key="show_morphology_graph").set_value(False).run()
            app.toggle(key="show_morphology_graph").set_value(True).run()
            self.assertEqual(app.multiselect(key="morphology_graph_fields").value, ["concentration"])
            self.assertEqual(app.number_input(key="morphology_graph_neighbors").value, 1)
            self.assertEqual(build.call_count, 2)
            app.session_state.fixture = pd.DataFrame({'object_id': ['1', '2'], 'cluster': [0, 1]})
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(load.call_count, 2)
            self.assertEqual(build.call_count, 3)
            self.assertEqual(app.multiselect(key="morphology_graph_fields").value, list(cluster_graph_ui.MORPHOLOGY_GRAPH_FIELDS))
            self.assertEqual(app.number_input(key="morphology_graph_neighbors").value, 2)

    def test_node_click_opens_analysis_without_selector_and_deduplicates_reruns(self):
        app = AppTest.from_string(GRAPH_UI_APP)
        with patch.object(cluster_graph_ui, "source_signature", return_value={"generation": 1}), \
                patch.object(cluster_graph_ui, "load_physical_measurements", return_value=graph_ui_measurements()), \
                patch.object(cluster_graph_ui.st, "plotly_chart", return_value={}) as chart:
            app.run()
            app.toggle(key="show_morphology_graph").set_value(True).run()
            self.assertIsNone(app.session_state["graph_requested"])
            chart.return_value = {"selection": {"points": [{"curve_number": 2, "customdata": [2]}]}}
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["graph_requested"], 2)
            self.assertEqual(len(app.selectbox), 0)
            self.assertEqual(len(app.get("download_button")), 0)
            app.run()
            self.assertIsNone(app.session_state["graph_requested"])
            chart.return_value = {"selection": {"points": [{"curve_number": 2, "customdata": [1]}]}}
            app.run()
            self.assertEqual(app.session_state["graph_requested"], 1)

    def test_canonical_cutouts_are_automatic_and_missing_images_keep_the_node(self):
        app = AppTest.from_string(GRAPH_UI_APP)
        app.session_state["canonical_rows"] = True
        with patch.object(cluster_graph_ui, "source_signature", return_value={"generation": 1}), \
                patch.object(cluster_graph_ui, "load_physical_measurements", return_value=graph_ui_measurements()), \
                patch.object(cluster_graph_ui, "object_image_path", side_effect=["first.jpg", OSError(), None]), \
                patch.object(cluster_graph_ui, "thumbnail_image_src", return_value="first-image") as thumbnail, \
                patch.object(cluster_graph_ui, "build_morphology_graph_figure", wraps=build_morphology_graph_figure) as figure:
            app.run()
            app.toggle(key="show_morphology_graph").set_value(True).run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.checkbox), 0)
            thumbnail.assert_called_once_with("first.jpg")
            self.assertEqual(figure.call_args.args[1], {0: "first-image"})
            self.assertEqual(figure.call_args.args[0]["nodes"].cluster.tolist(), [0, 1, 2])

    def test_reprioritized_previews_use_applied_parameters_without_reloading(self):
        app = AppTest.from_string(GRAPH_UI_APP)
        with patch.object(cluster_graph_ui, "source_signature", return_value={"generation": 1}), \
                patch.object(cluster_graph_ui, "load_physical_measurements", return_value=graph_ui_measurements()) as load:
            app.run()
            app.toggle(key="show_morphology_graph").set_value(True).run()
            app.multiselect(key="morphology_graph_fields").set_value(["concentration"]).run()
            app.number_input(key="morphology_graph_neighbors").set_value(1).run()
            app.session_state["visible_ids"] = [1, 2, 3]
            app.run()
            self.assertFalse(app.exception)
            model = app.session_state["morphology_profile_state"]["model"]
            self.assertEqual(model["nodes"].cluster.tolist(), [1, 2, 3])
            self.assertEqual(model["meta"]["fields"], list(cluster_graph_ui.MORPHOLOGY_GRAPH_FIELDS))
            self.assertEqual(model["meta"]["neighbors"], 2)
            self.assertEqual(load.call_count, 1)

    def test_empty_properties_and_failed_refresh_preserve_the_applied_graph(self):
        app = AppTest.from_string(GRAPH_UI_APP)
        with patch.object(cluster_graph_ui, "source_signature", return_value={"generation": 1}) as source, \
                patch.object(cluster_graph_ui, "load_physical_measurements", return_value=graph_ui_measurements()):
            app.run()
            app.toggle(key="show_morphology_graph").set_value(True).run()
            original = app.session_state["morphology_profile_state"]["model"]
            app.multiselect(key="morphology_graph_fields").set_value([]).run()
            next(b for b in app.button if b.label == "Build / refresh graph").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.info), 1)
            self.assertIs(app.session_state["morphology_profile_state"]["model"], original)
            self.assertEqual(source.call_count, 1)
            app.multiselect(key="morphology_graph_fields").set_value(["concentration"]).run()
            source.side_effect = TimeoutError
            next(b for b in app.button if b.label == "Build / refresh graph").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.warning), 1)
            self.assertIs(app.session_state["morphology_profile_state"]["model"], original)

    def test_cache_invalidates_on_source_version_change(self):
        membership = pd.DataFrame({"object_id": ["1"], "cluster": [0]})
        physical = membership.assign(concentration=2.)
        with patch.object(cluster_graph_ui, "load_physical_measurements", return_value=physical) as load:
            for source in ('{"generation": 1}', '{"generation": 1}', '{"generation": 2}'):
                cluster_graph_ui.load_morphology_profiles("catalogue", source, membership_signature(membership), membership)
            self.assertEqual(load.call_count, 2)

    def test_catalogue_failure_leaves_app_usable(self):
        app = AppTest.from_string(GRAPH_UI_APP)
        with patch.object(cluster_graph_ui, "source_signature", side_effect=TimeoutError) as source, \
                patch.object(cluster_graph_ui, "load_physical_measurements") as load:
            app.run()
            app.toggle(key="show_morphology_graph").set_value(True).run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.warning), 1)
            self.assertEqual(source.call_count, 1)
            app.number_input(key="morphology_graph_neighbors").set_value(1).run()
            self.assertEqual(source.call_count, 1)
            next(b for b in app.button if b.label == "Build / refresh graph").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.warning), 1)
            self.assertEqual(source.call_count, 2)
            load.assert_not_called()
