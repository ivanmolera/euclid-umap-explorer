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

    def test_compact_controls_keep_accessible_labels_and_concise_help(self):
        app = AppTest.from_string("""
import pandas as pd
from euclid_umap_explorer.cluster_graph_ui import render_morphology_graph
render_morphology_graph(pd.DataFrame({'object_id': ['1'], 'cluster': [0]}), {}, {}, 'Enrichment')
""")
        with patch.object(cluster_graph_ui, "load_physical_measurements") as load:
            app.run()
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
            cutouts = app.checkbox(key="morphology_graph_cutouts")
            self.assertEqual(cutouts.label, "Show canonical cutouts")
            self.assertEqual(cutouts.help, "Same PCA representatives as the previews.")
            cutouts.check().run()
            self.assertTrue(app.checkbox(key="morphology_graph_cutouts").value)
            self.assertFalse(app.exception)
            load.assert_not_called()

    def test_loading_is_explicit_and_profiles_reused_across_controls(self):
        app = AppTest.from_string("""
import pandas as pd
import streamlit as st
from euclid_umap_explorer.cluster_graph_ui import render_morphology_graph
if 'fixture' not in st.session_state:
    st.session_state.fixture = pd.DataFrame({'object_id': ['1', '2', '3'], 'cluster': [0, 1, 2]})
summary = pd.DataFrame({'cluster': [0, 1, 2], 'n_objects': [1, 1, 1], 'n_lenses': [1, 0, 0], 'enrichment': [3., 0., 0.]})
st.session_state['graph_requested'] = render_morphology_graph(st.session_state.fixture, {
    'cluster_download_df': summary,
    'clusters': [{'cluster_id': i, 'canonical_row': None} for i in range(3)],
}, {}, 'Labelled-candidate count')
""")
        physical = pd.DataFrame({"object_id": ["1", "2", "3"], "concentration": [1., 2., 3.]})
        with patch.object(cluster_graph_ui, "source_signature", return_value={"generation": 1}), \
                patch.object(cluster_graph_ui, "load_physical_measurements", return_value=physical) as load:
            app.run()
            load.assert_not_called()
            app.toggle(key="show_morphology_graph").set_value(True).run()
            load.assert_not_called()
            app.multiselect(key="morphology_graph_fields").set_value(["concentration"]).run()
            next(b for b in app.button if b.label == "Build / refresh graph").click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(load.call_count, 1)
            app.number_input(key="morphology_graph_neighbors").set_value(1).run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(load.call_count, 1)
            next(b for b in app.button if b.label == "Build / refresh graph").click().run()
            self.assertEqual(load.call_count, 1)
            app.selectbox[0].set_value(2).run()
            next(b for b in app.button if b.label == "Analyse").click().run()
            self.assertEqual(app.session_state["graph_requested"], 2)
            app.session_state.fixture = pd.DataFrame({'object_id': ['1', '2'], 'cluster': [0, 1]})
            app.run()
            next(b for b in app.button if b.label == "Build / refresh graph").click().run()
            self.assertEqual(load.call_count, 2)

    def test_cache_invalidates_on_source_version_change(self):
        membership = pd.DataFrame({"object_id": ["1"], "cluster": [0]})
        physical = membership.assign(concentration=2.)
        with patch.object(cluster_graph_ui, "load_physical_measurements", return_value=physical) as load:
            for source in ('{"generation": 1}', '{"generation": 1}', '{"generation": 2}'):
                cluster_graph_ui.load_morphology_profiles("catalogue", source, membership_signature(membership), membership)
            self.assertEqual(load.call_count, 2)

    def test_catalogue_failure_leaves_app_usable(self):
        app = AppTest.from_string("""
import pandas as pd
from euclid_umap_explorer.cluster_graph_ui import render_morphology_graph
render_morphology_graph(pd.DataFrame({'object_id': ['1'], 'cluster': [0]}), {}, {}, 'Enrichment')
""")
        with patch.object(cluster_graph_ui, "source_signature", side_effect=TimeoutError), \
                patch.object(cluster_graph_ui, "load_physical_measurements") as load:
            app.run()
            app.toggle(key="show_morphology_graph").set_value(True).run()
            next(b for b in app.button if b.label == "Build / refresh graph").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.warning), 1)
            load.assert_not_called()
