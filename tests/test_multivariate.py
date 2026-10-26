import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from streamlit.testing.v1 import AppTest

from euclid_umap_explorer import multivariate_ui
from euclid_umap_explorer.cluster_graph import build_cluster_profiles
from euclid_umap_explorer.multivariate import (
    build_parallel_figure, cluster_parallel_frame, complete_parallel_profiles,
    filter_parallel_profiles, normalize_parallel_ranges, object_parallel_frame,
    parallel_chart_sample, parallel_signature,
)
from euclid_umap_explorer.physical import PHYSICAL_ANALYSIS_FIELDS


def fixture():
    objects = pd.DataFrame({
        "object_id": ["2700244936644435563", "-625302350493849139", "3", "4"],
        "cluster": [0, 0, 1, 1], "is_lens": [True, False, True, False],
        "lens_grade": ["A", "A", "C", None], "point_role": ["Canonical", None, None, None],
    })
    measurements = pd.DataFrame({
        "object_id": objects.object_id,
        **{field: [1., 2., 3., 4.] for field in PHYSICAL_ANALYSIS_FIELDS},
        "phz_flags": [0, 1, 0, 0], "phys_param_flags": 0, "sersic_visnir_flags": 0,
    })
    return objects, measurements


UI_APP = """
import pandas as pd
import streamlit as st
from euclid_umap_explorer.multivariate_ui import render_cluster_multivariate, render_object_multivariate
from tests.test_multivariate import fixture
if 'fixture' not in st.session_state:
    st.session_state.fixture, st.session_state.measurements = fixture()
objects, measurements = st.session_state.fixture, st.session_state.measurements
summary = pd.DataFrame({'cluster': [0, 1], 'n_objects': [2, 2], 'n_lenses': [1, 1],
                        'lens_rate': [.5, .5], 'enrichment': [1., 1.], 'label_entropy_bits': [1., 1.]})
st.session_state.requested = render_cluster_multivariate(objects, {
    'cluster_download_df': summary, 'clusters': [{'cluster_id': 0}],
})
render_object_multivariate(objects, objects.iloc[:1], measurements)
"""


class MultivariateTests(unittest.TestCase):
    def test_object_identity_labels_and_quality_flags(self):
        objects, physical = fixture()
        frame = object_parallel_frame(objects, physical)
        self.assertEqual(frame.object_id.tolist(), objects.object_id.tolist())
        self.assertEqual(frame.label.tolist(), ["Grade A", "Unknown", "Grade C", "Unknown"])
        self.assertTrue(pd.isna(frame.phz_median.iloc[1]))
        self.assertEqual(frame.concentration.iloc[1], 2.)
        self.assertNotIn("point_role", frame)
        self.assertIn("point_role", objects)
        with self.assertRaises(ValueError):
            object_parallel_frame(pd.concat([objects, objects]), physical)

    def test_missing_values_are_not_imputed_and_unselected_fields_do_not_exclude(self):
        objects, physical = fixture()
        physical.loc[2, "concentration"] = np.inf
        frame = object_parallel_frame(objects, physical.iloc[:3])
        complete = complete_parallel_profiles(frame, ["concentration", "asymmetry"], "label")
        self.assertEqual(complete.object_id.tolist(), objects.object_id.iloc[:2].tolist())
        self.assertEqual(len(complete_parallel_profiles(frame, ["concentration", "phz_median"], "label")), 1)
        self.assertEqual(len(complete_parallel_profiles(frame, ["concentration", "asymmetry"], "phz_median")), 1)

    def test_clusters_use_full_membership_medians_and_selected_field_coverage(self):
        objects, physical = fixture()
        profiles = build_cluster_profiles(objects, physical)
        summary = pd.DataFrame({"cluster": [1, 0], "n_objects": 2, "n_lenses": 1,
                                "lens_rate": .5, "enrichment": 2., "label_entropy_bits": 1.})
        frame = cluster_parallel_frame(profiles, summary)
        self.assertEqual(frame.cluster.tolist(), [1, 0])
        self.assertEqual(frame.concentration.tolist(), [3.5, 1.5])
        self.assertEqual(frame.density_percent.tolist(), [50., 50.])
        self.assertEqual(len(complete_parallel_profiles(frame, ["concentration", "asymmetry"], "enrichment", min_coverage=.8)), 2)
        self.assertEqual(len(complete_parallel_profiles(frame, ["concentration", "phz_median"], "enrichment", min_coverage=.8)), 1)
        frame.loc[:, "concentration_coverage"] = [.8, .79]
        self.assertEqual(complete_parallel_profiles(frame, ["concentration", "asymmetry"], "enrichment", min_coverage=.8).cluster.tolist(), [1])

    def test_invalid_properties_and_coverage_are_rejected(self):
        frame = object_parallel_frame(*fixture())
        for fields in ([], ["enrichment"], ["gini", "gini"]):
            with self.assertRaises(ValueError):
                complete_parallel_profiles(frame, fields, "label")
        with self.assertRaises(ValueError):
            complete_parallel_profiles(frame, ["gini", "asymmetry"], "label", min_coverage=0)

    def test_ranges_are_inclusive_with_or_within_axes_and_and_between_axes(self):
        frame = object_parallel_frame(*fixture())
        ranges = {"gini": [[1., 1.], [3., 4.]], "asymmetry": [4., 2.]}
        selected = filter_parallel_profiles(frame, ranges, ["gini", "asymmetry"])
        self.assertEqual(selected.object_id.tolist(), ["3", "4"])
        self.assertEqual(normalize_parallel_ranges({"gini": [True, 4], "asymmetry": [[1, np.inf]], "foo": [1, 4]}, ["gini", "asymmetry"]), {})
        assert_frame_equal(filter_parallel_profiles(frame, None, ["gini"]), frame)

    def test_signatures_reset_on_data_colour_axes_or_identifier_changes(self):
        frame = object_parallel_frame(*fixture())
        first = parallel_signature(frame, ["gini", "asymmetry"], "label", "object_id")
        self.assertNotEqual(first, parallel_signature(frame, ["asymmetry", "gini"], "label", "object_id"))
        self.assertNotEqual(first, parallel_signature(frame, ["gini", "asymmetry"], "gini", "object_id"))
        frame.loc[0, "gini"] = 99
        self.assertNotEqual(first, parallel_signature(frame, ["gini", "asymmetry"], "label", "object_id"))

    def test_figure_keeps_units_axis_order_and_brush_without_changing_profiles(self):
        frame = complete_parallel_profiles(object_parallel_frame(*fixture()), ["gini", "asymmetry"], "label")
        brush = {"order": ["asymmetry", "gini"], "ranges": {"gini": [2, 3]}}
        figure = build_parallel_figure(frame, ["gini", "asymmetry"], "label", brush)
        dimensions = figure["data"][0]["dimensions"]
        self.assertEqual([d["name"] for d in dimensions], ["asymmetry", "gini"])
        self.assertEqual(dimensions[1]["values"], [1., 2., 3., 4.])
        self.assertEqual(dimensions[1]["constraintrange"], [2., 3.])
        self.assertEqual(figure["data"][0]["line"]["colorbar"]["ticktext"], ["Unknown", "Grade C", "Grade A"])
        json.dumps(figure, allow_nan=False)
        self.assertEqual(frame.gini.tolist(), [1., 2., 3., 4.])

    def test_constant_dimensions_remain_visible_and_sample_keeps_full_axis_limits(self):
        frame = object_parallel_frame(*fixture())
        frame.loc[:, "gini"] = 5.
        figure = build_parallel_figure(frame.iloc[:2], ["gini", "asymmetry"], "label", reference=frame)
        self.assertEqual(figure["data"][0]["dimensions"][0]["range"], [4.5, 5.5])
        self.assertEqual(figure["data"][0]["dimensions"][1]["range"], [1., 4.])

    def test_large_chart_sampling_is_bounded_reproducible_and_keeps_labels(self):
        frame = pd.DataFrame({"object_id": [str(i) for i in range(20_010)], "is_lens": [True] * 10 + [False] * 20_000})
        first, second = parallel_chart_sample(frame), parallel_chart_sample(frame.iloc[::-1])
        self.assertEqual(len(first), 20_000)
        self.assertEqual(set(first.object_id), set(second.object_id))
        self.assertEqual(first.is_lens.sum(), 10)


class MultivariateUITests(unittest.TestCase):
    def setUp(self):
        multivariate_ui.load_morphology_profiles.clear()

    def test_disabled_views_do_not_load_profiles_or_render_component(self):
        app = AppTest.from_string(UI_APP)
        with patch.object(multivariate_ui, "source_signature") as source, \
                patch.object(multivariate_ui, "parallel_component") as component:
            app.run()
            self.assertFalse(app.exception)
            source.assert_not_called()
            component.assert_not_called()
            self.assertEqual([toggle.value for toggle in app.toggle], [False, False])

    def test_views_are_independent_reuse_profiles_and_keep_scope_local(self):
        app = AppTest.from_string(UI_APP)
        renderer = lambda **kwargs: {"brush": kwargs["data"]["brush"]}
        objects, physical = fixture()
        with patch.object(multivariate_ui, "source_signature", return_value={"generation": 1}), \
                patch.object(multivariate_ui, "load_morphology_profiles", return_value=build_cluster_profiles(objects, physical)) as load, \
                patch.object(multivariate_ui, "parallel_component", return_value=renderer):
            app.run()
            app.toggle(key="show_cluster_multivariate").set_value(True).run()
            self.assertFalse(app.exception)
            self.assertEqual(app.segmented_control(key="cluster_multivariate_scope").proto.label_visibility.value, 2)
            self.assertTrue(multivariate_ui.CLUSTER_HELP.startswith("Each line is a cluster's median profile. Each selected property"))
            self.assertNotIn("not an individual galaxy", multivariate_ui.CLUSTER_HELP)
            self.assertEqual(len(app.dataframe[0].value), 1)
            app.segmented_control(key="cluster_multivariate_scope").set_value("All clusters").run()
            self.assertEqual(len(app.dataframe[0].value), 2)
            self.assertEqual(load.call_count, 1)
            app.toggle(key="show_object_multivariate").set_value(True).run()
            self.assertFalse(app.exception)
            self.assertEqual(app.segmented_control(key="object_multivariate_scope").proto.label_visibility.value, 2)
            self.assertEqual(len(app.dataframe[1].value), 4)
            app.segmented_control(key="object_multivariate_scope").set_value("After active filters").run()
            self.assertFalse(app.exception)
            self.assertEqual(app.dataframe[1].value.object_id.tolist(), ["2700244936644435563"])
            self.assertEqual(len(app.dataframe[0].value), 2)

    def test_missing_fields_failed_load_and_empty_axis_selection_leave_app_usable(self):
        app = AppTest.from_string(UI_APP)
        with patch.object(multivariate_ui, "source_signature", side_effect=TimeoutError):
            app.run()
            app.toggle(key="show_cluster_multivariate").set_value(True).run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.warning), 1)
        with patch.object(multivariate_ui, "parallel_component", return_value=lambda **kwargs: {"brush": {}}):
            app.toggle(key="show_cluster_multivariate").set_value(False).run()
            app.toggle(key="show_object_multivariate").set_value(True).run()
            app.multiselect(key="object_multivariate_fields").set_value([]).run()
            self.assertFalse(app.exception)
            self.assertEqual(app.info[0].value, "Select at least two properties.")

    def test_table_click_is_explicit_and_same_cluster_can_be_reselected(self):
        app = AppTest.from_string(UI_APP)
        objects, physical = fixture()
        selection = SimpleNamespace(rows=[0])
        with patch.object(multivariate_ui, "source_signature", return_value={"generation": 1}), \
                patch.object(multivariate_ui, "load_morphology_profiles", return_value=build_cluster_profiles(objects, physical)), \
                patch.object(multivariate_ui, "parallel_component", return_value=lambda **kwargs: {"brush": {}}), \
                patch.object(multivariate_ui.st, "dataframe", return_value=SimpleNamespace(selection=selection)):
            app.run()
            app.toggle(key="show_cluster_multivariate").set_value(True).run()
            self.assertEqual(app.session_state.requested, 0)
            app.run()
            self.assertIsNone(app.session_state.requested)
            selection.rows = []
            app.run()
            selection.rows = [0]
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state.requested, 0)

    def test_object_table_opens_exact_identifier_without_numeric_rounding(self):
        app = AppTest.from_string(UI_APP)
        selection = SimpleNamespace(rows=[0])
        aladin = "https://aladin.cds.unistra.fr/AladinLite/?target=12.5%20-4.0"
        with patch.object(multivariate_ui, "parallel_component", return_value=lambda **kwargs: {"brush": {}}), \
                patch.object(multivariate_ui.st, "dataframe", return_value=SimpleNamespace(selection=selection)), \
                patch.object(multivariate_ui, "cluster_preview_aladin_links", return_value={"2700244936644435563": aladin}) as links, \
                patch.object(multivariate_ui, "show_thumbnail") as modal:
            app.run()
            app.toggle(key="show_object_multivariate").set_value(True).run()
            self.assertFalse(app.exception)
            self.assertEqual(modal.call_args.args[0].object_id, "2700244936644435563")
            self.assertEqual(modal.call_args.args[1], "Grade A")
            self.assertEqual(modal.call_args.kwargs, {"aladin_link": aladin, "open_modal": True})
            links.assert_called_once_with(multivariate_ui.MORPH_PATH, ("2700244936644435563",))
            self.assertFalse(any(heading.value == "Selected object" for heading in app.subheader))

            modal.reset_mock()
            app.run()
            modal.assert_not_called()
            selection.rows = [1]
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(modal.call_args.args[0].object_id, "-625302350493849139")
            self.assertEqual(modal.call_args.args[1], "Unknown")
            self.assertIsNone(modal.call_args.kwargs["aladin_link"])
            modal.reset_mock()
            selection.rows = []
            app.run()
            modal.assert_not_called()
            selection.rows = [1]
            app.run()
            self.assertFalse(app.exception)
            modal.assert_called_once()

    def test_graph_profile_reuse_and_new_fit_reset_chart_context(self):
        app = AppTest.from_string(UI_APP)
        objects, physical = fixture()
        rendered_keys = []
        def renderer(**kwargs):
            rendered_keys.append(kwargs["key"])
            return {"brush": kwargs["data"]["brush"]}
        with patch.object(multivariate_ui, "source_signature", return_value={"generation": 1}) as source, \
                patch.object(multivariate_ui, "load_morphology_profiles", return_value=build_cluster_profiles(objects, physical)) as load, \
                patch.object(multivariate_ui, "parallel_component", return_value=renderer):
            app.run()
            app.session_state["morphology_profile_state"] = {
                "data": app.session_state.fixture, "profiles": build_cluster_profiles(objects, physical),
            }
            app.toggle(key="show_cluster_multivariate").set_value(True).run()
            self.assertFalse(app.exception)
            source.assert_not_called()
            load.assert_not_called()
            first_key = rendered_keys[-1]
            app.session_state.fixture = app.session_state.fixture.copy()
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(load.call_count, 1)
            self.assertNotEqual(first_key, rendered_keys[-1])


if __name__ == "__main__":
    unittest.main()
