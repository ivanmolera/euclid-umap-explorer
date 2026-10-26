import unittest
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from euclid_umap_explorer import components, ui
from euclid_umap_explorer.subclustering import _compute_hierarchical_subclusters_impl


class AnalysisUITests(unittest.TestCase):
    def test_filters_reach_hierarchical_and_semisupervised_inputs(self):
        cluster = pd.DataFrame({
            "object_id": [str(index) for index in range(24)],
            "id_str": [f"Q1_R1_1_{index}" for index in range(24)],
            "cluster": 0,
            "feat_pca_0": np.arange(24, dtype=float),
            "is_lens": [index % 3 == 0 for index in range(24)],
            "lens_grade": ["A" if index % 3 == 0 else "" for index in range(24)],
        })
        physical = pd.DataFrame({
            "object_id": cluster["object_id"],
            "phz_median": np.arange(24) / 10,
            "phz_flags": [1 if index == 8 else 0 for index in range(24)],
        })

        def fake_umap(data, *_args, **_kwargs):
            return data.assign(umap_1=data["feat_pca_0"], umap_2=0.0)

        def fake_semi_umap(data, *_args, **_kwargs):
            return data.assign(
                semi_umap_1=data["feat_pca_0"],
                semi_umap_2=0.0,
                semi_supervised_target=np.where(data["is_lens"], 2, -1),
                semi_supervised_label=np.where(data["is_lens"], "Grade A", "Unknown"),
            )

        app = AppTest.from_string("from euclid_umap_explorer.ui import main\nmain()")
        for key, value in {
            "cluster_ready": True,
            "cluster_result": (cluster, ["feat_pca_0"], None),
            "cluster_params": {
                "lens_grades": ("A", "B", "C"), "threshold": 6.2,
                "branching_factor": 50, "batch_size": 25000,
            },
            "analysis_cluster_id": 0,
            "pca_filter_count": 1,
            "pca_filter_0_feature": "feat_pca_0",
            "pca_filter_0_operator": ">=",
            "pca_filter_0_value": 4.0,
            "physical_filter_fields_0": ["phz_median"],
            "physical_filter_0_phz_median_operator": "<=",
            "physical_filter_0_phz_median_value": 1.8,
        }.items():
            app.session_state[key] = value

        with ExitStack() as stack:
            stack.enter_context(patch.object(ui, "path_exists", return_value=True))
            stack.enter_context(patch.object(
                ui, "selected_cluster_physical_measurements", return_value=physical
            ))
            # Keep the orchestration real; isolate plots, remote images and numerical UMAP.
            for name in (
                "render_cluster_preview_card", "render_lens_grade_help_label",
                "render_help_label", "render_selected_cluster_pca_characterization",
                "render_selected_cluster_physical_characterization",
                "render_cluster_umap_interaction", "render_semisupervised_umap_interaction",
                "render_subcluster_physical_characterization", "render_dendrogram_preview_section",
            ):
                stack.enter_context(patch.object(ui, name))
            umap = stack.enter_context(patch.object(
                ui, "compute_umap_embedding", side_effect=fake_umap
            ))
            hierarchy = stack.enter_context(patch.object(
                ui, "compute_hierarchical_subclusters",
                side_effect=_compute_hierarchical_subclusters_impl,
            ))
            semi = stack.enter_context(patch.object(
                ui, "compute_semisupervised_umap_embedding", side_effect=fake_semi_umap
            ))

            def click(label):
                next(button for button in app.button if button.label == label).click().run()
                self.assertEqual(len(app.exception), 0)

            app.run()
            self.assertEqual(len(app.exception), 0)
            click("Compute UMAP")
            expected = {str(index) for index in range(4, 19) if index != 8}
            self.assertEqual(set(umap.call_args.args[0]["object_id"]), expected)
            click("Compute hierarchical clustering")
            self.assertEqual(set(hierarchy.call_args.args[0]["object_id"]), expected)
            click("Compute semi-supervised UMAP")
            semi_input = semi.call_args.args[0]
            self.assertGreaterEqual(len(semi_input), 3)
            self.assertTrue(set(semi_input["object_id"]).issubset(expected))
            self.assertEqual(semi_input["hierarchical_subcluster"].nunique(), 1)

            # Editing filters must hide stale downstream results until explicit recomputation.
            app.number_input(key="pca_filter_0_value").set_value(10.0).run()
            self.assertEqual(len(app.exception), 0)
            self.assertFalse(any(b.label == "Compute semi-supervised UMAP" for b in app.button))
            self.assertEqual(semi.call_count, 1)
            click("Recompute UMAP")
            click("Compute hierarchical clustering")
            self.assertEqual(
                set(hierarchy.call_args.args[0]["object_id"]),
                {str(index) for index in range(10, 19)},
            )

    def test_complete_physical_rows_are_cached_per_object_and_returned_as_copies(self):
        physical = pd.DataFrame({"object_id": ["-2"], "flux_detection_total": [123.0]})
        with patch.object(components.st, "session_state", {}), patch.object(
            components, "load_physical_measurement_object", return_value=physical
        ) as load:
            first = components.object_physical_measurements(-2)
            first.loc[0, "flux_detection_total"] = 0.0
            second = components.object_physical_measurements("-2")
            load.assert_called_once_with(components.PHYSICAL_MEASUREMENTS_PATH, "-2")
            self.assertEqual(second.loc[0, "flux_detection_total"], 123.0)

    def test_object_tables_have_equal_eight_row_height_and_no_expander(self):
        morphology = pd.DataFrame([{f"morph_{index}": index for index in range(12)}])
        physical = pd.DataFrame([{
            "object_id": "1", "phz_median": 0.5, "phz_flags": 0,
            **{f"extra_{index}": index for index in range(12)},
        }])
        app = AppTest.from_string("""
import pandas as pd
from euclid_umap_explorer.components import show_selected_object_tables
row = pd.Series({'object_id': '1', **{f'feat_pca_{i}': i for i in range(10)}})
show_selected_object_tables(row, [f'feat_pca_{i}' for i in range(10)])
""")
        with patch.object(components, "load_morphology_object", return_value=morphology), \
                patch.object(components, "load_physical_measurement_object", return_value=physical), \
                patch.object(components.st, "dataframe", wraps=components.st.dataframe) as render_table:
            app.run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.expander), 0)
        self.assertEqual(len(app.dataframe), 4)
        self.assertEqual(len(app.get("column")), 4)
        self.assertTrue(all(len(column.get("dataframe")) == 1 for column in app.get("column")))
        for call in render_table.call_args_list:
            self.assertEqual(call.kwargs["height"], components.OBJECT_DETAIL_TABLE_HEIGHT)
        self.assertNotIn("object_id", set(app.dataframe[0].value["field"]))
        self.assertEqual(list(app.dataframe[3].value.columns), ["field", "value"])
        self.assertIn("extra_11", set(app.dataframe[3].value["field"]))

    def test_both_umap_details_show_copyable_id_and_four_tables(self):
        for mode in ("unsupervised", "semisupervised"):
            with self.subTest(mode=mode):
                app = AppTest.from_string(f"""
import pandas as pd
from euclid_umap_explorer.ui import (
    render_cluster_umap_interaction, render_semisupervised_umap_interaction
)
data = pd.DataFrame([{{
    'object_id': '-625302350493849139', 'cluster': 506,
    'id_str': 'Q1_R1_102020063_-625302350493849139',
    'feat_pca_0': 1.25, 'is_lens': False, 'umap_1': 2.5, 'umap_2': 3.5,
}}])
if {mode!r} == 'unsupervised':
    render_cluster_umap_interaction(None, data, ['feat_pca_0'], 506, pd.DataFrame())
else:
    render_semisupervised_umap_interaction(None, data, ['feat_pca_0'], (), 1, pd.DataFrame())
""")
                with ExitStack() as stack:
                    columns = stack.enter_context(patch.object(
                        components.st, "columns", wraps=components.st.columns
                    ))
                    stack.enter_context(patch.object(ui.st, "plotly_chart", return_value={
                        "selection": {"points": [{"customdata": [0]}]},
                    }))
                    stack.enter_context(patch.object(ui, "umap_download_df", return_value=pd.DataFrame()))
                    stack.enter_context(patch.object(components, "selected_umap_cutout_path", return_value="/test/cutout.jpg"))
                    stack.enter_context(patch.object(components, "lens_image_path", return_value=None))
                    image = stack.enter_context(patch.object(components, "show_detectable_image"))
                    stack.enter_context(patch.object(components, "aladin_caption_for_object", return_value="Morphology cutout"))
                    stack.enter_context(patch.object(components, "load_morphology_object", return_value=pd.DataFrame([{"gini": 0.5}])))
                    stack.enter_context(patch.object(components, "load_physical_measurement_object", return_value=pd.DataFrame([{"phz_median": 0.3}])))
                    app.run()
                self.assertEqual(len(app.exception), 0)
                self.assertEqual(len(app.code), 1)
                self.assertEqual(app.code[0].value, "-625302350493849139")
                columns.assert_any_call([1, 3], gap="small", vertical_alignment="center")
                self.assertEqual(len(app.dataframe), 4)
                summary = app.dataframe[0].value.set_index("field")["value"]
                self.assertEqual(summary["cluster"], "506")
                self.assertEqual(summary["umap_1"], "2.5")
                image.assert_called_once()

    def test_object_search_aligns_mosaic_and_physical_in_a_separate_row(self):
        app = AppTest.from_string("""
from euclid_umap_explorer.components import render_euclid_object_search
render_euclid_object_search('-625302350493849139')
""")
        app.session_state["euclid_search_result"] = {
            "requested_object_id": "-625302350493849139",
            "result": {
                "object_id": "-625302350493849139",
                "object_summary": {
                    "object_id": "-625302350493849139",
                    "right_ascension": 100.0, "declination": 30.0,
                },
                "mosaic_summary": {"tile_index": 102020063, "instrument_name": "VIS"},
                "cutout_path": None,
            },
            "morphology_df": pd.DataFrame([{"object_id": "-625302350493849139", "gini": 0.5}]),
            "physical_df": pd.DataFrame([{"phz_median": 0.3, "flux_detection_total": 15.0}]),
            "lens_candidate_summary": {"is_lens_candidate": "Yes", "lens_candidate_grade": "A"},
        }
        with patch.object(components.st, "dataframe", wraps=components.st.dataframe) as render_table:
            app.run()
        self.assertEqual(len(app.exception), 0)
        data_columns = [column for column in app.get("column") if column.get("dataframe")]
        self.assertEqual(len(data_columns), 3)
        summary_titles = [element.value for element in data_columns[0].get("markdown")]
        left_titles = [element.value for element in data_columns[1].get("markdown")]
        right_titles = [element.value for element in data_columns[2].get("markdown")]
        self.assertEqual(summary_titles, ["**Object summary** (Lens candidate grade A)"])
        self.assertEqual(left_titles, ["**Mosaic summary**", "**Morphology catalogue features**"])
        self.assertEqual(right_titles, ["**Physical measurements**"])
        self.assertEqual([len(column.get("dataframe")) for column in data_columns], [1, 2, 1])
        self.assertEqual(render_table.call_args_list[2].kwargs["height"], components.OBJECT_DETAIL_TABLE_HEIGHT)
        self.assertEqual(render_table.call_args_list[3].kwargs["height"], "stretch")
        self.assertIn("flux_detection_total", set(app.dataframe[3].value["field"]))


if __name__ == "__main__":
    unittest.main()
