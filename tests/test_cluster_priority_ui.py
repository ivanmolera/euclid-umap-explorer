import unittest
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from euclid_umap_explorer import ui
from euclid_umap_explorer.analysis import prioritize_cluster_summary


class ClusterPriorityTests(unittest.TestCase):
    def test_both_orders_and_deterministic_tie_breaks(self):
        summary = pd.DataFrame({
            "cluster": [4, 3, 2, 1], "n_objects": [500, 500, 1000, 100],
            "n_lenses": [50, 50, 100, 30], "lens_rate": [.1, .1, .1, .3],
            "enrichment": [10, 10, 10, 30],
        })
        self.assertEqual(prioritize_cluster_summary(summary).cluster.tolist(), [2, 3, 4, 1])
        self.assertEqual(prioritize_cluster_summary(summary, "Enrichment").cluster.tolist(), [1, 2, 3, 4])
        self.assertEqual(len(summary), 4)

    def test_catalogue_presets_keep_independent_user_edits(self):
        state = {}
        with patch.object(ui.st, "session_state", state):
            ui.synchronize_birch_defaults(True)
            self.assertEqual((state["birch_threshold"], state["birch_branching_factor"]), (5.5, 42))
            state["birch_threshold"] = 5.7
            ui.synchronize_birch_defaults(True)
            self.assertEqual(state["birch_threshold"], 5.7)
            ui.synchronize_birch_defaults(False)
            self.assertEqual((state["birch_threshold"], state["birch_branching_factor"]), (6.2, 50))
            ui.synchronize_birch_defaults(True)
            self.assertEqual(state["birch_threshold"], 5.7)

    def test_widgets_show_filtered_and_complete_defaults(self):
        app = AppTest.from_string("from euclid_umap_explorer.ui import main\nmain()")
        with patch.object(ui, "path_exists", return_value=True):
            app.run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(app.number_input(key="birch_threshold").value, 5.5)
            self.assertEqual(app.number_input(key="birch_branching_factor").value, 42)
            app.toggle(key="straight_line_artifacts_filter_enabled").set_value(False).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.number_input(key="birch_threshold").value, 6.2)
        self.assertEqual(app.number_input(key="birch_branching_factor").value, 50)

    def test_summary_keeps_initial_priority_without_selector_or_reclustering(self):
        catalogue = pd.DataFrame([
            {"object_id": str(cluster * 1000 + i), "cluster": cluster,
             "is_lens": i < 20 - cluster // 2, "lens_grade": "A" if i < 20 - cluster // 2 else "",
             "feat_pca_0": float(i), "id_str": f"Q1_R1_1_{cluster * 1000 + i}"}
            for cluster in range(22) for i in range(100 - 4 * cluster)
        ])
        app = AppTest.from_string("from euclid_umap_explorer.ui import main\nmain()")
        for key, value in {
            "cluster_ready": True, "cluster_result": (catalogue, ["feat_pca_0"], None),
            "cluster_params": {"lens_grades": ("A", "B", "C"), "threshold": 5.5,
                               "branching_factor": 42, "batch_size": 25000},
            "cluster_summary_expanded": True,
            "cluster_summary_priority": "Enrichment",
        }.items():
            app.session_state[key] = value
        with patch.object(ui, "path_exists", return_value=True), \
                patch.object(ui, "render_cluster_visual_summary_view_model", return_value=None), \
                patch.object(ui, "render_morphology_graph", return_value=None) as graph, \
                patch.object(ui, "run_birch_clustering") as fit:
            app.run(timeout=30)
            self.assertFalse(app.exception)
            first = app.session_state["cluster_summary_view_model"]
            self.assertEqual({m["cluster_id"] for m in first["clusters"]}, set(range(20)))
            self.assertEqual(first["clusters"][0]["cluster_id"], 1)
            self.assertFalse(any(s.label == "Prioritize by" for s in app.selectbox))
            self.assertEqual(app.dataframe[0].value.cluster.iloc[0], 1)
            self.assertTrue(any(c.value == (
                "Select any table row or use **Analyse this cluster** in a visual preview. "
                "Initial priority: labelled-candidate count, then density, then cluster size (descending)."
            ) for c in app.caption))
            app.run(timeout=30)
            self.assertFalse(app.exception)
            second = app.session_state["cluster_summary_view_model"]
            self.assertEqual(second["cluster_download_df"].cluster.tolist(), first["cluster_download_df"].cluster.tolist())
            self.assertEqual(app.dataframe[0].value.cluster.iloc[0], 1)
            self.assertEqual([m["cluster_id"] for m in graph.call_args.args[1]["clusters"]],
                             second["cluster_download_df"].head(20).cluster.tolist())
            self.assertEqual(graph.call_args.args[3], "Labelled-candidate count")
            expander_id = next(e for e in app.expander if e.label == "Clustering summary").proto.id
            graph_calls = graph.call_count
            app.session_state["cluster_summary_expanded"] = False
            app.run(timeout=30)
            self.assertEqual(graph.call_count, graph_calls)
            app.session_state["cluster_summary_expanded"] = True
            app.run(timeout=30)
            self.assertFalse(app.exception)
            self.assertEqual(graph.call_count, graph_calls + 1)
            self.assertEqual(next(e for e in app.expander if e.label == "Clustering summary").proto.id,
                             expander_id)
            fit.assert_not_called()
