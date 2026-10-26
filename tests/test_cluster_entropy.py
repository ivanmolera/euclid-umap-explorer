import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from euclid_umap_explorer import components, ui
from euclid_umap_explorer.analysis import binary_label_entropy, build_cluster_summary
from euclid_umap_explorer.config import ENRICHMENT_HELP, LABEL_DENSITY_HELP, LABEL_ENTROPY_HELP
from euclid_umap_explorer.downloads import cluster_summary_download_df


def example_catalogue():
    rows = []
    for cluster, n_objects, n_labels in ((0, 4, 0), (1, 4, 2), (2, 4, 4), (3, 10, 1)):
        for index in range(n_objects):
            rows.append({
                "cluster": cluster,
                "object_id": str(len(rows)),
                "feat_pca_0": float(index),
                "is_lens": index < n_labels,
                "lens_grade": "A" if index < n_labels else "",
            })
    return pd.DataFrame(rows)


class ClusterEntropyTests(unittest.TestCase):
    def test_entropy_boundaries_mixtures_and_symmetry(self):
        with np.errstate(divide="raise", invalid="raise"):
            actual = binary_label_entropy(np.array([0, 0.1, 0.5, 0.9, 1]))
        np.testing.assert_allclose(actual, [0, 0.4689955935892812, 1, 0.4689955935892812, 0])
        self.assertEqual(binary_label_entropy(0), 0)
        self.assertEqual(binary_label_entropy(1), 0)
        self.assertEqual(binary_label_entropy(0.5), 1)

    def test_invalid_label_fractions_are_not_reported_as_zero_entropy(self):
        for value in (-0.1, 1.1, np.nan, np.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                binary_label_entropy(value)

    def test_summary_preserves_support_order_and_handles_an_empty_catalogue(self):
        catalogue = example_catalogue()
        summary = build_cluster_summary(catalogue)
        self.assertEqual(list(summary["cluster"]), [2, 1, 3, 0])
        by_cluster = summary.set_index("cluster")
        self.assertEqual(by_cluster.loc[0, "label_entropy_bits"], 0)
        self.assertEqual(by_cluster.loc[2, "label_entropy_bits"], 0)
        self.assertEqual(by_cluster.loc[1, "label_entropy_bits"], 1)
        self.assertAlmostEqual(by_cluster.loc[3, "label_entropy_bits"], 0.4689955935892812)
        empty = build_cluster_summary(catalogue.iloc[:0])
        self.assertTrue(empty.empty)
        self.assertIn("label_entropy_bits", empty)

    def test_preview_selected_cluster_and_both_exports_share_entropy_values(self):
        catalogue = example_catalogue()
        summary = build_cluster_summary(catalogue)
        with patch.object(components, "SUMMARY_VISUAL_CLUSTER_LIMIT", 1):
            view = components.build_cluster_summary_view_model(
                catalogue, summary, ["feat_pca_0"], ["feat_pca_0"]
            )
        self.assertEqual(len(view["clusters"]), 1)
        self.assertEqual(view["clusters"][0]["label_entropy_bits"], 0)
        selected = components.build_selected_cluster_view_model(
            catalogue, summary, 1, ["feat_pca_0"], ["feat_pca_0"]
        )
        self.assertEqual(selected["label_entropy_bits"], 1)
        exported = cluster_summary_download_df(catalogue, summary, ["feat_pca_0"])
        for table in (view["cluster_download_df"], exported):
            np.testing.assert_allclose(table["label_entropy_bits"], summary["label_entropy_bits"])

    def test_cluster_metrics_expose_formula_and_caveats(self):
        catalogue = example_catalogue()
        model = components.build_selected_cluster_view_model(
            catalogue, build_cluster_summary(catalogue), 1, ["feat_pca_0"], ["feat_pca_0"]
        )
        app = AppTest.from_string("""
import streamlit as st
from euclid_umap_explorer.components import render_cluster_preview_card
render_cluster_preview_card(st.session_state['entropy_test_model'],
                            show_analysis_button=False, aladin_links={})
""")
        app.session_state["entropy_test_model"] = model
        with patch.object(components, "show_thumbnail_group"):
            app.run(timeout=30)
        self.assertFalse(app.exception)
        metric = next(item for item in app.metric if item.label == "Entropy")
        self.assertEqual(metric.value, "1,000")
        self.assertEqual(metric.proto.help, LABEL_ENTROPY_HELP)
        density = next(item for item in app.metric if item.label == "Density")
        self.assertEqual(density.value, "50,00%")
        self.assertEqual(density.proto.help, LABEL_DENSITY_HELP)
        enrichment = next(item for item in app.metric if item.label == "Enrichment")
        self.assertEqual(enrichment.value, "1,57x")
        self.assertEqual(enrichment.proto.help, ENRICHMENT_HELP)
        for help_text, formula in (
            (LABEL_ENTROPY_HELP, r"H = -p\log_2 p - (1-p)\log_2(1-p)"),
            (ENRICHMENT_HELP, r"E_c = \frac{L_c/N_c}{L/N}"),
        ):
            with self.subTest(formula=formula):
                self.assertIn(f"\n$$\n{formula}\n$$\n", help_text)
                self.assertIn("not a confirmed non-lens", help_text)

    def test_metric_tooltips_only_emphasize_their_titles(self):
        for title, text, word_limit in (
            ("Labelled candidate fraction", LABEL_DENSITY_HELP, 70),
            ("Label enrichment", ENRICHMENT_HELP, 120),
            ("Binary Shannon entropy", LABEL_ENTROPY_HELP, 160),
        ):
            with self.subTest(title=title):
                self.assertTrue(text.startswith(f"**{title}**\n\n"))
                self.assertEqual(text.count("**"), 2)
                self.assertLessEqual(len(text.split()), word_limit)

    def test_entropy_help_defines_each_symbol_and_a_worked_example(self):
        for symbol in ("H", "L_c", "N_c", "p = L_c/N_c", "1-p"):
            with self.subTest(symbol=symbol):
                self.assertIn(f"- ${symbol}$:", LABEL_ENTROPY_HELP)
        self.assertIn("cluster $c$", LABEL_ENTROPY_HELP)
        self.assertIn(r"$\log_2$: base-2 logarithm", LABEL_ENTROPY_HELP)
        self.assertIn("selected grades (A/B/C by default), pooled", LABEL_ENTROPY_HELP)
        self.assertIn("not a percentage", LABEL_ENTROPY_HELP)
        self.assertIn(
            rf"$H \approx {binary_label_entropy(10 / 100):.3f}$ bits",
            LABEL_ENTROPY_HELP,
        )

    def test_summary_table_migrates_cached_counts_without_reclustering(self):
        catalogue = example_catalogue()
        legacy_summary = build_cluster_summary(catalogue).drop(columns="label_entropy_bits")
        app = AppTest.from_string("from euclid_umap_explorer.ui import main\nmain()")
        for key, value in {
            "cluster_ready": True,
            "cluster_result": (catalogue, ["feat_pca_0"], None),
            "cluster_params": {"lens_grades": ("A", "B", "C"), "threshold": 6.2,
                               "branching_factor": 50, "batch_size": 25000},
            "cluster_summary_df": legacy_summary,
            "cluster_summary_view_signature": ("legacy",),
            "cluster_summary_view_model": {"stale": True},
        }.items():
            app.session_state[key] = value
        with patch.object(ui, "path_exists", return_value=True), \
                patch.object(ui, "run_birch_clustering") as clustering:
            app.run(timeout=30)
        self.assertFalse(app.exception)
        clustering.assert_not_called()
        table = app.dataframe[0]
        self.assertIn("label_entropy_bits", table.value)
        self.assertEqual(table.value.loc[1, "label_entropy_bits"], 1)
        column = json.loads(table.proto.columns)["label_entropy_bits"]
        self.assertEqual(column["label"], "Entropy")
        self.assertIsNone(column.get("help"))
        self.assertIn("label_entropy_bits", app.session_state["cluster_summary_df"])
        self.assertIn("label_entropy_bits", app.session_state["cluster_summary_view_model"]["cluster_download_df"])


if __name__ == "__main__":
    unittest.main()
