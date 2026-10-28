import html
import re
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from euclid_umap_explorer import components, config, storage, umap
from euclid_umap_explorer.analysis import (
    add_cluster_extreme_roles,
    build_cluster_summary,
    recommended_numeric_filter,
    sample_for_display,
)
from euclid_umap_explorer.catalogs import load_lens_catalog
from euclid_umap_explorer.help_tooltips import HELP_TOOLTIP_SCRIPT


class HelpTooltipTests(unittest.TestCase):
    def test_custom_tooltip_bodies_are_regular_weight_and_flow_steps_are_spaced(self):
        with patch.object(components.st, "markdown") as render, patch.object(
            components.st_components, "html"
        ):
            components.inject_plot_cursor_css()
        css = render.call_args.args[0]
        for selector, declaration in (
            (".concept-popover", "font-weight: 400;"),
            (".concept-popover strong", "font-weight: 700;"),
            (".lens-grade-help__description", "font-weight: 400;"),
            (".lens-grade-help__section + .lens-grade-help__section", "margin-top: 0.7rem;"),
            (".concept-popover__steps li + li", "margin-top: 0.5rem;"),
        ):
            with self.subTest(selector=selector):
                rule = re.search(re.escape(selector) + r"\s*\{([^}]+)\}", css)
                self.assertIsNotNone(rule)
                self.assertIn(declaration, rule.group(1))

    def test_shared_help_stays_concise(self):
        for name, text in {
            **config.PARAMETER_HELP,
            **config.CLUSTER_EXTREME_HELP,
            "Lens grades": config.LENS_GRADE_HELP_NOTE,
            "Straight-line artifacts": config.STRAIGHT_LINE_ARTIFACT_FILTER_HELP,
            "Recommended thresholds": config.RECOMMENDED_FILTER_HELP,
        }.items():
            with self.subTest(name=name):
                self.assertLessEqual(len(text.split()), 65)
        self.assertLessEqual(sum(len(step.split()) for step in config.ANALYSIS_FLOW_STEPS), 150)

    def test_long_popovers_can_scroll_and_have_a_mobile_layout(self):
        with patch.object(components.st, "markdown") as render, patch.object(
            components.st_components, "html"
        ) as install:
            components.inject_plot_cursor_css()
        css = render.call_args.args[0]
        self.assertIn("max-height: calc(100dvh - 1.5rem)", css)
        self.assertIn("overflow-y: auto", css)
        self.assertIn("overscroll-behavior: contain", css)
        self.assertIn(".concept-popover[data-escope-open]", css)
        self.assertNotIn(".stColumn:last-child:not(:first-child) .concept-popover", css)
        self.assertNotIn("top: 4.75rem", css)
        install.assert_called_once_with(HELP_TOOLTIP_SCRIPT, height=0, width=0)

    def test_tooltip_controller_anchors_to_the_control_and_handles_dismissal(self):
        for expected in (
            "getBoundingClientRect()", "showPopover()", "width - box.width - margin",
            "height - box.height - margin", "'Escape'", "'pointerdown'",
            "'resize'", "'scroll'", "setTimeout(close, 300)",
        ):
            self.assertIn(expected, HELP_TOOLTIP_SCRIPT)

    def test_parameter_help_covers_every_parameter_rendered_in_the_ui(self):
        self.assertEqual(set(config.PARAMETER_HELP), {
            "threshold", "branching_factor", "batch_size", "n_neighbors",
            "min_dist", "target_weight", "Maximum objects",
        })
        self.assertIn(f"{config.DEFAULT_BIRCH_BATCH_SIZE:,}", config.PARAMETER_HELP["batch_size"])
        self.assertIn("RMS radius", config.PARAMETER_HELP["threshold"])
        self.assertIn("not a guaranteed minimum", config.PARAMETER_HELP["min_dist"].lower())
        self.assertIn("both to fit and to draw", config.PARAMETER_HELP["Maximum objects"])
        for label, text in config.PARAMETER_HELP.items():
            with self.subTest(label=label), patch.object(components.st, "markdown") as render:
                components.render_help_label(label, text)
                self.assertIn(html.escape(text), render.call_args.args[0])

    def test_flow_help_matches_review_priority_and_export_limit(self):
        rows = []
        for cluster, size, labelled in ((0, 100, 20), (1, 5, 4), (2, 50, 20)):
            rows.extend({"cluster": cluster, "object_id": f"{cluster}_{i}",
                         "is_lens": i < labelled} for i in range(size))
        self.assertEqual(build_cluster_summary(pd.DataFrame(rows))["cluster"].tolist(), [2, 0, 1])
        with patch.object(components.st, "markdown") as render:
            components.render_app_flow_help()
        markup = render.call_args.args[0]
        self.assertIn('<ol class="concept-popover__steps">', markup)
        self.assertEqual(markup.count("<li>"), len(config.ANALYSIS_FLOW_STEPS))
        self.assertIn("<strong>Analysis flow</strong>", markup)
        self.assertNotIn("<br>", markup)
        for step in config.ANALYSIS_FLOW_STEPS:
            self.assertIn(f"<li>{html.escape(step)}</li>", markup)
        self.assertIn("A+B enrichment, A count and canonical score", markup)
        self.assertIn("morphology exploration by score", markup)
        self.assertIn(f"{config.DOWNLOAD_MAX_UMAP_ROWS:,} rows", markup)
        self.assertNotIn("Export Unknown objects close", markup)

    def test_grades_are_catalogue_metadata_not_an_application_score_cut(self):
        with TemporaryDirectory() as tmp, patch.object(storage, "USE_LOCAL_CACHE", False):
            path = Path(tmp) / "lenses.csv"
            pd.DataFrame({"object_id": ["1", "2"], "grade": ["C", "A"],
                          "expert_score": [0.6, 3.0]}).to_csv(path, index=False)
            self.assertEqual(load_lens_catalog(str(path), ("C",))["object_id"].tolist(), ["1"])
        with patch.object(components.st, "markdown") as render:
            components.render_lens_grade_help_label("Lens grades", config.LENS_GRADE_HELP, {})
        self.assertIn(html.escape(config.LENS_GRADE_HELP_NOTE), render.call_args.args[0])
        self.assertNotIn("expert score >", render.call_args.args[0])

    def test_lens_grade_intro_has_the_same_spacing_as_other_sections(self):
        with patch.object(components.st, "markdown") as render:
            components.render_lens_grade_help_label("Lens grades", config.LENS_GRADE_HELP, {})
        markup = render.call_args.args[0]
        self.assertIn(
            '<div class="lens-grade-help__section">'
            '<div class="lens-grade-help__description">'
            f'{html.escape(config.LENS_GRADE_HELP_NOTE)}</div></div>'
            '<div class="lens-grade-help__section">'
            '<div class="lens-grade-help__description"><strong>Grade A:</strong>',
            markup,
        )

    def test_extremes_use_within_cluster_standardization_and_can_change_after_filters(self):
        data = pd.DataFrame({"object_id": list("abcdef"), "is_lens": False,
                             "feat_pca_0": [0, 1000, 1050, 1100, 1150, 1200],
                             "feat_pca_1": [0, 0, 0, 0, 0, 100]})
        features = ["feat_pca_0", "feat_pca_1"]
        marked = add_cluster_extreme_roles(data, features)
        self.assertEqual(marked.loc[marked["is_anomaly"], "object_id"].item(), "f")
        raw = data[features].to_numpy()
        self.assertEqual(np.linalg.norm(raw - raw.mean(axis=0), axis=1).argmax(), 0)
        filtered = add_cluster_extreme_roles(data.iloc[:-1], features)
        self.assertEqual(filtered.loc[filtered["is_anomaly"], "object_id"].item(), "a")
        with patch.object(components.st, "markdown") as render:
            components.render_thumbnail_group_title("Canonical / anomalous")
        self.assertNotIn("opacity:", render.call_args.args[0])
        for text in config.CLUSTER_EXTREME_HELP.values():
            self.assertIn(html.escape(text), render.call_args.args[0])

    def test_max_objects_samples_the_fit_input_with_landmarks_then_labels_first(self):
        data = pd.DataFrame({"object_id": np.arange(20), "is_lens": [i < 3 for i in range(20)],
                             "is_canonical": [i == 18 for i in range(20)],
                             "is_anomaly": [i == 19 for i in range(20)]})
        small = sample_for_display(data, 3)
        self.assertEqual(len(small), 3)
        self.assertTrue({18, 19}.issubset(set(small["object_id"])))
        self.assertEqual(int(small["is_lens"].sum()), 1)
        pd.testing.assert_frame_equal(sample_for_display(data, 8), sample_for_display(data, 8))

    def test_umap_help_matches_fitted_neighborhood_and_categorical_targets(self):
        calls = []

        class FakeUMAP:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def fit_transform(self, values, y=None):
                calls.append((self.kwargs, values, y))
                return np.zeros((len(values), 2))

        data = pd.DataFrame({"feat_pca_0": [0.0, 1.0, 2.0, 3.0],
                             "lens_grade": ["A", "B", "C", ""]})
        with patch.dict(sys.modules, {"umap": SimpleNamespace(UMAP=FakeUMAP)}):
            umap._compute_umap_embedding_impl(data, ["feat_pca_0"], 50, 0.1)
            umap._compute_semisupervised_umap_embedding_impl(data, ["feat_pca_0"], 50, 0.1, 0.5)
        for parameters, values, _ in calls:
            self.assertEqual(parameters["n_neighbors"], 3)
            self.assertEqual(parameters["n_components"], 2)
            np.testing.assert_allclose(values.mean(axis=0), 0, atol=1e-12)
        self.assertIsNone(calls[0][2])
        self.assertEqual(calls[1][0]["target_metric"], "categorical")
        np.testing.assert_array_equal(calls[1][2], [2, 1, 0, -1])

    def test_recommended_cut_enrichment_uses_its_cluster_not_the_catalogue(self):
        data = pd.DataFrame({"feature": np.arange(100), "is_lens": np.arange(100) < 10})
        result = recommended_numeric_filter(
            data, "feature", min_valid_objects=2, min_selected_objects=1,
            min_lenses=1, min_unknowns=1, min_recall=0.05,
        )
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["base_lens_rate"], 0.1)
        self.assertAlmostEqual(result["enrichment"], result["lens_rate"] / 0.1)
        self.assertAlmostEqual(result["score"], result["enrichment"] * np.sqrt(result["recall"]))
        self.assertIn("not the whole catalogue", config.RECOMMENDED_FILTER_HELP)


if __name__ == "__main__":
    unittest.main()
