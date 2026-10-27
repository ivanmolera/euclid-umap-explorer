import hashlib
import json
import unittest
import zipfile
from io import BytesIO
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from streamlit.testing.v1 import AppTest

from euclid_umap_explorer import cluster_diagnostics_ui as ui
from euclid_umap_explorer.analysis import add_cluster_extreme_roles
from euclid_umap_explorer.cluster_diagnostics import (
    add_canonical_scores, add_reference_comparison, attach_evidence, build_diagnostic_figure,
    build_diagnostic_profiles, classify_diagnostic_regions, diagnostic_signature,
    diagnostic_statistics, independent_validation_auc, read_validation_labels,
    select_diagnostic_cohort,
)
from euclid_umap_explorer.lens_plausibility import (
    DETECTOR_CONFIG, DETECTOR_VERSION, analyze_image, angular_span_deg,
    long_border_lines, score_image_bytes,
)


FEATURES = ["feat_pca_0", "feat_pca_1"]


def fixture():
    return pd.DataFrame([
        {"cluster": cluster, "object_id": str(2700244936644435500 + cluster * 10 + i),
         "id_str": f"Q1_R1_1_{2700244936644435500 + cluster * 10 + i}",
         "is_lens": i < cluster, "lens_grade": "ABC"[i % 3] if i < cluster else "",
         "feat_pca_0": float(cluster * 10 + i), "feat_pca_1": float(i * (cluster + 1))}
        for cluster in range(4) for i in range(5)
    ])


def test_images():
    rng = np.random.default_rng(11)
    size = 160
    yy, xx = np.mgrid[:size, :size]
    core = 130 * np.exp(-((xx - 80) ** 2 + (yy - 80) ** 2) / (2 * 9 ** 2))
    base = np.clip(15 + core + rng.normal(0, 3, (size, size)), 0, 255).astype(np.uint8)
    arc = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)
    line = arc.copy()
    cv2.ellipse(arc, (80, 80), (42, 42), 0, 20, 130, (225, 225, 225), 3)
    cv2.ellipse(arc, (80, 80), (42, 42), 0, 205, 280, (190, 190, 190), 3)
    cv2.line(line, (1, 55), (158, 105), (225, 225, 225), 3)
    return arc, line


def png(image):
    output = BytesIO()
    Image.fromarray(image).save(output, format="PNG")
    return output.getvalue()


def scored_frame():
    data = fixture()
    profiles, canonical = build_diagnostic_profiles(data, FEATURES)
    evidence = {str(row.canonical): {"score": int(row.cluster * 30), "status": "exploratory_visual_only"}
                for row in profiles.itertuples()}
    return data, classify_diagnostic_regions(add_reference_comparison(attach_evidence(profiles, evidence), data)), canonical


class VisualEvidenceTests(unittest.TestCase):
    def test_synthetic_arcs_score_above_line_and_preserve_original(self):
        arc, line = test_images()
        before = arc.copy()
        result = analyze_image(arc, center_override=(80, 80))
        line_result = analyze_image(line, center_override=(80, 80))
        self.assertGreater(result["score"], line_result["score"])
        self.assertGreater(len(result["accepted"]), 0)
        np.testing.assert_array_equal(arc, before)
        np.testing.assert_array_equal(result["original"], before)
        self.assertFalse(np.array_equal(result["overlay"], before))
        self.assertEqual(result["score"], round(sum(result["contributions"].values())))
        self.assertEqual(result["score"], analyze_image(arc, center_override=(80, 80))["score"])

    def test_non_evaluable_is_missing_not_a_negative_score(self):
        self.assertIsNone(analyze_image(np.zeros((30, 30, 3), dtype=np.uint8))["score"])
        self.assertIsNone(analyze_image(np.zeros((100, 100, 3), dtype=np.uint8))["score"])
        for bad in (np.zeros((100, 100)), np.zeros((100, 100, 3), dtype=float)):
            with self.assertRaises(ValueError):
                analyze_image(bad)
        with self.assertRaises(ValueError):
            analyze_image(test_images()[0], center_override=(np.nan, 2))

    def test_hough_shapes_and_zero_length_lines_do_not_crash(self):
        gray = np.random.default_rng(4).random((160, 160)).astype(np.float32)
        for shape in ((1, 1, 4), (1, 4), (4,)):
            with patch("cv2.HoughLinesP", return_value=np.array([0, 0, 0, 0]).reshape(shape)):
                self.assertEqual(long_border_lines(gray, DETECTOR_CONFIG), [])
        self.assertAlmostEqual(angular_span_deg(np.deg2rad([350, 0, 10])), 20.)

    def test_byte_result_has_original_dimensions_and_json_safe_metadata(self):
        result = score_image_bytes(png(test_images()[0]))
        self.assertEqual((result["image_width"], result["image_height"]), (160, 160))
        self.assertIn("overlay_png", result)
        with Image.open(BytesIO(result["overlay_png"])) as overlay:
            self.assertEqual(overlay.size, (160, 160))
        json.dumps({key: value for key, value in result.items() if key != "overlay_png"}, allow_nan=False)


class ClusterDiagnosticTests(unittest.TestCase):
    def test_canonical_matches_previews_and_does_not_depend_on_labels(self):
        data = fixture()
        profiles, canonical = build_diagnostic_profiles(data, FEATURES)
        changed = data.assign(is_lens=~data.is_lens, lens_grade="C")
        other, changed_canonical = build_diagnostic_profiles(changed, FEATURES)
        self.assertEqual(profiles.canonical.tolist(), other.canonical.tolist())
        np.testing.assert_allclose(profiles.pca_rms, other.pca_rms)
        for group_id, row in canonical.items():
            members = data[data.cluster.eq(group_id)].reset_index(drop=True)
            expected = add_cluster_extreme_roles(members, FEATURES)
            self.assertEqual(row.object_id, expected.loc[expected.is_canonical].iloc[0].object_id)
            self.assertEqual(row.object_id, changed_canonical[group_id].object_id)
        self.assertNotEqual(diagnostic_signature(data, FEATURES), diagnostic_signature(changed, FEATURES))

    def test_sampling_is_label_blind_and_independent_of_row_order(self):
        profiles, _ = build_diagnostic_profiles(fixture(), FEATURES)
        selected = select_diagnostic_cohort(profiles, min_objects=5, max_clusters=2)
        other = select_diagnostic_cohort(profiles.iloc[::-1].assign(n_lenses=999), min_objects=5, max_clusters=2)
        self.assertEqual(selected.cluster.tolist(), other.cluster.tolist())
        self.assertEqual(select_diagnostic_cohort(profiles, min_objects=6).shape[0], 0)
        self.assertEqual(select_diagnostic_cohort(profiles, min_objects=1, preview_ids=[3]).cluster.tolist(), [3])

    def test_summary_scores_keep_real_zero_missing_values_and_canonical_ids(self):
        profiles, _ = build_diagnostic_profiles(fixture(), FEATURES)
        original = profiles.copy()
        evidence = {
            profiles.canonical.iloc[0]: {"score": 0, "status": "exploratory_visual_only"},
            profiles.canonical.iloc[1]: {"score": 89, "status": "exploratory_visual_only"},
            profiles.canonical.iloc[2]: {"score": None, "status": "image_error"},
        }
        summary = add_canonical_scores(profiles, evidence)
        self.assertEqual(summary.canonical_score.iloc[:2].tolist(), [0., 89.])
        self.assertTrue(summary.canonical_score.iloc[2:].isna().all())
        self.assertEqual(summary.canonical_score_status.iloc[-1], "not_scored")
        self.assertEqual(summary.canonical.tolist(), original.canonical.tolist())
        pd.testing.assert_frame_equal(profiles, original)
        with patch.object(ui.st, "session_state", {"cluster_diagnostics_state": {
            "_data": original, "features": tuple(FEATURES), "evidence": evidence,
            "canonical_features": tuple(FEATURES),
        }}):
            self.assertEqual(ui.summary_with_canonical_scores(profiles, original, FEATURES).canonical_score.iloc[1], 89.)
            self.assertTrue(ui.summary_with_canonical_scores(profiles, original.copy(), FEATURES).canonical_score.isna().all())
            self.assertTrue(ui.summary_with_canonical_scores(profiles, original, FEATURES[:1]).canonical_score.isna().all())

    def test_dispersion_uses_fixed_catalogue_scaling_for_subsets(self):
        data = fixture()
        full, _ = build_diagnostic_profiles(data, FEATURES)
        subset, _ = build_diagnostic_profiles(data[data.cluster.eq(2)], FEATURES, reference=data)
        self.assertAlmostEqual(subset.pca_rms.iloc[0], full.loc[full.cluster.eq(2), "pca_rms"].iloc[0])

    def test_preview_pca_selection_changes_canonical_but_not_full_pca_dispersion(self):
        data = fixture().assign(feat_pca_1=np.where(np.arange(20) % 5 == 2, 100., 0.))
        full, _ = build_diagnostic_profiles(data, FEATURES)
        selected, canonical = build_diagnostic_profiles(data, FEATURES, canonical_features=FEATURES[:1])
        self.assertTrue(full.canonical.ne(selected.canonical).all())
        np.testing.assert_allclose(full.pca_rms, selected.pca_rms)
        for group_id, row in canonical.items():
            expected = add_cluster_extreme_roles(data[data.cluster.eq(group_id)], FEATURES[:1])
            self.assertEqual(row.object_id, expected.loc[expected.is_canonical].iloc[0].object_id)
        with self.assertRaises(ValueError):
            build_diagnostic_profiles(data, FEATURES, canonical_features=["feat_pca_39"])

    def test_full_and_leave_canonical_out_reference_counts(self):
        data = fixture()
        profiles, _ = build_diagnostic_profiles(data, FEATURES)
        full = add_reference_comparison(profiles, data, exclude_canonical=False)
        others = add_reference_comparison(profiles, data)
        row = others.loc[others.cluster.eq(3)].iloc[0]
        self.assertTrue(row.canonical_is_lens)
        self.assertEqual((row.n_reference_compared, row.n_members_compared), (2, 4))
        self.assertAlmostEqual(row.reference_baseline, 5 / 19)
        self.assertAlmostEqual(row.enrichment, (2 / 4) / (5 / 19))
        self.assertAlmostEqual(full.loc[full.cluster.eq(3), "enrichment"].iloc[0], (3 / 5) / (6 / 20))
        a_only = add_reference_comparison(profiles, data, grades=("A",))
        self.assertEqual(a_only.n_unknown.tolist(), full.n_unknown.tolist())
        self.assertEqual(a_only.n_reference_full.tolist(), [0, 1, 1, 1])

    def test_no_references_missing_features_and_failed_images(self):
        data = fixture().assign(is_lens=False, lens_grade="")
        profiles, _ = build_diagnostic_profiles(data, FEATURES)
        scored = attach_evidence(profiles, {profiles.canonical.iloc[0]: {"score": None, "status": "image_error"}})
        self.assertTrue(scored.visual_score.isna().all())
        self.assertTrue(add_reference_comparison(scored, data).enrichment.isna().all())
        self.assertEqual(diagnostic_statistics(add_reference_comparison(scored, data))["n_compared"], 0)
        with self.assertRaises(ValueError):
            build_diagnostic_profiles(pd.concat([data, data]), FEATURES)
        with self.assertRaises(ValueError):
            build_diagnostic_profiles(data, ["is_lens"])
        with self.assertRaises(ValueError):
            attach_evidence(profiles, {profiles.canonical.iloc[0]: {"score": 101}})

    def test_figure_preserves_ids_zero_enrichment_and_unknowns_without_confirmation(self):
        _, frame, _ = scored_frame()
        frame.loc[0, "n_unknown"] = 0
        figure = build_diagnostic_figure(frame)
        self.assertEqual(figure.data[0].customdata[0][1], frame.canonical.iloc[0])
        self.assertEqual(figure.data[0].marker.symbol[0], "x")
        self.assertIn(0., figure.data[0].y)
        self.assertTrue(all("confirmed" not in name.lower() for name in frame.diagnostic_region))
        self.assertTrue(np.isfinite(diagnostic_statistics(frame)["pearson"]))
        constant = frame.assign(visual_score=10)
        self.assertIsNone(diagnostic_statistics(constant)["spearman"])
        with self.assertRaises(ValueError):
            classify_diagnostic_regions(frame, enrichment_threshold=-1)

    def test_auc_requires_explicit_independent_binary_labels(self):
        data, frame, _ = scored_frame()
        no_negatives = pd.DataFrame({"object_id": data.object_id, "is_lens": 1})
        self.assertIsNone(independent_validation_auc(data, frame, no_negatives)["auc"])
        independent = pd.DataFrame({"object_id": data.object_id, "is_lens": data.cluster.ge(2).astype(int)})
        result = independent_validation_auc(data, frame, independent)
        self.assertAlmostEqual(result["auc"], 1.)
        self.assertEqual(result["n_validated"], len(data) - len(frame))
        self.assertEqual(independent_validation_auc(data, frame, independent, exclude_canonical=False)["n_validated"], len(data))
        for bad in (b"object_id,is_lens\n2700244936644435563,Unknown\n",
                    b"object_id,is_lens\n2.7002449366444356e18,1\n", b"object_id,is_lens\n1,1\n1,0\n"):
            with self.assertRaises(ValueError):
                read_validation_labels(bad)
        parsed = read_validation_labels(b"object_id,is_lens\n2700244936644435563,1\n-625302350493849139,0\n")
        self.assertEqual(parsed.object_id.iloc[0], "2700244936644435563")
        for invalid in (independent.assign(is_lens="Unknown"),
                        independent.assign(object_id=independent.object_id.astype(float)),
                        pd.concat([independent, independent])):
            with self.assertRaises(ValueError):
                independent_validation_auc(data, frame, invalid)


UI_APP = """
import streamlit as st
from tests.test_cluster_diagnostics import fixture, FEATURES
from euclid_umap_explorer.cluster_diagnostics_ui import render_cluster_diagnostics
if 'data' not in st.session_state:
    st.session_state.data = fixture()
render_cluster_diagnostics(st.session_state.data, FEATURES, preview_ids=[3, 2])
"""


class ClusterDiagnosticUITests(unittest.TestCase):
    def setUp(self):
        ui.cached_visual_evidence.clear()

    def test_disabled_panel_does_not_read_images_or_compute_profiles(self):
        with patch.object(ui, "build_diagnostic_profiles") as profiles, patch.object(ui, "_load_canonical_evidence") as load:
            app = AppTest.from_string(UI_APP).run()
            self.assertFalse(app.exception)
            profiles.assert_not_called()
            load.assert_not_called()

    def test_panel_computes_once_has_original_and_overlay_and_preserves_membership(self):
        image_bytes = png(test_images()[0])
        evidence = {**score_image_bytes(image_bytes), "image_path": "fixture.png",
                    "image_sha256": hashlib.sha256(image_bytes).hexdigest()}
        app = AppTest.from_string(UI_APP)
        with patch.object(ui, "_load_canonical_evidence", return_value=evidence) as load, \
                patch.object(ui, "load_image_bytes", return_value=image_bytes):
            app.run()
            app.toggle(key="cluster_diagnostics_show").set_value(True).run()
            self.assertFalse(app.exception)
            load.assert_not_called()
            app.number_input(key="cluster_diagnostics_min").set_value(1)
            app.button(key="cluster_diagnostics_run").click().run(timeout=30)
            self.assertFalse(app.exception)
            self.assertEqual(load.call_count, 4)
            self.assertEqual(len(app.get("image")), 2)
            self.assertEqual(next(m for m in app.metric if m.label == "Validation ROC AUC").value, "-")
            app.number_input(key="cluster_diagnostics_score_threshold").set_value(50).run()
            self.assertFalse(app.exception)
            self.assertEqual(load.call_count, 4)
            app.checkbox(key="cluster_diagnostics_exclude").set_value(False).run()
            self.assertFalse(app.exception)
            self.assertEqual(load.call_count, 4)
            pd.testing.assert_frame_equal(app.session_state["data"], fixture())
            state = app.session_state["cluster_diagnostics_state"]
            frame = classify_diagnostic_regions(add_reference_comparison(state["cohort"], state["_reference"]))
            exported = ui._export_diagnostics(
                frame, state, grades=None, exclude_canonical=True, score_threshold=70., enrichment_threshold=10.,
                group_column="cluster", scope_label="BIRCH clusters", validation={"auc": None},
            )
            with zipfile.ZipFile(BytesIO(exported)) as archive:
                manifest = json.loads(archive.read("manifest.json"))
                self.assertEqual(manifest["detector_version"], DETECTOR_VERSION)
                self.assertIn("not P(lens)", manifest["score_semantics"])
                self.assertEqual(len(manifest["images"]), 4)
                self.assertIn("marked_regions.csv", archive.namelist())

    def test_missing_images_are_not_plotted_as_negatives_and_data_changes_reset_results(self):
        app = AppTest.from_string(UI_APP)
        with patch.object(ui, "_load_canonical_evidence", return_value={"score": None, "status": "missing_cutout"}):
            app.run()
            app.toggle(key="cluster_diagnostics_show").set_value(True).run()
            app.number_input(key="cluster_diagnostics_min").set_value(1)
            app.button(key="cluster_diagnostics_run").click().run(timeout=30)
            self.assertFalse(app.exception)
            self.assertEqual(next(m for m in app.metric if m.label == "Compared groups").value, "0 / 4")
            self.assertTrue(app.session_state["cluster_diagnostics_state"]["cohort"].visual_score.isna().all())
            app.session_state["data"] = fixture().assign(feat_pca_0=99.)
            app.run()
            self.assertFalse(app.exception)
            self.assertIsNone(app.session_state["cluster_diagnostics_state"]["cohort"])

    def test_subcluster_scope_has_separate_state_and_uses_catalogue_baseline(self):
        sub_app = UI_APP.replace("preview_ids=[3, 2]", "reference=st.session_state.data, group_column='cluster', prefix='subcluster_diagnostics', scope_label='UMAP subclusters'")
        app = AppTest.from_string(sub_app).run()
        app.toggle(key="subcluster_diagnostics_show").set_value(True).run()
        self.assertFalse(app.exception)
        self.assertIn("subcluster_diagnostics_state", app.session_state)
        self.assertNotIn("cluster_diagnostics_state", app.session_state)

    def test_main_summary_receives_scores_immediately_without_changing_priority(self):
        from euclid_umap_explorer import ui as main_ui

        catalogue = fixture().assign(feat_pca_1=np.where(np.arange(20) % 5 == 2, 100., 0.))
        image_bytes = png(test_images()[0])
        evidence = {**score_image_bytes(image_bytes), "image_path": "fixture.png"}
        app = AppTest.from_string("from euclid_umap_explorer.ui import main\nmain()")
        for key, value in {
            "cluster_ready": True, "cluster_result": (catalogue, FEATURES, None),
            "cluster_params": {"lens_grades": ("A", "B", "C"), "threshold": 5.5,
                               "branching_factor": 42, "batch_size": 25000},
            "cluster_summary_expanded": True,
        }.items():
            app.session_state[key] = value
        with patch.object(main_ui, "path_exists", return_value=True), \
                patch.object(main_ui, "render_cluster_visual_summary_view_model", return_value=None), \
                patch.object(main_ui, "render_morphology_graph", return_value=None), \
                patch.object(main_ui, "run_birch_clustering") as fit, \
                patch.object(ui, "_load_canonical_evidence", return_value=evidence), \
                patch.object(ui, "load_image_bytes", return_value=image_bytes):
            app.run(timeout=30)
            self.assertFalse(app.exception)
            app.multiselect(key="selected_pca_components").set_value(FEATURES[:1]).run(timeout=30)
            before = app.dataframe[0].value
            self.assertTrue(before.canonical_score.isna().all())
            app.toggle(key="cluster_diagnostics_show").set_value(True).run()
            app.number_input(key="cluster_diagnostics_min").set_value(1)
            app.button(key="cluster_diagnostics_run").click().run(timeout=30)
            self.assertFalse(app.exception)
            after = app.dataframe[0].value
            self.assertEqual(after.canonical_score.tolist(), [evidence["score"]] * 4)
            self.assertEqual(after.cluster.tolist(), before.cluster.tolist())
            self.assertEqual(after.canonical.tolist(), before.canonical.tolist())
            self.assertEqual(app.session_state["cluster_diagnostics_state"]["canonical_features"], tuple(FEATURES[:1]))
            app.toggle(key="cluster_diagnostics_show").set_value(False).run()
            self.assertFalse(app.exception)
            self.assertEqual(app.dataframe[0].value.canonical_score.tolist(), after.canonical_score.tolist())
            app.multiselect(key="selected_pca_components").set_value(FEATURES).run(timeout=30)
            self.assertTrue(app.dataframe[0].value.canonical_score.isna().all())
            fit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
