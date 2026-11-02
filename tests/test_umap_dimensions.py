import sys
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from euclid_umap_explorer import ui, umap
from euclid_umap_explorer.downloads import umap_download_df
from euclid_umap_explorer.umap_3d import build_3d_payload


class UmapDimensionsTests(unittest.TestCase):
    def data(self, count=12):
        return pd.DataFrame({'object_id': [str(2700244936644435500 + i) for i in range(count)],
                             'id_str': [f'Q1_R1_1_{i}' for i in range(count)],
                             'feat_pca_0': np.arange(count, dtype=float),
                             'lens_grade': ['A', 'B', 'C', ''] * (count // 4),
                             'is_lens': [True, True, True, False] * (count // 4)})

    def test_both_embeddings_fit_requested_dimensions_and_preserve_identifiers(self):
        calls = []

        class FakeUMAP:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                calls.append(kwargs)

            def fit_transform(self, values, y=None):
                self.targets = y
                return np.arange(len(values) * self.kwargs['n_components']).reshape(len(values), -1)

        for dimensions in (2, 3):
            with patch.dict(sys.modules, {'umap': SimpleNamespace(UMAP=FakeUMAP)}):
                data = self.data().assign(umap_3=-999, semi_umap_3=-999)
                result = umap._compute_umap_embedding_impl(data, ['feat_pca_0'], 5, .1, dimensions)
                semi = umap._compute_semisupervised_umap_embedding_impl(data, ['feat_pca_0'], 5, .1, .5, dimensions)
            self.assertEqual(result.object_id.tolist(), data.object_id.tolist())
            self.assertEqual(result.attrs['n_components'], dimensions)
            self.assertEqual('umap_3' in result, dimensions == 3)
            self.assertEqual('semi_umap_3' in semi, dimensions == 3)
            self.assertEqual(semi.semi_supervised_target.tolist()[:4], [2, 1, 0, -1])
            self.assertTrue(all(call['n_components'] == dimensions for call in calls[-2:]))
        for invalid in (1, 4):
            with self.assertRaises(ValueError):
                umap._compute_umap_embedding_impl(self.data(), ['feat_pca_0'], 5, .1, invalid)

    def test_small_3d_sample_avoids_spectral_initialization_failure(self):
        constructor = SimpleNamespace()

        class FakeUMAP:
            def __init__(self, **kwargs):
                constructor.args = kwargs

            def fit_transform(self, values):
                return np.zeros((len(values), 3))

        with patch.dict(sys.modules, {'umap': SimpleNamespace(UMAP=FakeUMAP)}):
            result = umap._compute_umap_embedding_impl(self.data().head(3), ['feat_pca_0'], 5, .1, 3)
        self.assertEqual(len(result), 3)
        self.assertEqual(constructor.args['init'], 'random')

    def test_signature_distinguishes_fit_dimensions_but_not_pca_subclusters(self):
        args = dict(selected_cluster=7, selected_features=['feat_pca_0'], pca_filters=(),
                    n_neighbors=5, min_dist=.1, max_objects=100,
                    cluster_params={'threshold': 5.5, 'branching_factor': 42, 'batch_size': 25000})
        two = umap.build_umap_signature(**args)
        three = umap.build_umap_signature(**args, n_components=3)
        self.assertNotEqual(two, three)
        self.assertEqual(two[:-1], three[:-1])
        self.assertEqual(two[-1], 2)

    def test_payload_and_download_keep_third_axis_and_exact_object_ids(self):
        frame = self.data().assign(umap_1=1., umap_2=2., umap_3=3.,
                                   point_index=np.arange(12), umap_point_label='Unknown')
        payload = build_3d_payload(frame)
        self.assertEqual(payload['points'][0]['id'], '2700244936644435500')
        self.assertEqual(payload['points'][0]['xyz'], [1., 2., 3.])
        with patch('euclid_umap_explorer.downloads.load_morphology_objects', return_value=pd.DataFrame()):
            exported = umap_download_df(frame, ['feat_pca_0'], [0], physical_df=pd.DataFrame())
        self.assertEqual(exported.umap_3.tolist(), [3.])
        self.assertEqual(exported.object_id.tolist(), ['2700244936644435500'])
        with self.assertRaises(ValueError):
            build_3d_payload(frame.assign(umap_3=np.nan))

    def test_dimension_callbacks_only_refit_existing_projections_and_reset_for_new_cluster(self):
        state = {}
        with patch.object(ui.st, 'session_state', state):
            ui.request_umap_dimension_change()
            self.assertNotIn('umap_requested', state)
            state['umap_signature'] = ('existing',)
            ui.request_umap_dimension_change()
            self.assertTrue(state['umap_requested'])
            state['semisupervised_umap_signature'] = ('existing',)
            ui.request_semi_dimension_change()
            self.assertTrue(state['semisupervised_umap_requested'])
            state['umap_3d'] = state['semi_umap_3d'] = True
            ui.clear_selected_cluster_analysis(clear_selection=False)
            self.assertFalse(state['umap_3d'])
            self.assertFalse(state['semi_umap_3d'])

    def test_switches_auto_refit_both_modes_and_preserve_ward_assignments(self):
        catalogue = self.data(24).assign(cluster=0)

        def fake_umap(data, *args, n_components=2, **kwargs):
            result = data.copy()
            for axis in range(1, n_components + 1):
                result[f'umap_{axis}'] = result['feat_pca_0'] * axis
            result.attrs['n_components'] = n_components
            return result

        def fake_semi(data, *args, n_components=2, **kwargs):
            result = fake_umap(data, n_components=n_components)
            for axis in range(1, n_components + 1):
                result[f'semi_umap_{axis}'] = result[f'umap_{axis}']
            result['semi_supervised_target'] = np.where(result.is_lens, 2, -1)
            result['semi_supervised_label'] = np.where(result.is_lens, 'Grade A', 'Unknown')
            return result

        app = AppTest.from_string('from euclid_umap_explorer.ui import main\nmain()', default_timeout=30)
        for key, value in {
            'cluster_ready': True, 'cluster_result': (catalogue, ['feat_pca_0'], None),
            'cluster_params': {'lens_grades': ('A', 'B', 'C'), 'threshold': 5.5,
                               'branching_factor': 42, 'batch_size': 25000},
            'analysis_cluster_id': 0, 'cluster_summary_expanded': False,
        }.items():
            app.session_state[key] = value
        with ExitStack() as stack:
            stack.enter_context(patch.object(ui, 'path_exists', return_value=True))
            stack.enter_context(patch.object(ui, 'selected_cluster_physical_measurements', return_value=pd.DataFrame()))
            for name in ('render_cluster_preview_card', 'render_cluster_visual_summary_view_model',
                         'render_selected_cluster_pca_characterization', 'render_selected_cluster_physical_characterization',
                         'render_dendrogram_preview_section', 'render_subcluster_physical_characterization',
                         'render_cluster_umap_interaction', 'render_semisupervised_umap_interaction'):
                stack.enter_context(patch.object(ui, name))
            fit = stack.enter_context(patch.object(ui, 'compute_umap_embedding', side_effect=fake_umap))
            semi = stack.enter_context(patch.object(ui, 'compute_semisupervised_umap_embedding', side_effect=fake_semi))
            hierarchy = stack.enter_context(patch.object(ui, 'compute_hierarchical_subclusters',
                side_effect=lambda data, *a: data.assign(hierarchical_subcluster=np.arange(len(data)) % 2)))
            app.run()
            self.assertFalse(app.exception)
            self.assertFalse(app.toggle(key='umap_3d').value)
            next(b for b in app.button if b.label == 'Compute UMAP').click().run()
            self.assertFalse(app.exception)
            self.assertEqual(fit.call_args.kwargs['n_components'], 2)
            next(b for b in app.button if b.label == 'Compute hierarchical clustering').click().run()
            self.assertFalse(app.exception)
            before = app.session_state['hierarchical_subcluster_signature']
            next(b for b in app.button if b.label == 'Compute semi-supervised UMAP').click().run()
            self.assertFalse(app.exception)
            self.assertEqual(semi.call_args.kwargs['n_components'], 2)
            app.toggle(key='semi_umap_3d').set_value(True).run()
            self.assertFalse(app.exception)
            self.assertEqual(semi.call_args.kwargs['n_components'], 3)
            app.toggle(key='semi_umap_3d').set_value(False).run()
            self.assertFalse(app.exception)
            self.assertEqual(semi.call_args.kwargs['n_components'], 2)
            self.assertNotIn('semi_umap_3', app.session_state['semisupervised_umap_df'])
            app.toggle(key='umap_3d').set_value(True).run()
            self.assertFalse(app.exception)
            self.assertEqual(fit.call_args.kwargs['n_components'], 3)
            self.assertEqual(app.session_state['hierarchical_subcluster_signature'], before)
            self.assertEqual(hierarchy.call_count, 1)
            app.toggle(key='umap_3d').set_value(False).run()
            self.assertFalse(app.exception)
            self.assertEqual(fit.call_args.kwargs['n_components'], 2)
            self.assertEqual(fit.call_count, 3)
            self.assertEqual(hierarchy.call_count, 1)


if __name__ == '__main__':
    unittest.main()
