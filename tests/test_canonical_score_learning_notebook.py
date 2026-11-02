import ast
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


SOURCE = Path(__file__).resolve().parents[1] / 'notebooks/escope_canonical_score_learning_colab.py'
HAS_TORCH = importlib.util.find_spec('torch') is not None


def notebook_cells():
    cells = []
    current = None
    for line in SOURCE.read_text().splitlines(keepends=True):
        if line.startswith('# %%'):
            current = {'cell_type': 'markdown' if '[markdown]' in line else 'code',
                       'isolated': '[escope]' in line, 'source': ''}
            cells.append(current)
        elif current is not None:
            if current['cell_type'] == 'markdown':
                line = line[2:] if line.startswith('# ') else line[1:] if line.startswith('#') else line
            current['source'] += line
    return cells


def notebook_helpers():
    constants = {'FEATURE_NAMES', 'INITIAL_REGION_WEIGHTS', 'INITIAL_POOL_WEIGHTS',
                 'DETECTOR_COMMIT', 'DETECTOR_SHA256'}
    nodes = [node for node in ast.parse(SOURCE.read_text()).body
             if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))
             or (isinstance(node, ast.Assign) and any(
                 isinstance(target, ast.Name) and target.id in constants for target in node.targets))]
    module = types.ModuleType('canonical_learning_notebook_helpers')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), module.__dict__)
    return module


class NotebookSourceTests(unittest.TestCase):
    def test_python_and_notebook_are_synchronized_and_not_executed(self):
        notebook = json.loads(SOURCE.with_suffix('.ipynb').read_text())
        expected = notebook_cells()
        self.assertEqual(len(notebook['cells']), len(expected))
        for cell, actual in zip(expected, notebook['cells']):
            self.assertEqual(cell['cell_type'], actual['cell_type'])
            expected_source = ('%%escope\n' if cell['isolated'] else '') + cell['source']
            self.assertEqual(expected_source, ''.join(actual['source']))
            if cell['cell_type'] == 'code':
                compile(cell['source'], str(SOURCE), 'exec')
                self.assertIsNone(actual['execution_count'])
                self.assertEqual(actual['outputs'], [])

    def test_unknown_and_nearby_are_not_declared_negative(self):
        source = SOURCE.read_text()
        self.assertIn("'unknown_are_negatives': False", source)
        self.assertIn("'nearby_are_negatives': False", source)
        self.assertIn('priors_are_hypotheses', source)
        self.assertIn('manual_float_score', source)
        self.assertIn('no precision', source)
        self.assertIn('no es un umbral fisico', source)


class RuntimeSetupTests(unittest.TestCase):
    def setUp(self):
        source = next(cell['source'] for cell in notebook_cells()
                      if 'def create_notebook_environment' in cell['source'])
        nodes = [node for node in ast.parse(source).body
                 if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))]
        self.nb = types.ModuleType('runtime_setup')
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), self.nb.__dict__)

    def test_all_numeric_cells_use_the_isolated_kernel(self):
        numerical = {'numpy', 'pandas', 'torch', 'cv2', 'matplotlib', 'sklearn', 'pyarrow'}
        for cell in notebook_cells():
            if cell['cell_type'] != 'code':
                continue
            for node in ast.walk(ast.parse(cell['source'])):
                names = [alias.name.split('.')[0] for alias in node.names] if isinstance(node, ast.Import) else (
                    [node.module.split('.')[0]] if isinstance(node, ast.ImportFrom) and node.module else [])
                if numerical.intersection(names):
                    self.assertTrue(cell['isolated'])
        self.assertNotIn('def runtime_dependency_plan', SOURCE.read_text())

    def test_only_private_environment_receives_numeric_installs(self):
        python = '/content/private-env/bin/python'
        commands = self.nb.notebook_install_commands(python, linux=True)
        self.assertEqual(len(commands), 2)
        for command in commands:
            self.assertEqual(command[:5], [python, '-I', '-m', 'pip', '--isolated'])
        self.assertIn('https://download.pytorch.org/whl/cpu', commands[0])
        self.assertIn('--no-deps', commands[0])
        self.assertIn('numpy==2.2.6', commands[1])
        self.assertIn('pandas==2.2.3', commands[1])
        self.assertIn('torch==2.7.1', commands[1])
        self.assertEqual(len(self.nb.notebook_install_commands(python, linux=False)), 1)

    def test_environment_removes_injected_package_paths(self):
        with patch.dict(os.environ, {'PYTHONPATH': '/broken/packages', 'PYTHONHOME': '/wrong',
                                    'PIP_TARGET': '/global/site-packages', 'PIP_USER': '1',
                                    'VIRTUAL_ENV': '/parent-env', 'KEEP_ME': 'unchanged'}):
            environment = self.nb.isolated_environment()
        for name in ('PYTHONPATH', 'PYTHONHOME', 'PIP_TARGET', 'PIP_USER', 'VIRTUAL_ENV'):
            self.assertNotIn(name, environment)
        self.assertEqual(environment['PYTHONNOUSERSITE'], '1')
        self.assertEqual(environment['KEEP_ME'], 'unchanged')

    def test_broken_parent_numpy_is_ignored_and_ready_environment_is_reused(self):
        calls = []

        def run(command, **kwargs):
            calls.append(command)
            if command[2:4] == ['-m', 'venv']:
                Path(command[-1]).mkdir(parents=True)

        with tempfile.TemporaryDirectory() as root, \
                patch.dict(sys.modules, {'numpy': types.SimpleNamespace(__version__='2.4.6')}), \
                patch.object(self.nb.subprocess, 'check_call', side_effect=run), \
                patch.object(self.nb.subprocess, 'check_output', return_value='{"numpy": "2.2.6"}') as probe:
            python = self.nb.create_notebook_environment(root, '/parent/python')
            self.assertTrue((python.parent.parent / 'escope-ready.json').exists())
            first_count = len(calls)
            self.assertEqual(self.nb.create_notebook_environment(root, '/parent/python'), python)
            self.assertEqual(calls[first_count:], [[str(python), '-I', '-m', 'pip', '--isolated', 'check']])
            self.assertEqual(sys.modules['numpy'].__version__, '2.4.6')
            self.assertIn('assert sys.prefix != sys.base_prefix', probe.call_args.args[0][-1])
        for command in calls:
            if 'install' in command:
                self.assertEqual(command[0], str(python))

    def test_colab_without_ensurepip_bootstraps_virtualenv_without_global_install(self):
        calls = []

        def run(command, **kwargs):
            calls.append(command)
            if command[2:4] == ['-m', 'venv']:
                Path(command[-1]).mkdir(parents=True)
                raise self.nb.subprocess.CalledProcessError(1, command)

        with tempfile.TemporaryDirectory() as root, \
                patch.object(self.nb.subprocess, 'check_call', side_effect=run), \
                patch.object(self.nb.subprocess, 'check_output', return_value='{}'):
            self.nb.create_notebook_environment(root, '/parent/python')
        bootstrap = next(command for command in calls if 'virtualenv==20.35.4' in command)
        self.assertIn('--target', bootstrap)
        self.assertIn('--ignore-installed', bootstrap)
        self.assertTrue(any('from virtualenv import cli_run' in command[-1] for command in calls))

    def test_failed_import_probe_never_marks_environment_ready(self):
        def run(command, **kwargs):
            if command[2:4] == ['-m', 'venv']:
                Path(command[-1]).mkdir(parents=True)

        with tempfile.TemporaryDirectory() as root, \
                patch.object(self.nb.subprocess, 'check_call', side_effect=run), \
                patch.object(self.nb.subprocess, 'check_output', side_effect=RuntimeError('compiled import failed')):
            with self.assertRaisesRegex(RuntimeError, 'compiled import failed'):
                self.nb.create_notebook_environment(root, '/parent/python')
            self.assertEqual(list(Path(root).rglob('escope-ready.json')), [])

    def session(self, reply):
        session = object.__new__(self.nb.IsolatedNotebookSession)
        session.manager = Mock()
        session.manager.is_alive.return_value = True
        session.client = Mock()
        session.client.execute_interactive.return_value = {'content': reply}
        return session

    def test_execution_forwards_rich_outputs_and_propagates_worker_errors(self):
        session = self.session({'status': 'ok'})
        hook = Mock()
        session.execute('display(table); plt.show()', output_hook=hook)
        self.assertFalse(session.client.execute_interactive.call_args.kwargs['allow_stdin'])
        self.assertIs(session.client.execute_interactive.call_args.kwargs['output_hook'], hook)
        session = self.session({'status': 'error', 'ename': 'ValueError', 'evalue': 'missing data'})
        with self.assertRaisesRegex(RuntimeError, 'ValueError: missing data'):
            session.execute('bad()')

    def test_rich_output_forwarding_does_not_depend_on_private_kernel_sockets(self):
        display = types.ModuleType('IPython.display')
        display.display, display.update_display, display.clear_output = Mock(), Mock(), Mock()
        data = {'text/html': '<table><tr><td>2700244936644435563</td></tr></table>'}
        with patch.dict(sys.modules, {'IPython.display': display}):
            self.nb.forward_notebook_output({'header': {'msg_type': 'display_data'},
                                            'content': {'data': data, 'metadata': {}}})
            self.nb.forward_notebook_output({'header': {'msg_type': 'clear_output'},
                                            'content': {'wait': True}})
        display.display.assert_called_once_with(data, metadata={}, raw=True)
        display.clear_output.assert_called_once_with(wait=True)

    def test_interrupt_stops_worker_and_restarting_first_cell_closes_previous_worker(self):
        session = self.session({'status': 'ok'})
        session.client.execute_interactive.side_effect = KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            session.execute('long_calculation()')
        session.manager.interrupt_kernel.assert_called_once()
        self.nb._escope_session = session
        with patch.object(self.nb, 'create_notebook_environment', return_value='/env/bin/python'), \
                patch.object(self.nb, 'IsolatedNotebookSession') as constructor:
            self.nb.prepare_notebook_runtime()
        session.client.stop_channels.assert_called_once()
        constructor.assert_called_once_with('/env/bin/python')


@unittest.skipUnless(HAS_TORCH, 'Optional notebook tests require torch; not a production dependency.')
class CanonicalLearningNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nb = notebook_helpers()
        cls.np, cls.pd, cls.torch = cls.nb.np, cls.nb.pd, cls.nb.torch
        cls.torch.set_num_threads(1)

    def test_identifiers_are_exact_and_float_ids_are_rejected(self):
        ids = ['2700244936644435563', '-610719047485205149']
        self.assertEqual(self.nb.exact_ids(ids).tolist(), ids)
        for bad in ([float(ids[0])], ['nan'], [None]):
            with self.assertRaises(ValueError):
                self.nb.exact_ids(bad)

    def test_catalogue_join_and_photoz_flags_do_not_create_negatives(self):
        pd = self.pd
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            ids = ['2700244936644435563', '-610719047485205149', '1', '2']
            pd.DataFrame({'id_str': ['Q1_R1_100_' + i for i in ids], 'object_id': ids}).to_parquet(folder / 'm.parquet')
            pd.DataFrame({'object_id': ids[:2], 'grade': ['A', 'C']}).to_csv(folder / 'l.csv', index=False)
            pd.DataFrame({'object_id': ids, 'phz_median': [.02, .01, .03, -.1],
                          'phz_flags': [0, 1, 0, 0]}).to_parquet(folder / 'p.parquet')
            frame = self.nb.load_catalogue(folder / 'm.parquet', folder / 'l.csv', folder / 'p.parquet')
        self.assertEqual(frame.object_id.tolist(), ids)
        self.assertEqual(frame.grade.tolist(), ['A', 'C', 'Unknown', 'Unknown'])
        self.assertTrue(self.np.isnan(frame.valid_phz.iloc[1]))
        self.assertTrue(self.np.isnan(frame.valid_phz.iloc[3]))
        self.assertEqual(frame.valid_phz.iloc[0], .02)

    def test_nearby_controls_are_audit_only_and_c_are_not_unknown(self):
        frame = self.pd.DataFrame({'object_id': [str(i) for i in range(200)],
                                  'grade': ['A', 'B', 'C'] + ['Unknown'] * 197,
                                  'valid_phz': .02, 'tile': '1', 'id_str': 'unused'})
        first = self.nb.sample_cases(frame, 50, 20, .1, 7)
        second = self.nb.sample_cases(frame, 50, 0, .1, 7)
        self.assertEqual(first.loc[first.primary_pool, 'object_id'].tolist(), second.object_id.tolist())
        self.assertEqual(first.loc[~first.primary_pool, 'grade'].unique().tolist(), ['Unknown'])
        self.assertEqual(first.loc[first.grade.eq('C'), 'primary_pool'].tolist(), [True])
        self.assertTrue(first.object_id.is_unique)
        with self.assertRaises(ValueError):
            self.nb.sample_cases(frame.assign(valid_phz=self.np.nan), 50, 20, .1, 7)

    def grouped_fixture(self):
        count = 600
        return self.pd.DataFrame({'object_id': [str(i) for i in range(count)],
            'grade': self.np.tile(['A', 'B', 'C', 'Unknown', 'Unknown', 'Unknown'], 100),
            'tile': [str(i // 30) for i in range(count)],
            'image_sha256': ['h' + str(i) for i in range(count)],
            'valid_image': True, 'primary_pool': True})

    def test_splits_keep_tiles_and_duplicate_images_together(self):
        frame = self.grouped_fixture()
        frame.loc[31, 'image_sha256'] = frame.loc[0, 'image_sha256']
        split, groups = self.nb.split_groups(frame, 17)
        other, _ = self.nb.split_groups(frame, 17)
        self.np.testing.assert_array_equal(split, other)
        frame['split'], frame['group'] = split, groups
        self.assertEqual(frame.groupby('tile')['split'].nunique().max(), 1)
        self.assertEqual(frame.groupby('image_sha256')['split'].nunique().max(), 1)
        self.assertEqual(groups[0], groups[31])
        self.assertEqual(set(split), {'train', 'validation', 'test'})
        with self.assertRaises(ValueError):
            self.nb.split_groups(frame.assign(tile='only-one'), 17)

    def test_network_starts_at_manual_weights_and_is_region_order_invariant(self):
        torch = self.torch
        model = self.nb.InterpretableScoreNet()
        x = torch.rand(8, 6, 5)
        mask = torch.ones(8, 6)
        mask[:, 3:] = 0
        geometry = torch.linspace(0, 1, 8)
        q = (x * torch.tensor(self.nb.INITIAL_REGION_WEIGHTS)).sum(-1) * mask
        top = q.topk(2, dim=1).values
        expected = .65 * top[:, 0] + .20 * top[:, 1] + .15 * geometry
        torch.testing.assert_close(model.evidence(x, mask, geometry), expected)
        order = torch.tensor([5, 3, 1, 4, 0, 2])
        torch.testing.assert_close(model.evidence(x[:, order], mask[:, order], geometry), expected)
        weights = model.weights()
        self.assertAlmostEqual(sum(weights['region'].values()), 1, places=6)
        self.assertAlmostEqual(sum(weights['aggregation'].values()), 1, places=6)
        self.assertTrue(all(weight >= 0 for level in weights.values() for weight in level.values()))

    def test_frozen_detector_reproduces_baseline_and_marks_invalid_images_missing(self):
        from euclid_umap_explorer import lens_plausibility as detector
        self.assertEqual(self.nb.sha256_file(detector.__file__), self.nb.DETECTOR_SHA256)
        np, cv2 = self.np, self.nb.cv2
        yy, xx = np.mgrid[:160, :160]
        base = np.clip(15 + 130 * np.exp(-((xx - 80) ** 2 + (yy - 80) ** 2) / 162), 0, 255).astype('uint8')
        image = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)
        cv2.ellipse(image, (80, 80), (42, 42), 0, 20, 130, (225, 225, 225), 3)
        result = self.nb.extract_record(image, detector)
        self.assertTrue(result['valid'])
        x, mask = np.zeros((1, 6, 5), dtype='float32'), np.zeros((1, 6), dtype='float32')
        n = len(result['features'])
        x[0, :n] = np.asarray(result['features']).reshape(n, 5)
        mask[0, :n] = 1
        score = self.nb.predict_scores(self.nb.InterpretableScoreNet(), x, mask,
                                      np.array([result['pair_agreement']], dtype='float32'), [True])[0]
        self.assertLessEqual(abs(score - result['score']), .5001)
        invalid = self.nb.extract_record(np.zeros((30, 30, 3), dtype='uint8'), detector)
        self.assertFalse(invalid['valid'])
        self.assertIsNone(invalid['score'])
        self.assertTrue(np.isnan(self.nb.predict_scores(self.nb.InterpretableScoreNet(), x, mask,
                                                        np.array([0.], dtype='float32'), [False])[0]))

    def test_nnpu_matches_formula_and_corrective_branch_has_gradients(self):
        torch, F = self.torch, self.nb.F
        p = torch.tensor([-1., 1.], requires_grad=True)
        u = torch.tensor([-2., 2.], requires_grad=True)
        update, objective = self.nb.nnpu_loss(p, u, .2)
        expected = .2 * F.softplus(-p).mean() + (F.softplus(u).mean() - .2 * F.softplus(p).mean()).clamp(min=0)
        torch.testing.assert_close(objective, expected)
        update.backward()
        self.assertTrue(torch.isfinite(p.grad).all())
        p, u = torch.tensor([10.], requires_grad=True), torch.tensor([-10.], requires_grad=True)
        update, objective = self.nb.nnpu_loss(p, u, .5)
        self.assertGreater(update.item(), objective.item())
        update.backward()
        self.assertTrue(torch.isfinite(u.grad).all())
        with self.assertRaises(ValueError):
            self.nb.nnpu_loss(p, u, 0.)

    def test_tied_scores_have_expected_recovery_and_c_is_not_a_negative(self):
        result = self.nb.reference_metrics(['A', 'Unknown', 'Unknown', 'C'], [0., 0., 0., 100.], ('A',), .2)
        self.assertAlmostEqual(result['reference_recall'], 1 / 3)
        self.assertAlmostEqual(result['reference_ap'], 1 / 3)
        self.assertEqual(result['review_k'], 1)

    def test_fit_updates_weights_without_access_to_test_or_extra_controls(self):
        np, pd = self.np, self.pd
        count = 180
        positive = np.arange(count) % 3 == 0
        x = np.full((count, 6, 5), 0., dtype='float32')
        x[:, 0, :] = np.where(positive[:, None], .2, .8)
        x[:, 0, 0] = np.where(positive, .9, .1)
        mask = np.zeros((count, 6), dtype='float32')
        mask[:, 0] = 1
        geometry = np.zeros(count, dtype='float32')
        frame = pd.DataFrame({'grade': np.where(positive, 'A', 'Unknown'), 'valid_image': True,
            'primary_pool': True, 'split': ['train'] * 100 + ['validation'] * 40 + ['test'] * 40})
        frame.loc[90:99, 'primary_pool'] = False
        original = self.nb.predict_scores(self.nb.InterpretableScoreNet(), x, mask, geometry, frame.valid_image)
        validation = frame.split.eq('validation')
        baseline = self.nb.reference_metrics(frame.loc[validation, 'grade'], original[validation], ('A',))['reference_ap']
        model, history, metric = self.nb.fit_weights(x, mask, geometry, frame, .2, 9, ('A',),
                                                    epochs=120, patience=12)
        self.assertGreater(metric, baseline)
        self.assertGreater(model.weights()['region']['tangential'], .3)
        self.assertFalse(history.empty)
        changed = frame.copy()
        changed.loc[frame.split.eq('test'), 'grade'] = 'Unknown'
        changed.loc[~frame.primary_pool, 'grade'] = 'Unknown'
        other_x = x.copy()
        other_x[frame.split.eq('test')] = 1
        other_x[~frame.primary_pool] = 1
        other, _, _ = self.nb.fit_weights(other_x, mask, geometry, changed, .2, 9, ('A',),
                                          epochs=120, patience=12)
        for key, value in model.state_dict().items():
            self.torch.testing.assert_close(value, other.state_dict()[key], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
