"""Zero-install notebook contracts and opt-in, offline end-to-end execution."""
import ast
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / 'notebooks'


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load_module('canonical_batch_builder', NOTEBOOKS / 'build_canonical_score_batch_notebook.py')
worker = load_module('canonical_batch_worker', NOTEBOOKS / 'canonical_score_batch_worker.py')


def code_cells():
    return [''.join(cell['source']) for cell in builder.make_notebook()['cells'] if cell['cell_type'] == 'code']


class BatchNotebookTests(unittest.TestCase):
    def test_committed_notebook_matches_builder_and_all_cells_compile(self):
        self.assertEqual(json.loads(builder.TARGET.read_text()), builder.make_notebook())
        for index, source in enumerate(code_cells()):
            compile(source, f'cell-{index}', 'exec')
        for cell in builder.make_notebook()['cells']:
            self.assertIn('id', cell)
            if cell['cell_type'] == 'code':
                self.assertIsNone(cell['execution_count'])
                self.assertEqual(cell['outputs'], [])

    def test_no_installers_custom_magics_or_second_kernel(self):
        sources = [*code_cells(), (NOTEBOOKS / 'canonical_score_batch_worker.py').read_text()]
        sources.extend(builder.scientific_cells())
        for source in sources:
            self.assertNotRegex(source, r'\b(?:pip|ensurepip|venv|virtualenv|jupyter_client|KernelManager)\b')
            self.assertNotIn('%%escope', source)
        self.assertIn('subprocess.Popen', code_cells()[0])
        self.assertIn('PACKAGES_BEFORE', code_cells()[0])
        self.assertIn('un fallo de datos no requiere reiniciar Colab', code_cells()[0])
        self.assertIn('packages_unchanged', sources[len(code_cells())])

    def test_bundle_is_self_contained_and_detector_is_frozen(self):
        import hashlib
        with zipfile.ZipFile(io.BytesIO(builder.bundle_bytes())) as archive:
            self.assertEqual(set(archive.namelist()), {'batch_worker.py', 'science_cells.json', 'frozen_detector.py'})
            self.assertEqual(hashlib.sha256(archive.read('frozen_detector.py')).hexdigest(),
                             'b45f1d4e8866aa5ad2805ea6205c1a138abf4da3459ecd116b44511a822600a4')
            self.assertEqual(json.loads(archive.read('science_cells.json')), builder.scientific_cells())

    def test_scientific_functions_and_model_are_not_reimplemented(self):
        for source in builder.scientific_cells():
            before = ast.parse(source)
            after = worker.scientific_tree(source)
            # Only the top-level inline-matplotlib magic and display import are removed.
            worker.scientific_source(source)
            self.assertEqual([ast.dump(n) for n in before.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))],
                             [ast.dump(n) for n in after.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))])

    def test_configuration_does_not_execute_user_path_as_code(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            morph, lens, cutouts = root / "morph'); raise RuntimeError('oops", root / 'lenses.csv', root / 'cutouts'
            morph.touch()
            lens.touch()
            cutouts.mkdir()
            config = {'morph_path': str(morph), 'lens_path': str(lens), 'cutout_base': str(cutouts), 'n_near': 0}
            overrides = worker.configuration(config, root, root / 'output')
            namespace = {'Path': Path}
            exec(worker.scientific_source('MORPH_PATH = None', overrides), namespace)
            self.assertEqual(namespace['MORPH_PATH'], morph)
            with self.assertRaisesRegex(FileNotFoundError, 'no es un error de librerias'):
                worker.configuration({**config, 'n_near': 1}, root, root / 'output')

    def test_real_data_is_opt_in_and_no_drive_access_by_default(self):
        cells = code_cells()
        config = next(source for source in cells if 'RUN_REAL_DATA = False' in source)
        namespace = {'__name__': '__test__'}
        exec(config, namespace)
        self.assertFalse(namespace['RUN_REAL_DATA'])
        self.assertEqual(namespace['CONFIG']['n_near'], 0)
        self.assertIn('if RUN_REAL_DATA:', cells[-1])


@unittest.skipUnless(os.getenv('ESCOPE_TEST_BATCH') == '1', 'Uses existing science packages; never installs them.')
class BatchOfflineIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.bundle = cls.root / 'bundle'
        cls.bundle.mkdir()
        with zipfile.ZipFile(io.BytesIO(builder.bundle_bytes())) as archive:
            archive.extractall(cls.bundle)
        cls.demo = cls.execute_worker('demo')

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @classmethod
    def execute_worker(cls, mode, config=None, expected=0):
        import uuid
        output = cls.root / (mode + '-' + uuid.uuid4().hex[:6])
        output.mkdir()
        config_file = output / 'input_config.json'
        config_file.write_text(json.dumps(config or {}))
        result = subprocess.run([sys.executable, '-B', '-u', str(cls.bundle / 'batch_worker.py'),
                                 '--mode', mode, '--output', str(output), '--config', str(config_file)],
                                capture_output=True, text=True, timeout=300)
        print(result.stdout, flush=True)
        if result.returncode != expected:
            raise AssertionError(result.stdout + '\n' + result.stderr)
        payload = json.loads((output / 'result.json').read_text())
        if not payload['packages_unchanged']:
            raise AssertionError('Installed packages changed')
        return output

    def test_demo_runs_the_whole_experiment_with_reports_and_no_package_changes(self):
        output = self.demo
        expected = ('manifest.json', 'all_predictions.csv', 'heldout_reference_metrics.csv',
                    'selected_weights.pt', 'selected_weights.json', 'region_measurements.csv',
                    'manual_vs_learned_weights.png', 'blind_visual_audit.csv',
                    'reference_catalogue_audit.csv', 'reference_catalogue_report.json')
        for filename in expected:
            self.assertTrue((output / filename).is_file(), filename)
        manifest = json.loads((output / 'manifest.json').read_text())
        self.assertFalse(manifest['unknown_are_negatives'])
        self.assertFalse(manifest['nearby_are_negatives'])
        self.assertEqual(json.loads((output / 'packages_before.json').read_text()),
                         json.loads((output / 'packages_after.json').read_text()))
        report = json.loads((output / 'report/index.json').read_text())
        self.assertGreaterEqual(sum(row['kind'] == 'png' for row in report), 6)
        self.assertGreaterEqual(sum(row['kind'] == 'html' for row in report), 7)
        for row in report:
            self.assertGreater((output / 'report' / row['path']).stat().st_size, 100)

    def test_real_data_code_path_on_fixture_with_no_physical_catalogue(self):
        import pandas as pd
        data = self.demo / 'synthetic_data'
        references = pd.read_csv(data / 'strong_lensing_catalogue/q1_discovery_engine_lens_catalog.csv',
                                 dtype={'object_id': 'string'})
        morphology = pd.read_parquet(data / 'morphology_catalogue/morphology_catalogue.parquet')
        references = references.merge(morphology[['object_id', 'id_str']], on='object_id', how='left')
        references.loc[0, 'object_id'] = pd.NA
        references = pd.concat([references, pd.DataFrame({'object_id': [None, None],
            'id_str': ['12345', 'SAS_OTF_RELEASE_100_NEG99999'], 'grade': ['A', 'C']})], ignore_index=True)
        lens_path = self.root / 'reference_with_missing_ids.csv'
        references.to_csv(lens_path, index=False)
        output = self.execute_worker('real', {'root': str(data), 'n_unknown': 120, 'n_near': 0,
                                             'lens_path': str(lens_path),
                                             'physical_path': str(data / 'not-needed.parquet'),
                                             'priors': [.005], 'fit_seeds': [20261031],
                                             'epochs': 10, 'bootstraps': 20})
        manifest = json.loads((output / 'manifest.json').read_text())
        self.assertEqual(manifest['n_extra_near'], 0)
        report = manifest['reference_catalogue_report']
        self.assertEqual(report['blank_object_id_rows'], 3)
        self.assertEqual(report['recovered_from_id_str'], 1)
        self.assertEqual(report['unresolved_reference_rows'], 2)
        self.assertEqual(report['matched_unique_objects'], 240)
        self.assertFalse(json.loads((output / 'result.json').read_text())['synthetic_only'])

    def test_missing_data_error_is_persisted_without_installing_anything(self):
        output = self.execute_worker('real', {'root': str(self.root / 'missing')}, expected=1)
        result = json.loads((output / 'result.json').read_text())
        self.assertIn('FileNotFoundError', result['error'])
        self.assertIn('no es un error de librerias', result['error'])
        self.assertTrue(result['packages_unchanged'])

    def test_generated_notebook_executes_without_numeric_imports_in_parent(self):
        from IPython.display import HTML, Image
        namespace = {'__name__': '__notebook_test__'}
        emitted = []
        broken_parent_numpy = types.SimpleNamespace(__version__='bad-parent')
        with patch.dict(sys.modules, {'numpy': broken_parent_numpy}):
            exec(code_cells()[0], namespace)
            namespace['display'] = emitted.append
            exec(code_cells()[1], namespace)
            self.assertIs(sys.modules['numpy'], broken_parent_numpy)
            self.assertTrue(any(isinstance(item, Image) for item in emitted))
            emitted.clear()
            # Verify the complete demonstration through the actual notebook runner as well.
            exec(code_cells()[2], namespace)
            self.assertTrue(any(isinstance(item, HTML) for item in emitted))
            self.assertTrue(any(isinstance(item, Image) for item in emitted))
            exec(code_cells()[3], namespace)
            exec(code_cells()[4], namespace)
        import shutil
        shutil.rmtree(namespace['WORKSPACE'])


if __name__ == '__main__':
    unittest.main()
