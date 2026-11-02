"""Opt-in installation and end-to-end execution; no private catalogues required."""
import ast
import base64
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from test_canonical_score_learning_notebook import SOURCE, notebook_cells


@unittest.skipUnless(os.getenv('ESCOPE_TEST_ISOLATED_KERNEL') == '1',
                     'Opt-in test installs the isolated notebook environment.')
class IsolatedNotebookIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = next(cell['source'] for cell in notebook_cells()
                      if 'def create_notebook_environment' in cell['source'])
        nodes = [node for node in ast.parse(source).body
                 if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))]
        cls.runtime = types.ModuleType('notebook_runtime_integration')
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), cls.runtime.__dict__)
        cls.python = cls.runtime.create_notebook_environment(root=os.getenv('ESCOPE_TEST_RUNTIME_ROOT'))
        print('ISOLATED_PYTHON=' + str(cls.python), flush=True)
        cls.session = cls.runtime.IsolatedNotebookSession(cls.python)
        cls.outputs = []
        cls.scientific_cells = [cell['source'] for cell in notebook_cells() if cell['isolated']]
        cls.execute(cls.scientific_cells[0])

    @classmethod
    def tearDownClass(cls):
        cls.session.close()

    @classmethod
    def execute(cls, code):
        cls.outputs.clear()
        return cls.session.execute(code, timeout=600, output_hook=cls.outputs.append)

    def test_cells_share_variables_and_preserve_large_identifiers(self):
        self.execute("shared_object_id = '2700244936644435563'")
        self.execute("print(shared_object_id); display(pd.DataFrame({'object_id': [shared_object_id]}))")
        self.assertTrue(any('2700244936644435563' in message['content'].get('text', '')
                            for message in self.outputs))
        self.assertTrue(any('2700244936644435563' in message['content'].get('data', {}).get('text/html', '')
                            for message in self.outputs))

    def test_broken_numpy_in_parent_cannot_change_isolated_imports(self):
        broken = types.SimpleNamespace(__version__='2.4.6')
        with patch.dict(sys.modules, {'numpy': broken}):
            self.execute("assert np.__version__ == '2.2.6'; assert pd.__version__ == '2.2.3'")
            self.assertIs(sys.modules['numpy'], broken)
        self.assertNotIn('numpy', sys.modules)

    def test_technical_smoke_cell_really_trains_reads_parquet_and_renders_png(self):
        self.execute(self.scientific_cells[1])
        self.assertTrue(any('PRUEBA TECNICA: OK' in message['content'].get('text', '')
                            for message in self.outputs))
        pngs = [message['content'].get('data', {}).get('image/png') for message in self.outputs
                if message['content'].get('data', {}).get('image/png')]
        self.assertTrue(pngs)
        png = base64.b64decode(pngs[0])
        self.assertEqual(png[:8], b'\x89PNG\r\n\x1a\n')
        self.assertTrue(all(value > 100 for value in struct.unpack('>II', png[16:24])))

    def test_native_cell_magic_redisplays_tables_and_plots_in_parent_notebook(self):
        from IPython.core.interactiveshell import InteractiveShell
        from IPython.utils.capture import capture_output

        shell = InteractiveShell.instance()
        self.runtime._escope_session = self.session
        shell.register_magic_function(self.runtime._escope_magic, magic_kind='cell', magic_name='escope')
        with capture_output() as captured:
            result = shell.run_cell("%%escope\ndisplay(pd.DataFrame({'object_id': ['2700244936644435563']}))\n"
                                    "plt.figure(); plt.plot([0, 1]); plt.show()")
        self.assertIsNone(result.error_in_exec)
        self.assertIsNone(result.error_before_exec)
        self.assertTrue(any('text/html' in output.data for output in captured.outputs))
        self.assertTrue(any('image/png' in output.data for output in captured.outputs))

    def test_real_virtualenv_fallback_when_system_ensurepip_is_missing(self):
        check_call = subprocess.check_call

        def missing_ensurepip(command, **kwargs):
            if command[2:4] == ['-m', 'venv']:
                raise subprocess.CalledProcessError(1, command)
            return check_call(command, **kwargs)

        with tempfile.TemporaryDirectory() as root, \
                patch.object(self.runtime.subprocess, 'check_call', side_effect=missing_ensurepip):
            python = self.runtime.create_notebook_environment(root)
            check_call([str(python), '-I', '-c',
                        'import numpy, torch; assert numpy.__version__ == "2.2.6"; '
                        'assert torch.tensor([1.]).numpy().tolist() == [1.]'])

    def test_worker_error_is_visible_and_next_cell_can_run(self):
        with self.assertRaisesRegex(RuntimeError, 'ValueError: deliberate test failure'):
            self.execute("raise ValueError('deliberate test failure')")
        self.assertTrue(any(message['header']['msg_type'] == 'error' for message in self.outputs))
        self.execute('assert 2 + 2 == 4')

    def test_full_notebook_with_synthetic_cutouts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.execute(f"""
ROOT = Path({str(root)!r})
(ROOT / 'morphology_catalogue').mkdir()
(ROOT / 'strong_lensing_catalogue').mkdir()
rng = np.random.default_rng(17)
rows, references, physical = [], [], []
for index in range(480):
    tile, object_id = str(100 + index // 24), str(2700244936644435000 + index)
    grade = ('A', 'B', 'C', 'Unknown', 'Unknown', 'Unknown')[index % 6]
    row = {{'id_str': f'Q1_R1_{{tile}}_{{object_id}}', 'object_id': object_id,
           'segmentation_area': 120 + index}}
    rows.append(row)
    physical.append({{'object_id': object_id, 'phz_median': .04, 'phz_flags': 0}})
    if grade != 'Unknown':
        references.append({{'object_id': object_id, 'grade': grade}})
    yy, xx = np.mgrid[:160, :160]
    gray = np.clip(15 + 130 * np.exp(-((xx - 80)**2 + (yy - 80)**2) / 162)
                   + rng.normal(0, 2, xx.shape), 0, 255).astype('uint8')
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    if grade != 'Unknown' or index % 10 == 0:
        cv2.ellipse(rgb, (80, 80), (40 + index % 7, 40 + index % 7), 0,
                    10 + index % 30, 120 + index % 40, (225, 225, 225), 3)
    path = ROOT / 'morphology_catalogue/cutouts_jpg_gz_arcsinh_vis_only' / tile
    path.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path / f'{{tile}}_{{object_id}}_gz_arcsinh_vis_only.jpg')
pd.DataFrame(rows).to_parquet(ROOT / 'morphology_catalogue/morphology_catalogue.parquet')
pd.DataFrame(physical).to_parquet(ROOT / 'morphology_catalogue/useful_physical_measurements.parquet')
pd.DataFrame(references).to_csv(ROOT / 'strong_lensing_catalogue/q1_discovery_engine_lens_catalog.csv', index=False)
""")
            configuration = ast.parse(self.scientific_cells[2])
            overrides = {
                ('ROOT',): f'ROOT = Path({str(root)!r})',
                ('N_UNKNOWN', 'N_NEAR_CONTROLS', 'NEAR_Z_MAX'): 'N_UNKNOWN, N_NEAR_CONTROLS, NEAR_Z_MAX = 120, 20, .1',
                ('PU_PRIORS',): 'PU_PRIORS = (.005,)',
                ('FIT_SEEDS',): 'FIT_SEEDS = (SEED,)',
                ('EPOCHS', 'LEARNING_RATE', 'PATIENCE'): 'EPOCHS, LEARNING_RATE, PATIENCE = 20, .02, 4',
                ('RECALL_FRACTION', 'BOOTSTRAPS'): 'RECALL_FRACTION, BOOTSTRAPS = .05, 30',
            }
            for i, node in enumerate(configuration.body):
                if not isinstance(node, ast.Assign):
                    continue
                target = node.targets[0]
                names = (target.id,) if isinstance(target, ast.Name) else tuple(
                    element.id for element in target.elts) if isinstance(target, ast.Tuple) else ()
                if names in overrides:
                    configuration.body[i] = ast.parse(overrides[names]).body[0]
            self.execute(ast.unparse(configuration))
            for index, cell in enumerate(self.scientific_cells[3:], 3):
                print(f'Executing scientific notebook cell {index + 1}/{len(self.scientific_cells)}', flush=True)
                self.execute(cell)
            output = root / 'morphology_catalogue/escope_canonical_score_learning'
            expected = ('manifest.json', 'all_predictions.csv', 'heldout_reference_metrics.csv',
                        'selected_weights.pt', 'selected_weights.json', 'region_measurements.csv',
                        'manual_vs_learned_weights.png', 'blind_visual_audit.csv')
            for filename in expected:
                self.assertTrue((output / filename).is_file(), filename)
            import json
            manifest = json.loads((output / 'manifest.json').read_text())
            self.assertFalse(manifest['unknown_are_negatives'])
            self.assertFalse(manifest['nearby_are_negatives'])
            self.assertEqual(manifest['versions']['numpy'], '2.2.6')
            self.assertEqual(manifest['versions']['pandas'], '2.2.3')


if __name__ == '__main__':
    unittest.main()
