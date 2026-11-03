"""Offline batch adapter for the canonical-score experiment. Never installs packages."""
import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import traceback


def package_snapshot():
    return sorted((item.metadata['Name'] or '', item.version, str(item.locate_file('')))
                  for item in importlib.metadata.distributions())


def scientific_tree(source, overrides=None):
    """Adapt notebook presentation/configuration, leaving scientific functions unchanged."""
    tree = ast.parse(source)
    body = []
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            function = node.value.func
            if isinstance(function, ast.Attribute) and function.attr == 'run_line_magic':
                continue
        if isinstance(node, ast.ImportFrom) and node.module == 'IPython.display':
            continue
        if overrides and isinstance(node, ast.Assign):
            target = node.targets[0]
            names = (target.id,) if isinstance(target, ast.Name) else tuple(
                item.id for item in target.elts) if (isinstance(target, ast.Tuple)
                and all(isinstance(item, ast.Name) for item in target.elts)) else ()
            if names in overrides:
                node = ast.Assign(targets=node.targets, value=ast.parse(overrides[names], mode='eval').body)
        body.append(node)
    return ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))


def scientific_source(source, overrides=None):
    return compile(scientific_tree(source, overrides), '<science>', 'exec')


class Report:
    def __init__(self, output):
        self.root = Path(output) / 'report'
        self.root.mkdir(parents=True)
        self.entries = []
        self.phase = ''

    def add(self, kind, path):
        self.entries.append({'kind': kind, 'path': path.name, 'phase': self.phase})
        (self.root / 'index.json').write_text(json.dumps(self.entries, indent=2))

    def display(self, *items):
        for item in items:
            stem = f'{len(self.entries):03d}'
            if hasattr(item, 'to_html'):
                path = self.root / (stem + '.html')
                path.write_text(item.to_html(max_rows=60, escape=True))
                item.to_csv(self.root / (stem + '.csv'))
                self.add('html', path)
            else:
                print(item, flush=True)

    def show(self, pyplot):
        for number in pyplot.get_fignums():
            figure = pyplot.figure(number)
            path = self.root / f'{len(self.entries):03d}.png'
            figure.savefig(path, dpi=120, bbox_inches='tight')
            self.add('png', path)
            pyplot.close(figure)


def bundled_detector(namespace, bundle):
    def load(destination):
        destination = Path(destination)
        content = (bundle / 'frozen_detector.py').read_bytes()
        if hashlib.sha256(content).hexdigest() != namespace['DETECTOR_SHA256']:
            raise ValueError('El detector incluido no coincide con su SHA256.')
        if not destination.exists():
            destination.write_bytes(content)
        if destination.read_bytes() != content:
            raise ValueError('Detector distinto del fijado; usa una carpeta de resultados nueva.')
        spec = namespace['importlib'].util.spec_from_file_location('escope_frozen_detector', destination)
        module = namespace['importlib'].util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return load


def demo_catalogue(namespace, root):
    """Synthetic fixtures exercise I/O and training, not scientific performance."""
    np, pd, cv2, Image = (namespace[name] for name in ('np', 'pd', 'cv2', 'Image'))
    morphology = root / 'morphology_catalogue'
    lenses = root / 'strong_lensing_catalogue'
    morphology.mkdir(parents=True)
    lenses.mkdir()
    rng = np.random.default_rng(17)
    rows, references, physical = [], [], []
    yy, xx = np.mgrid[:160, :160]
    for index in range(480):
        tile, object_id = str(100 + index // 24), str(2700244936644435000 + index)
        grade = ('A', 'B', 'C', 'Unknown', 'Unknown', 'Unknown')[index % 6]
        rows.append({'id_str': f'Q1_R1_{tile}_{object_id}', 'object_id': object_id,
                     'segmentation_area': 120 + index})
        physical.append({'object_id': object_id, 'phz_median': .04, 'phz_flags': 0})
        if grade != 'Unknown':
            references.append({'object_id': object_id, 'grade': grade})
        gray = np.clip(15 + 130 * np.exp(-((xx - 80)**2 + (yy - 80)**2) / 162)
                       + rng.normal(0, 2, xx.shape), 0, 255).astype('uint8')
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        if grade != 'Unknown' or index % 10 == 0:
            cv2.ellipse(rgb, (80, 80), (40 + index % 7, 40 + index % 7), 0,
                        10 + index % 30, 120 + index % 40, (225, 225, 225), 3)
        directory = morphology / 'cutouts_jpg_gz_arcsinh_vis_only' / tile
        directory.mkdir(exist_ok=True, parents=True)
        Image.fromarray(rgb).save(directory / f'{tile}_{object_id}_gz_arcsinh_vis_only.jpg')
    pd.DataFrame(rows).to_parquet(morphology / 'morphology_catalogue.parquet')
    pd.DataFrame(physical).to_parquet(morphology / 'useful_physical_measurements.parquet')
    pd.DataFrame(references).to_csv(lenses / 'q1_discovery_engine_lens_catalog.csv', index=False)


def configuration(config, root, output):
    paths = {
        'ROOT': root,
        'MORPH_PATH': Path(config.get('morph_path') or root / 'morphology_catalogue/morphology_catalogue.parquet'),
        'LENS_PATH': Path(config.get('lens_path') or root / 'strong_lensing_catalogue/q1_discovery_engine_lens_catalog.csv'),
        'PHYSICAL_PATH': Path(config.get('physical_path') or root / 'morphology_catalogue/useful_physical_measurements.parquet'),
        'CUTOUT_BASE': Path(config.get('cutout_base') or root / 'morphology_catalogue/cutouts_jpg_gz_arcsinh_vis_only'),
        'OUTPUT_DIR': output,
    }
    fields = {
        ('SEED',): repr(config.get('seed', 20261031)),
        ('N_UNKNOWN', 'N_NEAR_CONTROLS', 'NEAR_Z_MAX'): repr((config.get('n_unknown', 3000), config.get('n_near', 0), config.get('near_z_max', .1))),
        ('PU_PRIORS',): repr(tuple(config.get('priors', [.001, .005, .01]))),
        ('FIT_SEEDS',): repr(tuple(config.get('fit_seeds', [20261031, 20261032, 20261033]))),
        ('EPOCHS', 'LEARNING_RATE', 'PATIENCE'): repr((config.get('epochs', 400), config.get('learning_rate', .02), config.get('patience', 12))),
        ('RECALL_FRACTION', 'BOOTSTRAPS'): repr((config.get('recall_fraction', .05), config.get('bootstraps', 500))),
    }
    fields.update({(name,): f'Path({str(path)!r})' for name, path in paths.items()})
    required = ['MORPH_PATH', 'LENS_PATH', 'CUTOUT_BASE']
    if config.get('n_near', 0):
        required.append('PHYSICAL_PATH')
    for name in required:
        path = paths[name]
        if not path.exists():
            raise FileNotFoundError(f'Falta {path}. Revisa las rutas en CONFIG; no es un error de librerias.')
    return fields


PHASES = ('Imports', 'Prueba tecnica', 'Catalogos y muestra', 'Regiones y particiones',
          'Modelo nnPU', 'Entrenamiento y seleccion', 'Evaluacion reservada',
          'Pesos e imagenes', 'Exportacion')


def run(mode, config, output, bundle):
    os.environ['MPLBACKEND'] = 'Agg'
    os.environ['MPLCONFIGDIR'] = str(output / 'plot_cache')
    report = Report(output)
    cells = json.loads((bundle / 'science_cells.json').read_text())
    namespace = {'__name__': '__escope_batch__', 'display': report.display}
    report.phase = PHASES[0]
    print('[0/8] Imports de las librerias existentes', flush=True)
    exec(scientific_source(cells[0]), namespace)
    pyplot = namespace['plt']
    namespace['cv2'].setNumThreads(1)
    pyplot.show = lambda *args, **kwargs: report.show(pyplot)
    namespace['load_detector'] = bundled_detector(namespace, bundle)
    versions = {name: str(namespace[alias].__version__) for name, alias in
                [('numpy', 'np'), ('pandas', 'pd'), ('torch', 'torch'), ('opencv', 'cv2'), ('sklearn', 'sklearn')]}
    print('Python', platform.python_version(), json.dumps(versions), flush=True)
    if mode == 'demo':
        root = output / 'synthetic_data'
        demo_catalogue(namespace, root)
        config = {'n_unknown': 120, 'n_near': 20, 'priors': [.005], 'fit_seeds': [20261031],
                  'epochs': 20, 'patience': 4, 'bootstraps': 30}
    else:
        root = Path(config.get('root', '/content/drive/MyDrive/catalogues'))
    for index, source in enumerate(cells[1:], 1):
        if mode == 'check' and index > 1:
            break
        report.phase = PHASES[index]
        print(f'[{index}/{len(cells) - 1}] {report.phase}', flush=True)
        overrides = configuration(config, root, output) if index == 2 else None
        exec(scientific_source(source, overrides), namespace)
    return {'status': 'ok', 'mode': mode, 'versions': versions, 'report': str(report.root),
            'synthetic_only': mode != 'real'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['check', 'demo', 'real'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'result.json').exists():
        parser.error('Usa una carpeta nueva para no mezclar experimentos.')
    before = package_snapshot()
    (args.output / 'packages_before.json').write_text(json.dumps(before, indent=2))
    try:
        config = json.loads(args.config.read_text()) if args.config else {}
        result = run(args.mode, config, args.output, Path(__file__).resolve().parent)
    except Exception as exc:
        traceback.print_exc()
        result = {'status': 'error', 'mode': args.mode, 'error': f'{type(exc).__name__}: {exc}',
                  'traceback': traceback.format_exc()}
    after = package_snapshot()
    result['packages_unchanged'] = before == after
    if before != after:
        result.update(status='error', error='Las versiones instaladas cambiaron durante la ejecucion.')
    (args.output / 'packages_after.json').write_text(json.dumps(after, indent=2))
    (args.output / 'result.json').write_text(json.dumps(result, indent=2))
    print('Paquetes instalados sin cambios:', before == after, flush=True)
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    sys.exit(main())
