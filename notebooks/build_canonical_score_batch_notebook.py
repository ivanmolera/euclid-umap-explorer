"""Build the self-contained, zero-install Colab notebook using only the stdlib."""
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile


HERE = Path(__file__).resolve().parent
TARGET = HERE / 'escope_canonical_score_batch_colab.ipynb'


def scientific_cells():
    notebook = json.loads((HERE / 'escope_canonical_score_learning_colab.ipynb').read_text())
    return [''.join(cell['source']).removeprefix('%%escope\n') for cell in notebook['cells']
            if 'escope-isolated' in cell.get('metadata', {}).get('tags', [])]


def bundle_bytes():
    detector = (HERE.parent / 'euclid_umap_explorer/lens_plausibility.py').read_bytes()
    expected = 'b45f1d4e8866aa5ad2805ea6205c1a138abf4da3459ecd116b44511a822600a4'
    if hashlib.sha256(detector).hexdigest() != expected:
        raise ValueError('Review the frozen detector before rebuilding this experiment.')
    files = {'batch_worker.py': (HERE / 'canonical_score_batch_worker.py').read_bytes(),
             'science_cells.json': json.dumps(scientific_cells()).encode(),
             'frozen_detector.py': detector}
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            info = zipfile.ZipInfo(name, (2026, 11, 3, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, content)
    return output.getvalue()


def make_notebook():
    bundle = bundle_bytes()
    cells = []

    def add(kind, source):
        cell = {'cell_type': kind, 'id': f'cell-{len(cells):02d}',
                'metadata': {}, 'source': source.splitlines(keepends=True)}
        if kind == 'code':
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add('markdown', '''# E-SCOPE: aprendizaje del canonical score, sin instalaciones

**Esta version no instala, actualiza ni desinstala paquetes.** Tampoco usa un segundo kernel Jupyter ni magias personalizadas. Ejecuta el analisis en un proceso Python nuevo, con las librerias que ya tiene Colab, y devuelve tablas, imagenes y archivos. El detector y el codigo cientifico estan incluidos en este notebook.

Si ya ejecutaste celdas antiguas que cambiaron NumPy, guarda primero los resultados locales que necesites. Una sola vez, usa **Entorno de ejecucion > Desconectar y eliminar entorno de ejecucion** y reconecta. Reiniciar solo Python no restaura paquetes cambiados. No vuelvas a ejecutar las celdas de instalacion del notebook anterior.

Ejecuta las secciones 1-3 primero, sin Drive. Deben terminar con `OK: check` y `OK: demo`. La demostracion verifica el flujo completo, pero sus imagenes y metricas sinteticas no validan el score cientificamente.
''')
    add('markdown', '## 1. Preparar el ejecutor\nSolo descomprime el codigo incluido en una carpeta temporal. No descarga codigo ni modifica librerias.\n')
    add('code', f'''import base64
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import uuid
import zipfile
from IPython.display import HTML, Image, display

BUNDLE_B64 = {base64.b64encode(bundle).decode()!r}
BUNDLE_SHA256 = {hashlib.sha256(bundle).hexdigest()!r}

def package_snapshot():
    return sorted((item.metadata['Name'] or '', item.version, str(item.locate_file('')))
                  for item in importlib.metadata.distributions())

PACKAGES_BEFORE = package_snapshot()
WORKSPACE = Path(tempfile.mkdtemp(prefix='escope-batch-'))
payload = base64.b64decode(BUNDLE_B64)
assert hashlib.sha256(payload).hexdigest() == BUNDLE_SHA256
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    assert set(archive.namelist()) == {{'batch_worker.py', 'science_cells.json', 'frozen_detector.py'}}
    archive.extractall(WORKSPACE)

def run_experiment(mode, config=None, output=None):
    output = (Path(output) if output is not None else WORKSPACE / (mode + '-' + uuid.uuid4().hex[:8])).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / 'input_config.json'
    config_path.write_text(json.dumps(config or {{}}, indent=2))
    command = [sys.executable, '-B', '-u', str(WORKSPACE / 'batch_worker.py'),
               '--mode', mode, '--config', str(config_path), '--output', str(output)]
    with (output / 'execution.log').open('w') as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, cwd=WORKSPACE)
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            returncode = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        finally:
            process.stdout.close()
    if package_snapshot() != PACKAGES_BEFORE:
        raise RuntimeError('Los paquetes del entorno cambiaron durante esta sesion.')
    result_file = output / 'result.json'
    if not result_file.is_file():
        raise RuntimeError(f'El proceso termino sin resultado (codigo {{returncode}}). Ver {{output / "execution.log"}}')
    result = json.loads(result_file.read_text())
    if returncode or result['status'] != 'ok':
        print('No se ha instalado ni reparado ningun paquete. Si fallan imports, usa un entorno Colab nuevo.')
        raise RuntimeError(result.get('error', 'Error del proceso') + f'\\nLog: {{output / "execution.log"}}')
    report = output / 'report'
    for entry in json.loads((report / 'index.json').read_text()):
        path = report / entry['path']
        display(HTML(path.read_text()) if entry['kind'] == 'html' else Image(filename=str(path)))
    print(f'OK: {{mode}}. Paquetes sin cambios. Resultados: {{output}}')
    return output

print('Ejecutor listo. Python:', sys.version.split()[0])
''')
    add('markdown', '## 2. Comprobar imports, detector, Parquet, entrenamiento y figuras\nEsta prueba usa las versiones presentes, sin exigir coincidencia con versiones de paquetes ajenos al experimento.\n')
    add('code', "CHECK_OUTPUT = run_experiment('check')\n")
    add('markdown', '## 3. Demostracion completa sin Drive\n480 cutouts sinteticos, particiones por tiles, extraccion de regiones, ajuste nnPU, test reservado, tablas, overlays y exportacion. Usa un ajuste corto para verificar la ejecucion; no interpretes sus metricas como rendimiento real.\n')
    add('code', "DEMO_OUTPUT = run_experiment('demo')\n")
    add('markdown', '''## 4. Datos reales: configurar y autorizar Drive

Tras completar la demostracion, cambia `RUN_REAL_DATA` a `True`, revisa las rutas y ejecuta las secciones 4-5. Se necesitan el Parquet morfologico (`id_str`, y `object_id` si existe), el CSV de candidatos (`object_id`, `grade`) y los JPG VIS completos. Cada JPG debe estar en `<cutout_base>/<tile>/<tile>_<object_id>_gz_arcsinh_vis_only.jpg`; para IDs negativos, el nombre usa `NEG` en lugar del signo menos. No se necesitan embeddings ni GPU.

Por defecto no se exige el Parquet de propiedades fisicas. Para incluir controles cercanos, indica `physical_path` y aumenta `n_near`; debe contener `object_id`, `phz_median`, `phz_flags`. Esos controles son solo de auditoria. Los Unknown y los objetos cercanos **no son negativos confirmados**.

La red ajusta cinco pesos de regiones y tres de agregacion; no aprende a segmentar nuevos arcos. A/B son referencias positivas, C se evalua aparte y Unknown se trata mediante aprendizaje positivo-no etiquetado. Se mantienen el detector fijo, la separacion por tiles/imagenes duplicadas y la seleccion en validacion antes del test. Los priors son hipotesis, no prevalencias medidas. El resultado es un indice de evidencia visual, **no una probabilidad calibrada de lente**.
''')
    add('code', '''RUN_REAL_DATA = False
ROOT = '/content/drive/MyDrive/catalogues'
CONFIG = {
    'root': ROOT,
    'morph_path': ROOT + '/morphology_catalogue/morphology_catalogue.parquet',
    'lens_path': ROOT + '/strong_lensing_catalogue/q1_discovery_engine_lens_catalog.csv',
    'cutout_base': ROOT + '/morphology_catalogue/cutouts_jpg_gz_arcsinh_vis_only',
    'physical_path': ROOT + '/morphology_catalogue/useful_physical_measurements.parquet',
    'n_unknown': 3000,
    'n_near': 0,
    'near_z_max': 0.1,
    'seed': 20261031,
    'priors': [0.001, 0.005, 0.01],
    'fit_seeds': [20261031, 20261032, 20261033],
    'epochs': 400,
    'learning_rate': 0.02,
    'patience': 12,
    'bootstraps': 500,
    'recall_fraction': 0.05,
}
if RUN_REAL_DATA:
    from google.colab import drive
    if not os.path.ismount('/content/drive'):
        drive.mount('/content/drive')
else:
    print('Datos reales desactivados. Primero completa la demostracion.')
''')
    add('markdown', '''## 5. Ejecutar y guardar el experimento real
Cada ejecucion escribe en una carpeta nueva en Drive, sin sobrescribir experimentos anteriores. La extraccion informa de su progreso cada 100 cutouts. Se conservan los fallos de cobertura; nunca se convierten en score cero. Si faltan imagenes o grupos independientes, se detiene con el error en `execution.log`.

Se exportan pesos, curvas de entrenamiento, metricas de referencia reservadas, contribuciones por region, imagenes original/overlay, CSV para revision ciega y un manifiesto con hashes y versiones. AP y recall miden recuperacion de candidatos catalogados, no pureza astrofisica. Los pesos no son importancias causales y no se incorporan automaticamente a E-SCOPE.
''')
    add('code', '''if RUN_REAL_DATA:
    from datetime import datetime, timezone
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:6]
    REAL_OUTPUT = run_experiment('real', CONFIG,
        Path(ROOT) / 'morphology_catalogue' / 'escope_canonical_score_learning' / stamp)
else:
    print('No se ha accedido a Drive ni ejecutado el analisis real.')
''')
    add('markdown', '''## Si falla
- **En la seccion 2:** conserva el traceback. Este notebook no cambia paquetes; un entorno que ya fue alterado debe sustituirse una sola vez por uno nuevo. No ejecutes comandos para subir o bajar NumPy.
- **En la seccion 3:** comparte `execution.log` de la carpeta indicada. No depende de tus datos.
- **En la seccion 5:** revisa el archivo o columna nombrados en el error. Un fallo de rutas, cobertura o muestra no requiere reinstalar librerias.

Los resultados de demostracion viven en almacenamiento temporal de Colab y desaparecen al eliminar el entorno. Los resultados reales se guardan en Drive. La igualdad de las listas de paquetes antes/despues se comprueba en el notebook y en cada proceso.
''')
    return {'cells': cells, 'metadata': {'colab': {'name': TARGET.name},
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 5}


if __name__ == '__main__':
    TARGET.write_text(json.dumps(make_notebook(), indent=1) + '\n')
    print(TARGET)
