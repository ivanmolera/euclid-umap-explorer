# %% [markdown]
# # ESCOPE: aprender los pesos del canonical score
#
# **Pregunta:** conservando el detector de regiones de ESCOPE, mejora la
# recuperacion de candidatos reservados si aprendemos sus pesos con una red
# pequena, en lugar de fijarlos manualmente?
#
# OpenCV extrae las regiones; la red aprende a combinarlas. No entrenamos una
# CNN segmentadora sin disponer de mascaras de arcos. Las etiquetas son del
# cutout completo, no de cada region: sus contribuciones no son validaciones
# fisicas ni explicaciones causales. Un score alto del representante tampoco
# implica que todos los objetos de su cluster sean lentes.
#
# **A/B/C son candidatos de referencia, no confirmaciones universales. Unknown
# no significa no-lente.** Una galaxia cercana tambien puede actuar como lente:
# ESO325-G004 tiene z=0.0345 ([Smith et al. 2005](https://arxiv.org/abs/astro-ph/0504453)).
# Los controles de bajo redshift se mantienen como Unknown y no se usan como
# negativos. El piloto usa aprendizaje positivo-no etiquetado (PU), bajo
# supuestos explicitos, no una probabilidad calibrada de lente.

# %%
get_ipython().run_line_magic('pip', "install -q 'numpy<2' 'pandas>=2.2,<3' 'scikit-learn>=1.4,<2' 'torch>=2.2,<3' 'opencv-python-headless==4.10.0.84' 'pyarrow>=15' matplotlib")

# %%
from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import math
import os
import platform
import urllib.request

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import sklearn
import torch
from PIL import Image
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.nn import functional as F

FEATURE_NAMES = ('tangential', 'elongation', 'angular_span', 'radial_coherence', 'contrast')
INITIAL_REGION_WEIGHTS = (0.30, 0.20, 0.25, 0.15, 0.10)
INITIAL_POOL_WEIGHTS = (0.65, 0.20, 0.15)
DETECTOR_COMMIT = 'd25e4c4feeecf4c369bceb8798882ad4d5b25eef'
DETECTOR_SHA256 = 'b45f1d4e8866aa5ad2805ea6205c1a138abf4da3459ecd116b44511a822600a4'


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_detector(destination):
    destination = Path(destination)
    if not destination.exists():
        url = f'https://raw.githubusercontent.com/ivanmolera/euclid-umap-explorer/{DETECTOR_COMMIT}/euclid_umap_explorer/lens_plausibility.py'
        with urllib.request.urlopen(url, timeout=60) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != DETECTOR_SHA256:
            raise ValueError('El detector descargado no coincide con el SHA256 fijado.')
        destination.write_bytes(content)
    if sha256_file(destination) != DETECTOR_SHA256:
        raise ValueError('Detector distinto del fijado: no mezclar versiones en una corrida.')
    spec = importlib.util.spec_from_file_location('escope_frozen_detector', destination)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exact_ids(values):
    if any(isinstance(value, (float, np.floating)) for value in values):
        raise ValueError('Identificadores float: recargar como texto, sin redondearlos.')
    clean = pd.Series(values, dtype='string').str.strip()
    if clean.isna().any() or not clean.str.fullmatch(r'-?\d+').all():
        raise ValueError('object_id ausente o no entero exacto.')
    return clean


def load_catalogue(morph_path, lens_path, physical_path=None):
    schema = pq.ParquetFile(morph_path).schema_arrow.names
    columns = ['id_str'] + (['object_id'] if 'object_id' in schema else [])
    columns += [name for name in ('segmentation_area', 'cutout_width_arcsec') if name in schema]
    frame = pd.read_parquet(morph_path, columns=columns)
    frame['id_str'] = frame['id_str'].astype('string').str.strip()
    suffix = frame['id_str'].str.rsplit('_', n=1).str[-1]
    frame['object_id'] = exact_ids(frame['object_id'] if 'object_id' in frame else suffix)
    if not frame['object_id'].eq(suffix).all() or frame['object_id'].duplicated().any():
        raise ValueError('id_str inconsistente o object_id duplicados en morfologia.')
    frame['tile'] = frame['id_str'].str.rsplit('_', n=2).str[-2]
    lenses = pd.read_csv(lens_path, dtype={'object_id': 'string'})
    lenses['object_id'] = exact_ids(lenses['object_id'])
    lenses['grade'] = lenses['grade'].astype('string').str.strip().str.upper()
    lenses = lenses[lenses['grade'].isin(['A', 'B', 'C'])].copy()
    if lenses.groupby('object_id')['grade'].nunique().gt(1).any():
        raise ValueError('Grados de referencia contradictorios.')
    frame = frame.merge(lenses[['object_id', 'grade']].drop_duplicates(),
                        on='object_id', how='left', validate='one_to_one')
    frame['grade'] = frame['grade'].fillna('Unknown')
    frame['valid_phz'] = np.nan
    if physical_path is not None:
        physical = pd.read_parquet(physical_path, columns=['object_id', 'phz_median', 'phz_flags'])
        physical['object_id'] = exact_ids(physical['object_id'])
        if physical['object_id'].duplicated().any():
            raise ValueError('Redshifts duplicados: resolver antes de unir.')
        z = pd.to_numeric(physical['phz_median'], errors='coerce')
        flags = pd.to_numeric(physical['phz_flags'], errors='coerce')
        physical['valid_phz'] = z.where(np.isfinite(z) & z.ge(0) & flags.eq(0))
        frame = frame.drop(columns='valid_phz').merge(
            physical[['object_id', 'valid_phz']], on='object_id', how='left', validate='one_to_one')
    return frame


def sample_cases(frame, n_unknown, n_near, z_max, seed):
    if n_unknown < 1 or n_near < 0 or not 0 < z_max < 1:
        raise ValueError('Parametros de muestreo invalidos.')
    references = frame[frame['grade'].isin(['A', 'B', 'C'])].copy()
    unknown = frame[frame['grade'].eq('Unknown')].sort_values('object_id')
    if len(unknown) < n_unknown:
        raise ValueError('No hay suficientes Unknown para la muestra solicitada.')
    random = unknown.sample(n=n_unknown, random_state=seed).copy()
    primary = pd.concat([references, random], ignore_index=True).assign(primary_pool=True)
    # Extra nearby draws are audit-only; they cannot change the PU fit population.
    near = unknown[unknown['valid_phz'].between(0, z_max, inclusive='both')
                   & ~unknown['object_id'].isin(primary['object_id'])]
    if len(near) < n_near:
        raise ValueError(f'Solo {len(near)} controles cercanos validos; reducir N_NEAR_CONTROLS explicitamente.')
    extra = near.sample(n=n_near, random_state=seed + 1).assign(primary_pool=False)
    selected = pd.concat([primary, extra], ignore_index=True)
    selected['near_control'] = selected['grade'].eq('Unknown') & selected['valid_phz'].between(0, z_max)
    assert selected['object_id'].is_unique
    return selected


def cutout_path(row, base):
    tile, object_id = str(row['tile']), str(row['object_id'])
    file_id = 'NEG' + object_id[1:] if object_id.startswith('-') else object_id
    return Path(base) / tile / f'{tile}_{file_id}_gz_arcsinh_vis_only.jpg'

# %% [markdown]
# ## 1. Datos y protocolo fijados antes de mirar resultados
# En Colab se usa Drive; tambien se pueden cambiar las rutas por copias locales.
# Todos los grupos usan el mismo JPG morfologico VIS completo, sin reescalar ni
# sustituirlo por un thumbnail o una imagen de otro catalogo. Si falta, queda
# como fallo de cobertura, nunca como score cero.
#
# Se incluyen todos los A/B/C presentes, 3.000 Unknown aleatorios y hasta 500
# controles adicionales con 0 <= phz_median <= 0.1 y phz_flags=0. El corte 0.1
# es configurable: **no es un umbral fisico que descarte lentes**. Un photo-z
# valido segun flags sigue teniendo incertidumbre; no es distancia exacta.
# Los controles extra no entran en el ajuste ni en la seleccion del modelo.
# Ni redshift, ni grado, ni cluster, ni enrichment son entradas de la red.

# %%
try:
    from google.colab import drive
except ImportError:
    drive = None
if drive is not None and not os.path.ismount('/content/drive'):
    drive.mount('/content/drive')

ROOT = Path('/content/drive/MyDrive/catalogues')
MORPH_PATH = ROOT / 'morphology_catalogue/morphology_catalogue.parquet'
LENS_PATH = ROOT / 'strong_lensing_catalogue/q1_discovery_engine_lens_catalog.csv'
PHYSICAL_PATH = ROOT / 'morphology_catalogue/useful_physical_measurements.parquet'
CUTOUT_BASE = ROOT / 'morphology_catalogue/cutouts_jpg_gz_arcsinh_vis_only'
OUTPUT_DIR = ROOT / 'morphology_catalogue/escope_canonical_score_learning'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 20261031
N_UNKNOWN, N_NEAR_CONTROLS, NEAR_Z_MAX = 3000, 500, 0.1
POSITIVE_GRADES = ('A', 'B')  # Sensitivity with A only requires a new experiment.
PU_PRIORS = (0.001, 0.005, 0.01)  # Illustrative hypotheses, NOT measured prevalence.
FIT_SEEDS = (SEED, SEED + 1, SEED + 2)
EPOCHS, LEARNING_RATE, PATIENCE = 400, 0.02, 12
RECALL_FRACTION, BOOTSTRAPS = 0.05, 500
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
detector = load_detector(OUTPUT_DIR / 'frozen_lens_plausibility.py')
catalogue = load_catalogue(MORPH_PATH, LENS_PATH, PHYSICAL_PATH if N_NEAR_CONTROLS else None)
cases = sample_cases(catalogue, N_UNKNOWN, N_NEAR_CONTROLS, NEAR_Z_MAX, SEED)
cases.to_csv(OUTPUT_DIR / 'sample_before_image_loading.csv', index=False)
display(cases.groupby(['grade', 'primary_pool', 'near_control']).size().rename('objects').reset_index())

# %% [markdown]
# ## 2. Misma extraccion que ESCOPE, con trazabilidad
# El detector se fija por commit y SHA256. Usa gris normalizado P1-P99.5,
# resta gaussiana positiva, anillo de 0.09-0.46 veces el lado menor y umbral
# max(P91 del residual, mediana + 2.5 * 1.4826 * MAD, 0.04): **no Otsu**.
# Aplica cierre eliptico 3x3, mascara de lineas de borde Canny/Hough y componentes
# conexos de 8 vecinos. Se conservan las reglas de aceptacion del detector.
#
# La red recibe los cinco proxies normalizados [0,1] de las seis regiones mejor
# puntuadas por el detector original, con padding enmascarado, y la concordancia
# geometrica entre ellas. Puede reordenar esas seis regiones, pero no recuperar
# arcos rechazados o ausentes: esto afina la combinacion, no el segmentador.
# El cache incluye el hash de los bytes y del detector, no solo object_id.

# %%
def extract_record(rgb, detector_module):
    result = detector_module.analyze_image(rgb)
    record = {key: result.get(key) for key in ('status', 'score', 'pair_agreement', 'quality_flag')}
    record['valid'] = result.get('score') is not None
    record['features'] = [[float(c['features'][key]) for key in FEATURE_NAMES]
                          for c in result.get('accepted', [])[:6]]
    record['regions'] = [{key: value for key, value in c.items()
                          if key not in ('pixels', 'features', 'accepted')}
                         for c in result.get('accepted', [])[:6]]
    record['n_accepted'] = len(result.get('accepted', []))
    return record


def extract_cases(cases, base, output, detector_module):
    cache = Path(output) / ('features_' + DETECTOR_SHA256[:12])
    cache.mkdir(exist_ok=True, parents=True)
    records, rows, regions = [], [], []
    for i, row in enumerate(cases.to_dict('records')):
        path = cutout_path(row, base)
        image_hash = ''
        try:
            content = path.read_bytes()
            image_hash = hashlib.sha256(content).hexdigest()
            cached = cache / f'{image_hash}.json'
            if cached.exists():
                record = json.loads(cached.read_text())
            else:
                from io import BytesIO
                with Image.open(BytesIO(content)) as image:
                    if image.width * image.height > 10_000_000:
                        raise ValueError('Imagen demasiado grande.')
                    rgb = np.asarray(image.convert('RGB'))
                record = extract_record(rgb, detector_module)
                cached.write_text(json.dumps(record, allow_nan=False))
        except Exception as exc:
            record = {'status': 'read_error:' + type(exc).__name__, 'score': None,
                      'valid': False, 'features': [], 'regions': [], 'n_accepted': 0}
        records.append(record)
        rows.append({**row, 'image_path': str(path), 'image_sha256': image_hash,
                     'valid_image': record['valid'], 'detector_status': record['status'],
                     'original_score': record['score'], 'n_accepted': record['n_accepted']})
        for rank, region in enumerate(record['regions'], 1):
            regions.append({'object_id': row['object_id'], 'original_region_rank': rank, **region,
                            **dict(zip(FEATURE_NAMES, record['features'][rank - 1]))})
        if (i + 1) % 100 == 0:
            print(f'{i + 1}/{len(cases)} cutouts procesados')
    frame = pd.DataFrame(rows)
    tensor = np.zeros((len(rows), 6, len(FEATURE_NAMES)), dtype=np.float32)
    mask = np.zeros((len(rows), 6), dtype=bool)
    pair = np.zeros(len(rows), dtype=np.float32)
    for i, record in enumerate(records):
        n = len(record['features'])
        tensor[i, :n] = np.asarray(record['features'], dtype=np.float32).reshape(n, 5)
        mask[i, :n] = True
        pair[i] = record.get('pair_agreement') or 0.
    pd.DataFrame(regions).to_csv(Path(output) / 'region_measurements.csv', index=False)
    return frame, tensor, mask, pair


def split_groups(frame, seed, positive_grades=('A', 'B')):
    # Union keeps a tile OR byte-identical images together across partitions.
    parents = list(range(len(frame)))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for field in ('tile', 'image_sha256'):
        seen = {}
        for i, value in enumerate(frame[field].fillna('').astype(str)):
            if not value:
                continue
            if value in seen:
                parents[root(i)] = root(seen[value])
            else:
                seen[value] = i
    groups = np.array([root(i) for i in range(len(frame))])
    if len(np.unique(groups)) < 5:
        raise ValueError('Se necesitan al menos cinco grupos independientes de tiles/imagenes.')
    strata = frame['grade'].astype(str).to_numpy()
    folds = np.full(len(frame), -1)
    for fold, (_, held) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(frame, strata, groups)):
        folds[held] = fold
    split = np.where(folds == 0, 'test', np.where(folds == 1, 'validation', 'train'))
    for name in ('train', 'validation', 'test'):
        eligible = (split == name) & frame['valid_image'] & frame['primary_pool']
        for label in (frame['grade'].isin(positive_grades), frame['grade'].eq('Unknown')):
            if int((eligible & label).sum()) < 5:
                raise ValueError(f'Pocos P/U validos en {name}; ampliar muestra, no probar semillas hasta ganar.')
    return split, groups


frame, region_features, region_mask, pair = extract_cases(cases, CUTOUT_BASE, OUTPUT_DIR, detector)
frame['split'], frame['leakage_group'] = split_groups(frame, SEED, POSITIVE_GRADES)
frame.to_csv(OUTPUT_DIR / 'sample_and_split.csv', index=False)
display(frame.groupby(['split', 'grade', 'primary_pool', 'valid_image']).size().rename('objects').reset_index())
print('Faltan/fallan imagenes:', int((~frame['valid_image']).sum()))

# %% [markdown]
# ## 3. Red interpretable y perdida positivo-no etiquetado
# Pesos compartidos entre regiones: w = softmax(a), cinco pesos no negativos
# que suman 1. q_i = sum_j w_j f_ij. Pesos de agregacion v = softmax(b):
# S = 100 * (v_1 q_(1) + v_2 q_(2) + v_g G), donde q_(1), q_(2) son las dos
# regiones mas puntuadas por los pesos aprendidos, y G es la geometria fija.
#
# Inicializacion exacta: w=(.30,.20,.25,.15,.10), v=(.65,.20,.15).
# Para entrenar: g = softplus(t) * (S/100 - .5) + bias. La sigmoide de g
# **no se interpreta como probabilidad calibrada**. Se exporta S, no 100*sigmoid(g).
#
# La perdida logistico-nnPU es pi*E_P[softplus(-g)] +
# max(0, E_U[softplus(g)] - pi*E_P[softplus(g)]). En la rama de riesgo negativo
# se aplica el paso correctivo -R_neg del algoritmo de
# [Kiryo et al. 2017](https://proceedings.neurips.cc/paper_files/paper/2017/hash/7cce53cf90577442771720a370c3c723-Abstract.html).
#
# pi es la fraccion hipotetica de lentes en U, NO la fraccion de etiquetas del
# catalogo. Probamos tres valores ilustrativos; no estimamos prevalencia.
# El estimador requiere positivos representativos de los positivos ocultos y
# U representativo de la mezcla objetivo. Los candidatos conocidos pueden
# violar ese supuesto por seleccion visual y contaminacion de las etiquetas.
# Por ello los resultados son de recuperacion de referencias, no pureza real.

# %%
class InterpretableScoreNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.region_logits = nn.Parameter(torch.log(torch.tensor(INITIAL_REGION_WEIGHTS)))
        self.pool_logits = nn.Parameter(torch.log(torch.tensor(INITIAL_POOL_WEIGHTS)))
        self.raw_gain = nn.Parameter(torch.tensor(4.))
        self.bias = nn.Parameter(torch.tensor(-3.))

    def evidence(self, x, mask, geometry):
        region_scores = (x * self.region_logits.softmax(0)).sum(-1) * mask
        best = region_scores.topk(2, dim=1).values
        branches = torch.column_stack([best[:, 0], best[:, 1], geometry])
        return (branches * self.pool_logits.softmax(0)).sum(1)

    def forward(self, x, mask, geometry):
        return F.softplus(self.raw_gain) * (self.evidence(x, mask, geometry) - .5) + self.bias

    def weights(self):
        return {'region': dict(zip(FEATURE_NAMES, self.region_logits.softmax(0).detach().tolist())),
                'aggregation': dict(zip(('best_region', 'second_region', 'geometry'), self.pool_logits.softmax(0).detach().tolist()))}


def nnpu_loss(positive_logits, unlabeled_logits, prior, corrective=True):
    if not 0 < prior < 1 or not positive_logits.numel() or not unlabeled_logits.numel():
        raise ValueError('nnPU necesita pi en (0,1), positivos y no etiquetados.')
    positive_risk = prior * F.softplus(-positive_logits).mean()
    negative_risk = F.softplus(unlabeled_logits).mean() - prior * F.softplus(positive_logits).mean()
    objective = positive_risk + negative_risk.clamp(min=0)
    update = -negative_risk if corrective and negative_risk.detach().item() < 0 else objective
    return update, objective


def reference_metrics(grades, scores, positive_grades=('A', 'B'), fraction=.05):
    grades, scores = np.asarray(grades), np.asarray(scores, dtype=float)
    keep = np.isfinite(scores) & np.isin(grades, [*positive_grades, 'Unknown'])
    labels, values = np.isin(grades[keep], positive_grades), scores[keep]
    if not len(values) or not labels.any() or labels.all():
        return {'reference_ap': np.nan, 'reference_recall': np.nan, 'review_k': 0}
    k = max(1, math.ceil(fraction * len(values)))
    cutoff = np.sort(values)[-k]
    higher, tied = values > cutoff, values == cutoff
    # Expected recovery under random ordering of ties, not object-ID ordering.
    found = labels[higher].sum() + labels[tied].sum() * (k - higher.sum()) / tied.sum()
    return {'reference_ap': float(average_precision_score(labels, values)),
            'reference_recall': float(found / labels.sum()), 'review_k': k}


def fit_weights(x, mask, geometry, frame, prior, seed, positive_grades=('A', 'B'),
                epochs=400, learning_rate=.02, patience=12):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = InterpretableScoreNet()
    with torch.no_grad():
        model.region_logits.add_(torch.tensor(rng.normal(0, .03, 5), dtype=torch.float32))
        model.pool_logits.add_(torch.tensor(rng.normal(0, .03, 3), dtype=torch.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    eligible = frame['valid_image'].to_numpy() & frame['primary_pool'].to_numpy()
    positive = frame['grade'].isin(positive_grades).to_numpy()
    unknown = frame['grade'].eq('Unknown').to_numpy()
    train = frame['split'].eq('train').to_numpy() & eligible
    validation = frame['split'].eq('validation').to_numpy() & eligible & (positive | unknown)
    p, u = np.flatnonzero(train & positive), np.flatnonzero(train & unknown)
    v = np.flatnonzero(validation)
    if not len(p) or not len(u) or not (positive[v].any() and unknown[v].any()):
        raise ValueError('Faltan P/U en train o validation.')
    tensors = [torch.tensor(value, dtype=torch.float32) for value in (x, mask, geometry)]
    best_state, best_metric, stale, history = None, -np.inf, 0, []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(*(value[np.r_[p, u]] for value in tensors))
        update, objective = nnpu_loss(logits[:len(p)], logits[len(p):], prior)
        # Weak shrinkage towards the existing rule; correlated proxies are not uniquely identifiable.
        penalty = .001 * ((model.region_logits.softmax(0) - torch.tensor(INITIAL_REGION_WEIGHTS)) ** 2).sum()
        (update + penalty).backward()
        optimizer.step()
        if epoch % 5 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                score = model.evidence(*(value[v] for value in tensors)).numpy()
            metric = reference_metrics(frame.iloc[v]['grade'], score, positive_grades)['reference_ap']
            history.append({'epoch': epoch, 'nnpu_risk': float(objective.detach()), 'validation_reference_ap': metric})
            if metric > best_metric + 1e-6:
                best_state, best_metric, stale = copy.deepcopy(model.state_dict()), metric, 0
            else:
                stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return model, pd.DataFrame(history), best_metric


def predict_scores(model, x, mask, geometry, valid):
    scores = np.full(len(x), np.nan)
    idx = np.flatnonzero(valid)
    model.eval()
    with torch.no_grad():
        tensors = [torch.tensor(value[idx], dtype=torch.float32) for value in (x, mask, geometry)]
        scores[idx] = 100 * model.evidence(*tensors).numpy()
    return scores

# %% [markdown]
# ## 4. Ajuste y seleccion SOLO en desarrollo
# Separacion 60/20/20 por tiles y duplicados exactos de imagen; las rotaciones
# o aumentos de un mismo objeto deberian permanecer en su grupo si se anaden.
# Tiles adyacentes pueden compartir sistemas: revisar duplicados astrofisicos
# antes de afirmar independencia espacial completa. No ajustamos umbrales del
# detector, ni usamos los 20 clusters mas enriquecidos como conjunto de prueba.
#
# Se selecciona prior/semilla/checkpoint por AP de recuperacion de A/B frente a
# Unknown en validation. Esta AP mide reconocimiento del catalogo, **no precision
# astrofisica**. C se conserva para evaluacion por grado, no se convierte en negativo.
# Las repeticiones muestran sensibilidad del ajuste, no intervalos de confianza.

# %%
initial_model = InterpretableScoreNet()
initial_float = predict_scores(initial_model, region_features, region_mask, pair, frame['valid_image'])
valid = frame['valid_image'].to_numpy()
if not np.all(np.abs(initial_float[valid] - frame.loc[valid, 'original_score']) <= .5001):
    raise ValueError('La regla de partida no reproduce el detector: detener experimento.')
frame['manual_float_score'] = initial_float
models, candidate_rows, weight_rows = {}, [], []
for prior in PU_PRIORS:
    for fit_seed in FIT_SEEDS:
        key = f'pi{prior:g}_seed{fit_seed}'
        model, history, val_ap = fit_weights(region_features, region_mask, pair, frame, prior,
                                             fit_seed, POSITIVE_GRADES, EPOCHS, LEARNING_RATE, PATIENCE)
        models[key] = model
        history.to_csv(OUTPUT_DIR / f'training_{key}.csv', index=False)
        candidate_rows.append({'model': key, 'prior_hypothesis': prior, 'seed': fit_seed,
                               'validation_reference_ap': val_ap})
        for level, weights in model.weights().items():
            weight_rows.extend({'model': key, 'level': level, 'feature': name, 'weight': weight}
                               for name, weight in weights.items())
candidates = pd.DataFrame(candidate_rows).sort_values(['validation_reference_ap', 'model'], ascending=[False, True])
selected_key = candidates.iloc[0]['model']
selected_model = models[selected_key]
candidates.to_csv(OUTPUT_DIR / 'validation_model_selection.csv', index=False)
weights = pd.DataFrame(weight_rows)
weights.to_csv(OUTPUT_DIR / 'all_fit_weights.csv', index=False)
torch.save(selected_model.state_dict(), OUTPUT_DIR / 'selected_weights.pt')
(OUTPUT_DIR / 'selected_weights.json').write_text(json.dumps({
    'selected_model': selected_key, 'weights': selected_model.weights(),
    'detector_commit': DETECTOR_COMMIT, 'detector_sha256': DETECTOR_SHA256,
    'interpretation': 'Learned visual ranking index, NOT calibrated lens probability.'}, indent=2))
display(candidates)

# %% [markdown]
# ## 5. Evaluacion reservada y mejora frente al score manual
# Test se consulta tras congelar el modelo. Se compara AP de referencia y
# recall al revisar el 5% de la muestra test (presupuesto relativo identico).
# Las metricas son condicionales a los JPG evaluables; la tabla de cobertura
# conserva todos los fallos por grado y particion, sin convertirlos en ceros.
# Tambien se compara contra la regla manual sin redondear, para no atribuir
# a los pesos una mejora debida solamente a eliminar empates del score entero.
# Los empates se reparten proporcionalmente. Se informa por grado y separado
# para el contraste con controles cercanos. Ese contraste no reemplaza al test
# con Unknown aleatorios ni estima la tasa de falsos positivos.
#
# Un bootstrap pareado por grupos estima la incertidumbre de la diferencia
# aprendida - manual en recall. Sus intervalos son condicionales a este modelo
# y muestra, no validan las etiquetas ni incluyen incertidumbre de seleccion.
# No reutilizar este test para modificar los pesos; una nueva iteracion exige
# datos externos o una nueva evaluacion reservada declarada de antemano.

# %%
def grouped_recall_difference(frame, original, learned, groups, positive_grades,
                              fraction, repeats, seed):
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    draws = []
    for _ in range(repeats):
        sampled = rng.choice(unique, len(unique), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == group) for group in sampled])
        old = reference_metrics(frame.iloc[idx]['grade'], original[idx], positive_grades, fraction)['reference_recall']
        new = reference_metrics(frame.iloc[idx]['grade'], learned[idx], positive_grades, fraction)['reference_recall']
        if np.isfinite(old) and np.isfinite(new):
            draws.append(new - old)
    if len(draws) < 20:
        return {'valid_bootstraps': len(draws), 'delta_low': None, 'delta_high': None}
    low, high = np.quantile(draws, [.025, .975])
    return {'valid_bootstraps': len(draws), 'delta_low': float(low), 'delta_high': float(high)}


frame['learned_score'] = predict_scores(selected_model, region_features, region_mask, pair, valid)
frame['learned_rounded_score'] = np.rint(frame['learned_score'])
test = frame['split'].eq('test') & frame['primary_pool'] & frame['valid_image']
evaluation = []
for grade_set in (POSITIVE_GRADES, ('A',), ('B',), ('C',)):
    for column in ('original_score', 'manual_float_score', 'learned_score', 'learned_rounded_score'):
        evaluation.append({'method': column, 'positive_grades': '+'.join(grade_set),
                           **reference_metrics(frame.loc[test, 'grade'], frame.loc[test, column], grade_set, RECALL_FRACTION)})
evaluation = pd.DataFrame(evaluation)
evaluation.to_csv(OUTPUT_DIR / 'heldout_reference_metrics.csv', index=False)
test_frame = frame.loc[test].reset_index(drop=True)
uncertainty = grouped_recall_difference(test_frame, test_frame['manual_float_score'].to_numpy(),
    test_frame['learned_score'].to_numpy(), test_frame['leakage_group'].to_numpy(),
    POSITIVE_GRADES, RECALL_FRACTION, BOOTSTRAPS, SEED + 100)
(OUTPUT_DIR / 'heldout_recall_difference.json').write_text(json.dumps(uncertainty, indent=2))
display(evaluation)
print('Diferencia de recall (IC bootstrap 95%):', uncertainty)

# Held-out near controls remain unlabeled; this is a restricted retrieval contrast.
near_test = frame['split'].eq('test') & frame['valid_image'] & (
    frame['grade'].isin(POSITIVE_GRADES) | frame['near_control'])
near_metrics = pd.DataFrame([{'method': column,
    **reference_metrics(frame.loc[near_test, 'grade'], frame.loc[near_test, column], POSITIVE_GRADES, RECALL_FRACTION)}
    for column in ('manual_float_score', 'learned_score')])
near_metrics.to_csv(OUTPUT_DIR / 'nearby_unknown_contrast_NOT_specificity.csv', index=False)
display(near_metrics)

# Redshift-stratified retrieval is diagnostic only, never a new selection metric.
redshift_rows = []
for low, high in ((0., .1), (.1, .5), (.5, 1.), (1., np.inf)):
    in_bin = test & frame['valid_phz'].ge(low) & frame['valid_phz'].lt(high)
    for column in ('manual_float_score', 'learned_score'):
        redshift_rows.append({'z_low': low, 'z_high': high, 'objects': int(in_bin.sum()),
            'method': column, **reference_metrics(frame.loc[in_bin, 'grade'], frame.loc[in_bin, column],
                                                  POSITIVE_GRADES, RECALL_FRACTION)})
pd.DataFrame(redshift_rows).to_csv(OUTPUT_DIR / 'redshift_stratified_retrieval.csv', index=False)

# %% [markdown]
# ## 6. Pesos, estabilidad y auditoria visual
# Los pesos relativos no son importancias causales: hay correlaciones entre
# elongacion, arco y coherencia radial. Comparar estabilidad entre priors y
# semillas antes de trasladar una regla a ESCOPE. Esta version no cambia
# automaticamente el score desplegado ni demuestra que un peso sea optimo.

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, level, initial in zip(axes, ('region', 'aggregation'), (INITIAL_REGION_WEIGHTS, INITIAL_POOL_WEIGHTS)):
    chosen = selected_model.weights()[level]
    names = list(chosen)
    x = np.arange(len(names))
    ax.bar(x - .18, initial, width=.36, label='Manual')
    ax.bar(x + .18, list(chosen.values()), width=.36, label='Aprendido')
    ax.set_xticks(x, names, rotation=25, ha='right')
    ax.set_ylim(0, 1)
    ax.set_title(level)
    ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'manual_vs_learned_weights.png', dpi=150)
plt.show()
display(weights.groupby(['level', 'feature'])['weight'].agg(['min', 'median', 'max']))

context_fields = [name for name in ('valid_phz', 'segmentation_area') if name in frame]
fig, axes = plt.subplots(1, len(context_fields), figsize=(5 * len(context_fields), 4), squeeze=False)
for ax, field in zip(axes[0], context_fields):
    for label, color in (('Unknown', 'gray'), ('A', 'green'), ('B', 'purple'), ('C', 'orange')):
        subset = frame[test & frame['grade'].eq(label)]
        context = np.log10(subset[field].where(subset[field] > 0)) if field == 'segmentation_area' else subset[field]
        ax.scatter(context, subset['learned_score'], s=10, alpha=.45, color=color, label=label)
    xlabel = 'log10 segmentation_area (pixels)' if field == 'segmentation_area' else field
    ax.set(xlabel=xlabel, ylabel='Learned visual index')
    ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'context_confounding_checks.png', dpi=150)
plt.show()


def show_case(object_id):
    matching = frame[frame['object_id'].eq(str(object_id))]
    if matching.empty:
        raise ValueError('Objeto fuera de la muestra.')
    index = matching.index[0]
    row = matching.iloc[0]
    if not row['valid_image']:
        print(row['detector_status'])
        return
    if sha256_file(row['image_path']) != row['image_sha256']:
        raise ValueError('La imagen ha cambiado desde la extraccion.')
    with Image.open(row['image_path']) as image:
        rgb = np.asarray(image.convert('RGB'))
    result = detector.analyze_image(rgb)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, image, title in zip(axes, (rgb, result['overlay']),
                               ('Cutout original', 'Regiones del detector fijo')):
        ax.imshow(image)
        ax.set_title(title)
        ax.axis('off')
    plt.show()
    print(f"Manual: {row['original_score']:.0f}; aprendido: {row['learned_score']:.1f}; grado: {row['grade']}; split: {row['split']}")
    w = np.array(list(selected_model.weights()['region'].values()))
    v = np.array(list(selected_model.weights()['aggregation'].values()))
    values = region_features[index] @ w
    order = np.flatnonzero(region_mask[index])
    order = order[np.argsort(-values[order], kind='stable')]
    details = []
    for rank, i in enumerate(order):
        region = result['accepted'][i]
        branch = v[rank] if rank < 2 else 0.
        details.append({'overlay_region': int(i + 1), 'learned_rank': rank + 1,
                        'radius_px': region['radius_px'], 'angular_span_deg': region['angular_span_deg'],
                        'learned_region_score': values[i],
                        'region_contribution_points': 100 * branch * values[i],
                        **{name + '_points': 100 * branch * w[j] * region_features[index, i, j]
                           for j, name in enumerate(FEATURE_NAMES)}})
    display(pd.DataFrame(details))
    print('Geometria (puntos adicionales):', 100 * v[2] * pair[index])


for object_id in frame.loc[test].sort_values('learned_score', ascending=False).head(3)['object_id']:
    show_case(object_id)

# %% [markdown]
# ## 7. Exportacion reproducible y revision ciega
# Se conservan imagenes fallidas, particiones, hashes, medidas por region,
# pesos por ajuste, curvas de entrenamiento, evaluacion y versiones. El CSV
# de revision mezcla 50 Unknown aleatorios y 50 con score alto; se ocultan sus
# grados y scores para la inspeccion. Es una auditoria dirigida, no una muestra
# para estimar pureza poblacional. No convertir cercania ni ausencia de arco
# evidente en una confirmacion automatica de no-lente.

# %%
frame.to_csv(OUTPUT_DIR / 'all_predictions.csv', index=False)
unknown_test = frame[test & frame['grade'].eq('Unknown')]
audit_random = unknown_test.sample(n=min(50, len(unknown_test)), random_state=SEED)
audit_top = unknown_test.sort_values('learned_score', ascending=False).head(50)
audit = pd.concat([audit_random, audit_top]).drop_duplicates('object_id').sample(frac=1, random_state=SEED)
audit = audit[['object_id', 'image_path', 'image_sha256']].copy()
audit['review_class'] = ''  # lens_like / non_lens_like / uncertain / artifact
audit['reviewer'], audit['notes'] = '', ''
audit.to_csv(OUTPUT_DIR / 'blind_visual_audit.csv', index=False)
manifest = {
    'experiment': 'frozen_detector_interpretable_neural_nnPU',
    'detector_commit': DETECTOR_COMMIT, 'detector_sha256': DETECTOR_SHA256,
    'detector_config': detector.DETECTOR_CONFIG,
    'files': {str(path): sha256_file(path) for path in (MORPH_PATH, LENS_PATH, PHYSICAL_PATH)
              if path is not None and Path(path).is_file()},
    'selected_model': selected_key, 'positive_grades': POSITIVE_GRADES,
    'unknown_are_negatives': False, 'nearby_are_negatives': False,
    'near_z_max': NEAR_Z_MAX, 'n_random_unknown': N_UNKNOWN, 'n_extra_near': N_NEAR_CONTROLS,
    'sample_sha256': sha256_file(OUTPUT_DIR / 'sample_and_split.csv'),
    'model_sha256': sha256_file(OUTPUT_DIR / 'selected_weights.pt'),
    'feature_names': FEATURE_NAMES, 'region_cap': 6,
    'priors_are_hypotheses': PU_PRIORS, 'seeds': FIT_SEEDS, 'split_seed': SEED,
    'epochs': EPOCHS, 'learning_rate': LEARNING_RATE, 'patience_checks': PATIENCE,
    'selection_metric': 'validation_reference_ap', 'recall_fraction': RECALL_FRACTION,
    'bootstrap_repeats': BOOTSTRAPS, 'versions': {
        'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__,
        'pandas': pd.__version__, 'sklearn': sklearn.__version__, 'opencv': cv2.__version__},
    'claim': 'Exploratory reference retrieval; no calibrated probabilities or confirmed non-lens labels.',
}
(OUTPUT_DIR / 'manifest.json').write_text(json.dumps(manifest, indent=2))
print('Resultados guardados en', OUTPUT_DIR)
