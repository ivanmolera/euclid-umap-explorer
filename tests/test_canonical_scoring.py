import copy
import hashlib
import importlib.util
import json
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from euclid_umap_explorer import canonical_scoring as scoring, lens_plausibility as frozen


def arc_png():
    yy, xx = np.mgrid[:160, :160]
    gray = np.clip(15 + 130 * np.exp(-((xx - 80)**2 + (yy - 80)**2) / 162), 0, 255).astype('uint8')
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    cv2.ellipse(rgb, (80, 80), (42, 42), 0, 20, 130, (225, 225, 225), 3)
    cv2.ellipse(rgb, (80, 80), (42, 42), 0, 205, 280, (190, 190, 190), 3)
    output = BytesIO()
    Image.fromarray(rgb).save(output, format='PNG')
    return output.getvalue()


class LearnedCanonicalScoringTests(unittest.TestCase):
    def test_model_records_exact_exported_weights_and_keeps_training_detector_frozen(self):
        model = json.loads(scoring.MODEL_PATH.read_bytes())
        self.assertEqual(model['selected_model'], 'pi0.01_seed20261033')
        self.assertEqual(scoring.REGION_WEIGHTS['contrast'], 0.381090372800827)
        self.assertEqual(scoring.POOL_WEIGHTS['second_region'], 0.5778163075447083)
        self.assertEqual(hashlib.sha256(Path(frozen.__file__).read_bytes()).hexdigest(), model['detector_sha256'])
        self.assertIn(scoring.MODEL_SHA256, scoring.SCORING_KEY)
        fingerprint = scoring.scoring_fingerprint()
        fingerprint['region_weights']['contrast'] = 0.
        self.assertNotEqual(fingerprint, scoring.scoring_fingerprint())

    def test_reranks_only_the_original_top_six_and_preserves_overlay_numbering(self):
        vectors = [*np.eye(5), np.zeros(5), np.ones(5)]
        baseline = {'score': 66, 'pair_agreement': .4, 'overlay_png': b'original-overlay',
                    'regions': [{'component_score': 1 - i / 10, 'original_number': i + 1,
                                 'features': dict(zip(scoring.FEATURE_NAMES, vector))}
                                for i, vector in enumerate(vectors)]}
        original = copy.deepcopy(baseline)
        with patch.object(frozen, 'score_image_bytes', return_value=baseline):
            result = scoring.score_image_bytes(b'image')
        self.assertEqual(len(result['regions']), 6)
        self.assertEqual([row['original_number'] for row in result['regions']], list(range(1, 7)))
        self.assertEqual([row['learned_rank'] for row in result['regions']], [3, 5, 4, 2, 1, 6])
        self.assertEqual(result['overlay_png'], b'original-overlay')
        self.assertEqual(result['manual_score'], 66)
        expected = 100 * (scoring.POOL_WEIGHTS['best_region'] * scoring.REGION_WEIGHTS['contrast']
                          + scoring.POOL_WEIGHTS['second_region'] * scoring.REGION_WEIGHTS['radial_coherence']
                          + scoring.POOL_WEIGHTS['geometry'] * .4)
        self.assertAlmostEqual(result['score'], expected, places=4)
        self.assertEqual(result['regions'][0]['manual_component_score'], original['regions'][0]['component_score'])
        self.assertEqual(original['regions'], baseline['regions'])
        self.assertEqual(result['regions'][0]['region_contribution_points'], 0.)
        self.assertGreater(result['regions'][4]['region_contribution_points'], 0.)

    def test_no_regions_is_zero_but_non_evaluable_remains_none(self):
        for baseline, expected in (({'score': 0, 'pair_agreement': 0., 'regions': []}, 0.),
                                   ({'score': None, 'status': 'insufficient_resolution'}, None)):
            with patch.object(frozen, 'score_image_bytes', return_value=baseline):
                result = scoring.score_image_bytes(b'image')
            self.assertEqual(result['score'], expected)
            self.assertEqual(result['scoring_model'], scoring.scoring_fingerprint())

    def test_real_detector_output_changes_only_scoring_not_segmentation_or_images(self):
        image = arc_png()
        baseline = frozen.score_image_bytes(image)
        result = scoring.score_image_bytes(image)
        self.assertNotAlmostEqual(result['score'], baseline['score'])
        self.assertEqual(result['overlay_png'], baseline['overlay_png'])
        self.assertEqual(result['center'], baseline['center'])
        self.assertEqual(result['pair_agreement'], baseline['pair_agreement'])
        self.assertEqual([row['features'] for row in result['regions']],
                         [row['features'] for row in baseline['regions']])
        self.assertAlmostEqual(result['score'], sum(result['contributions'].values()), places=4)
        self.assertNotIn('overlay_png', scoring.score_image_bytes(image, include_overlay=False))
        json.dumps({key: value for key, value in result.items() if key != 'overlay_png'}, allow_nan=False)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Optional parity check; app does not require PyTorch.')
    def test_matches_notebook_float32_neural_aggregation(self):
        import torch

        image = arc_png()
        baseline = frozen.score_image_bytes(image, include_overlay=False)
        x = np.zeros((6, 5), dtype=np.float32)
        for index, region in enumerate(baseline['regions']):
            x[index] = [region['features'][name] for name in scoring.FEATURE_NAMES]
        weights = torch.tensor([scoring.REGION_WEIGHTS[name] for name in scoring.FEATURE_NAMES])
        best = (torch.tensor(x) * weights).sum(-1).topk(2).values
        branches = torch.stack([best[0], best[1], torch.tensor(baseline['pair_agreement'])])
        pool = torch.tensor([scoring.POOL_WEIGHTS[name] for name in scoring.POOL_NAMES])
        expected = float(100 * (branches * pool).sum())
        self.assertAlmostEqual(scoring.score_image_bytes(image)['score'], expected, places=4)


if __name__ == '__main__':
    unittest.main()
