import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from classical_models import make_estimator, positive_scores
from site_features import FEATURE_NAMES, NUMERIC_COLUMNS, summarize_site

ROOT = Path(__file__).resolve().parents[1]


class FeatureTests(unittest.TestCase):
    def test_feature_values_order_and_one_read(self):
        reads = np.tile(np.array([0., 2., 4., 6.])[:, None], (1, 9))
        actual = summarize_site(reads, 'ACGTACG')
        self.assertEqual(actual.shape, (65,))
        self.assertEqual(len(set(FEATURE_NAMES)), 65)
        np.testing.assert_allclose(actual[:9], 3.)
        np.testing.assert_allclose(actual[9:18], np.sqrt(5.))
        np.testing.assert_allclose(actual[18:27], 1.5)
        np.testing.assert_allclose(actual[27:36], 4.5)
        np.testing.assert_array_equal(actual[36:64].reshape(
            7, 4), np.eye(4)[[0, 1, 2, 3, 0, 1, 2]])
        self.assertEqual(actual[-1], np.log1p(4))
        np.testing.assert_array_equal(summarize_site(
            reads[:1], 'AAAAAAA')[9:18], np.zeros(9))
        with self.assertRaises(ValueError):
            summarize_site(reads, 'AAAAAAN')

    def test_scaler_and_calibration_are_fitted_on_training(self):
        rng = np.random.default_rng(8)
        x = rng.normal(size=(30, 65))
        y = np.tile([0, 1], 15)
        with threadpool_limits(limits=1):
            for name in ('logistic_regression', 'rbf_svm'):
                model = make_estimator(name, n_jobs=1)
                model.fit(x, y)
                pipeline = model if name == 'logistic_regression' else model.calibrated_classifiers_[
                    0].estimator
                scaler = pipeline.named_steps['preprocessing'].named_transformers_[
                    'numeric']
                np.testing.assert_allclose(
                    scaler.mean_, x[:, NUMERIC_COLUMNS].mean(axis=0))
                before = scaler.mean_.copy()
                scores = positive_scores(model, np.full((5, 65), 1000.))
                np.testing.assert_array_equal(before, scaler.mean_)
                self.assertTrue(np.isfinite(scores).all())
                self.assertTrue(((scores >= 0) & (scores <= 1)).all())
                if name == 'rbf_svm':
                    self.assertFalse(model.ensemble)
                    self.assertEqual(model.cv.n_splits, 3)


class RunnerTests(unittest.TestCase):
    def test_all_models_share_splits_and_export_reloadable_probabilities(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rng = np.random.default_rng(6)
            for number in (0, 1, 2):
                folder = root / 'data' / f'data{number}'
                folder.mkdir(parents=True)
                labels = [i % 2 * (0.25 if number == 2 else 1)
                          for i in range(40)]
                pd.DataFrame({'transcript_id': ['same'] * 40, 'transcript_position': range(40), 'label': labels}).to_csv(
                    folder / ('data.info.labelled' if number == 0 else 'data.info'), index=False)
                (folder / f'dataset{number}.json').write_text(json.dumps({'same': {
                    str(i): {'ACGTACG': rng.normal(loc=labels[i], size=(3, 9)).tolist()} for i in range(40)
                }}))
            params = {
                'extra_trees': [{'n_estimators': 4}, {'n_estimators': 5, 'max_depth': 2}],
                'random_forest': {'n_estimators': 4},
                'gradient_boosting': {'max_iter': 3, 'min_samples_leaf': 2},
                'logistic_regression': {}, 'rbf_svm': {},
            }
            env = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                   'MPLCONFIGDIR': str(root / 'mpl'), 'XDG_CACHE_HOME': str(root / 'cache-home')}
            reference = None
            for model in (*params, 'mil'):
                output = root / model
                command = [sys.executable, str(ROOT / 'train.py'), '--data-dir', str(root / 'data'),
                           '--output-dir', str(output), '--cache-dir', str(root / 'cache'),
                           '--model', model, '--train-datasets', '0', '--n-jobs', '1']
                if model != 'mil':
                    path = root / 'params.json'
                    path.write_text(json.dumps(params[model]))
                    command += ['--model-params', str(path)]
                else:
                    command += ['--epochs', '1', '--batch-size', '32',
                                '--n-reads-per-site', '2', '--eval-iterations', '1']
                result = subprocess.run(
                    command, env=env, capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode, 0,
                                 result.stdout + result.stderr)
                folder = next(output.iterdir())
                assignments = [pd.read_csv(
                    folder / f'dataset{n}_splits.csv') for n in (0, 1, 2)]
                if reference is None:
                    reference = assignments
                else:
                    for expected, actual in zip(reference, assignments):
                        pd.testing.assert_frame_equal(expected, actual)
                predictions = pd.read_csv(
                    folder / 'dataset2_test_predictions.csv')
                self.assertEqual(set(predictions.label), {0, 1})
                self.assertTrue(predictions.score.between(0, 1).all())
                self.assertFalse(predictions.duplicated(
                    ['dataset', 'row_id']).any())
                metrics = json.loads((folder / 'metrics.json').read_text())
                self.assertEqual(
                    metrics['train']['pooled'], metrics['train']['datasets']['0'])
                self.assertEqual(set(metrics['val']['datasets']), {'0', '1'})
                if model != 'mil':
                    saved = joblib.load(folder / 'best_model.joblib')
                    features = np.load(
                        root / 'cache' / 'dataset2' / 'site_features.npy')
                    with threadpool_limits(limits=1):
                        np.testing.assert_allclose(positive_scores(
                            saved['estimator'], features), predictions.score)
            # Reuse the feature cache for full-data training without validation.
            (root /
             'full.json').write_text(json.dumps({'0': [1, 0, 0], '1': [1, 0, 0]}))
            result = subprocess.run([
                sys.executable, str(
                    ROOT / 'train.py'), '--data-dir', str(root / 'data'),
                '--cache-dir', str(root /
                                   'cache'), '--output-dir', str(root / 'full'),
                '--model', 'logistic_regression', '--dataset-splits', str(
                    root / 'full.json'),
                '--val-datasets', '--test-datasets', '2',
            ], env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0,
                             result.stdout + result.stderr)
            self.assertIn('Using cached data0 site features', result.stdout)
            folder = next((root / 'full').iterdir())
            info = json.loads((folder / 'run_info.json').read_text())
            self.assertEqual(info['checkpoint_selection'], 'fixed_parameters')
            self.assertIsNone(info['best_validation_value'])
