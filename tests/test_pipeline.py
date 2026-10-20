import unittest
import numpy as np
import pandas as pd

from data_prep import normalize_labels, split_sites
from train import compute_metrics


class PipelineTests(unittest.TestCase):
    def test_row_splits_are_fixed_and_stratified(self):
        rows = pd.DataFrame({'label': [0, 1] * 100, 'transcript_id': ['same'] * 200,
                             'transcript_position': [1] * 200})
        tables = {n: split_sites(rows, seed=42 + n).assign(dataset=n)
                  for n in (0, 1)}
        for table in tables.values():
            self.assertEqual(table['split'].value_counts().to_dict(), {
                             'train': 140, 'val': 30, 'test': 30})
            self.assertTrue(table.groupby('split')[
                            'label'].mean().eq(0.5).all())
        combined = pd.concat(tables.values())
        for n in (0, 1):
            pd.testing.assert_frame_equal(
                combined[combined.dataset == n], tables[n])
            pd.testing.assert_frame_equal(split_sites(
                rows, seed=42 + n), tables[n].drop(columns='dataset'))

    def test_full_partitions_and_invalid_ratios(self):
        rows = pd.DataFrame({'label': [0, 1]})
        for ratios, expected in [([1, 0, 0], 'train'), ([0, 0, 1], 'test')]:
            self.assertTrue(split_sites(rows, ratios=ratios)
                            ['split'].eq(expected).all())
        for ratios in ([0, 0, 0], [-1, 1, 1], [np.nan, 0, 1]):
            with self.assertRaises(ValueError):
                split_sites(rows, ratios=ratios)

    def test_binary_data2(self):
        rows = normalize_labels(pd.DataFrame({'label': [0, 0.01, 0.5, 1]}), 2)
        self.assertEqual(rows.label.tolist(), [0, 1, 1, 1])
        metrics = compute_metrics(rows.label, [0.1, 0.8, 0.9, 1.0])
        self.assertEqual(metrics['roc_auc'], 1.0)
        self.assertNotIn('rmse', metrics)
        with self.assertRaises(ValueError):
            normalize_labels(pd.DataFrame({'label': [0.5]}), 0)


if __name__ == '__main__':
    unittest.main()
