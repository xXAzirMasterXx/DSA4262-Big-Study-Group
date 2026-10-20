import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from evaluation import partition_metrics


class Data2MetricsTests(unittest.TestCase):
    def test_inverts_only_data2_metrics(self):
        labels = np.array([0, 1, 1, 1])
        scores = np.array([0.4, 0.2, 0.7, 0.9])
        sites = pd.DataFrame({'dataset': [2]*4, 'label': labels})
        metrics = partition_metrics(sites, scores)
        p, r, _ = precision_recall_curve(1-labels, 1-scores)
        self.assertAlmostEqual(metrics['datasets']['2']['pr_auc'], auc(r, p))
        self.assertAlmostEqual(
            metrics['datasets']['2']['roc_auc'], roc_auc_score(labels, scores))
        self.assertEqual(metrics['datasets']['2']['auc_positive_sites'], 1)
        self.assertEqual(metrics['pooled']['auc_positive_sites'], 3)
        sites['dataset'] = 0
        other = partition_metrics(sites, scores)
        self.assertEqual(other['datasets']['0']['auc_positive_sites'], 3)
