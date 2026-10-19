import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
import torch

from export_model import export_model
from mil_model import M6ANetInspiredMIL
from model_bundle import load_bundle
from predict import predict
from site_features import FEATURE_NAMES, summarize_site


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "run"
        self.source.mkdir()
        self.input = self.root / "input.json"
        self.reads = np.arange(27, dtype=np.float32).reshape(3, 9)
        self.input.write_text(json.dumps({"tx": {
            "1": {"ACGTACG": self.reads.tolist()},
            "2": {"ACGTACG": (self.reads + 1).tolist()},
        }}))

    def classical_bundle(self):
        values = np.stack([summarize_site(self.reads + i, "ACGTACG")
                          for i in range(8)])
        estimator = LogisticRegression().fit(values, [0, 0, 0, 0, 1, 1, 1, 1])
        joblib.dump({"estimator": estimator, "model": "logistic_regression",
                     "feature_names": FEATURE_NAMES, "sklearn_version": sklearn.__version__},
                    self.source / "best_model.joblib")
        bundle = export_model(self.source, self.root / "bundle")
        return bundle, estimator, values

    def test_classical_matches_saved_estimator_without_labels(self):
        bundle, estimator, values = self.classical_bundle()
        output = self.root / "scores.csv"
        self.assertEqual(predict(bundle, self.input, output, batch_size=1), 2)
        result = pd.read_csv(output)
        np.testing.assert_allclose(
            result.score, estimator.predict_proba(values[:2])[:, 1])
        self.assertNotIn("label", result.columns)
        self.assertTrue(output.with_suffix(".csv.json").is_file())
        with self.assertRaisesRegex(ValueError, "already exists"):
            predict(bundle, self.input, output)

    def test_corruption_and_lfs_pointer(self):
        bundle, _, _ = self.classical_bundle()
        artifact = bundle / "model.joblib"
        artifact.write_bytes(artifact.read_bytes() + b"corrupt")
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_bundle(bundle)
        artifact.write_text("version https://git-lfs.github.com/spec/v1\n")
        with self.assertRaisesRegex(ValueError, "git lfs pull"):
            load_bundle(bundle)

    def test_bad_input_does_not_publish_partial_predictions(self):
        bundle, _, _ = self.classical_bundle()
        self.input.write_text(self.input.read_text() + self.input.read_text())
        output = self.root / "scores.csv"
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            predict(bundle, self.input, output, batch_size=1)
        self.assertFalse(output.exists())
        self.assertFalse(list(self.root.glob(".predictions-*")))

    def test_mil_sampling_is_independent_of_batch_size(self):
        params = {"embed_dim": 2, "hidden_dim": 8,
                  "dropout": 0.1, "n_reads_per_site": 2}
        model = M6ANetInspiredMIL(2, 8, 0.1)
        torch.save({"baseline_parameters": params, "model_state_dict": model.state_dict(),
                    "feature_mean": torch.ones(9), "feature_std": torch.full((9,), 2.0)},
                   self.source / "best_model.pt")
        bundle = export_model(self.source, self.root /
                              "bundle", eval_iterations=3, seed=17)
        first, second = self.root / "first.csv", self.root / "second.csv"
        predict(bundle, self.input, first, batch_size=1)
        predict(bundle, self.input, second, batch_size=2)
        np.testing.assert_allclose(pd.read_csv(
            first).score, pd.read_csv(second).score, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
