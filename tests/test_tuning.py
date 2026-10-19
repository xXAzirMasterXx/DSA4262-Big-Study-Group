"""Synthetic tuning checks. Run explicitly with the optional tuning dependencies."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

if importlib.util.find_spec("optuna") is None:
    raise unittest.SkipTest(
        "Install the tuning dependency group to run tuning tests")

import joblib
import numpy as np
import optuna
import pandas as pd
from optuna.trial import TrialState

from data_prep import split_sites
from tune import (make_objective, open_study, parse_args, sample_parameters,
                  save_sampler, validate_config)
from tuning_data import TuningData

ROOT = Path(__file__).resolve().parents[1]


class StudyTests(unittest.TestCase):
    def test_resume_contract_and_interrupted_trial(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            args = parse_args(["--data-dir", temporary])
            contract = {"test": "fixed"}
            study = open_study(folder, contract, args)
            baseline = study.ask()
            self.assertTrue(baseline.user_attrs["baseline"])
            study.tell(baseline, 0.5)
            save_sampler(folder / "sampler.joblib", study.sampler)
            abandoned = study.ask()
            resumed = open_study(folder, contract, args)
            self.assertEqual(
                resumed.trials[abandoned.number].state, TrialState.FAIL)
            self.assertEqual(resumed.best_trial.number, baseline.number)
            with self.assertRaisesRegex(ValueError, "changed"):
                open_study(folder, {"test": "changed"}, args)

    def test_sampling_applies_numeric_and_special_branches(self):
        config = {"baseline": {"C": 1.0, "gamma": "scale"}, "space": {
            "C": {"type": "float", "low": 0.01, "high": 100, "log": True},
            "gamma": {"type": "float_or", "choices": ["scale"], "low": 0.0001, "high": 1.0},
        }}
        trial = optuna.trial.FixedTrial(
            {"C": 2.0, "gamma__branch": "numeric", "gamma": 0.2})
        self.assertEqual(sample_parameters(trial, config),
                         {"C": 2.0, "gamma": 0.2})
        trial = optuna.trial.FixedTrial({"C": 2.0, "gamma__branch": 0})
        self.assertEqual(sample_parameters(trial, config),
                         {"C": 2.0, "gamma": "scale"})
        trial = optuna.trial.FixedTrial({})
        trial.set_user_attr("baseline", True)
        self.assertEqual(sample_parameters(trial, config), config["baseline"])

    def test_mil_pruning_uses_epoch_validation_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = parse_args(["--data-dir", temporary])
            config = json.loads(
                (ROOT / "configs/tuning.json").read_text())["models"]["mil"]
            trial = optuna.trial.FixedTrial({
                "learning_rate": 0.001, "weight_decay__branch": 0,
                "hidden_dim": 64, "embed_dim": 2, "dropout": 0.1,
            })
            # FixedTrial normally never prunes. Force its pruning signal for this boundary check.
            data = TuningData({"train": None, "val": None},
                              {}, mean=None, std=None)

            def simulated_epoch(*arguments, **kwargs):
                self.assertFalse(kwargs["evaluate_train"])
                kwargs["epoch_callback"](
                    6, {"val": {"pooled": {"pr_auc": 0.2}}}, 0.2)

            with patch.object(trial, "should_prune", return_value=True), patch(
                "tune.train_baseline_model", side_effect=simulated_epoch
            ):
                objective = make_objective(
                    args, "mil", config, data, Path(temporary), "cpu")
                with self.assertRaises(optuna.TrialPruned):
                    objective(trial)
            self.assertIn("last_validation", trial.user_attrs)


class TuningRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        rng = np.random.default_rng(5)
        # No data2 files: optimization must succeed without accessing test-only datasets.
        for number in (0, 1):
            folder = self.root / "data" / f"data{number}"
            folder.mkdir(parents=True)
            labels = np.tile([0, 1], 20)
            pd.DataFrame({"transcript_id": ["same"] * len(labels),
                          "transcript_position": range(len(labels)), "label": labels}).to_csv(
                folder / ("data.info.labelled" if number == 0 else "data.info"), index=False)
            (folder / f"dataset{number}.json").write_text(json.dumps({"same": {
                str(i): {"ACGTACG": rng.normal(loc=int(label), size=(3, 9)).tolist()}
                for i, label in enumerate(labels)
            }}))
        self.config = json.loads((ROOT / "configs/tuning.json").read_text())
        for name, spec in self.config["models"].items():
            spec["trials"] = 1
            if name in ("extra_trees", "random_forest"):
                spec["baseline"]["n_estimators"] = 3
            if name == "gradient_boosting":
                spec["baseline"].update(max_iter=2, min_samples_leaf=2)
            if name == "mil":
                spec.update(baseline_epochs=1, max_epochs=1)
                spec["baseline"].update(hidden_dim=64)
        # Freeze this search to make parameter application independently checkable.
        self.config["models"]["logistic_regression"]["space"]["C"] = {
            "type": "categorical", "choices": [0.2],
        }
        self.config_path = self.root / "config.json"
        self.config_path.write_text(json.dumps(self.config))
        self.env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                    "MPLCONFIGDIR": str(self.root / "mpl"), "XDG_CACHE_HOME": str(self.root / "xdg")}

    def command(self, model, *extra):
        result = subprocess.run([
            sys.executable, str(
                ROOT / "tune.py"), "--data-dir", str(self.root / "data"),
            "--cache-dir", str(self.root /
                               "cache"), "--output-dir", str(self.root / "studies"),
            "--config", str(self.config_path), "--models", model, "--variants", "data0",
            "--n-jobs", "1", "--threads", "1", "--eval-iterations", "1",
            "--n-reads-per-site", "2", "--batch-size", "32", *extra,
        ], env=self.env, capture_output=True, text=True, timeout=120)
        return result

    def test_all_baselines_without_test_access_and_winner_export(self):
        for model, spec in self.config["models"].items():
            validate_config(spec, model)
            result = self.command(model, "--no-final-eval")
            self.assertEqual(result.returncode, 0,
                             result.stdout + result.stderr)
            folder = self.root / "studies" / f"{model}_data0"
            self.assertFalse((folder / "final").exists())
            best = json.loads((folder / "best_trial.json").read_text())
            self.assertEqual(best["parameters"], spec["baseline"])
            self.assertEqual(best["number"], 0)
            self.assertEqual(set(best["validation"]["datasets"]), {"0", "1"})
            sites = pd.read_csv(folder / "dataset0_splits.csv")
            expected = split_sites(pd.DataFrame(
                {"label": np.tile([0, 1], 20)}), seed=42)
            self.assertEqual(sites.split.tolist(), expected.split.tolist())
        # Same study, same budget: no new trials. Then extend the budget by one trial.
        model = "logistic_regression"
        result = self.command(model, "--no-final-eval")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        folder = self.root / "studies" / f"{model}_data0"
        self.assertEqual(len(pd.read_csv(folder / "trials.csv")), 1)
        result = self.command(model, "--no-final-eval", "--trials", "2")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        log = json.loads((folder / "trials" / "trial_00001" /
                         "parameters.json").read_text())
        self.assertEqual(log, {"C": 0.2})
        # Create data2 only after tuning. This permits final evaluation with the original contract.
        import shutil
        shutil.copytree(self.root / "data" / "data1",
                        self.root / "data" / "data2")
        (self.root / "data" / "data2" / "dataset1.json").rename(self.root /
                                                                "data" / "data2" / "dataset2.json")
        result = self.command(model, "--trials", "2")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(pd.read_csv(folder / "trials.csv")), 2)
        best = json.loads((folder / "best_trial.json").read_text())
        output = folder / "final" / f"trial_{best['number']:05d}"
        self.assertTrue((output / "internal_test_metrics.json").exists())
        self.assertTrue((output / "dataset2_test_predictions.csv").exists())
        fitted = joblib.load(output / "best_model.joblib")["estimator"]
        self.assertEqual(
            fitted.named_steps["classifier"].C, best["parameters"]["C"])
        features = np.load(self.root / "cache" /
                           "dataset2" / "site_features.npy")
        predictions = pd.read_csv(output / "dataset2_test_predictions.csv")
        np.testing.assert_allclose(
            fitted.predict_proba(features)[:, 1], predictions.score.to_numpy()
        )
        # Changing the model seed rejects resume but leaves exported split rows intact.
        result = self.command(model, "--no-final-eval", "--model-seed", "9")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("changed", result.stderr)
