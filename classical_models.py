"""Classical classifiers with training-only preprocessing and calibration."""
import json
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from evaluation import partition_metrics, print_metrics, export_predictions, metric_records
from site_features import FEATURE_NAMES, NUMERIC_COLUMNS, prepare_site_features, partition_features


def make_estimator(name, seed=42, balanced=True, n_jobs=-1, calibration_folds=3, params=None):
    params = dict(params or {})
    # Keep probability output, training-only selection, and balancing under runner control.
    reserved = {"class_weight", "random_state", "n_jobs",
                "probability", "kernel", "early_stopping", "warm_start"}
    if reserved.intersection(params):
        raise ValueError(
            f"Runner-controlled parameters cannot be overridden: {sorted(reserved.intersection(params))}")
    common = {"random_state": seed,
              "class_weight": "balanced" if balanced else None}
    if name == "extra_trees":
        estimator = ExtraTreesClassifier(
            **({"n_estimators": 300, "n_jobs": n_jobs, **common, **params}))
    elif name == "random_forest":
        estimator = RandomForestClassifier(
            **({"n_estimators": 300, "n_jobs": n_jobs, **common, **params}))
    elif name == "gradient_boosting":
        # Use the shared validation partition for candidate selection, not an implicit holdout.
        estimator = HistGradientBoostingClassifier(
            **({"max_iter": 100, "early_stopping": False, **common, **params}))
    elif name == "logistic_regression":
        estimator = LogisticRegression(
            **({"max_iter": 2000, **common, **params}))
    elif name == "rbf_svm":
        estimator = SVC(**({"kernel": "rbf", **common, **params}))
    else:
        raise ValueError(f"Unknown classical model: {name}")
    if name in ("logistic_regression", "rbf_svm"):
        preprocessing = ColumnTransformer(
            [("numeric", StandardScaler(), NUMERIC_COLUMNS)], remainder="passthrough"
        )
        estimator = Pipeline(
            [("preprocessing", preprocessing), ("classifier", estimator)])
    if name == "rbf_svm":
        # The pipeline is cloned inside each fold so its scaler cannot see held-out rows.
        # No calibration sample weights: calibrate to the natural training distribution.
        estimator = CalibratedClassifierCV(
            estimator, method="sigmoid", ensemble=False, n_jobs=n_jobs,
            cv=StratifiedKFold(calibration_folds,
                               shuffle=True, random_state=seed),
        )
    return estimator


def positive_scores(estimator, features):
    positive_column = np.flatnonzero(estimator.classes_ == 1)
    if len(positive_column) != 1:
        raise ValueError("Classifier must include positive class 1")
    return estimator.predict_proba(features)[:, positive_column[0]]


def train_classical(args, prepared, tables, partitions, read_maps, run_dir, run_info):
    candidates = json.loads(args.model_params.read_text()
                            ) if args.model_params else {}
    candidates = [candidates] if isinstance(candidates, dict) else candidates
    if not isinstance(candidates, list) or not candidates or not all(isinstance(p, dict) for p in candidates):
        raise ValueError(
            "--model-params must contain an object or a nonempty list of objects")
    has_validation = not partitions["val"].empty
    if len(candidates) > 1 and not has_validation:
        raise ValueError(
            "Multiple parameter candidates require validation datasets")
    labels = partitions["train"]["label"].to_numpy(dtype=int)
    counts = np.bincount(labels, minlength=2)
    if np.any(counts == 0):
        raise ValueError("Classical training requires both binary classes")
    if args.model == "rbf_svm" and counts.min() < args.calibration_folds:
        raise ValueError(
            "Each training class must have at least --calibration-folds rows")
    estimators = [make_estimator(args.model, args.seed, not args.no_balanced_sampler,
                                 args.n_jobs, args.calibration_folds, p) for p in candidates]
    features = prepare_site_features(prepared, tables, read_maps)
    x_train = partition_features(partitions["train"], features)
    x_val = partition_features(
        partitions["val"], features) if has_validation else None
    best_model, best_value, best_index = None, None, None
    history = []
    if args.model == "rbf_svm":
        print("Fitting exact RBF SVM with training-only calibration; no rows are subsampled.", flush=True)
    for index, estimator in enumerate(estimators):
        print(
            f"Fitting {args.model} candidate {index + 1}/{len(estimators)}", flush=True)
        started = time.perf_counter()
        estimator.fit(x_train, labels)
        metrics = partition_metrics(partitions["val"], positive_scores(
            estimator, x_val)) if has_validation else None
        value = metrics["pooled"][args.selection_metric] if has_validation else None
        if has_validation:
            print_metrics(f"Candidate {index + 1} val", metrics)
            if value is None:
                raise ValueError("Validation selection metric is undefined")
        improved = best_model is None or (has_validation and (
            value < best_value if args.selection_metric == "loss" else value > best_value
        ))
        if improved:
            best_model, best_value, best_index = estimator, value, index
        history.append({"candidate": index, "parameters": candidates[index],
                        "fit_and_validation_seconds": time.perf_counter() - started,
                        "selected_as_best": improved, "val": metrics})
        # Release losing fitted candidates, especially large forests and SVMs.
        estimators[index] = None
        (run_dir / "search_history.json").write_text(json.dumps(history,
                                                                indent=2, allow_nan=False))
    joblib.dump({"estimator": best_model, "feature_names": FEATURE_NAMES,
                 "model": args.model, "sklearn_version": sklearn.__version__}, run_dir / "best_model.joblib")
    (run_dir / "feature_names.json").write_text(json.dumps(FEATURE_NAMES, indent=2))
    final_metrics, rows = {}, []
    for split, sites in partitions.items():
        if sites.empty:
            continue
        x = x_train if split == "train" else x_val if split == "val" else partition_features(
            sites, features)
        scores = positive_scores(best_model, x)
        metrics = partition_metrics(sites, scores)
        final_metrics[split] = metrics
        print_metrics(f"Selected {args.model} {split}", metrics)
        export_predictions(run_dir, split, sites, scores)
        rows.extend(metric_records(split, metrics, "site_features"))
    pd.DataFrame(rows).to_csv(run_dir / "metrics.csv", index=False)
    (run_dir / "metrics.json").write_text(json.dumps(final_metrics, indent=2, allow_nan=False))
    run_info.update({
        "checkpoint_selection": "validation" if has_validation else "fixed_parameters",
        "best_candidate": best_index, "best_validation_value": best_value,
        "model_parameters": candidates[best_index],
        "resolved_estimator_parameters": {key: value if isinstance(value, (str, int, float, bool, type(None))) else repr(value)
                                          for key, value in best_model.get_params(deep=True).items()},
        "balanced_class_weights": not args.no_balanced_sampler,
        "calibration": {"method": "sigmoid", "folds": args.calibration_folds, "source": "training_only"} if args.model == "rbf_svm" else None,
        "feature_count": len(FEATURE_NAMES), "sklearn_version": sklearn.__version__,
    })
    (run_dir / "run_info.json").write_text(json.dumps(run_info, indent=2, allow_nan=False))
