"""Resumable Optuna studies. Trials only receive training and validation data."""
import argparse
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import shutil
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from threadpoolctl import threadpool_limits

try:
    import optuna
    from optuna.trial import TrialState
except ImportError as error:
    raise SystemExit(
        "Install tuning dependencies: uv sync --locked --group tuning") from error

from classical_models import make_estimator, positive_scores
from data_prep import validate_ratios
from evaluation import partition_metrics, export_predictions, metric_records, compute_metrics
from mil_model import M6ANetInspiredMIL
from site_features import FEATURE_NAMES, prepare_site_features, partition_features
from train import train_baseline_model, evaluate_partition, save_curves
from tuning_data import prepare_tuning_data, load_partitions, select_partition, sha256, data_fingerprints

ROOT = Path(__file__).resolve().parent
MODELS = ("extra_trees", "random_forest", "gradient_boosting",
          "logistic_regression", "mil")
VARIANTS = {"data0": [0], "data1": [1], "data01": [0, 1], "data012": [0, 1, 2]}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def save_sampler(path, sampler):
    temporary = path.with_suffix(".tmp")
    joblib.dump(sampler, temporary)
    temporary.replace(path)


@contextmanager
def study_lock(folder):
    """One local writer per study; the OS releases the lock after an interruption."""
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"Study is already running: {folder}") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def sample_parameters(trial, config):
    if trial.user_attrs.get("baseline"):
        return dict(config["baseline"])
    parameters = {}
    for name, spec in config["space"].items():
        kind = spec["type"]
        if kind == "categorical":
            value = trial.suggest_categorical(name, spec["choices"])
        elif kind == "int":
            value = trial.suggest_int(name, spec["low"], spec["high"],
                                      step=spec.get("step", 1), log=spec.get("log", False))
        elif kind == "float":
            value = trial.suggest_float(
                name, spec["low"], spec["high"], log=spec.get("log", False))
        elif kind == "float_or":
            # Separate branch and numeric distributions avoid mixing sentinel strings with floats.
            branch = trial.suggest_categorical(
                name + "__branch", ["numeric", *range(len(spec["choices"]))])
            value = (trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
                     if branch == "numeric" else spec["choices"][branch])
        else:
            raise ValueError(f"Unknown distribution type: {kind}")
        parameters[name] = value
    return parameters


def validate_config(config, model):
    if not isinstance(config, dict) or not isinstance(config.get("baseline"), dict) or not isinstance(config.get("space"), dict):
        raise ValueError(f"Invalid baseline/search configuration for {model}")
    if config.get("trials", 0) < 1:
        raise ValueError("Trial budget must be positive")
    if set(config["baseline"]) != set(config["space"]):
        raise ValueError(
            "Baseline and search-space parameter names must match")
    if model == "mil":
        allowed = {"learning_rate", "weight_decay",
                   "hidden_dim", "embed_dim", "dropout"}
        if set(config["space"]) != allowed:
            raise ValueError(
                "MIL search must specify learning rate, weight decay, dimensions and dropout")
        if min(config.get("baseline_epochs", 0), config.get("max_epochs", 0)) < 1:
            raise ValueError("MIL epoch budgets must be positive")
    else:
        make_estimator(model, params=config["baseline"])
    # Validate distributions without starting a study or fitting a model.
    for spec in config["space"].values():
        kind = spec.get("type")
        if kind == "categorical":
            optuna.distributions.CategoricalDistribution(spec["choices"])
        elif kind == "int":
            optuna.distributions.IntDistribution(spec["low"], spec["high"],
                                                 log=spec.get("log", False), step=spec.get("step", 1))
        elif kind in ("float", "float_or"):
            optuna.distributions.FloatDistribution(
                spec["low"], spec["high"], log=spec.get("log", False))
            if kind == "float_or" and not spec.get("choices"):
                raise ValueError(
                    "float_or requires at least one special value")
        else:
            raise ValueError(f"Unknown distribution type: {kind}")


def study_contract(args, model, train_numbers, config, fingerprint, device):
    code_files = ("tune.py", "tuning_data.py", "train.py", "classical_models.py",
                  "data_prep.py", "evaluation.py", "site_features.py", "mil_model.py")
    return {
        "protocol_version": 1, "search_version": args.search_version, "model": model,
        "train_datasets": train_numbers, "val_datasets": args.val_datasets,
        "test_datasets": args.test_datasets, "split_ratios": args.split_ratios,
        "split_seed": args.split_seed, "model_seed": args.model_seed,
        "sampler_seed": args.sampler_seed, "evaluation_seed": args.evaluation_seed,
        "balanced": not args.no_class_weight, "selection_metric": "pr_auc",
        "search": {k: v for k, v in config.items() if k != "trials"},
        "n_reads_per_site": args.n_reads_per_site, "batch_size": args.batch_size,
        "eval_iterations": args.eval_iterations,
        "startup_trials": args.startup_trials, "prune_warmup_epochs": args.prune_warmup_epochs,
        "n_jobs": args.n_jobs, "threads": args.threads, "device": str(device),
        "data_fingerprints": fingerprint,
        "code_sha256": {name: sha256(ROOT / name) for name in code_files},
        "versions": {"optuna": optuna.__version__, "sklearn": sklearn.__version__,
                     "torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__},
    }


def open_study(folder, contract, args):
    sampler_path = folder / "sampler.joblib"
    sampler = (joblib.load(sampler_path) if sampler_path.exists() else
               optuna.samplers.TPESampler(seed=args.sampler_seed, n_startup_trials=args.startup_trials))
    study = optuna.create_study(
        study_name=folder.name, storage="sqlite:///" +
        str((folder / "study.db").resolve()),
        direction="maximize", sampler=sampler,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=args.startup_trials,
                                           n_warmup_steps=args.prune_warmup_epochs),
        load_if_exists=True,
    )
    existing = study.user_attrs.get("contract")
    if existing is not None and existing != contract:
        raise ValueError(
            "Study inputs, seeds, code, versions or search space changed. Use a new --output-dir.")
    if existing is None:
        if study.trials:
            raise ValueError(
                "Existing trials have no compatibility contract; use a new output directory")
        study.set_user_attr("contract", contract)
    # The exclusive lock proves no local writer owns these interrupted trials.
    for trial in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
        study.tell(trial.number, state=TrialState.FAIL)
    trials = study.get_trials(deepcopy=False)
    if trials and not sampler_path.exists():
        raise ValueError(
            "Sampler checkpoint is missing; use a new output directory for a reproducible study")
    if not any(t.user_attrs.get("baseline") and t.state in (TrialState.COMPLETE, TrialState.WAITING) for t in trials):
        study.enqueue_trial({}, user_attrs={"baseline": True})
    write_json(folder / "study_info.json", contract)
    save_sampler(sampler_path, study.sampler)
    return study


def mil_parameters(args, config, trial, parameters):
    return {**parameters,
            "epochs": config["baseline_epochs"] if trial.user_attrs.get("baseline") else config["max_epochs"],
            "batch_size": args.batch_size, "n_reads_per_site": args.n_reads_per_site,
            "balanced": not args.no_class_weight}


def make_objective(args, model, config, data, folder, device):
    """No test table, test feature matrix or test evaluator is available here."""
    def objective(trial):
        started = time.perf_counter()
        trial_dir = folder / "trials" / f"trial_{trial.number:05d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        params = sample_parameters(trial, config)
        trial.set_user_attr("resolved_parameters", params)
        trial.set_user_attr("model_seed", args.model_seed)
        write_json(trial_dir / "parameters.json", params)
        try:
            if model == "mil":
                resolved = mil_parameters(args, config, trial, params)
                trial.set_user_attr("fit_parameters", resolved)

                def epoch_finished(epoch, metrics, best_value):
                    trial.set_user_attr("last_validation", metrics["val"])
                    trial.report(best_value, step=epoch)
                    if not trial.user_attrs.get("baseline") and trial.should_prune():
                        raise optuna.TrialPruned(f"Pruned at epoch {epoch}")

                fitted, epoch, value = train_baseline_model(
                    data.partitions, data.read_maps, data.mean, data.std, resolved,
                    device, args.model_seed, trial_dir, args.eval_iterations, "pr_auc",
                    epoch_callback=epoch_finished, evaluate_train=False, plot_curves=False,
                    evaluation_seed=args.evaluation_seed,
                )
                _, metrics = evaluate_partition(
                    fitted, data.partitions["val"], data.read_maps, data.mean, data.std,
                    resolved, device, args.eval_iterations, args.evaluation_seed + 1,
                )
                trial.set_user_attr("best_epoch", epoch)
            else:
                fitted = make_estimator(model, args.model_seed, not args.no_class_weight,
                                        args.n_jobs, params)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    fitted.fit(
                        data.x_train, data.partitions["train"].label.to_numpy(dtype=int))
                trial.set_user_attr(
                    "warnings", [str(w.message) for w in caught])
                metrics = partition_metrics(
                    data.partitions["val"], positive_scores(fitted, data.x_val))
                value = metrics["pooled"]["pr_auc"]
                completed = trial.study.get_trials(
                    deepcopy=False, states=(TrialState.COMPLETE,))
                if not completed or value > trial.study.best_value:
                    joblib.dump({"estimator": fitted, "feature_names": FEATURE_NAMES,
                                 "model": model, "sklearn_version": sklearn.__version__},
                                trial_dir / "best_model.joblib")
            trial.set_user_attr("validation", metrics)
            write_json(trial_dir / "validation.json", metrics)
            if value is None or not np.isfinite(value):
                raise ValueError("Validation PR AUC must be finite")
            return value
        except Exception as error:
            trial.set_user_attr("error", f"{type(error).__name__}: {error}")
            raise
        finally:
            trial.set_user_attr("fit_validation_seconds",
                                time.perf_counter() - started)
    return objective


def persist_progress(study, folder):
    save_sampler(folder / "sampler.joblib", study.sampler)
    study.trials_dataframe().to_csv(folder / "trials.csv", index=False)
    completed = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
    best = study.best_trial if completed else None
    if best is not None:
        write_json(folder / "best_parameters.json",
                   best.user_attrs["resolved_parameters"])
        write_json(folder / "best_trial.json", {
            "number": best.number, "validation_pr_auc": best.value,
            "parameters": best.user_attrs["resolved_parameters"],
            "validation": best.user_attrs["validation"],
            "artifact_directory": str(folder / "trials" / f"trial_{best.number:05d}"),
        })
    # Retain trial logs, but only retain the winning fitted model to limit disk use.
    for trial in study.get_trials(deepcopy=False):
        if trial.state.is_finished() and (best is None or trial.number != best.number):
            for name in ("best_model.pt", "best_model.joblib"):
                (folder / "trials" /
                 f"trial_{trial.number:05d}" / name).unlink(missing_ok=True)


def export_winner(args, model, train_numbers, study, folder, device):
    """Evaluate the selected checkpoint after optimization."""
    best = study.best_trial
    source = folder / "trials" / f"trial_{best.number:05d}"
    output = folder / "final" / f"trial_{best.number:05d}"
    numbers = sorted(
        set(train_numbers + args.val_datasets + args.test_datasets))
    prepared, tables, read_maps = load_partitions(args, numbers)
    fingerprints = data_fingerprints(prepared, tables)
    for number, previous in study.user_attrs["contract"]["data_fingerprints"].items():
        if fingerprints[number] != previous:
            raise ValueError(
                "Training/validation inputs changed during tuning; winner export refused")
    if (output / "run_info.json").exists():
        previous = json.loads((output / "run_info.json").read_text())
        if previous.get("data_fingerprints") != fingerprints:
            raise ValueError(
                "Inputs changed since the saved final evaluation; use a new output directory")
        print(f"Winner already evaluated: {output}", flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    partitions = {split: select_partition(tables, selected, split)
                  for split, selected in (("train", train_numbers), ("val", args.val_datasets), ("test", args.test_datasets))}
    for number, table in tables.items():
        table.to_csv(output / f"dataset{number}_splits.csv", index=False)
    if model == "mil":
        checkpoint = torch.load(source / "best_model.pt",
                                map_location="cpu", weights_only=True)
        params = checkpoint["baseline_parameters"]
        fitted = M6ANetInspiredMIL(
            params["embed_dim"], params["hidden_dim"], params["dropout"]).to(device)
        fitted.load_state_dict(checkpoint["model_state_dict"])
        mean = checkpoint["feature_mean"].numpy()
        std = checkpoint["feature_std"].numpy()
        shutil.copy2(source / "best_model.pt", output / "best_model.pt")
        shutil.copy2(source / "history.json", output / "history.json")
        save_curves(json.loads((source / "history.json").read_text()),
                    output, checkpoint["best_epoch"])
    else:
        fitted = joblib.load(source / "best_model.joblib")["estimator"]
        features = prepare_site_features(prepared, tables, read_maps)
        shutil.copy2(source / "best_model.joblib",
                     output / "best_model.joblib")
        write_json(output / "feature_names.json", FEATURE_NAMES)
    metrics_by_split, rows = {}, []
    for split, sites in partitions.items():
        if sites.empty:
            continue
        if model == "mil":
            scores, metrics = evaluate_partition(
                fitted, sites, read_maps, mean, std, params, device, args.eval_iterations,
                args.evaluation_seed + ("train", "val", "test").index(split),
            )
        else:
            scores = positive_scores(
                fitted, partition_features(sites, features))
            metrics = partition_metrics(sites, scores)
        metrics_by_split[split] = metrics
        export_predictions(output, split, sites, scores)
        rows.extend(metric_records(split, metrics,
                    "noisy_or" if model == "mil" else "site_features"))
        if split == "test":
            mask = sites.dataset.isin([0, 1]).to_numpy()
            if mask.any():
                internal = compute_metrics(
                    sites.loc[mask, "label"], scores[mask])
                write_json(output / "internal_test_metrics.json", internal)
                rows.append({"split": "test", "dataset": "0+1", **internal})
    write_json(output / "metrics.json", metrics_by_split)
    pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
    write_json(output / "run_info.json", {
        "model": model, "trained_on_datasets": train_numbers,
        "datasets": {"train": train_numbers, "val": args.val_datasets, "test": args.test_datasets},
        "seed": args.split_seed, "model_seed": args.model_seed,
        "split_ratios": args.split_ratios, "selection_metric": "pr_auc",
        "eval_iterations": args.eval_iterations,
        "best_trial": best.number, "best_validation_value": best.value,
        "best_epoch": best.user_attrs.get("best_epoch"),
        "baseline_parameters": best.user_attrs.get("fit_parameters"),
        "parameters": best.user_attrs["resolved_parameters"],
        "study_contract": study.user_attrs["contract"], "data2_used_for_selection": 2 in args.val_datasets,
        "data_fingerprints": fingerprints,
    })
    print(f"Saved selected model and evaluation: {output}", flush=True)


def run_study(args, model, train_numbers, config):
    name = model + "_data" + "-".join(map(str, train_numbers))
    folder = args.output_dir.resolve() / name
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if model != "mil":
        device = torch.device("cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    with study_lock(folder), threadpool_limits(limits=args.threads):
        data, tables, fingerprint = prepare_tuning_data(
            args, model, train_numbers)
        contract = study_contract(
            args, model, train_numbers, config, fingerprint, device)
        study = open_study(folder, contract, args)
        for n, table in tables.items():
            table.to_csv(folder / f"dataset{n}_splits.csv", index=False)
        target = args.trials if args.trials is not None else config["trials"]
        attempted = sum(t.state.is_finished()
                        for t in study.get_trials(deepcopy=False))
        remaining = max(0, target - attempted)
        write_json(folder / "budget.json",
                   {"target_trials": target, "timeout_seconds_this_invocation": args.timeout})
        print(
            f"{name}: {remaining} trials remaining within total budget {target}", flush=True)
        try:
            if remaining:
                study.optimize(make_objective(args, model, config, data, folder, device),
                               n_trials=remaining, timeout=args.timeout, n_jobs=1,
                               callbacks=[
                                   lambda study, trial: persist_progress(study, folder)],
                               gc_after_trial=True)
        finally:
            persist_progress(study, folder)
        if study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)) and not args.no_final_eval:
            export_winner(args, model, train_numbers, study, folder, device)
    return folder


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("runs/cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/tuning"))
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/tuning.json")
    parser.add_argument("--models", choices=MODELS,
                        nargs="+", default=list(MODELS))
    parser.add_argument("--variants", choices=tuple(VARIANTS),
                        nargs="+", default=["data012"])
    parser.add_argument("--train-datasets", type=int, nargs="+",
                        help="Custom training sources instead of --variants")
    parser.add_argument("--val-datasets", type=int,
                        nargs="+", default=[0, 1, 2])
    parser.add_argument("--test-datasets", type=int,
                        nargs="*", default=[0, 1, 2])
    parser.add_argument("--split-ratios", type=float,
                        nargs=3, default=[0.7, 0.15, 0.15])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--sampler-seed", type=int, default=42)
    parser.add_argument("--evaluation-seed", type=int, default=42)
    parser.add_argument(
        "--trials", type=int, help="Total attempted trials per study, including baseline and previous trials")
    parser.add_argument("--timeout", type=float,
                        help="Seconds per study per invocation; does not interrupt a running fit")
    parser.add_argument("--startup-trials", type=int, default=5)
    parser.add_argument("--prune-warmup-epochs", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Jobs inside each forest")
    parser.add_argument("--threads", type=int, default=1,
                        help="BLAS/OpenMP/PyTorch CPU threads")
    parser.add_argument("--no-class-weight", action="store_true")
    parser.add_argument("--n-reads-per-site", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-iterations", type=int, default=5)
    parser.add_argument("--device", choices=("auto",
                        "cpu", "cuda"), default="auto")
    parser.add_argument("--no-final-eval", action="store_true",
                        help="Save study winner without final test evaluation")
    args = parser.parse_args(argv)
    for key in ("threads", "startup_trials", "prune_warmup_epochs", "n_reads_per_site", "batch_size", "eval_iterations"):
        if getattr(args, key) < 1:
            parser.error(f"{key} must be positive")
    if args.n_jobs == 0:
        parser.error(
            "n-jobs must be nonzero")
    if args.trials is not None and args.trials < 1:
        parser.error("trials must be positive")
    if args.timeout is not None and (not np.isfinite(args.timeout) or args.timeout <= 0):
        parser.error("timeout must be finite and positive")
    sources = (args.train_datasets or []) + \
        args.val_datasets + args.test_datasets
    if any(n < 0 for n in sources):
        parser.error("Dataset IDs must be nonnegative")
    try:
        validate_ratios(args.split_ratios)
    except ValueError as error:
        parser.error(str(error))
    if args.split_ratios[0] <= 0 or args.split_ratios[1] <= 0:
        parser.error(
            "Tuning requires nonempty train and validation partitions")
    return args


def main():
    args = parse_args()
    config = json.loads(args.config.read_text())
    args.search_version = config["version"]
    for model in args.models:
        validate_config(config["models"][model], model)
    torch.set_num_threads(args.threads)
    variants = ([sorted(set(args.train_datasets))] if args.train_datasets else
                [VARIANTS[name] for name in dict.fromkeys(args.variants)])
    for model in dict.fromkeys(args.models):
        for train_numbers in variants:
            run_study(args, model, train_numbers, config["models"][model])


if __name__ == "__main__":
    main()
