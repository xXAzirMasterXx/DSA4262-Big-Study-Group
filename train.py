import argparse
from datetime import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from data_prep import dataset_files, normalize_labels, prepare_datasets, split_sites, validate_ratios
from evaluation import compute_metrics, partition_metrics, print_metrics, export_predictions, metric_records
from mil_model import M6ANetInspiredMIL, SiteBagDataset

SPLITS = ("train", "val", "test")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def open_read_maps(prepared, sites):
    read_maps = {}
    for dataset_number, (_, reads_path) in prepared.items():
        n_reads = int(sites.loc[sites["dataset"] ==
                      dataset_number, "n_reads"].sum())
        read_maps[dataset_number] = np.memmap(
            reads_path,
            dtype=np.float32,
            mode="r",
            shape=(n_reads, 9),
        )
    return read_maps


def feature_statistics(train_sites, read_maps):
    total = 0
    sums = np.zeros(9, dtype=np.float64)
    sums_sq = np.zeros(9, dtype=np.float64)
    for row in train_sites.itertuples(index=False):
        reads = read_maps[int(row.dataset)][
            int(row.read_start): int(row.read_start) + int(row.n_reads)
        ]
        values = np.asarray(reads, dtype=np.float64)
        sums += values.sum(axis=0)
        sums_sq += np.square(values).sum(axis=0)
        total += len(values)
    mean = sums / max(total, 1)
    variance = np.maximum(sums_sq / max(total, 1) - np.square(mean), 1e-12)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def make_loader(
    sites, read_maps, n_reads, mean, std, batch_size, shuffle, balanced=False
):
    dataset = SiteBagDataset(sites, read_maps, n_reads, mean, std)
    if balanced:
        labels = sites["label"].to_numpy(dtype=int)
        counts = np.bincount(labels, minlength=2)
        if np.any(counts == 0):
            raise ValueError(
                "Both label classes must be present in the training split")
        sample_weights = np.where(
            labels == 1, 1.0 / counts[1], 1.0 / counts[0])
        sampler = WeightedRandomSampler(
            torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=2 * int(max(counts)),
            replacement=True,
        )
        return dataset, DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return dataset, DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def score_dataset(model, loader, n_sites, device, iterations=1):
    scores = np.zeros(n_sites, dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for _ in range(iterations):
            for indices, reads, kmers, _labels in loader:
                probabilities = (
                    model(
                        reads.to(device),
                        kmers.to(device),
                    )
                    .cpu()
                    .numpy()
                )
                scores[indices.numpy()] += probabilities
    return scores / iterations


def evaluate(
    model, sites, read_maps, n_reads, mean, std, batch_size, device, iterations, seed=42
):
    # Use the same evaluation bags each epoch without changing training RNGs.
    numpy_state, torch_state = np.random.get_state(), torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        np.random.seed(seed)
        torch.manual_seed(seed)
        _, loader = make_loader(
            sites, read_maps, n_reads, mean, std, batch_size, shuffle=False
        )
        return score_dataset(model, loader, len(sites), device, iterations)
    finally:
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def evaluate_partition(
    model, sites, read_maps, mean, std, params, device, iterations, seed
):
    sites = sites.reset_index(drop=True)
    scores = evaluate(
        model,
        sites,
        read_maps,
        params["n_reads_per_site"],
        mean,
        std,
        params["batch_size"],
        device,
        iterations,
        seed,
    )
    return scores, partition_metrics(sites, scores)


def train_baseline_model(
    partitions,
    read_maps,
    mean,
    std,
    params,
    device,
    seed,
    output_dir,
    iterations,
    selection_metric,
    log_every_batches=100,
):
    set_seed(seed)
    model = M6ANetInspiredMIL(
        embed_dim=params["embed_dim"],
        hidden_dim=params["hidden_dim"],
        dropout=params["dropout"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"],
    )
    _, loader = make_loader(
        partitions["train"],
        read_maps,
        params["n_reads_per_site"],
        mean,
        std,
        params["batch_size"],
        shuffle=not params["balanced"],
        balanced=params["balanced"],
    )
    criterion = nn.BCELoss()
    best_value, best_state, best_epoch = None, None, None
    history = []
    for epoch in range(1, params["epochs"] + 1):
        model.train()
        epoch_start = time.perf_counter()
        loss_sum, processed = 0.0, 0
        for batch_number, (_, reads, kmers, labels) in enumerate(loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            predictions = model(reads.to(device), kmers.to(device))
            loss = criterion(predictions.clamp(
                1e-7, 1 - 1e-7), labels.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(labels)
            processed += len(labels)
            if batch_number % log_every_batches == 0 or batch_number == len(loader):
                print(
                    f"Epoch {epoch}/{params['epochs']} | batch {batch_number}/{len(loader)} | "
                    f"sampled training BCE={loss_sum / processed:.6f}",
                    flush=True,
                )
        # Natural, unweighted site distributions are used for all reported metrics.
        epoch_metrics = {}
        for name in ("train", "val"):
            if partitions[name].empty:
                continue
            _, metrics = evaluate_partition(
                model,
                partitions[name],
                read_maps,
                mean,
                std,
                params,
                device,
                iterations,
                seed + SPLITS.index(name),
            )
            epoch_metrics[name] = metrics
            print_metrics(f"Epoch {epoch} {name}", metrics)
        has_validation = "val" in epoch_metrics
        value = epoch_metrics["val"]["pooled"][selection_metric] if has_validation else None
        if has_validation and value is None:
            raise ValueError(
                f"Validation {selection_metric} is undefined; use --selection-metric loss"
            )
        improved = not has_validation or best_value is None or (
            value < best_value if selection_metric == "loss" else value > best_value
        )
        if improved:
            best_value, best_epoch = value, epoch
            best_state = {
                key: tensor.detach().cpu().clone()
                for key, tensor in model.state_dict().items()
            }
            torch.save(
                {
                    "model_state_dict": best_state,
                    "baseline_parameters": params,
                    "feature_mean": torch.from_numpy(mean),
                    "feature_std": torch.from_numpy(std),
                    "best_epoch": best_epoch,
                    "selection_metric": selection_metric if has_validation else None,
                    "validation_value": best_value,
                    "auc_target": "label > 0",
                    "prior_correction": False,
                },
                output_dir / "best_model.pt",
            )
        history.append(
            {
                "epoch": epoch,
                "sampled_training_loss": loss_sum / processed,
                "elapsed_seconds": time.perf_counter() - epoch_start,
                "selected_as_best": improved,
                **epoch_metrics,
            }
        )
        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2, allow_nan=False)
        )
    save_curves(history, output_dir, best_epoch)
    model.load_state_dict(best_state)
    return model, best_epoch, best_value


def save_curves(history, output_dir, best_epoch):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [
        {"epoch": epoch["epoch"], "split": split,
            "dataset": dataset, **metrics}
        for epoch in history
        for split in ("train", "val") if split in epoch
        for dataset, metrics in [
            ("pooled", epoch[split]["pooled"]),
            *epoch[split]["datasets"].items(),
        ]
    ]
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "history.csv", index=False)
    datasets = table["dataset"].unique()
    fig, axes = plt.subplots(
        len(datasets), 3, figsize=(12, 3 * len(datasets)), squeeze=False
    )
    for row, dataset in enumerate(datasets):
        for col, metric in enumerate(("loss", "pr_auc", "roc_auc")):
            ax = axes[row, col]
            for split in ("train", "val"):
                values = table[
                    (table["dataset"] == dataset) & (table["split"] == split)
                ]
                ax.plot(values["epoch"], values[metric],
                        marker="o", label=split)
            ax.axvline(best_epoch, color="gray",
                       linestyle=":", label="selected epoch")
            ax.set(
                title=f"{dataset if dataset == 'pooled' else 'data' + str(dataset)}: {metric}",
                xlabel="Epoch",
            )
            ax.xaxis.get_major_locator().set_params(integer=True)
            if metric != "loss":
                ax.set_ylim(0, 1.02)
            ax.grid(alpha=0.2)
            ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=160)
    fig.savefig(output_dir / "training_curves.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Train MIL or classical site classifiers with shared dataset splits."
    )
    parser.add_argument("--model", choices=("mil", "extra_trees", "random_forest",
                        "gradient_boosting", "logistic_regression", "rbf_svm"), default="mil")
    parser.add_argument("--model-params", type=Path,
                        help="Classical estimator parameters: JSON object or list of candidate objects")
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="Parallel jobs for forests and SVM calibration")
    parser.add_argument("--calibration-folds", type=int, default=3)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/baseline")
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--split-ratios", type=float, nargs=3, default=[0.7, 0.15, 0.15]
    )
    parser.add_argument("--train-datasets", type=int,
                        nargs="+", default=[0, 1])
    parser.add_argument("--val-datasets", type=int, nargs="*", default=[0, 1])
    parser.add_argument("--test-datasets", type=int,
                        nargs="*", default=[0, 1, 2])
    parser.add_argument("--dataset-splits", type=Path,
                        help='JSON mapping dataset numbers to [train, val, test] ratios')
    parser.add_argument("--split-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-reads-per-site", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--eval-iterations", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--log-every-batches", type=int, default=100)
    parser.add_argument(
        "--selection-metric", choices=("loss", "pr_auc", "roc_auc"), default="pr_auc"
    )
    parser.add_argument("--no-balanced-sampler", "--no-class-weight",
                        dest="no_balanced_sampler", action="store_true")
    parser.add_argument("--device", choices=("auto",
                        "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.n_jobs == 0:
        parser.error("--n-jobs cannot be zero")
    if args.calibration_folds < 2:
        parser.error("--calibration-folds must be at least two")
    if args.model == "mil" and args.model_params:
        parser.error("--model-params is for classical models")
    for name in (
        "epochs",
        "eval_iterations",
        "n_reads_per_site",
        "batch_size",
        "log_every_batches",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if 2 in args.train_datasets or 2 in args.val_datasets:
        parser.error("data2 is reserved exclusively for testing")
    selected = {"train": args.train_datasets,
                "val": args.val_datasets, "test": args.test_datasets}
    numbers = sorted(set(sum(selected.values(), [])))
    overrides = json.loads(args.dataset_splits.read_text()
                           ) if args.dataset_splits else {}
    if not isinstance(overrides, dict):
        parser.error(
            "Dataset splits must be a JSON object mapping dataset numbers to ratios")
    ratios = {n: overrides.get(
        str(n), [0, 0, 1] if n == 2 else args.split_ratios) for n in numbers}
    if 2 in ratios and ratios[2] != [0, 0, 1]:
        parser.error("data2 must use split ratios [0, 0, 1]")
    for n in numbers:
        try:
            dataset_files(n)
            validate_ratios(ratios[n])
        except ValueError as error:
            parser.error(str(error))
    run_dir = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True)
    if args.split_only:
        tables = {n: pd.read_csv(
            args.data_dir / dataset_files(n)[1], dtype={"transcript_id": str}) for n in numbers}
    else:
        prepared = prepare_datasets(
            args.data_dir, args.cache_dir or args.output_dir / "cache", dataset_numbers=numbers)
        tables = {n: pd.read_csv(prepared[n][0], dtype={
                                 "transcript_id": str}) for n in numbers}
    for number in numbers:
        tables[number] = split_sites(normalize_labels(
            tables[number], number), seed=args.seed + number, ratios=ratios[number])
    summary = {}
    for number, sites in tables.items():
        sites["dataset"] = number
        sites["row_id"] = np.arange(len(sites))
        sites.to_csv(run_dir / f"dataset{number}_splits.csv", index=False)
        summary[str(number)] = (
            sites.groupby("split")["label"].agg(
                ["count", "mean"]).to_dict("index")
        )
        print(
            f"data{number} split summary: {summary[str(number)]}", flush=True)
    run_info = {
        "model": args.model,
        "seed": args.seed,
        "split_ratios": ratios,
        "datasets": selected,
        "split_summary": summary,
        "trained_on_datasets": args.train_datasets,
        "data2_used_for_selection": False,
        "prior_correction": False,
        "eval_iterations": args.eval_iterations,
        "selection_metric": args.selection_metric if args.val_datasets else None,
    }
    (run_dir / "run_info.json").write_text(json.dumps(run_info, indent=2))
    if args.split_only:
        print("Saved split assignments to:", run_dir.resolve())
        return
    all_sites = pd.concat(tables.values(), ignore_index=True)
    partitions = {
        name: all_sites[(all_sites["split"] == name) & all_sites["dataset"].isin(
            selected[name])].reset_index(drop=True)
        for name in SPLITS
    }
    if partitions["train"].empty:
        parser.error("The selected training partition is empty")
    if args.val_datasets and partitions["val"].empty:
        parser.error(
            "Validation is empty; use --val-datasets with no values for final-epoch training")
    if not partitions["val"].empty and args.selection_metric != "loss" and partitions["val"]["label"].nunique() < 2:
        parser.error(
            "Validation has one binary class; choose --selection-metric loss")
    read_maps = open_read_maps(prepared, pd.concat(
        tables.values(), ignore_index=True))
    if args.model != "mil":
        from classical_models import train_classical
        train_classical(args, prepared, tables, partitions,
                        read_maps, run_dir, run_info)
        print("Saved outputs to:", run_dir.resolve())
        return
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    mean, std = feature_statistics(partitions["train"], read_maps)
    params = {
        "n_reads_per_site": args.n_reads_per_site,
        "embed_dim": 2,
        "hidden_dim": 150,
        "dropout": 0.1,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "balanced": not args.no_balanced_sampler,
    }
    model, best_epoch, best_value = train_baseline_model(
        partitions,
        read_maps,
        mean,
        std,
        params,
        device,
        args.seed,
        run_dir,
        args.eval_iterations,
        args.selection_metric,
        args.log_every_batches,
    )
    final_metrics, metric_rows = {}, []
    for name in SPLITS:
        if partitions[name].empty:
            continue
        scores, metrics = evaluate_partition(
            model,
            partitions[name],
            read_maps,
            mean,
            std,
            params,
            device,
            args.eval_iterations,
            args.seed + SPLITS.index(name),
        )
        final_metrics[name] = metrics
        print_metrics(f"Selected epoch {best_epoch} {name}", metrics)
        export_predictions(run_dir, name, partitions[name], scores)
        metric_rows.extend(metric_records(name, metrics, "noisy_or"))
    pd.DataFrame(metric_rows).to_csv(run_dir / "metrics.csv", index=False)
    (run_dir / "metrics.json").write_text(
        json.dumps(final_metrics, indent=2, allow_nan=False)
    )
    run_info.update(
        {
            "checkpoint_selection": "validation" if not partitions["val"].empty else "final_epoch",
            "best_epoch": best_epoch,
            "best_validation_value": best_value,
            "baseline_parameters": params,
        }
    )
    (run_dir / "run_info.json").write_text(json.dumps(run_info, indent=2))
    print("Saved outputs to:", run_dir.resolve())


if __name__ == "__main__":
    main()
