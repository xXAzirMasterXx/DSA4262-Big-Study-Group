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

from data_prep import prepare_datasets
from mil_model import M6ANetInspiredMIL, SiteBagDataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def open_read_maps(prepared, sites):
    read_maps = {}
    for dataset_number, (_, reads_path) in prepared.items():
        n_reads = int(
            sites.loc[sites["dataset"] == dataset_number, "n_reads"].sum()
        )
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
            int(row.read_start):int(row.read_start) + int(row.n_reads)
        ]
        values = np.asarray(reads, dtype=np.float64)
        sums += values.sum(axis=0)
        sums_sq += np.square(values).sum(axis=0)
        total += len(values)
    mean = sums / max(total, 1)
    variance = np.maximum(sums_sq / max(total, 1) - np.square(mean), 1e-12)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def make_loader(sites, read_maps, n_reads, mean, std, batch_size, shuffle, balanced=False):
    dataset = SiteBagDataset(sites, read_maps, n_reads, mean, std)
    if balanced:
        labels = sites["label"].to_numpy(dtype=int)
        counts = np.bincount(labels, minlength=2)
        if np.any(counts == 0):
            raise ValueError("Both label classes must be present in the training split")
        sample_weights = np.where(labels == 1, 1.0 / counts[1], 1.0 / counts[0])
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
                probabilities = model(
                    reads.to(device),
                    kmers.to(device),
                ).cpu().numpy()
                scores[indices.numpy()] += probabilities
    return scores / iterations


def correct_balanced_prior(scores, natural_prevalence):
    """Undo the 50/50 sampler prior using the labeled training prevalence."""
    clipped = np.clip(np.asarray(scores, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    balanced_odds = clipped / (1.0 - clipped)
    natural_odds = balanced_odds * natural_prevalence / (1.0 - natural_prevalence)
    return natural_odds / (1.0 + natural_odds)


def train_baseline_model(
    sites,
    read_maps,
    mean,
    std,
    params,
    device,
    seed,
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
    dataset, loader = make_loader(
        sites,
        read_maps,
        params["n_reads_per_site"],
        mean,
        std,
        params["batch_size"],
        shuffle=False,
        balanced=True,
    )
    criterion = nn.BCELoss()
    batches_per_epoch = len(loader)

    for epoch in range(params["epochs"]):
        model.train()
        epoch_start = time.perf_counter()
        running_loss = 0.0
        processed_sites = 0
        for batch_number, (_indices, reads, kmers, labels) in enumerate(loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            predictions = model(reads.to(device), kmers.to(device))
            loss = criterion(predictions, labels.to(device))
            loss.backward()
            optimizer.step()
            batch_sites = len(labels)
            running_loss += float(loss.detach().cpu()) * batch_sites
            processed_sites += batch_sites

            if batch_number % log_every_batches == 0 or batch_number == batches_per_epoch:
                elapsed = time.perf_counter() - epoch_start
                print(
                    f"Epoch {epoch + 1}/{params['epochs']} | "
                    f"batch {batch_number}/{batches_per_epoch} | "
                    f"mean loss={running_loss / max(processed_sites, 1):.4f} | "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )

        epoch_elapsed = time.perf_counter() - epoch_start
        print(
            f"Completed epoch {epoch + 1}/{params['epochs']} | "
            f"mean balanced site BCE={running_loss / max(processed_sites, 1):.4f} | "
            f"time={epoch_elapsed:.1f}s",
            flush=True,
        )
    return model


def evaluate(model, sites, read_maps, n_reads, mean, std, batch_size, device, iterations):
    _dataset, loader = make_loader(
        sites,
        read_maps,
        n_reads,
        mean,
        std,
        batch_size,
        shuffle=False,
    )
    return score_dataset(model, loader, len(sites), device, iterations)


def main():
    parser = argparse.ArgumentParser(
        description="Train an m6Anet-inspired multiple-instance m6A site classifier."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/m6anet_baseline"))
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-reads-per-site", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--eval-iterations", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--log-every-batches", type=int, default=100)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if args.log_every_batches < 1:
        parser.error("--log-every-batches must be at least 1")

    if args.cache_dir is None:
        args.cache_dir = args.output_dir / "cache"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    prepared = prepare_datasets(
        args.data_dir,
        args.cache_dir,
        dataset_numbers=(0, 1, 2),
    )
    site_tables = {}
    for dataset_number in (0, 1, 2):
        sites_path, _reads_path = prepared[dataset_number]
        sites = pd.read_csv(sites_path, dtype={"transcript_id": str})
        site_tables[dataset_number] = sites
        print(f"Dataset {dataset_number}: {len(sites):,} sites")

    # Fit on all labeled sites in datasets 0 and 1. Dataset 2 labels are not
    # used; it is included only in the final prediction CSV.
    train_sites = pd.concat(
        [site_tables[0], site_tables[1]],
        ignore_index=True,
    )
    train_sites["label"] = train_sites["label"].astype(int)
    if train_sites["label"].nunique() != 2:
        raise ValueError("Training data from datasets 0 and 1 must contain both classes")
    training_prevalence = float(train_sites["label"].mean())
    print(
        f"Training sites: {len(train_sites):,}; "
        f"positive fraction={train_sites['label'].mean():.4f}"
    )
    sites_for_maps = pd.concat(
        [site_tables[number] for number in (0, 1, 2)],
        ignore_index=True,
    )
    read_maps = open_read_maps(prepared, sites_for_maps)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this PyTorch install")
    print("Training device:", device)

    train_mean, train_std = feature_statistics(train_sites, read_maps)
    # One fixed, reproducible baseline configuration; no hyperparameter search.
    baseline_parameters = {
        "n_reads_per_site": args.n_reads_per_site,
        "embed_dim": 2,
        "hidden_dim": 150,
        "dropout": 0.1,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "class_balance": "WeightedRandomSampler with replacement",
    }
    print("\nTraining one baseline model:", baseline_parameters)
    baseline_model = train_baseline_model(
        train_sites,
        read_maps,
        train_mean,
        train_std,
        baseline_parameters,
        device,
        args.seed,
        log_every_batches=args.log_every_batches,
    )

    prediction_paths = {}
    for dataset_number in (0, 1, 2):
        prediction_sites = site_tables[dataset_number].reset_index(drop=True)
        set_seed(args.seed + dataset_number + 1)
        scores = evaluate(
            baseline_model,
            prediction_sites,
            read_maps,
            args.n_reads_per_site,
            train_mean,
            train_std,
            args.batch_size,
            device,
            iterations=args.eval_iterations,
        )
        scores = correct_balanced_prior(scores, training_prevalence)
        dataset_predictions = prediction_sites[
            ["transcript_id", "transcript_position"]
        ].copy()
        dataset_predictions["score"] = scores
        prediction_path = args.output_dir / (
            f"dataset{dataset_number}_predictions_{run_id}.csv"
        )
        dataset_predictions.to_csv(
            prediction_path,
            index=False,
            columns=["transcript_id", "transcript_position", "score"],
        )
        prediction_paths[str(dataset_number)] = prediction_path.name
        print(f"Scored dataset {dataset_number}: {len(dataset_predictions):,} sites")

    run_info = {
        "trained_on_datasets": [0, 1],
        "predicted_datasets": [0, 1, 2],
        "training_sites": int(len(train_sites)),
        "training_positive_fraction": training_prevalence,
        "balanced_prior_corrected_to_training_prevalence": True,
        "prediction_sites_by_dataset": {
            str(dataset_number): int(len(site_tables[dataset_number]))
            for dataset_number in (0, 1, 2)
        },
        "baseline_parameters": baseline_parameters,
        "eval_iterations": args.eval_iterations,
        "prediction_files": prediction_paths,
        "note": "Dataset 0 and 1 scores are in-sample; dataset 2 was excluded from fitting.",
    }
    (args.output_dir / "run_info.json").write_text(
        json.dumps(run_info, indent=2),
        encoding="utf-8",
    )

    torch.save(
        {
            "model_state_dict": baseline_model.state_dict(),
            "baseline_parameters": baseline_parameters,
            "feature_mean": train_mean,
            "feature_std": train_std,
            "training_prevalence": training_prevalence,
        },
        args.output_dir / "baseline_model.pt",
    )
    print("\nRun information:")
    print(json.dumps(run_info, indent=2))
    print("Saved prediction files:", prediction_paths)
    print("Saved outputs to:", args.output_dir.resolve())


if __name__ == "__main__":
    main()
