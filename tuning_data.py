"""Fixed data preparation and resume fingerprints for local tuning studies."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_prep import prepare_datasets, normalize_labels, split_sites
from site_features import prepare_site_features, partition_features
from train import open_read_maps, feature_statistics


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_partitions(args, numbers):
    prepared = prepare_datasets(args.data_dir, args.cache_dir, numbers)
    tables = {}
    for number in numbers:
        sites = pd.read_csv(prepared[number][0], dtype={"transcript_id": str})
        tables[number] = split_sites(
            normalize_labels(sites, number), seed=args.split_seed + number,
            ratios=(0, 0, 1) if number == 2 else args.split_ratios,
        )
        tables[number]["dataset"] = number
        tables[number]["row_id"] = np.arange(len(sites))
    sites = pd.concat(tables.values(), ignore_index=True)
    return prepared, tables, open_read_maps(prepared, sites)


def select_partition(tables, numbers, split):
    sites = pd.concat(tables.values(), ignore_index=True)
    return sites[(sites["dataset"].isin(numbers)) & (sites["split"] == split)].reset_index(drop=True)


def data_fingerprints(prepared, tables):
    return {
        str(n): {
            "source": json.loads((paths[0].parent / "manifest.json").read_text()),
            "sites_sha256": sha256(paths[0]),
            "splits_sha256": hashlib.sha256(
                tables[n][["row_id", "label", "split"]].to_csv(
                    index=False).encode()
            ).hexdigest(),
        }
        for n, paths in prepared.items()
    }


@dataclass
class TuningData:
    # Only training and validation tables are exposed to trial objectives.
    partitions: dict
    read_maps: dict
    x_train: object = None
    x_val: object = None
    mean: object = None
    std: object = None


def prepare_tuning_data(args, model, train_numbers):
    numbers = sorted(set(train_numbers + args.val_datasets))
    prepared, tables, read_maps = load_partitions(args, numbers)
    partitions = {
        "train": select_partition(tables, train_numbers, "train"),
        "val": select_partition(tables, args.val_datasets, "val"),
    }
    for split, sites in partitions.items():
        if sites.empty or sites.label.nunique() != 2:
            raise ValueError(f"Tuning requires both binary classes in {split}")
    if model == "rbf_svm" and partitions["train"].label.value_counts().min() < args.calibration_folds:
        raise ValueError(
            "Training classes must each contain at least calibration-folds rows")
    data = TuningData(partitions, read_maps)
    if model == "mil":
        data.mean, data.std = feature_statistics(
            partitions["train"], read_maps)
    else:
        features = prepare_site_features(prepared, tables, read_maps)
        data.x_train = partition_features(partitions["train"], features)
        data.x_val = partition_features(partitions["val"], features)
    return data, tables, data_fingerprints(prepared, tables)
