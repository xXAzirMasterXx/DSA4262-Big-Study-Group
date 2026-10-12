import json
from pathlib import Path

import ijson
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

DATA_FILES = {
    0: ("data0/dataset0.json", "data0/data.info.labelled"),
    1: ("data1/dataset1.json", "data1/data.info"),
    2: ("data2/dataset2.json", "data2/data.info"),
}


def iter_feature_sites(feature_path: Path):
    """Stream top-level transcript objects from the feature JSON."""
    with feature_path.open("rb") as stream:
        records = ijson.kvitems(
            stream,
            "",
            multiple_values=True,
            use_float=True,
        )
        for transcript_id, positions in records:
            for position, contexts in positions.items():
                yield str(transcript_id), int(position), contexts


def prepare_one(
    data_dir: Path,
    cache_dir: Path,
    dataset_number: int,
    binary_labels: bool = True,
):
    feature_name, metadata_name = DATA_FILES[dataset_number]
    feature_path = data_dir / feature_name
    metadata_path = data_dir / metadata_name
    out_dir = cache_dir / f"dataset{dataset_number}"
    out_dir.mkdir(parents=True, exist_ok=True)
    reads_path = out_dir / "reads.f32"
    sites_path = out_dir / "sites.csv"
    manifest_path = out_dir / "manifest.json"

    stat = feature_path.stat()
    signature = {
        "feature_path": str(feature_path.resolve()),
        "feature_size": stat.st_size,
        "feature_mtime_ns": stat.st_mtime_ns,
        "metadata_path": str(metadata_path.resolve()),
        "metadata_size": metadata_path.stat().st_size,
        "metadata_mtime_ns": metadata_path.stat().st_mtime_ns,
        "n_features": 9,
        "cache_version": 4,
    }
    if manifest_path.exists() and sites_path.exists() and reads_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) == signature:
            print(f"Using cached dataset {dataset_number} features")
            return sites_path, reads_path

    sites = pd.read_csv(metadata_path, dtype={"transcript_id": str})
    required = {"transcript_id", "transcript_position", "label"}
    if not required.issubset(sites.columns):
        raise ValueError(f"{metadata_path} must contain {sorted(required)}")
    has_read_counts = "n_reads" in sites.columns
    sites["transcript_id"] = sites["transcript_id"].astype(str)
    sites["transcript_position"] = pd.to_numeric(
        sites["transcript_position"], errors="raise"
    ).astype(int)
    if has_read_counts:
        sites["n_reads"] = pd.to_numeric(sites["n_reads"], errors="raise").astype(int)
    # Validate before casting: converting a fractional target to int first
    # could silently turn an invalid binary label into zero.
    sites["label"] = pd.to_numeric(sites["label"], errors="raise")
    if binary_labels and not sites["label"].isin([0, 1]).all():
        raise ValueError(f"Dataset {dataset_number} training labels must be binary")
    if not binary_labels and (
        not np.isfinite(sites["label"]).all() or not sites["label"].between(0, 1).all()
    ):
        raise ValueError(f"Dataset {dataset_number} labels must be in [0, 1]")
    sites["label"] = sites["label"].astype(int if binary_labels else float)
    if sites.duplicated(["transcript_id", "transcript_position"]).any():
        raise ValueError(f"Duplicate site keys in {metadata_path}")
    if has_read_counts and (sites["n_reads"] <= 0).any():
        raise ValueError(f"Nonpositive n_reads values in {metadata_path}")

    row_lookup = {
        (row.transcript_id, int(row.transcript_position)): int(index)
        for index, row in sites[["transcript_id", "transcript_position"]].iterrows()
    }

    # data.info.labelled (dataset 0) contains labels but no read counts. Get
    # counts from its feature arrays in a first streaming pass.
    if not has_read_counts:
        print(f"Counting dataset {dataset_number} reads from feature JSON...")
        counts = np.zeros(len(sites), dtype=np.int64)
        for transcript_id, position, contexts in iter_feature_sites(feature_path):
            key = (transcript_id, position)
            if key not in row_lookup:
                raise ValueError(f"Feature site absent from metadata: {key}")
            row_index = row_lookup[key]
            if counts[row_index] != 0:
                raise ValueError(f"Duplicate feature site in JSON: {key}")
            if len(contexts) != 1:
                raise ValueError(f"Expected one sequence context for {key}")
            _sequence, read_vectors = next(iter(contexts.items()))
            counts[row_index] = len(read_vectors)
        if np.any(counts <= 0):
            missing = int(np.count_nonzero(counts <= 0))
            raise ValueError(
                f"Could not determine read counts for {missing} metadata sites "
                f"in {feature_name}"
            )
        sites["n_reads"] = counts

    sites["read_start"] = np.r_[0, sites["n_reads"].cumsum().to_numpy()[:-1]]
    sites["sequence"] = ""
    sites["dataset"] = dataset_number
    seen = np.zeros(len(sites), dtype=bool)
    total_reads = int(sites["n_reads"].sum())
    read_map = np.memmap(
        reads_path,
        dtype=np.float32,
        mode="w+",
        shape=(total_reads, 9),
    )

    processed = 0
    for transcript_id, position, contexts in iter_feature_sites(feature_path):
        key = (transcript_id, position)
        if key not in row_lookup:
            raise ValueError(f"Feature site absent from metadata: {key}")
        row_index = row_lookup[key]
        if seen[row_index]:
            raise ValueError(f"Duplicate feature site in JSON: {key}")
        if len(contexts) != 1:
            raise ValueError(f"Expected one sequence context for {key}")
        sequence, read_vectors = next(iter(contexts.items()))
        if len(sequence) != 7:
            raise ValueError(f"Expected a 7-mer sequence context for {key}")
        values = np.asarray(read_vectors, dtype=np.float32)
        if (
            values.ndim != 2
            or values.shape[1] != 9
            or values.shape[0] == 0
            or not np.isfinite(values).all()
        ):
            raise ValueError(f"Invalid 9-channel read matrix for {key}")
        expected_reads = int(sites.at[row_index, "n_reads"])
        if len(values) != expected_reads:
            raise ValueError(
                f"Read-count mismatch at {key}: metadata={expected_reads}, "
                f"JSON={len(values)}"
            )
        start = int(sites.at[row_index, "read_start"])
        read_map[start : start + expected_reads] = values
        sites.at[row_index, "sequence"] = sequence.upper()
        seen[row_index] = True
        processed += 1
        if processed % 20000 == 0:
            print(f"Dataset {dataset_number}: prepared {processed:,} sites")

    read_map.flush()
    del read_map
    if not seen.all():
        missing = int((~seen).sum())
        raise ValueError(f"{missing} metadata sites were missing from {feature_name}")

    sites.to_csv(sites_path, index=False)
    manifest_path.write_text(json.dumps(signature, indent=2), encoding="utf-8")
    print(
        f"Prepared dataset {dataset_number}: {len(sites):,} sites, "
        f"{total_reads:,} reads"
    )
    return sites_path, reads_path


def prepare_datasets(data_dir: Path, cache_dir: Path, dataset_numbers=(0, 1)):
    cache_dir.mkdir(parents=True, exist_ok=True)
    prepared = {}
    for dataset_number in dataset_numbers:
        prepared[dataset_number] = prepare_one(
            data_dir,
            cache_dir,
            dataset_number,
            binary_labels=dataset_number in (0, 1),
        )
    return prepared


def split_sites(sites, seed=42, ratios=(0.7, 0.15, 0.15)):
    """Stratify one dataset by label, keeping each transcript in one partition."""
    sites = sites.copy().reset_index(drop=True)
    folds = min(20, sites["transcript_id"].nunique())
    if folds < 3:
        raise ValueError("At least three transcripts are required for train/val/test")
    train_end = int(np.clip(round(folds * ratios[0]), 1, folds - 2))
    val_end = int(
        np.clip(round(folds * (ratios[0] + ratios[1])), train_end + 1, folds - 1)
    )
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (_, indices) in enumerate(
        splitter.split(sites, sites["label"], sites["transcript_id"])
    ):
        name = "train" if fold < train_end else "val" if fold < val_end else "test"
        sites.loc[indices, "split"] = name
    if sites.groupby("transcript_id")["split"].nunique().gt(1).any():
        raise AssertionError("Transcript appears in multiple partitions")
    return sites
