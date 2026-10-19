"""Deterministic, label-independent site features shared by classical models."""
import hashlib
import json

import numpy as np

STATISTICS = ("mean", "std", "q25", "q75")
FEATURE_NAMES = (
    [f"channel_{channel}_{stat}" for stat in STATISTICS for channel in range(
        9)]
    + [f"base_{position}_{base}" for position in range(7) for base in "ACGT"]
    + ["log1p_n_reads"]
)
NUMERIC_COLUMNS = list(range(36)) + [64]


def summarize_site(reads, sequence):
    reads = np.asarray(reads, dtype=np.float64)
    sequence = str(sequence).upper()
    if reads.ndim != 2 or reads.shape[1] != 9 or not len(reads) or not np.isfinite(reads).all():
        raise ValueError(
            "Site features require a nonempty finite matrix with nine channels")
    if len(sequence) != 7 or any(base not in "ACGT" for base in sequence):
        raise ValueError(
            f"Classical features require a seven-base A/C/G/T sequence: {sequence}")
    quartiles = np.quantile(reads, [0.25, 0.75], axis=0, method="linear")
    one_hot = np.asarray(
        [[base == nucleotide for nucleotide in "ACGT"] for base in sequence])
    return np.concatenate([
        reads.mean(axis=0), reads.std(axis=0, ddof=0),
        quartiles.ravel(), one_hot.ravel(), [np.log1p(len(reads))],
    ])


def prepare_site_features(prepared, tables, read_maps):
    """Cache all-read summaries in source metadata row order, independent of splits."""
    features = {}
    for number, (sites_path, reads_path) in prepared.items():
        folder = sites_path.parent
        feature_path = folder / "site_features.npy"
        manifest_path = folder / "site_features_manifest.json"
        signature = {
            "version": 1,
            "source": json.loads((folder / "manifest.json").read_text()),
            "sites_sha256": hashlib.sha256(sites_path.read_bytes()).hexdigest(),
            "reads_size": reads_path.stat().st_size,
            "reads_mtime_ns": reads_path.stat().st_mtime_ns,
            "feature_names": FEATURE_NAMES,
        }
        sites = tables[number]
        if feature_path.exists() and manifest_path.exists():
            if json.loads(manifest_path.read_text()) == signature:
                cached = np.load(feature_path, mmap_mode="r",
                                 allow_pickle=False)
                if cached.shape == (len(sites), len(FEATURE_NAMES)):
                    features[number] = cached
                    print(
                        f"Using cached data{number} site features", flush=True)
                    continue
        # Remove the old signature first so interrupted rebuilds cannot appear valid.
        manifest_path.unlink(missing_ok=True)
        values = np.empty((len(sites), len(FEATURE_NAMES)), dtype=np.float64)
        for index, row in enumerate(sites.itertuples(index=False)):
            reads = read_maps[number][row.read_start:row.read_start + row.n_reads]
            values[index] = summarize_site(reads, row.sequence)
            if (index + 1) % 20000 == 0:
                print(
                    f"data{number}: summarized {index + 1:,} sites", flush=True)
        np.save(feature_path, values, allow_pickle=False)
        manifest_path.write_text(json.dumps(signature, indent=2))
        features[number] = np.load(
            feature_path, mmap_mode="r", allow_pickle=False)
    return features


def partition_features(sites, features):
    values = np.empty((len(sites), len(FEATURE_NAMES)), dtype=np.float64)
    for number in sites["dataset"].unique():
        mask = sites["dataset"].to_numpy() == number
        values[mask] = features[number][sites.loc[mask,
                                                  "row_id"].to_numpy(dtype=int)]
    return values
