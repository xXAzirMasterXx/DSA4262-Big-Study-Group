"""Predict site modification probabilities from a feature JSON, with optional label-based evaluation."""
import argparse
import csv
import json
from pathlib import Path
import tempfile

import pandas as pd
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from data_prep import iter_feature_sites
from evaluation import compute_metrics, data2_metrics
from mil_model import encode_5mer
from model_bundle import load_bundle
from site_features import summarize_site


def input_sites(path):
    seen = set()
    for row_id, (transcript, position, contexts) in enumerate(iter_feature_sites(path)):
        key = (transcript, position)
        if key in seen:
            raise ValueError(f"Duplicate site key in input: {key}")
        seen.add(key)
        if len(contexts) != 1:
            raise ValueError(f"Expected one sequence context for {key}")
        sequence, vectors = next(iter(contexts.items()))
        sequence = sequence.upper()
        values = np.asarray(vectors, dtype=np.float32)
        if len(sequence) != 7 or any(base not in "ACGT" for base in sequence):
            raise ValueError(
                f"Expected a seven-base A/C/G/T sequence at {key}")
        if values.ndim != 2 or values.shape[1] != 9 or not len(values) or not np.isfinite(values).all():
            raise ValueError(
                f"Expected nonempty, finite nine-channel reads at {key}")
        yield row_id, transcript, position, sequence, values


def sample_bags(reads, row_id, seed, iterations, params, mean, std):
    size = params["n_reads_per_site"]
    # Independent per-site seeds make sampled bags invariant to inference batch size.
    bags = []
    for iteration in range(iterations):
        rng = np.random.default_rng(
            np.random.SeedSequence([seed, row_id, iteration]))
        indices = rng.choice(len(reads), size=size, replace=len(reads) < size)
        bags.append((reads[indices] - mean) / np.maximum(std, 1e-6))
    return np.stack(bags).astype(np.float32)


def load_labels(input_path, labels_path=None, no_labels=False):
    """Discover course metadata or read an explicit CSV; align by site key."""
    if no_labels:
        if labels_path is not None:
            raise ValueError("Cannot combine labels_path with no_labels")
        return None, None
    if labels_path is None:
        candidates = [input_path.parent /
                      name for name in ("data.info.labelled", "data.info")]
        labels_path = next(
            (path for path in candidates if path.is_file()), None)
        if labels_path is None:
            return None, None
        frame = pd.read_csv(labels_path, dtype={"transcript_id": str})
        if "label" not in frame.columns:
            return None, None
    else:
        labels_path = Path(labels_path)
        frame = pd.read_csv(labels_path, dtype={"transcript_id": str})
    required = {"transcript_id", "transcript_position", "label"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Label CSV must contain {sorted(required)}")
    positions = pd.to_numeric(frame["transcript_position"], errors="raise")
    targets = pd.to_numeric(frame["label"], errors="raise")
    if (frame["transcript_id"].isna().any() or not np.isfinite(positions).all()
            or (positions % 1 != 0).any()):
        raise ValueError("Label CSV contains invalid transcript/site keys")
    if not np.isfinite(targets).all() or (targets < 0).any():
        raise ValueError("Labels must be finite and nonnegative")
    keys = list(zip(frame["transcript_id"], positions.astype(int)))
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate site keys in label CSV")
    return dict(zip(keys, (targets > 0).astype(int))), labels_path


def predict(model_dir, input_path, output_path, batch_size=256, device="cpu",
            seed=None, iterations=None, dataset_name=None, labels_path=None, no_labels=False, n_jobs=None):
    input_path, output_path = Path(input_path), Path(output_path)
    sidecar = output_path.with_suffix(output_path.suffix + ".json")
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    if output_path.exists() or sidecar.exists():
        raise ValueError(
            "Prediction output already exists; choose a new output filename")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; use --device cpu")
    is_data2 = dataset_name in (
        "2", "data2", "dataset2") or input_path.name == "dataset2.json" or input_path.parent.name == "data2"
    label_rule = "label > 0"
    labels, label_source = load_labels(input_path, labels_path, no_labels)
    evaluated_labels, evaluated_scores = [], []
    manifest, payload, fitted = load_bundle(model_dir, device)
    if n_jobs == 0:
        raise ValueError("n_jobs must be nonzero")
    if n_jobs is not None and manifest["kind"] == "classical":
        updates = {key: n_jobs for key in fitted.get_params(deep=True)
                   if key == "n_jobs" or key.endswith("__n_jobs")}
        fitted.set_params(**updates)
    seed = manifest["inference_defaults"]["seed"] if seed is None else seed
    iterations = manifest["inference_defaults"]["eval_iterations"] if iterations is None else iterations
    if seed < 0 or iterations < 1:
        raise ValueError(
            "Seed must be nonnegative and evaluation iterations positive")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    name = dataset_name or input_path.stem
    temporary = None
    count = 0
    try:
        with tempfile.NamedTemporaryFile(mode="w", newline="", encoding="utf-8", delete=False,
                                         dir=output_path.parent, prefix=".predictions-", suffix=".csv") as stream:
            temporary = Path(stream.name)
            writer = csv.writer(stream)
            writer.writerow(["dataset", "row_id", "transcript_id",
                            "transcript_position", "n_reads", "score"] + (["label"] if labels is not None else []))
            batch, metadata, kmers = [], [], []
            if manifest["kind"] == "mil":
                mean = payload["feature_mean"].numpy()
                std = payload["feature_std"].numpy()

            def flush():
                if not batch:
                    return
                values = np.stack(batch)
                if manifest["kind"] == "classical":
                    scores = fitted.predict_proba(values)[:, 1]
                else:
                    kmer_tensor = torch.tensor(
                        kmers, dtype=torch.long, device=device)
                    scores = np.zeros(len(batch), dtype=np.float64)
                    with torch.inference_mode():
                        for iteration in range(iterations):
                            bags = torch.from_numpy(np.ascontiguousarray(
                                values[:, iteration])).to(device)
                            scores += fitted(bags, kmer_tensor).cpu().numpy()
                    scores /= iterations
                if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
                    raise ValueError("Model produced invalid probabilities")
                for meta, score in zip(metadata, scores):
                    row = [*meta, float(score)]
                    if labels is not None:
                        label = labels.get((meta[2], meta[3]))
                        row.append(label)
                        if label is not None:
                            evaluated_labels.append(label)
                            evaluated_scores.append(float(score))
                    writer.writerow(row)
                batch.clear()
                metadata.clear()
                kmers.clear()

            for row_id, transcript, position, sequence, reads in input_sites(input_path):
                metadata.append(
                    [name, row_id, transcript, position, len(reads)])
                if manifest["kind"] == "classical":
                    batch.append(summarize_site(reads, sequence))
                else:
                    batch.append(sample_bags(reads, row_id, seed, iterations,
                                             payload["baseline_parameters"], mean, std))
                    kmers.append([encode_5mer(sequence[i:i + 5])
                                 for i in range(3)])
                count += 1
                if len(batch) >= batch_size:
                    flush()
            flush()
            if count == 0:
                raise ValueError("Input contains no transcript sites")
        stat = input_path.stat()
        metadata = {
            "model": manifest["model"], "model_sha256": manifest["sha256"],
            "input": str(input_path.resolve()), "input_size": stat.st_size,
            "input_mtime_ns": stat.st_mtime_ns, "sites": count, "dataset": name,
            "device": device, "batch_size": batch_size, "n_jobs": n_jobs,
            "torch_threads": torch.get_num_threads(),
            "seed": seed if manifest["kind"] == "mil" else None,
            "eval_iterations": iterations if manifest["kind"] == "mil" else None,
            "sampling": "per_site_iteration_seed_v1" if manifest["kind"] == "mil" else "all_reads",
        }
        if labels is not None:
            if not evaluated_labels:
                raise ValueError(
                    "No label CSV sites matched the input feature sites")
            evaluator = data2_metrics if is_data2 else compute_metrics
            metrics = evaluator(evaluated_labels, evaluated_scores)
            metrics.pop("loss")
            metadata["metrics"] = metrics
            metadata["labels"] = {
                "path": str(label_source.resolve()), "rule": label_rule,
                "metadata_sites": len(labels), "matched_sites": len(evaluated_labels),
                "unlabeled_prediction_sites": count - len(evaluated_labels),
                "metadata_sites_without_predictions": len(labels) - len(evaluated_labels),
            }
        temporary.replace(output_path)
        sidecar.write_text(json.dumps(metadata, indent=2, allow_nan=False))
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True,
                        help="Feature JSON; no labels or metadata CSV required")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dataset-name", help="Dataset identifier written to the output")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--eval-iterations", type=int)
    label_options = parser.add_mutually_exclusive_group()
    label_options.add_argument("--labels", type=Path,
                               help="Label CSV; defaults to adjacent data.info.labelled or data.info")
    label_options.add_argument(
        "--no-labels", action="store_true", help="Disable label discovery/evaluation")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Parallel estimator prediction jobs")
    parser.add_argument("--threads", type=int, default=1,
                        help="BLAS/OpenMP/PyTorch CPU threads")
    args = parser.parse_args()
    if args.n_jobs == 0 or args.threads < 1:
        parser.error("--n-jobs must be nonzero and --threads must be positive")
    torch.set_num_threads(args.threads)
    with threadpool_limits(limits=args.threads):
        count = predict(args.model_dir, args.input, args.output, args.batch_size, args.device,
                        args.seed, args.eval_iterations, args.dataset_name, args.labels, args.no_labels, args.n_jobs)
    print(f"Saved {count:,} predictions to {args.output}")
    summary = json.loads(args.output.with_suffix(
        args.output.suffix + ".json").read_text())
    if "metrics" in summary:
        metrics = summary["metrics"]
        print(
            f"Evaluated {metrics['sites']:,} labeled sites out of {count:,} predictions")
        for name in ("roc_auc", "pr_auc"):
            value = metrics[name]
            print(f"{name.upper()}: {value:.6f}" if value is not None else
                  f"{name.upper()}: N/A ({metrics['auc_unavailable_reason']})")


if __name__ == "__main__":
    main()
