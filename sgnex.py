"""Download pinned SG-NEx samples and run resumable, sample-level inference."""
import argparse
import csv
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def verify_single_part(path, etag):
    if "-" in etag:
        return False
    value = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    if value.hexdigest() != etag:
        raise ValueError(f"Source checksum mismatch: {path}")
    return True


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def select_samples(manifest, cell_lines=None, samples=None):
    rows = manifest["samples"]
    if manifest["version"] != 1:
        raise ValueError("Unsupported sample manifest")
    ids = [row["sample_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate samples in manifest")
    for row in rows:
        if not re.fullmatch(r"SGNex_[A-Za-z0-9]+_directRNA_replicate\d+_run\d+", row["sample_id"]):
            raise ValueError("Invalid sample identifier")
        expected = "https://sg-nex-data.s3.amazonaws.com/data/processed_data/m6Anet/" + \
            row["sample_id"] + "/data.json"
        if row["url"] != expected or row["bytes"] <= 0:
            raise ValueError("Invalid sample source")
        if not re.fullmatch(r"[a-f0-9]{32}(?:-\d+)?", row["etag"]):
            raise ValueError("Invalid source ETag")
    if cell_lines and set(cell_lines) - {row["cell_line"] for row in rows}:
        raise ValueError(
            "Unknown cell line; use --list to see available names")
    if samples and set(samples) - set(ids):
        raise ValueError("Unknown sample; use --list to see available names")
    return [row for row in rows if (not cell_lines or row["cell_line"] in cell_lines)
            and (not samples or row["sample_id"] in samples)]


def download(row, data_dir):
    folder = data_dir / row["sample_id"]
    folder.mkdir(parents=True, exist_ok=True)
    target, receipt = folder / "data.json", folder / "download.json"
    if target.exists():
        if not receipt.exists():
            if target.stat().st_size != row["bytes"] or not verify_single_part(target, row["etag"]):
                raise ValueError(
                    f"Cannot verify existing {target} without a download receipt. Use another --data-dir or move this file before downloading.")
            write_json(receipt, {"source": row, "sha256": digest(target)})
        saved = json.loads(receipt.read_text())
        if saved["source"] != row or target.stat().st_size != row["bytes"] or saved["sha256"] != digest(target):
            raise ValueError(
                f"Download verification failed for {target}; use a clean data directory")
        print(f"Verified cached download: {row['sample_id']}", flush=True)
        return target, saved["sha256"]
    partial = folder / "data.json.part"
    source_record = folder / "partial_source.json"
    if partial.exists() and (not source_record.exists() or json.loads(source_record.read_text()) != row):
        raise ValueError(f"Partial download has a different source: {partial}")
    write_json(source_record, row)
    downloaded = partial.stat().st_size if partial.exists() else 0
    if downloaded > row["bytes"]:
        raise ValueError(
            f"Partial download is larger than its source: {partial}")
    required = row["bytes"] - downloaded + 1024 ** 3
    if shutil.disk_usage(folder).free < required:
        raise ValueError(
            "Insufficient disk space for the next download plus 1 GiB reserve")
    if downloaded < row["bytes"]:
        print(
            f"Downloading {row['sample_id']} ({row['bytes']:,} bytes)", flush=True)
        subprocess.run(["curl", "--fail", "--location", "--retry", "5", "--connect-timeout", "30",
                        "--continue-at", "-", "--header", f'If-Match: "{row["etag"]}"',
                        "--output", str(partial), row["url"]], check=True)
    if partial.stat().st_size != row["bytes"]:
        raise ValueError(
            f"Incomplete download: {partial}; repeat the command to resume")
    verify_single_part(partial, row["etag"])
    checksum = digest(partial)
    write_json(receipt, {"source": row, "sha256": checksum})
    partial.replace(target)
    return target, checksum


def completed(folder, contract):
    marker = folder / "complete.json"
    if not marker.exists():
        if folder.exists():
            raise ValueError(
                f"Incomplete output folder: {folder}; choose a new --output-dir")
        return False
    info = json.loads(marker.read_text())
    if info["contract"] != contract:
        raise ValueError(
            f"Model, input, environment or settings changed for {folder.name}; choose a new --output-dir")
    for name in ("predictions.csv", "predictions.csv.json"):
        if not (folder / name).is_file() or digest(folder / name) != info["sha256"][name]:
            raise ValueError(
                f"Completed predictions failed integrity checks: {folder / name}")
    return True


def run_sample(row, args, contract):
    folder = args.output_dir / row["sample_id"]
    contract = {**contract, "sample": row}
    if completed(folder, contract):
        print(f"Skipping verified predictions: {row['sample_id']}", flush=True)
        return
    source, checksum = download(row, args.data_dir)
    temporary = Path(tempfile.mkdtemp(prefix=".sgnex-", dir=args.output_dir))
    try:
        output = temporary / "predictions.csv"
        subprocess.run([sys.executable, str(ROOT / "predict.py"), "--model-dir", str(args.model_dir),
                        "--input", str(source), "--output", str(
                            output), "--dataset-name", row["sample_id"],
                        "--no-labels", "--n-jobs", str(
                            args.n_jobs), "--threads", str(args.threads),
                        "--batch-size", str(args.batch_size), "--seed", str(args.seed),
                        "--eval-iterations", str(args.eval_iterations)], check=True)
        write_json(temporary / "complete.json", {
            "contract": contract, "input_sha256": checksum,
            "sha256": {name: digest(temporary / name) for name in ("predictions.csv", "predictions.csv.json")},
        })
        temporary.rename(folder)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def summary(output_dir, rows):
    records = []
    for row in rows:
        folder = output_dir / row["sample_id"]
        if not (folder / "complete.json").exists():
            continue
        info = json.loads((folder / "predictions.csv.json").read_text())
        records.append({key: row[key] for key in ("sample_id", "cell_line", "replicate", "run")} |
                       {"sites": info["sites"], "model": info["model"], "model_sha256": info["model_sha256"],
                        "predictions": row["sample_id"] + "/predictions.csv"})
    temporary = output_dir / "samples.csv.tmp"
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id", "cell_line", "replicate", "run", "sites",
                                                    "model", "model_sha256", "predictions"])
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(output_dir / "samples.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "configs/sgnex_samples.json")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--cell-lines", nargs="+")
    selection.add_argument("--samples", nargs="+")
    parser.add_argument("--list", action="store_true",
                        help="List selection and download size without downloading")
    parser.add_argument("--data-dir", type=Path, default=Path("data/sgnex"))
    parser.add_argument("--model-dir", type=Path,
                        default=Path("models/extra_trees_data01"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-iterations", type=int, default=5)
    args = parser.parse_args()
    if args.n_jobs == 0 or min(args.threads, args.batch_size, args.eval_iterations) < 1 or args.seed < 0:
        parser.error(
            "Jobs must be nonzero; threads, batch size and iterations positive; seed nonnegative")
    manifest = json.loads(args.manifest.read_text())
    rows = select_samples(manifest, args.cell_lines, args.samples)
    for row in rows:
        print(f"{row['sample_id']} | {row['bytes'] / 1e9:.2f} GB", flush=True)
    print(
        f"Selected {len(rows)} runs; {sum(row['bytes'] for row in rows) / 1e9:.2f} GB of feature JSON", flush=True)
    if args.list:
        return
    if not shutil.which("curl"):
        parser.error("curl is required; follow the README setup")
    from model_bundle import load_bundle
    import numpy
    import sklearn
    import torch
    model, payload, fitted = load_bundle(args.model_dir)
    del payload, fitted
    args.model_dir = args.model_dir.resolve()
    args.data_dir = args.data_dir.resolve()
    args.output_dir = (args.output_dir or Path(
        "predictions/sgnex") / args.model_dir.name).resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    contract = {
        "manifest_sha256": digest(args.manifest),
        "model_manifest": model, "n_jobs": args.n_jobs, "threads": args.threads,
        "batch_size": args.batch_size, "seed": args.seed, "eval_iterations": args.eval_iterations,
        "code": {name: digest(ROOT / name) for name in
                 ("sgnex.py", "predict.py", "model_bundle.py", "site_features.py", "data_prep.py", "mil_model.py")},
        "versions": {"numpy": numpy.__version__, "sklearn": sklearn.__version__, "torch": str(torch.__version__)},
    }
    with (args.data_dir / ".sgnex.lock").open("a") as data_lock, (args.output_dir / ".sgnex.lock").open("a") as output_lock:
        try:
            fcntl.flock(data_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(output_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(
                "Another SG-NEx workflow is using these data or output folders")
        contract_path = args.output_dir / "run_contract.json"
        if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
            raise ValueError(
                "Output directory belongs to different settings, model or manifest; choose a new --output-dir")
        write_json(contract_path, contract)
        write_json(args.output_dir / "sample_manifest.json", manifest)
        for row in rows:
            try:
                run_sample(row, args, contract)
            finally:
                summary(args.output_dir, manifest["samples"])
    print(
        f"Finished. Sample index: {args.output_dir / 'samples.csv'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"SG-NEx workflow stopped: {exc}")
