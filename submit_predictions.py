"""Generate the eight course leaderboard CSVs using the full Extra Trees model."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import tempfile

import torch
from threadpoolctl import threadpool_limits

from predict import predict

COLUMNS = ["transcript_id", "transcript_position", "score"]


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--team-name", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("predictions/leaderboard"))
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if not args.team_name.strip() or any(c in args.team_name for c in '/\\\n\r'):
        parser.error("Team name must be nonempty and cannot contain path separators or newlines")
    if args.n_jobs == 0 or args.threads < 1:
        parser.error("n-jobs must be nonzero and threads must be positive")
    if args.output_dir.exists():
        parser.error("Output directory already exists; choose a new directory to protect official files")
    with Path("reference/sample_submission.csv").open(newline="") as stream:
        if next(csv.reader(stream)) != COLUMNS:
            raise ValueError("Submission sample columns changed")
    inputs = [Path(f"data/data{i}/dataset{i}.json") for i in range(8)]
    for source in inputs:
        if not source.is_file():
            raise FileNotFoundError(source)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    manifest = {"team_name": args.team_name, "model_dir": "models/full/extra_trees",
                "columns": COLUMNS, "score": "modification probability; no data2 inversion",
                "datasets": []}
    # Publish the directory only when every dataset has completed successfully.
    with tempfile.TemporaryDirectory(prefix="leaderboard-", dir=args.output_dir.parent) as temporary:
        staging = Path(temporary)
        final = staging / "submission"
        final.mkdir()
        with threadpool_limits(limits=args.threads):
            for i, source in enumerate(inputs):
                raw = staging / f"dataset{i}.csv"
                count = predict(Path("models/full/extra_trees"), source, raw,
                                no_labels=True, n_jobs=args.n_jobs)
                target = final / f"{args.team_name}_dataset{i}.csv"
                seen = set()
                with raw.open(newline="") as src, target.open("w", newline="") as dst:
                    writer = csv.DictWriter(dst, fieldnames=COLUMNS, lineterminator="\n")
                    writer.writeheader()
                    for row in csv.DictReader(src):
                        key = (row["transcript_id"], int(row["transcript_position"]))
                        if not key[0] or key in seen or not 0 <= float(row["score"]) <= 1:
                            raise ValueError(f"Invalid submission row: {row}")
                        seen.add(key)
                        writer.writerow({column: row[column] for column in COLUMNS})
                if len(seen) != count:
                    raise ValueError("Submission row count does not match inference")
                metadata = json.loads(raw.with_suffix(".csv.json").read_text())
                manifest["datasets"].append({"dataset": i, "file": target.name,
                    "sites": count, "sha256": sha256(target), "input": str(source),
                    "input_sha256": sha256(source), "inference": metadata})
                print(f"Validated {target.name}: {count:,} sites", flush=True)
        (final / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        final.rename(args.output_dir)
    print(f"Saved all eight submission CSVs to {args.output_dir}")


if __name__ == "__main__":
    main()
