"""Package a selected run/checkpoint as a model folder for inference and Git LFS."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile

import joblib
import sklearn
import torch

from model_bundle import file_sha256, read_artifact
from site_features import FEATURE_NAMES


def export_model(source, output, description="", eval_iterations=None, seed=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError(
            f"Destination already exists: {output}; choose a new model folder")
    # A study folder points to its selected trial; do not depend on absolute paths saved on another machine.
    if (source / "best_trial.json").is_file():
        number = int(json.loads(
            (source / "best_trial.json").read_text())["number"])
        source = source / "trials" / f"trial_{number:05d}"
    candidates = [(source / name, kind) for name, kind in
                  (("best_model.pt", "mil"), ("best_model.joblib", "classical"))
                  if (source / name).is_file()]
    if len(candidates) != 1:
        raise ValueError(
            "Source must contain exactly one selected best_model.pt or best_model.joblib")
    artifact, kind = candidates[0]
    payload, _ = read_artifact(artifact, kind)
    info = {}
    if (source / "run_info.json").exists():
        info = json.loads((source / "run_info.json").read_text())
    elif (source.parent.parent / "study_info.json").exists():
        info = json.loads(
            (source.parent.parent / "study_info.json").read_text())
    contract = info.get("study_contract", info)
    eval_iterations = eval_iterations if eval_iterations is not None else contract.get(
        "eval_iterations", 5)
    seed = seed if seed is not None else contract.get(
        "evaluation_seed", info.get("seed", 42))
    if eval_iterations < 1 or seed < 0:
        raise ValueError(
            "Evaluation iterations must be positive and seed nonnegative")
    provenance_keys = ("trained_on_datasets", "train_datasets", "datasets", "split_ratios", "split_seed",
                       "seed", "model_seed", "sampler_seed", "best_epoch", "best_trial",
                       "best_validation_value", "parameters", "model_parameters", "balanced_class_weights")
    provenance = {key: info[key] for key in provenance_keys if key in info}
    if contract is not info:
        provenance["training_settings"] = {
            key: contract[key] for key in provenance_keys if key in contract}
    if (source / "parameters.json").exists():
        provenance["parameters"] = json.loads(
            (source / "parameters.json").read_text())
    if (source / "validation.json").exists():
        provenance["validation"] = json.loads(
            (source / "validation.json").read_text())
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=".model-export-", dir=output.parent))
    try:
        filename = "model.pt" if kind == "mil" else "model.joblib"
        if kind == "classical":
            joblib.dump(payload, temporary / filename, compress=3)
        else:
            shutil.copy2(artifact, temporary / filename)
        manifest = {
            "format_version": 1, "model": "mil" if kind == "mil" else payload["model"],
            "kind": kind, "description": description, "artifact": filename,
            "sha256": file_sha256(temporary / filename),
            "exported_at_utc": datetime.now(timezone.utc).isoformat(),
            "export_environment": {"sklearn": sklearn.__version__, "torch": str(torch.__version__)},
            "input_schema": {"numeric_channels": 9, "sequence_length": 7, "bases": "ACGT"},
            "inference_defaults": {"eval_iterations": eval_iterations, "seed": seed},
        }
        if kind == "classical":
            manifest["feature_names"] = FEATURE_NAMES
            manifest["training_sklearn_version"] = payload["sklearn_version"]
        else:
            manifest["parameters"] = payload["baseline_parameters"]
        (temporary / "model.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
        (temporary / "provenance.json").write_text(json.dumps(provenance,
                                                              indent=2, allow_nan=False))
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="Completed run, selected trial, final export, or study folder")
    parser.add_argument("--output", type=Path, required=True,
                        help="New model folder, for example models/extra_trees")
    parser.add_argument("--description", default="")
    parser.add_argument("--eval-iterations", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    print("Packaged model:", export_model(args.source, args.output,
          args.description, args.eval_iterations, args.seed))


if __name__ == "__main__":
    main()
