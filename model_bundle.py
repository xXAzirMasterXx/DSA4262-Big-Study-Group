"""Versioned model folders for portable, training-free inference."""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import sklearn
import torch

from mil_model import M6ANetInspiredMIL
from site_features import FEATURE_NAMES

CLASSICAL_MODELS = {"extra_trees", "random_forest",
                    "gradient_boosting", "logistic_regression", "rbf_svm"}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_artifact(path, kind, device="cpu"):
    with path.open("rb") as stream:
        if stream.read(80).startswith(b"version https://git-lfs.github.com/spec/"):
            raise ValueError(
                "Model weights are a Git LFS pointer. Run git lfs pull first.")
    if kind == "classical":
        payload = joblib.load(path)
        if payload.get("model") not in CLASSICAL_MODELS:
            raise ValueError("Unrecognized classical model artifact")
        if payload.get("feature_names") != FEATURE_NAMES:
            raise ValueError(
                "Saved classical feature schema does not match this code")
        if payload.get("sklearn_version") != sklearn.__version__:
            raise ValueError(
                "scikit-learn version differs from the saved model; use its matching locked environment")
        fitted = payload["estimator"]
        if not hasattr(fitted, "predict_proba") or not np.array_equal(fitted.classes_, [0, 1]):
            raise ValueError(
                "Expected a fitted binary classifier with classes 0 and 1")
        if getattr(fitted, "n_features_in_", None) != len(FEATURE_NAMES):
            raise ValueError(
                "Expected a fitted classifier with the documented site features")
        return payload, fitted
    if kind != "mil":
        raise ValueError(f"Unknown artifact type: {kind}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    params = payload["baseline_parameters"]
    if params["n_reads_per_site"] < 1:
        raise ValueError("Saved read bag size must be positive")
    for name in ("feature_mean", "feature_std"):
        values = payload[name].detach().cpu().numpy()
        if values.shape != (9,) or not np.isfinite(values).all():
            raise ValueError(f"Invalid saved normalization: {name}")
    if (payload["feature_std"].numpy() <= 0).any():
        raise ValueError("Saved feature standard deviations must be positive")
    fitted = M6ANetInspiredMIL(
        params["embed_dim"], params["hidden_dim"], params["dropout"])
    fitted.load_state_dict(payload["model_state_dict"], strict=True)
    fitted.to(device).eval()
    return payload, fitted


def load_bundle(folder, device="cpu"):
    folder = Path(folder)
    manifest_path = folder / "model.json"
    if not manifest_path.is_file():
        raise ValueError(
            f"No model.json in {folder}. Export a selected checkpoint with export_model.py first.")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format_version") != 1:
        raise ValueError("Unsupported model folder format")
    artifact = manifest.get("artifact", "")
    if artifact not in ("model.pt", "model.joblib"):
        raise ValueError(
            "Model artifact must be model.pt or model.joblib inside the folder")
    path = folder / artifact
    # Give a useful LFS message before attempting checksum validation.
    with path.open("rb") as stream:
        if stream.read(80).startswith(b"version https://git-lfs.github.com/spec/"):
            raise ValueError(
                "Model weights are a Git LFS pointer. Run git lfs pull first.")
    if file_sha256(path) != manifest.get("sha256"):
        raise ValueError(
            "Model checksum mismatch; restore or re-download the model folder")
    payload, fitted = read_artifact(path, manifest["kind"], device)
    model_name = "mil" if manifest["kind"] == "mil" else payload["model"]
    if manifest.get("model") != model_name:
        raise ValueError("Model manifest and artifact disagree")
    if manifest.get("input_schema") != {"numeric_channels": 9, "sequence_length": 7, "bases": "ACGT"}:
        raise ValueError("Unsupported input schema")
    return manifest, payload, fitted
