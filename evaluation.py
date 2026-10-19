"""Shared binary metrics for MIL and classical site classifiers."""
import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score


def compute_metrics(labels, scores):
    """Binary BCE, trapezoidal PR AUC (as in m6Anet), and ROC AUC."""
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if not len(labels) or labels.shape != scores.shape:
        raise ValueError(
            "Metrics require nonempty, equally sized labels and scores")
    if not np.isfinite(labels).all() or not np.isfinite(scores).all():
        raise ValueError("Metrics require finite targets and probabilities")
    if not np.isin(labels, [0, 1]).all() or not ((scores >= 0) & (scores <= 1)).all():
        raise ValueError(
            "Classification targets must be binary and scores in [0, 1]")
    clipped = np.clip(scores, 1e-7, 1 - 1e-7)
    binary = (labels > 0).astype(int)
    metrics = {
        "loss": float(
            -(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped)).mean()
        ),
        "pr_auc": None,
        "roc_auc": None,
        "sites": int(len(labels)),
        "auc_positive_sites": int(binary.sum()),
        "auc_target": "label > 0",
        "auc_unavailable_reason": None,
    }
    if np.unique(binary).size < 2:
        metrics["auc_unavailable_reason"] = (
            "Only one binary target class is present in this split"
        )
    else:
        precision, recall, _ = precision_recall_curve(binary, scores)
        metrics["pr_auc"] = float(auc(recall, precision))
        metrics["roc_auc"] = float(roc_auc_score(binary, scores))
    return metrics


def partition_metrics(sites, scores):
    scores = np.asarray(scores)
    metrics = {"pooled": compute_metrics(
        sites["label"].to_numpy(), scores), "datasets": {}}
    for number in sorted(sites["dataset"].unique()):
        mask = sites["dataset"].to_numpy() == number
        metrics["datasets"][str(number)] = compute_metrics(
            sites.loc[mask, "label"].to_numpy(), scores[mask])
    return metrics


def print_metrics(prefix, metrics):
    for name, values in [
        ("pooled", metrics["pooled"]),
        *[(f"data{n}", m) for n, m in metrics["datasets"].items()],
    ]:

        def display(value):
            return "N/A (single target class)" if value is None else f"{value:.6f}"

        print(
            f"{prefix} {name}: loss={values['loss']:.6f} | "
            f"PR AUC={display(values['pr_auc'])} | ROC AUC={display(values['roc_auc'])}",
            flush=True,
        )


def export_predictions(output_dir, split, sites, scores):
    predictions = sites[["dataset", "row_id", "transcript_id",
                         "transcript_position", "label", "split"]].copy()
    predictions["score"] = scores
    for number in sorted(predictions["dataset"].unique()):
        predictions[predictions["dataset"] == number].to_csv(
            output_dir / f"dataset{number}_{split}_predictions.csv", index=False
        )


def metric_records(split, metrics, pooling):
    return [
        {"split": split, "dataset": dataset, "pooling": pooling, **values}
        for dataset, values in [("pooled", metrics["pooled"]), *metrics["datasets"].items()]
    ]
