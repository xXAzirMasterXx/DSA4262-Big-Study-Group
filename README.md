# m6A site classifier baseline

This project trains one m6Anet-inspired MIL model on binary site labels from data0 and data1. Each site supplies nine numeric features per read and three overlapping 5-mers. Training uses fixed-size read bags and noisy-OR pooling. Data2 is an external test dataset and never contributes to training, normalization, or checkpoint selection.

## Setup and run

From the repository root, with [uv](https://docs.astral.sh/uv/getting-started/installation/) installed:

```bash
uv sync --locked
uv run --locked python train.py --data-dir ./data --output-dir ./runs/grouped_baseline --cache-dir ./runs/m6anet_baseline/cache --device cpu
```

The project pins Python 3.12 in `.python-version`. Dependencies are declared in `pyproject.toml`, and `uv.lock` records the resolved versions. uv creates the root `.venv`; environment activation is optional when using `uv run`.

The supplied data belong in the root `data/` directory:

```text
data/
  data0/dataset0.json
  data0/data.info.labelled
  data1/dataset1.json
  data1/data.info
  data2/dataset2.json
  data2/data.info
```

## Splitting and training

Data0 and data1 are independently divided into approximate 70/15/15 train/validation/test partitions using scikit-learn's `StratifiedGroupKFold`: 20 folds are combined into 14 train, 3 validation, and 3 test folds. Smaller datasets use fewer folds. The algorithm balances site-label proportions as group sizes permit. Cross-dataset transcript overlap does not constrain splitting.

The grouping key is only `transcript_id`; `gene_id` is ignored. All sites from one transcript remain in one partition within their dataset. Data0 is split first on its own labels and transcript IDs, and data1 is split separately on its own labels and transcript IDs. Only then are the corresponding training, validation, and test partitions combined. The same transcript ID in different datasets is independent and may receive different partition assignments.

```bash
uv run --locked python train.py --data-dir ./data --split-only
```

Change proportions with `--split-ratios`, or the reproducible seed with `--seed`. The model trains on combined data0/data1 training sites. Means and standard deviations use training reads only. Inverse-class-frequency sampling is enabled for training; `--no-balanced-sampler` disables it. Reported metrics always use the natural, unweighted site distribution. No prior correction is applied to the read probabilities or pooled scores.

Defaults: 10 epochs, batch size 256, 20 reads per bag, and 5 evaluation iterations. Each epoch logs BCE loss, PR AUC, ROC AUC. Evaluation bags are deterministic across epochs without changing training RNGs. The checkpoint is selected by pooled validation PR AUC; `--selection-metric loss` or `roc_auc` changes selection. Test is evaluated only after the best checkpoint is restored.

## Data2 evaluation

The entire dataset is evaluated once after checkpoint selection. Every read is scored in bounded chunks using training normalization and the selected model. Those same read probabilities supply two outputs:

- **Noisy-OR:** average fixed-size bag scores over `--eval-iterations` (default five), sampling without replacement when enough reads exist. Compare with binary `label > 0` using BCE loss, PR AUC, and ROC AUC.
- **Mean pooling:** average probabilities across all reads at each site. Compare with the original fraction using site-averaged MAE and RMSE. Fraction accuracy is exploratory because training uses only binary site supervision.

Errors are also summarized by original fraction label. All sites have equal metric weight. Single-class partitions report null AUC values with a reason. Data2 is never subdivided or used to tune the model. Its full dataset contains both binary classes under `label > 0`.

## Outputs

Each run creates a timestamped directory containing:

- `datasetN_splits.csv` and `run_info.json`: assignments, actual counts, and run settings. Every data2 site is marked test.
- `history.json`, `history.csv`, `training_curves.png`, and `training_curves.pdf`: epoch train/validation loss and AUC curves for pooled, data0, and data1 results, with the selected epoch marked. These training metrics are measured after each epoch on natural training sites; batch logs separately show sampled training BCE.
- `best_model.pt`: selected weights and training-only normalization.
- `metrics.json` and `metrics.csv`: selected-checkpoint metrics for data0/data1 train/val/test and data2's two pooling methods.
- Six data0/data1 partition prediction CSVs; `dataset2_test_predictions.csv` with original fractions, binary labels, noisy-OR scores, and mean scores; `dataset2_by_fraction.csv` with mean predictions, MAE, and RMSE per fraction.
