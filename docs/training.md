# Training

Run on Ubuntu RONIN c8i.2xlarge (16 GiB RAM, 8 vCPUs), from the repository root after the README setup. Five models are supported: Extra Trees, Random Forest, histogram Gradient Boosting, Logistic Regression and MIL.

## Retrain selected parameters

Selected parameters and tuning provenance are stored in `configs/best/`. Both modes train on data0/data1/data2 using seed 42. Original studies and Optuna are not required.

Train on 70% of each dataset, select checkpoints on 15% validation, evaluate on 15% held-out test, and export inference bundles.

```bash
uv run --locked python retrain.py --mode split --n-jobs 8 --threads 8 --export-dir models/retrained_split
```

Train on 100% of each dataset, evaluate on the training rows, and export inference bundles.

```bash
uv run --locked python retrain.py --mode full --n-jobs 8 --threads 8 --export-dir models/retrained_full
```

- Results are saved in timestamped folders under `runs/retrained/split/` and `runs/retrained/full/`. Each includes `evaluation.csv` and `evaluation.md`: per-dataset ROC AUC and PR AUC tables. Model subfolders retain checkpoints, predictions and split assignments.
- Split scores use held-out test rows. Full scores use training rows and are biased. Row splitting is independent, without transcript/gene grouping.
- Data2 training labels use `label > 0`. Its per-dataset evaluation treats original `label == 0` as positive and uses `1 - score`. Pooled metrics use modification-positive labels/scores across all datasets.
- MIL split training allows 30 epochs and selects by pooled validation PR AUC. Full training uses a fixed 28 epochs. Class balancing and training-only normalization remain enabled.
- The commands export to `models/retrained_split/<model>` and `models/retrained_full/<model>` to preserve the published bundles in `models/split/` and `models/full/`. Destinations must not already exist. Each command starts fresh fits; `--models` accepts a subset of the five model names.
- Parameters were selected from the recorded data0/data1 studies. Retraining reuses them without additional tuning. Local analysis reports and training runs are ignored by Git.

## Train individual models

Train Extra Trees with the selected parameters on stratified 70/15/15 partitions of all three datasets.

```bash
uv run --locked python train.py --data-dir data --model extra_trees --model-params configs/best/extra_trees.json --cache-dir runs/cache --output-dir runs/extra_trees --n-jobs 8 --threads 8
```

`--train-datasets`, `--val-datasets` and `--test-datasets` default to `0 1 2`. `--split-ratios` controls proportions; `--dataset-splits` accepts a JSON mapping of dataset IDs to ratios. `configs/final_splits.json` assigns every row to training. Use empty `--val-datasets --test-datasets` for full fitting. Data3/data4 can be selected explicitly with the same folder convention.

Classical models use 65 features: mean/std/25th/75th percentile of nine read channels (36), seven-base A/C/G/T one-hot encoding (28), and `log1p(n_reads)` (1). Logistic Regression scales numeric features using training rows. MIL uses read bags, noisy-OR pooling and training-only normalization.

## Optional tuning

Install tuning dependencies.

```bash
uv sync --locked --group tuning
```

Tune the five models on all three datasets.

```bash
uv run --locked --group tuning python tune.py --data-dir data --variants data012 --n-jobs 8 --threads 8
```

See [tuning](tuning.md) for budgets and resuming. See [model publication](../models/README.md) for Git LFS export details. Saved models include preprocessing and provenance.
