# m6A site classifier

A m6Anet-inspired multiple instance learning (MIL) model with nine features per read, sequence embeddings, and noisy-OR pooling. All targets are binary; data2 labels are converted using `label > 0`.

## Run

```bash
uv sync --locked
# Run each training variant against the same validation and test sets.
uv run python train.py --data-dir data --train-datasets 0 --output-dir runs/data0 --cache-dir runs/cache
uv run python train.py --data-dir data --train-datasets 1 --output-dir runs/data1 --cache-dir runs/cache
uv run python train.py --data-dir data --train-datasets 0 1 --output-dir runs/combined --cache-dir runs/cache
```

Expected files: `data/dataN/datasetN.json` and `data/dataN/data.info`. Data0 uses `data.info.labelled` instead. New data3, data4, etc. follow the same convention and require binary labels.

## Dataset selection

- Each dataset is split independently into stratified 70/15/15 train/validation/test rows. Transcript and gene IDs do not constrain splitting. Matching sites across datasets remain independent.
- Splits depend on the seed (default 42), dataset number, row order, labels, and ratios, never on which training variant is selected. Keep these inputs fixed for comparisons.
- `--train-datasets` defaults to `0 1`. Combined training uses the same training rows as the individual variants.
- `--val-datasets` defaults to `0 1`; `--test-datasets` defaults to `0 1 2`. These stay fixed across training variants. Metrics include pooled and individual dataset results. Pooled test includes data2.
- Data2 always uses 100% test rows and cannot be selected for training or validation.
- `--split-ratios 0.7 0.15 0.15` changes the default ratios for datasets other than data2. Integer row counts may require rounding. Very small classes may not support stratification.
- `--dataset-splits splits.json` overrides ratios per dataset, for example `{"0": [1, 0, 0], "1": [1, 0, 0]}` for final training on all data0/data1 rows:

```bash
uv run python train.py --data-dir data --dataset-splits splits.json --train-datasets 0 1 --val-datasets --test-datasets 2
```

With no validation datasets, the final epoch is saved. Use `--split-only` to export assignments without loading features or training. Assignment files include all partitions for each loaded dataset; the dataset selectors determine which partitions are actually used.

## Training and outputs

Defaults: 10 epochs, batch size 256, 20 reads per bag, and 5 evaluation iterations. Normalization uses training reads only. Inverse-class-frequency sampling is enabled; disable with `--no-balanced-sampler`.

Checkpoint selection uses pooled validation PR AUC (`--selection-metric loss` or `roc_auc` also supported). Tests run after checkpoint selection. Every dataset uses the same sampled-bag noisy-OR evaluation with BCE, PR AUC, and ROC AUC. Single-class AUCs are null. There is no fraction regression evaluation.

Each timestamped output directory contains split CSVs, settings and split counts in `run_info.json`, checkpoint weights and normalization in `best_model.pt`, epoch history and training curves, final metrics, and per-dataset prediction CSVs. `--cache-dir` shares prepared features across runs.

Run checks with `uv run python -m unittest discover -s tests`.
