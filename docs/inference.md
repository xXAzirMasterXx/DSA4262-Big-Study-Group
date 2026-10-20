# Inference details

Follow the root README setup first. Commands target RONIN c8i.2xlarge with `--n-jobs 8 --threads 8`.

## Included models

All are baseline models trained on the combined data0/data1 training partitions. Select the folder using `--model-dir`.

| Model                       | Folder                              |
| --------------------------- | ----------------------------------- |
| MIL (m6Anet)                | `models/mil_data01`                 |
| Extra Trees                 | `models/extra_trees_data01`         |
| Random forest               | `models/random_forest_data01`       |
| Histogram gradient boosting | `models/gradient_boosting_data01`   |
| Logistic regression         | `models/logistic_regression_data01` |

Evaluate the MIL model on data2.

```bash
uv run --locked python predict.py --model-dir models/mil_data01 --input data/data2/dataset2.json --output predictions/mil-data2-$(date +%Y%m%d_%H%M%S).csv --n-jobs 8 --threads 8
```

## Inputs and outputs

- Inference needs only a trusted model folder and precomputed feature JSON. Labels are optional.
- Adjacent `data.info.labelled` or `data.info` files containing labels are discovered automatically. Use `--labels` for a custom CSV or `--no-labels` to disable evaluation. See [formats](data.md#custom-inference-labels).
- Predictions include dataset, row ID, transcript ID, site position, read count, score and matched labels when available. The companion `.csv.json` records settings, model checksum, ROC AUC, trapezoidal PR AUC and label coverage. Single-class AUCs are null with an explanation.
- Classical models reuse fitted preprocessing and all reads. MIL uses saved normalization and sampled read bags. `--seed` and `--eval-iterations` override packaged defaults. Preserve input order and settings for repeatability; sampling differs from training evaluation.

## Troubleshooting

- If weights remain Git LFS pointers or data cannot be parsed, rerun `git lfs pull` and check that it completes successfully.
- If `uv` is not found, rerun the README PATH command in the current terminal.
- If a model reports an incompatible scikit-learn version, use the repository's locked environment with `uv sync --locked`.

See [export and publication](../models/README.md) for packaging other checkpoints.

## Retrained models

The retraining workflow exports `models/split/` and `models/full/`, each containing the five model folders. Both modes train on data0/data1/data2. Use the split run's `evaluation.md` for held-out test performance; whole-file inference includes training rows. Full-run metrics are biased training-set evaluations.

After full-data retraining, predict and evaluate data2 with the exported Extra Trees model.

```bash
uv run --locked python predict.py --model-dir models/full/extra_trees --input data/data2/dataset2.json --output predictions/full-data2-$(date +%Y%m%d_%H%M%S).csv --n-jobs 8 --threads 8
```
