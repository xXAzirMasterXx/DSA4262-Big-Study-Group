# Inference details

Follow the root README setup first. Commands target RONIN c8i.2xlarge with `--n-jobs 8 --threads 8`. The first flag controls supported estimator prediction jobs; the second controls numerical/PyTorch CPU threads. Models without job-based prediction parallelism ignore the first setting.

## Included models

All are baseline models trained on the combined data0/data1 training partitions, not Optuna winners. Select the folder using `--model-dir`.

| Model | Folder |
| --- | --- |
| MIL | `models/mil_data01` |
| Extra Trees | `models/extra_trees_data01` |
| Random forest | `models/random_forest_data01` |
| Histogram gradient boosting | `models/gradient_boosting_data01` |
| Logistic regression | `models/logistic_regression_data01` |
| RBF SVM | `models/rbf_svm_data01` |

Evaluate the MIL model on data2.
```bash
uv run --locked python predict.py --model-dir models/mil_data01 --input data/data2/dataset2.json --output predictions/mil-data2-$(date +%Y%m%d_%H%M%S).csv --n-jobs 8 --threads 8
```

## Inputs and outputs

- Inference needs only a trusted model folder and precomputed feature JSON. Joblib loading can execute Python code; checksums verify file integrity, not trustworthiness.
- Adjacent `data.info.labelled` or `data.info` files containing labels are discovered automatically. Use `--labels` for a custom CSV or `--no-labels` to disable evaluation. See [formats](data.md#custom-inference-labels).
- Predictions include dataset, row ID, transcript ID, site position, read count, score and matched labels when available. The companion `.csv.json` records settings, model checksum, ROC AUC, trapezoidal PR AUC and label coverage. Single-class AUCs are null with an explanation.
- Classical models reuse fitted preprocessing/calibration and all reads. MIL uses saved normalization and sampled read bags. `--seed` and `--eval-iterations` override packaged defaults. Preserve input order and settings for repeatability; sampling differs from training evaluation.
- Existing outputs are not overwritten. README commands use timestamps to support repeated evaluation.

## Troubleshooting

- If weights remain Git LFS pointers or data cannot be parsed, rerun `git lfs pull` and check that it completes successfully. Downloads require the repository's LFS objects to have been pushed and hosting quota to be available.
- If `uv` is not found, rerun the README PATH command in the current terminal.
- If a model reports an incompatible scikit-learn version, use the repository's locked environment with `uv sync --locked`.

See [export and publication](../models/README.md) for packaging other checkpoints.
