# Inference details

Follow the root README setup first. Commands target RONIN c8i.2xlarge with `--n-jobs 8 --threads 8`.

## Included models

Each model is provided in `models/full/` (100% training) and `models/split/` (70% training), using data0/data1/data2. The table lists full models; replace `full` with `split` to select a split bundle. Select the folder using `--model-dir`.

| Model                       | Folder                            |
| --------------------------- | --------------------------------- |
| MIL (m6Anet)                | `models/full/mil`                 |
| Extra Trees                 | `models/full/extra_trees`         |
| Random forest               | `models/full/random_forest`       |
| Histogram gradient boosting | `models/full/gradient_boosting`   |
| Logistic regression         | `models/full/logistic_regression` |

Evaluate the MIL model on data2.

```bash
uv run --locked python predict.py --model-dir models/full/mil --input data/data2/dataset2.json --output predictions/mil-data2-$(date +%Y%m%d_%H%M%S).csv --n-jobs 8 --threads 8
```

## Leaderboard submission

Use `submit_predictions.py --team-name "Big Study Group" --output-dir predictions/leaderboard-reproduced` through `uv run --locked python` to generate all eight sample-format CSVs with the full Extra Trees model. The official files are already available in [predictions/leaderboard/](../predictions/leaderboard/). See the [root README](../README.md#leaderboard-predictions) for the reproduction command. Datasets 3 through 7 contain feature JSON only; no label metadata is required.

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

## Evaluation

The [split results](../results/split/evaluation.md) evaluate only the held-out 15% of each dataset. These are the unbiased row-level benchmark results, with no transcript/gene grouping. Data2 has 199 test sites, including 29 evaluation positives.

The [full results](../results/full/evaluation.md) evaluate the same rows used for training and are biased. Whole-file inference also includes training rows even when using a split bundle; it does not reproduce the held-out table. MIL sampling can differ between training evaluation and standalone inference.

Data0/data1 metrics use modification-positive labels and scores. Data2 metrics use original `label == 0` as positive and `1 - score`. Data2 training labels remain `label > 0`, and saved prediction scores always represent modification probability. PR AUC uses trapezoidal integration of the precision-recall curve.
