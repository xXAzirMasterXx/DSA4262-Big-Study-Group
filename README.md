# m6A site classifiers

Compare a m6Anet-inspired MIL model with Extra Trees, random forest, histogram gradient boosting, logistic regression, and RBF SVM. All predict a modification probability per transcript site.

## Setup

Commands assume a **Linux machine**, with Git and curl installed, and run from the repository root. Other operating systems use similar setup. CPU inference is the default.

Install [uv using the official instructions](https://docs.astral.sh/uv/getting-started/installation/), then verify it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# Reopen your shell after installation, then:
uv --version

git clone https://github.com/xXAzirMasterXx/DSA4262-Big-Study-Group.git
cd DSA4262-Big-Study-Group
uv sync --locked
```

Keep `.python-version` and `uv.lock` unchanged to reproduce the environment. `uv run` uses the project environment without manual activation.

## Inference without training

Six pretrained models are included under `models/`, using the combined data0/data1 training partitions. These are the completed baseline runs, not Optuna winners. Install [Git LFS](https://git-lfs.com/) using its official Linux instructions, then fetch the weights inside the clone:

```bash
git lfs version
git lfs install
git lfs pull
```

Put your uncompressed feature JSON at `data/new/dataset.json` (or pass its actual path). No labels or metadata CSV are required. This must contain precomputed read features, not raw sequencing files. See [input format and dataset preparation](docs/data.md).

Choose a model folder:

| Model                       | Folder                              |
| --------------------------- | ----------------------------------- |
| MIL                         | `models/mil_data01`                 |
| Extra Trees                 | `models/extra_trees_data01`         |
| Random forest               | `models/random_forest_data01`       |
| Histogram gradient boosting | `models/gradient_boosting_data01`   |
| Logistic regression         | `models/logistic_regression_data01` |
| RBF SVM                     | `models/rbf_svm_data01`             |

Run inference:

```bash
uv run --locked python predict.py --model-dir models/extra_trees_data01 --input data/new/dataset.json --output predictions/new.csv
# Use --model-dir models/mil_data01 for the packaged MIL model.
```

When `data.info.labelled` or `data.info` beside the input contains labels, inference automatically prints ROC AUC and PR AUC and saves them in the companion JSON (`predictions/new.csv.json`). This works with the course datasets:

```bash
uv run --locked python predict.py --model-dir models/extra_trees_data01 --input data/data2/dataset2.json --output predictions/data2.csv
# For data0/data1, use data/data0/dataset0.json or data/data1/dataset1.json.
# For custom labels, add --labels /path/to/labels.csv; disable evaluation with --no-labels.
```

Metrics match labels by transcript ID and position, using `label > 0` as the binary target (including data2).

- Only the model folder and feature JSON are needed. No fitting, labels or training data are required.
- Output contains dataset name, row ID, transcript ID, position, read count and `score` (modification probability). A companion JSON records model checksum and inference settings. Existing outputs are never overwritten.
- Classical models reuse saved preprocessing/calibration and all reads. MIL reuses saved normalization and averages sampled read bags; `--seed` and `--eval-iterations` override packaged defaults. Keep input order and settings fixed for repeatability. Its per-site sampling differs from training-time evaluation, so scores can differ from run prediction files.
- Use only trusted model folders: loading joblib artifacts can execute Python code. Checksums detect changed files, not trustworthiness.

For training, hyperparameter tuning, custom splits and model export, see the [training guide](docs/training.md).

For custom evaluation labels, see the [labels.csv format and example](docs/data.md#custom-inference-labels).
