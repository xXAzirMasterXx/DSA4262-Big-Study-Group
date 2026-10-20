# m6A site inference

Use an **Ubuntu Linux RONIN c8i.2xlarge machine (16 GiB RAM, 8 vCPUs)**. Open its terminal and run the commands below in order. Course datasets and five pretrained models in split and full versions are included through Git LFS; no training is required.

## Setup

Update the package list.

```bash
sudo apt-get update
```

Install Git, Git LFS and curl.

```bash
sudo apt-get install -y git git-lfs curl
```

Enable Git LFS.

```bash
git lfs install
```

Install [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Add uv to the current terminal's PATH.

```bash
source "$HOME/.local/bin/env"
```

Verify uv is available.

```bash
uv --version
```

Clone the repository without downloading large files during checkout.

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/xXAzirMasterXx/DSA4262-Big-Study-Group.git
```

Enter the repository.

```bash
cd DSA4262-Big-Study-Group
```

Download the extracted course datasets and pretrained weights.

```bash
git lfs pull
```

Install the locked Python environment.

```bash
uv sync --locked
```

Data layout: `data/data0/` contains `dataset0.json` and `data.info.labelled`; `data/data1/` and `data/data2/` each contain `dataset1.json` or `dataset2.json` and `data.info`. No decompression is needed.

## Predict and evaluate

These commands use full Extra Trees, our best model on the held-out data0/data1 benchmarks. All course rows were used to train this model, so the metrics below are biased training-set evaluations. Each prints ROC AUC and PR AUC and saves predictions plus a metrics JSON in `predictions/`. Existing output files are protected from overwriting; the timestamp gives each run a new filename.

Evaluate data0.

```bash
uv run --locked python predict.py --model-dir models/full/extra_trees --input data/data0/dataset0.json --output predictions/data0-$(date +%Y%m%d_%H%M%S).csv --n-jobs 8 --threads 8
```

Evaluate data1.

```bash
uv run --locked python predict.py --model-dir models/full/extra_trees --input data/data1/dataset1.json --output predictions/data1-$(date +%Y%m%d_%H%M%S).csv --n-jobs 8 --threads 8
```

Evaluate data2.

```bash
uv run --locked python predict.py --model-dir models/full/extra_trees --input data/data2/dataset2.json --output predictions/data2-$(date +%Y%m%d_%H%M%S).csv --n-jobs 8 --threads 8
```

## Model performance

Each cell is **ROC AUC / PR AUC**. **Unbiased (split)** means evaluation on the held-out 15% of each dataset, after 70% training and 15% validation. Rows are split independently, without transcript/gene grouping. **Biased (full)** means training and evaluation on 100% of the same data; these scores do not measure generalization.

Data2 metrics treat original `label == 0` as positive and use `1 - score`; training and prediction scores remain modification-positive (`label > 0`). See [evaluation details](docs/inference.md#evaluation).

### Unbiased split

| Model | data0 | data1 | data2 |
|---|---:|---:|---:|
| Extra Trees | 0.9376 / 0.5739 | 0.8632 / 0.4704 | 0.9462 / 0.7271 |
| Random Forest | 0.9326 / 0.5282 | 0.8461 / 0.4176 | 0.9318 / 0.6595 |
| Gradient Boosting | 0.9316 / 0.5226 | 0.8313 / 0.3893 | 0.8826 / 0.5590 |
| Logistic Regression | 0.8992 / 0.4002 | 0.8122 / 0.3098 | 0.6897 / 0.2350 |
| MIL | 0.9342 / 0.4971 | 0.8393 / 0.3872 | 0.9377 / 0.8501 |

[Full-precision results](results/split/evaluation.csv).

### Biased full

| Model | data0 | data1 | data2 |
|---|---:|---:|---:|
| Extra Trees | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| Random Forest | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| Gradient Boosting | 0.9916 / 0.7921 | 0.9882 / 0.7986 | 0.9841 / 0.9454 |
| Logistic Regression | 0.8977 / 0.4038 | 0.8140 / 0.3307 | 0.7706 / 0.3057 |
| MIL | 0.9427 / 0.5144 | 0.8750 / 0.4128 | 0.9653 / 0.8831 |

[Full-precision results](results/full/evaluation.csv).

Optional: [other models and inference details](docs/inference.md), [custom datasets](docs/data.md), [training and tuning](docs/training.md), [SG-NEx predictions for Task 2](docs/sgnex.md).
