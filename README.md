# m6A site inference

Use an **Ubuntu Linux RONIN c8i.2xlarge machine (16 GiB RAM, 8 vCPUs)**. Open its terminal and run the commands below in order. Course datasets and five pretrained models are included through Git LFS; no training is required.

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

These commands use the bundled Extra Trees model. Each prints ROC AUC and PR AUC and saves predictions plus a metrics JSON in `predictions/`. Existing output files are protected from overwriting; the timestamp gives each run a new filename.

Evaluate data0.

```bash
uv run --locked python predict.py --model-dir models/extra_trees_data01 --input data/data0/dataset0.json --output predictions/data0-$(date +%Y%m%d_%H%M%S).csv --n-jobs 8 --threads 8
```

Evaluate data1.

```bash
uv run --locked python predict.py --model-dir models/extra_trees_data01 --input data/data1/dataset1.json --output predictions/data1-$(date +%Y%m%d_%H%M%S).csv --n-jobs 8 --threads 8
```

Evaluate data2.

```bash
uv run --locked python predict.py --model-dir models/extra_trees_data01 --input data/data2/dataset2.json --output predictions/data2-$(date +%Y%m%d_%H%M%S).csv --n-jobs 8 --threads 8
```

The models were trained on data0/data1 training partitions, so whole-file data0/data1 metrics include training sites. Data2 training labels use `label > 0`; its per-dataset metrics use original `label == 0` and inverted scores.

Optional: [other pretrained models and inference details](docs/inference.md), [custom input and label formats](docs/data.md), [training and tuning](docs/training.md).

Task 2: [download and predict SG-NEx cell-line datasets](docs/sgnex.md), including instructions and expected outputs.

See [selected-parameter retraining](docs/training.md#retrain-selected-parameters) for all-data training, split/full inference bundles and the two generated metric tables. Whole-file inference includes training rows; use the split run’s table for held-out results.
