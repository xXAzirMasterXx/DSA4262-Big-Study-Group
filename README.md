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

Data layout: `data/data0/` contains `dataset0.json` and `data.info.labelled`; `data/data1/` and `data/data2/` contain their feature JSON and `data.info`. The unlabelled final leaderboard datasets are in `data/data3/dataset3.json` through `data/data7/dataset7.json`. All eight datasets are already extracted; no decompression is needed.

## Leaderboard predictions

The official submission files for **Big Study Group** are in [predictions/leaderboard/](predictions/leaderboard/). Newcomers can inspect or download these CSVs after `git lfs pull` without running inference. Submit exactly these eight CSVs, named `Big Study Group_dataset0.csv` through `Big Study Group_dataset7.csv`. Do not submit the accompanying `manifest.json`.

Each CSV follows [sample_submission.csv](reference/sample_submission.csv) exactly: `transcript_id,transcript_position,score`, with no index or diagnostic columns. Scores are modification probabilities from the **full Extra Trees** model, using all reads. Dataset2 scores are not inverted for submission. The manifest records site counts, input/output checksums, model checksum and inference settings.

### Reproduce all eight files

After completing Setup, run this command from the repository root. It writes to a fresh directory because the official files are already included and protected from overwriting.

```bash
uv run --locked python submit_predictions.py --team-name "Big Study Group" --output-dir predictions/leaderboard-reproduced --n-jobs 8 --threads 8
```

The command processes datasets 0 through 7, disables label-based evaluation, checks the sample header, rejects duplicate site keys or invalid scores, and publishes the output directory only after all eight files succeed. To create the official directory in a checkout where it does not yet exist, omit `--output-dir`.

To publish a newly generated official set after reviewing the CSVs and manifest:

```bash
git add predictions/leaderboard
git commit -m "feat: publish leaderboard predictions for datasets 0-7"
git push origin main
```

## Optional evaluation

The full Extra Trees model was trained on all labelled rows from data0/data1/data2, so evaluations on those datasets are biased training-set evaluations. Datasets 3 through 7 have no labels and cannot be evaluated locally. For example, evaluate data0 and save a diagnostic CSV plus metrics separately:

```bash
uv run --locked python predict.py --model-dir models/full/extra_trees --input data/data0/dataset0.json --output predictions/data0-$(date +%Y%m%d_%H%M%S).csv --n-jobs 8 --threads 8
```

Use the corresponding paths for data1 or data2. Diagnostic CSVs from `predict.py` contain extra columns and must not be submitted directly. See [inference details](docs/inference.md).

## Model performance

Each cell is **ROC AUC / PR AUC**. **Unbiased (split)** means evaluation on the held-out 15% of each dataset, after 70% training and 15% validation. Rows are split independently, without transcript/gene grouping. **Biased (full)** means training and evaluation on 100% of the same data; these scores do not measure generalization.

Data2 metrics treat original `label == 0` as positive and use `1 - score`; training and prediction scores remain modification-positive (`label > 0`). See [evaluation details](docs/inference.md#evaluation).

### Unbiased split

| Model               |           data0 |           data1 |           data2 |
| ------------------- | --------------: | --------------: | --------------: |
| Extra Trees         | 0.9376 / 0.5739 | 0.8632 / 0.4704 | 0.9462 / 0.7271 |
| Random Forest       | 0.9326 / 0.5282 | 0.8461 / 0.4176 | 0.9318 / 0.6595 |
| Gradient Boosting   | 0.9316 / 0.5226 | 0.8313 / 0.3893 | 0.8826 / 0.5590 |
| Logistic Regression | 0.8992 / 0.4002 | 0.8122 / 0.3098 | 0.6897 / 0.2350 |
| MIL                 | 0.9342 / 0.4971 | 0.8393 / 0.3872 | 0.9377 / 0.8501 |

[Full-precision results](results/split/evaluation.csv).

### Biased full

| Model               |           data0 |           data1 |           data2 |
| ------------------- | --------------: | --------------: | --------------: |
| Extra Trees         | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| Random Forest       | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| Gradient Boosting   | 0.9916 / 0.7921 | 0.9882 / 0.7986 | 0.9841 / 0.9454 |
| Logistic Regression | 0.8977 / 0.4038 | 0.8140 / 0.3307 | 0.7706 / 0.3057 |
| MIL                 | 0.9427 / 0.5144 | 0.8750 / 0.4128 | 0.9653 / 0.8831 |

[Full-precision results](results/full/evaluation.csv).

## Optional

- [SG-NEx predictions for Task 2](docs/sgnex.md): explore the included Extra Trees predictions without running inference, or reproduce them from public inputs.
- [Inference details](docs/inference.md): choose another model and understand prediction files and evaluation metrics.
- [Custom datasets](docs/data.md): prepare your own feature JSON and optional labels.
- [Training and tuning](docs/training.md): retrain models or search for different hyperparameters.
