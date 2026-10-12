# m6A site classifier baseline

This standalone project adapts the multiple-instance learning approach in the supplied `m6anet-master.zip` to the provided datasets. It trains one fixed baseline model on all labeled sites in datasets 0 and 1, then predicts all sites in datasets 0, 1, and 2. Dataset 2 labels are parsed only as metadata and are never used for training.

For each site, the model randomly samples 20 reads by default, predicts read-level modification probabilities from the nine numeric read features plus the three overlapping 5-mers in the site's 7-mer context, then combines those predictions with noisy-OR pooling. Training addresses low positive-class prevalence with a weighted sampler that oversamples positive sites with replacement. Since this creates a balanced training prior, output scores receive a prior correction using the observed positive fraction in datasets 0 and 1. Validation and test splits are not used in this full-data prediction run.

Each run writes three timestamped prediction CSVs, one per dataset, so previous runs are preserved. Each has exactly these columns:

```csv
transcript_id,transcript_position,score
```

Scores for datasets 0 and 1 are in-sample predictions because those sites were used to train the model. Dataset 2 was excluded from fitting and is the unseen dataset. The output intentionally has no dataset column, as requested; if the same transcript and position occurs in more than one input dataset, each dataset's row is kept.

Dataset 0's `data.info.labelled` has no `n_reads` column. On the first run, the project counts reads from `data/data0/dataset0.json` before building the feature cache, requiring an extra pass over that JSON file. Later runs reuse the cache unless the source files change.

## Setup

Run these commands from the repository root. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if it is not already available.

```bash
uv sync --locked
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

The loader reads the decompressed JSON files. Data, virtual environments, caches, and runs are ignored by Git; commit `pyproject.toml`, `.python-version`, and `uv.lock`.

To add or remove a dependency, use `uv add PACKAGE` or `uv remove PACKAGE`. Use `uv lock --upgrade` followed by `uv sync --locked` for an intentional dependency upgrade.

## Run

```bash
uv run --locked python train.py --data-dir ./data --output-dir ./runs/m6anet_baseline --device cpu
```

On macOS the current trainer uses CPU. On a machine with a supported NVIDIA GPU, use `--device cuda` or `--device auto`. The current CLI does not support Apple's MPS backend.

The defaults train for 10 epochs with 20 reads per site and batch size 256. Training prints batch progress every 100 batches and an epoch-complete summary. Change that interval with `--log-every-batches` (for example, `--log-every-batches 25`). You can also adjust `--n-reads-per-site`, `--epochs`, `--batch-size`, and `--eval-iterations`. Evaluation iterations average predictions over repeated random read bags for each site; this still trains only one model. Set `--device cuda` if a supported NVIDIA GPU and CUDA-enabled PyTorch are available; otherwise device selection is automatic.

## Outputs

- `dataset0_predictions_<run_id>.csv`, `dataset1_predictions_<run_id>.csv`, `dataset2_predictions_<run_id>.csv`: the requested three-column scores, one file for each dataset. Each execution gets a unique set of files.
- `baseline_model.pt`: model weights, feature normalization statistics, and baseline settings.
- `run_info.json`: training data counts, prediction counts, and baseline settings.
- `cache/`: prepared site tables and memory-mapped read features.

The supplied data contain read-level numeric signals, not raw eventalign/signal files. This adaptation omits m6Anet's raw-signal dataprep and motif-specific normalization tables; it standardizes the nine input channels using training-read statistics. Treat it as an adapted baseline, not the official m6Anet model or pretrained weights.
