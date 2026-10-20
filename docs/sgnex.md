# SG-NEx prediction workflow (Task 2)

## Explore the included predictions

Predictions for all 23 pinned SG-NEx runs are included in [predictions/sgnex/extra_trees/](../predictions/sgnex/extra_trees/), generated with `models/full/extra_trees`. No raw feature downloads or inference are needed to explore these results. Prediction CSVs use Git LFS; metadata uses ordinary Git. The prediction files occupy approximately 2.39 GB.

The root README setup downloads these files already. For an existing checkout, download just the published predictions after pulling the latest commit.

```bash
git lfs pull --include="predictions/sgnex/extra_trees/**"
```

Start with [samples.csv](../predictions/sgnex/extra_trees/samples.csv), which maps each sample to its cell line, replicate, run, site count and relative prediction path. Read one sample at a time for exploration or notebook analysis. The accompanying manifests and completion files preserve model, input and output checksums and inference settings. See [what to expect](#what-to-expect) for the file schema and score interpretation.

## Run the analysis notebook

Open [SG-NEx exploration](../notebooks/sgnex_exploration.ipynb) to compare coverage, score distributions, replicate agreement and candidate differences across cell lines. Interactive plots support hovering and site selection; the notebook also exports standalone HTML explorers. It uses the published predictions and does not train models or download raw feature inputs.

After the root README setup, install the optional locked analysis environment on your RONIN machine.

```bash
uv sync --locked --group analysis
```

Start JupyterLab from the repository root.

```bash
uv run --locked --group analysis jupyter lab notebooks/sgnex_exploration.ipynb --no-browser --ip=127.0.0.1 --port=8888
```

In your SSH client's port-forwarding settings, forward local port 8888 to `127.0.0.1:8888` on RONIN. Open the localhost URL with the token printed by JupyterLab in your local browser. Keep the SSH connection and Jupyter process running. Select **Kernel → Restart Kernel and Run All Cells**. The notebook uses one file at a time in chunks and does not need `--n-jobs` or `--threads`.

Outputs are written to ignored `reports/sgnex_analysis/`: summary CSVs, figures, candidate sites, interactive HTML and generated discussion points. MCF7 transcript identifiers differ between its runs, so the notebook retains MCF7 in distribution summaries but excludes it from matched-site comparisons pending annotation harmonization. Read the caveats in the notebook: coverage and score thresholds are exploratory, missing sites are not negative labels, and score differences are not evidence of statistically significant differential modification. The interactive notebook is a starting point for the Task 2 visualization, not a completed biological analysis.

## Reproduce predictions (optional)

Complete the root README setup first. Use Ubuntu RONIN c8i.2xlarge with 16 GiB RAM and 8 vCPUs. The commands below download public inputs and write to `predictions/sgnex/reproduced_extra_trees/`, keeping the published results intact.

## Start with one sample

List the pinned samples and download sizes without downloading anything.

```bash
uv run --locked python sgnex.py --list
```

Download and predict one A549 run using the bundled Extra Trees model.

```bash
uv run --locked python sgnex.py --samples SGNex_A549_directRNA_replicate5_run1 --model-dir models/full/extra_trees --output-dir predictions/sgnex/reproduced_extra_trees --n-jobs 8 --threads 8
```

Repeat that same command after an interruption. Download byte ranges resume automatically. A completed prediction is skipped only after checking its contract and output checksums. An interrupted prediction restarts that sample; it does not restart completed samples.

The sample from the earlier manual example can be reused at `data/sgnex/SGNex_A549_directRNA_replicate5_run1/data.json`: the workflow verifies its size and single-part S3 checksum and creates a receipt. Other manually downloaded files with multipart S3 ETags cannot be adopted without a receipt; move those files aside or select another data directory. The script does not overwrite unverified existing data.

## Run the collection

Download and predict all pinned SG-NEx runs, reusing completed samples.

```bash
uv run --locked python sgnex.py --model-dir models/full/extra_trees --output-dir predictions/sgnex/reproduced_extra_trees --n-jobs 8 --threads 8
```

Alternatively, process only the MCF7 cell line.

```bash
uv run --locked python sgnex.py --model-dir models/full/extra_trees --output-dir predictions/sgnex/reproduced_extra_trees --cell-lines MCF7 --n-jobs 8 --threads 8
```

The manifest `configs/sgnex_samples.json` was captured on 20 October 2026. It contains 23 processed runs across A549, H9, HEYA8, Hct116, HepG2, K562 and MCF7. Feature JSON downloads total approximately 33.37 GB.

- Keep sufficient free disk space for downloads, predictions and the Python environment. The downloader checks space for the next JSON plus a 1 GiB reserve, but this is not a guarantee that the entire collection fits. Inspect space before the full run.
- Raw downloads in `data/sgnex/` and newly reproduced predictions are ignored by Git. Only the published Extra Trees predictions and their metadata are tracked; raw feature inputs are not included.
- Public HTTPS downloads need no AWS credentials or AWS CLI. `curl` is installed by the README setup.
- Do not run two workflows against the same data/output directories simultaneously. The script takes exclusive locks.

Check free disk space on the current drive.

```bash
df -h .
```

## What to expect

The published folder `predictions/sgnex/extra_trees/` and newly generated output folders have the same layout:

| File                                 | Contents                                                                                                |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------- |
| `samples.csv`                        | Sample ID, cell line, replicate, run, predicted site count, model/checksum and relative prediction path |
| `sample_manifest.json`               | Full pinned source inventory                                                                            |
| `run_contract.json`                  | Model, source manifest, code, environment and inference settings                                        |
| Each sample's `predictions.csv`      | Dataset/sample ID, row ID, transcript ID, transcript position, read count and modification score        |
| Each sample's `predictions.csv.json` | Input and inference provenance                                                                          |
| Each sample's `complete.json`        | Completion contract, input checksum and output checksums                                                |

The sample index is refreshed after each attempted sample and includes completed samples in this output folder. Join its `sample_id` with the predictions' `dataset` column for cell-line analysis. Raw CSVs remain separate so that you do not need to load the whole collection into RAM.

SG-NEx metadata supplies indexing and coverage, not ground-truth labels. This workflow deliberately uses unlabeled inference and produces **no ROC AUC or PR AUC**. Scores are site-modification predictions, not modification fractions among reads. Outputs include low-coverage sites; downstream analysis must choose and report a coverage filter.

## Another model or new settings

You can use any of the bundled models (in `models/`) or a tuned model you export. For example, run the same collection with the bundled MIL (m6Anet) model:

```bash
uv run --locked python sgnex.py --samples SGNex_A549_directRNA_replicate5_run1 --model-dir models/full/mil --n-jobs 8 --threads 8
```

Refer to [model packaging](../models/README.md) for instructions on exporting a tuned model. Then pass that folder with `--model-dir`. The default output folder uses the model folder's name. Changed model contents, code, library versions, manifest or inference settings require a new `--output-dir`; they cannot be mixed into existing results. `--seed` defaults to 42 and `--eval-iterations` to 5 explicitly for this workflow, including MIL. Inference uses CPU.

If a command fails, it stops with the sample/error visible. Already completed outputs and partial downloads remain available. Repeating the command resumes them. An uncatchable process kill can leave an ignored temporary `.sgnex-` folder; it is not considered a completed sample. Changing the sample selection alone does not invalidate completed predictions.
