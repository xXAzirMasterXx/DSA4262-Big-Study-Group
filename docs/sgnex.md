# SG-NEx prediction workflow (Task 2)

Complete the root README setup first. Use the same Ubuntu RONIN c8i.2xlarge machine with 16 GiB RAM and 8 vCPUs. This optional workflow downloads public data and predicts modifications without training or labels.

## Start with one sample

List the pinned samples and download sizes without downloading anything.

```bash
uv run --locked python sgnex.py --list
```

Download and predict one A549 run using the bundled Extra Trees model.

```bash
uv run --locked python sgnex.py --samples SGNex_A549_directRNA_replicate5_run1 --model-dir models/full/extra_trees --n-jobs 8 --threads 8
```

Repeat that same command after an interruption. Download byte ranges resume automatically. A completed prediction is skipped only after checking its contract and output checksums. An interrupted prediction restarts that sample; it does not restart completed samples.

The sample from the earlier manual example can be reused at `data/sgnex/SGNex_A549_directRNA_replicate5_run1/data.json`: the workflow verifies its size and single-part S3 checksum and creates a receipt. Other manually downloaded files with multipart S3 ETags cannot be adopted without a receipt; move those files aside or select another data directory. The script does not overwrite unverified existing data.

## Run the collection

Download and predict all pinned SG-NEx runs, reusing completed samples.

```bash
uv run --locked python sgnex.py --model-dir models/full/extra_trees --n-jobs 8 --threads 8
```

Alternatively, process only the MCF7 cell line.

```bash
uv run --locked python sgnex.py --model-dir models/full/extra_trees --cell-lines MCF7 --n-jobs 8 --threads 8
```

The manifest `configs/sgnex_samples.json` was captured on 20 October 2026. It contains 23 processed runs across A549, H9, HEYA8, Hct116, HepG2, K562 and MCF7. Feature JSON downloads total approximately 33.37 GB.

- Keep sufficient free disk space for downloads, predictions and the Python environment. The downloader checks space for the next JSON plus a 1 GiB reserve, but this is not a guarantee that the entire collection fits. Inspect space before the full run.
- Raw downloads remain in `data/sgnex/`; predictions remain in `predictions/sgnex/`. Both are ignored by Git. Do not put this collection in Git LFS.
- Public HTTPS downloads need no AWS credentials or AWS CLI. `curl` is installed by the README setup.
- Do not run two workflows against the same data/output directories simultaneously. The script takes exclusive locks.

Check free disk space on the current drive.

```bash
df -h .
```

## What to expect

The default output folder is `predictions/sgnex/extra_trees/` (assuming you use the Extra Trees model as above) (or you can specify `--output-dir` to change it) contains:

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
