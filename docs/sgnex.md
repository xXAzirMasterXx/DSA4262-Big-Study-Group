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

You will use **two terminal windows**: one connected to RONIN, and one on your own computer. Jupyter runs on RONIN; your browser displays it through the second connection.

### 1. In your RONIN terminal

After the root README setup, run these commands from the repository folder. Install the notebook environment.

```bash
uv sync --locked --group analysis
```

Start JupyterLab with the repository as its file-browser root and leave this terminal open. Run this from `DSA4262-Big-Study-Group`, not its `notebooks` subfolder.

```bash
uv run --locked --group analysis jupyter lab --ServerApp.root_dir="$PWD" --no-browser --ip=127.0.0.1 --port=8888 --ServerApp.port_retries=0
```

Jupyter prints a URL beginning with `http://127.0.0.1:8888/` or `http://localhost:8888/` and containing `token=`. Keep it available for step 3. For example, it may look like `http://127.0.0.1:8888/lab?token=example-token-from-your-terminal`. This is an illustrative pattern, not a working URL: copy the complete URL printed by your own Jupyter process.

### 2. Open a second terminal on your own computer

**Do not run this step inside RONIN.** Open Terminal on macOS/Linux, or PowerShell on Windows. This connection lets your browser reach Jupyter without exposing it publicly.

Have the same connection details you normally use to SSH into RONIN ready:

- **SSH destination:** the username and machine address joined by `@`, copied from your usual SSH command or RONIN connection instructions. For example, `ubuntu@rna-project-team.nus.cloud` shows the expected pattern; use your own machine address.
- **Private-key path:** if your usual SSH command uses `-i`, enter the path to that same key file on your own computer, without surrounding quotes. Examples are `~/.ssh/rna-project-team.pem` on macOS/Linux or `C:\Users\Alex\.ssh\rna-project-team.pem` on Windows. Use your own filename; `~/` is supported by the commands below. Otherwise press Enter to leave it blank. Enter the path, not the contents of the key.

Run **only the command for your computer's operating system**. It asks for these details, so you do not need to edit the command.

On macOS or Linux, create the connection using the prompted SSH details.

```bash
bash -c 'read -r -p "RONIN SSH destination: " ronin_destination; read -r -p "Full private-key path (Enter if not needed): " ronin_key; case "$ronin_key" in "~/"*) ronin_key="$HOME/${ronin_key#\~/}" ;; esac; key_args=(); if [ -n "$ronin_key" ]; then key_args=(-i "$ronin_key"); fi; ssh "${key_args[@]}" -N -T -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -L 127.0.0.1:8888:127.0.0.1:8888 "$ronin_destination"'
```

On Windows PowerShell, create the connection using the prompted SSH details.

```powershell
$roninDestination = Read-Host "RONIN SSH destination"; $roninKey = Read-Host "Full private-key path (Enter if not needed)"; if ($roninKey.StartsWith("~/") -or $roninKey.StartsWith("~\")) { $roninKey = Join-Path $HOME $roninKey.Substring(2) }; $keyArgs = @(); if ($roninKey) { $keyArgs = @("-i", $roninKey) }; ssh @keyArgs -N -T -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -L 127.0.0.1:8888:127.0.0.1:8888 $roninDestination
```

Authenticate as you normally do for SSH. **A blank terminal with no new prompt is expected after connecting.** Leave it open. If an error appears, resolve it before continuing.

### 3. Open the notebook in your browser

- On your own computer, paste the full Jupyter URL from step 1 into your browser, including `token=` and everything after it. Use the localhost URL, not the RONIN machine's public address.
- The left file browser should show repository folders such as `data`, `docs`, `models`, `notebooks` and `predictions`. Double-click `notebooks`, then `sgnex_exploration.ipynb`. Use the file browser's home/breadcrumb control to return to the repository root.
- Select **Kernel → Restart Kernel and Run All Cells**. Keep both terminal windows open while it runs.
- The notebook processes files in chunks and does not need `--n-jobs` or `--threads`.

### If something goes wrong

- **Only the notebook appears in the file browser:** save your work, stop the old Jupyter process with Ctrl+C, return to the repository folder and run the updated launch command in step 1. Open the new URL printed by that process.

- **Browser cannot connect:** check that Jupyter is still running in the RONIN terminal and that the second terminal has no SSH error.
- **Permission denied / key file not found:** use the same username, address and full key path as your working SSH connection. The key must be accessible on your own computer.
- **Address already in use:** port 8888 is occupied on the computer reporting the error. If it belongs to an earlier notebook connection you started, close that connection with Ctrl+C, then retry. Do not stop unrelated processes.
- **Jupyter asks for a token:** copy the value after `token=` from the RONIN terminal, or reopen the complete URL.
- **Finished:** save the notebook, stop Jupyter with Ctrl+C in the RONIN terminal and follow its shutdown prompt, then press Ctrl+C in the second terminal to close the connection.

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
