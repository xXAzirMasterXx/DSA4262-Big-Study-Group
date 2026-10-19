# Optuna tuning

`tune.py` uses the existing split, feature, model and metric functions. Each study selects a single model configuration by pooled validation trapezoidal PR AUC. The first trial fits the current baseline configuration. All other trials use Optuna's seeded TPE sampler.

## Launch and resume

```bash
uv sync --locked --group tuning
# One model, one training source
uv run --locked --group tuning python tune.py --data-dir data --models extra_trees --variants data0
# All six models on data0, data1 and combined, sequentially
uv run --locked --group tuning python tune.py --data-dir data
# A chosen subset
uv run --locked --group tuning python tune.py --data-dir data --models random_forest gradient_boosting --variants data0 data1 data01
```

Default storage is `runs/tuning/<model>_data<IDs>/`. For combined training the suffix is `data0-1`. Each directory contains a separate SQLite database and allows only one active writer. Stop with Ctrl+C and repeat the same command to resume completed work. A model fit interrupted before completion restarts as a new trial; it does not resume halfway through training.

`--trials` is the **total attempted trial budget per study**, including baseline, complete, pruned and failed trials. Repeating a completed budget performs no new trials. Increase the budget to continue searching:

```bash
uv run --locked --group tuning python tune.py --data-dir data --models extra_trees --variants data0 --trials 50
```

`--timeout 3600` sets a time budget per study per invocation. It stops launching additional trials, but an ongoing fit may exceed it. Final test evaluation is also outside this budget. SVM trials use the exact kernel and training-only calibration without subsampling, and can be slow.

Unexpected fitting errors stop the invocation and are recorded. A process terminated abruptly can leave a RUNNING trial; the next exclusive writer marks it failed. Sampler state is saved after trials. Completed work is retained, but a hard process termination can prevent exact continuation of the sampler's random sequence. A missing sampler checkpoint is rejected rather than silently resetting it.

## Search settings

Edit a copy of `configs/tuning.json` and pass it with `--config`. Each model has a trial budget, baseline parameters and search distributions. Supported distributions are `int`, `float`, `categorical`, and `float_or` (a numeric range or special values such as zero or `scale`). The baseline's resolved parameters are stored in trial attributes, including when it is outside the sampled range.

| Model                       | Default trials per source | Parameters searched                                               |
| --------------------------- | ------------------------: | ----------------------------------------------------------------- |
| Extra Trees                 |                        30 | Tree count, depth, leaf size, feature fraction                    |
| Random forest               |                        30 | Tree count, depth, leaf size, feature fraction                    |
| Histogram gradient boosting |                        30 | Learning rate, iteration count, leaf count, leaf size, L2 penalty |
| Logistic regression         |                        30 | L2 regularization strength through `C`                            |
| RBF SVM                     |                        10 | `C`, `gamma`                                                      |
| MIL                         |                        15 | Learning rate, weight decay, hidden/embedding dimensions, dropout |

MIL keeps bag size, batch size and evaluation bags fixed across trials. Its baseline retains the existing ten-epoch limit; searched configurations have a thirty-epoch limit. Validation chooses the best checkpoint within each trial. A median pruner can end weak nonbaseline trials after warm-up. `--startup-trials` and `--prune-warmup-epochs` both default to 5. Classical fits have no mid-fit pruning.

Balancing is fixed across trials and enabled by default; `--no-class-weight` disables it for an entire study. Feature ablations, extra calibration methods and stacking are outside this runner.

## Fixed data and reproducibility

- Default variants preserve the existing independent stratified splits and source selections. `--variants data01` uses data0 and data1 training rows. Validation defaults to data0/data1; tests default to data0/data1/data2.
- `--train-datasets` replaces the variants with a custom training-source combination. `--val-datasets`, `--test-datasets` and `--split-ratios` are available. Data2 remains test-only. Tuning requires both classes in nonempty train and validation partitions; full-data final training remains available through `train.py`.
- `--split-seed`, `--model-seed`, `--sampler-seed` and `--evaluation-seed` each default to 42. Model/search seed changes do not alter which rows belong to each split.
- Resume checks cover source signatures, cached metadata and split hashes, source-code hashes, search configuration and library versions. Changed inputs or settings require a new `--output-dir`; trial-count and time budgets may change on resume.
- `--n-jobs` controls jobs inside forests and SVM calibration; `--threads` controls numerical/PyTorch threads. Both default to 1 to avoid overlapping worker pools. Studies themselves execute sequentially. Different thread settings require a fresh study because they are recorded in its contract.
- These are finite-budget, single-model-seed searches. Budget and runtime differences are logged and should accompany comparisons; a tuned validation winner is not an unbiased estimate of future performance.

## Test evaluation and artifacts

The objective receives only train and validation partitions. Data2 is not prepared until the selected winner is evaluated. `--no-final-eval` skips test evaluation entirely. Repeat the command with the same total budget and omit that flag to evaluate the saved winner later.

- `study.db`, `sampler.joblib`, `study_info.json`: study history, sampler state and compatibility contract.
- `trials.csv`, `budget.json`: trial records and requested budget.
- `best_parameters.json`, `best_trial.json`: resolved winner settings and validation metrics.
- `trials/trial_NNNNN/`: parameter/validation files, plus MIL epoch history where applicable. Only the current winning fitted model is retained to limit disk use.
- `final/trial_NNNNN/`: selected estimator/checkpoint, split assignments, per-dataset predictions, metrics and run metadata. MIL also exports validation learning curves. Final metrics include training metrics for the selected checkpoint.
- `internal_test_metrics.json` in that final directory reports data0/data1 test pooling separately from data2, while standard `metrics.json` retains all selected test datasets.

A saved final evaluation is reused when the winner and data fingerprints match. Extending a study may select a different winner and produce a new final directory. Use `--no-final-eval` while extending searches to avoid repeated test inspection.

## Local dashboard and checks

```bash
uv run --locked --group tuning optuna-dashboard sqlite:///runs/tuning/extra_trees_data0/study.db
uv run --locked --group tuning python -m unittest discover -s tests
```

The dashboard reads a chosen study database locally. Tests use temporary synthetic datasets; they cover model baselines, split identity, training-only trial inputs, parameter sampling, pruning, resumption, incompatible-study rejection and winner exports.
