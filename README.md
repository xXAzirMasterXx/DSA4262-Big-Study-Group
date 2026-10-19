# m6A site classifiers

Compare a m6Anet-inspired MIL model with Extra Trees, random forest, histogram gradient boosting, logistic regression, and exact RBF SVM. Each predicts a site-level modification probability. Data2 labels are binarized with `label > 0`.

## Run

```bash
uv sync --locked
# MIL is the default. Repeat with --train-datasets 1 or --train-datasets 0 1.
uv run python train.py --data-dir data --train-datasets 0 --output-dir runs/mil_data0 --cache-dir runs/cache
# Choose extra_trees, random_forest, gradient_boosting, logistic_regression, or rbf_svm.
uv run python train.py --data-dir data --model extra_trees --train-datasets 0 --output-dir runs/et_data0 --cache-dir runs/cache
```

Files follow `data/dataN/datasetN.json` and `data/dataN/data.info`; data0 uses `data.info.labelled`. Future data3/data4 folders follow the same convention and require binary labels.

## Shared splits and evaluation

- Independent, stratified 70/15/15 row splits per dataset. Transcript and gene IDs do not constrain splitting; matching sites across datasets remain independent.
- `--train-datasets` defaults to `0 1`; `--val-datasets` to `0 1`; `--test-datasets` to `0 1 2`. Changing the training selection keeps validation/test rows fixed. Preserve the seed (default 42), inputs, row order, and ratios when comparing models.
- Data2 always uses 100% test rows and cannot train or select models. Test evaluation follows model selection.
- All models report BCE, trapezoidal PR AUC, and ROC AUC, pooled and per dataset. Pooled means the datasets selected for that split, so train pooled equals train data0 when training only on data0. Single-class AUCs are null.
- Change default proportions with `--split-ratios`; override individual datasets with `--dataset-splits splits.json`. Integer counts require rounding; small classes may not support stratification. `--split-only` exports assignments without loading features.

For final training, put `{"0": [1, 0, 0], "1": [1, 0, 0]}` in `splits.json`:

```bash
uv run python train.py --data-dir data --model logistic_regression --dataset-splits splits.json --train-datasets 0 1 --val-datasets --test-datasets 2
```

Without validation, MIL saves the final epoch; classical models fit one fixed parameter configuration.

## Classical models

All use the same **65 features**, computed once over all reads per site and cached:

- 36 numeric summaries: mean, population standard deviation (`ddof=0`), 25th and 75th percentiles (linear interpolation) for each of nine channels.
- 28 position-specific A/C/G/T one-hot features for the 7-mer. Other bases raise an error.
- `log1p(n_reads)` using the natural logarithm.

Logistic regression and SVM standardize numeric features using training rows only; one-hot columns pass through. Trees use unscaled features. Balanced class weights default on; `--no-class-weight` disables them (alias of `--no-balanced-sampler`). SVM uses sigmoid calibration from training-only stratified folds (`--calibration-folds 3`), with preprocessing fitted inside each fold and a final base model fitted on all training rows. Calibration uses natural class frequencies.

Defaults: 300 trees for forests, 100 boosting iterations with internal early stopping disabled, L2 logistic regression, and exact RBF SVM. `--n-jobs` controls forest/calibration parallelism. Exact SVM can be slow on these datasets; it never automatically subsamples or substitutes an approximation.

Use `--model-params params.json` for estimator parameters, such as `{"n_estimators": 500, "max_depth": 20}` for a forest. A list of objects runs candidate configurations and selects by pooled validation PR AUC (`--selection-metric loss` or `roc_auc` also supported). A single object fits one configuration. Balancing, seeds, jobs, SVM kernel/probability handling, and internal early stopping are controlled by the runner.

These summaries use all reads, whereas MIL uses sampled read bags. The site splits match, but the representations differ. Feature ablations and ensemble fitting are future experiments; ensemble weights must be learned from held-out or out-of-fold predictions, never test results.

## Hyperparameter tuning

Optuna tunes each model and training source independently, using the same fixed validation rows. Search ranges and default budgets are in `configs/tuning.json`.

```bash
uv sync --locked --group tuning
# Tune one model on combined training; repeat the same command to resume.
uv run --locked --group tuning python tune.py --data-dir data --models extra_trees --variants data01
# Tune all models on all three training variants, sequentially.
uv run --locked --group tuning python tune.py --data-dir data
```

The baseline is trial zero. Budgets default to 30 trials for forests, boosting and logistic regression, 10 for SVM, and 15 for MIL. `--trials` sets the total budget per study; `--timeout` stops new trials after a time budget but cannot interrupt an ongoing fit. MIL supports epoch pruning; exact SVM retains training-only calibration.

Studies save locally under `runs/tuning/`, including a database, trial CSV, best settings and winning model. Tests run only after selecting a winner; use `--no-final-eval` to defer them. Separate split/model/search seeds preserve comparison rows. See [tuning details](docs/tuning.md) for settings, resume behavior, artifacts and the local dashboard.

## MIL and outputs

MIL defaults: 10 epochs, batch size 256, 20 reads per bag, and 5 evaluation iterations. Training-only normalization, balanced sampling, and noisy-OR pooling are used. The checkpoint with the best pooled validation PR AUC is selected.

Each timestamped run saves split assignments, `run_info.json`, `metrics.json`/`.csv`, and per-dataset prediction CSVs. Predictions include `dataset`, source `row_id`, transcript/position, binary label, split, and `score`. Use `(dataset, row_id)` to align identical source datasets for future ensembles. Assignment files include unused partitions; dataset selectors determine actual usage.

- **MIL:** `best_model.pt`, epoch history, and training curves.
- **Classical:** `best_model.joblib` (estimator including preprocessing/calibration), `feature_names.json`, and `search_history.json`. Logged settings include resolved parameters and the scikit-learn version. Saved classical estimators accept matrices in the documented feature order and expose `predict_proba`.

Run all checks with `uv run --locked --group tuning python -m unittest discover -s tests`.
