# Model folders

The six `*_data01` folders contain the completed baseline models trained on the combined data0/data1 training partitions. Each manifest identifies its original run; provenance records dataset selections and splits. Preprocessing/calibration and MIL normalization are included.

Use `predict.py --model-dir models/extra_trees_data01` (or another model folder) as shown in the root README. Fetch binaries with `git lfs pull` after cloning.

## Export a selected checkpoint

Run from the repository root, after selecting a completed run or study. For example:

Run this command from the repository root.

```bash
uv run --locked python export_model.py --source runs/tuning/extra_trees_data0-1 --output models/extra_trees --description "Selected Extra Trees model"
```

Replace the source with the actual study directory. A study folder containing `best_trial.json` resolves to its selected trial. A timestamped training run, trial folder, or final evaluation folder containing `best_model.joblib` or `best_model.pt` also works. Export does not fit or evaluate the model. The destination must not already exist.

Each folder contains:

- `model.json`: model type, input schema, checksum, environment versions and inference defaults.
- `model.joblib` or `model.pt`: fitted estimator with preprocessing/calibration, or MIL weights with normalization and architecture settings.
- `provenance.json`: available training settings and validation information.

Classical artifacts require the saved scikit-learn version; retain the matching repository revision and lockfile. MIL bag sampling uses defaults from the run when available; `--eval-iterations` and `--seed` can set explicit export defaults.

## Publish selected weights

Install [Git LFS](https://git-lfs.com/) first. `.gitattributes` tracks model binaries in these folders through LFS. Large forests exceed [GitHub's regular file size limit](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).

Run this command from the repository root.

```bash
git lfs install
```

Run this command from the repository root.

```bash
git add .gitattributes models/
```

Run this command from the repository root.

```bash
git lfs status
```

Run this command from the repository root.

```bash
git commit -m "feat: add selected pretrained models"
```

Run this command from the repository root.

```bash
git push
```

Before publishing, check the staged files and validate inference with each selected model. Retain a short description of its training datasets and selection criterion. LFS storage and bandwidth must be available on the hosting account. Newcomers run `git lfs pull` after cloning, then pass the chosen folder to `predict.py --model-dir`.
