# Model bundles

Five models are available in both `split/` and `full/`: Extra Trees, Random Forest, histogram Gradient Boosting, Logistic Regression and MIL. All train on data0/data1/data2 with the parameters in `configs/best/`.

- `split/`: 70% training, 15% validation, 15% held-out test per dataset.
- `full/`: 100% training; reported evaluation on training rows is biased.
- Default inference uses `models/full/extra_trees`. See the [quickstart](../README.md), [metrics](../results/) and [inference details](../docs/inference.md).

Each folder contains `model.json` (schema, checksum and inference defaults), `provenance.json` (training settings), and `model.joblib` or `model.pt` (weights and preprocessing). Binary files use Git LFS.

Download weights after cloning.

```bash
git lfs pull
```

Use the locked environment to match the saved scikit-learn version. MIL bundles include architecture and normalization. Only load trusted model files.

For new weights, follow [selected-parameter retraining](../docs/training.md#retrain-selected-parameters). `export_model.py` can also package a completed run or selected study without training. Its `--source` must identify an existing checkpoint folder and `--output` must be a new folder. Exported weights are covered by the existing LFS rules.
