# Dataset inputs

The three course datasets and metadata files are included through Git LFS. Follow the root README to download them already extracted. Custom datasets under other data folders are ignored by Git.

## Course layout

| Directory | Read features | Labels |
| --- | --- | --- |
| `data/data0` | `dataset0.json` | `data.info.labelled` |
| `data/data1` | `dataset1.json` | `data.info` |
| `data/data2` | `dataset2.json` | `data.info` |

## Feature JSON

Both training and inference expect transcript IDs mapping to integer site positions, each with one seven-base sequence context and a list of reads. Each read has nine finite numeric channels in exactly the same order and units used for training. Example structure with illustrative values:

```json
{
  "transcript_example": {
    "100": {
      "ACGTACG": [
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        [2, 3, 4, 5, 6, 7, 8, 9, 10]
      ]
    }
  }
}
```

- A single JSON object or consecutive top-level transcript objects is supported. Use uncompressed JSON.
- Each site needs at least one read. Contexts must contain seven A/C/G/T bases. Inference accepts lowercase and converts to uppercase.
- Each transcript/position key must be unique within a dataset. Matching keys across separate datasets remain independent.
- Raw FASTQ, BAM or signal files need upstream feature extraction first. This repository does not implement that extraction. A matching array shape alone does not establish compatibility with a model's training features.

## Your own labeled training dataset

Place your feature JSON at `data/data3/dataset3.json` and label CSV at `data/data3/data.info`. Use `train.py` with `--train-datasets 3 --val-datasets 3 --test-datasets 3 --n-jobs 8 --threads 8` after preparing these files. See the training guide for complete course-data commands.



`data.info` is CSV with `transcript_id,transcript_position,label` columns. Labels must be 0 or 1 (data2 alone allows nonnegative values, binarized with `> 0`). Optional `n_reads` must match the feature JSON. Each labeled site must have matching reads. Supply enough rows from both classes for stratified splits; tiny datasets may not split successfully. Data0 uses the filename `data.info.labelled` instead.

Data3/data4 and later datasets follow the same naming convention. Explicit validation/test selectors avoid requiring the default course datasets. See [training details](training.md) for custom ratios and full-data training.

## Your own inference dataset

Pass your feature JSON path through `predict.py --input`, choose a bundled folder with `--model-dir`, and choose an output CSV with `--output`. Include `--n-jobs 8 --threads 8` on the recommended RONIN machine. See the README for complete runnable commands using included data.



This requires an exported model folder. No metadata CSV or labels are needed. You can also pass a course feature JSON directly. Use `--dataset-name` to set the output dataset identifier; otherwise it is the input filename stem. Scores are probabilities, not thresholded labels.

## Custom inference labels

To calculate ROC AUC and PR AUC, supply a CSV with a header and these required columns:

```csv
transcript_id,transcript_position,label
transcript_example,100,0
transcript_example,101,1
```

- `transcript_id` must match the feature JSON transcript ID exactly.
- `transcript_position` must be an integer matching the JSON site position. Each transcript/position pair must be unique in the CSV.
- `label` should be `0` (unmodified) or `1` (modified). Inference also accepts finite nonnegative numeric labels and converts every value greater than zero to `1`, as for data2. Missing, negative and nonnumeric labels are rejected.
- Row order does not matter: labels are matched by transcript ID and position. Extra columns are ignored.
- You can label a subset of sites. Metrics use only matching labeled sites; coverage and unmatched counts are recorded in the companion JSON. No matches cause an error. Both binary classes are needed for AUCs; otherwise they are saved as null with an explanation.



A file named `labels.csv` requires `--labels`. For automatic discovery, name it `data.info.labelled` or `data.info` and place it beside the feature JSON. Use `--no-labels` to skip automatic evaluation.
