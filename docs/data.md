# Dataset inputs

Run commands from the repository root. Data files are not included in Git. Obtain the course dataset through the course's distribution channel; this repository does not download it.

## Course datasets

Extract compressed archives first. For a `.json.gz` file, `gzip -dk /path/to/dataset0.json.gz` creates the JSON while keeping the archive. Copy the extracted files, replacing `/path/to/course` with your location:

```bash
mkdir -p data/data0 data/data1 data/data2
cp /path/to/course/dataset0.json data/data0/
cp /path/to/course/data.info.labelled data/data0/
cp /path/to/course/dataset1.json data/data1/
cp /path/to/course/data1/data.info data/data1/
cp /path/to/course/dataset2.json data/data2/
cp /path/to/course/data2/data.info data/data2/
```

The source layout depends on your downloaded archive; the destination layout in the README is required for training.

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

```bash
mkdir -p data/data3
cp /path/to/features.json data/data3/dataset3.json
cp /path/to/labels.csv data/data3/data.info
uv run --locked python train.py --data-dir data --model extra_trees --train-datasets 3 --val-datasets 3 --test-datasets 3 --output-dir runs/et_data3 --cache-dir runs/cache
```

`data.info` is CSV with `transcript_id,transcript_position,label` columns. Labels must be 0 or 1 (data2 alone allows nonnegative values, binarized with `> 0`). Optional `n_reads` must match the feature JSON. Each labeled site must have matching reads. Supply enough rows from both classes for stratified splits; tiny datasets may not split successfully. Data0 uses the filename `data.info.labelled` instead.

Data3/data4 and later datasets follow the same naming convention. Explicit validation/test selectors avoid requiring the default course datasets. See [training details](training.md) for custom ratios and full-data training.

## Your own inference dataset

```bash
mkdir -p data/new
cp /path/to/features.json data/new/dataset.json
uv run --locked python predict.py --model-dir models/extra_trees_data01 --input data/new/dataset.json --output predictions/new.csv
```

This requires an exported model folder. No metadata CSV or labels are needed. You can also pass a course feature JSON directly. Use `--dataset-name` to set the output dataset identifier; otherwise it is the input filename stem. Scores are probabilities, not thresholded labels.
