"""Retrain saved parameter selections, without Optuna or the original studies."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
MODELS = ('extra_trees', 'random_forest', 'gradient_boosting',
          'logistic_regression', 'rbf_svm', 'mil')


def training_command(args, model, output):
    selection = json.loads((ROOT / 'configs/best/selection.json').read_text())
    command = [sys.executable, str(ROOT / 'train.py'), '--data-dir', str(args.data_dir),
               '--cache-dir', str(args.cache_dir), '--output-dir', str(output),
               '--model', model, '--model-params', str(
                   ROOT / f'configs/best/{model}.json'),
               '--train-datasets', '0', '1', '--seed', str(selection['seed']),
               '--n-jobs', str(args.n_jobs), '--threads', str(args.threads),
               '--device', args.device, '--selection-metric', 'pr_auc']
    if args.mode == 'full':
        command += ['--dataset-splits', str(ROOT / 'configs/final_splits.json'),
                    '--val-datasets', '--test-datasets', '2']
    else:
        command += ['--split-ratios', '0.7', '0.15', '0.15',
                    '--val-datasets', '0', '1', '--test-datasets', '0', '1', '2']
    if model == 'mil':
        command += ['--epochs', str(selection[f'mil_{args.mode}_epochs'])]
    return command


def evaluation_rows(model, mode, metrics):
    rows = []
    for dataset in ('0', '1', '2'):
        split = 'train' if mode == 'full' and dataset != '2' else 'test'
        metric = metrics[split]['datasets'][dataset]
        rows.append(dict(model=model, dataset=dataset,
                         evaluation='training_set' if split == 'train' else 'held_out_test',
                         sites=metric['sites'], roc_auc=metric['roc_auc'], pr_auc=metric['pr_auc']))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('split', 'full'), required=True)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--cache-dir', type=Path, default=Path('runs/cache'))
    parser.add_argument('--output-dir', type=Path,
                        default=Path('runs/retrained'))
    parser.add_argument('--models', nargs='+',
                        choices=MODELS, default=list(MODELS))
    parser.add_argument('--n-jobs', type=int, default=8)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--device', choices=('auto',
                        'cpu', 'cuda'), default='auto')
    args = parser.parse_args()
    if args.n_jobs == 0 or args.threads < 1:
        parser.error('n-jobs must be nonzero and threads must be positive')
    folder = args.output_dir / args.mode / \
        datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    folder.mkdir(parents=True)
    rows = []
    for model in dict.fromkeys(args.models):
        output = folder / model
        command = training_command(args, model, output)
        (folder /
         f'{model}_command.json').write_text(json.dumps(command, indent=2) + '\n')
        subprocess.run(command, check=True)
        runs = list(output.glob('*/metrics.json'))
        if len(runs) != 1:
            raise RuntimeError(f'Expected one metrics file in {output}')
        rows.extend(evaluation_rows(model, args.mode,
                    json.loads(runs[0].read_text())))
        with (folder / 'evaluation.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        for row in rows[-3:]:
            print(row, flush=True)
    print(
        f'Saved per-dataset ROC AUC and PR AUC: {folder / "evaluation.csv"}', flush=True)


if __name__ == '__main__':
    main()
