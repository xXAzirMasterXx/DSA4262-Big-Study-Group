import unittest
from argparse import Namespace
from pathlib import Path
from retrain import evaluation_rows, training_command


class RetrainTests(unittest.TestCase):
    def args(self, mode):
        return Namespace(mode=mode, data_dir=Path('data'), cache_dir=Path('runs/cache'),
                         n_jobs=8, threads=8, device='cpu')

    def test_full_excludes_training_sources_from_test_selection(self):
        command = training_command(self.args('full'), 'mil', Path('output'))
        index = command.index('--val-datasets')
        self.assertEqual(command[index:index+3],
                         ['--val-datasets', '--test-datasets', '2'])
        self.assertEqual(command[command.index('--epochs')+1], '28')
        self.assertIn('--dataset-splits', command)

    def test_split_uses_validation_and_all_test_sources(self):
        command = training_command(self.args('split'), 'mil', Path('output'))
        index = command.index('--test-datasets')
        self.assertEqual(command[index+1:index+4], ['0', '1', '2'])
        self.assertEqual(command[command.index('--epochs')+1], '30')

    def test_full_summary_does_not_present_training_metrics_as_held_out(self):
        def metric(score):
            return dict(sites=10, roc_auc=score, pr_auc=score)
        metrics = {'train': {'datasets': {'0': metric(.99), '1': metric(.98)}},
                   'test': {'datasets': {'2': metric(.7)}}}
        rows = evaluation_rows('mil', 'full', metrics)
        self.assertEqual([r['evaluation'] for r in rows], [
                         'training_set', 'training_set', 'held_out_test'])
        self.assertEqual([r['pr_auc'] for r in rows], [.99, .98, .7])


if __name__ == '__main__':
    unittest.main()
