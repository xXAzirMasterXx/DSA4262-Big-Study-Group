import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import sgnex


class SGNExTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.content = b'{"tx":{"1":{"ACGTACG":[[1,2,3,4,5,6,7,8,9]]}}}\n'
        sample = "SGNex_A549_directRNA_replicate5_run1"
        self.row = {"sample_id": sample, "cell_line": "A549", "replicate": 5, "run": 1,
                    "url": "https://sg-nex-data.s3.amazonaws.com/data/processed_data/m6Anet/" + sample + "/data.json",
                    "bytes": len(self.content), "etag": hashlib.md5(self.content).hexdigest()}

    def test_selection_validates_names(self):
        manifest = {"version": 1, "samples": [self.row]}
        self.assertEqual(sgnex.select_samples(manifest, ["A549"]), [self.row])
        with self.assertRaisesRegex(ValueError, "Unknown"):
            sgnex.select_samples(manifest, ["wrong"])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            sgnex.select_samples(
                {"version": 1, "samples": [self.row, self.row]})

    def test_download_resume_receipt_and_tampering(self):
        folder = self.root / self.row["sample_id"]
        folder.mkdir()
        (folder / "data.json.part").write_bytes(self.content[:10])
        sgnex.write_json(folder / "partial_source.json", self.row)

        def fetch(command, check):
            self.assertIn("--continue-at", command)
            self.assertIn(f'If-Match: "{self.row["etag"]}"', command)
            Path(command[command.index("--output") + 1]
                 ).write_bytes(self.content)

        with patch("sgnex.subprocess.run", side_effect=fetch) as request:
            target, checksum = sgnex.download(self.row, self.root)
            self.assertEqual(checksum, sgnex.digest(target))
            sgnex.download(self.row, self.root)
            self.assertEqual(request.call_count, 1)
            target.write_bytes(b"x" * len(self.content))
            with self.assertRaisesRegex(ValueError, "verification"):
                sgnex.download(self.row, self.root)

    def test_adopts_verified_manual_download(self):
        folder = self.root / self.row["sample_id"]
        folder.mkdir()
        (folder / "data.json").write_bytes(self.content)
        with patch("sgnex.subprocess.run") as request:
            sgnex.download(self.row, self.root)
            request.assert_not_called()
        self.assertTrue((folder / "download.json").exists())

    def test_resume_contract_integrity_and_summary(self):
        data = self.root / "data"
        output = self.root / "output"
        output.mkdir()
        source = self.root / "input.json"
        source.write_bytes(self.content)
        args = SimpleNamespace(data_dir=data, output_dir=output, model_dir=self.root / "model",
                               n_jobs=8, threads=8, batch_size=256, seed=42, eval_iterations=5)

        def infer(command, check):
            path = Path(command[command.index("--output") + 1])
            path.write_text("dataset,score\nexample,0.5\n")
            sgnex.write_json(path.with_suffix(".csv.json"), {
                             "sites": 1, "model": "mil", "model_sha256": "test"})

        with patch("sgnex.download", return_value=(source, sgnex.digest(source))) as fetch, \
                patch("sgnex.subprocess.run", side_effect=infer) as predict:
            sgnex.run_sample(self.row, args, {"model": "first"})
            sgnex.run_sample(self.row, args, {"model": "first"})
            self.assertEqual(predict.call_count, 1)
            self.assertEqual(fetch.call_count, 1)
            with self.assertRaisesRegex(ValueError, "settings changed"):
                sgnex.run_sample(self.row, args, {"model": "second"})
            sgnex.summary(output, [self.row])
            self.assertIn("A549,5,1,1,mil",
                          (output / "samples.csv").read_text())
            (output / self.row["sample_id"] /
             "predictions.csv").write_text("changed")
            with self.assertRaisesRegex(ValueError, "integrity"):
                sgnex.run_sample(self.row, args, {"model": "first"})

    def test_failed_prediction_is_not_completed(self):
        args = SimpleNamespace(output_dir=self.root, data_dir=self.root, model_dir=self.root,
                               n_jobs=8, threads=8, batch_size=256, seed=42, eval_iterations=5)
        with patch("sgnex.download", return_value=(self.root / "input", "test")), \
                patch("sgnex.subprocess.run", side_effect=subprocess.CalledProcessError(1, "predict")):
            with self.assertRaises(subprocess.CalledProcessError):
                sgnex.run_sample(self.row, args, {})
        self.assertFalse((self.root / self.row["sample_id"]).exists())
        self.assertFalse(list(self.root.glob(".sgnex-*")))


if __name__ == "__main__":
    unittest.main()
