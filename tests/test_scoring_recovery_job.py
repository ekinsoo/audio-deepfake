import importlib.util
from pathlib import Path
import sys
import unittest
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import scoring_recovery_job as recovery


class RecoveryJobTests(unittest.TestCase):
    def test_completion_rejects_wrong_scope_before_accepting_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "provenance"):
                recovery.validate_completion({"status": "ok", "scope": "pilot"}, Path(directory))

    def test_completion_rejects_changed_score_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "scores.csv").write_text("changed")
            report = {"status": "ok", "scope": "full-official-train-dev-adaptation-scoring-recovery",
                      "configuration": {"data": {"eval_enabled": False}},
                      "model_sha256": "701fc3d8107cc8b2696b5552dd9e1b4751e44d2d197cfd5d1dfe7e4814add101",
                      "output_sha256": {"scores.csv": "incorrect"}}
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                recovery.validate_completion(report, path)

    def test_pipeline_only_scores_exact_remaining_range(self):
        stages = list(recovery.stages())
        self.assertEqual(stages[0], ("preflight", ["--mode", "preflight"]))
        self.assertEqual(stages[-1], ("finalize", ["--mode", "finalize"]))
        ranges = []
        for _, args in stages[1:-1]:
            self.assertEqual(args[:2], ["--mode", "chunk"])
            start, stop = int(args[3]), int(args[5])
            self.assertLessEqual(stop - start, 1000)
            ranges.extend(range(start, stop))
        self.assertEqual(ranges, list(range(22815, 24844)))
        for name, args in stages:
            command = recovery.worker_command(args)
            self.assertIn("--worker", command)
            self.assertNotIn("scripts/run_full_baseline.py", command)
            self.assertNotIn("scripts/prepare_full_data.py", command)


if __name__ == "__main__":
    unittest.main()
