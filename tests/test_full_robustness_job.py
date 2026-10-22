"""Budget and bootstrap checks without launching scoring or reading audio."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import full_robustness_job as job


class RobustnessJobTests(unittest.TestCase):
    def test_seed_budget_accumulates_chunks_and_includes_startup(self):
        stages = [dict(name="preflight", elapsed_seconds=120),
                  dict(name="seed42_00000_02000", elapsed_seconds=300),
                  dict(name="seed42_02000_04000", elapsed_seconds=400),
                  dict(name="seed43_00000_02000", elapsed_seconds=800)]
        self.assertEqual(job.remaining_budget(stages, 42), 1100)
        self.assertEqual(job.remaining_budget(stages, 43), 1000)
        self.assertEqual(job.remaining_budget(stages, 42, smoke_seconds=20), 1080)
        stages.append(dict(name="seed42_04000_06000", elapsed_seconds=1100))
        with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
            job.remaining_budget(stages, 42)

    def test_worker_uses_actual_interpreter_and_explicit_allowlist(self):
        with patch.object(job.supervisor, "base_python", return_value="actual-python"):
            command = job.worker_command("scripts/full_robustness_worker.py", "--mode", "smoke")
        self.assertEqual(command[:3], ["actual-python", "-S", "-u"])
        self.assertEqual(command[-4:], ["--worker-script", "scripts/full_robustness_worker.py", "--mode", "smoke"])
        with self.assertRaises(ValueError):
            job.worker_command("scripts/run_full_baseline.py")

    def test_any_source_or_protocol_change_stops(self):
        with patch.object(job, "source_hashes", return_value={"a": "one"}), patch.object(
                job, "digest", return_value=job.CONTRACT_SHA):
            job.check_pins({"a": "one"})
            with self.assertRaises(RuntimeError):
                job.check_pins({"a": "two"})
        with patch.object(job, "digest", return_value="wrong"):
            with self.assertRaises(RuntimeError):
                job.check_pins({})

    def test_preflight_failure_stops_before_smoke_and_preserves_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "outputs/jobs/full-robustness-test"
            directory.mkdir(parents=True)
            output = root / "outputs/robustness-full-v1"
            with patch.object(job, "ROOT", root), patch.object(job, "OUTPUT", output), patch.object(
                    job, "source_hashes", return_value={}), patch.object(job, "check_pins"), patch.object(
                    job.supervisor, "wait_for_resources"), patch.object(job.supervisor, "run_stage",
                    side_effect=RuntimeError("preflight_failed")) as launch:
                code = job.main(["--job-dir", str(directory)])
            self.assertEqual(code, 1)
            self.assertEqual(launch.call_count, 1)
            self.assertEqual(launch.call_args.kwargs["name"], "preflight")
            status = json.loads((directory / "status.json").read_text())
            self.assertEqual(status["status"], "stopped")
            self.assertIn("preflight_failed", status["reason"])
            self.assertTrue((output / "execution.json").exists())
            self.assertFalse((directory.parent / "full-robustness.lock").exists())
