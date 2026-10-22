"""Queue safety gates without downloads, real workers or model fitting."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

SPEC = importlib.util.spec_from_file_location("localization_pilot_job",
    Path(__file__).resolve().parents[1] / "scripts/localization_pilot_job.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class QueueGateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "data/partialspoof").mkdir(parents=True)
        self.job = self.root / "job"
        self.job.mkdir()

    def receipt(self, split, digest="selection"):
        path = self.root / f"data/partialspoof/audio_{split}_complete.json"
        path.write_text(json.dumps({"status": "complete", "selection_sha256": digest}))

    def test_both_matching_receipts_required(self):
        self.assertFalse(module.audio_ready(self.root, "selection"))
        self.receipt("train")
        self.assertFalse(module.audio_ready(self.root, "selection"))
        self.receipt("dev")
        self.assertTrue(module.audio_ready(self.root, "selection"))

    def test_wrong_selection_stops(self):
        self.receipt("train", "wrong")
        with self.assertRaisesRegex(RuntimeError, "selection"):
            module.audio_ready(self.root, "selection")

    def test_stopped_audio_never_starts_training(self):
        (self.job / "status.json").write_text('{"status":"stopped"}')
        status = {}
        with self.assertRaisesRegex(RuntimeError, "ended"):
            module.wait_audio(root=self.root, job_dir=self.job, status=status,
                audio_job_dir=self.job, selection_sha256="selection", sleep=lambda _: None)
        self.assertFalse(status["training_started"])

    def test_queue_timeout_is_bounded(self):
        (self.job / "status.json").write_text('{"status":"running"}')
        ticks = iter([0, 1800])
        with self.assertRaisesRegex(RuntimeError, "expired"):
            module.wait_audio(root=self.root, job_dir=self.job, status={},
                audio_job_dir=self.job, selection_sha256="selection", clock=lambda: next(ticks))

    def test_resource_guards(self):
        module.check_resources({"system_available_ram_bytes": 2 * module.GIB,
                                "child_peak_rss_bytes": module.GIB}, 20 * module.GIB)
        for available, peak, disk in [(module.GIB, 0, 20 * module.GIB),
                (3 * module.GIB, module.GIB + 1, 20 * module.GIB),
                (3 * module.GIB, 0, 20 * module.GIB - 1)]:
            with self.assertRaises(RuntimeError):
                module.check_resources({"system_available_ram_bytes": available,
                    "child_peak_rss_bytes": peak}, disk)

    def test_changed_frozen_input_stops(self):
        file = self.root / "input.json"
        file.write_text("original")
        pins = {"input.json": module.sha(file)}
        module.verify_sources(pins, self.root)
        file.write_text("changed")
        with self.assertRaisesRegex(RuntimeError, "changed"):
            module.verify_sources(pins, self.root)

    def test_default_plan_does_not_start_workers(self):
        with patch.object(module.sys, "argv", ["job"]), patch.object(module.subprocess, "Popen") as popen:
            module.main()
        popen.assert_not_called()

    def test_worker_timeout_stops_owned_process(self):
        child = SimpleNamespace(pid=123, poll=lambda: None)
        status = {"stages": []}
        with patch.object(module, "windows_memory", return_value={"system_available_ram_bytes": 3 * module.GIB}), \
                patch.object(module.shutil, "disk_usage", return_value=SimpleNamespace(free=30 * module.GIB)), \
                patch.object(module.subprocess, "Popen", return_value=child), \
                patch.object(module, "worker_command", return_value=["test-worker"]), \
                patch.object(module.time, "monotonic", side_effect=[0, 301]), \
                patch.object(module, "stop_owned_child") as stop:
            with self.assertRaisesRegex(RuntimeError, "runtime budget"):
                module.run_stage("pilot", [], job_dir=self.job, status=status, budget_seconds=300)
        stop.assert_called_once_with(child)
        self.assertEqual(status["stages"][0]["status"], "stopped")

    def test_failed_alignment_never_launches_pilot(self):
        jobs = self.root / "outputs/jobs"
        jobs.mkdir(parents=True)
        directory = jobs / "pilot"
        args = ["job", "--execute", "--job-dir", str(directory), "--audio-job-dir", str(jobs / "audio")]
        pins = {"docs/validation/partial_selection_v1.json": "selection"}
        with patch.object(module, "ROOT", self.root), patch.object(module.sys, "argv", args), \
                patch.object(module, "freeze_sources", return_value=pins), \
                patch.object(module, "verify_sources"), patch.object(module, "wait_audio"), \
                patch.object(module, "sha", return_value="hash"), \
                patch.object(module, "run_stage", side_effect=RuntimeError("alignment failure")) as stage:
            with self.assertRaisesRegex(RuntimeError, "alignment failure"):
                module.main()
        self.assertEqual(stage.call_count, 1)
        self.assertEqual(stage.call_args.args[0], "alignment")
        final = json.loads((directory / "status.json").read_text())
        self.assertFalse(final["training_started"])
        self.assertEqual(final["status"], "stopped")

    def test_worker_bootstrap_preserves_execute_and_alignment_arguments(self):
        argv = ["job", "--worker", "pilot", "--execute", "--alignment", "audit.json",
                "--output-dir", "pilot-output"]
        seen = []
        with patch.object(module.sys, "argv", argv), patch.object(module, "configure_project_environment"), \
                patch.object(module.runpy, "run_path", side_effect=lambda *a, **k: seen.extend(module.sys.argv)):
            module.main()
        self.assertEqual(seen, [module.WORKERS["pilot"], "--execute", "--alignment", "audit.json",
                                "--output-dir", "pilot-output"])

    def test_zero_exit_is_not_completion_without_fit_results(self):
        directory = self.job / "pilot"
        directory.mkdir()
        (directory / "status.json").write_text('{"phase":"complete","training_started":true}')
        (directory / "results.json").write_text('{"status":"plan"}')
        with self.assertRaisesRegex(RuntimeError, "complete training"):
            module.require_pilot_complete(self.job)
        (directory / "results.json").write_text('{"status":"complete"}')
        self.assertTrue(module.require_pilot_complete(self.job)["training_started"])


if __name__ == "__main__":
    unittest.main()
