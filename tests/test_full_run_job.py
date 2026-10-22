"""Small child processes exercise the detached supervisor without data or fits."""

import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import os
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "full_run_job", Path(__file__).resolve().parents[1] / "scripts/full_run_job.py")
job = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(job)


class FullRunJobTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "nt", "Windows ctypes probe regression")
    def test_repeated_memory_probes_do_not_grow_pointer_cache(self):
        import ctypes
        job.windows_memory(os.getpid())
        before = len(ctypes._pointer_type_cache)
        for _ in range(2000):
            measured = job.windows_memory(os.getpid())
        self.assertEqual(len(ctypes._pointer_type_cache), before)
        self.assertGreater(measured["child_peak_rss_bytes"], 0)

    def resource_wait(self, readings, budget=30, disk=25 * job.GIB):
        now = [0]
        def sleep(seconds):
            now[0] += seconds
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            status = {"stages": []}
            values = iter(readings)
            with patch.object(job.subprocess, "Popen") as child:
                try:
                    job.wait_for_resources(job_dir=path, status=status, phase="startup",
                        resource_probe=lambda: {"system_available_ram_bytes": next(values)},
                        disk_probe=lambda: disk, clock=lambda: now[0], sleep=sleep,
                        budget_seconds=budget)
                    failure = None
                except RuntimeError as error:
                    failure = str(error)
                child.assert_not_called()
            return json.loads((path / "status.json").read_text()), failure

    def test_resource_wait_requires_two_consecutive_readings(self):
        status, failure = self.resource_wait([4, 3, 4, 4][i] * job.GIB for i in range(4))
        self.assertIsNone(failure)
        self.assertEqual(status["status"], "running")
        self.assertIsNone(status["child_pid"])
        wait = status["resource_waits"][0]
        self.assertEqual(wait["elapsed_seconds"], 15)
        self.assertEqual(wait["consecutive_ready_readings"], 2)
        self.assertEqual(wait["status"], "ready")
        self.assertIn("deadline_utc", wait)
        self.assertEqual(status["stages"], [])

    def test_resource_wait_expiry_preserves_observed_resources(self):
        status, failure = self.resource_wait([2 * job.GIB] * 2, budget=10)
        self.assertEqual(failure, "startup_resource_wait_expired")
        self.assertEqual(status["status"], "waiting_resources")  # main sets final stopped
        wait = status["resource_waits"][0]
        self.assertEqual(wait["status"], "stopped")
        self.assertEqual(wait["elapsed_seconds"], 10)
        self.assertEqual(wait["last_resources"]["system_available_ram_bytes"], 2 * job.GIB)

    def test_resource_wait_rejects_insufficient_disk(self):
        status, failure = self.resource_wait([4 * job.GIB], disk=19 * job.GIB)
        self.assertIn("20 GiB free disk", failure)
        self.assertEqual(status["resource_waits"][0]["elapsed_seconds"], 0)

    def run_child(self, code, *, budget=5, memory=None):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            status = {"stages": []}
            probe = lambda pid: memory or {"child_peak_rss_bytes": 1, "system_available_ram_bytes": 4 * job.GIB}
            try:
                executable = job.base_python() if os.name == "nt" else sys.executable
                job.run_stage([executable, "-c", code], root=root, job_dir=root,
                              name="test", budget_seconds=budget, status=status,
                              resource_probe=probe, poll_seconds=0.01)
                failure = None
            except RuntimeError as error:
                failure = str(error)
            saved = json.loads((root / "status.json").read_text())
            logs = ((root / "test.stdout.log").read_bytes(),
                    (root / "test.stderr.log").read_bytes())
            return saved, failure, logs

    def test_success_records_exit_and_logs(self):
        status, failure, logs = self.run_child("print('done')")
        self.assertIsNone(failure)
        self.assertEqual(status["stages"][0]["status"], "complete")
        self.assertEqual(status["stages"][0]["exit_code"], 0)
        self.assertIsNone(status["child_pid"])
        self.assertIn(b"done", logs[0])

    @unittest.skipUnless(os.name == "nt", "Actual Windows venv redirector regression")
    def test_real_worker_pid_and_venv_package_versions(self):
        from importlib.metadata import version
        command = [job.base_python(), "-S", "-u", str(job.ROOT / "scripts/full_run_job.py"), "--worker-probe"]
        with subprocess.Popen(command, cwd=job.ROOT, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True) as child:
            record = json.loads(child.stdout.readline())
            self.assertEqual(record["pid"], child.pid)
            measured = job.windows_memory(child.pid, process_handle=child._handle)
            self.assertGreater(measured["child_peak_rss_bytes"], 0)
            _, errors = child.communicate(timeout=10)
            self.assertEqual(child.returncode, 0, errors)
        self.assertEqual(record["versions"], {
            name: version(name) for name in ("numpy", "scipy", "scikit-learn", "soundfile", "joblib")})
        self.assertEqual(record["imported_versions"], record["versions"])
        self.assertEqual(Path(record["executable"]).resolve(), Path(job.base_python()).resolve())

    def test_worker_script_allowlist(self):
        with self.assertRaisesRegex(ValueError, "Only the fixed"):
            job.worker_command("scripts/run_robustness.py")

    def test_failure_records_actual_exit(self):
        status, failure, _ = self.run_child("raise SystemExit(7)")
        self.assertEqual(failure, "child_exit_7")
        self.assertEqual(status["stages"][0]["exit_code"], 7)
        self.assertEqual(status["stages"][0]["status"], "stopped")

    def test_timeout_terminates_owned_child(self):
        status, failure, _ = self.run_child("import time; time.sleep(30)", budget=0.05)
        self.assertEqual(failure, "stage_runtime_budget_exceeded")
        self.assertIsNotNone(status["stages"][0]["exit_code"])
        self.assertLess(status["stages"][0]["elapsed_seconds"], 10)

    def test_peak_memory_stops_child(self):
        status, failure, _ = self.run_child("import time; time.sleep(30)", memory={
            "child_peak_rss_bytes": 2 * job.GIB + 1, "system_available_ram_bytes": 4 * job.GIB})
        self.assertEqual(failure, "child_peak_memory_budget_exceeded")
        self.assertEqual(status["stages"][0]["peak_rss_bytes"], 2 * job.GIB + 1)

    def test_system_pressure_stops_child(self):
        _, failure, _ = self.run_child("import time; time.sleep(30)", memory={
            "child_peak_rss_bytes": 1, "system_available_ram_bytes": job.GIB - 1})
        self.assertEqual(failure, "system_memory_pressure")

    def test_lock_refuses_duplicate_and_preserves_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "job.lock"
            job.acquire_lock(path, {"pid": 123})
            with self.assertRaises(FileExistsError):
                job.acquire_lock(path, {"pid": 456})
            self.assertEqual(json.loads(path.read_text())["pid"], 123)

    def test_status_atomic_replace_retries_transient_windows_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "status.json"
            actual = job.os.replace
            calls = []
            def replace(source, destination):
                calls.append(1)
                if len(calls) == 1:
                    raise PermissionError("brief Windows reader lock")
                return actual(source, destination)
            with patch.object(job.os, "replace", side_effect=replace):
                job.write_json(target, {"status": "complete"})
            self.assertEqual(len(calls), 2)
            self.assertEqual(json.loads(target.read_text())["status"], "complete")

    def test_incomplete_success_artifact_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "Unexpected full baseline scope"):
                job.validate_baseline_completion({"scope": "pilot"}, Path(temporary))
            path = Path(temporary)
            (path / "scores.csv").write_text("utterance_id,split,score\nx,dev,0\n")
            with self.assertRaisesRegex(RuntimeError, "exact train/dev splits"):
                job.validate_baseline_completion({"scope": "full-official-train-dev-adaptation"}, path)


if __name__ == "__main__":
    unittest.main()
