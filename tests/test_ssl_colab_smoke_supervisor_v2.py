"""Regression coverage for exit races and strict CUDA-smoke supervision."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

PATH = Path(__file__).resolve().parents[1] / "scripts/ssl_colab_smoke_supervisor_v2.py"
spec = importlib.util.spec_from_file_location("colab_supervisor_v2", PATH)
supervisor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(supervisor)


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class Child:
    pid = 1234

    def __init__(self, clock, *, exit_after=None, code=0):
        self.clock, self.exit_after, self.code = clock, exit_after, code

    def poll(self):
        return self.code if self.exit_after is not None and self.clock() >= self.exit_after else None

    def wait(self, timeout=None):
        if self.poll() is not None:
            return self.code
        assert timeout is not None
        self.clock.sleep(timeout)
        if self.poll() is not None:
            return self.code
        raise subprocess.TimeoutExpired("owned-child", timeout)


def missing_metrics(pid):
    raise supervisor.ProcessProbeUnavailable(f"Process {pid} lacks VmRSS,VmHWM; state=R")


class ExitRaceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.clock = Clock()
        self.state = {"peak_rss_bytes": 100}
        system = patch.object(supervisor, "guard_system", return_value={
            "system_available_ram_bytes": 5 * supervisor.GIB})
        process = patch.object(supervisor, "process_memory", side_effect=missing_metrics)
        system.start()
        process.start()
        self.addCleanup(system.stop)
        self.addCleanup(process.stop)

    def monitor(self, child, deadline=180):
        return supervisor.monitor_child(child, self.root, self.state, deadline,
                                        clock=self.clock, sleep=self.clock.sleep)

    def test_exit_race_waits_for_owned_child(self):
        self.assertEqual(self.monitor(Child(self.clock, exit_after=.2)), 0)
        self.assertEqual(self.state["last_process_probe_error"]["error_type"], "ProcessProbeUnavailable")
        self.assertLess(self.clock(), supervisor.EXIT_GRACE_SECONDS)

    def test_missing_metrics_does_not_allow_live_child_indefinitely(self):
        with self.assertRaisesRegex(supervisor.ResourceProbeError, "beyond exit grace"):
            self.monitor(Child(self.clock))
        self.assertLessEqual(self.clock(), supervisor.EXIT_GRACE_SECONDS + .001)

    def test_system_ram_gate_remains_active_during_exit_grace(self):
        def guard(output):
            if self.clock() >= .1:
                raise supervisor.ResourceLimitError("Available system RAM boundary reached")
            return {"system_available_ram_bytes": 5 * supervisor.GIB}

        with patch.object(supervisor, "guard_system", side_effect=guard):
            with self.assertRaisesRegex(supervisor.ResourceLimitError, "RAM"):
                self.monitor(Child(self.clock, exit_after=.5))

    def test_exit_grace_cannot_extend_overall_deadline(self):
        with self.assertRaisesRegex(supervisor.ResourceLimitError, "180-second"):
            self.monitor(Child(self.clock, exit_after=.5), deadline=.15)
        self.assertLessEqual(self.clock(), .151)

    def test_nonzero_exit_rejected_even_with_complete_result(self):
        (self.root / "result.json").write_text('{"status":"complete"}')
        with self.assertRaisesRegex(RuntimeError, "code 7"):
            supervisor.validate_completion(self.root, self.root, {"exit_code": 7}, SimpleNamespace(), 180)

    def test_live_process_peak_gate_remains_active(self):
        with patch.object(supervisor, "process_memory", return_value={
                "rss_bytes": 4 * supervisor.GIB, "peak_rss_bytes": 4 * supervisor.GIB}):
            with self.assertRaisesRegex(supervisor.ResourceLimitError, "peak RSS"):
                self.monitor(Child(self.clock))

    def test_resource_failure_terminates_owned_child_and_records_type(self):
        bundle = self.root / "bundle"
        bundle.mkdir()
        (bundle / "manifest.json").write_text('{}')
        output = self.root / "attempt-v2"
        child = Mock(pid=1234)
        child.poll.return_value = None
        child.wait.return_value = -15
        frozen = Mock()
        with patch.object(supervisor.sys, "platform", "linux"), \
             patch.object(supervisor, "MANIFEST_SHA256", supervisor.sha(bundle / "manifest.json")), \
             patch.object(supervisor, "load_frozen_worker", return_value=(frozen, bundle / "ssl_colab_smoke.py")), \
             patch.object(supervisor.subprocess, "Popen", return_value=child), \
             patch.object(supervisor.os, "killpg", create=True) as kill, \
             patch.object(supervisor, "process_memory", side_effect=supervisor.ResourceLimitError("peak RSS")):
            with self.assertRaisesRegex(supervisor.ResourceLimitError, "peak RSS"):
                supervisor.supervise(bundle, output)
        kill.assert_called_once_with(1234, supervisor.signal.SIGTERM)
        child.wait.assert_called_once_with(timeout=5)
        result = json.loads((output / "status.json").read_text())
        self.assertEqual(result["status"], "stopped")
        self.assertEqual(result["error_type"], "ResourceLimitError")
        self.assertFalse(result["training_started"])


class ProcessParsingTests(unittest.TestCase):
    def test_missing_fields_are_explicit_process_probe_error(self):
        with patch.object(supervisor.Path, "read_text", return_value="State:\tZ (zombie)\n"):
            with self.assertRaisesRegex(supervisor.ProcessProbeUnavailable, "VmRSS,VmHWM"):
                supervisor.process_memory(1234)

    def test_process_memory_reads_peak_not_only_current_rss(self):
        with patch.object(supervisor.Path, "read_text", return_value="VmRSS:\t10 kB\nVmHWM:\t20 kB\n"):
            self.assertEqual(supervisor.process_memory(1234), {"rss_bytes": 10240, "peak_rss_bytes": 20480})


if __name__ == "__main__":
    unittest.main()
