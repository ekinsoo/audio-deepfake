"""Real Windows allocation/PID probe; no model, optimizer, audio or download."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from full_run_job import base_python, stop_owned_child, windows_memory


def module(filename, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@unittest.skipUnless(os.name == "nt", "Windows real-process working-set probe")
class SSLWorkerProcessMonitorTests(unittest.TestCase):
    def test_actual_python_pid_peak_and_venv_bootstrap(self):
        runner = module("run_localization_ssl_head.py", "head_monitor_probe")
        adapter = module("evaluate_localization_ssl.py", "metric_monitor_probe")
        commands = [runner.worker_command(ROOT / "outputs/unused-cache", ROOT / "outputs/unused-head"),
                    adapter.worker_command(ROOT / "outputs/unused-head", ROOT / "outputs/unused-errors", "errors"),
                    adapter.worker_command(ROOT / "outputs/unused-head", ROOT / "outputs/unused-range", "range")]
        for command in commands:
            self.assertTrue(Path(command[0]).samefile(base_python()))
            self.assertEqual(command[1:3], ["-S", "-u"])
        # Use the production launch prefix, replacing the workload with a
        # bounded 64 MiB allocation. The child waits until measurements finish.
        code = "\n".join([
            "import json, os, sys",
            f"sys.path.insert(0, {str(ROOT / 'scripts')!r})",
            "from full_run_job import configure_project_environment",
            "configure_project_environment()",
            "from audio_deepfake.resources import memory_snapshot",
            "from importlib.metadata import version",
            "allocation = bytearray(64 * 1024**2)",
            "for i in range(0, len(allocation), 4096): allocation[i] = 123",
            "print(json.dumps({'worker_pid':os.getpid(),'worker_executable':sys.executable,",
            "'allocated_bytes':len(allocation),'memory':memory_snapshot(),",
            "'numpy_version':version('numpy'),'training_started':False}), flush=True)",
            "sys.stdin.readline()",
        ])
        command = commands[0][:3] + ["-c", code]
        child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            proof = json.loads(child.stdout.readline())
            measured = windows_memory(child.pid, process_handle=child._handle)
            self.assertEqual(proof["worker_pid"], child.pid)
            self.assertTrue(Path(proof["worker_executable"]).samefile(base_python()))
            self.assertGreaterEqual(measured["child_peak_rss_bytes"], 64 * 1024**2)
            self.assertGreaterEqual(measured["child_peak_rss_bytes"], proof["memory"]["peak_rss_bytes"])
            expected_numpy = json.loads((ROOT / "docs/validation/localization_pilot_v1.json").read_text())["results"]["training"]["versions"]["numpy"]
            self.assertEqual(proof["numpy_version"], expected_numpy)
            child.stdin.write("release\n")
            child.stdin.flush()
            self.assertEqual(child.wait(timeout=10), 0)
            final = windows_memory(child.pid, process_handle=child._handle)
            self.assertGreaterEqual(final["child_peak_rss_bytes"], measured["child_peak_rss_bytes"])
            print(json.dumps({"probe": "real_windows_same_pid_64MiB", "worker": proof,
                "monitor_pid": child.pid, "observed_peak_rss_bytes": measured["child_peak_rss_bytes"],
                "final_peak_rss_bytes": final["child_peak_rss_bytes"], "exit_code": child.returncode}))
        finally:
            stop_owned_child(child)
            for stream in (child.stdin, child.stdout, child.stderr):
                stream.close()


if __name__ == "__main__":
    unittest.main()
