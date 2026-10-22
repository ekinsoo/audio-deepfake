import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ssl_metric_adapter", ROOT / "scripts/evaluate_localization_ssl.py")
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


class SSLMetricAdapterTests(unittest.TestCase):
    def test_reject_missing_nonfinite_and_out_of_range_metric_rows(self):
        self.assertEqual(adapter.parse_selected_row("BoundedSearch selected_row=[-2.0,0.2,0.21,0.01]"),
                         [-2., .2, .21, .01])
        for text in ("worker ended without results", "BoundedSearch selected_row=[0,NaN,0.2,0.1]",
                     "BoundedSearch selected_row=[0,1.2,0.2,1]", "BoundedSearch selected_row=[0,0.1]"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                adapter.parse_selected_row(text)

    def test_worker_routes_ssl_scores_to_unchanged_metric_function(self):
        argv = ["evaluate_localization_ssl.py", "--worker", "--pilot",
                str(ROOT / "outputs/localization/ssl-test"), "--output-dir",
                str(ROOT / "outputs/localization/ssl-diagnostic-test")]
        with patch("sys.argv", argv), patch.dict(adapter.frozen.PILOTS), patch.object(adapter, "save"), \
             patch("full_run_job.configure_project_environment"), patch.object(adapter.frozen, "metric_worker") as worker:
            adapter.main()
            worker.assert_called_once_with("ssl", ROOT / "outputs/localization/ssl-diagnostic-test")
            self.assertEqual(adapter.frozen.PILOTS["ssl"], "outputs/localization/ssl-test")

    @unittest.skipUnless(adapter.os.name == "nt", "Windows resource supervisor")
    def test_external_memory_violation_stops_owned_worker_without_evaluating(self):
        import full_run_job
        class Child:
            pid, _handle = 123, "retained-handle"
            def poll(self):
                return None
        child = Child()
        healthy = {"system_available_ram_bytes": 4 * 1024**3}
        violation = {"system_available_ram_bytes": 1024**3, "child_peak_rss_bytes": 100}
        with tempfile.TemporaryDirectory() as directory, patch.object(full_run_job, "windows_memory", side_effect=[healthy, violation]), \
             patch.object(full_run_job, "stop_owned_child") as stop, patch.object(adapter.subprocess, "Popen", return_value=child):
            with self.assertRaises(RuntimeError):
                adapter.bounded(["never-executed"], Path(directory))
            stop.assert_called_once_with(child)


if __name__ == "__main__":
    unittest.main()
