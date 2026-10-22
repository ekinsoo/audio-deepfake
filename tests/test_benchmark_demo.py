import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('benchmark_demo', Path(__file__).resolve().parents[1] / 'scripts/benchmark_demo.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class BenchmarkTests(unittest.TestCase):
    def test_aggregate_rtf_uses_total_duration_not_unweighted_average(self):
        result = benchmark.summarize([
            {'wall_seconds': 0.2, 'duration_seconds': 1},
            {'wall_seconds': 0.2, 'duration_seconds': 3},
        ])
        self.assertAlmostEqual(result['aggregate_rtf'], 0.1)
        self.assertAlmostEqual(result['mean_record_rtf'], (0.2 + 0.2 / 3) / 2)
        self.assertEqual(result['mean_ms'], 200)

    def test_invalid_measurements_refused(self):
        for rows in ([], [{'wall_seconds': float('nan'), 'duration_seconds': 2}],
                     [{'wall_seconds': 1, 'duration_seconds': 0}],
                     [{'wall_seconds': -1, 'duration_seconds': 2}]):
            with self.assertRaises(ValueError):
                benchmark.summarize(rows)

    def test_existing_output_refused_before_model_load_or_measurement(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = Path(directory) / 'plan.json'
            plan.write_text('keep')
            with patch.object(benchmark, 'build_plan') as build:
                with self.assertRaises(ValueError):
                    benchmark.execute(plan, Path(directory) / 'result.json')
                build.assert_not_called()
            self.assertEqual(plan.read_text(), 'keep')

    def test_resource_gate_blocks_low_ram_even_with_disk_space(self):
        with patch.object(benchmark, 'memory_snapshot', return_value={
                'system_available_ram_bytes': 1024**3 - 1, 'peak_rss_bytes': 100}), \
                patch.object(benchmark.shutil, 'disk_usage', return_value=type('Usage', (), {'free': 100 * 1024**3})()):
            with self.assertRaises(RuntimeError):
                benchmark.resource_check()
