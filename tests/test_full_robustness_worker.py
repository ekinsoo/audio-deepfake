import importlib.util
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from audio_deepfake.manifests import ManifestRow

spec = importlib.util.spec_from_file_location('full_robustness_worker', Path(__file__).resolve().parents[1] / 'scripts/full_robustness_worker.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class WorkerTests(unittest.TestCase):
    def test_chunk_bounds_disallow_bool_empty_eval_and_oversize(self):
        for start, stop in ((True, 2), (0, False), (-1, 1), (1, 1), (0, 2001), (24843, 24845)):
            with self.assertRaises(ValueError):
                worker.validate_range(start, stop)
        worker.validate_range(24000, 24844)

    def test_fixed_average_and_delta_keep_metadata(self):
        row = ManifestRow('dev', 'D1', 'LA_D_1', 'A01', 'spoof', 'unused')
        record = worker.paired_record(row, 1e308, 1e308, 'analysis')
        self.assertEqual(record['s_average'], 1e308)
        self.assertEqual(record['delta'], 0)
        self.assertEqual(record['attack_id'], 'A01')
        self.assertEqual(record['group'], 'analysis')
        for value in (math.inf, math.nan):
            with self.assertRaises(ValueError):
                worker.paired_record(row, value, 1, 'analysis')
        with self.assertRaises(ValueError):
            worker.paired_record(row, -1e308, 1e308, 'analysis')

    def test_hash_failure_precedes_unpickle(self):
        contract = {'seeds': {'42': {'model_path': 'fake', 'model_sha256': 'pinned'}}}
        with patch.object(worker.full, 'digest', return_value='changed'), patch.object(worker, 'load_model') as load:
            with self.assertRaisesRegex(ValueError, 'refusing unpickle'):
                worker.trusted_model(42, contract, {})
            load.assert_not_called()

    def test_overwrite_is_refused_before_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / 'chunks').mkdir()
            (output / 'chunks/seed42_00000_00020.json').write_text('preserve')
            with patch.object(worker, 'score_subset') as score:
                with self.assertRaises(ValueError):
                    worker.chunk(output, 42, 0, 20)
                score.assert_not_called()
            (output / 'smoke.json').write_text('preserve')
            with self.assertRaises(ValueError):
                worker.smoke(output)
            self.assertEqual((output / 'smoke.json').read_text(), 'preserve')

    def test_execution_rejects_changed_sources(self):
        execution = {'status': 'frozen', 'protocol_sha256': worker.CONTRACT_SHA256,
                     'seeds': [42, 43, 44], 'chunk_size': 2000, 'per_seed_budget_seconds': 1800,
                     'source_sha256': {name: 'old' for name in worker.EXECUTION_SOURCES}}
        with patch.object(worker, 'read_json', return_value=execution), patch.object(worker.full, 'digest', return_value='changed'):
            with self.assertRaisesRegex(ValueError, 'Execution sources'):
                worker.validate_execution(Path('.'))


if __name__ == '__main__':
    unittest.main()
