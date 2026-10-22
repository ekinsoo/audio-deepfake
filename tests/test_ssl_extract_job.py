"""Offline resource, cumulative-resume and cache-integrity regression tests."""
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import ssl_extract_job as job
import ssl_extract_worker as worker


def npy(path, values, shape=(1, 768), dtype='<f4'):
    header = repr({'descr': dtype, 'fortran_order': False, 'shape': shape}).encode('latin1') + b'\n'
    path.write_bytes(b'\x93NUMPY\x01\x00' + struct.pack('<H', len(header)) + header +
                     struct.pack('<' + 'f' * len(values), *values))


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root / 'test.npy'

    def tearDown(self):
        self.temp.cleanup()

    def test_npy_finite_and_geometry(self):
        npy(self.path, [0.1] * 768)
        job.validate_npy(self.path, [1, 768])
        with self.assertRaises(ValueError):
            job.validate_npy(self.path, [2, 768])

    def test_nan_rejected(self):
        npy(self.path, [float('nan')] + [0.] * 767)
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            job.validate_npy(self.path, [1, 768])

    def test_trailing_or_truncated_data_rejected(self):
        npy(self.path, [0.] * 768)
        self.path.write_bytes(self.path.read_bytes() + b'x')
        with self.assertRaisesRegex(ValueError, 'size'):
            job.validate_npy(self.path, [1, 768])
        self.path.write_bytes(self.path.read_bytes()[:-10])
        with self.assertRaisesRegex(ValueError, 'size'):
            job.validate_npy(self.path, [1, 768])

    def test_wrong_dtype_rejected(self):
        npy(self.path, [0.] * 768, dtype='<f8')
        with self.assertRaises(ValueError):
            job.validate_npy(self.path, [1, 768])

    def test_final_receipt_hash_identity_and_audio(self):
        row = {'split': 'train', 'utterance_id': 'ID', 'coarse_count_20ms': 1,
               'sample_count': 400, 'audio_sha256': 'audio'}
        target = self.root / 'cache/train/ID.npy'
        target.parent.mkdir(parents=True)
        npy(target, [0.] * 768)
        receipt = {'status': 'complete', 'identity_sha256': 'identity', 'split': 'train',
                   'recording_id': 'ID', 'audio_sha256': 'audio', 'shape': [1, 768], 'dtype': 'float32',
                   'source_frame_count': 1, 'features_sha256': worker.sha(target),
                   'features_bytes': target.stat().st_size}
        worker.save(target.with_suffix('.json'), receipt)
        checked = job.validate_all(self.root, {'records': [row]}, 'identity')
        self.assertEqual(checked[0]['features_sha256'], receipt['features_sha256'])
        orphan = target.parent / 'orphan.npy.tmp'
        orphan.write_bytes(b'orphan')
        with self.assertRaisesRegex(ValueError, 'inventory'):
            job.validate_all(self.root, {'records': [row]}, 'identity')
        orphan.unlink()
        with self.assertRaisesRegex(ValueError, 'provenance'):
            job.validate_all(self.root, {'records': [row]}, 'other-device')
        target.write_bytes(target.read_bytes()[:-1] + b'X')
        with self.assertRaisesRegex(ValueError, 'hash'):
            job.validate_all(self.root, {'records': [row]}, 'identity')

    def test_atomic_json_has_no_leftover_temp(self):
        path = self.root / 'result.json'
        worker.save(path, {'status': 'stopped'})
        self.assertEqual(json.loads(path.read_text())['status'], 'stopped')
        self.assertFalse(path.with_suffix('.json.tmp').exists())

    def test_bundle_escape_rejected(self):
        with self.assertRaises(ValueError):
            worker.local(self.root, '../escape')


class ResourceTests(unittest.TestCase):
    def test_resume_counts_failed_elapsed_time(self):
        self.assertEqual(job.remaining_budget({'status': 'stopped', 'cumulative_elapsed_seconds': 1790}), 10)
        for value in (1800, 1801, -1, float('nan')):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                job.remaining_budget({'cumulative_elapsed_seconds': value})

    def test_ram_disk_rss_guards(self):
        good = {'system_available_ram_bytes': 5 * job.GIB, 'peak_rss_bytes': job.GIB}
        with patch.object(job, 'memory', return_value=good), patch.object(job, 'cache_bytes', return_value=0), \
                patch.object(job.shutil, 'disk_usage') as disk:
            disk.return_value.free = 21 * job.GIB
            job.guard(Path('.'), initial=True, pid=123)
            good['system_available_ram_bytes'] = 3 * job.GIB
            with self.assertRaisesRegex(RuntimeError, 'RAM'):
                job.guard(Path('.'), initial=True)
            job.guard(Path('.'))
            good['peak_rss_bytes'] = 4 * job.GIB
            with self.assertRaisesRegex(RuntimeError, 'RSS'):
                job.guard(Path('.'))
            good['peak_rss_bytes'] = 0
            disk.return_value.free = 19 * job.GIB
            with self.assertRaisesRegex(RuntimeError, 'disk'):
                job.guard(Path('.'))

    def test_external_deadline_stops_before_worker_poll(self):
        class Child:
            def poll(self):
                return None
        with self.assertRaisesRegex(RuntimeError, 'deadline'):
            job.monitor(Child(), Path('.'), {'peak_rss_bytes': 0}, 10, 0, 1790, clock=lambda: 10)

    def test_source_config_remains_pinned(self):
        self.assertEqual(worker.CONFIG_SHA, worker.sha(SCRIPTS.parent / 'configs/localization_ssl_v1.json'))
        self.assertEqual(job.SMOKE_AUDIT_SHA,
                         worker.sha(SCRIPTS.parent / 'docs/validation/localization_ssl_colab_smoke_v2.json'))


if __name__ == '__main__':
    unittest.main()
