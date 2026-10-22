"""Supervisor failures must stop the owned child and retain failure status."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import ssl_smoke_job as job


class SSLSupervisorTests(unittest.TestCase):
    def fixture(self, root):
        for relative in ['scripts/ssl_smoke_job.py', 'scripts/ssl_smoke_worker.py',
                         'src/audio_deepfake/localization_ssl_geometry.py', 'requirements/ssl-windows.lock',
                         'scripts/full_run_job.py', 'src/audio_deepfake/resources.py']:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('fixture', encoding='utf-8')
        path = root / 'configs/localization_ssl_v1.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({'budgets': {
            'minimum_start_available_ram_bytes': 4 * 1024**3,
            'minimum_runtime_available_ram_bytes': 2 * 1024**3,
            'minimum_free_disk_bytes': 20 * 1024**3,
            'worker_peak_bytes': 3 * 1024**3, 'smoke_timeout_seconds': 180}}))
        (root / 'outputs/jobs').mkdir(parents=True)

    def test_memory_pressure_terminates_owned_child_and_clears_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            child = Mock()
            child.pid = 123
            child._handle = 456
            child.poll.return_value = None
            child.wait.return_value = 0
            readings = [{'system_available_ram_bytes': 5 * 1024**3},
                        {'system_available_ram_bytes': 1024**3, 'child_peak_rss_bytes': 1024**3}]
            directory = root / 'outputs/jobs/test'
            with patch.object(job, 'ROOT', root), patch.object(job, 'windows_memory', side_effect=readings), \
                 patch.object(job.shutil, 'disk_usage', return_value=Mock(free=30 * 1024**3)), \
                 patch.object(job, 'base_python', return_value='python'), \
                 patch.object(job.subprocess, 'CREATE_NO_WINDOW', 0, create=True), \
                 patch.object(job.subprocess, 'Popen', return_value=child), \
                 patch.object(sys, 'argv', ['job', '--execute', '--job-dir', str(directory)]):
                with self.assertRaisesRegex(RuntimeError, 'available RAM'):
                    job.main()
            child.terminate.assert_called_once()
            status = json.loads((directory / 'status.json').read_text())
            self.assertEqual(status['status'], 'stopped')
            self.assertFalse(status['training_started'])
            self.assertFalse((root / 'outputs/ssl_smoke.lock').exists())

    def test_default_plan_does_not_spawn(self):
        with patch.object(sys, 'argv', ['job']), patch.object(job.subprocess, 'Popen') as spawn:
            job.main()
            spawn.assert_not_called()


if __name__ == '__main__':
    unittest.main()
