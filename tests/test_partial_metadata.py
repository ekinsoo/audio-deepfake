"""Bounded metadata input and traversal checks without network calls."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('partial_metadata', Path(__file__).resolve().parents[1] / 'scripts/prepare_partial_metadata.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class MetadataTests(unittest.TestCase):
    def test_existing_corrupt_artifact_is_not_replaced_or_trusted(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'tiny'
            target.write_bytes(b'bad')
            with patch.object(module, 'DEST', Path(directory)), patch.object(module, 'FILES', {'tiny': (3, '0' * 32)}):
                with self.assertRaisesRegex(RuntimeError, 'checksum mismatch'):
                    module.fetch({'transferred_bytes': 0})
            self.assertEqual(target.read_bytes(), b'bad')

    def test_transfer_plan_has_no_audio_and_fits_budget(self):
        self.assertEqual(sum(x[0] for x in module.FILES.values()), 92_230_352)
        self.assertLess(sum(x[0] for x in module.FILES.values()), module.BUDGET)
        self.assertFalse(any('database_train.' in x or 'database_dev.' in x or 'database_eval.' in x for x in module.FILES))

    def test_reject_traversal_windows_and_absolute_paths(self):
        for name in ['../evil', '/tmp/evil', 'a/../../evil', 'C:/evil', 'a\\evil']:
            with self.subTest(name=name), self.assertRaises(ValueError):
                module.safe_member(name)

    def test_eval_path_not_allowed(self):
        self.assertFalse(module.allowed_member('database/vad/eval/CON_E_1.vad'))
        self.assertFalse(module.allowed_member('database/segment_labels/eval_seglab_0.01.npy'))
        self.assertFalse(module.allowed_member('database/protocols/PartialSpoof_LA_cm_protocols/PartialSpoof.LA.cm.eval.trl.txt'))
        self.assertTrue(module.allowed_member('database/segment_labels/train_seglab_0.01.npy'))
        self.assertTrue(module.allowed_member('database/protocols/PartialSpoof_LA_cm_protocols/PartialSpoof.LA.cm.dev.trl.txt'))
        self.assertTrue(module.allowed_member('database/vad/train/CON_T_1.vad'))
        self.assertTrue(module.allowed_member('database/vad/dev/CON_D_1.vad'))

if __name__ == '__main__':
    unittest.main()
