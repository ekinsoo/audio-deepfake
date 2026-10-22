"""Real alignment adapters must reject changed samples or silent label repairs."""
import importlib.util
from pathlib import Path
import tempfile
import io
import tarfile
import time
import unittest
from unittest.mock import patch
import numpy as np
import soundfile as sf

spec = importlib.util.spec_from_file_location("partial_alignment", Path(__file__).resolve().parents[1] / "scripts/audit_partial_alignment.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class AlignmentTests(unittest.TestCase):
    def fixture(self, directory, *, samples=321, vad_samples=320):
        root = Path(directory)
        audio = root / "sample.wav"
        sf.write(audio, np.zeros(samples), 16000)
        vad = root / "verified_train_dev/database/vad/train/sample.vad"
        vad.parent.mkdir(parents=True)
        vad.write_text(f"0 {vad_samples / 16000} 2\n")
        selected = {"utterance_id": "sample", "vad_member": "database/vad/train/sample.vad",
                    "vad_sha256": module.digest(vad), "vad_duration_seconds": vad_samples / 16000}
        receipt = {"sha256": module.digest(audio), "frames": samples}
        return root, audio, selected, receipt

    def test_native_actual_sample_alignment_survives_early_vad_endpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root, audio, selected, receipt = self.fixture(directory)
            with patch.object(module, "DEST", root), patch.object(module, "ROOT", root):
                record = module.inspect_record(selected, [1, 0, 1], audio, receipt)
            self.assertTrue(record["candidate_match"])
            self.assertEqual(record["vad_audio_delta_samples"], 1)
            self.assertFalse(record["labels_trimmed_or_padded"])
            np.testing.assert_array_equal(module.load_aligned_labels(record, [1, 0, 1], 321), [0, 1])

    def test_missing_terminal_label_does_not_get_padded(self):
        with tempfile.TemporaryDirectory() as directory:
            root, audio, selected, receipt = self.fixture(directory)
            with patch.object(module, "DEST", root), patch.object(module, "ROOT", root):
                record = module.inspect_record(selected, [1, 0], audio, receipt)
            self.assertFalse(record["candidate_match"])
            self.assertIn("differs", record["alignment_error"])
            with self.assertRaises(ValueError):
                module.load_aligned_labels(record, [1, 0], 321)

    def test_changed_audio_hash_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, audio, selected, receipt = self.fixture(directory)
            receipt["sha256"] = "0" * 64
            with patch.object(module, "DEST", root), self.assertRaisesRegex(ValueError, "receipt/header"):
                module.inspect_record(selected, [1, 0, 1], audio, receipt)

    def test_replay_changed_samples_rejected(self):
        record = {"candidate_match": True, "sample_count": 321, "native_count_10ms": 3, "coarse_count_20ms": 2}
        with self.assertRaisesRegex(ValueError, "samples"):
            module.load_aligned_labels(record, [1, 0, 1], 320)
        with self.assertRaisesRegex(ValueError, "Label count"):
            module.load_aligned_labels(record, [1, 0], 321)

    def test_inventory_finds_extensionless_rttm_without_reading_eval_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = {"artifacts": {}}
            for name in ("database_protocols.tar.gz", "database_segment_labels_v1.2.tar.gz", "database_vad.tar.gz"):
                path = root / name
                with tarfile.open(path, "w:gz") as archive:
                    for member_name in ("database/dev/con_data/rttm_2cls_0sil", "database/segment_labels/eval_seglab_0.01.npy"):
                        entry = tarfile.TarInfo(member_name)
                        entry.size = 8
                        archive.addfile(entry, io.BytesIO(b"notnumpy"))
                metadata["artifacts"][name] = {"size_bytes": path.stat().st_size,
                    "published_md5": module.digest(path, "md5"), "sha256": module.digest(path)}
            with patch.object(module, "DEST", root), patch.object(tarfile.TarFile, "extractfile", side_effect=AssertionError("Payload opened")):
                inventory = module.native_rttm_inventory(metadata, deadline=time.monotonic() + 5)
            for item in inventory.values():
                self.assertEqual(item["rttm_member_names"], ["database/dev/con_data/rttm_2cls_0sil"])
                self.assertEqual(item["payloads_extracted"], 0)


if __name__ == "__main__":
    unittest.main()
