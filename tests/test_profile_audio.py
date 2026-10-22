"""Check profile provenance and split guards with generated audio only."""

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from audio_deepfake.manifests import ManifestRow, write_manifest

SPEC = importlib.util.spec_from_file_location(
    "profile_audio", Path(__file__).resolve().parents[1] / "scripts/profile_audio.py"
)
profile = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(profile)


class ProfileAudioTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.audio_root = self.root / "audio"
        self.output = self.root / "profile"
        self.manifest = self.root / "selected.csv"
        self.journal = self.root / "journal.json"
        rows = []
        records = {}
        tone = (0.2 * np.sin(2 * np.pi * 440 * np.arange(1600) / 16000)).astype(np.float32)
        for split, prefix, speaker in [("train", "T", "LA_0001"), ("dev", "D", "LA_0002")]:
            identifier = f"LA_{prefix}_1000001"
            relative = f"ASVspoof2019_LA_{split}/flac/{identifier}.flac"
            rows.append(ManifestRow(split, speaker, identifier, "-", "bonafide", relative))
            path = self.audio_root / relative
            path.parent.mkdir(parents=True)
            sf.write(path, tone, 16000, subtype="PCM_16")
            payload = path.read_bytes()
            records[relative] = {"sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}
        write_manifest(rows, self.manifest)
        self.journal.write_text(json.dumps({"records": records}), encoding="utf-8")

    def test_profile_counts_and_public_summary(self):
        result = profile.profile_manifest(self.manifest, self.audio_root, self.journal,
                                          self.output, expected_count=2)
        self.assertEqual(result["records"], 2)
        self.assertAlmostEqual(result["audio_duration_seconds"], 0.2)
        self.assertEqual(result["workers"], 1)
        self.assertGreater(result["feature_bytes_if_cached"], 0)
        self.assertGreater(result["memory_after"]["peak_rss_bytes"], 0)
        self.assertNotIn(str(self.root), (self.output / "summary.json").read_text())
        self.assertTrue((self.output / "recordings.csv").is_file())

    def test_corrupt_journal_blocks_profile_before_export(self):
        journal = json.loads(self.journal.read_text())
        next(iter(journal["records"].values()))["sha256"] = "0" * 64
        self.journal.write_text(json.dumps(journal), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "differs from download journal"):
            profile.profile_manifest(self.manifest, self.audio_root, self.journal,
                                     self.output, expected_count=2)
        self.assertFalse(self.output.exists())

    def test_eval_and_count_mismatch_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Expected exactly"):
            profile.profile_manifest(self.manifest, self.audio_root, self.journal, self.output)
        write_manifest([ManifestRow("eval", "LA_0003", "LA_E_1000001", "-", "bonafide",
                                    "ASVspoof2019_LA_eval/flac/LA_E_1000001.flac")], self.manifest)
        with self.assertRaisesRegex(ValueError, "only accepts train/dev"):
            profile.profile_manifest(self.manifest, self.audio_root, self.journal,
                                     self.output, expected_count=1)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
