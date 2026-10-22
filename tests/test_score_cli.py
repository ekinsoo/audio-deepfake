"""Single-recording diagnostics using generated audio and a tiny saved model."""
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from audio_deepfake.baseline import GMMConfig, save_model, train_feature_stream
from audio_deepfake.cli import main
from audio_deepfake.features import LFCCConfig, extract_lfcc
from audio_deepfake.manifests import ManifestRow


class ScoreCLITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        cls.root = Path(cls.temporary.name)
        config = LFCCConfig(n_coeffs=4, delta_order=0)
        time = np.arange(8000, dtype=np.float32) / 16000
        cls.waves = {label: (0.2 * np.sin(2 * np.pi * frequency * time)).astype(np.float32)
                     for label, frequency in (("bonafide", 220), ("spoof", 1800))}
        rows = [ManifestRow("train", "speaker", f"LA_T_{index}",
                            "-" if label == "bonafide" else "A01", label,
                            f"ASVspoof2019_LA_train/flac/LA_T_{index}.flac")
                for index, label in enumerate(cls.waves)]
        model = train_feature_stream(rows, lambda row: extract_lfcc(cls.waves[row.label], config),
                                     lfcc_config=config,
                                     gmm_config=GMMConfig(n_components=1, frames_per_class=100, max_iter=5))
        cls.model_path = cls.root / "trusted.joblib"
        save_model(model, cls.model_path)
        for label, waveform in cls.waves.items():
            sf.write(cls.root / f"{label}.wav", waveform, 16000, subtype="FLOAT")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def invoke(self, *arguments):
        output, errors = io.StringIO(), io.StringIO()
        with patch.object(sys, "argv", ["audio-deepfake", *map(str, arguments)]), redirect_stdout(output), redirect_stderr(errors):
            try:
                status = main()
            except SystemExit as exc:
                status = exc.code
        return status, output.getvalue(), errors.getvalue()

    def test_score_direction_output_contract_and_no_private_path(self):
        for label, sign in (("bonafide", 1), ("spoof", -1)):
            status, output, errors = self.invoke("score", self.root / f"{label}.wav", "--model", self.model_path)
            self.assertEqual(status, 0, errors)
            result = json.loads(output)
            self.assertGreater(sign * result["recording_score"], 0)
            self.assertEqual(result["score_direction"], "higher_is_bonafide")
            self.assertEqual(result["sample_rate"], 16000)
            self.assertEqual(result["samples"], 8000)
            self.assertEqual(result["duration_seconds"], 0.5)
            self.assertEqual(result["lfcc_config"]["n_coeffs"], 4)
            self.assertEqual(result["gmm_config"]["n_components"], 1)
            self.assertEqual(result["model_sha256"], hashlib.sha256(self.model_path.read_bytes()).hexdigest())
            self.assertEqual(result["scope"], "single_recording_diagnostic")
            self.assertNotIn(str(self.root), output)
            self.assertNotIn("probability", result)
            self.assertNotIn("prediction", result)

    def test_invalid_artifact_or_audio_fails_cleanly(self):
        corrupt = self.root / "corrupt.joblib"
        corrupt.write_bytes(b"invalid artifact")
        wrong_rate = self.root / "wrong_rate.wav"
        sf.write(wrong_rate, self.waves["bonafide"], 8000)
        for audio, model in ((self.root / "bonafide.wav", self.root / "missing.joblib"),
                             (self.root / "bonafide.wav", corrupt),
                             (self.root / "missing.wav", self.model_path),
                             (wrong_rate, self.model_path)):
            with self.subTest(audio=audio.name, model=model.name):
                status, output, errors = self.invoke("score", audio, "--model", model)
                self.assertEqual(status, 1)
                self.assertEqual(output, "")
                self.assertIn("Error:", errors)
                self.assertNotIn("Traceback", errors)

    def test_model_required_and_help_explains_trust(self):
        status, _, errors = self.invoke("score", self.root / "bonafide.wav")
        self.assertEqual(status, 2)
        self.assertIn("--model", errors)
        status, output, _ = self.invoke("score", "--help")
        self.assertEqual(status, 0)
        self.assertIn("created and trust", output)
        self.assertIn("execute code", output)


if __name__ == "__main__":
    unittest.main()
