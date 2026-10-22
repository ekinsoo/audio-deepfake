"""Check input rules that could otherwise change benchmark data silently."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from audio_deepfake.audio import load_audio
from audio_deepfake.cli import smoke_check


class AudioTests(unittest.TestCase):
    def test_flac_round_trip(self):
        self.assertEqual(smoke_check()["status"], "ok")

    def test_invalid_inputs_are_rejected(self):
        cases = [
            ("stereo.wav", np.zeros((160, 2)), 16_000, "PCM_16", "mono"),
            ("rate.wav", np.zeros(160), 8_000, "PCM_16", "16000 Hz"),
            ("empty.wav", np.zeros(0), 16_000, "PCM_16", "empty"),
            ("nan.wav", np.array([0.0, np.nan]), 16_000, "FLOAT", "non-finite"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            for name, samples, rate, subtype, message in cases:
                with self.subTest(name=name):
                    path = Path(directory) / name
                    sf.write(path, samples, rate, subtype=subtype)
                    with self.assertRaisesRegex(ValueError, message):
                        load_audio(path)


if __name__ == "__main__":
    unittest.main()
