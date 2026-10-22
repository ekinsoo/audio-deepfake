"""Exact sample-boundary and context checks for temporal LFCC adaptation."""

import unittest
from unittest.mock import patch

import numpy as np

from audio_deepfake.features import LFCCConfig, extract_lfcc
from audio_deepfake.localization import time_grid
from audio_deepfake.localization_features import (
    _nearest_frame_indices, extract_localization_features,
)


class LocalizationFeatureTests(unittest.TestCase):
    def test_one_sample_and_window_hop_boundaries(self):
        cases = [(1, [0]), (319, [0]), (320, [0]), (321, [0, 1]),
                 (479, [0, 1]), (480, [0, 1]), (481, [0, 2]),
                 (640, [0, 2]), (641, [0, 2, 3]), (960, [0, 2, 4])]
        for length, indices in cases:
            with self.subTest(length=length):
                x = np.random.default_rng(9).normal(size=length).astype(np.float32)
                result = extract_localization_features(x, sample_rate=16_000)
                np.testing.assert_array_equal(result.frame_indices, indices)
                np.testing.assert_array_equal(result.features, extract_lfcc(x)[indices])
                self.assertEqual(result.features.shape, (len(indices), 60))
                self.assertEqual(result.features.dtype, np.float32)
                self.assertEqual(result.bin_starts[0], 0)
                self.assertEqual(result.bin_ends[-1], length / 16_000)

    def test_exact_tail_center_tie_chooses_earlier_frame(self):
        # Integer center comparison preserves ties and one-half-sample changes.
        np.testing.assert_array_equal(_nearest_frame_indices(
            np.array([0, 320, 480, 481, 640, 999]), np.array([320, 640])),
            [0, 0, 0, 1, 1, 1])
        x = np.zeros(481, dtype=np.float32)
        # For 481 samples finalcenter=400.5, the later center480 wins strictly.
        result = extract_localization_features(x, sample_rate=16_000)
        np.testing.assert_array_equal(result.frame_indices, [0, 2])

    def test_long_recording_grid_matches_explicit_seconds_contract(self):
        for length in (16_000, 16_001, 22_400, 22_401):
            with self.subTest(length=length):
                x = np.random.default_rng(12).normal(size=length).astype(np.float32)
                original = x.copy()
                result = extract_localization_features(x, sample_rate=16_000)
                grid = time_grid(duration=length / 16_000, step=.02)
                np.testing.assert_array_equal(result.bin_starts, grid[:, 0])
                np.testing.assert_array_equal(result.bin_ends, grid[:, 1])
                np.testing.assert_array_equal(x, original)
                np.testing.assert_array_equal(
                    result.features, extract_localization_features(x, sample_rate=16_000).features)

    def test_custom_coefficients_keep_geometry_explicit(self):
        config = LFCCConfig(n_coeffs=10, delta_order=0)
        result = extract_localization_features(np.zeros(641), sample_rate=16_000, config=config)
        self.assertEqual(result.features.shape, (3, 10))

    def test_reject_sample_rate_geometry_and_invalid_audio(self):
        for rate in (8000, 16000.0, True, "16000"):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                extract_localization_features(np.zeros(320), sample_rate=rate)
        for config in (None, LFCCConfig(window_ms=10), LFCCConfig(hop_ms=20),
                       LFCCConfig(sample_rate=8000)):
            with self.subTest(config=config), self.assertRaises(ValueError):
                extract_localization_features(np.zeros(320), sample_rate=16000, config=config)
        for x in ([], np.zeros((320, 2)), np.zeros(320, dtype=int),
                  np.array([np.nan]), np.array([np.inf])):
            with self.subTest(x=x), self.assertRaises(ValueError):
                extract_localization_features(x, sample_rate=16000)

    def test_refuse_extractor_count_or_nonfinite_contract_drift(self):
        for output in (np.zeros((3, 60)), np.zeros((1, 20)), np.full((1, 60), np.nan)):
            with self.subTest(shape=output.shape):
                with patch("audio_deepfake.localization_features.extract_lfcc", return_value=output):
                    with self.assertRaises(ValueError):
                        extract_localization_features(np.zeros(320), sample_rate=16_000)


if __name__ == "__main__":
    unittest.main()
