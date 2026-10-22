"""Signal-processing invariants and explicit LFCC boundary policies."""

import unittest

import numpy as np

from audio_deepfake.features import LFCCConfig, extract_lfcc, linear_filterbank, temporal_delta


class LFCCTests(unittest.TestCase):
    def test_silence_is_finite_and_stationary(self):
        features = extract_lfcc(np.zeros(1600, dtype=np.float32))
        self.assertEqual(features.shape, (9, 60))
        self.assertEqual(features.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(features)))
        np.testing.assert_allclose(features[:, 1:20], 0, atol=1e-12)
        np.testing.assert_array_equal(features[:, 20:], 0)
        self.assertLess(features[0, 0], 0)

    def test_filterbank_partition_and_frequency_selectivity(self):
        config = LFCCConfig()
        bank = linear_filterbank(config)
        self.assertEqual(bank.shape, (20, 257))
        self.assertTrue(np.all((bank >= 0) & (bank <= 1)))
        np.testing.assert_array_equal(bank[:, [0, -1]], 0)
        # Between the first and last triangle peaks, adjacent filters sum to 1.
        hz = np.linspace(0, 8000, 257)
        interior = (hz >= 8000 / 21) & (hz <= 8000 * 20 / 21)
        np.testing.assert_allclose(bank.sum(axis=0)[interior], 1, atol=1e-14)
        # A spectrum containing only a 1 kHz bin activates its two neighbors.
        response = bank[:, 32]
        self.assertEqual(np.count_nonzero(response), 2)
        self.assertEqual(int(response.argmax()), 2)

    def test_short_audio_and_tail_boundaries(self):
        for samples, expected_frames in [(1, 1), (319, 1), (320, 1), (321, 2),
                                         (479, 2), (480, 2), (481, 3)]:
            with self.subTest(samples=samples):
                features = extract_lfcc(np.zeros(samples, dtype=np.float32))
                self.assertEqual(features.shape, (expected_frames, 60))

    def test_delta_slope_and_replicated_edges(self):
        ramp = np.arange(5, dtype=np.float64)[:, None]
        np.testing.assert_array_equal(temporal_delta(ramp)[:, 0], [1, 2, 2, 2, 1])
        np.testing.assert_array_equal(temporal_delta(np.ones((1, 2))), 0)
        np.testing.assert_array_equal(temporal_delta(np.ones((5, 2))), 0)

    def test_gain_only_changes_c0_without_preemphasis(self):
        # A full frame avoids padded-tail effects; power gain becomes a constant
        # log-filterbank shift, which must affect c0 alone.
        x = np.random.default_rng(42).normal(size=320)
        config = LFCCConfig(pre_emphasis=0, delta_order=0)
        difference = extract_lfcc(2 * x, config) - extract_lfcc(x, config)
        self.assertAlmostEqual(float(difference[0, 0]), np.sqrt(20) * np.log10(4), places=5)
        np.testing.assert_allclose(difference[:, 1:], 0, atol=1e-6)

    def test_repeatability_and_input_preservation(self):
        x = np.random.default_rng(7).normal(size=2000).astype(np.float32)
        original = x.copy()
        np.testing.assert_array_equal(extract_lfcc(x), extract_lfcc(x))
        np.testing.assert_array_equal(x, original)
        self.assertEqual(extract_lfcc(x, LFCCConfig(delta_order=1)).shape[1], 40)

    def test_reject_invalid_inputs_and_configs(self):
        for x in [np.array([]), np.zeros((4, 2)), np.array([np.nan]), np.ones(4, dtype=int)]:
            with self.subTest(x=x):
                with self.assertRaises(ValueError):
                    extract_lfcc(x)
        for kwargs in [dict(n_fft=128), dict(n_filters=1000), dict(n_coeffs=21),
                       dict(delta_order=3), dict(hop_ms=30), dict(window_ms=0.001),
                       dict(pre_emphasis=float("nan")), dict(n_fft=511)]:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    LFCCConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
