"""Signal-level checks for the frozen narrowband transformation."""

import unittest

import numpy as np

from audio_deepfake.transforms import TRANSFORM_SPEC, narrowband_roundtrip


class NarrowbandTests(unittest.TestCase):
    def test_odd_even_and_short_lengths_are_exact_and_finite(self):
        for count in (1, 2, 3, 17, 16000, 16001):
            with self.subTest(count=count):
                source = np.linspace(-0.3, 0.4, count, dtype=np.float32)
                result = narrowband_roundtrip(source)
                self.assertEqual(result.shape, source.shape)
                self.assertEqual(result.dtype, np.float32)
                self.assertTrue(np.isfinite(result).all())

    def test_deterministic_nonmutating_for_readonly_view(self):
        storage = np.random.default_rng(42).normal(size=2003).astype(np.float32)
        snapshot = storage.copy()
        source = storage[::2]
        source.setflags(write=False)
        first = narrowband_roundtrip(source)
        second = narrowband_roundtrip(source)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(storage, snapshot)
        self.assertFalse(np.shares_memory(first, storage))

    def test_reject_invalid_audio(self):
        for source in ([], np.zeros((3, 2)), np.array([1, 2]),
                       np.array([True]), np.array([1 + 2j]),
                       np.array(["0.3"]), np.array([np.nan]),
                       np.array([np.inf]), np.array([-np.inf])):
            with self.subTest(source=repr(source)), self.assertRaises(ValueError):
                narrowband_roundtrip(source)

    def test_reject_other_rates_and_noninteger_rate(self):
        for rate in (8000, 48000, 16000.0, True, "16000", None):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                narrowband_roundtrip(np.zeros(16, dtype=np.float32), rate)

    def test_low_frequency_preserved_and_high_frequency_attenuated(self):
        time = np.arange(32000, dtype=np.float64) / 16000
        rms = lambda value: float(np.sqrt(np.mean(np.square(value[512:-512], dtype=np.float64))))
        low = 0.4 * np.sin(2 * np.pi * 1000 * time)
        high = 0.4 * np.sin(2 * np.pi * 6000 * time)
        low_ratio = rms(narrowband_roundtrip(low)) / rms(low)
        high_ratio = rms(narrowband_roundtrip(high)) / rms(high)
        self.assertGreater(low_ratio, 0.98)
        self.assertLess(low_ratio, 1.02)
        self.assertLess(high_ratio, 0.01)

    def test_gain_is_linear_without_normalization_or_clipping(self):
        time = np.arange(16000, dtype=np.float64) / 16000
        quiet = 0.2 * np.sin(2 * np.pi * 1000 * time)
        quiet_result = narrowband_roundtrip(quiet)
        loud_result = narrowband_roundtrip(quiet * 10)
        np.testing.assert_allclose(loud_result, quiet_result * 10, rtol=2e-6, atol=2e-7)
        self.assertGreater(float(np.max(np.abs(loud_result[512:-512]))), 1.9)

    def test_metadata_describes_fixed_processing(self):
        self.assertEqual(TRANSFORM_SPEC["window"], ["kaiser", 5.0])
        self.assertEqual(TRANSFORM_SPEC["processing_dtype"], "float64")
        self.assertEqual(TRANSFORM_SPEC["output_dtype"], "float32")
        self.assertEqual(TRANSFORM_SPEC["padtype"], "constant")
        self.assertFalse(TRANSFORM_SPEC["clipping"])


if __name__ == "__main__":
    unittest.main()
