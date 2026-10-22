import unittest
import importlib.util
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

from audio_deepfake.localization_context import recording_context


class RecordingContextTests(unittest.TestCase):
    def test_windows_status_replace_retries_then_succeeds(self):
        path = Path(__file__).resolve().parents[1] / "scripts/run_localization_context.py"
        spec = importlib.util.spec_from_file_location("context_runner_test", path)
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        original = Path.replace
        count = 0

        def transient(source, target):
            nonlocal count
            count += 1
            if count < 3:
                raise PermissionError("temporary Windows reader lock")
            return original(source, target)

        with tempfile.TemporaryDirectory() as directory, patch.object(Path, "replace", transient):
            runner.save_status(Path(directory), "fitting", True)
            self.assertIn('"training_started": true', (Path(directory) / "status.json").read_text())
        self.assertEqual(count, 3)

    def test_formula_all_dimensions_and_no_input_mutation(self):
        values = np.tile(np.array([1., 3., 9.])[:, None], (1, 60))
        original = values.copy()
        actual = recording_context(values)
        np.testing.assert_allclose(actual[:, 0], np.array([-2., 0., 6.]) / (1.4826 * 2.))
        np.testing.assert_allclose(actual[:, 59], actual[:, 0])
        np.testing.assert_array_equal(values, original)
        self.assertEqual(actual.dtype, np.float64)

    def test_constant_and_single_bin_remain_zero(self):
        np.testing.assert_array_equal(recording_context(np.full((4, 60), 8.)), np.zeros((4, 60)))
        np.testing.assert_array_equal(recording_context(np.full((1, 60), 8.)), np.zeros((1, 60)))

    def test_fixed_floor_preserves_sparse_difference(self):
        values = np.zeros((3, 60))
        values[2] = .002
        np.testing.assert_array_equal(recording_context(values)[2], np.full(60, 2.))

    def test_recordings_do_not_share_statistics(self):
        values = np.tile(np.array([1., 3., 9.])[:, None], (1, 60))
        expected = recording_context(values)
        recording_context(values * 999 + 500)
        np.testing.assert_array_equal(recording_context(values), expected)
        np.testing.assert_allclose(recording_context(values + 500), expected)

    def test_reject_invalid_or_nonfinite(self):
        for value in (np.zeros((0, 60)), np.zeros((4, 59)), np.zeros(60),
                      np.ones((2, 60), dtype=int), np.full((2, 60), np.nan),
                      np.full((2, 60), np.inf)):
            with self.subTest(shape=value.shape):
                with self.assertRaises(ValueError):
                    recording_context(value)


if __name__ == "__main__":
    unittest.main()
