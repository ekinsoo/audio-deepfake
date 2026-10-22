"""Timeline boundary contracts, separate from dataset/metric verification."""

import unittest

import numpy as np

from audio_deepfake.localization import (
    intervals_to_spoof_mask, native_labels_to_spoof_mask,
    scores_to_spoof_intervals, spoof_mask_to_intervals,
    spoof_overlap_fraction, time_grid, validate_intervals,
)


class LocalizationTests(unittest.TestCase):
    def test_exact_decimal_grid_and_short_tail(self):
        self.assertEqual(time_grid(duration=0.14, step=0.02).shape, (7, 2))
        grid = time_grid(duration=0.045, step=0.02)
        np.testing.assert_allclose(grid, [[0, .02], [.02, .04], [.04, .045]])
        np.testing.assert_array_equal(time_grid(duration=.005, step=.02), [[0, .005]])

    def test_reject_invalid_grid_and_resource_explosion(self):
        for duration, step in [(0, .02), (-1, .02), (np.nan, .02), (1, np.inf),
                               (True, .02), (1, 0), (1, 1e-10)]:
            with self.subTest(duration=duration, step=step):
                with self.assertRaises(ValueError):
                    time_grid(duration=duration, step=step)

    def test_invalid_and_overlapping_intervals_are_not_repaired(self):
        invalid = [[(-.01, .02)], [(0, 0)], [(.03, .02)], [(0, .11)],
                   [(0, np.inf)], [(0, np.nan)], [(False, .02)], [(0,)],
                   [(0, .04), (.03, .05)], [(.05, .06), (0, .02)]]
        for ranges in invalid:
            with self.subTest(ranges=ranges), self.assertRaises(ValueError):
                validate_intervals(ranges, duration=.1)
        self.assertEqual(validate_intervals([(0, .02), (.02, .1)], duration=.1),
                         ((0., .02), (.02, .1)))

    def test_coverage_handles_adjacent_ranges_and_short_final_bin(self):
        fractions = spoof_overlap_fraction(
            [(0, .01), (.01, .02), (.042, .045)], duration=.045, step=.02)
        np.testing.assert_allclose(fractions, [1, 0, .6])
        np.testing.assert_array_equal(
            spoof_overlap_fraction([], duration=.04, step=.02), [0, 0])

    def test_rasterization_policy_is_explicit_and_half_open(self):
        kwargs = dict(duration=.04, step=.02)
        # First midpoint at .01 is excluded from [0,.01); second at .03 is included.
        ranges = [(0, .01), (.03, .04)]
        np.testing.assert_array_equal(
            intervals_to_spoof_mask(ranges, policy="midpoint", **kwargs), [False, True])
        np.testing.assert_array_equal(
            intervals_to_spoof_mask(ranges, policy="majority", **kwargs), [False, False])
        np.testing.assert_array_equal(
            intervals_to_spoof_mask(ranges, policy="any_overlap", **kwargs), [True, True])
        # Exact decimal comparison must preserve a real, very small majority.
        np.testing.assert_array_equal(intervals_to_spoof_mask(
            [(0, .010000000000001), (.030000000000001, .04)],
            policy="majority", **kwargs), [True, False])
        with self.assertRaises(ValueError):
            intervals_to_spoof_mask(ranges, policy="unknown", **kwargs)

    def test_native_label_direction_unknown_and_length_guards(self):
        kwargs = dict(duration=.045, step=.02)
        np.testing.assert_array_equal(native_labels_to_spoof_mask([0, 1, 0], **kwargs),
                                      [True, False, True])
        for labels in [[0, 2, 1], [0, np.nan, 1], [0, 1], [[0, 1, 0]],
                       ["0", "1", "0"], [True, False, True], [.5, 0, 1]]:
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                native_labels_to_spoof_mask(labels, **kwargs)

    def test_runs_have_no_off_by_one_and_clip_tail(self):
        kwargs = dict(duration=.085, step=.02)
        self.assertEqual(spoof_mask_to_intervals([True, True, False, True, True], **kwargs),
                         ((0., .04), (.06, .085)))
        self.assertEqual(spoof_mask_to_intervals([False] * 5, **kwargs), ())
        self.assertEqual(spoof_mask_to_intervals([True] * 5, **kwargs), ((0., .085),))
        with self.assertRaises(ValueError):
            spoof_mask_to_intervals([0, 1, 0, 1, 0], **kwargs)

    def test_score_direction_equality_and_invalid_scores(self):
        kwargs = dict(threshold=.5, duration=.065, step=.02)
        self.assertEqual(scores_to_spoof_intervals([.49, .5, .8, -.1], **kwargs),
                         ((0., .02), (.06, .065)))
        for scores in [[np.nan, 0, 0, 0], [np.inf, 0, 0, 0], [False] * 4, [0]]:
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                scores_to_spoof_intervals(scores, **kwargs)
        with self.assertRaises(ValueError):
            scores_to_spoof_intervals([0] * 4, threshold=np.nan, duration=.065, step=.02)


if __name__ == "__main__":
    unittest.main()
