import unittest

import numpy as np

from audio_deepfake.partialspoof import (
    binary_labels, coarse_binary_labels, labels_for_audio,
    parse_vad, vad_spoof_intervals,
)


class PartialSpoofTests(unittest.TestCase):
    def test_publisher_strings_and_minimum_rule_keep_partial_group(self):
        labels = np.array(["1", "1", "1", "0", "1"], dtype="<U21")
        np.testing.assert_array_equal(binary_labels(labels), [1, 1, 1, 0, 1])
        np.testing.assert_array_equal(coarse_binary_labels(labels, factor=2), [1, 0, 1])
        np.testing.assert_array_equal(coarse_binary_labels(labels, factor=10), [0])

    def test_unknown_nonspeech_or_permissive_codes_rejected(self):
        for values in (["0", "2"], ["1.0"], [0, np.nan], [True, False], [], [[0, 1]]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                binary_labels(values)

    def test_actual_sample_count_never_silently_trims(self):
        np.testing.assert_array_equal(labels_for_audio([1, 0, 1], sample_count=321), [0, 1])
        np.testing.assert_array_equal(labels_for_audio([1], sample_count=1), [1])
        with self.assertRaises(ValueError):
            labels_for_audio([1, 0], sample_count=321)

    def test_vad_preserves_nonspeech_and_validates_complete_audio(self):
        segments = parse_vad("0 .01 2\n.01 .03 0\n.03 .04 1\n", duration=.04)
        self.assertEqual([s.native_label for s in segments], [2, 0, 1])
        self.assertEqual(vad_spoof_intervals(segments), ((.01, .03),))
        cases = ["", "0 .02 3", "0 .02 0\n.03 .04 1", "0 .03 0\n.02 .04 1",
                 "0 .03 0", "0 inf 0", "0 .05 0", ".01 .04 0"]
        for text in cases:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_vad(text, duration=.04)


if __name__ == "__main__":
    unittest.main()
