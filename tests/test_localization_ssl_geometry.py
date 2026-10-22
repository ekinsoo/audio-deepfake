import unittest

import numpy as np

from audio_deepfake.localization_ssl_geometry import convolution_geometry, frame_mapping


class SSLGeometryTests(unittest.TestCase):
    def test_checkpoint_convolutions(self):
        self.assertEqual(convolution_geometry([10, 3, 3, 3, 3, 2, 2], [5, 2, 2, 2, 2, 2, 2]), (400, 320))

    def test_counts_follow_valid_convolutions_and_clipped_bins(self):
        for n in [400, 401, 639, 640, 719, 720, 721, 120001]:
            frames = n
            for kernel, stride in zip([10, 3, 3, 3, 3, 2, 2], [5, 2, 2, 2, 2, 2, 2]):
                frames = (frames - kernel) // stride + 1
            out = frame_mapping(n)
            self.assertEqual(out['source_frame_count'], frames)
            self.assertEqual(len(out['frame_indices']), (n + 319) // 320)
            self.assertEqual(out['bin_ends_samples'][-1], n)
            self.assertTrue(np.all(out['frame_indices'] < frames))

    def test_last_bin_uses_its_actual_midpoint(self):
        out = frame_mapping(960)
        np.testing.assert_array_equal(out['frame_indices'], [0, 1, 1])

    def test_tie_uses_earlier_frame(self):
        out = frame_mapping(1000, receptive_samples=320, hop_samples=640)
        self.assertEqual(out['frame_indices'][1], 0)

    def test_invalid_and_short_input(self):
        for n in [0, -1, True, 399, 400.0]:
            with self.assertRaises(ValueError):
                frame_mapping(n)
        with self.assertRaises(ValueError):
            convolution_geometry([10], [0])


if __name__ == '__main__':
    unittest.main()
