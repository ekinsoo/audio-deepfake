import unittest

import numpy as np

from audio_deepfake.localization_model import (
    TemporalConfig, score_temporal, train_temporal,
)


class TemporalModelTests(unittest.TestCase):
    def test_reject_dev_before_loading_any_features(self):
        calls = []
        with self.assertRaises(ValueError):
            train_temporal([{"split": "dev", "utterance_id": "D1"}], calls.append)
        self.assertEqual(calls, [])

    def test_balanced_reservoir_reproducible_under_record_order(self):
        rows = [{"split": "train", "utterance_id": f"T{i}"} for i in range(3)]
        rng = np.random.default_rng(1)
        values = {}
        for row in rows:
            labels = np.r_[np.zeros(6), np.ones(10)]
            x = rng.normal(size=(16, 60)).astype(np.float32) + labels[:, None] * 2
            values[row["utterance_id"]] = x, labels
        loader = lambda row: values[row["utterance_id"]]
        config = TemporalConfig(frames_per_class=8)
        first = train_temporal(rows, loader, config=config)
        second = train_temporal(rows[::-1], loader, config=config)
        np.testing.assert_array_equal(first.scaler.mean_, second.scaler.mean_)
        np.testing.assert_array_equal(first.classifier.coef_, second.classifier.coef_)
        self.assertEqual(first.metadata["frames_seen_by_native_class"], [18, 30])
        self.assertEqual(first.metadata["frames_retained_per_class"], 8)
        self.assertGreater(score_temporal(first, np.full((1, 60), 2., dtype=np.float32))[0],
                           score_temporal(first, np.zeros((1, 60), dtype=np.float32))[0])

    def test_normalization_train_only_and_invalid_label_alignment(self):
        rows = [{"split": "train", "utterance_id": "T1"}]
        x = np.repeat([[1.], [2.], [3.], [4.]], 60, axis=1).astype(np.float32)
        model = train_temporal(rows, lambda _: (x, [0, 0, 1, 1]))
        np.testing.assert_allclose(model.scaler.mean_, 2.5)
        original = model.scaler.mean_.copy()
        score_temporal(model, x + 100)
        np.testing.assert_array_equal(model.scaler.mean_, original)
        for labels in ([0, 0, 2, 1], [0, 1], [True, False, True, False]):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                train_temporal(rows, lambda _: (x, labels))


if __name__ == "__main__":
    unittest.main()
