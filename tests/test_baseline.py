"""Training boundaries, deterministic sampling, likelihood direction and artifacts."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from audio_deepfake.baseline import (
    GMMConfig, load_model, save_model, score_features, score_rows, train_feature_stream,
)
from audio_deepfake.features import LFCCConfig
from audio_deepfake.manifests import ManifestRow


def row(index, label, split="train"):
    prefix = {"train": "T", "dev": "D", "eval": "E"}[split]
    uid = f"LA_{prefix}_{index}"
    attack = "-" if label == "bonafide" else ("A07" if split == "eval" else "A01")
    return ManifestRow(split, f"speaker_{split}", uid, attack, label,
                       f"ASVspoof2019_LA_{split}/flac/{uid}.flac")


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.lfcc = LFCCConfig(n_coeffs=2, delta_order=0)
        self.config = GMMConfig(n_components=2, frames_per_class=40, max_iter=5)
        self.rows = [row(1, "bonafide"), row(2, "spoof"),
                     row(3, "bonafide"), row(4, "spoof")]
        rng = np.random.default_rng(17)
        self.features = {r.utterance_id: rng.normal(
            4 if r.label == "bonafide" else -4, 0.2, (35, 2)).astype(np.float32)
            for r in self.rows}

    def train(self, rows=None, loader=None, config=None):
        return train_feature_stream(self.rows if rows is None else rows,
                                    loader or (lambda r: self.features[r.utterance_id]),
                                    lfcc_config=self.lfcc, gmm_config=config or self.config)

    def test_score_direction_and_sampling_metadata(self):
        model = self.train()
        self.assertGreater(score_features(model, self.features[self.rows[0].utterance_id]), 0)
        self.assertLess(score_features(model, self.features[self.rows[1].utterance_id]), 0)
        for info in model.metadata["classes"].values():
            self.assertEqual(info["frames_seen"], 70)
            self.assertEqual(info["frames_sampled"], 40)
            self.assertGreater(info["iterations"], 0)
        self.assertEqual(model.metadata["normalization"], "none")

    def test_input_order_does_not_change_reservoir_or_model(self):
        first, second = self.train(), self.train(list(reversed(self.rows)))
        for label in ("bonafide", "spoof"):
            np.testing.assert_array_equal(getattr(first, label).means_, getattr(second, label).means_)
            np.testing.assert_array_equal(getattr(first, label).covariances_, getattr(second, label).covariances_)

    def test_reject_split_class_and_duplicate_before_loading(self):
        def fail_loader(r):
            self.fail("Invalid manifests must be rejected before features are loaded")
        for records in ([], [row(1, "bonafide")], self.rows + [self.rows[0]],
                        [row(1, "bonafide", "dev"), row(2, "spoof", "dev")],
                        [row(1, "bonafide", "eval"), row(2, "spoof", "eval")]):
            with self.subTest(records=records), self.assertRaises(ValueError):
                self.train(records, fail_loader)

    def test_eval_scoring_requires_explicit_flag(self):
        with self.assertRaisesRegex(ValueError, "allow_eval"):
            score_rows(self.train(), [row(1, "bonafide", "eval")], ".")

    def test_scarce_data_and_finite_validation(self):
        for matrix in (np.ones((1, 2), dtype=np.float32),
                       np.ones((4, 3), dtype=np.float32),
                       np.full((4, 2), np.nan), np.ones((4, 2), dtype=int)):
            with self.subTest(shape=matrix.shape), self.assertRaises(ValueError):
                self.train(loader=lambda r: matrix,
                           rows=[row(1, "bonafide"), row(2, "spoof")])
        # Scarcity below the budget is allowed when each mixture has enough frames.
        model = self.train(config=GMMConfig(n_components=2, frames_per_class=100, max_iter=5))
        self.assertEqual(model.metadata["classes"]["bonafide"]["frames_sampled"], 70)
        with self.assertRaises(ValueError):
            score_features(model, np.full((4, 2), np.inf))
        model.bonafide.means_[:] = np.nan
        with np.errstate(all="ignore"), self.assertRaises(ValueError):
            score_features(model, np.ones((4, 2)))

    def test_save_load_predictions_and_configs(self):
        model = self.train()
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            path = Path(directory) / "model.joblib"
            save_model(model, path)
            loaded = load_model(path)
        self.assertEqual(loaded.lfcc_config, model.lfcc_config)
        self.assertEqual(loaded.gmm_config, model.gmm_config)
        self.assertEqual(loaded.metadata, model.metadata)
        for features in self.features.values():
            self.assertEqual(score_features(loaded, features), score_features(model, features))

    def test_invalid_configuration(self):
        for kwargs in (dict(n_components=0), dict(frames_per_class=1),
                       dict(n_components=10, frames_per_class=5), dict(seed=-1),
                       dict(max_iter=True), dict(reg_covar=0), dict(tol=float("nan"))):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                GMMConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
