import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from audio_deepfake.localization_model import TemporalConfig, train_temporal
from audio_deepfake.localization_ssl_head import (
    reference_sampling, score_ssl_head, train_ssl_head,
)

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ssl_head_runner", ROOT / "scripts/run_localization_ssl_head.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class FakeClassifier:
    """Capture inputs without running even a synthetic optimizer."""
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.n_iter_ = np.array([1])
        self.classes_ = np.array([0, 1])

    def fit(self, x, y):
        self.fitted_x, self.fitted_y = x.copy(), y.copy()
        return self

    def decision_function(self, x):
        return x[:, 0]


class SSLHeadTests(unittest.TestCase):
    def fixture(self, dimensions=60):
        rows = [{"split": "train", "utterance_id": f"T{i}"} for i in range(5)]
        rng = np.random.default_rng(103)
        values = {r["utterance_id"]: (rng.normal(size=(31, dimensions)).astype(np.float32),
                                   np.r_[np.zeros(9), np.ones(22)]) for r in rows}
        return rows, values

    @patch("audio_deepfake.localization_ssl_head.LogisticRegression", FakeClassifier)
    @patch("audio_deepfake.localization_model.LogisticRegression", FakeClassifier)
    def test_exact_frozen_lfcc_sampler_equivalence_with_repeated_cap_pruning(self):
        rows, values = self.fixture()
        config = TemporalConfig(frames_per_class=12)
        loader = lambda r: values[r["utterance_id"]]
        reference = reference_sampling(rows, lambda r: loader(r)[1], config=config)
        lfcc = train_temporal(rows, loader, config=config)
        ssl = train_ssl_head(rows[::-1], loader, dimensions=60, config=config,
                            expected_sample_digest=reference["retained_sample_ids_sha256"])
        self.assertEqual(ssl.metadata["retained_feature_bytes_sha256"],
                         lfcc.metadata["retained_feature_bytes_sha256"])
        np.testing.assert_array_equal(ssl.classifier.fitted_x, lfcc.classifier.fitted_x)
        np.testing.assert_array_equal(ssl.classifier.fitted_y, lfcc.classifier.fitted_y)
        self.assertEqual(ssl.classifier.kwargs, lfcc.classifier.kwargs)
        self.assertEqual(ssl.metadata["frames_retained_per_class"], 12)
        self.assertEqual(ssl.metadata["frames_seen_by_native_class"], [45, 110])

    @patch("audio_deepfake.localization_ssl_head.LogisticRegression", FakeClassifier)
    def test_768_dimension_scaler_train_only_and_reference_balance(self):
        rows, values = self.fixture(768)
        loader = lambda r: values[r["utterance_id"]]
        reference = reference_sampling(rows, lambda r: loader(r)[1])
        model = train_ssl_head(rows, loader, expected_retained_per_class=45,
                               expected_sample_digest=reference["retained_sample_ids_sha256"])
        before = model.scaler.mean_.copy()
        self.assertEqual(len(score_ssl_head(model, np.full((3, 768), 1000., dtype=np.float32))), 3)
        np.testing.assert_array_equal(model.scaler.mean_, before)
        self.assertEqual(model.scaler.n_samples_seen_, 90)

    def test_reject_dev_duplicate_and_wrong_inputs_before_fit(self):
        calls = []
        for rows in ([{"split": "dev", "utterance_id": "D"}],
                     [{"split": "train", "utterance_id": "T"}] * 2):
            with self.assertRaises(ValueError):
                train_ssl_head(rows, calls.append)
        self.assertEqual(calls, [])
        rows = [{"split": "train", "utterance_id": "T"}]
        for x, y in ((np.zeros((4, 60), np.float32), [0, 0, 1, 1]),
                     (np.zeros((4, 768)), [0, 0, 1, 1]),
                     (np.full((4, 768), np.nan, np.float32), [0, 0, 1, 1]),
                     (np.zeros((4, 768), np.float32), [0, 0, 2, 1])):
            with self.assertRaises(ValueError):
                train_ssl_head(rows, lambda _: (x, y))

    def test_mismatch_stops_before_optimizer_and_convergence_warning_is_fatal(self):
        rows, values = self.fixture(768)
        loader = lambda r: values[r["utterance_id"]]
        with patch("audio_deepfake.localization_ssl_head.LogisticRegression") as classifier:
            for kwargs in ({"expected_retained_per_class": 4524}, {"expected_sample_digest": "wrong"}):
                with self.assertRaises(ValueError):
                    train_ssl_head(rows, loader, **kwargs)
            classifier.assert_not_called()
            classifier.return_value.fit.side_effect = lambda *_: warnings.warn("not converged", ConvergenceWarning)
            with self.assertRaises(ConvergenceWarning):
                train_ssl_head(rows, loader)


class CacheGateTests(unittest.TestCase):
    def test_completed_identity_arrays_and_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = {"config.json": runner.CONFIG,
                "localization_ssl_geometry.py": "src/audio_deepfake/localization_ssl_geometry.py",
                "ssl_extract_worker.py": "scripts/ssl_extract_worker.py", "ssl_extract_job.py": "scripts/ssl_extract_job.py",
                "partial_selection_v1.json": "docs/validation/partial_selection_v1.json",
                "localization_alignment.json": "docs/validation/localization_alignment.json"}
            source_hashes = {name: runner.sha(ROOT / path) for name, path in sources.items()}
            identity_body = {"model_sha256": {"pytorch_model.bin": "checkpoint", "config.json": "model-config",
                "preprocessor_config.json": "processor"}, "source_sha256": source_hashes,
                "worker_sha256": source_hashes["ssl_extract_worker.py"],
                "supervisor_sha256": source_hashes["ssl_extract_job.py"], "device": "cuda",
                "environment": [["numpy", "test"]]}
            import hashlib
            digest = hashlib.sha256(json.dumps(identity_body, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()
            (root / "identity.json").write_text(json.dumps({"identity": identity_body, "identity_sha256": digest}))
            (root / "worker-result.json").write_text(json.dumps({"status": "complete", "identity_sha256": digest,
                "record_count": 192, "training_started": False, "eval_accessed": False}))
            indexed = {"train": {"T": {"sample_count": 800, "audio_sha256": "audioT"}},
                       "dev": {"D": {"sample_count": 800, "audio_sha256": "audioD"}}}
            checked = []
            for split, uid in (("train", "T"), ("dev", "D")):
                folder = root / "cache" / split
                folder.mkdir(parents=True)
                path = folder / f"{uid}.npy"
                np.save(path, np.zeros((3, 768), np.float32), allow_pickle=False)
                (folder / f"{uid}.json").write_text(json.dumps({"identity_sha256": digest,
                    "status": "complete", "recording_id": uid, "split": split,
                    "audio_sha256": f"audio{uid}", "features_sha256": runner.sha(path),
                    "features_bytes": path.stat().st_size, "shape": [3, 768], "dtype": "float32",
                    "source_frame_count": 2}))
                checked.append({"split": split, "recording_id": uid, "features_sha256": runner.sha(path),
                                "receipt_sha256": runner.sha(folder / f"{uid}.json")})
            (root / "validation.json").write_text(json.dumps({"identity_sha256": digest, "records": checked}))
            (root / "status.json").write_text(json.dumps({"status": "complete", "identity_sha256": digest,
                "record_count": 192, "exit_code": 0, "training_started": False, "eval_accessed": False,
                "validation_sha256": runner.sha(root / "validation.json")}))
            config = {"checkpoint": {"sha256": "checkpoint", "metadata_sha256": {
                "config.json": "model-config", "preprocessor_config.json": "processor"}}}
            runner.checked_cache(root, indexed, config)
            for key in ("config.json", "preprocessor_config.json"):
                config["checkpoint"]["metadata_sha256"][key] += "changed"
                with self.assertRaises(ValueError):
                    runner.checked_cache(root, indexed, config)
                config["checkpoint"]["metadata_sha256"][key] = config["checkpoint"]["metadata_sha256"][key][:-7]
            np.save(root / "cache/dev/D.npy", np.ones((3, 768), np.float32), allow_pickle=False)
            with self.assertRaises(ValueError):
                runner.checked_cache(root, indexed, config)
            (root / "status.json").write_text(json.dumps({"status": "running", "identity_sha256": digest}))
            with self.assertRaises(ValueError):
                runner.checked_cache(root, indexed, config)


if __name__ == "__main__":
    unittest.main()
