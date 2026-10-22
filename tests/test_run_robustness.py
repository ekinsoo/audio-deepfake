"""Guarded frozen-model experiment orchestration using synthetic fixtures."""

import csv
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from audio_deepfake.baseline import GMMConfig
from audio_deepfake.features import LFCCConfig
from audio_deepfake.manifests import ManifestRow, summarize
from audio_deepfake.reliability import evaluate_error_ranking, partition_dev

SPEC = importlib.util.spec_from_file_location("run_robustness", Path(__file__).resolve().parents[1] / "scripts/run_robustness.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def make_row(index, split, speaker, label, attack="A01"):
    prefix = "T" if split == "train" else "D"
    utterance = f"LA_{prefix}_{index}"
    return ManifestRow(split, speaker, utterance, "-" if label == "bonafide" else attack,
                       label, f"ASVspoof2019_LA_{split}/flac/{utterance}.flac")


class RobustnessRunnerTests(unittest.TestCase):
    def fixture(self, root):
        config_path = root / "pilot.toml"
        config_path.write_bytes((runner.REPO_ROOT / "configs/pilot.toml").read_bytes())
        train = [make_row(index, "train", "T_speaker", "bonafide" if index < 1000 else "spoof") for index in range(2000)]
        dev = [make_row(index, "dev", f"D_{index // 50}", "bonafide" if index % 50 < 25 else "spoof",
                        f"A{index % 6 + 1:02d}") for index in range(1000)]
        groups = {"train": train, "dev": dev}
        inputs = {"selection_manifest": runner.FROZEN_SELECTION_SHA256, "audio_journal": "journalhash",
                  "asv_scores": "asvhash", "reference_evaluator": "referencehash"}
        plan = {"groups": groups, "gmm": GMMConfig(), "lfcc": LFCCConfig(),
                "audio_root": root / "audio", "asv": root / "asv", "reference": root / "reference",
                "input_sha256": inputs}
        model_metadata = {"training_ids": sorted(r.utterance_id for r in train),
                          "gmm_config": asdict(GMMConfig()), "lfcc_config": asdict(LFCCConfig())}
        mixture = SimpleNamespace(n_components=32, covariance_type="diag")
        model = SimpleNamespace(metadata=model_metadata, gmm_config=GMMConfig(), lfcc_config=LFCCConfig(),
                                bonafide=mixture, spoof=mixture)
        baseline_dir = root / "baseline"
        baseline_dir.mkdir()
        (baseline_dir / "model.joblib").write_bytes(b"mock local model; never deserialize")
        (baseline_dir / "config.json").write_text("{}")
        with (baseline_dir / "scores.csv").open("w", encoding="utf-8", newline="") as target:
            writer = csv.writer(target)
            writer.writerow(("utterance_id", "split", "label", "attack_id", "score"))
            for index, row in enumerate(dev):
                clean = (1 if row.label == "bonafide" else -1) * (0.1 if index % 3 else -0.1) + index / 10000
                writer.writerow((row.utterance_id, row.split, row.label, row.attack_id, clean))
        with config_path.open("rb") as source:
            config = runner.tomllib.load(source)
        baseline = {"status": "ok", "scope": "pilot", "experiment": config["experiment"],
                    "splits": {key: summarize(rows) for key, rows in groups.items()},
                    "input_sha256": {**inputs, "config": runner.digest(config_path)},
                    "output_sha256": {name: runner.digest(baseline_dir / name) for name in ("model.joblib", "scores.csv", "config.json")},
                    "source_sha256": {name: runner.digest(runner.REPO_ROOT / name) for name in
                                      ("src/audio_deepfake/baseline.py", "src/audio_deepfake/features.py", "src/audio_deepfake/audio.py")},
                    "model_metadata": model_metadata}
        (baseline_dir / "summary.json").write_text(json.dumps(baseline))
        _, _, audit = partition_dev(train, dev)
        protocol_path = root / "protocol.md"
        protocol_path.write_text("Synthetic frozen protocol fixture")
        audit.update({"selection_sha256": inputs["selection_manifest"],
                      "protocol_sha256": runner.digest(protocol_path),
                      "source_sha256": runner.digest(runner.REPO_ROOT / "src/audio_deepfake/reliability.py")})
        audit_path = root / "audit.json"
        audit_path.write_text(json.dumps(audit))
        return baseline_dir, config_path, audit_path, protocol_path, root / "new-output", plan, model

    def prepare(self, fixture):
        baseline, config, audit, protocol, output, plan, model = fixture
        with patch.object(runner, "preflight", return_value=plan), patch.object(runner, "load_model", return_value=model):
            return runner.prepare_run(baseline, config, audit, protocol, output)

    def test_tampered_model_rejected_before_deserialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.fixture(Path(temporary))
            (fixture[0] / "model.joblib").write_bytes(b"tampered")
            with patch.object(runner, "preflight", return_value=fixture[5]), patch.object(runner, "load_model") as load:
                with self.assertRaisesRegex(ValueError, "artifact SHA-256 mismatch"):
                    runner.prepare_run(*fixture[:5])
                load.assert_not_called()

    def test_missing_clean_id_rejected_even_if_summary_hash_updated(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.fixture(Path(temporary))
            score_path = fixture[0] / "scores.csv"
            score_path.write_text("\n".join(score_path.read_text().splitlines()[:-1]) + "\n")
            summary_path = fixture[0] / "summary.json"
            summary = json.loads(summary_path.read_text())
            summary["output_sha256"]["scores.csv"] = runner.digest(score_path)
            summary_path.write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "Clean score IDs"):
                self.prepare(fixture)

    def test_frozen_partition_and_protocol_changes_rejected(self):
        for target in ("partition", "protocol", "helper"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temporary:
                fixture = self.fixture(Path(temporary))
                audit = json.loads(fixture[2].read_text())
                if target == "partition":
                    audit["calibration"]["utterance_ids"] = []
                elif target == "protocol":
                    audit["protocol_sha256"] = "changed"
                else:
                    audit["source_sha256"] = "changed"
                fixture[2].write_text(json.dumps(audit))
                with self.assertRaisesRegex(ValueError, "Frozen"):
                    self.prepare(fixture)

    def test_eval_configuration_rejected_by_original_preflight(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "eval.toml"
            config.write_text((runner.REPO_ROOT / "configs/pilot.toml").read_text().replace("eval_enabled = false", "eval_enabled = true"))
            with self.assertRaisesRegex(ValueError, "eval_enabled must be false"):
                runner.prepare_run(root / "missingbaseline", config, root / "audit", root / "protocol", root / "out")

    def test_output_existing_rejected_by_original_preflight(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "Output already exists"):
                runner.prepare_run(root, runner.REPO_ROOT / "configs/pilot.toml", root, root, root)

    def test_loaded_model_training_ids_must_match(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.fixture(Path(temporary))
            fixture[6].metadata["training_ids"] = []
            with self.assertRaisesRegex(ValueError, "metadata differs"):
                self.prepare(fixture)

    def test_complete_mocked_pipeline_persists_calibration_before_analysis(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.fixture(Path(temporary))
            output = fixture[4]
            def rank(*args):
                persisted = json.loads((output / "calibration.json").read_text())
                self.assertEqual(persisted["threshold"], args[-1])
                self.assertIn("manifest_sha256", persisted["partition"])
                return evaluate_error_ranking(*args)
            with patch.object(runner, "preflight", return_value=fixture[5]), \
                 patch.object(runner, "load_model", return_value=fixture[6]), \
                 patch.object(runner, "load_audio", return_value=np.zeros(16, dtype=np.float32)) as audio, \
                 patch.object(runner, "narrowband_roundtrip", side_effect=lambda value, **kwargs: value), \
                 patch.object(runner, "extract_lfcc", return_value=np.zeros((1, 60), dtype=np.float32)), \
                 patch.object(runner, "score_features", return_value=0.25) as score, \
                 patch.object(runner, "evaluate_scores", return_value={"eer": 0.2, "min_tDCF": 0.4}) as metric, \
                 patch.object(runner, "evaluate_error_ranking", side_effect=rank), \
                 patch.object(runner, "environment_info", return_value={"synthetic": True}):
                summary = runner.run_experiment(*fixture[:5])
            self.assertEqual(audio.call_count, 1000)
            self.assertEqual(score.call_count, 1000)
            self.assertEqual(metric.call_count, 3)
            self.assertEqual(summary["status"], "ok")
            self.assertEqual(summary["thread_limit"], 1)
            with (output / "paired_scores.csv").open() as source:
                records = list(csv.DictReader(source))
            self.assertEqual(len(records), 1000)
            self.assertEqual({r["group"] for r in records}, {"calibration", "analysis"})
            for name, expected in summary["output_sha256"].items():
                self.assertEqual(runner.digest(output / name), expected)
            self.assertLessEqual(len(summary["failure_cases"]["examples"]), 10)

    def test_failure_examples_report_unavailable_strata(self):
        result = runner.select_failure_examples({"records": [
            {"utterance_id": "LA_D_1", "error": False, "delta": 1, "abs_delta": 1, "negative_margin": -1}]})
        self.assertEqual(result["examples"], [])
        self.assertTrue(all(value["status"] == "not available" for value in result["availability"].values()))


if __name__ == "__main__":
    unittest.main()
