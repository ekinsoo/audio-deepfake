"""Runner provenance and held-out split guards, using generated audio fixtures."""

import copy
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from audio_deepfake.manifests import ManifestRow, write_manifest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_baseline", ROOT / "scripts/run_baseline.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.audio = self.root / "audio"
        self.manifest = self.root / "selection.csv"
        self.journal = self.root / "journal.json"
        self.output = self.root / "output"
        self.asv, self.reference = self.root / "asv.txt", self.root / "evaluator.py"
        self.asv.write_text("fixture ASV", encoding="utf-8")
        self.reference.write_text("fixture evaluator", encoding="utf-8")
        self.rows, records = [], {}
        for split, prefix in (("train", "T"), ("dev", "D")):
            for index in range(4):
                uid = f"LA_{prefix}_{index}"
                label = "bonafide" if index < 2 else "spoof"
                relative = f"ASVspoof2019_LA_{split}/flac/{uid}.flac"
                row = ManifestRow(split, f"speaker_{split}", uid,
                                   "-" if label == "bonafide" else "A01", label, relative)
                self.rows.append(row)
                path = self.audio / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(path, np.linspace(-0.2, 0.2, 640), 16000)
                records[relative] = {"sha256": runner.digest(path), "size_bytes": path.stat().st_size}
        write_manifest(self.rows, self.manifest)
        self.journal.write_text(json.dumps({"records": records}), encoding="utf-8")
        self.config = {
            "experiment": {"seed": 42, "stage": "smoke"},
            "data": {"train_limit": 4, "dev_limit": 4, "eval_enabled": False,
                     "dataset": "ASVspoof2019-LA", "version": "Zenodo-6906306-v1", "sample_rate": 16000},
            "model": {"n_components": 2, "max_total_training_frames": 40, "max_iter": 5},
            "features": {"sample_rate": 16000, "n_coeffs": 2, "delta_order": 0},
            "paths": {"selection_manifest": str(self.manifest), "audio_root": str(self.audio),
                      "audio_journal": str(self.journal), "selection_sha256": runner.digest(self.manifest),
                      "asv_scores": str(self.asv), "reference_evaluator": str(self.reference)},
        }
        # Tiny fixtures replace only the expected digests; production constants stay pinned.
        self.asv_patch = patch.object(runner, "DEV_ASV_SHA256", runner.digest(self.asv))
        self.ref_patch = patch.object(runner, "REFERENCE_SHA256", runner.digest(self.reference))
        self.asv_patch.start()
        self.ref_patch.start()
        self.addCleanup(self.asv_patch.stop)
        self.addCleanup(self.ref_patch.stop)

    def update_manifest(self, rows):
        write_manifest(rows, self.manifest)
        self.config["paths"]["selection_sha256"] = runner.digest(self.manifest)

    def test_complete_preflight(self):
        plan = runner.preflight(self.config, self.output)
        self.assertEqual(len(plan["audio_hashes"]), 8)
        self.assertEqual(len(plan["groups"]["train"]), 4)
        self.assertEqual(plan["gmm"].frames_per_class, 20)
        self.assertFalse(self.output.exists())

    def test_regularization_and_tolerance_are_used(self):
        self.config["model"].update(reg_covar=1e-4, tol=0.01)
        plan = runner.preflight(self.config, self.output)
        self.assertEqual(plan["gmm"].reg_covar, 1e-4)
        self.assertEqual(plan["gmm"].tol, 0.01)

    def test_selection_and_output_guards(self):
        self.config["paths"]["selection_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "pinned selection"):
            runner.preflight(self.config, self.output)
        self.output.mkdir()
        with self.assertRaisesRegex(ValueError, "already exists"):
            runner.preflight({}, self.output)

    def test_asv_and_reference_mismatch(self):
        self.asv.write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "pinned official LA dev"):
            runner.preflight(self.config, self.output)
        self.asv.write_text("fixture ASV", encoding="utf-8")
        self.reference.write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Evaluator differs"):
            runner.preflight(self.config, self.output)

    def test_missing_and_changed_audio_journal(self):
        path = self.audio / self.rows[0].audio_path
        original = path.read_bytes()
        path.unlink()
        with self.assertRaisesRegex(ValueError, "Missing selected audio"):
            runner.preflight(self.config, self.output)
        path.write_bytes(original + b"changed")
        with self.assertRaisesRegex(ValueError, "differs from download journal"):
            runner.preflight(self.config, self.output)
        path.write_bytes(original)
        journal = json.loads(self.journal.read_text())
        del journal["records"][self.rows[0].audio_path]
        self.journal.write_text(json.dumps(journal))
        with self.assertRaisesRegex(ValueError, "Missing download journal entry"):
            runner.preflight(self.config, self.output)

    def test_eval_and_speaker_leak_and_imbalance(self):
        uid = "LA_E_100"
        eval_row = ManifestRow("eval", "speaker_eval", uid, "-", "bonafide",
                                f"ASVspoof2019_LA_eval/flac/{uid}.flac")
        self.update_manifest(self.rows + [eval_row])
        with self.assertRaisesRegex(ValueError, "eval is excluded"):
            runner.preflight(self.config, self.output)
        self.update_manifest([replace(r, speaker_id="shared") for r in self.rows])
        with self.assertRaisesRegex(ValueError, "Speaker overlap"):
            runner.preflight(self.config, self.output)
        self.update_manifest(self.rows[:-1])
        with self.assertRaisesRegex(ValueError, "balanced records"):
            runner.preflight(self.config, self.output)

    def test_unsafe_experiment_settings(self):
        for section, key, value in (("data", "eval_enabled", True), ("experiment", "stage", "eval"),
                                    ("model", "max_total_training_frames", 41),
                                    ("data", "train_limit", 3), ("model", "typo", 10),
                                    ("data", "dataset", "other"), ("data", "version", "other"),
                                    ("data", "sample_rate", 8000), ("features", "sample_rate", 8000)):
            config = copy.deepcopy(self.config)
            config[section][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.preflight(config, self.output)

    def test_run_exports_public_reproducibility_artifacts(self):
        config_path = self.root / "config.toml"
        sections = []
        for section, settings in self.config.items():
            sections.append(f"[{section}]\n" + "\n".join(f"{key} = {json.dumps(value)}" for key, value in settings.items()))
        config_path.write_text("\n\n".join(sections), encoding="utf-8")
        scores = [(r.utterance_id, float(index)) for index, r in enumerate(self.rows) if r.split == "dev"]
        model = SimpleNamespace(metadata={"test": "fixture"})
        def fake_save(model, path):
            path.write_bytes(b"trusted local fixture artifact")
        with patch.object(runner, "train_baseline", return_value=model) as training, \
             patch.object(runner, "score_rows", return_value=scores), \
             patch.object(runner, "evaluate_scores", return_value={"eer": 0.25, "min_tDCF": 0.5}), \
             patch.object(runner, "save_model", side_effect=fake_save):
            summary = runner.run_experiment(config_path, self.output)
        self.assertEqual(training.call_args.args[0], self.rows[:4])
        self.assertEqual(summary["scope"], "smoke")
        self.assertIn("scripts/run_baseline.py", summary["source_sha256"])
        self.assertEqual(summary["workers"], 1)
        for filename in ("summary.json", "config.json", "scores.csv", "model.joblib"):
            self.assertTrue((self.output / filename).is_file())
        for filename in ("summary.json", "config.json"):
            self.assertNotIn(str(self.root), (self.output / filename).read_text())
        with patch.object(runner, "train_baseline") as training:
            with self.assertRaisesRegex(ValueError, "already exists"):
                runner.run_experiment(config_path, self.output)
            training.assert_not_called()


if __name__ == "__main__":
    unittest.main()
