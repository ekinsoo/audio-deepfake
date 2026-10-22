"""Validate public diagnostics, score integrity and threshold boundaries."""

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("analyze_baseline", Path(__file__).resolve().parents[1] / "scripts/analyze_baseline.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "audit.json"
        self.records = [
            ("LA_D_1", "dev", "bonafide", "-", -1),
            ("LA_D_2", "dev", "bonafide", "-", 1),
            ("LA_D_3", "dev", "spoof", "A01", 0),
            ("LA_D_4", "dev", "spoof", "A01", -2),
        ]
        self.summary = {"status": "ok", "scope": "pilot",
                        "splits": {"dev": {"count": 4, "attacks": {"-": 2, "A01": 2}}},
                        "metrics": {"eer_threshold": 0, "cm_counts": {"bonafide": 2, "spoof": 2}},
                        "model_metadata": {"training_ids": ["LA_T_1"]}, "output_sha256": {}}
        self.write_run()

    def write_run(self):
        path = self.root / "scores.csv"
        with path.open("w", newline="", encoding="utf-8") as target:
            writer = csv.writer(target)
            writer.writerow(("utterance_id", "split", "label", "attack_id", "score"))
            writer.writerows(self.records)
        self.summary["output_sha256"]["scores.csv"] = hashlib.sha256(path.read_bytes()).hexdigest()
        (self.root / "summary.json").write_text(json.dumps(self.summary), encoding="utf-8")

    def test_literal_threshold_counts_boundary_as_accepted(self):
        audit = module.export_audit(self.root, self.output)
        errors = audit["error_analysis"]
        self.assertEqual(errors["by_attack"], {"-": {"count": 2, "errors": 1}, "A01": {"count": 2, "errors": 1}})
        self.assertEqual([case["utterance_id"] for case in errors["failure_cases"]], ["LA_D_1", "LA_D_3"])
        self.assertEqual(errors["failure_cases"][1]["error"], "spoof_accepted")
        with self.assertRaisesRegex(ValueError, "already exists"):
            module.export_audit(self.root, self.output)

    def test_modified_scores_fail_before_export(self):
        with (self.root / "scores.csv").open("a") as target:
            target.write("unexpected\n")
        with self.assertRaisesRegex(ValueError, "hash"):
            module.export_audit(self.root, self.output)
        self.assertFalse(self.output.exists())

    def test_invalid_dev_contracts_fail(self):
        for records in (
            self.records[:-1],
            self.records[:3] + [self.records[0]],
            [(*self.records[0][:4], float("nan")), *self.records[1:]],
            [("LA_T_1", *self.records[0][1:]), *self.records[1:]],
            [("LA_D_1", "eval", "bonafide", "-", -1), *self.records[1:]],
            [("LA_D_1", "dev", "bonafide", "A01", -1), *self.records[1:]],
        ):
            with self.subTest(records=records):
                self.records = records
                self.write_run()
                with self.assertRaises(ValueError):
                    module.export_audit(self.root, self.output)
                self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
