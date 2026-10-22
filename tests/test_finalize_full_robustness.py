import importlib.util
from pathlib import Path
import unittest

from audio_deepfake.manifests import ManifestRow

spec = importlib.util.spec_from_file_location("finalize_full_robustness", Path(__file__).resolve().parents[1] / "scripts/finalize_full_robustness.py")
finalizer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(finalizer)


class FinalizerTests(unittest.TestCase):
    def fixture(self):
        rows = [ManifestRow("dev", "LA_01", "LA_D_1", "-", "bonafide", "a"),
                ManifestRow("dev", "LA_02", "LA_D_2", "A01", "spoof", "b")]
        records = []
        for index, row in enumerate(rows):
            clean, narrow = 1.0 + index, -2.0 - index
            records.append({name: getattr(row, name) for name in finalizer.METADATA} |
                           {"s_clean": clean, "s_narrow": narrow, "s_average": clean / 2 + narrow / 2,
                            "delta": narrow - clean, "abs_delta": abs(narrow - clean),
                            "group": "calibration" if index == 0 else "analysis"})
        return rows, records, {row.utterance_id: record["s_clean"] for row, record in zip(rows, records)}

    def test_exact_join_works_and_reversed_duplicate_missing_rows_fail(self):
        rows, records, clean = self.fixture()
        self.assertEqual(finalizer.validate_paired_rows(records, rows, clean, {"LA_D_1"}), records)
        for corrupt in (records[::-1], records[:1], [records[0], records[0]]):
            with self.subTest(corrupt=corrupt), self.assertRaises(ValueError):
                finalizer.validate_paired_rows(corrupt, rows, clean, {"LA_D_1"})

    def test_corrupt_metadata_derived_values_clean_group_and_nonfinite_rejected(self):
        rows, records, clean = self.fixture()
        corruptions = {"speaker_id": "other", "label": "spoof", "s_clean": 1.00000001,
                       "s_average": 0.0, "delta": 3.0, "abs_delta": -3.0, "group": "analysis",
                       "s_narrow": float("nan")}
        for name, value in corruptions.items():
            corrupt = [dict(record) for record in records]
            corrupt[0][name] = value
            with self.subTest(field=name), self.assertRaises(ValueError):
                finalizer.validate_paired_rows(corrupt, rows, clean, {"LA_D_1"})

    def test_complete_aggregate_population_std(self):
        result = finalizer.aggregate_values([1.0, 2.0, 3.0])
        self.assertEqual(result["mean"], 2)
        self.assertAlmostEqual(result["std"], (2 / 3) ** .5)
        self.assertTrue(result["complete"])
        self.assertEqual(result["defined_seed_count"], 3)

    def test_undefined_seed_marks_aggregate_incomplete(self):
        result = finalizer.aggregate_values([None, 1.0, 3.0])
        self.assertEqual((result["mean"], result["std"]), (2, 1))
        self.assertFalse(result["complete"])
        self.assertEqual(result["defined_seed_count"], 2)
        absent = finalizer.aggregate_values([None, None, None])
        self.assertIsNone(absent["mean"])
        self.assertIsNone(absent["std"])
        self.assertEqual(absent["defined_seed_count"], 0)

    def test_nonfinite_or_selected_seed_aggregate_fails(self):
        for values in ([1, 2], [1, 2, float("nan")], [1, 2, True]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                finalizer.aggregate_values(values)

    def test_supervision_reconciles_stages_and_rejects_modified_wall_budget(self):
        chunks = {str(seed): [{"file": f"chunks/seed{seed}_00000_00002.json"}] for seed in finalizer.SEEDS}
        names = ["preflight", "smoke"] + [f"seed{seed}_00000_00002" for seed in finalizer.SEEDS]
        stages = [{"name": name, "status": "complete", "exit_code": 0, "elapsed_seconds": 2.0,
                   "budget_seconds": 1800, "peak_rss_bytes": 100,
                   "last_memory": {"system_available_ram_bytes": 2 * 1024**3, "child_peak_rss_bytes": 100}}
                  for name in names]
        preflight = {"execution_source_sha256": {"runner": "pin"}}
        supervision = {"status": "scoring_complete", "source_sha256": preflight["execution_source_sha256"],
                       "stages": stages, "seed_wall_seconds": {str(seed): 3.0 for seed in finalizer.SEEDS},
                       "seed_smoke_seconds": {str(seed): 1.0 for seed in finalizer.SEEDS}}
        smoke_seconds = supervision["seed_smoke_seconds"]
        finalizer.validate_supervision(supervision, preflight, chunks, smoke_seconds)
        supervision["seed_wall_seconds"]["42"] = 1.0
        with self.assertRaisesRegex(ValueError, "wall time"):
            finalizer.validate_supervision(supervision, preflight, chunks, smoke_seconds)
        supervision["seed_wall_seconds"]["42"] = 3.0
        stages[2]["last_memory"]["system_available_ram_bytes"] = 100
        with self.assertRaisesRegex(ValueError, "memory guard"):
            finalizer.validate_supervision(supervision, preflight, chunks, smoke_seconds)


if __name__ == "__main__":
    unittest.main()
