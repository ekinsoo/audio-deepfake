"""Protocol guards and diagnostic metrics without audio or real scores."""

from dataclasses import replace
import unittest

from audio_deepfake.manifests import ManifestRow
from audio_deepfake.reliability import calibration_threshold, evaluate_error_ranking, partition_dev


def row(index, label="bonafide", *, speaker="Dspeaker", split="dev", attack="A01"):
    prefix = {"train": "T", "dev": "D", "eval": "E"}[split]
    utterance = f"LA_{prefix}_{index}"
    return ManifestRow(split, speaker, utterance, "-" if label == "bonafide" else attack,
                       label, f"ASVspoof2019_LA_{split}/flac/{utterance}.flac")


class ReliabilityTests(unittest.TestCase):
    def fixture(self):
        train = [row(1, speaker="Tspeaker", split="train")]
        dev = []
        for speaker in range(20):
            dev.append(row(7 * speaker, speaker=f"D_{speaker}"))
            dev.extend(row(7 * speaker + attack, "spoof", speaker=f"D_{speaker}", attack=f"A{attack:02d}")
                       for attack in range(1, 7))
        return train, dev

    def test_partition_is_order_invariant_and_speaker_disjoint(self):
        train, dev = self.fixture()
        cal, analysis, metadata = partition_dev(train, dev)
        self.assertEqual((cal, analysis, metadata), partition_dev(train[::-1], dev[::-1]))
        self.assertEqual(metadata["calibration"]["speakers"], 8)
        self.assertEqual(metadata["analysis"]["speakers"], 12)
        self.assertFalse({r.speaker_id for r in cal} & {r.speaker_id for r in analysis})
        self.assertEqual(set(r.utterance_id for r in cal + analysis), set(r.utterance_id for r in dev))

    def test_train_dev_leakage_and_eval_are_rejected(self):
        train, dev = self.fixture()
        with self.assertRaisesRegex(ValueError, "Speaker overlap"):
            partition_dev([replace(train[0], speaker_id=dev[0].speaker_id)], dev)
        with self.assertRaises(ValueError):
            partition_dev(train, dev + [row(999, split="eval")])

    def test_missing_attack_or_label_fails_without_alternate_partition(self):
        train, dev = self.fixture()
        with self.assertRaisesRegex(ValueError, "every attack"):
            partition_dev(train, [r for r in dev if r.attack_id != "A06"])
        with self.assertRaisesRegex(ValueError, "both labels"):
            partition_dev(train, [r for r in dev if r.label == "spoof"])

    def test_coverage_is_required_in_each_partition_not_only_overall(self):
        train, dev = self.fixture()
        cal, _, _ = partition_dev(train, dev)
        cal_speakers = {r.speaker_id for r in cal}
        reduced = [r for r in dev if not (r.speaker_id in cal_speakers and r.attack_id == "A06")]
        self.assertIn("A06", {r.attack_id for r in reduced})
        with self.assertRaisesRegex(ValueError, "calibration requires every attack"):
            partition_dev(train, reduced)

    def test_threshold_uses_exact_order_statistic_and_ties(self):
        rows = [row(index) for index in range(40)] + [row(40, "spoof")]
        scores = [(r.utterance_id, -9 if index == 0 else 2 if index < 4 else index)
                  for index, r in enumerate(rows)]
        self.assertEqual(calibration_threshold(rows, scores), 2)
        self.assertEqual(calibration_threshold(rows[::-1], scores[::-1]), 2)
        self.assertEqual(calibration_threshold(rows, scores[:-1] + [(rows[-1].utterance_id, -1e9)]), 2)
        result = evaluate_error_ranking([row(1)], [("LA_D_1", 2)], [("LA_D_1", 2)], 2)
        self.assertEqual(result["records"][0]["predicted_label"], "bonafide")

    def test_join_rejects_missing_extra_duplicate_nonfinite_scores(self):
        rows = [row(1), row(2, "spoof")]
        valid = [("LA_D_1", 1), ("LA_D_2", -1)]
        for invalid in (valid[:1], valid + [("LA_D_3", 0)], valid + [valid[0]],
                        [("LA_D_1", float("nan")), valid[1]]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                calibration_threshold(rows, invalid)
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                evaluate_error_ranking(rows, valid, invalid, 0)

    def test_error_ranking_and_counts(self):
        rows = [row(1), row(2), row(3, "spoof"), row(4, "spoof")]
        clean = [("LA_D_1", 3), ("LA_D_2", -0.1), ("LA_D_3", -3), ("LA_D_4", 0.1)]
        narrow = [("LA_D_1", 3.01), ("LA_D_2", 2), ("LA_D_3", -3.01), ("LA_D_4", -2)]
        result = evaluate_error_ranking(rows, clean, narrow, 0)
        self.assertEqual(result["errors"], 2)
        self.assertEqual(result["error_prevalence"], 0.5)
        for diagnostic in result["rankings"].values():
            self.assertEqual(diagnostic["auroc"], 1)
            self.assertEqual(diagnostic["average_precision"], 1)
        self.assertEqual(result["by_label"]["bonafide"], {"count": 2, "errors": 1, "correct": 1})
        self.assertEqual(result["by_attack_id"]["A01"]["errors"], 1)

    def test_one_error_class_is_undefined(self):
        rows = [row(1), row(2, "spoof")]
        for clean, reason in (([("LA_D_1", 1), ("LA_D_2", -1)], "No errors"),
                              ([("LA_D_1", -1), ("LA_D_2", 1)], "No correct predictions")):
            result = evaluate_error_ranking(rows, clean, clean, 0)
            for metric in result["rankings"].values():
                self.assertIsNone(metric["auroc"])
                self.assertIsNone(metric["average_precision"])
                self.assertEqual(metric["undefined_reason"], reason)


if __name__ == "__main__":
    unittest.main()
