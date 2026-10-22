"""Gate and duration diagnostics checks; no real data fitting."""

import copy
import importlib.util
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("localization_pilot", ROOT / "scripts/run_localization_pilot.py")
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def fixture():
    selection, audit = {"splits": {}, "seed": 45, "dataset": "PartialSpoof v1.2"}, {"training_alignment_ready": True,
        "selection_sha256": "selection", "compact_labels_sha256": "labels", "splits": {}}
    for split, count in (("train", 128), ("dev", 64)):
        rows, records = [], []
        for i in range(count):
            uid = f"{split}_{i}"
            rows.append({"utterance_id": uid, "utterance_class": "bonafide" if i < count // 2 else "spoof"})
            records.append({"utterance_id": uid, "sample_count": 321, "sample_rate": 16000,
                            "channels": 1, "audio_sha256": "audio", "native_count_10ms": 3,
                            "candidate_match": True})
        selection["splits"][split] = {"selected_records": rows}
        audit["splits"][split] = {"records": records}
    selection["splits"]["dev"]["timeline_ids"] = [r["utterance_id"] for r in records[:5] + records[32:37]]
    return selection, audit


class PilotTests(unittest.TestCase):
    def test_gate_rejects_stale_missing_and_mismatched_alignment(self):
        selection, audit = fixture()
        self.assertEqual(len(pilot.validate_gate(selection, audit, "selection", "labels")["train"]), 128)
        for change in (lambda a: a.update(training_alignment_ready=False),
                       lambda a: a.update(selection_sha256="stale"),
                       lambda a: a["splits"]["train"]["records"][0].update(native_count_10ms=2),
                       lambda a: a["splits"]["dev"]["records"].pop()):
            broken = copy.deepcopy(audit)
            change(broken)
            with self.assertRaises(ValueError):
                pilot.validate_gate(selection, broken, "selection", "labels")

    def test_duration_eer_clips_tail_and_crosses_ties_as_group(self):
        # Equal scores cross together: no artificial perfect split within ties.
        tied = pilot.duration_eer([0, 1], [0., 0.], [.02, .001])
        self.assertAlmostEqual(tied["eer_fraction"], .5)
        self.assertFalse(tied["official_segment_eer"])
        # Unequal final widths count as duration, rather than one vote per bin.
        weighted = pilot.duration_eer([0, 0, 1, 1], [-1., 1., 0., 2.], [.019, .001, .001, .019])
        self.assertAlmostEqual(weighted["eer_fraction"], .05)
        self.assertAlmostEqual(weighted["spoof_seconds"], .02)

    def test_duration_eer_rejects_one_class_and_invalid_widths(self):
        for labels, scores, widths in (([1], [0.], [.02]), ([0, 1], [0., 1.], [.02, 0.]),
                                     ([0, 1], [0., np.nan], [.02, .02])):
            with self.assertRaises(ValueError):
                pilot.duration_eer(labels, scores, widths)

    def test_reference_intervals_merge_only_adjacent_spoof_bins(self):
        intervals = pilot.spoof_intervals([0, 0, 1, 0], [0, .02, .04, .06], [.02, .04, .06, .061])
        self.assertEqual(intervals, [(0., .04), (.06, .061)])


if __name__ == "__main__":
    unittest.main()
