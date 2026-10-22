import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("errors", Path(__file__).resolve().parents[1] / "scripts/analyze_localization_errors.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class ErrorTests(unittest.TestCase):
    def test_native_vad_intersection_keeps_binary_truth_and_tail(self):
        row = {"scores": [-1, 1], "bin_starts": [0, .02], "bin_ends": [.02, .035]}
        values = m.native_vad_stats(row, [1, 0, 0, 1], [(0, .015, 2), (.015, .03, 1)], .035)
        self.assertAlmostEqual(values["nonspeech"]["false_alarm_seconds"], .01)
        self.assertAlmostEqual(values["nonspeech"]["spoof_seconds"], .005)
        self.assertAlmostEqual(values["speech"]["missed_spoof_seconds"], .01)
        self.assertAlmostEqual(values["vad_uncovered"]["bonafide_seconds"], .005)
        self.assertAlmostEqual(sum(v["spoof_seconds"] + v["bonafide_seconds"] for v in values.values()), .035)

    def test_fragmentation_and_no_overlap_are_distinct(self):
        v = m.fragment_stats([(0, .1), (.2, .3)], [(0, .02), (.04, .08), (.4, .5)])
        self.assertEqual(v["reference_runs_with_no_overlap"], 1)
        self.assertEqual(v["reference_runs_split_over_multiple_predictions"], 1)
        self.assertEqual(v["predicted_runs_with_no_reference_overlap"], 1)

    def test_duration_groups_endpoints(self):
        self.assertEqual(m.duration_group(.1), "up_to_100ms")
        self.assertEqual(m.duration_group(.1001), "over_100_to_500ms")
        self.assertEqual(m.duration_group(.5), "over_100_to_500ms")
        self.assertEqual(m.duration_group(.501), "over_500ms")

    def test_equality_is_bonafide_and_no_truth_denominator_is_null(self):
        stats = m.empty()
        m.add(stats, 0, 0 < 0, .02)
        self.assertEqual(m.finish(stats)["miss_fraction"], 1)
        self.assertIsNone(m.finish(stats)["false_alarm_fraction"])

    def test_vad_gaps_rejected(self):
        with self.assertRaises(ValueError):
            m.native_vad_stats({"scores": [1], "bin_starts": [0], "bin_ends": [.02]}, [1, 1], [(0, .005, 1), (.01, .02, 2)], .02)


if __name__ == "__main__":
    unittest.main()
