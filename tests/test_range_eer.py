"""Run in .cache/partial-metric-env/venv, not the detection environment."""
import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1] / "scripts/check_range_eer.py"
spec = importlib.util.spec_from_file_location("range_conformance", PATH)
conformance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conformance)


class RangeEERConformanceTests(unittest.TestCase):
    def test_actual_upstream_duration_rates_and_main(self):
        audit = conformance.run_checks()
        self.assertEqual(len(audit["cases"]), 13)
        self.assertTrue(all(case["status"] == "passed" for case in audit["cases"]))
        self.assertEqual(audit["original_failure"]["type"], "NameError")
        bounded = audit["bounded_search_variant"]
        self.assertEqual(bounded["stop_reason"], "repeated_full_state")
        self.assertGreater(bounded["iterations"], 5)
        self.assertGreater(bounded["threshold_repeat_count"], 0)
        self.assertAlmostEqual(bounded["range_eer"], (.75 + 2/3)/2)

    def test_source_pin_rejects_changed_source_layout(self):
        original = (conformance.CACHE / "RangeEER.original.py").read_text()
        with self.assertRaises(ValueError):
            conformance.compatibility_patch(original.replace("abs_diffs = np.abs(frr - fpr)", "abs_diffs = np.abs(fnr - fpr)"))

    def test_compatibility_patch_preserves_bisection(self):
        original = (conformance.CACHE / "RangeEER.original.py").read_text()
        patched = conformance.compatibility_patch(original)
        start = original.index("    while(th_left < th_right")
        end = original.index("    if(os.path.exists(save_file))", start)
        self.assertEqual(original[start:end], patched[start:end])


if __name__ == "__main__":
    unittest.main()
