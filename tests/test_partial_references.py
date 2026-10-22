"""Derived references preserve binary truth and clipped terminal bins."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('partial_refs', Path(__file__).resolve().parents[1]/'scripts/prepare_partial_references.py')
refs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(refs)


class ReferenceTests(unittest.TestCase):
    def test_exact_native_runs_and_clipped_terminal_interval(self):
        self.assertEqual(refs.spoof_ranges([0, 0, 1, 0], .035), [(0, .02), (.03, .035)])

    def test_bonafide_has_no_spoof_and_all_spoof_keeps_tail(self):
        self.assertEqual(refs.spoof_ranges([1, 1], .016), [])
        self.assertEqual(refs.spoof_ranges([0, 0], .016), [(0, .016)])

    def test_rejects_repaired_unknown_or_nonfinite_reference(self):
        for labels, duration in [([0], .021), ([0, 2], .02), ([False], .01), ([0], float('nan')), ([], .01)]:
            with self.assertRaises(ValueError):
                refs.spoof_ranges(labels, duration)


if __name__ == '__main__':
    unittest.main()
