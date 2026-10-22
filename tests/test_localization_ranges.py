"""Independent native-duration counting keeps scale, clipping and ties explicit."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('range_adapter', Path(__file__).resolve().parents[1]/'scripts/evaluate_localization_ranges.py')
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


class NativeDurationTests(unittest.TestCase):
    def test_native10ms_truth_differs_from_min_pooled20ms(self):
        rates = adapter.direct_rates({'u': {'scores': [.1, .9, .1]}},
                                     {'u': [0,0,0,1,1,1]}, {'u': .06}, .5)
        self.assertAlmostEqual(rates['fnr'], 1/3)
        self.assertAlmostEqual(rates['fpr'], 2/3)

    def test_terminal_clip_and_upstream_equality_spoof(self):
        rates = adapter.direct_rates({'u': {'scores': [.5, .9, .5]}},
                                     {'u': [0,0,0,1,1,1]}, {'u': .055}, .5)
        self.assertAlmostEqual(rates['spoof_seconds'], .03)
        self.assertAlmostEqual(rates['bonafide_seconds'], .025)
        self.assertAlmostEqual(rates['false_alarm_seconds'], .015)
        self.assertAlmostEqual(rates['fpr'], .6)


if __name__ == '__main__':
    unittest.main()
