"""Run portable unit tests; external metric/process checks are opt-in."""
import argparse
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
INTEGRATION_MODULES = {'test_range_eer', 'test_ssl_worker_process_monitor'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=('unit', 'range', 'process'), default='unit',
                        help='unit: portable checks; range: isolated upstream evaluator; process: Windows audited venv')
    args = parser.parse_args()
    loader = unittest.TestLoader()
    # Exclude integration modules before import: a clean checkout has no
    # downloaded upstream evaluator or dedicated metric environment.
    suite = unittest.TestSuite()
    sys.path.insert(0, str(ROOT / 'tests'))
    for path in sorted((ROOT / 'tests').glob('test_*.py')):
        if args.suite == 'unit' and path.stem in INTEGRATION_MODULES:
            continue
        if args.suite == 'range' and path.stem != 'test_range_eer':
            continue
        if args.suite == 'process' and path.stem != 'test_ssl_worker_process_monitor':
            continue
        suite.addTests(loader.loadTestsFromName(path.stem))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
