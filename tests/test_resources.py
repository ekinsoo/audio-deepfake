"""Native resource probes must remain bounded across long experiment loops."""

import ctypes
import gc
import os
import unittest
from unittest.mock import patch

from audio_deepfake import resources


@unittest.skipUnless(os.name == "nt", "Windows native memory probes")
class WindowsMemoryTests(unittest.TestCase):
    def test_snapshot_reports_current_and_peak_working_set(self):
        value = resources.memory_snapshot()
        self.assertEqual(set(value), {"method", "rss_bytes", "peak_rss_bytes",
                                     "system_total_ram_bytes", "system_available_ram_bytes"})
        self.assertGreater(value["rss_bytes"], 0)
        self.assertGreaterEqual(value["peak_rss_bytes"], value["rss_bytes"])
        self.assertGreater(value["system_available_ram_bytes"], 0)
        self.assertLessEqual(value["system_available_ram_bytes"], value["system_total_ram_bytes"])

    def test_repeated_probes_do_not_retain_new_pointer_types(self):
        resources.memory_snapshot()
        gc.collect()
        initial_types = dict(ctypes._pointer_type_cache)
        for _ in range(2000):
            resources.memory_snapshot()
        gc.collect()
        self.assertEqual(ctypes._pointer_type_cache, initial_types)

    def test_failed_native_measurement_raises_instead_of_reporting_zero(self):
        with patch.object(resources.psapi, "GetProcessMemoryInfo", return_value=False):
            with self.assertRaises(OSError):
                resources.memory_snapshot()
        with patch.object(resources.kernel, "GlobalMemoryStatusEx", return_value=False):
            with self.assertRaises(OSError):
                resources.memory_snapshot()


if __name__ == "__main__":
    unittest.main()
