"""Recovery never gets a new seed budget or starts original training."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import full_robustness_recovery_job as job


class RecoveryJobTests(unittest.TestCase):
    def test_failed_attempt_counts_against_total_seed_budget(self):
        stages = [dict(name="recovery_preflight", elapsed_seconds=600),
                  dict(name="seed44_00000_02000", elapsed_seconds=100)]
        self.assertEqual(job.remaining_budget(stages, 95), 1605)
        with self.assertRaisesRegex(RuntimeError, "including failed attempt"):
            job.remaining_budget(stages, 1700)

    def test_audit_bootstrap_uses_actual_python(self):
        with patch.object(job.supervisor, "base_python", return_value="actual-python"):
            command = job.audit_command("verify")
        self.assertEqual(command[:3], ["actual-python", "-S", "-u"])
        self.assertIn("--audit-worker", command)
        self.assertNotIn("run_full_baseline.py", " ".join(command))
