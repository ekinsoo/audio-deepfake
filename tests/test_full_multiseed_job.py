"""Safety and aggregation checks; never read audio, fit or launch workers."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import tomllib
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import full_multiseed_job as job


def report(value, converged=True):
    return dict(status="ok", metrics=dict(eer=value, min_tDCF=value, split="dev",
                reference_sha256="93ea505282d329ed14537d977a74b2d1a25d161ac6767a16e0216a1a18391049",
                asv_score_sha256="fe8114cf608c3062c141df5979892eba9664aa37a8c99e97157f129d679d3389",
                rate_units="fraction", score_direction="higher_is_bonafide",
                cm_counts={"bonafide":2548,"spoof":22296}),
                model_metadata=dict(classes={label: dict(converged=converged, iterations=30, warnings=[])
                                             for label in ("bonafide", "spoof")}))


class MultiseedTests(unittest.TestCase):
    def test_public_stages_exclude_local_commands_paths_and_pid(self):
        private = dict(name="seed43", status="complete", budget_seconds=1800,
                       elapsed_seconds=600, peak_rss_bytes=200000000, exit_code=0,
                       started_utc="start", ended_utc="end", last_memory={"child_peak_rss_bytes":200000000},
                       command=["C:\\Users\\private\\python.exe", "--config"], pid=1234,
                       unrelated_private_path="C:\\Users\\private\\output")
        exported = job.public_stages([private])
        self.assertNotIn("command", exported[0])
        self.assertNotIn("pid", exported[0])
        self.assertNotIn("C:\\", json.dumps(exported))
        self.assertNotIn("private", json.dumps(exported))
        self.assertEqual(exported[0]["elapsed_seconds"], 600)
        self.assertEqual(exported[0]["last_memory"], {"child_peak_rss_bytes":200000000})
        self.assertIn("command", private)

    def test_aggregate_population_std_and_preserve_nonconvergence(self):
        result = job.aggregate({42: report(0.1), 43: report(0.2, False), 44: report(0.3)})
        self.assertAlmostEqual(result["metrics"]["eer"]["mean"], 0.2)
        self.assertAlmostEqual(result["metrics"]["eer"]["std"], (0.02/3)**0.5)
        self.assertEqual(result["std_ddof"], 0)
        self.assertEqual(result["convergence_failures"], [43])
        self.assertFalse(result["eval_enabled"])

    def test_no_partial_aggregate_or_nonfinite_metrics(self):
        with self.assertRaises(RuntimeError):
            job.aggregate({42: report(0.1), 43: report(0.2)})
        with self.assertRaises(RuntimeError):
            job.aggregate({42: report(0.1), 43: report(float("nan")), 44: report(0.3)})
        failed = report(0.2); failed["status"] = "failed"
        with self.assertRaises(RuntimeError):
            job.aggregate({42: report(0.1), 43: failed, 44: report(0.3)})

    def test_aggregate_rejects_metric_protocol_drift(self):
        altered = report(0.2); altered["metrics"]["score_direction"] = "higher_is_spoof"
        with self.assertRaisesRegex(RuntimeError, "Metric protocol"):
            job.aggregate({42: report(0.1), 43: altered, 44: report(0.3)})

    def test_configs_differ_only_by_fixed_seed(self):
        job.validate_configs()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root / "configs").mkdir()
            for seed in (42, 43, 44):
                source = (job.ROOT / f"configs/full_seed{seed}.toml").read_text()
                if seed == 44:
                    source = source.replace("n_components = 32", "n_components = 64")
                (root / f"configs/full_seed{seed}.toml").write_text(source)
            with patch.object(job, "ROOT", root), self.assertRaises(RuntimeError):
                job.validate_configs()

    def test_conflicting_lock_and_existing_outputs_rejected(self):
        for name in ("outputs/jobs/full-seed42.lock", "outputs/jobs/scoring-recovery-seed42.lock",
                     "outputs/baseline-full-seed43", "outputs/baseline-full-seed44", job.AUDIT):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                root = Path(temp); target = root / name
                target.parent.mkdir(parents=True); target.write_text("preserve")
                with patch.object(job, "ROOT", root), self.assertRaises(RuntimeError):
                    job.reject_existing()
                self.assertEqual(target.read_text(), "preserve")

    def test_missing_prior_review_stops_before_worker(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); directory = root / "outputs/jobs/full-multiseed-test"
            with patch.object(job, "ROOT", root), patch.object(job, "validate_configs"), \
                    patch.object(job, "validate_prior", side_effect=RuntimeError("review missing")), \
                    patch.object(job.supervisor, "run_stage") as stage:
                self.assertEqual(job.main(["--job-dir", str(directory)]), 1)
                stage.assert_not_called()
                status = json.loads((directory / "status.json").read_text())
                self.assertIn("review missing", status["reason"])
                self.assertFalse((root / "outputs/jobs/full-multiseed.lock").exists())

    def test_existing_job_status_is_immutable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); directory = root / "outputs/jobs/full-multiseed-test"
            directory.mkdir(parents=True); status = directory / "status.json"
            status.write_text("original")
            with patch.object(job, "ROOT", root), self.assertRaises(SystemExit):
                job.main(["--job-dir", str(directory)])
            self.assertEqual(status.read_text(), "original")

    def test_prior_independent_review_must_match_artifacts(self):
        import scoring_recovery_job
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "configs").mkdir()
            (root / "configs/full_seed42.toml").write_bytes((job.ROOT / "configs/full_seed42.toml").read_bytes())
            summary = report(0.1)
            summary["configuration"] = tomllib.loads((root / "configs/full_seed42.toml").read_text())
            summary["output_sha256"] = {"scores.csv": "score-pin"}
            files = {
                "outputs/scoring-recovery-seed42/summary.json": summary,
                "docs/validation/full_baseline_seed42.json": dict(status="ok", model_sha256=job.MODEL_SHA,
                    metrics=summary["metrics"], eval_enabled=False, output_sha256=summary["output_sha256"]),
                "docs/validation/full_independent_review.json": dict(status="ok", seed=42,
                    model_sha256=job.MODEL_SHA, summary_sha256="wrong", score_sha256="score-pin", eval_enabled=False),
            }
            for name, value in files.items():
                target = root / name; target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(value))
            with patch.object(job, "ROOT", root), patch.object(job, "digest", return_value="actual-pin"), \
                    patch.object(scoring_recovery_job, "validate_completion"), self.assertRaisesRegex(RuntimeError, "Independent"):
                job.validate_prior()

    def test_new_completion_rejects_eval_and_wrong_seed_before_artifact_read(self):
        for value in (dict(status="ok", experiment=dict(seed=42)),
                      dict(status="ok", experiment=dict(seed=43), configuration=dict(data=dict(eval_enabled=True)))):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                job.validate_new(value, Path("nonexistent"), 43)

    def test_failed_first_seed_never_launches_second_or_writes_aggregate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); directory = root / "outputs/jobs/full-multiseed-test"
            with patch.object(job, "ROOT", root), patch.object(job, "validate_configs"), \
                    patch.object(job, "validate_prior", return_value=report(0.1)), \
                    patch.object(job, "source_hashes", return_value={}), \
                    patch.object(job.supervisor, "wait_for_resources"), \
                    patch.object(job.supervisor, "worker_command", return_value=["fixed-worker"]), \
                    patch.object(job.supervisor, "run_stage", side_effect=RuntimeError("resource stop")) as stage:
                self.assertEqual(job.main(["--job-dir", str(directory)]), 1)
                self.assertEqual(stage.call_count, 1)
                self.assertEqual(stage.call_args.kwargs["name"], "seed43")
                self.assertEqual(stage.call_args.kwargs["budget_seconds"], 1800)
                self.assertFalse((root / job.AUDIT).exists())
                self.assertFalse((root / "reports.md").exists())


if __name__ == "__main__":
    unittest.main()
