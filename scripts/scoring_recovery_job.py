"""Supervise one scoring-only recovery; never invoke fitting or extraction setup."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import sys

import full_run_job as supervisor

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/scoring-recovery-seed42"
SOURCES = ("scripts/scoring_recovery_job.py", "scripts/start_scoring_recovery.ps1", "scripts/recover_full_scoring.py",
           "scripts/full_run_job.py", "scripts/run_full_baseline.py",
           "src/audio_deepfake/resources.py", "src/audio_deepfake/baseline.py",
           "src/audio_deepfake/audio.py", "src/audio_deepfake/features.py",
           "src/audio_deepfake/manifests.py", "src/audio_deepfake/metrics.py",
           "src/audio_deepfake/cli.py",
           "configs/full_seed42.toml")


def source_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}


def stages():
    """Fixed remaining official dev range; zero-based, stop exclusive."""
    yield "preflight", ["--mode", "preflight"]
    for start in range(22815, 24844, 1000):
        stop = min(start + 1000, 24844)
        yield f"score-{start}-{stop}", ["--mode", "chunk", "--start", str(start), "--stop", str(stop)]
    yield "finalize", ["--mode", "finalize"]


def worker_command(arguments):
    return [supervisor.base_python(), "-S", "-u", str(Path(__file__).resolve()),
            "--worker", *arguments, "--output-dir", str(OUTPUT)]


def validate_completion(report, output):
    if (report.get("status") != "ok"
            or report.get("scope") != "full-official-train-dev-adaptation-scoring-recovery"
            or report.get("configuration", {}).get("data", {}).get("eval_enabled") is not False
            or report.get("model_sha256") != "701fc3d8107cc8b2696b5552dd9e1b4751e44d2d197cfd5d1dfe7e4814add101"):
        raise RuntimeError("Unexpected recovery completion provenance")
    for name in ("scores.csv", "plan.json"):
        if hashlib.sha256((output / name).read_bytes()).hexdigest() != report.get("output_sha256", {}).get(name):
            raise RuntimeError(f"Recovery artifact hash mismatch: {name}")
    protocol = ROOT / "data/metadata/asvspoof2019/ASVspoof2019.LA.cm.dev.trl.txt"
    if hashlib.sha256(protocol.read_bytes()).hexdigest() != "2fedaebf1ba3ae1a9bb7d3ec916804ac237b4eee063a27fa92e6549f03d129c7":
        raise RuntimeError("Official dev protocol hash mismatch")
    expected = [line.split() for line in protocol.read_text().splitlines() if line.strip()]
    with (output / "scores.csv").open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != ["utterance_id", "split", "label", "attack_id", "score"]:
            raise RuntimeError("Unexpected recovery score header")
        rows = list(reader)
    if len(rows) != 24844 or len(expected) != 24844 or len({row["utterance_id"] for row in rows}) != 24844:
        raise RuntimeError("Recovery requires 24844 unique dev scores")
    for row, fields in zip(rows, expected):
        if (row["utterance_id"] != fields[1] or row["attack_id"] != fields[3]
                or row["label"] != fields[4] or row["split"] != "dev"
                or not math.isfinite(float(row["score"]))):
            raise RuntimeError("Recovery scores differ from official dev order/metadata")
    if report.get("metrics", {}).get("cm_counts") != {"bonafide": 2548, "spoof": 22296}:
        raise RuntimeError("Recovery must score complete official dev counts")
    if any(not math.isfinite(report["metrics"][name]) for name in ("eer", "min_tDCF")):
        raise RuntimeError("Recovery metrics must be finite")


def record_completed_result(report, job_dir, status):
    """Save a small public audit only after independent completion checks pass."""
    audit_path = ROOT / "docs/validation/full_baseline_seed42.json"
    if audit_path.exists():
        raise RuntimeError("Public full baseline audit already exists; preserve and review")
    audit = {"status": "ok", "scope": report["scope"], "seed": 42,
             "eval_enabled": False, "finished_utc": report["finished_utc"],
             "splits": report["splits"], "metrics": report["metrics"],
             "model_sha256": report["model_sha256"],
             "input_sha256": report["input_sha256"], "output_sha256": report["output_sha256"],
             "source_sha256": report["source_sha256"],
             "supervisor_source_sha256": status["source_sha256"],
             "recovery": report["recovery"], "limitations": report["limitations"],
             "completion_check": "Separate supervisor verified score hashes, exact official dev order, metadata and finite scores.",
             "local_summary": "outputs/scoring-recovery-seed42/summary.json",
             "local_job": job_dir.relative_to(ROOT).as_posix()}
    supervisor.write_json(audit_path, audit)
    with (ROOT / "reports.md").open("a", encoding="utf-8") as target:
        target.write("\n\n## Completed full dev scoring recovery\n\n"
                     f"Finished: {report['finished_utc']}. The saved seed 42 model now has all "
                     "24,844 official dev scores. No retraining or eval scoring was performed. "
                     f"Dev EER: {report['metrics']['eer'] * 100:.4f}%; minimum normalized t-DCF: "
                     f"{report['metrics']['min_tDCF']:.6f}. These are development results for "
                     "the Python LFCC-GMM adaptation, not held-out benchmark claims.\n\n"
                     "The supervisor independently checked the complete score artifact against "
                     "the pinned official dev protocol. Recovery kept the failed original run "
                     "and model unchanged. Timings describe recovery stages only. "
                     "See [the compact audit](docs/validation/full_baseline_seed42.json) and "
                     "[recovery guide](docs/SCORING_RECOVERY.md).\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    directory = args.job_dir.resolve()
    jobs = (ROOT / "outputs/jobs").resolve()
    if directory.parent != jobs or not directory.name.startswith("scoring-recovery-seed42-"):
        parser.error("Use a new outputs/jobs/scoring-recovery-seed42-* directory")
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "status.json").exists():
        parser.error("Preserve previous job status")
    status = dict(status="starting", pid=os.getpid(), child_pid=None, stages=[],
                  seed=42, eval_enabled=False, scope="scoring_only_recovery",
                  started_utc=supervisor.utc_now(), output_dir=str(OUTPUT))
    lock = jobs / "scoring-recovery-seed42.lock"
    owns_lock = False
    code = 1
    try:
        supervisor.acquire_lock(lock, {"pid": os.getpid(), "job_dir": str(directory)})
        owns_lock = True
        if (jobs / "full-seed42.lock").exists():
            raise RuntimeError("Original full job lock exists; review before recovery")
        if OUTPUT.exists():
            raise RuntimeError("Recovery output already exists; preserve and review")
        status["source_sha256"] = source_hashes()
        for name, arguments in stages():
            if source_hashes() != status["source_sha256"]:
                raise RuntimeError("Recovery source changed after launch")
            supervisor.wait_for_resources(job_dir=directory, status=status, phase=name)
            supervisor.run_stage(worker_command(arguments), root=ROOT, job_dir=directory,
                                 name=name, budget_seconds=1800, status=status)
        report = json.loads((OUTPUT / "summary.json").read_text(encoding="utf-8"))
        validate_completion(report, OUTPUT)
        record_completed_result(report, directory, status)
        status.update(status="complete", metrics=report["metrics"],
                      summary_sha256=hashlib.sha256((OUTPUT / "summary.json").read_bytes()).hexdigest())
        code = 0
    except BaseException as error:
        status.update(status="stopped", reason=f"{type(error).__name__}: {error}")
    finally:
        status.update(ended_utc=supervisor.utc_now(), exit_code=code)
        supervisor.write_json(directory / "status.json", status)
        if owns_lock:
            lock.unlink()
    return code


if __name__ == "__main__":
    supervisor.configure_project_environment()
    if sys.argv[1:2] == ["--worker"]:
        sys.argv = [str(ROOT / "scripts/recover_full_scoring.py"), *sys.argv[2:]]
        runpy.run_path(sys.argv[0], run_name="__main__")
    else:
        raise SystemExit(main())
