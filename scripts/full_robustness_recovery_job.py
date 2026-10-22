"""Resume only missing seed 44 chunks, preserving the frozen experiment."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys

import full_run_job as supervisor
import full_robustness_job as original

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/robustness-full-v1"
ORIGINAL_JOB = ROOT / "outputs/jobs/full-robustness-20261010T095134725Z"
SOURCES = ("scripts/full_robustness_recovery_job.py",
           "scripts/full_robustness_recovery_audit.py", "scripts/start_full_robustness_recovery.ps1")


def source_hashes():
    return {name: original.digest(ROOT / name) for name in SOURCES}


def remaining_budget(stages, spent):
    elapsed = sum(item["elapsed_seconds"] for item in stages if item["name"].startswith("seed44_"))
    remaining = 1800 - spent - elapsed
    if remaining <= 0:
        raise RuntimeError("Seed 44 total processing budget exhausted, including failed attempt")
    return remaining


def audit_command(mode):
    return [supervisor.base_python(), "-S", "-u", str(Path(__file__).resolve()),
            "--audit-worker", "--mode", mode, "--output-dir", str(OUTPUT),
            "--original-job-dir", str(ORIGINAL_JOB)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    jobs = ROOT / "outputs/jobs"
    directory = args.job_dir.resolve()
    if directory.parent != jobs.resolve() or not directory.name.startswith("full-robustness-recovery-"):
        parser.error("Use a new outputs/jobs/full-robustness-recovery-* directory")
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "status.json").exists():
        parser.error("Preserve previous status")
    status = dict(status="starting", pid=os.getpid(), child_pid=None, stages=[],
                  started_utc=supervisor.utc_now(), eval_enabled=False, recovered=True,
                  output_dir=OUTPUT.relative_to(ROOT).as_posix(),
                  original_job=ORIGINAL_JOB.relative_to(ROOT).as_posix())
    lock = jobs / "full-robustness-recovery.lock"
    owned = False
    code = 1
    try:
        supervisor.acquire_lock(lock, dict(pid=os.getpid(), job_dir=str(directory)))
        owned = True
        for name in ("full-robustness.lock", "full-seed42.lock", "scoring-recovery-seed42.lock", "full-multiseed.lock"):
            if (jobs / name).exists():
                raise RuntimeError(f"Conflicting job lock: {name}")
        from full_robustness_recovery_audit import validate_resume
        resume = validate_resume(OUTPUT, ORIGINAL_JOB)
        for name in ("recovery_execution.json", "recovery_preflight.json", "supervision.json", "summary.json"):
            if (OUTPUT / name).exists():
                raise RuntimeError(f"Preserve existing artifact: {name}")
        pins = source_hashes()
        execution = dict(status="frozen", source_sha256=pins,
                         original_job=resume["original_job"], original_status_sha256=resume["original_status_sha256"],
                         chunk_sha256=resume["chunk_sha256"], started_utc=status["started_utc"],
                         preflight_sha256=original.digest(OUTPUT / "preflight.json"),
                         smoke_sha256=original.digest(OUTPUT / "smoke.json"), failure_history=resume["failed_stages"],
                         recovery_job=directory.relative_to(ROOT).as_posix(),
                         seed_spent_seconds=resume["seed_spent_seconds"],
                         reason="Original seed 44 worker stopped on system available RAM guard; keep completed chunks.")
        supervisor.write_json(OUTPUT / "recovery_execution.json", execution)
        status.update(source_sha256=pins, seed_spent_seconds=resume["seed_spent_seconds"])
        execution_hash = original.digest(OUTPUT / "recovery_execution.json")

        def check_pins():
            if source_hashes() != pins or original.digest(OUTPUT / "recovery_execution.json") != execution_hash:
                raise RuntimeError("Recovery execution sources or plan changed")
            original.check_pins(json.loads((OUTPUT / "execution.json").read_text())["source_sha256"])

        def stage(name, command, budget):
            check_pins()
            supervisor.wait_for_resources(job_dir=directory, status=status, phase=name)
            check_pins()
            supervisor.run_stage(command, root=ROOT, job_dir=directory, name=name,
                                 budget_seconds=budget, status=status)

        stage("recovery_preflight", audit_command("verify"), 1800)
        for start in range(0, 24844, original.CHUNK_SIZE):
            stop = min(start + original.CHUNK_SIZE, 24844)
            command = original.worker_command("scripts/full_robustness_worker.py", "--mode", "chunk",
                         "--output-dir", str(OUTPUT), "--seed", "44", "--start", str(start), "--stop", str(stop))
            stage(f"seed44_{start:05d}_{stop:05d}", command,
                  remaining_budget(status["stages"], resume["seed_spent_seconds"]["44"]))
        check_pins()
        from full_multiseed_job import public_stages
        complete_new = public_stages([item for item in status["stages"] if item["name"].startswith("seed44_")])
        wall = dict(resume["seed_spent_seconds"])
        wall["44"] = 1800 - remaining_budget(status["stages"], wall["44"])
        supervisor.write_json(OUTPUT / "supervision.json", dict(status="scoring_complete", recovered=True,
                              source_sha256=json.loads((OUTPUT / "execution.json").read_text())["source_sha256"],
                              stages=resume["success_stages"] + complete_new,
                              recovery_stages=public_stages([item for item in status["stages"] if item["name"] == "recovery_preflight"]),
                              seed_wall_seconds=wall, seed_smoke_seconds=resume["seed_smoke_seconds"],
                              failure_history=resume["failed_stages"], original_job=resume["original_job"],
                              original_status_sha256=resume["original_status_sha256"],
                              recovery_execution_sha256=execution_hash,
                              recovery_preflight_sha256=original.digest(OUTPUT / "recovery_preflight.json")))
        stage("recovery_finalize", audit_command("finalize"), 1800)
        result = json.loads((OUTPUT / "summary.json").read_text())
        if result.get("status") != "ok" or result.get("supervision", {}).get("recovered") is not True:
            raise RuntimeError("Missing validated recovery summary")
        check_pins()
        status.update(status="complete", summary="outputs/robustness-full-v1/summary.json")
        code = 0
    except BaseException as error:
        status.update(status="stopped", reason=f"{type(error).__name__}: {error}")
    finally:
        status.update(ended_utc=supervisor.utc_now(), exit_code=code)
        supervisor.write_json(directory / "status.json", status)
        if owned:
            lock.unlink()
    return code


if __name__ == "__main__":
    supervisor.configure_project_environment()
    if len(sys.argv) > 1 and sys.argv[1] == "--audit-worker":
        script = ROOT / "scripts/full_robustness_recovery_audit.py"
        sys.argv = [str(script), *sys.argv[2:]]
        runpy.run_path(str(script), run_name="__main__")
    else:
        raise SystemExit(main())
