"""Supervise frozen narrowband scoring in sequential, recoverable chunks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

import full_run_job as supervisor

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/robustness-full-v1"
CONTRACT = "docs/validation/full_robustness_protocol_v1.json"
CONTRACT_SHA = "806fd09f488449d6e276e4b36b60ddc016441be2b62e7a223d3bc74d7ea6d913"
SOURCES = ("scripts/full_robustness_job.py", "scripts/full_robustness_worker.py",
           "scripts/finalize_full_robustness.py", "scripts/start_full_robustness.ps1")
SEEDS = (42, 43, 44)
CHUNK_SIZE = 2000


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes():
    return {name: digest(ROOT / name) for name in SOURCES}


def check_pins(expected):
    if digest(ROOT / CONTRACT) != CONTRACT_SHA or source_hashes() != expected:
        raise RuntimeError("Protocol or execution source changed")


def remaining_budget(stages, seed, budget=1800, smoke_seconds=0):
    """Count every worker's wall time, including startup, against the seed cap."""
    elapsed = sum(stage["elapsed_seconds"] for stage in stages
                  if stage["name"].startswith(f"seed{seed}_"))
    remaining = budget - elapsed - smoke_seconds
    if remaining <= 0:
        raise RuntimeError(f"Seed {seed} processing budget exhausted")
    return remaining


def worker_command(script, *arguments):
    if script not in ("scripts/full_robustness_worker.py", "scripts/finalize_full_robustness.py"):
        raise ValueError("Unexpected worker")
    return [supervisor.base_python(), "-S", "-u", str(Path(__file__).resolve()),
            "--worker-script", script, *arguments]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    jobs = ROOT / "outputs/jobs"
    directory = args.job_dir.resolve()
    if directory.parent != jobs.resolve() or not directory.name.startswith("full-robustness-"):
        parser.error("Use a new outputs/jobs/full-robustness-* directory")
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "status.json").exists():
        parser.error("Preserve previous job status")
    status = dict(status="starting", pid=os.getpid(), child_pid=None, stages=[],
                  started_utc=supervisor.utc_now(), seeds=list(SEEDS), eval_enabled=False,
                  output_dir=OUTPUT.relative_to(ROOT).as_posix())
    lock = jobs / "full-robustness.lock"
    owns_lock = False
    code = 1
    try:
        supervisor.acquire_lock(lock, dict(pid=os.getpid(), job_dir=str(directory)))
        owns_lock = True
        for other in ("full-seed42.lock", "scoring-recovery-seed42.lock", "full-multiseed.lock"):
            if (jobs / other).exists():
                raise RuntimeError(f"Other job lock exists: {other}")
        if OUTPUT.exists():
            raise RuntimeError("Preserve existing narrowband output; review before recovery")
        pins = source_hashes()
        check_pins(pins)
        OUTPUT.mkdir()
        execution = dict(status="frozen", protocol_sha256=CONTRACT_SHA, source_sha256=pins,
                         seeds=list(SEEDS), chunk_size=CHUNK_SIZE, per_seed_budget_seconds=1800,
                         started_utc=status["started_utc"], job=directory.relative_to(ROOT).as_posix())
        supervisor.write_json(OUTPUT / "execution.json", execution)
        status["source_sha256"] = pins

        def stage(name, script, arguments, budget):
            check_pins(pins)
            supervisor.wait_for_resources(job_dir=directory, status=status, phase=name)
            check_pins(pins)
            supervisor.run_stage(worker_command(script, "--output-dir", str(OUTPUT), *arguments),
                                 root=ROOT, job_dir=directory, name=name,
                                 budget_seconds=budget, status=status)

        stage("preflight", "scripts/full_robustness_worker.py", ["--mode", "preflight"], 1800)
        stage("smoke", "scripts/full_robustness_worker.py", ["--mode", "smoke"], 300)
        smoke = json.loads((OUTPUT / "smoke.json").read_text(encoding="utf-8"))
        if smoke.get("status") != "ok":
            raise RuntimeError("Measured smoke did not pass")
        smoke_seconds = {str(item["seed"]): item["processing_seconds"] for item in smoke["seeds"]}
        if set(smoke_seconds) != {str(seed) for seed in SEEDS}:
            raise RuntimeError("Missing measured smoke seed")
        for seed in SEEDS:
            for start in range(0, 24844, CHUNK_SIZE):
                stop = min(start + CHUNK_SIZE, 24844)
                stage(f"seed{seed}_{start:05d}_{stop:05d}", "scripts/full_robustness_worker.py",
                      ["--mode", "chunk", "--seed", str(seed), "--start", str(start), "--stop", str(stop)],
                      remaining_budget(status["stages"], seed, smoke_seconds=smoke_seconds[str(seed)]))
        # Save measured supervision before the finalizer builds its public audit.
        check_pins(pins)
        from full_multiseed_job import public_stages
        supervisor.write_json(OUTPUT / "supervision.json", dict(status="scoring_complete",
                              stages=public_stages(status["stages"]), source_sha256=pins,
                              seed_smoke_seconds=smoke_seconds,
                              seed_wall_seconds={str(seed): 1800 - remaining_budget(status["stages"], seed,
                                                 smoke_seconds=smoke_seconds[str(seed)]) for seed in SEEDS}))
        stage("finalize", "scripts/finalize_full_robustness.py", [], 1800)
        result = json.loads((OUTPUT / "summary.json").read_text(encoding="utf-8"))
        if result.get("status") != "ok":
            raise RuntimeError("Missing successful validated summary")
        check_pins(pins)
        status.update(status="complete", summary="outputs/robustness-full-v1/summary.json")
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
    if len(sys.argv) > 2 and sys.argv[1] == "--worker-script":
        script = sys.argv[2]
        if script not in ("scripts/full_robustness_worker.py", "scripts/finalize_full_robustness.py"):
            raise SystemExit("Unexpected worker")
        sys.argv = [str(ROOT / script), *sys.argv[3:]]
        runpy.run_path(str(ROOT / script), run_name="__main__")
    else:
        raise SystemExit(main())
