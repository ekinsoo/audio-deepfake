"""External 300 second guard for the single frozen recording-context ablation."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from full_run_job import (base_python, configure_project_environment, windows_memory,
                          write_json, acquire_lock, stop_owned_child)
from localization_pilot_job import check_resources, verify_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--job-dir", type=Path)
    args = parser.parse_args()
    if args.worker:
        configure_project_environment()
        sys.argv = ["scripts/run_localization_context.py", "--execute", "--output-dir", str(args.job_dir / "pilot")]
        runpy.run_path(str(ROOT / sys.argv[0]), run_name="__main__")
        return
    if not args.execute:
        print(json.dumps({"status": "plan_only", "worker_seconds": 300, "training_started": False}))
        return
    if args.job_dir is None:
        raise ValueError("Require fresh --job-dir")
    directory = args.job_dir.resolve()
    if not directory.is_relative_to(ROOT / "outputs/jobs") or directory.exists():
        raise ValueError("Require fresh outputs/jobs directory")
    frozen = json.loads((ROOT / "docs/validation/localization_context_frozen_v2.json").read_text(encoding="utf-8"))
    pins = frozen["source_sha256"]
    verify_sources(pins)
    memory = windows_memory()
    check_resources(memory, shutil.disk_usage(ROOT).free)
    directory.mkdir(parents=True)
    lock = ROOT / "outputs/localization_context.lock"
    acquire_lock(lock, {"pid": os.getpid(), "job_dir": str(directory)})
    status = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
              "training_started": False, "preflight": memory, "peak_rss_bytes": 0,
              "worker_seconds": 300, "eval_enabled": False, "source_sha256": pins}
    child = None
    start = time.monotonic()
    try:
        write_json(directory / "status.json", status)
        command = [base_python(), "-S", "-u", str(Path(__file__).resolve()),
                   "--worker", "--job-dir", str(directory)]
        environment = os.environ.copy()
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            environment[key] = "1"
        with (directory / "stdout.log").open("wb") as stdout, (directory / "stderr.log").open("wb") as stderr:
            child = subprocess.Popen(command, cwd=ROOT, env=environment,
                stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            status["child_pid"] = child.pid
            while child.poll() is None:
                if time.monotonic() - start >= 300:
                    raise RuntimeError("Context worker external 300 second timeout")
                try:
                    memory = windows_memory(child.pid, process_handle=child._handle)
                except Exception:
                    if child.poll() is not None:
                        break
                    raise
                status["peak_rss_bytes"] = max(status["peak_rss_bytes"], memory["child_peak_rss_bytes"])
                check_resources(memory, shutil.disk_usage(ROOT).free)
                runner_path = directory / "pilot/status.json"
                if runner_path.exists():
                    state = json.loads(runner_path.read_text(encoding="utf-8"))
                    status["runner_phase"] = state["phase"]
                    status["training_started"] = state["training_started"]
                write_json(directory / "status.json", status)
                time.sleep(.25)
            status["exit_code"] = child.wait()
        if status["exit_code"]:
            raise RuntimeError(f"Context worker exited {status['exit_code']}")
        verify_sources(pins)
        state = json.loads((directory / "pilot/status.json").read_text(encoding="utf-8"))
        result = json.loads((directory / "pilot/results.json").read_text(encoding="utf-8"))
        if state["phase"] != "complete" or result["status"] != "complete":
            raise RuntimeError("Context worker did not complete")
        status.update(status="complete", training_started=True)
    except BaseException as error:
        status.update(status="stopped", error=str(error))
        raise
    finally:
        if child is not None:
            stop_owned_child(child)
        runner_path = directory / "pilot/status.json"
        if runner_path.exists():
            state = json.loads(runner_path.read_text(encoding="utf-8"))
            status["training_started"] = state["training_started"]
            status["runner_phase"] = state["phase"]
        status["finished_utc"] = datetime.now(timezone.utc).isoformat()
        status["elapsed_seconds"] = time.monotonic() - start
        status["child_pid"] = None
        write_json(directory / "status.json", status)
        lock.unlink()


if __name__ == "__main__":
    main()
