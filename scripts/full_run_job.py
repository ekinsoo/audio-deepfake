"""Supervise one detached, local-only full train/dev seed 42 job."""

import argparse
import ctypes
from datetime import datetime, timezone, timedelta
import json
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from functools import lru_cache

ROOT = Path(__file__).resolve().parents[1]
GIB = 1024 ** 3


def configure_project_environment():
    """Use the venv packages while running the real Python executable directly.

    Windows venv python.exe is a redirector which spawns another interpreter.
    Supervising its process handle would measure/terminate the wrong process.
    """
    import site
    packages = str(ROOT / ".venv/Lib/site-packages")
    # addsitedir alone appends: bundled base NumPy would otherwise win.
    sys.path.insert(0, packages)
    site.addsitedir(packages)
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))


def base_python():
    configuration = dict(line.split(" = ", 1) for line in
                         (ROOT / ".venv/pyvenv.cfg").read_text().splitlines() if " = " in line)
    executable = Path(configuration["executable"])
    if not executable.is_absolute() or not executable.is_file():
        raise RuntimeError("Venv base executable is missing or is not absolute")
    return str(executable)


def worker_command(script, *arguments):
    if script not in ("scripts/prepare_full_data.py", "scripts/run_full_baseline.py"):
        raise ValueError("Only the fixed preparation and full baseline workers are allowed")
    return [base_python(), "-S", "-u", str(ROOT / "scripts/full_run_job.py"),
            "--worker-script", script, *arguments]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    # Windows readers/antivirus can briefly retain the destination handle.
    for attempt in range(10):
        try:
            os.replace(temporary, path)
            break
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.05)


@lru_cache(maxsize=1)
def windows_memory_api():
    """Cache ctypes types: POINTER's global cache otherwise retains every probe."""
    if os.name != "nt":
        raise RuntimeError("The detached full-run supervisor requires Windows")
    from ctypes import wintypes

    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD)] + [
            (name, ctypes.c_ulonglong) for name in (
                "total", "available", "total_page", "available_page", "total_virtual",
                "available_virtual", "available_extended")]

    class ProcessMemory(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in (
                "peak", "working", "peak_paged", "paged", "peak_nonpaged",
                "nonpaged", "pagefile", "peak_pagefile")]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MemoryStatus)]
    kernel.GlobalMemoryStatusEx.restype = wintypes.BOOL
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemory), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    return kernel, psapi, MemoryStatus, ProcessMemory


def windows_memory(pid=None, *, process_handle=None):
    """Read system availability and the owned process lifetime peak working set."""
    kernel, psapi, MemoryStatus, ProcessMemory = windows_memory_api()
    system = MemoryStatus()
    system.length = ctypes.sizeof(system)
    if not kernel.GlobalMemoryStatusEx(ctypes.byref(system)):
        raise ctypes.WinError(ctypes.get_last_error())
    result = {"system_available_ram_bytes": int(system.available)}
    if pid is not None:
        # Popen retains this handle until child cleanup: it cannot refer to a reused PID.
        handle = process_handle or kernel.OpenProcess(0x1000 | 0x0010, False, pid)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            process = ProcessMemory()
            process.cb = ctypes.sizeof(process)
            if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(process), process.cb):
                raise ctypes.WinError(ctypes.get_last_error())
            result["child_peak_rss_bytes"] = int(process.peak)
        finally:
            if process_handle is None:
                kernel.CloseHandle(handle)
    return result


def acquire_lock(path, metadata):
    """A stale lock requires manual review; never automatically take it over."""
    with path.open("x", encoding="utf-8") as target:
        json.dump(metadata, target, indent=2)
        target.write("\n")


def stop_owned_child(child):
    if child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=5)


def wait_for_resources(*, job_dir, status, phase, resource_probe=windows_memory,
                       disk_probe=None, clock=time.monotonic, sleep=time.sleep,
                       budget_seconds=900, poll_seconds=5):
    """Wait with the job lock held; do not spend any worker runtime budget."""
    disk_probe = disk_probe or (lambda: shutil.disk_usage(ROOT).free)
    started = clock()
    record = {"phase": phase, "started_utc": utc_now(),
              "deadline_utc": (datetime.now(timezone.utc) + timedelta(seconds=budget_seconds)).isoformat(),
              "budget_seconds": budget_seconds, "required_ram_bytes": int(3.5 * GIB),
              "required_disk_bytes": 20 * GIB, "required_consecutive_readings": 2,
              "status": "waiting"}
    status.setdefault("resource_waits", []).append(record)
    status.update(status="waiting_resources", stage=f"{phase}_resource_wait", child_pid=None)
    consecutive = 0
    try:
        while True:
            elapsed = clock() - started
            if elapsed >= budget_seconds:
                raise RuntimeError(f"{phase}_resource_wait_expired")
            resources = {**resource_probe(), "disk_free_bytes": disk_probe()}
            record.update(last_resources=resources, elapsed_seconds=elapsed)
            if resources["disk_free_bytes"] < record["required_disk_bytes"]:
                raise RuntimeError("Start requires at least 20 GiB free disk")
            consecutive = (consecutive + 1 if resources["system_available_ram_bytes"]
                           >= record["required_ram_bytes"] else 0)
            record["consecutive_ready_readings"] = consecutive
            write_json(job_dir / "status.json", status)
            if consecutive >= 2:
                record["status"] = "ready"
                status["start_resources"] = resources
                status["status"] = "running"
                return
            sleep(min(poll_seconds, max(0, budget_seconds - (clock() - started))))
    except BaseException as error:
        record.update(status="stopped", reason=str(error))
        raise
    finally:
        record.update(ended_utc=utc_now(), elapsed_seconds=clock() - started)
        write_json(job_dir / "status.json", status)


def run_stage(command, *, root, job_dir, name, budget_seconds, status,
              resource_probe=windows_memory, poll_seconds=1.0):
    started = time.monotonic()
    env = os.environ.copy()
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[variable] = "1"
    record = {"name": name, "command": command, "started_utc": utc_now(),
              "status": "running", "budget_seconds": budget_seconds,
              "peak_rss_bytes": 0, "exit_code": None}
    status["stages"].append(record)
    status["stage"] = name
    child = None
    reason = None
    try:
        with (job_dir / f"{name}.stdout.log").open("wb") as stdout, (
                job_dir / f"{name}.stderr.log").open("wb") as stderr:
            child = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                     stdout=stdout, stderr=stderr,
                                     creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            record["pid"] = child.pid
            status["child_pid"] = child.pid
            write_json(job_dir / "status.json", status)
            while child.poll() is None:
                elapsed = time.monotonic() - started
                if elapsed >= budget_seconds:
                    reason = "stage_runtime_budget_exceeded"
                    break
                try:
                    memory = (windows_memory(child.pid, process_handle=child._handle)
                              if resource_probe is windows_memory else resource_probe(child.pid))
                except Exception:
                    if child.poll() is not None:
                        break
                    raise
                record["peak_rss_bytes"] = max(record["peak_rss_bytes"], memory["child_peak_rss_bytes"])
                record["last_memory"] = memory
                record["elapsed_seconds"] = elapsed
                write_json(job_dir / "status.json", status)
                if record["peak_rss_bytes"] > 2 * GIB:
                    reason = "child_peak_memory_budget_exceeded"
                    break
                if memory["system_available_ram_bytes"] < GIB:
                    reason = "system_memory_pressure"
                    break
                time.sleep(poll_seconds)
            if reason:
                stop_owned_child(child)
            record["exit_code"] = child.wait()
            if reason:
                raise RuntimeError(reason)
            if record["exit_code"] != 0:
                raise RuntimeError(f"child_exit_{record['exit_code']}")
            record["status"] = "complete"
    except BaseException as error:
        if child is not None:
            stop_owned_child(child)
            record["exit_code"] = child.poll()
        record["status"] = "stopped"
        record["reason"] = str(error)
        raise
    finally:
        record["ended_utc"] = utc_now()
        record["elapsed_seconds"] = time.monotonic() - started
        status["child_pid"] = None
        write_json(job_dir / "status.json", status)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    job_dir = args.job_dir.resolve()
    jobs = (ROOT / "outputs/jobs").resolve()
    if job_dir.parent != jobs or not job_dir.name.startswith("full-seed42-"):
        parser.error("job-dir must be an outputs/jobs/full-seed42-* directory")
    job_dir.mkdir(parents=True, exist_ok=True)
    if (job_dir / "status.json").exists():
        parser.error("Job status already exists; preserve this run and choose a new directory")
    status = {"status": "starting", "pid": os.getpid(), "child_pid": None,
              "started_utc": utc_now(), "stages": [], "seed": 42,
              "scope": "official_train_dev_confirmation", "eval_enabled": False,
              "output_dir": "outputs/baseline-full-seed42"}
    lock = jobs / "full-seed42.lock"
    owns_lock = False
    try:
        acquire_lock(lock, {"pid": os.getpid(), "job_dir": str(job_dir),
                            "started_utc": status["started_utc"],
                            "command": [sys.executable, *sys.argv]})
        owns_lock = True
        write_json(job_dir / "status.json", status)
        if (ROOT / "outputs/baseline-full-seed42").exists():
            raise RuntimeError("Full seed 42 output already exists; preserve and review it")
        status["source_sha256"] = {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
                "scripts/full_run_job.py", "scripts/start_full_run.ps1",
                "scripts/prepare_full_data.py", "scripts/run_full_baseline.py",
                "scripts/prepare_pilot_data.py", "scripts/fetch_pilot_audio.py",
                "scripts/fetch_protocols.py", "src/audio_deepfake/baseline.py",
                "src/audio_deepfake/features.py", "src/audio_deepfake/audio.py",
                "src/audio_deepfake/manifests.py", "src/audio_deepfake/metrics.py",
                "src/audio_deepfake/resources.py", "src/audio_deepfake/cli.py", "configs/full_seed42.toml",
                "docs/FULL_CONFIRMATION_PLAN.md")}
        commands = [
            ("prepare", 3600, worker_command("scripts/prepare_full_data.py", "--execute")),
            ("baseline", 1800, worker_command("scripts/run_full_baseline.py", "--config",
                                "configs/full_seed42.toml", "--output-dir", "outputs/baseline-full-seed42")),
        ]
        for name, budget, command in commands:
            wait_for_resources(job_dir=job_dir, status=status,
                               phase="startup" if name == "prepare" else "baseline_start")
            run_stage(command, root=ROOT, job_dir=job_dir, name=name,
                      budget_seconds=budget, status=status)
            artifact = ROOT / ("outputs/full-data/prepare_report.json" if name == "prepare"
                               else "outputs/baseline-full-seed42/summary.json")
            report = json.loads(artifact.read_text(encoding="utf-8"))
            if report.get("status") != "ok":
                raise RuntimeError(f"{name} completion artifact has no successful status")
            if name == "prepare":
                if (report.get("expected_records") != 50224 or report.get("verified_records") != 50224
                        or len(report.get("records", {})) != 50224
                        or report.get("archive", {}).get("verified") is not True):
                    raise RuntimeError("Preparation completion must verify all 50224 records")
            if name == "baseline":
                validate_baseline_completion(report, ROOT / "outputs/baseline-full-seed42")
        status["status"] = "complete"
        return_code = 0
    except BaseException as error:
        status["status"] = "stopped"
        status["reason"] = f"{type(error).__name__}: {error}"
        return_code = 1
    finally:
        status["ended_utc"] = utc_now()
        status["exit_code"] = locals().get("return_code", 1)
        write_json(job_dir / "status.json", status)
        if owns_lock:
            lock.unlink()
    return return_code


def validate_baseline_completion(report, output):
    if report.get("scope") != "full-official-train-dev-adaptation":
        raise RuntimeError("Unexpected full baseline scope")
    expected = {"train": (25380, {"bonafide": 2580, "spoof": 22800}),
                "dev": (24844, {"bonafide": 2548, "spoof": 22296})}
    if set(report.get("splits", {})) != set(expected):
        raise RuntimeError("Full baseline must report exact train/dev splits")
    for name, (count, labels) in expected.items():
        if report["splits"][name].get("count") != count or report["splits"][name].get("labels") != labels:
            raise RuntimeError("Full baseline official count/labels mismatch")
    for name in ("model.joblib", "scores.csv", "config.json"):
        actual = hashlib.sha256((output / name).read_bytes()).hexdigest()
        if report.get("output_sha256", {}).get(name) != actual:
            raise RuntimeError(f"Full baseline artifact hash mismatch: {name}")
    # Verify the runner's complete finite score artifact, not just its exit code.
    import csv
    import math
    with (output / "scores.csv").open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    if len(rows) != 24844 or len({row["utterance_id"] for row in rows}) != 24844:
        raise RuntimeError("Full baseline must contain 24844 unique dev scores")
    if any(row["split"] != "dev" or not math.isfinite(float(row["score"])) for row in rows):
        raise RuntimeError("Full baseline contains invalid dev scores")
    if not (output / "model.joblib").is_file():
        raise RuntimeError("Completed full baseline model is missing")


if __name__ == "__main__":
    configure_project_environment()
    if sys.argv[1:2] == ["--worker-script"]:
        import runpy
        if len(sys.argv) < 3:
            raise SystemExit("Missing fixed worker script")
        script = sys.argv[2]
        worker_command(script)  # Validate the exact allowlist before running.
        sys.argv = [str(ROOT / script), *sys.argv[3:]]
        runpy.run_path(sys.argv[0], run_name="__main__")
    elif sys.argv[1:] == ["--worker-probe"]:
        # Diagnostic regression check only: imports packages, never reads audio or fits.
        from importlib.metadata import version
        import numpy, scipy, sklearn, soundfile, joblib
        print(json.dumps({"pid": os.getpid(), "executable": sys.executable,
                          "versions": {name: version(name) for name in (
                              "numpy", "scipy", "scikit-learn", "soundfile", "joblib")},
                          "imported_versions": {"numpy": numpy.__version__, "scipy": scipy.__version__,
                              "scikit-learn": sklearn.__version__, "soundfile": soundfile.__version__,
                              "joblib": joblib.__version__}}),
              flush=True)
        time.sleep(0.2)
    else:
        raise SystemExit(main())
