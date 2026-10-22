"""Queue one bounded localization pilot behind verified audio and alignment."""

import argparse
from datetime import datetime, timezone
import hashlib
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

GIB = 1024 ** 3
WORKERS = {"alignment": "scripts/audit_partial_alignment.py",
           "pilot": "scripts/run_localization_pilot.py"}
PIN_PATHS = ["scripts/localization_pilot_job.py", "scripts/start_localization_pilot.ps1",
             "scripts/full_run_job.py", *WORKERS.values(),
             "docs/validation/partial_selection_v1.json",
             "docs/validation/partial_metadata.json",
             "data/partialspoof/selected_labels_v1.json"]
PIN_PATHS += [f"src/audio_deepfake/{name}.py" for name in (
    "localization", "localization_features", "localization_model", "localization_timeline",
    "partialspoof", "features", "audio", "manifests", "metrics", "resources")]


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_sources(root=ROOT):
    return {name: sha(root / name) for name in PIN_PATHS}


def verify_sources(pins, root=ROOT):
    if any(not (root / name).is_file() or sha(root / name) != digest
           for name, digest in pins.items()):
        raise RuntimeError("Frozen source or input changed")


def check_resources(memory, disk):
    if memory["system_available_ram_bytes"] < 2 * GIB:
        raise RuntimeError("Available system RAM below 2 GiB")
    if memory.get("child_peak_rss_bytes", 0) > GIB:
        raise RuntimeError("Worker lifetime peak RAM exceeds 1 GiB")
    if disk < 20 * GIB:
        raise RuntimeError("Free disk below 20 GiB")


def audio_ready(root, selection_sha256):
    ready = True
    for split in ("train", "dev"):
        path = root / f"data/partialspoof/audio_{split}_complete.json"
        if not path.exists():
            ready = False
            continue
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if receipt.get("status") != "complete" or receipt.get("selection_sha256") != selection_sha256:
            raise RuntimeError(f"{split} audio receipt does not match the frozen selection")
    return ready


def validate_alignment(result, pins, root=ROOT):
    expected = {"selection_sha256": "docs/validation/partial_selection_v1.json",
        "compact_labels_sha256": "data/partialspoof/selected_labels_v1.json",
        "metadata_sha256": "docs/validation/partial_metadata.json",
        "alignment_source_sha256": WORKERS["alignment"],
        "adapter_source_sha256": "src/audio_deepfake/partialspoof.py"}
    if result.get("training_alignment_ready") is not True:
        raise RuntimeError("Alignment gate failed; no training started")
    if any(result.get(key) != pins[path] for key, path in expected.items()):
        raise RuntimeError("Alignment audit source or input pins differ")
    for split, count in (("train", 128), ("dev", 64)):
        entry = result["splits"][split]
        receipt = f"data/partialspoof/audio_{split}_complete.json"
        if entry.get("receipt_sha256") != pins[receipt] or len(entry["records"]) != count:
            raise RuntimeError("Alignment audit receipt or record count differs")
        for record in entry["records"]:
            path = (root / record["audio_path"]).resolve()
            if not path.is_relative_to((root / "data/partialspoof/database" / split).resolve()):
                raise RuntimeError("Audited audio path escaped its split")
            if record.get("candidate_match") is not True or sha(path) != record["audio_sha256"]:
                raise RuntimeError("Audited audio changed or alignment failed")


def wait_audio(*, root, job_dir, status, audio_job_dir, selection_sha256,
               budget_seconds=1800, clock=time.monotonic, sleep=time.sleep):
    started = clock()
    status.update(status="waiting_audio", stage="audio", training_started=False)
    while True:
        if audio_ready(root, selection_sha256):
            return
        if clock() - started >= budget_seconds:
            raise RuntimeError("Audio preparation wait expired after 1800 seconds")
        source_status = audio_job_dir / "status.json"
        if not source_status.is_file():
            raise RuntimeError("Audio job status is missing")
        upstream = json.loads(source_status.read_text(encoding="utf-8"))
        if upstream.get("status") in ("stopped", "failed", "complete"):
            raise RuntimeError("Audio job ended without both matching completion receipts")
        status["audio_wait_elapsed_seconds"] = clock() - started
        write_json(job_dir / "status.json", status)
        sleep(5)


def worker_command(name, *arguments):
    return [base_python(), "-S", "-u", str(Path(__file__).resolve()),
            "--worker", name, *map(str, arguments)]


def require_pilot_complete(job_dir):
    directory = job_dir / "pilot"
    state = json.loads((directory / "status.json").read_text(encoding="utf-8"))
    result = json.loads((directory / "results.json").read_text(encoding="utf-8"))
    if (state.get("phase") != "complete" or state.get("training_started") is not True
            or result.get("status") != "complete"):
        raise RuntimeError("Pilot exited without complete training status and results")
    return state


def run_stage(name, arguments, *, job_dir, status, budget_seconds):
    check_resources(windows_memory(), shutil.disk_usage(ROOT).free)
    environment = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[key] = "1"
    record = {"name": name, "started_utc": now(), "budget_seconds": budget_seconds,
              "peak_rss_bytes": 0, "status": "running"}
    status["stages"].append(record)
    status.update(status="running", stage=name)
    child = None
    started = time.monotonic()
    try:
        with (job_dir / f"{name}.stdout.log").open("wb") as stdout, (
                job_dir / f"{name}.stderr.log").open("wb") as stderr:
            child = subprocess.Popen(worker_command(name, *arguments), cwd=ROOT,
                env=environment, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            status["child_pid"] = record["pid"] = child.pid
            while child.poll() is None:
                if time.monotonic() - started >= budget_seconds:
                    raise RuntimeError(f"{name} runtime budget exceeded")
                try:
                    memory = windows_memory(child.pid, process_handle=child._handle)
                except Exception:
                    if child.poll() is not None:
                        break
                    raise
                record["peak_rss_bytes"] = max(record["peak_rss_bytes"], memory["child_peak_rss_bytes"])
                check_resources(memory, shutil.disk_usage(ROOT).free)
                runner_status = job_dir / "pilot" / "status.json"
                if name == "pilot" and runner_status.is_file():
                    runner = json.loads(runner_status.read_text(encoding="utf-8"))
                    status["training_started"] = bool(runner.get("training_started", False))
                    status["runner_phase"] = runner.get("phase", runner.get("status"))
                write_json(job_dir / "status.json", status)
                time.sleep(1)
            record["exit_code"] = child.wait()
            if record["exit_code"]:
                raise RuntimeError(f"{name} worker exited with code {record['exit_code']}")
            record["status"] = "complete"
    except BaseException as error:
        record.update(status="stopped", error=str(error))
        raise
    finally:
        if child is not None:
            stop_owned_child(child)
        runner_status = job_dir / "pilot/status.json"
        if name == "pilot" and runner_status.is_file():
            try:
                state = json.loads(runner_status.read_text(encoding="utf-8"))
                status["training_started"] = bool(state.get("training_started", False))
                status["runner_phase"] = state.get("phase")
            except Exception as error:
                status["runner_status_read_error"] = str(error)
        status["child_pid"] = None
        record["finished_utc"] = now()
        write_json(job_dir / "status.json", status)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--job-dir", type=Path)
    parser.add_argument("--audio-job-dir", type=Path)
    parser.add_argument("--alignment-audit", type=Path, default=ROOT / "docs/validation/localization_alignment.json")
    parser.add_argument("--worker", choices=WORKERS)
    args, remaining = parser.parse_known_args()
    if args.worker:
        configure_project_environment()
        sys.argv = [WORKERS[args.worker], *( ["--execute"] if args.execute else []), *remaining]
        runpy.run_path(str(ROOT / WORKERS[args.worker]), run_name="__main__")
        return
    if remaining:
        parser.error("Unknown arguments")
    if not args.execute:
        print(json.dumps({"status": "plan", "audio_wait_seconds": 1800,
            "alignment_seconds": 240, "pilot_seconds": 300, "eval_enabled": False,
            "required_available_ram_bytes": 2 * GIB, "worker_peak_limit_bytes": GIB,
            "required_disk_bytes": 20 * GIB, "training_started": False}, indent=2))
        return
    if args.job_dir is None or args.audio_job_dir is None:
        parser.error("Execute requires --job-dir and --audio-job-dir")
    directory, audio_directory = args.job_dir.resolve(), args.audio_job_dir.resolve()
    for path in (directory, audio_directory):
        if not path.is_relative_to(ROOT / "outputs/jobs"):
            raise ValueError("Job directories must stay under outputs/jobs")
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "status.json").exists():
        raise ValueError("Use a new job directory; previous status is preserved")
    lock = ROOT / "outputs/localization_pilot.lock"
    acquire_lock(lock, {"pid": os.getpid(), "job_dir": str(directory), "started_utc": now()})
    status = {"status": "queued", "started_utc": now(), "training_started": False,
              "eval_enabled": False, "stages": [], "audio_job_dir": str(audio_directory)}
    try:
        pins = freeze_sources()
        status["source_sha256"] = pins
        write_json(directory / "plan.json", {"source_sha256": pins, "seed": 45,
            "audio_wait_seconds": 1800, "alignment_seconds": 240, "pilot_seconds": 300,
            "eval_enabled": False})
        write_json(directory / "status.json", status)
        wait_audio(root=ROOT, job_dir=directory, status=status, audio_job_dir=audio_directory,
                   selection_sha256=pins["docs/validation/partial_selection_v1.json"])
        verify_sources(pins)
        for split in ("train", "dev"):
            name = f"data/partialspoof/audio_{split}_complete.json"
            pins[name] = sha(ROOT / name)
        write_json(directory / "verified_input_pins.json", pins)
        plan = json.loads((directory / "plan.json").read_text(encoding="utf-8"))
        plan["source_sha256"] = pins
        plan["audio_receipts_verified_utc"] = now()
        write_json(directory / "plan.json", plan)
        alignment = directory / "alignment.json"
        if args.alignment_audit.exists():
            existing = args.alignment_audit.resolve()
            if not existing.is_relative_to(ROOT):
                raise ValueError("Alignment audit must stay in the repository")
            result = json.loads(existing.read_text(encoding="utf-8"))
            validate_alignment(result, pins)
            alignment.write_bytes(existing.read_bytes())
            status["alignment_reused_from"] = str(existing)
        else:
            run_stage("alignment", ["--execute", "--output", alignment],
                      job_dir=directory, status=status, budget_seconds=240)
        result = json.loads(alignment.read_text(encoding="utf-8"))
        validate_alignment(result, pins)
        verify_sources(pins)
        status["alignment_sha256"] = sha(alignment)
        run_stage("pilot", ["--execute", "--alignment", alignment, "--output-dir", directory / "pilot"],
                  job_dir=directory, status=status, budget_seconds=300)
        verify_sources(pins)
        state = require_pilot_complete(directory)
        status["training_started"] = state["training_started"]
        status.update(status="complete", finished_utc=now())
    except BaseException as error:
        status.update(status="stopped", finished_utc=now(), error=str(error))
        raise
    finally:
        write_json(directory / "status.json", status)
        lock.unlink()


if __name__ == "__main__":
    main()
