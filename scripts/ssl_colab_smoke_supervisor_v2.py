"""Bounded Linux supervisor for the unchanged, hash-frozen CUDA smoke worker.

Handles disappearing /proc memory fields during child exit without treating
a worker result alone as a successful supervised run. No downloads or training.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

GIB = 1024 ** 3
WORKER_SHA256 = "c3c38b21e0d1b3a672831bff57cd8fec33d9dc7f2d91e2227176b11528e21337"
MANIFEST_SHA256 = "0f4df17c238c3df1b369239342c86fd4552011cd3e4c8c745fbf8e65048f5c7c"
WORKER_SECONDS = 180
EXIT_GRACE_SECONDS = 1.0


class ResourceProbeError(RuntimeError):
    """A required resource measurement is unavailable or malformed."""


class ProcessProbeUnavailable(ResourceProbeError):
    """Process metrics can disappear before waitpid observes child exit."""


class ResourceLimitError(RuntimeError):
    """A frozen resource limit has been reached."""


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def guard_system(output, *, initial=False):
    try:
        fields = {line.split(":", 1)[0]: line.split(":", 1)[1].split()
                  for line in Path("/proc/meminfo").read_text().splitlines() if ":" in line}
        available = int(fields["MemAvailable"][0]) * 1024
    except (OSError, KeyError, IndexError, ValueError) as error:
        raise ResourceProbeError("Cannot read system MemAvailable") from error
    if available < (4 if initial else 2) * GIB:
        raise ResourceLimitError("Available system RAM boundary reached")
    disk_free = shutil.disk_usage(output).free
    if disk_free < 20 * GIB:
        raise ResourceLimitError("Free disk below 20 GiB")
    return {"system_available_ram_bytes": available, "disk_free_bytes": disk_free}


def process_memory(pid):
    try:
        lines = Path(f"/proc/{pid}/status").read_text().splitlines()
    except FileNotFoundError as error:
        raise ProcessProbeUnavailable(f"/proc/{pid}/status disappeared") from error
    except OSError as error:
        raise ResourceProbeError(f"Cannot read /proc/{pid}/status") from error
    fields = {line.split(":", 1)[0]: line.split(":", 1)[1].strip()
              for line in lines if ":" in line}
    missing = [name for name in ("VmRSS", "VmHWM") if name not in fields]
    if missing:
        raise ProcessProbeUnavailable(
            f"Process {pid} lacks {','.join(missing)}; state={fields.get('State', 'unknown')}")
    try:
        return {"rss_bytes": int(fields["VmRSS"].split()[0]) * 1024,
                "peak_rss_bytes": int(fields["VmHWM"].split()[0]) * 1024}
    except (ValueError, IndexError) as error:
        raise ResourceProbeError(f"Malformed process memory for {pid}") from error


def monitor_child(child, output, status, deadline, *, clock=time.monotonic, sleep=time.sleep):
    missing_since = None
    while child.poll() is None:
        now = clock()
        if now >= deadline:
            raise ResourceLimitError("External 180-second timeout reached")
        system = guard_system(output)
        try:
            process = process_memory(child.pid)
        except ProcessProbeUnavailable as error:
            if missing_since is None:
                missing_since = now
            status["last_process_probe_error"] = {
                "error_type": type(error).__name__, "error": str(error),
                "grace_seconds": EXIT_GRACE_SECONDS}
            remaining = min(deadline, missing_since + EXIT_GRACE_SECONDS) - clock()
            if remaining <= 0:
                if clock() >= deadline:
                    raise ResourceLimitError("External 180-second timeout reached") from error
                raise ResourceProbeError("Process memory unavailable beyond exit grace; child still running") from error
            save(output / "status.json", status)
            try:
                return child.wait(timeout=min(.1, remaining))
            except subprocess.TimeoutExpired:
                continue
        missing_since = None
        if process["peak_rss_bytes"] > 3 * GIB:
            raise ResourceLimitError("Worker peak RSS exceeds 3 GiB")
        status["peak_rss_bytes"] = max(status["peak_rss_bytes"], process["peak_rss_bytes"])
        status["last_memory"] = {**system, **process}
        save(output / "status.json", status)
        sleep(min(.1, max(0, deadline - clock())))
    return child.wait()


def load_frozen_worker(bundle):
    path = bundle / "ssl_colab_smoke.py"
    if sha(path) != WORKER_SHA256:
        raise ValueError("Frozen worker source hash differs")
    spec = importlib.util.spec_from_file_location("frozen_ssl_colab_smoke_v1", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def validate_completion(bundle, output, status, frozen, deadline):
    if status["exit_code"] != 0:
        raise RuntimeError(f"Worker exited with code {status['exit_code']}; inspect retained logs")
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    if result.get("status") != "complete":
        raise RuntimeError("Real worker completion is missing")
    if any(result.get(key) is not False for key in
           ("training_started", "labels_opened", "dev_audio_accessed", "eval_accessed")):
        raise RuntimeError("Worker scope flags differ from the frozen smoke")
    if result.get("source_sha256") != WORKER_SHA256:
        raise RuntimeError("Worker result source hash differs")
    if result.get("manifest_sha256") != status["manifest_sha256"]:
        raise RuntimeError("Worker result manifest hash differs")
    final_peak = result["resources_before_completion"]["peak_rss_bytes"]
    status["peak_rss_bytes"] = max(status["peak_rss_bytes"], final_peak)
    if status["peak_rss_bytes"] > 3 * GIB:
        raise ResourceLimitError("Final recorded worker peak exceeds 3 GiB")
    if result["resources_before_model_load"]["system_available_ram_bytes"] < 4 * GIB:
        raise ResourceLimitError("Worker immediate pre-load RAM check did not pass")
    if result.get("source_frame_count") != 368 or result.get("aligned_shape") != [369, 768]:
        raise RuntimeError("Worker result geometry differs")
    status["resources_after_worker_exit"] = guard_system(output)
    frozen.verify_bundle(bundle)
    if sha(bundle / "ssl_colab_smoke.py") != WORKER_SHA256:
        raise RuntimeError("Worker source changed during smoke")
    if sha(__file__) != status["supervisor_sha256"]:
        raise RuntimeError("Supervisor source changed during smoke")
    if sha(bundle / "manifest.json") != status["manifest_sha256"]:
        raise RuntimeError("Manifest changed during smoke")
    if sha(output / f"{frozen.SMOKE_ID}.npy") != result["features_sha256"]:
        raise RuntimeError("Feature artifact changed")
    if time.monotonic() >= deadline:
        raise ResourceLimitError("External 180-second timeout reached during final validation")


def supervise(bundle, output):
    if sys.platform != "linux":
        raise RuntimeError("The supervisor requires Linux /proc")
    if output.exists() or output == bundle or output.is_relative_to(bundle):
        raise ValueError("Require a fresh output directory outside the input bundle")
    guard_system(output.parent, initial=True)
    frozen, worker_path = load_frozen_worker(bundle)
    if sha(bundle / "manifest.json") != MANIFEST_SHA256:
        raise ValueError("Frozen input manifest hash differs")
    frozen.verify_bundle(bundle)
    output.mkdir(parents=True)
    status = {"status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
              "training_started": False, "worker_seconds": WORKER_SECONDS,
              "peak_rss_bytes": 0, "worker_sha256": WORKER_SHA256,
              "supervisor_sha256": sha(__file__), "manifest_sha256": sha(bundle / "manifest.json"),
              "exit_grace_seconds": EXIT_GRACE_SECONDS}
    child = None
    start = time.monotonic()
    deadline = start + WORKER_SECONDS
    try:
        save(output / "status.json", status)
        environment = os.environ.copy()
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            environment[key] = "2"
        environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
        with (output / "stdout.log").open("wb") as stdout, (output / "stderr.log").open("wb") as stderr:
            child = subprocess.Popen([sys.executable, "-u", str(worker_path), "--worker", "--execute",
                "--bundle-dir", str(bundle), "--output-dir", str(output)], env=environment,
                stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, start_new_session=True)
            status["worker_pid"] = child.pid
            status["exit_code"] = monitor_child(child, output, status, deadline)
        validate_completion(bundle, output, status, frozen, deadline)
        status["status"] = "complete"
    except BaseException as error:
        status.update(status="stopped", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait(timeout=5)
        status.update(worker_pid=None, elapsed_seconds=time.monotonic() - start,
                      finished_utc=datetime.now(timezone.utc).isoformat())
        save(output / "status.json", status)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--bundle-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"status": "plan_only", "training_started": False,
                          "worker_sha256": WORKER_SHA256}))
        return
    if args.bundle_dir is None or args.output_dir is None:
        parser.error("Execution requires --bundle-dir and --output-dir")
    supervise(args.bundle_dir.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
