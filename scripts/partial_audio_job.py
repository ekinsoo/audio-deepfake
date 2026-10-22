"""Supervise bounded train/dev audio preparation independently of the chat."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from full_run_job import base_python, configure_project_environment


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=["train", "dev"])
    parser.add_argument("--job-dir", type=Path)
    args = parser.parse_args()
    configure_project_environment()
    if args.worker:
        sys.argv = ["prepare_partial_audio.py", "--split", args.worker, "--execute"]
        runpy.run_path(str(ROOT / "scripts/prepare_partial_audio.py"), run_name="__main__")
        return
    directory = args.job_dir.resolve()
    if not directory.is_relative_to(ROOT / "outputs/jobs"):
        raise ValueError("Job directory must stay in outputs/jobs")
    directory.mkdir(parents=True, exist_ok=True)
    status_file = directory / "status.json"
    if status_file.exists():
        raise ValueError("Use a new job directory; existing status is preserved")
    sources = [ROOT / p for p in ("scripts/prepare_partial_audio.py", "scripts/partial_audio_job.py",
               "docs/validation/partial_selection_v1.json", "docs/validation/partial_metadata.json")]
    pins = {p.relative_to(ROOT).as_posix(): sha(p) for p in sources}
    status = {"status": "running", "started_utc": now(), "source_sha256": pins,
              "training_started": False, "eval_enabled": False, "completed_splits": []}
    save(status_file, status)
    child = None
    try:
        from audio_deepfake.resources import memory_snapshot
        import shutil
        status["preflight"] = {"memory": memory_snapshot(), "disk_free_bytes": shutil.disk_usage(ROOT).free}
        for split in ("train", "dev"):
            if any(sha(ROOT / p) != digest for p, digest in pins.items()):
                raise RuntimeError("Pinned source or selection changed")
            receipt = ROOT / f"data/partialspoof/audio_{split}_complete.json"
            if receipt.exists():
                record = json.loads(receipt.read_text())
                if (record.get("status") != "complete"
                        or record.get("selection_sha256") != pins["docs/validation/partial_selection_v1.json"]):
                    raise RuntimeError("Existing split receipt does not match the selection")
                status["completed_splits"].append(split)
                continue
            status["current_split"] = split
            child = subprocess.Popen([base_python(), "-S", "-u", str(Path(__file__).resolve()),
                                      "--worker", split], cwd=ROOT)
            status["worker_pid"] = child.pid
            status["updated_utc"] = now()
            save(status_file, status)
            try:
                code = child.wait(timeout=1865)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=15)
                raise RuntimeError("Hard worker timeout; partial archive retained. Review stale lock before resume.")
            if code:
                raise RuntimeError(f"Audio worker stopped with exit code {code}; inspect partial_audio.json and logs")
            status["completed_splits"].append(split)
        status.update(status="complete", finished_utc=now())
    except BaseException as exc:
        if child is not None and child.poll() is None:
            child.kill()
            child.wait(timeout=15)
        status.update(status="stopped", finished_utc=now(), error=str(exc))
        save(status_file, status)
        raise
    save(status_file, status)


if __name__ == "__main__":
    main()
