"""Bounded resumable PartialSpoof audio preparation; default is a dry plan.

Only official train/dev archives are supported. All archives are MD5-verified
before tar inspection, and only preselected WAV payloads are written. Transfer
accounting conservatively charges each requested read before the read, including
failed/interrupted reads; network/TLS overhead is outside this payload budget.
"""

import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile
import time
import urllib.request
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from audio_deepfake.resources import memory_snapshot
import soundfile as sf

FILES = {"train": (2_027_101_056, "c4853ddd831e8e96b0e279fc0a512e7e"),
         "dev": (2_027_972_850, "ddd4cd3221b7210ac879f67452fb209e")}
BUDGET = 4_500_000_000
EXTRACT_BUDGET = 256 * 1024**2
FILE_LIMIT = 8 * 1024**2
READ_SIZE = 1024**2
TIMEOUT = 60
JOB_SECONDS = 1800
MAX_AUDIO_SECONDS = 60
DEST = ROOT / "data/partialspoof"
AUDIT = ROOT / "docs/validation/partial_audio.json"
SELECTION = ROOT / "docs/validation/partial_selection_v1.json"
METADATA = ROOT / "docs/validation/partial_metadata.json"


def digest(path, algorithm="sha256"):
    result = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(READ_SIZE), b""):
            result.update(block)
    return result.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def event(split, value):
    """Append-only per-split history; receipts are separate immutable files."""
    path = DEST / f"audio_{split}_events.jsonl"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"time_unix": time.time(), **value}) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def plan():
    return {"dataset": "PartialSpoof v1.2", "record": "5766198",
            "files": {split: {"filename": f"database_{split}.tar.gz", "size_bytes": size,
                               "published_md5": md5,
                               "url": f"https://zenodo.org/records/5766198/files/database_{split}.tar.gz?download=1"}
                      for split, (size, md5) in FILES.items()},
            "transfer_budget_bytes": BUDGET, "extraction_budget_bytes": EXTRACT_BUDGET,
            "max_audio_bytes": FILE_LIMIT, "read_timeout_seconds": TIMEOUT,
            "max_audio_seconds": MAX_AUDIO_SECONDS,
            "job_timeout_seconds": JOB_SECONDS, "eval_access": False,
            "timeout_semantics": "Deadline checked between operations; a blocking read can add up to 60 seconds. Use an external supervisor for a strict wall-clock stop.",
            "minimum_free_ram_bytes": 2 * 1024**3,
            "maximum_process_peak_bytes": 1024**3,
            "disk_reserve_bytes": 20 * 1024**3,
            "source_sha256": digest(Path(__file__)),
            "selection_sha256": None, "metadata_sha256": None}


def guard(deadline, remaining_archive=0):
    if time.monotonic() >= deadline:
        raise RuntimeError("Audio job time budget exhausted")
    memory = memory_snapshot()
    available = memory.get("system_available_ram_bytes")
    peak = memory.get("peak_rss_bytes")
    if available is None or available < 2 * 1024**3 or peak is None or peak > 1024**3:
        raise RuntimeError("Audio RAM guard failed")
    free = shutil.disk_usage(ROOT).free
    if free < 20 * 1024**3 + remaining_archive + EXTRACT_BUDGET:
        raise RuntimeError("Audio disk reserve guard failed")
    return {"memory": memory, "disk_free_bytes": free}


def load_selection():
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    if (metadata.get("status") != "metadata_verified_selection_annotated_audio_alignment_pending"
            or metadata.get("eval_payload_parsed") is not False
            or metadata.get("eval_metadata_extracted") is not False
            or metadata.get("native_metadata_parsed") != {"train": True, "dev": True, "eval": False}):
        raise RuntimeError("Complete verified train/dev-only metadata is required")
    if digest(SELECTION) != metadata.get("selection_sha256"):
        raise RuntimeError("Selection differs from verified metadata")
    exported = metadata["selected_labels_export"]
    labels = (ROOT / exported["path"]).resolve()
    if (not labels.is_relative_to(DEST.resolve()) or not labels.is_file()
            or labels.stat().st_size != exported["size_bytes"]
            or digest(labels) != exported["sha256"] or exported.get("pickle_required") is not False):
        raise RuntimeError("Selected safe label export is missing or changed")
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    if selection.get("dataset") != "PartialSpoof v1.2" or selection.get("seed") != 45:
        raise RuntimeError("Unexpected selection contract")
    for split, count in (("train", 128), ("dev", 64)):
        entries = selection["splits"][split]["selected_records"]
        ids = [entry["utterance_id"] for entry in entries]
        letter = "T" if split == "train" else "D"
        if (len(ids) != count or len(set(ids)) != count
                or any(not re.fullmatch(rf"(?:CON|LA)_{letter}_[0-9]{{7}}", uid) for uid in ids)):
            raise RuntimeError("Unexpected selected IDs or split size")
    return selection


def load_audit(selection):
    pinned = plan()
    pinned["selection_sha256"] = digest(SELECTION)
    pinned["metadata_sha256"] = digest(METADATA)
    history = (any(DEST.glob("audio_*_events.jsonl"))
               or any(DEST.glob("audio_*_complete.json"))
               or any((DEST / f"database_{s}.tar.gz{suffix}").exists()
                      for s in FILES for suffix in ("", ".part")))
    if AUDIT.exists():
        audit = json.loads(AUDIT.read_text(encoding="utf-8"))
        if audit.get("plan", {}).get("selection_sha256") is None:
            if audit.get("charged_payload_bytes", 0) or audit.get("splits") or history:
                raise RuntimeError("Cannot repin an active audio plan")
            audit["plan"] = pinned
        elif audit.get("plan") != pinned:
            raise RuntimeError("Pinned plan/selection/metadata changed")
    else:
        if history:
            raise RuntimeError("Audio history exists without its cumulative transfer journal")
        audit = {"status": "planned", "charged_payload_bytes": 0,
                 "known_received_bytes": 0, "splits": {}, "plan": pinned}
    if not 0 <= audit.get("charged_payload_bytes", -1) <= BUDGET:
        raise RuntimeError("Invalid cumulative transfer budget")
    atomic_json(AUDIT, audit)
    return audit


def download(split, audit, deadline, opener=urllib.request.urlopen):
    size, expected_md5 = FILES[split]
    archive = DEST / f"database_{split}.tar.gz"
    partial = archive.with_suffix(archive.suffix + ".part")
    if archive.is_symlink() or partial.is_symlink():
        raise RuntimeError("Archive and partial paths must not be symlinks")
    if archive.exists():
        if archive.stat().st_size != size or digest(archive, "md5") != expected_md5:
            raise RuntimeError("Existing archive checksum mismatch; preserved for inspection")
        return archive
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > size:
        raise RuntimeError("Partial archive exceeds published size")
    if offset < size:
        guard(deadline, size - offset)
        headers = {"Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(audit["plan"]["files"][split]["url"], headers=headers)
        with opener(request, timeout=TIMEOUT) as response:
            status = response.status
            if response.headers.get("Content-Encoding", "identity") != "identity":
                raise RuntimeError("Compressed HTTP payload is not allowed")
            if status == 206:
                if response.headers.get("Content-Range") != f"bytes {offset}-{size-1}/{size}":
                    raise RuntimeError("Unexpected Content-Range; partial retained")
            elif status != 200 or offset:
                raise RuntimeError("Server did not honor resume; partial retained")
            length = response.headers.get("Content-Length")
            if length is not None and int(length) != size - offset:
                raise RuntimeError("Unexpected response length")
            with partial.open("ab") as stream:
                while offset < size:
                    guard(deadline, size - offset)
                    amount = min(READ_SIZE, size - offset)
                    if audit["charged_payload_bytes"] + amount > BUDGET:
                        raise RuntimeError("Cumulative transfer budget exhausted")
                    audit["charged_payload_bytes"] += amount
                    atomic_json(AUDIT, audit)
                    event(split, {"event": "read_reserved", "bytes": amount, "offset": offset})
                    try:
                        block = response.read(amount)
                    except http.client.IncompleteRead as exc:
                        block = exc.partial
                        stream.write(block)
                        stream.flush()
                        os.fsync(stream.fileno())
                        audit["known_received_bytes"] += len(block)
                        atomic_json(AUDIT, audit)
                        raise RuntimeError("Incomplete read; received partial bytes retained") from exc
                    if not block:
                        raise RuntimeError("Premature response EOF; partial retained")
                    stream.write(block)
                    stream.flush()
                    os.fsync(stream.fileno())
                    audit["known_received_bytes"] += len(block)
                    offset += len(block)
                    atomic_json(AUDIT, audit)
    guard(deadline)
    if partial.stat().st_size != size or digest(partial, "md5") != expected_md5:
        raise RuntimeError("Downloaded archive checksum mismatch; partial preserved")
    partial.replace(archive)
    event(split, {"event": "archive_verified", "md5": expected_md5, "sha256": digest(archive)})
    return archive


def safe_member(member, split):
    name = member.name
    path = PurePosixPath(name)
    if (path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name
            or member.issym() or member.islnk() or not (member.isfile() or member.isdir())):
        raise ValueError("Unsafe tar member")
    if len(path.parts) < 2 or path.parts[:2] != ("database", split):
        # Root database directory is harmless but no other split is allowed.
        if not (member.isdir() and path.parts == ("database",)):
            raise ValueError("Unexpected archive split/path")
    return path


def extract_selected(archive, split, records, deadline):
    size, expected_md5 = FILES[split]
    if archive.stat().st_size != size or digest(archive, "md5") != expected_md5:
        raise RuntimeError("Archive must be verified before tar inspection")
    wanted = {f"database/{split}/con_wav/{r['utterance_id']}.wav": r for r in records}
    found = {}
    total = 0
    target = DEST / "database" / split / "con_wav"
    if not target.resolve().is_relative_to(DEST.resolve()):
        raise RuntimeError("Selected audio destination escapes the data directory")
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as container:
        for member in container:
            guard(deadline)
            path = safe_member(member, split)
            key = str(path)
            if key not in wanted:
                continue
            if not member.isfile() or key in found or not 0 < member.size <= FILE_LIMIT:
                raise ValueError("Invalid or duplicate selected WAV member")
            total += member.size
            if total > EXTRACT_BUDGET:
                raise RuntimeError("Selected extraction byte budget exceeded")
            output = target / path.name
            temporary = output.with_suffix(".wav.tmp")
            if output.is_symlink() or temporary.is_symlink():
                raise RuntimeError("Selected audio output paths must not be symlinks")
            stream = container.extractfile(member)
            with stream, temporary.open("wb") as destination:
                shutil.copyfileobj(stream, destination, READ_SIZE)
            if temporary.stat().st_size != member.size:
                raise RuntimeError("Truncated tar audio")
            info = sf.info(temporary)
            if (info.samplerate != 16_000 or info.channels != 1
                    or not 0 < info.frames <= 16_000 * MAX_AUDIO_SECONDS or info.format != "WAV"):
                raise ValueError("Selected audio must be mono 16000 Hz WAV, longer than 0 and at most 60 seconds")
            audio, rate = sf.read(temporary, dtype="float32", always_2d=False)
            if len(audio) != info.frames or rate != 16_000 or not np.isfinite(audio).all():
                raise ValueError("Invalid decoded selected audio")
            sha256 = digest(temporary)
            if output.exists():
                if digest(output) != sha256:
                    raise RuntimeError("Existing selected audio differs; refusing overwrite")
                temporary.unlink()
            else:
                temporary.replace(output)
            found[key] = {"utterance_id": wanted[key]["utterance_id"], "path": str(path),
                          "size_bytes": member.size, "sha256": sha256, "sample_rate": rate,
                          "channels": 1, "frames": info.frames, "duration_seconds": info.frames / rate}
    if set(found) != set(wanted):
        raise RuntimeError("Selected audio members missing from archive")
    return [found[key] for key in sorted(found)]


def execute(split):
    selection = load_selection()
    DEST.mkdir(parents=True, exist_ok=True)
    lock = DEST / "audio_prepare.lock"
    # A stale lock after process death must be reviewed explicitly, not bypassed.
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        receipt = DEST / f"audio_{split}_complete.json"
        if receipt.exists():
            raise RuntimeError("Split already complete; immutable receipt retained")
        audit = load_audit(selection)
        deadline = time.monotonic() + JOB_SECONDS
        audit["status"] = "running"
        audit["preflight"] = guard(deadline, FILES[split][0])
        atomic_json(AUDIT, audit)
        try:
            archive = download(split, audit, deadline)
            records = extract_selected(archive, split, selection["splits"][split]["selected_records"], deadline)
            result = {"status": "complete", "selection_sha256": digest(SELECTION),
                      "archive_sha256": digest(archive), "archive_md5": FILES[split][1],
                      "records": records, "selected_count": len(records), "eval_payload_read": False}
            # Exclusive creation prevents replacing an earlier completed receipt.
            with receipt.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2)
                stream.write("\n")
            audit["splits"][split] = result
            audit["status"] = "complete" if len(audit["splits"]) == 2 else "partial_complete"
            atomic_json(AUDIT, audit)
            event(split, {"event": "complete", "selected_count": len(records)})
        except Exception as exc:
            audit["status"] = "stopped"
            audit["last_error"] = str(exc)
            atomic_json(AUDIT, audit)
            event(split, {"event": "stopped", "reason": str(exc)})
            raise
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=tuple(FILES), default="train")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute:
        execute(args.split)
    else:
        print(json.dumps({"status": "plan_only", "plan": plan()}, indent=2))


if __name__ == "__main__":
    main()
