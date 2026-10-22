"""Fetch at most 100 original train/dev FLAC members; never fetch an archive."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import tempfile
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener

import soundfile as sf

from audio_deepfake.audio import load_audio
from audio_deepfake.manifests import ManifestRow, sample_pilot, validate_splits, write_manifest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_protocols import ZenodoRedirectHandler, validate_url

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "https://zenodo.org/api/records/6906306/files/LA.zip/container/LA/"
FILE_CAP = 2 * 1024 * 1024
TOTAL_CAP = 32 * 1024 * 1024


def select_records(directory: Path) -> list[ManifestRow]:
    selected = []
    for split in ("train", "dev"):
        with (directory / f"pilot_{split}.csv").open(encoding="utf-8") as source:
            rows = [ManifestRow(**row) for row in csv.DictReader(source)]
        if {row.split for row in rows} != {split}:
            raise ValueError("Input pilot split mismatch")
        selected.extend(sample_pilot(rows, 50, seed=42))
    validate_splits({split: [r for r in selected if r.split == split] for split in ("train", "dev")})
    return selected


def inspect_audio(path: Path) -> dict:
    info = sf.info(path)
    if info.format != "FLAC" or info.subtype != "PCM_16":
        raise ValueError("Expected original PCM16 FLAC")
    waveform = load_audio(path)
    return {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size, "sample_rate": info.samplerate,
            "channels": info.channels, "subtype": info.subtype,
            "duration_seconds": len(waveform) / info.samplerate}


def safe_destination(root: Path, row: ManifestRow) -> Path:
    if row.split not in {"train", "dev"}:
        raise ValueError("Eval is forbidden in the audio pilot")
    expected = f"ASVspoof2019_LA_{row.split}/flac/{row.utterance_id}.flac"
    if row.audio_path != expected or "/" in row.utterance_id or "\\" in row.utterance_id:
        raise ValueError("Noncanonical audio path")
    result = root / row.audio_path
    if result.is_symlink() or not result.resolve().is_relative_to(root.resolve()):
        raise ValueError("Audio destination escapes data root")
    return result


def save_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def fetch_one(row: ManifestRow, destination: Path, report: dict, *, opener=None, budget_lock=None) -> dict:
    url = SOURCE + row.audio_path
    validate_url(url)
    opener = opener or build_opener(ZenodoRedirectHandler())
    budget_lock = budget_lock or threading.Lock()
    for attempt in range(3):  # Initial request plus at most two retries.
        temporary = None
        try:
            request = Request(url, headers={"User-Agent": "audio-deepfake-small-pilot/0.1"})
            with opener.open(request, timeout=30) as response:
                validate_url(response.geturl())
                length = response.headers.get("Content-Length")
                if length is not None and not 0 < int(length) <= FILE_CAP:
                    raise ValueError("Response exceeds 2 MiB file cap or is empty")
                destination.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".part", delete=False) as out:
                    temporary = Path(out.name)
                    count = 0
                    while True:
                        if length is not None and count == int(length):
                            break
                        with budget_lock:
                            allowance = min(FILE_CAP - count, TOTAL_CAP - report["network_bytes"])
                            if allowance <= 0:
                                # Unknown length at the cap is rejected conservatively.
                                raise ValueError("Download byte cap reached before known EOF")
                            chunk = response.read(min(65536, allowance))
                            report["network_bytes"] += len(chunk)
                        if not chunk:
                            break
                        count += len(chunk)
                        out.write(chunk)
            result = inspect_audio(temporary)
            if length is not None and result["size_bytes"] != int(length):
                raise ValueError("Truncated response")
            result["url"] = url
            temporary.replace(destination)
            return result
        except HTTPError as exc:
            report["errors"].append({"audio_path": row.audio_path, "attempt": attempt + 1, "error": str(exc)})
            if exc.code in {403, 429} or exc.code < 500 or attempt == 2:
                raise
            time.sleep(2 ** (attempt + 1))
        except (URLError, TimeoutError) as exc:
            report["errors"].append({"audio_path": row.audio_path, "attempt": attempt + 1, "error": str(exc)})
            if attempt == 2:
                raise
            time.sleep(2 ** (attempt + 1))
        finally:
            if temporary:
                temporary.unlink(missing_ok=True)
    raise RuntimeError("No download result")


def run(manifest_dir: Path, data_root: Path, output_dir: Path, limit: int = 100, workers: int = 1) -> dict:
    if not 1 <= limit <= 100:
        raise ValueError("Limit must be between 1 and 100")
    if workers not in {1, 2}:
        raise ValueError("Workers must be 1 or 2")
    rows = select_records(manifest_dir)
    source_hashes = {f"pilot_{split}.csv": hashlib.sha256(
        (manifest_dir / f"pilot_{split}.csv").read_bytes()).hexdigest() for split in ("train", "dev")}
    output_dir.mkdir(parents=True, exist_ok=True)
    selection = output_dir / "selected100.csv"
    if selection.exists():
        with selection.open(encoding="utf-8") as source:
            previous = [ManifestRow(**row) for row in csv.DictReader(source)]
        if previous != rows:
            raise ValueError("Existing selection differs; refusing to alter it")
    else:
        write_manifest(rows, selection)
    journal = output_dir / "download_report.json"
    report = json.loads(journal.read_text(encoding="utf-8")) if journal.exists() else {
        "source_record": "https://zenodo.org/records/6906306", "selection_seed": 42,
        "hash_provenance": "Locally observed member SHA-256; no independently published member checksum.",
        "network_bytes": 0, "elapsed_seconds": 0, "records": {}, "errors": []}
    if report.get("source_manifest_sha256", source_hashes) != source_hashes:
        raise ValueError("Pilot source manifests changed")
    report["source_manifest_sha256"] = source_hashes
    report["selection_policy"] = "50 train + 50 dev, 25 bonafide + 25 spoof per split; proportional attacks; seed 42; existing pilots only"
    report["selection_sha256"] = hashlib.sha256(selection.read_bytes()).hexdigest()
    started = time.monotonic()
    elapsed_before = report["elapsed_seconds"]
    current_path = None
    budget_lock = threading.Lock()
    try:
        pending = []
        for row in rows[:limit]:
            current_path = row.audio_path
            destination = safe_destination(data_root, row)
            cached = report["records"].get(row.audio_path)
            if destination.exists():
                if cached is None or inspect_audio(destination) != {k: v for k, v in cached.items() if k != "url"}:
                    raise ValueError(f"Unjournaled or changed cached audio: {row.audio_path}")
                print(f"verified: {row.utterance_id}", flush=True)
                continue
            if cached is not None:
                raise ValueError(f"Journaled audio missing: {row.audio_path}")
            pending.append((row, destination))
        # Batches contain at most two tasks. Finish and journal the other
        # in-flight result before stopping on a failure; no replacement files.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for offset in range(0, len(pending), workers):
                batch = pending[offset:offset + workers]
                futures = [(row, pool.submit(fetch_one, row, destination, report,
                                             budget_lock=budget_lock)) for row, destination in batch]
                failure = None
                for row, future in futures:
                    current_path = row.audio_path
                    try:
                        result = future.result()
                        report["records"][row.audio_path] = result
                        print(f"downloaded: {row.utterance_id}", flush=True)
                    except Exception as exc:
                        report["errors"].append({"error": str(exc), "stage": "member", "audio_path": row.audio_path})
                        failure = failure or exc
                report["elapsed_seconds"] = elapsed_before + time.monotonic() - started
                save_report(report, journal)
                if failure is not None:
                    raise failure
                time.sleep(0.5)
    except Exception as exc:
        report["errors"].append({"error": str(exc), "stage": "run", "audio_path": current_path})
        raise
    finally:
        report["elapsed_seconds"] = elapsed_before + time.monotonic() - started
        save_report(report, journal)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, default=ROOT / "outputs/manifests/asvspoof2019-la-seed42")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/asvspoof2019/LA")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/audio-pilot")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    try:
        report = run(args.manifest_dir, args.data_root, args.output_dir, args.limit, args.workers)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Pilot fetch error: {exc}\n")
    print(f"Verified {len(report['records'])} files; network bytes={report['network_bytes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
