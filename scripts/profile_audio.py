"""Measure selected train/dev audio and LFCC extraction without training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import sys
import time
import tomllib
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from audio_deepfake.audio import load_audio
from audio_deepfake.cli import environment_info
from audio_deepfake.features import LFCCConfig, extract_lfcc
from audio_deepfake.manifests import ManifestRow, check_audio_files, summarize, validate_splits
from audio_deepfake.resources import memory_snapshot


def profile_manifest(manifest: Path, audio_root: Path, journal_path: Path,
                     output_dir: Path, *, expected_count: int = 100,
                     config_path: Path | None = None) -> dict:
    """Profile a frozen, hash-verified selection in one fresh sequential process."""
    with manifest.open(encoding="utf-8", newline="") as source:
        rows = [ManifestRow(**row) for row in csv.DictReader(source)]
    if len(rows) != expected_count or not rows:
        raise ValueError(f"Expected exactly {expected_count} manifest rows, got {len(rows)}")
    splits = defaultdict(list)
    for row in rows:
        if row.split not in {"train", "dev"}:
            raise ValueError("Profiling only accepts train/dev recordings")
        splits[row.split].append(row)
    validate_splits(splits)
    if expected_count == 100:
        for split in ("train", "dev"):
            group = splits[split]
            if len(group) != 50 or summarize(group)["labels"] != {"bonafide": 25, "spoof": 25}:
                raise ValueError("The 100-record profile requires 50 train/50 dev, each with 25 records per class")
    missing = check_audio_files(rows, audio_root)
    if missing:
        raise ValueError(f"{len(missing)} selected audio files are missing; first: {missing[0]}")
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    records = journal["records"]
    # Audit cached files before measuring. These are observed member hashes,
    # not independent official checksums for the complete dataset archive.
    hashes = {}
    for row in rows:
        payload = (audio_root / row.audio_path).read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        recorded = records[row.audio_path]
        if digest != recorded["sha256"] or len(payload) != recorded["size_bytes"]:
            raise ValueError(f"Cached audio differs from download journal: {row.utterance_id}")
        hashes[row.utterance_id] = digest
    if output_dir.exists():
        raise ValueError("Profile output already exists; use a new directory for a new run")
    if config_path is None:
        config = LFCCConfig()
    else:
        with config_path.open("rb") as source:
            plan = tomllib.load(source)
        config = LFCCConfig(**plan["features"])
    measurements = []
    started_utc = datetime.now(timezone.utc).isoformat()
    cpu_identifier = platform.processor()
    if os.name == "nt":
        import winreg

        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                cpu_identifier = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    with threadpool_limits(limits=1):
        start_memory = memory_snapshot()
        pools = threadpool_info()
        start_wall = time.perf_counter()
        start_cpu = time.process_time()
        for index, row in enumerate(rows, 1):
            path = audio_root / row.audio_path
            started = time.perf_counter()
            waveform = load_audio(path, sample_rate=config.sample_rate)
            loaded = time.perf_counter()
            features = extract_lfcc(waveform, config)
            finished = time.perf_counter()
            measurements.append({
                "split": row.split, "utterance_id": row.utterance_id,
                "label": row.label, "attack_id": row.attack_id,
                "sha256": hashes[row.utterance_id], "audio_bytes": path.stat().st_size,
                "samples": int(waveform.size), "duration_seconds": waveform.size / config.sample_rate,
                "frames": features.shape[0], "dimensions": features.shape[1],
                "feature_bytes": features.nbytes, "feature_dtype": str(features.dtype),
                "load_seconds": loaded - started, "feature_seconds": finished - loaded,
                "load_and_feature_seconds": finished - started,
            })
            del waveform, features
            if index % 10 == 0:
                print(f"Profiled {index}/{len(rows)} recordings", file=sys.stderr, flush=True)
        wall = time.perf_counter() - start_wall
        cpu = time.process_time() - start_cpu
        end_memory = memory_snapshot()
    duration = sum(item["duration_seconds"] for item in measurements)
    feature_time = sum(item["feature_seconds"] for item in measurements)
    summary = {
        "status": "ok", "records": len(rows),
        "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "cpu_identifier": cpu_identifier, "logical_cpu_count": os.cpu_count(),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "download_journal_sha256": hashlib.sha256(journal_path.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest() if config_path is not None else None,
        "environment": environment_info(), "lfcc_config": asdict(config),
        "source_sha256": {
            name: hashlib.sha256((Path(__file__).resolve().parents[1] / name).read_bytes()).hexdigest()
            for name in ("src/audio_deepfake/features.py", "src/audio_deepfake/audio.py",
                         "src/audio_deepfake/resources.py", "scripts/profile_audio.py")
        },
        "workers": 1, "thread_limit": 1, "threadpools": pools,
        "memory_before": start_memory, "memory_after": end_memory,
        "free_disk_bytes_after": shutil.disk_usage(audio_root).free,
        "processing_wall_seconds": wall, "processing_cpu_seconds": cpu,
        "audio_duration_seconds": duration,
        "audio_bytes": sum(item["audio_bytes"] for item in measurements),
        "feature_bytes_if_cached": sum(item["feature_bytes"] for item in measurements),
        "feature_seconds": feature_time,
        "feature_rtf": feature_time / duration,
        "load_seconds": sum(item["load_seconds"] for item in measurements),
        "per_file_feature_seconds_p50": float(np.percentile([m["feature_seconds"] for m in measurements], 50)),
        "per_file_feature_seconds_p95": float(np.percentile([m["feature_seconds"] for m in measurements], 95)),
        "first_record_feature_seconds": measurements[0]["feature_seconds"],
        "splits": {split: summarize(group) for split, group in splits.items()},
        "limitations": [
            "No model training, inference score, EER or t-DCF was measured.",
            "Filesystem cache was not controlled; imports and preflight hash checks are outside processing time.",
            "Peak RSS is a process-lifetime high-water mark including imports and preflight, not training RAM.",
            "Feature arrays were discarded per recording; storage is estimated from array sizes, not written caches.",
            "Hashes were observed during member download; the full LA.zip checksum was not verified.",
            "Threadpool library paths are omitted from this public summary.",
        ],
    }
    # Native library paths may contain a local account name.
    for pool in summary["threadpools"]:
        pool.pop("filepath", None)
    projection = {}
    for split, target in [("train", 2000), ("dev", 1000)]:
        group = [item for item in measurements if item["split"] == split]
        if group:
            projection[split] = {
                "target_records": target, "sample_records": len(group),
                "feature_seconds": sum(item["feature_seconds"] for item in group) / len(group) * target,
                "feature_bytes_if_cached": sum(item["feature_bytes"] for item in group) / len(group) * target,
                "audio_bytes": sum(item["audio_bytes"] for item in group) / len(group) * target,
            }
    summary["rough_pilot_projection"] = projection
    summary["projection_note"] = "Split-weighted sample means, not a training-time/RAM estimate or download-time guarantee."
    output_dir.mkdir(parents=True)
    with (output_dir / "recordings.csv").open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=measurements[0].keys())
        writer.writeheader()
        writer.writerows(measurements)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("outputs/audio-pilot/selected100.csv"))
    parser.add_argument("--audio-root", type=Path, default=Path("data/asvspoof2019/LA"))
    parser.add_argument("--journal", type=Path, default=Path("outputs/audio-pilot/download_report.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/profile-audio-100"))
    parser.add_argument("--config", type=Path, default=Path("configs/pilot.toml"))
    args = parser.parse_args()
    try:
        result = profile_manifest(args.manifest, args.audio_root, args.journal, args.output_dir,
                                  config_path=args.config)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        parser.exit(1, f"Profile error: {error}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
