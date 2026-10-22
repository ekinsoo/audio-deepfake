"""Small environment and audio checks. No model is trained here."""

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import platform
import pickle
import tempfile
from pathlib import Path

import numpy as np
import scipy
import sklearn
import soundfile as sf

from .audio import load_audio
from .baseline import load_model, score_features
from .features import extract_lfcc
from .manifests import (
    check_audio_files,
    parse_protocol,
    sample_pilot,
    summarize,
    validate_splits,
    write_manifest,
)


def build_manifests(args: argparse.Namespace) -> dict:
    """Build protocol manifests and separate balanced train/dev pilot lists."""
    paths = {"train": args.train_protocol, "dev": args.dev_protocol}
    if args.eval_protocol is not None:
        paths["eval"] = args.eval_protocol
    splits = {split: parse_protocol(path, split) for split, path in paths.items()}
    validate_splits(splits)
    pilots = {
        "train": sample_pilot(splits["train"], args.train_limit, args.seed),
        "dev": sample_pilot(splits["dev"], args.dev_limit, args.seed),
    }
    if args.audio_root is not None:
        for split, rows in splits.items():
            missing = check_audio_files(rows, args.audio_root)
            if missing:
                raise ValueError(f"{split}: {len(missing)} missing audio files; first: {missing[0]}")
    names = [f"{split}.csv" for split in splits]
    names += ["pilot_train.csv", "pilot_dev.csv", "summary.json"]
    if args.eval_protocol is None and (args.output_dir / "eval.csv").exists():
        raise ValueError("Output contains an eval manifest but no eval protocol was supplied. Use a new output directory.")
    if not args.overwrite:
        existing = [name for name in names if (args.output_dir / name).exists()]
        if existing:
            raise ValueError(f"Output already exists: {', '.join(existing)}. Use --overwrite to replace it.")
    summary = {
        "dataset": "ASVspoof2019-LA",
        "audio_validation": "files_exist_only" if args.audio_root is not None else "not_checked",
        "official_distribution_verified": False,
        "note": "Protocol checks do not verify the LA.zip checksum or audio contents.",
        "protocol_sha256": {
            split: hashlib.sha256(path.read_bytes()).hexdigest() for split, path in paths.items()
        },
        "splits": {split: summarize(rows) for split, rows in splits.items()},
        "pilot": {
            "seed": args.seed,
            "policy": "equal class counts; proportional spoof attack quotas; SHA256-ranked IDs",
            "use": "custom train/dev pilot only; not a full benchmark or deployment prevalence estimate",
            "splits": {split: summarize(rows) for split, rows in pilots.items()},
        },
        "eval": "metadata inventory only; excluded from pilot selection and model development",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        write_manifest(rows, args.output_dir / f"{split}.csv")
    for split, rows in pilots.items():
        write_manifest(rows, args.output_dir / f"pilot_{split}.csv")
    summary["manifest_sha256"] = {
        name: hashlib.sha256((args.output_dir / name).read_bytes()).hexdigest()
        for name in names if name.endswith(".csv")
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def environment_info() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("audio-deepfake", "numpy", "scipy", "soundfile", "scikit-learn")
        },
        "libsndfile": sf.__libsndfile_version__,
    }


def smoke_check() -> dict:
    """Check FLAC support using a generated tone, not a dataset recording."""
    sample_rate = 16_000
    time = np.arange(sample_rate // 4, dtype=np.float32) / sample_rate
    expected = (0.25 * np.sin(2 * np.pi * 440 * time)).astype(np.float32)
    with tempfile.TemporaryDirectory(prefix="audio-deepfake-") as directory:
        path = Path(directory) / "tone.flac"
        sf.write(path, expected, sample_rate, subtype="PCM_16")
        actual = load_audio(path)
    if actual.shape != expected.shape:
        raise RuntimeError("FLAC round trip changed the sample count")
    max_error = float(np.max(np.abs(actual - expected)))
    if max_error > 1 / 32768:
        raise RuntimeError(f"FLAC round trip error is too large: {max_error}")
    return {
        "status": "ok",
        "check": "generated-tone FLAC round trip; no dataset or model used",
        "sample_rate": sample_rate,
        "samples": int(actual.size),
        "max_abs_error": max_error,
        "environment": environment_info(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Show Python and dependency versions")
    commands.add_parser("smoke", help="Check 16 kHz mono FLAC read/write")
    inspect = commands.add_parser("inspect", help="Check a local 16 kHz mono recording")
    inspect.add_argument("path", type=Path)
    score = commands.add_parser("score", help="Score one recording using a trusted local model",
                                description="Load only local joblib models you created and trust; loading can execute code.")
    score.add_argument("path", type=Path)
    score.add_argument("--model", type=Path, required=True,
                       help="Trusted local joblib artifact; never load unknown models")
    manifest = commands.add_parser("manifests", help="Validate LA CM protocols and build train/dev pilots")
    manifest.add_argument("--train-protocol", type=Path, required=True)
    manifest.add_argument("--dev-protocol", type=Path, required=True)
    manifest.add_argument("--eval-protocol", type=Path)
    manifest.add_argument("--output-dir", type=Path, default=Path("outputs/manifests"))
    manifest.add_argument("--audio-root", type=Path, help="Check all full-split audio paths under this root, not just the pilot")
    manifest.add_argument("--train-limit", type=int, default=2000)
    manifest.add_argument("--dev-limit", type=int, default=1000)
    manifest.add_argument("--seed", type=int, default=42)
    manifest.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "doctor":
            result = environment_info()
        elif args.command == "smoke":
            result = smoke_check()
        elif args.command == "manifests":
            result = build_manifests(args)
        elif args.command == "score":
            try:
                model_sha256 = hashlib.sha256(args.model.read_bytes()).hexdigest()
                model = load_model(args.model)
            except (pickle.UnpicklingError, EOFError, AttributeError, KeyError, TypeError, ImportError) as error:
                raise ValueError("Could not load the trusted local baseline artifact") from error
            waveform = load_audio(args.path, sample_rate=model.lfcc_config.sample_rate)
            result = {
                "recording_score": score_features(model, extract_lfcc(waveform, model.lfcc_config)),
                "score_direction": "higher_is_bonafide",
                "sample_rate": model.lfcc_config.sample_rate,
                "samples": int(waveform.size),
                "duration_seconds": waveform.size / model.lfcc_config.sample_rate,
                "model_sha256": model_sha256,
                "lfcc_config": asdict(model.lfcc_config),
                "gmm_config": asdict(model.gmm_config),
                "scope": "single_recording_diagnostic",
            }
        else:
            waveform = load_audio(args.path)
            result = {
                "sample_rate": 16_000,
                "samples": int(waveform.size),
                "duration_seconds": waveform.size / 16_000,
                "peak_amplitude": float(np.max(np.abs(waveform))),
            }
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(json.dumps(result, indent=2))
    return 0
