"""Run a frozen, hash-verified train/dev LFCC-GMM smoke test or pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import time
import tomllib
from datetime import datetime, timezone

from threadpoolctl import threadpool_info, threadpool_limits

from audio_deepfake.baseline import GMMConfig, train_baseline, score_rows, save_model
from audio_deepfake.cli import environment_info
from audio_deepfake.features import LFCCConfig
from audio_deepfake.manifests import ManifestRow, check_audio_files, summarize, validate_splits
from audio_deepfake.metrics import DEFAULT_REFERENCE, REFERENCE_SHA256, evaluate_scores
from audio_deepfake.resources import memory_snapshot


REPO_ROOT = Path(__file__).resolve().parents[1]
DEV_ASV_SHA256 = "fe8114cf608c3062c141df5979892eba9664aa37a8c99e97157f129d679d3389"
DEFAULT_ASV = REPO_ROOT / "data/metadata/asvspoof2019/metrics/ASVspoof2019.LA.asv.dev.gi.trl.scores.txt"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _public_config(config: dict) -> dict:
    """Keep configuration useful without exposing machine-specific roots."""
    public = json.loads(json.dumps(config))
    for key, value in public.get("paths", {}).items():
        if isinstance(value, str) and Path(value).is_absolute():
            public["paths"][key] = f"<local>/{Path(value).name}"
    return public


def preflight(config: dict, output_dir: Path) -> dict:
    """Validate the whole experiment before fitting any model or writing outputs."""
    if output_dir.exists():
        raise ValueError("Output already exists; use a new directory for a new run")
    experiment, data, paths = config["experiment"], config["data"], config["paths"]
    if experiment["stage"] not in {"smoke", "pilot"}:
        raise ValueError("Experiment stage must be smoke or pilot")
    if data.get("eval_enabled") is not False:
        raise ValueError("eval_enabled must be false")
    if data.get("dataset") != "ASVspoof2019-LA" or data.get("version") != "Zenodo-6906306-v1":
        raise ValueError("Dataset/version must be ASVspoof2019-LA / Zenodo-6906306-v1")
    lfcc = LFCCConfig(**config["features"])
    if data.get("sample_rate", 16000) != 16000 or lfcc.sample_rate != 16000:
        raise ValueError("Dataset and features sample_rate must both be 16000")
    settings = config["model"]
    allowed_model = {"family", "n_components", "covariance_type", "max_total_training_frames", "max_iter", "reg_covar", "tol"}
    if set(settings) - allowed_model:
        raise ValueError("Unknown model configuration keys: " + ", ".join(sorted(set(settings) - allowed_model)))
    if settings.get("family", "lfcc-gmm") != "lfcc-gmm" or settings.get("covariance_type", "diag") != "diag":
        raise ValueError("This runner supports only diagonal lfcc-gmm")
    budget = settings["max_total_training_frames"]
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0 or budget % 2:
        raise ValueError("max_total_training_frames must be a positive even integer")
    gmm = GMMConfig(n_components=settings["n_components"], seed=experiment["seed"],
                    frames_per_class=budget // 2, max_iter=settings.get("max_iter", 50),
                    reg_covar=settings.get("reg_covar", 1e-6), tol=settings.get("tol", 1e-3))
    manifest, root, journal_path = (Path(paths[key]) for key in
                                    ("selection_manifest", "audio_root", "audio_journal"))
    manifest_hash = digest(manifest)
    if manifest_hash != paths["selection_sha256"]:
        raise ValueError("Selection manifest differs from the pinned selection_sha256")
    with manifest.open(encoding="utf-8", newline="") as source:
        rows = [ManifestRow(**record) for record in csv.DictReader(source)]
    groups = {"train": [], "dev": []}
    for row in rows:
        if row.split not in groups:
            raise ValueError("Only train/dev rows are allowed; eval is excluded")
        groups[row.split].append(row)
    validate_splits(groups)
    for split in groups:
        count = data[f"{split}_limit"]
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0 or count % 2:
            raise ValueError(f"{split}_limit must be a positive even integer")
        actual = summarize(groups[split])
        if actual["count"] != count or actual["labels"] != {"bonafide": count // 2, "spoof": count // 2}:
            raise ValueError(f"{split} must contain exactly {count} balanced records")
    missing = check_audio_files(rows, root)
    if missing:
        raise ValueError(f"Missing selected audio: {missing[0]}")
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    if journal.get("selection_sha256") not in (None, manifest_hash):
        raise ValueError("Journal selection hash differs from pinned manifest")
    hashes = {}
    for row in rows:
        record = journal["records"].get(row.audio_path)
        if record is None:
            raise ValueError(f"Missing download journal entry: {row.utterance_id}")
        audio = root / row.audio_path
        sha256 = digest(audio)
        if sha256 != record["sha256"] or audio.stat().st_size != record["size_bytes"]:
            raise ValueError(f"Audio differs from download journal: {row.utterance_id}")
        hashes[row.utterance_id] = {"sha256": sha256, "size_bytes": audio.stat().st_size}
    asv = Path(paths.get("asv_scores", DEFAULT_ASV))
    reference = Path(paths.get("reference_evaluator", DEFAULT_REFERENCE))
    if digest(asv) != DEV_ASV_SHA256:
        raise ValueError("ASV scores do not match the pinned official LA dev file")
    if digest(reference) != REFERENCE_SHA256:
        raise ValueError("Evaluator differs from the pinned official 2019 reference")
    return {"groups": groups, "audio_root": root, "lfcc": lfcc, "gmm": gmm,
            "asv": asv, "reference": reference, "audio_hashes": hashes,
            "input_sha256": {"selection_manifest": manifest_hash, "audio_journal": digest(journal_path),
                             "asv_scores": digest(asv), "reference_evaluator": digest(reference)},
            "source_manifest_sha256": journal.get("source_manifest_sha256", {}),
            "download_provenance": {key: journal.get(key) for key in
                                    ("archive_md5", "archive_size_bytes", "archive_size", "hash_provenance")}}


def run_experiment(config_path: Path, output_dir: Path) -> dict:
    with config_path.open("rb") as source:
        config = tomllib.load(source)
    plan = preflight(config, output_dir)
    output_dir.mkdir(parents=True)
    (output_dir / "config.json").write_text(json.dumps(_public_config(config), indent=2) + "\n", encoding="utf-8")
    started = datetime.now(timezone.utc).isoformat()
    with threadpool_limits(limits=1):
        before = memory_snapshot()
        pools = [{key: value for key, value in pool.items() if key != "filepath"} for pool in threadpool_info()]
        wall, cpu = time.perf_counter(), time.process_time()
        model = train_baseline(plan["groups"]["train"], plan["audio_root"],
                               lfcc_config=plan["lfcc"], gmm_config=plan["gmm"])
        trained = time.perf_counter()
        pairs = score_rows(model, plan["groups"]["dev"], plan["audio_root"])
        scored = time.perf_counter()
        metrics = evaluate_scores(plan["groups"]["dev"], pairs, plan["asv"], plan["reference"], evaluation_scope="pilot")
        finished = time.perf_counter()
        cpu_seconds = time.process_time() - cpu
        after = memory_snapshot()
    summary = {
        "status": "ok", "experiment": config["experiment"],
        "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "smoke" if config["experiment"]["stage"] == "smoke" else "pilot",
        "splits": {key: summarize(rows) for key, rows in plan["groups"].items()},
        "input_sha256": {**plan["input_sha256"], "config": digest(config_path)},
        "source_manifest_sha256": plan["source_manifest_sha256"],
        "download_provenance": plan["download_provenance"],
        "audio_files": plan["audio_hashes"], "model_metadata": model.metadata,
        "metrics": metrics, "environment": environment_info(),
        "workers": 1, "thread_limit": 1, "threadpools": pools,
        "memory_before": before, "memory_after": after,
        "free_disk_bytes_after": shutil.disk_usage(plan["audio_root"]).free,
        "timings": {"training_seconds": trained - wall, "dev_scoring_seconds": scored - trained,
                    "metric_seconds": finished - scored, "processing_wall_seconds": finished - wall,
                    "processing_cpu_seconds": cpu_seconds},
        "source_sha256": {name: digest(REPO_ROOT / name) for name in (
            "scripts/run_baseline.py", "src/audio_deepfake/baseline.py", "src/audio_deepfake/features.py",
            "src/audio_deepfake/manifests.py", "src/audio_deepfake/metrics.py", "src/audio_deepfake/audio.py",
            "src/audio_deepfake/resources.py")},
        "limitations": [
            "Custom balanced train/dev subset; not a full official benchmark result.",
            "Dev EER and min t-DCF choose thresholds on dev; no deployment or held-out eval claim.",
            "Small sklearn GMM adaptation, not an exact official MATLAB baseline reproduction.",
            ("Journal reports published archive MD5 verification; this runner verifies selected member hashes only."
             if str(plan["download_provenance"].get("hash_provenance", "")).startswith("Archive verified against published MD5;")
             and plan["download_provenance"].get("archive_md5")
             else "Per-member hashes match the observed download journal; full LA.zip checksum was not verified."),
            "Peak RSS is a process-lifetime high-water mark; filesystem cache was not controlled.",
            "Model artifact uses joblib; load only local files you created and trust.",
        ],
    }
    save_model(model, output_dir / "model.joblib")
    score_map = dict(pairs)
    with (output_dir / "scores.csv").open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(("utterance_id", "split", "label", "attack_id", "score"))
        for row in plan["groups"]["dev"]:
            writer.writerow((row.utterance_id, row.split, row.label, row.attack_id, score_map[row.utterance_id]))
    summary["output_sha256"] = {name: digest(output_dir / name) for name in ("model.joblib", "scores.csv", "config.json")}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/smoke.toml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run_experiment(args.config, args.output_dir)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        parser.exit(1, f"Baseline error: {error}\n")
    print(json.dumps({"status": result["status"], "scope": result["scope"],
                      "metrics": result["metrics"], "timings": result["timings"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
