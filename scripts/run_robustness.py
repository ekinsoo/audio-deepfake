"""Run the frozen narrowband diagnostic on a trusted completed pilot model."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
import tomllib

from threadpoolctl import threadpool_info, threadpool_limits

from audio_deepfake.audio import load_audio
from audio_deepfake.baseline import GMMConfig, load_model, score_features
from audio_deepfake.cli import environment_info
from audio_deepfake.features import LFCCConfig, extract_lfcc
from audio_deepfake.manifests import summarize
from audio_deepfake.metrics import evaluate_scores
from audio_deepfake.reliability import (PROTOCOL, calibration_threshold,
                                       evaluate_error_ranking, partition_dev)
from audio_deepfake.resources import memory_snapshot
from audio_deepfake.transforms import TRANSFORM_SPEC, narrowband_roundtrip

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_baseline import REPO_ROOT, _public_config, digest, preflight

DEFAULT_AUDIT = REPO_ROOT / "docs/validation/reliability_partition_seed42.json"
DEFAULT_PROTOCOL = REPO_ROOT / "docs/validation/robustness_protocol_v1.md"
FROZEN_SELECTION_SHA256 = "fe2811f468705c571a5e2e617d0623d3d18fdd61782129bf07e5dc2d11b472df"


def _json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_clean_scores(path: Path, rows) -> dict[str, float]:
    expected = {row.utterance_id: row for row in rows}
    result = {}
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != ["utterance_id", "split", "label", "attack_id", "score"]:
            raise ValueError("Unexpected clean score CSV fields")
        for record in reader:
            utterance = record["utterance_id"]
            if utterance not in expected or utterance in result:
                raise ValueError("Clean score IDs must exactly match dev without duplicates")
            row = expected[utterance]
            if any(record[key] != getattr(row, key) for key in ("split", "label", "attack_id")):
                raise ValueError("Clean score metadata differs from selected dev rows")
            score = float(record["score"])
            if not math.isfinite(score):
                raise ValueError("Clean score must be finite")
            result[utterance] = score
    if set(result) != set(expected):
        raise ValueError("Clean score IDs must exactly match dev without duplicates")
    return result


def prepare_run(baseline_dir: Path, config_path: Path, audit_path: Path,
                protocol_path: Path, output_dir: Path) -> dict:
    with config_path.open("rb") as source:
        config = tomllib.load(source)
    plan = preflight(config, output_dir)
    if plan["input_sha256"]["selection_manifest"] != FROZEN_SELECTION_SHA256:
        raise ValueError("Selection differs from frozen seed42 pilot")
    if (config["experiment"]["stage"] != "pilot" or config["experiment"]["seed"] != 42
            or config["data"]["train_limit"] != 2000 or config["data"]["dev_limit"] != 1000
            or plan["gmm"] != GMMConfig() or plan["lfcc"] != LFCCConfig()):
        raise ValueError("Requires the frozen seed42 2000/1000 32-component pilot configuration")
    summary_path = baseline_dir / "summary.json"
    baseline = json.loads(summary_path.read_text(encoding="utf-8"))
    if baseline.get("status") != "ok" or baseline.get("scope") != "pilot" or baseline.get("experiment") != config["experiment"]:
        raise ValueError("Baseline must be the completed matching pilot")
    if baseline.get("splits") != {key: summarize(rows) for key, rows in plan["groups"].items()}:
        raise ValueError("Baseline train/dev counts differ from selected pilot")
    expected_inputs = {**plan["input_sha256"], "config": digest(config_path)}
    if baseline.get("input_sha256") != expected_inputs:
        raise ValueError("Baseline configuration or input hashes changed")
    for name in ("model.joblib", "scores.csv", "config.json"):
        if baseline.get("output_sha256", {}).get(name) != digest(baseline_dir / name):
            raise ValueError(f"Baseline artifact SHA-256 mismatch: {name}")
    # Check executable inference sources before loading the trusted joblib file.
    for name in ("src/audio_deepfake/baseline.py", "src/audio_deepfake/features.py", "src/audio_deepfake/audio.py"):
        if baseline.get("source_sha256", {}).get(name) != digest(REPO_ROOT / name):
            raise ValueError(f"Baseline inference source changed: {name}")
    calibration, analysis, partition = partition_dev(plan["groups"]["train"], plan["groups"]["dev"])
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("protocol") != PROTOCOL or any(audit.get(key) != value for key, value in partition.items()):
        raise ValueError("Frozen partition audit differs from recomputed speaker groups")
    if audit.get("selection_sha256") != expected_inputs["selection_manifest"]:
        raise ValueError("Frozen partition selection differs from pilot")
    if audit.get("protocol_sha256") != digest(protocol_path):
        raise ValueError("Frozen protocol snapshot SHA-256 mismatch")
    if audit.get("source_sha256") != digest(REPO_ROOT / "src/audio_deepfake/reliability.py"):
        raise ValueError("Frozen reliability helper SHA-256 mismatch")
    clean = read_clean_scores(baseline_dir / "scores.csv", plan["groups"]["dev"])
    # No analysis scores or labels enter threshold selection.
    threshold = calibration_threshold(calibration, [(row.utterance_id, clean[row.utterance_id]) for row in calibration])
    model = load_model(baseline_dir / "model.joblib")
    if model.metadata != baseline.get("model_metadata"):
        raise ValueError("Loaded model metadata differs from baseline summary")
    if (model.gmm_config != plan["gmm"] or model.lfcc_config != plan["lfcc"]
            or model.metadata.get("gmm_config") != asdict(plan["gmm"])
            or model.metadata.get("lfcc_config") != asdict(plan["lfcc"])
            or model.metadata.get("training_ids") != sorted(row.utterance_id for row in plan["groups"]["train"])):
        raise ValueError("Loaded model configuration or training IDs differ from pilot")
    for mixture in (model.bonafide, model.spoof):
        if mixture.n_components != 32 or mixture.covariance_type != "diag":
            raise ValueError("Loaded mixture differs from 32-component diagonal pilot")
    return {"plan": plan, "config": config, "baseline": baseline, "model": model,
            "calibration": calibration, "analysis": analysis, "partition": partition,
            "clean": clean, "threshold": threshold,
            "provenance": {"baseline_summary_sha256": digest(summary_path),
                           "model_sha256": digest(baseline_dir / "model.joblib"),
                           "clean_scores_sha256": digest(baseline_dir / "scores.csv"),
                           "partition_audit_sha256": digest(audit_path),
                           "protocol_snapshot_sha256": digest(protocol_path),
                           "config_sha256": digest(config_path)}}


def select_failure_examples(ranking: dict) -> dict:
    """Illustrative errors, not a random or performance-estimation sample."""
    records = ranking["records"]
    errors = [record for record in records if record["error"]]
    quota = math.ceil(len(records) / 4)
    delta_top = {r["utterance_id"] for r in sorted(records, key=lambda r: (-r["abs_delta"], r["utterance_id"]))[:quota]}
    margin_top = {r["utterance_id"] for r in sorted(records, key=lambda r: (-r["negative_margin"], r["utterance_id"]))[:quota]}
    strata = {
        "negative_delta_error": [r for r in errors if r["delta"] < 0],
        "positive_delta_error": [r for r in errors if r["delta"] > 0],
        "delta_top_quarter_only_error": [r for r in errors if r["utterance_id"] in delta_top - margin_top],
        "margin_top_quarter_only_error": [r for r in errors if r["utterance_id"] in margin_top - delta_top],
    }
    selected, availability = {}, {}
    for name, group in strata.items():
        group = sorted(group, key=lambda r: r["utterance_id"])
        availability[name] = {"eligible_count": len(group), "status": "available" if group else "not available",
                              "representative_id": group[0]["utterance_id"] if group else None}
        if group:
            selected[group[0]["utterance_id"]] = group[0]
    for record in sorted(errors, key=lambda r: (-r["abs_delta"], r["utterance_id"])):
        if len(selected) >= 10:
            break
        selected[record["utterance_id"]] = record
    return {"selection_rule": "Analysis errors only. First ID in each delta-sign and top-quarter-disagreement stratum, deduplicate; fill to 10 by descending abs_delta then ID. Top quarters use ceil(N/4) of all analysis records, ties by ID.",
            "availability": availability, "available_error_count": len(errors),
            "examples": list(selected.values())}


def run_experiment(baseline_dir: Path, config_path: Path, audit_path: Path,
                   protocol_path: Path, output_dir: Path) -> dict:
    prepared = prepare_run(baseline_dir, config_path, audit_path, protocol_path, output_dir)
    plan, model, clean = prepared["plan"], prepared["model"], prepared["clean"]
    output_dir.mkdir(parents=True)
    calibration_record = {"protocol": PROTOCOL, "threshold": prepared["threshold"],
                          "rule": "Clean BF ascending scores[n//20]; bonafide iff clean score >= threshold",
                          "target_calibration_bonafide_acceptance": 0.95,
                          "partition": prepared["partition"]["calibration"],
                          "clean_scores_sha256": prepared["provenance"]["clean_scores_sha256"],
                          "saved_utc": datetime.now(timezone.utc).isoformat()}
    _json(output_dir / "calibration.json", calibration_record)
    started = datetime.now(timezone.utc).isoformat()
    per_record, narrow = [], {}
    with threadpool_limits(limits=1):
        before = memory_snapshot()
        pools = [{key: value for key, value in pool.items() if key != "filepath"} for pool in threadpool_info()]
        wall, cpu = time.perf_counter(), time.process_time()
        for row in sorted(plan["groups"]["dev"], key=lambda row: row.utterance_id):
            tick = time.perf_counter()
            waveform = load_audio(plan["audio_root"] / row.audio_path, sample_rate=16000)
            loaded = time.perf_counter()
            transformed = narrowband_roundtrip(waveform, sample_rate=16000)
            transformed_at = time.perf_counter()
            features = extract_lfcc(transformed, model.lfcc_config)
            features_at = time.perf_counter()
            value = float(score_features(model, features))
            if not math.isfinite(value):
                raise ValueError(f"Non-finite transformed score: {row.utterance_id}")
            narrow[row.utterance_id] = value
            per_record.append({"utterance_id": row.utterance_id, "sample_count": len(waveform),
                               "load_seconds": loaded - tick, "transform_seconds": transformed_at - loaded,
                               "feature_seconds": features_at - transformed_at,
                               "score_seconds": time.perf_counter() - features_at})
        scored = time.perf_counter()
        average = {key: clean[key] / 2 + narrow[key] / 2 for key in clean}
        metrics = {name: evaluate_scores(plan["groups"]["dev"], values.items(), plan["asv"], plan["reference"], evaluation_scope="pilot")
                   for name, values in (("clean_pilot", clean), ("narrow_custom_stress", narrow), ("average_custom_stress", average))}
        analysis_pairs = lambda values: [(row.utterance_id, values[row.utterance_id]) for row in prepared["analysis"]]
        ranking = evaluate_error_ranking(prepared["analysis"], analysis_pairs(clean), analysis_pairs(narrow), prepared["threshold"])
        finished = time.perf_counter()
        cpu_seconds = time.process_time() - cpu
        after = memory_snapshot()
    groups = {row.utterance_id: name for name in ("calibration", "analysis") for row in prepared[name]}
    with (output_dir / "paired_scores.csv").open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination, lineterminator="\n")
        writer.writerow(("utterance_id", "split", "label", "attack_id", "group", "clean", "narrow", "average", "delta", "abs_delta"))
        for row in sorted(plan["groups"]["dev"], key=lambda row: row.utterance_id):
            key = row.utterance_id
            delta = narrow[key] - clean[key]
            if not math.isfinite(delta):
                raise ValueError("Non-finite derived delta")
            writer.writerow((key, row.split, row.label, row.attack_id, groups[key], clean[key], narrow[key], average[key], delta, abs(delta)))
    _json(output_dir / "partition.json", prepared["partition"])
    _json(output_dir / "recording_timings.json", {"records": per_record})
    summary = {"status": "ok", "scope": "custom-narrowband-pilot-diagnostic", "protocol": PROTOCOL,
               "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
               "transform": TRANSFORM_SPEC, "provenance": prepared["provenance"],
               "baseline_experiment": prepared["baseline"]["experiment"],
               "configuration": _public_config(prepared["config"]),
               "baseline_input_sha256": prepared["baseline"]["input_sha256"],
               "model_metadata": model.metadata, "partition": prepared["partition"],
               "calibration": calibration_record, "full_dev_metrics_exploratory": metrics,
               "analysis_error_ranking": ranking, "failure_cases": select_failure_examples(ranking),
               "environment": environment_info(), "workers": 1, "thread_limit": 1, "threadpools": pools,
               "memory_before": before, "memory_after": after,
               "timings": {"transformed_scoring_seconds": scored - wall,
                           "metrics_analysis_seconds": finished - scored,
                           "processing_wall_seconds": finished - wall, "processing_cpu_seconds": cpu_seconds},
               "source_sha256": {name: digest(REPO_ROOT / name) for name in (
                   "scripts/run_robustness.py", "src/audio_deepfake/transforms.py", "src/audio_deepfake/reliability.py",
                   "src/audio_deepfake/audio.py", "src/audio_deepfake/features.py", "src/audio_deepfake/baseline.py", "src/audio_deepfake/metrics.py",
                   "scripts/run_baseline.py", "src/audio_deepfake/manifests.py", "src/audio_deepfake/resources.py", "src/audio_deepfake/cli.py")},
               "limitations": ["Custom balanced dev pilot; full-dev EER/min t-DCF are exploratory, not official benchmark results.",
                               "Official dev ASV scores are held fixed; transformed CM diagnostics do not jointly transform the ASV system.",
                               "Analysis speakers are held apart from calibration but remain in the same dev domain and attack taxonomy.",
                               "Polyphase bandwidth reduction is not a real telephone channel or speech codec.",
                               "No learned fusion, analysis-label threshold tuning, significance test or novelty claim.",
                               "Illustrative failure examples are selected errors, not a representative random sample.",
                               "Process peak RSS includes imports; filesystem cache is uncontrolled; clean scores were reused."]}
    summary["output_sha256"] = {name: digest(output_dir / name) for name in
                                ("calibration.json", "partition.json", "paired_scores.csv", "recording_timings.json")}
    _json(output_dir / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/pilot.toml"))
    parser.add_argument("--partition-audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--protocol-snapshot", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        summary = run_experiment(args.baseline_run_dir, args.config, args.partition_audit, args.protocol_snapshot, args.output_dir)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        parser.exit(1, f"Robustness error: {error}\n")
    print(json.dumps({"status": summary["status"], "metrics": summary["full_dev_metrics_exploratory"],
                      "rankings": summary["analysis_error_ranking"]["rankings"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
