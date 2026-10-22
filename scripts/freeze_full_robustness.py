"""Freeze full-dev narrowband metadata only; never transform, score or fit audio."""

import argparse
import csv
from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import full_multiseed_job as multiseed
import run_full_baseline as full
from recover_full_scoring import validate_scores
from audio_deepfake.reliability import PROTOCOL, calibration_threshold, partition_dev
from audio_deepfake.transforms import TRANSFORM_SPEC

ROOT = full.ROOT
PROTOCOL_ID = "full-narrowband-confirmation-v1"
PLAN = ROOT / "docs/FULL_ROBUSTNESS_PLAN.md"
AUDIT = ROOT / "docs/validation/full_robustness_protocol_v1.json"
OUTPUT = ROOT / "outputs/full-robustness-protocol-v1"
SOURCES = ("scripts/freeze_full_robustness.py", "scripts/run_robustness.py",
           "scripts/full_multiseed_job.py", "scripts/run_full_baseline.py",
           "scripts/recover_full_scoring.py", "scripts/scoring_recovery_job.py",
           "scripts/full_run_job.py", "src/audio_deepfake/cli.py",
           "src/audio_deepfake/transforms.py", "src/audio_deepfake/reliability.py",
           "src/audio_deepfake/baseline.py", "src/audio_deepfake/features.py",
           "src/audio_deepfake/audio.py", "src/audio_deepfake/manifests.py",
           "src/audio_deepfake/metrics.py", "src/audio_deepfake/resources.py")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def freeze_threshold(rows, clean):
    """Only calibration bona fide scores determine a model's operating point."""
    threshold = calibration_threshold(rows, [(row.utterance_id, clean[row.utterance_id]) for row in rows])
    bona = [clean[row.utterance_id] for row in rows if row.label == "bonafide"]
    accepted = sum(value >= threshold for value in bona)
    if accepted / len(bona) < 0.95:
        raise ValueError("Calibration acceptance is below the frozen target")
    return {"threshold": threshold, "rule": "ascending clean calibration BF[n//20]; accept bona fide iff score >= threshold",
            "bonafide_count": len(bona), "ascending_index": len(bona) // 20,
            "accepted_count": accepted, "rejected_count": len(bona) - accepted,
            "acceptance_fraction": accepted / len(bona), "target_acceptance": 0.95}


def build_plan():
    multiseed.validate_configs()
    prior = multiseed.validate_prior()
    review_path = ROOT / "docs/validation/full_multiseed_review.json"
    review = read(review_path)
    if review.get("status") != "ok" or review.get("eval_enabled") is not False or review.get("aggregate_exact_match") is not True:
        raise ValueError("Requires successful complete three-seed post-run review")
    groups, hashes = full.official_selection(ROOT / prior["configuration"]["paths"]["protocol_dir"],
                                            ROOT / prior["configuration"]["paths"]["selection_manifest"])
    calibration, analysis, partition = partition_dev(groups["train"], groups["dev"])
    if (len(calibration) + len(analysis) != 24844
            or partition["calibration"]["speakers"] != 8 or partition["analysis"]["speakers"] != 12):
        raise ValueError("Unexpected complete full-dev partition")
    paths = {42: ROOT / "outputs/scoring-recovery-seed42",
             43: ROOT / "outputs/baseline-full-seed43", 44: ROOT / "outputs/baseline-full-seed44"}
    checks = {record["seed"]: record for record in review["checks"]}
    seeds = {}
    for seed, directory in paths.items():
        report = prior if seed == 42 else read(directory / "summary.json")
        if seed != 42:
            multiseed.validate_new(report, directory, seed, prior)
            check = checks.get(seed, {})
            if (check.get("summary_sha256") != full.digest(directory / "summary.json")
                    or check.get("scores_sha256") != full.digest(directory / "scores.csv")
                    or check.get("model_sha256") != report["output_sha256"]["model.joblib"]
                    or check.get("official_metrics_exact_match") is not True):
                raise ValueError("Post-run review artifact pins differ")
        packages = report["model_metadata"]["versions"]
        if packages != {name: version(name) for name in packages}:
            raise ValueError("Packages differ from the trained model")
        pairs = validate_scores(directory / "scores.csv", groups["dev"])
        clean = dict(pairs)
        model_path = ROOT / "outputs/baseline-full-seed42/model.joblib" if seed == 42 else directory / "model.joblib"
        seeds[str(seed)] = {"model_path": model_path.relative_to(ROOT).as_posix(),
                            "model_sha256": full.digest(model_path),
                            "clean_scores_path": (directory / "scores.csv").relative_to(ROOT).as_posix(),
                            "clean_scores_sha256": full.digest(directory / "scores.csv"),
                            "summary_sha256": full.digest(directory / "summary.json"),
                            "config_sha256": full.digest(ROOT / f"configs/full_seed{seed}.toml"),
                            "packages": packages, "calibration": freeze_threshold(calibration, clean)}
    compact_partition = {key: {name: value for name, value in partition[key].items() if name != "utterance_ids"}
                         for key in ("calibration", "analysis")}
    contract = {"status": "frozen", "protocol": PROTOCOL_ID, "created_utc": full.timestamp(),
                "eval_enabled": False, "model_seeds": [42, 43, 44], "retraining": False,
                "transform": TRANSFORM_SPEC, "partition_helper_protocol": PROTOCOL,
                "partition": compact_partition, "train_manifest_sha256": partition["train_manifest_sha256"],
                "official_input_sha256": prior["input_sha256"], "selection_sha256": hashes["selection_manifest"],
                "seeds": seeds, "protocol_text_sha256": full.digest(PLAN),
                "source_sha256": {name: full.digest(ROOT / name) for name in SOURCES},
                "parent_audit_sha256": full.digest(ROOT / multiseed.AUDIT),
                "parent_review_sha256": full.digest(review_path),
                "comparisons": ["clean", "narrowband", "average = clean/2 + narrowband/2"],
                "ranking_scores": ["abs(narrowband-clean)", "-abs(clean-calibration_threshold)"],
                "reporting": {"detection": ["EER", "minimum normalized t-DCF"],
                              "error_ranking": ["AUROC", "non-interpolated average precision", "error prevalence"],
                              "aggregate": "Report each fixed seed, mean and population std ddof=0; no seed selection.",
                              "ranking_aggregate": "Report defined-seed count; if any seed is undefined mark three-seed aggregate incomplete.",
                              "undefined": "One-class ranking returns null with a reason; never substitute 0.5."},
                "resources": {"workers": 1, "numeric_threads": 1, "max_worker_seconds": 1800,
                              "max_processing_seconds_per_seed": 1800,
                              "max_process_peak_gib": 2, "min_start_available_ram_gib": 3.5,
                              "min_running_available_ram_gib": 1, "min_free_disk_gib": 20},
                "execution_status": "Not started; metadata freeze only. A separate guarded full runner is required.",
                "limitations": ["Clean full-dev labels/results and pilot stress results were inspected before this freeze.",
                                "Pre-transformation protocol, not blind preregistration or independent validation.",
                                "Full dev includes the earlier pilot subset and the same attack taxonomy.",
                                "Speaker separation is internal to dev; the three model seeds are not independent datasets.",
                                "Approximate bandwidth stress, not a codec; official ASV scores remain fixed."]}
    return contract, calibration, analysis, partition


def save_plan(contract, calibration, analysis, partition, output=OUTPUT, audit=AUDIT):
    if output.exists() or audit.exists():
        raise ValueError("Preserve the existing frozen protocol; use an explicitly versioned new protocol")
    output.mkdir(parents=True)
    output_hashes = {}
    for name, rows in (("calibration", calibration), ("analysis", analysis)):
        path = output / f"{name}.csv"
        with path.open("x", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=tuple(rows[0].__dataclass_fields__), lineterminator="\n")
            writer.writeheader()
            writer.writerows(asdict(row) for row in rows)
        output_hashes[path.name] = full.digest(path)
    full.atomic_json(output / "partition.json", partition)
    output_hashes["partition.json"] = full.digest(output / "partition.json")
    contract["local_partition_dir"] = output.relative_to(ROOT).as_posix()
    contract["partition_file_sha256"] = output_hashes
    audit.parent.mkdir(parents=True, exist_ok=True)
    full.atomic_json(audit, contract)  # Publish the successful freeze last.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Write the immutable metadata freeze; no audio processing")
    args = parser.parse_args()
    contract, calibration, analysis, partition = build_plan()
    if args.execute:
        save_plan(contract, calibration, analysis, partition)
    print(json.dumps({"written": args.execute, "protocol": PROTOCOL_ID,
                      "partition": contract["partition"],
                      "calibration": {seed: record["calibration"] for seed, record in contract["seeds"].items()}}, indent=2))


if __name__ == "__main__":
    main()
