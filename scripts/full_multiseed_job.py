"""Run the fixed remaining full train/dev seeds sequentially, without eval."""
import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tomllib

import full_run_job as supervisor

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (43, 44)
AUDIT = "docs/validation/full_multiseed_summary.json"
MODEL_SHA = "701fc3d8107cc8b2696b5552dd9e1b4751e44d2d197cfd5d1dfe7e4814add101"
SOURCES = ("scripts/full_multiseed_job.py", "scripts/start_full_multiseed.ps1",
           "scripts/full_run_job.py", "scripts/run_full_baseline.py",
           "scripts/scoring_recovery_job.py", "scripts/recover_full_scoring.py",
           "src/audio_deepfake/baseline.py", "src/audio_deepfake/features.py",
           "src/audio_deepfake/audio.py", "src/audio_deepfake/manifests.py",
           "src/audio_deepfake/metrics.py", "src/audio_deepfake/resources.py",
           "src/audio_deepfake/cli.py", "configs/full_seed42.toml",
           "configs/full_seed43.toml", "configs/full_seed44.toml",
           "docs/FULL_CONFIRMATION_PLAN.md", "docs/validation/full_baseline_seed42.json",
           "docs/validation/full_independent_review.json")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def source_hashes():
    return {name: digest(ROOT / name) for name in SOURCES}


def convergence(report):
    classes = report["model_metadata"]["classes"]
    if set(classes) != {"bonafide", "spoof"}:
        raise RuntimeError("Missing class convergence metadata")
    return {key: {name: value[name] for name in ("converged", "iterations", "warnings")}
            for key, value in classes.items()}


def validate_configs():
    original = tomllib.loads((ROOT / "configs/full_seed42.toml").read_text())
    for seed in SEEDS:
        supplied = tomllib.loads((ROOT / f"configs/full_seed{seed}.toml").read_text())
        expected = copy.deepcopy(original)
        expected["experiment"]["seed"] = seed
        if supplied != expected or supplied["data"]["eval_enabled"] is not False:
            raise RuntimeError("Only the preplanned experiment seed may change")


def validate_prior():
    """Cross-check the recovered seed and independent auditor before any child."""
    import scoring_recovery_job
    report_path = ROOT / "outputs/scoring-recovery-seed42/summary.json"
    report = read(report_path)
    scoring_recovery_job.validate_completion(report, report_path.parent)
    audit = read(ROOT / "docs/validation/full_baseline_seed42.json")
    independent = read(ROOT / "docs/validation/full_independent_review.json")
    if (audit.get("status") != "ok" or audit.get("model_sha256") != MODEL_SHA
            or audit.get("metrics") != report["metrics"] or audit.get("eval_enabled") is not False
            or audit.get("output_sha256") != report.get("output_sha256")
            or report.get("configuration") != tomllib.loads((ROOT / "configs/full_seed42.toml").read_text())):
        raise RuntimeError("Public seed 42 recovery audit differs")
    expected = dict(status="ok", seed=42, eval_enabled=False, model_sha256=MODEL_SHA,
                    summary_sha256=digest(report_path),
                    score_sha256=digest(report_path.parent / "scores.csv"))
    if any(independent.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Independent seed 42 review does not match completed recovery")
    if digest(ROOT / "outputs/baseline-full-seed42/model.joblib") != MODEL_SHA:
        raise RuntimeError("Seed 42 saved model hash differs")
    classes = convergence(report)
    if any(value["converged"] is not True or value["warnings"] for value in classes.values()):
        raise RuntimeError("Seed 42 convergence gate failed")
    plan = read(report_path.parent / "plan.json")
    fix = plan["provenance"]["instrumentation_fix"]
    if (fix.get("original_sha256") != "ebc6d201f0cc9203d9d935ab712054fce809bbd73c274be3961b5941ddefc615"
            or fix.get("current_sha256") != "9d68202f7335b17e83f2138600f0f1c7312adab46c3d73609d61751800282e7a"
            or digest(ROOT / "src/audio_deepfake/resources.py") != fix["current_sha256"]):
        raise RuntimeError("Resource instrumentation fix history differs")
    for name, pinned in audit["source_sha256"].items():
        if digest(ROOT / name) != pinned:
            raise RuntimeError(f"Seed 42 source changed: {name}")
    return report


def validate_metric_protocol(metrics):
    expected = dict(reference_sha256="93ea505282d329ed14537d977a74b2d1a25d161ac6767a16e0216a1a18391049",
                    asv_score_sha256="fe8114cf608c3062c141df5979892eba9664aa37a8c99e97157f129d679d3389",
                    rate_units="fraction", score_direction="higher_is_bonafide", split="dev",
                    cm_counts={"bonafide": 2548, "spoof": 22296})
    if any(metrics.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Metric protocol differs from pinned full dev reference")


def validate_new(report, output, seed, prior=None):
    if (report.get("status") != "ok" or report.get("experiment", {}).get("seed") != seed
            or report.get("configuration", {}).get("data", {}).get("eval_enabled") is not False):
        raise RuntimeError("Unexpected new seed completion")
    supervisor.validate_baseline_completion(report, output)
    if report["configuration"] != tomllib.loads((ROOT / f"configs/full_seed{seed}.toml").read_text()):
        raise RuntimeError("Completed seed configuration differs")
    validate_metric_protocol(report["metrics"])
    if prior is None or any(report["input_sha256"].get(key) != value
                            for key, value in prior["input_sha256"].items()):
        raise RuntimeError("Completed seed official input provenance differs")
    metadata = report["model_metadata"]
    expected_gmm = copy.deepcopy(prior["model_metadata"]["gmm_config"])
    expected_gmm["seed"] = seed
    if metadata["gmm_config"] != expected_gmm:
        raise RuntimeError("Completed seed GMM configuration differs")
    protocols = ROOT / "data/metadata/asvspoof2019"
    train_ids = sorted(line.split()[1] for line in
                       (protocols / "ASVspoof2019.LA.cm.train.trn.txt").read_text().splitlines())
    if len(train_ids) != 25380 or metadata["training_ids"] != train_ids:
        raise RuntimeError("Completed seed training IDs differ")
    dev = [line.split() for line in (protocols / "ASVspoof2019.LA.cm.dev.trl.txt").read_text().splitlines()]
    with (output / "scores.csv").open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    for row, fields in zip(rows, dev, strict=True):
        if (row["utterance_id"] != fields[1] or row["label"] != fields[4]
                or row["attack_id"] != fields[3]):
            raise RuntimeError("Completed seed scores differ from official dev order/metadata")
    for name, pinned in report["source_sha256"].items():
        if digest(ROOT / name) != pinned:
            raise RuntimeError("Completed seed source differs")
    convergence(report)  # Preserve unsuccessful convergence rather than select another seed.


def aggregate(reports):
    if set(reports) != {42, 43, 44} or any(value.get("status") != "ok" for value in reports.values()):
        raise RuntimeError("Aggregation requires all three complete fixed seeds")
    runs = []
    for seed in (42, 43, 44):
        report = reports[seed]
        validate_metric_protocol(report["metrics"])
        metrics = {key: report["metrics"][key] for key in ("eer", "min_tDCF")}
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in metrics.values()):
            raise RuntimeError("Invalid seed metric")
        classes = convergence(report)
        runs.append(dict(seed=seed, metrics=metrics, classes=classes,
                         convergence_ok=all(v["converged"] is True and not v["warnings"] for v in classes.values()),
                         timings=report.get("timings", report.get("recovery", {})),
                         memory_after=report.get("memory_after"),
                         model_sha256=report.get("model_sha256", report.get("output_sha256", {}).get("model.joblib"))))
    return dict(status="ok", scope="full-official-train-dev-adaptation-three-seeds", eval_enabled=False,
                seeds=[42, 43, 44], runs=runs, rate_units="fraction", std_ddof=0,
                metrics={key: dict(mean=statistics.mean(run["metrics"][key] for run in runs),
                                   std=statistics.pstdev(run["metrics"][key] for run in runs))
                         for key in ("eer", "min_tDCF")},
                convergence_failures=[run["seed"] for run in runs if not run["convergence_ok"]],
                limitations=["Development results only; eval remains closed.",
                             "Fixed Python adaptation, not exact official MATLAB reproduction.",
                             "Seed 42 used scoring recovery; its timings exclude original training/scoring.",
                             "Three fixed seeds describe variation, not a significance test."])


def reject_existing():
    for name in ("full-seed42.lock", "scoring-recovery-seed42.lock"):
        if (ROOT / "outputs/jobs" / name).exists():
            raise RuntimeError(f"Other job lock exists: {name}")
    for name in (AUDIT, *(f"outputs/baseline-full-seed{seed}" for seed in SEEDS)):
        if (ROOT / name).exists():
            raise RuntimeError(f"Preserve existing output: {name}")


def public_stages(stages):
    """Export measured resources/timing without local commands, paths or PIDs."""
    fields = ("name", "status", "budget_seconds", "elapsed_seconds", "peak_rss_bytes",
              "exit_code", "started_utc", "ended_utc", "last_memory")
    return [{name: record[name] for name in fields if name in record} for record in stages]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    directory = args.job_dir.resolve()
    jobs = (ROOT / "outputs/jobs").resolve()
    if directory.parent != jobs or not directory.name.startswith("full-multiseed-"):
        parser.error("Use a new outputs/jobs/full-multiseed-* directory")
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "status.json").exists():
        parser.error("Preserve previous job status")
    status = dict(status="starting", pid=os.getpid(), child_pid=None, stages=[], seeds=list(SEEDS),
                  started_utc=supervisor.utc_now(), eval_enabled=False)
    lock = jobs / "full-multiseed.lock"
    owns_lock = False
    code = 1
    try:
        supervisor.acquire_lock(lock, dict(pid=os.getpid(), job_dir=str(directory)))
        owns_lock = True
        reject_existing()
        validate_configs()
        reports = {42: validate_prior()}
        status["source_sha256"] = source_hashes()
        for seed in SEEDS:
            if source_hashes() != status["source_sha256"]:
                raise RuntimeError("Frozen sources changed after launch")
            supervisor.wait_for_resources(job_dir=directory, status=status, phase=f"seed{seed}")
            if source_hashes() != status["source_sha256"]:
                raise RuntimeError("Frozen sources changed during resource wait")
            output = ROOT / f"outputs/baseline-full-seed{seed}"
            if output.exists():
                raise RuntimeError("Seed output appeared after preflight; preserve it")
            command = supervisor.worker_command("scripts/run_full_baseline.py", "--config",
                         f"configs/full_seed{seed}.toml", "--output-dir", str(output))
            supervisor.run_stage(command, root=ROOT, job_dir=directory, name=f"seed{seed}",
                                 budget_seconds=1800, status=status)
            reports[seed] = read(output / "summary.json")
            validate_new(reports[seed], output, seed, reports[42])
        if source_hashes() != status["source_sha256"]:
            raise RuntimeError("Frozen sources changed before aggregation")
        result = aggregate(reports)
        result.update(finished_utc=supervisor.utc_now(), source_sha256=status["source_sha256"],
                      job=directory.relative_to(ROOT).as_posix(), supervised_stages=public_stages(status["stages"]))
        if (ROOT / AUDIT).exists():
            raise RuntimeError("Multiseed audit already exists")
        supervisor.write_json(ROOT / AUDIT, result)
        with (ROOT / "reports.md").open("a", encoding="utf-8") as target:
            target.write("\n\n## Full dev confirmation across three seeds\n\n"
                         f"Completed {result['finished_utc']}. Fixed seeds 42, 43 and 44: "
                         f"mean dev EER {result['metrics']['eer']['mean']*100:.4f}% "
                         f"(population standard deviation {result['metrics']['eer']['std']*100:.4f} percentage points); "
                         f"mean minimum t-DCF {result['metrics']['min_tDCF']['mean']:.6f} "
                         f"(population standard deviation {result['metrics']['min_tDCF']['std']:.6f}). "
                         f"Seeds with convergence warnings/failure: {result['convergence_failures']}. "
                         "Eval stayed closed. These are development results for the Python adaptation. "
                         "Seed 42 timing covers recovery only. See "
                         "[the measured audit](docs/validation/full_multiseed_summary.json).\n")
        status.update(status="complete", metrics=result["metrics"], audit=AUDIT)
        code = 0
    except BaseException as error:
        status.update(status="stopped", reason=f"{type(error).__name__}: {error}")
    finally:
        status.update(ended_utc=supervisor.utc_now(), exit_code=code)
        supervisor.write_json(directory / "status.json", status)
        if owns_lock:
            lock.unlink()
    return code


if __name__ == "__main__":
    supervisor.configure_project_environment()
    raise SystemExit(main())
