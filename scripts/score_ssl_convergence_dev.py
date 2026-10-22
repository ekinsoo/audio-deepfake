"""Score the fixed dev subset only after independently verified train convergence.

Separate max_iter=3000 follow-up, not the failed frozen max_iter=300 comparison.
This script has no fit call, download, encoder execution or threshold tuning.
Default invocation prints a plan only.
"""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_localization_ssl_head import read, save, sha, worker_state, checked_cache


def worker_command(train, cache, verification, output):
    from full_run_job import base_python
    return [base_python(), "-S", "-u", str(Path(__file__).resolve()), "--worker",
            "--train-run", str(train), "--cache-run", str(cache), "--verification", str(verification),
            "--output-dir", str(output)]


def checked_train(train, verification, cache_identity):
    """Require a caller-provided independent audit before opening a model pickle."""
    audit = read(verification)
    config = read(train / "config.json")
    result = read(train / "results.json")
    status = read(train / "status.json")
    expected = {"status": "verified_train_only", "train_dir": train.relative_to(ROOT).as_posix(),
        "train_results_sha256": sha(train / "results.json"), "train_status_sha256": sha(train / "status.json"),
        "train_config_sha256": sha(train / "config.json"), "model_sha256": sha(train / "model.pkl"),
        "cache_identity_sha256": cache_identity["identity_sha256"], "convergence_verified": True,
        "dev_scores_absent": True, "eval_enabled": False}
    if any(audit.get(key) != value for key, value in expected.items()):
        raise ValueError("Matching independent train-only convergence verification required")
    if (status.get("status") != "complete" or status.get("exit_code") != 0
            or result.get("status") != "complete_train_only"
            or result.get("config_sha256") != expected["train_config_sha256"]
            or result.get("model_sha256") != expected["model_sha256"]
            or status.get("results_sha256") != expected["train_results_sha256"]
            or result.get("dev_scores_produced") is not False or result.get("eval_accessed") is not False
            or result.get("convergence_verified") is not True
            or (train / "dev_scores.json").exists() or (train / "timelines").exists()):
        raise ValueError("Require converged train-only artifact without dev scores or timelines")
    training = result["training"]
    if (training.get("dimensions") != 768 or training.get("seed") != 45
            or training.get("max_iter") != 3000 or training.get("C") != 1.0
            or training.get("solver") != "lbfgs" or training.get("numerical_threads") != 1
            or training.get("frames_retained_per_class") != 4524 or training.get("warnings") != []
            or not training.get("iterations") or any(not isinstance(i, int) or not 0 < i < 3000 for i in training["iterations"])
            or training.get("retained_sample_ids_sha256") != config["lfcc_sampling_reference"]["retained_sample_ids_sha256"]
            or config.get("cache_identity") != cache_identity or not config.get("protocol_id")
            or audit.get("protocol_id") != config["protocol_id"]):
        raise ValueError("Train artifact does not match the separately registered max_iter=3000 protocol")
    for path, digest in config["source_sha256"].items():
        if sha(ROOT / path) != digest:
            raise ValueError(f"Train-only source/input changed: {path}")
    return config, result, audit


def run_worker(train, cache, verification, output):
    from full_run_job import configure_project_environment
    configure_project_environment()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    import numpy as np
    from audio_deepfake.localization_ssl_head import SSLHead, score_ssl_head
    from audio_deepfake.localization_timeline import render_timeline_svg
    from audio_deepfake.partialspoof import labels_for_audio
    from audio_deepfake.resources import memory_snapshot
    from run_localization_pilot import validate_gate, duration_eer, spoof_intervals
    from importlib.metadata import version

    start = time.monotonic()
    worker_state(output, phase="verifying_train_only_gate", training_started=False, fitting_enabled=False)
    base = read(ROOT / "configs/localization_ssl_v1.json")
    for path, digest in base["input_sha256"].items():
        if sha(ROOT / path) != digest:
            raise ValueError("Frozen comparison input changed")
    selection = read(ROOT / "docs/validation/partial_selection_v1.json")
    alignment = read(ROOT / "docs/validation/localization_alignment.json")
    indexed = validate_gate(selection, alignment,
        base["input_sha256"]["docs/validation/partial_selection_v1.json"],
        base["input_sha256"]["data/partialspoof/selected_labels_v1.json"])
    cache_identity = read(cache / "identity.json")
    train_config, train_result, independent = checked_train(train, verification, cache_identity)
    identity, receipts = checked_cache(cache, indexed, base)
    training = train_result["training"]
    if any(version(package) != expected for package, expected in training["versions"].items()):
        raise ValueError("Scoring numerical environment differs from verified train-only environment")
    if (train_config["selected_ids"] != {s: sorted(indexed[s]) for s in indexed}
            or train_config["timeline_ids"] != selection["splits"]["dev"]["timeline_ids"]):
        raise ValueError("Train/development selection differs from frozen comparison")
    pins = dict(train_config["source_sha256"])
    for path in [Path(__file__), ROOT / "scripts/run_localization_ssl_head.py", ROOT / "scripts/full_run_job.py",
                 ROOT / "scripts/run_localization_pilot.py", ROOT / "src/audio_deepfake/localization_timeline.py",
                 ROOT / "src/audio_deepfake/partialspoof.py", ROOT / "src/audio_deepfake/resources.py",
                 verification, train / "config.json", train / "results.json", train / "status.json", train / "model.pkl"]:
        pins[path.relative_to(ROOT).as_posix()] = sha(path)

    def guard():
        memory = memory_snapshot()
        if (memory["system_available_ram_bytes"] is None or memory["system_available_ram_bytes"] < 2 * 1024**3
                or memory["peak_rss_bytes"] > 3 * 1024**3 or shutil.disk_usage(ROOT).free < 20 * 1024**3
                or time.monotonic()-start >= 300):
            raise RuntimeError("Score-only resource/time guard failed")
        return memory

    guard()
    # Read only the exact locally generated model approved by the independent
    # audit above; never load an unverified external pickle.
    with (train / "model.pkl").open("rb") as stream:
        model = pickle.load(stream)
    if (not isinstance(model, SSLHead) or model.metadata != training
            or not np.array_equal(model.classifier.classes_, [0, 1])
            or model.classifier.max_iter != 3000 or model.classifier.C != 1.0
            or model.classifier.solver != "lbfgs" or model.classifier.random_state != 45
            or model.scaler.n_features_in_ != 768 or model.classifier.n_features_in_ != 768
            or not np.array_equal(model.classifier.n_iter_, training["iterations"])
            or model.scaler.n_samples_seen_ != 9048):
        raise ValueError("Verified model object differs from its train-only metadata")
    labels = read(ROOT / "data/partialspoof/selected_labels_v1.json")["splits"]["dev"]
    save(output / "config.json", {"status": "frozen_before_dev_scoring", "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": train_config["protocol_id"], "comparison_scope": "separate max_iter3000 convergence follow-up; frozen max_iter300 failed",
        "fitting_enabled": False, "official_benchmark": False, "eval_enabled": False,
        "head": train_config["head"], "source_sha256": pins, "cache_identity": identity,
        "lfcc_sampling_reference": train_config["lfcc_sampling_reference"],
        "selected_ids": train_config["selected_ids"], "timeline_ids": train_config["timeline_ids"],
        "independent_train_verification": independent, "preflight": guard()})
    worker_state(output, phase="scoring_fixed_dev", training_started=False, fitting_enabled=False)
    score_rows, all_labels, all_scores, all_widths, diagnostics = {}, [], [], [], []
    for uid in sorted(indexed["dev"]):
        guard()
        path = cache / "cache/dev" / f"{uid}.npy"
        if sha(path) != receipts[f"dev/{uid}"]["features_sha256"]:
            raise ValueError("Validated dev feature cache changed")
        features = np.load(path, allow_pickle=False)
        scores = score_ssl_head(model, features)
        n = indexed["dev"][uid]["sample_count"]
        truth = labels_for_audio(labels[uid], sample_count=n)
        starts = np.arange(len(scores)) * 320 / 16000
        ends = np.minimum((np.arange(len(scores))+1) * 320, n) / 16000
        widths, predicted = ends-starts, scores < 0
        score_rows[uid] = {"scores": scores.tolist(), "labels_20ms": truth.tolist(),
                           "bin_starts": starts.tolist(), "bin_ends": ends.tolist()}
        all_labels.extend(truth.tolist()); all_scores.extend(scores.tolist()); all_widths.extend(widths.tolist())
        diagnostics.append({"utterance_id": uid,
            "missed_spoof_seconds": float(widths[(truth == 0) & ~predicted].sum()),
            "false_alarm_seconds": float(widths[(truth == 1) & predicted].sum()),
            "predicted_spoof_run_count": len(spoof_intervals(np.where(predicted, 0, 1), starts, ends))})
        if uid in train_config["timeline_ids"]:
            path = output / "timelines" / f"{uid}.svg"
            path.parent.mkdir(exist_ok=True)
            path.write_text(render_timeline_svg(title=f"{uid} â€” SSL convergence follow-up custom dev",
                duration=float(ends[-1]), reference_intervals=spoof_intervals(truth, starts, ends),
                scores=scores, threshold=0., include_curve=False), encoding="utf-8")
    checked_cache(cache, indexed, base)
    checked_train(train, verification, identity)
    if any(sha(ROOT / path) != digest for path, digest in pins.items()):
        raise RuntimeError("Verified model/source/input changed during scoring")
    # Keep the exact verified bytes, rather than serializing a new model.
    with (train / "model.pkl").open("rb") as source, (output / "model.pkl").open("xb") as target:
        shutil.copyfileobj(source, target)
    save(output / "dev_scores.json", score_rows)
    save(output / "results.json", {"status": "complete", "protocol_id": train_config["protocol_id"],
        "comparison_scope": "separate max_iter3000 convergence follow-up; not frozen max_iter300 result",
        "training_started": False, "fitting_enabled": False, "model_trained_in_verified_separate_stage": True,
        "official_benchmark": False, "eval_enabled": False,
        "config_sha256": sha(output / "config.json"), "model_sha256": sha(output / "model.pkl"),
        "dev_scores_sha256": sha(output / "dev_scores.json"), "training": training,
        "custom_duration_eer": duration_eer(all_labels, all_scores, all_widths),
        "display_threshold": 0.0, "display_threshold_selection": "fixed before scoring; classifier boundary zero",
        "per_record": diagnostics, "elapsed_seconds": time.monotonic()-start, "memory": guard(),
        "limitations": ["Original max_iter300 SSL attempt failed convergence and has no accepted dev result",
           "Separate max_iter3000 follow-up changes optimization budget; not a strict frozen-iteration comparison",
           "Same repeatedly inspected 64-dev subset; not held-out or official benchmark evaluation",
           "Whole-recording encoder uses future audio; pretraining overlap unknown",
           "Dev EER threshold diagnostic only; fixed-zero errors and timelines separate",
           "Derived-reference RangeEER diagnostic must be run separately"]})
    worker_state(output, phase="complete", training_started=False, fitting_enabled=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--train-run", type=Path)
    parser.add_argument("--cache-run", type=Path)
    parser.add_argument("--verification", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not args.execute and not args.worker:
        print(json.dumps({"status": "plan_only", "training_started": False, "fitting_enabled": False,
            "requires": "independently verified complete_train_only max_iter3000 model", "eval_enabled": False}))
        return
    if any(value is None for value in (args.train_run, args.cache_run, args.verification, args.output_dir)):
        parser.error("Execution requires --train-run --cache-run --verification and fresh --output-dir")
    train, cache, verification, output = (value.resolve() for value in
        (args.train_run, args.cache_run, args.verification, args.output_dir))
    if (not all(value.is_relative_to(ROOT / "outputs") for value in (train, cache, output))
            or not verification.is_relative_to(ROOT / "docs/validation")
            or verification.is_relative_to(train)):
        raise ValueError("Require ignored train/cache/output paths and separate docs/validation audit")
    if args.worker:
        try:
            run_worker(train, cache, verification, output)
        except BaseException as error:
            worker_state(output, phase="failed", training_started=False, fitting_enabled=False,
                         error_type=type(error).__name__, error=str(error))
            raise
        return
    if output.exists():
        raise ValueError("Preserve previous score attempts; output must be fresh")
    output.mkdir(parents=True)
    from full_run_job import windows_memory, stop_owned_child
    start, child = time.monotonic(), None
    command = worker_command(train, cache, verification, output)
    status = {"status": "stopped", "command": command, "training_started": False, "fitting_enabled": False,
        "monitor_pid": None, "worker_pid": None, "exit_code": None, "peak_rss_bytes": 0,
        "timeout_seconds": 300, "source_sha256": {Path(__file__).relative_to(ROOT).as_posix(): sha(__file__),
            "scripts/full_run_job.py": sha(ROOT / "scripts/full_run_job.py")}}
    def check(memory):
        status["peak_rss_bytes"] = max(status["peak_rss_bytes"], memory.get("child_peak_rss_bytes", 0))
        if (memory["system_available_ram_bytes"] < 2 * 1024**3 or status["peak_rss_bytes"] > 3 * 1024**3
                or shutil.disk_usage(ROOT).free < 20 * 1024**3 or time.monotonic()-start >= 300):
            raise RuntimeError("Score-only external timeout/RAM/disk guard failed")
    try:
        if os.name != "nt":
            raise RuntimeError("Score-only external resource monitor currently requires Windows")
        check(windows_memory())
        with (output / "stdout.log").open("xb") as stdout, (output / "stderr.log").open("xb") as stderr:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW)
            status["monitor_pid"] = child.pid
            while child.poll() is None:
                check(windows_memory(child.pid, process_handle=child._handle))
                time.sleep(.1)
            check(windows_memory(child.pid, process_handle=child._handle))
        process = read(output / "worker-state.json")
        if process["worker_pid"] != child.pid or process["training_started"] is not False:
            raise RuntimeError("Score-only worker PID or no-fit contract differs from supervisor")
        if child.returncode != 0 or read(output / "results.json").get("status") != "complete":
            raise RuntimeError("Score-only worker failed; no dev result accepted")
        status.update(status="complete", results_sha256=sha(output / "results.json"))
    except BaseException as error:
        status.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if child is not None:
            stop_owned_child(child)
            status["exit_code"] = child.returncode
        if (output / "worker-state.json").exists():
            process = read(output / "worker-state.json")
            status.update(worker_pid=process["worker_pid"], worker_executable=process["worker_executable"],
                          worker_phase=process["phase"], worker_state_sha256=sha(output / "worker-state.json"))
            if process.get("error_type"):
                status.update(worker_error_type=process["error_type"], worker_error=process["error"])
        status["elapsed_seconds"] = time.monotonic()-start
        save(output / "status.json", status)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
