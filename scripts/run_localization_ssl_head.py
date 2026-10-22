"""Fit the frozen SSL comparison only from a completed, validated 192-cache run.

Default invocation plans only. --execute launches an independent 300-second
worker; this script neither extracts encoder features nor downloads anything.
"""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/localization_ssl_v1.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def worker_state(output, **changes):
    path = output / "worker-state.json"
    state = read(path) if path.exists() else {
        "worker_pid": os.getpid(), "worker_executable": sys.executable,
        "training_started": False, "created_utc": datetime.now(timezone.utc).isoformat()}
    state.update(changes, updated_utc=datetime.now(timezone.utc).isoformat())
    temporary = output / "worker-state.tmp"
    temporary.write_text(json.dumps(state, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    return state


def worker_command(cache, output):
    sys.path.insert(0, str(ROOT / "scripts"))
    from full_run_job import base_python
    return [base_python(), "-S", "-u", str(Path(__file__).resolve()), "--worker", "--cache-run", str(cache),
            "--output-dir", str(output)]


def checked_cache(cache, indexed, config):
    """Verify every receipt and array before a single training loader is used."""
    import numpy as np
    status = read(cache / "status.json")
    identity = read(cache / "identity.json")
    validation = read(cache / "validation.json")
    worker = read(cache / "worker-result.json")
    digest = hashlib.sha256(json.dumps(identity["identity"], sort_keys=True,
                            separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if (status.get("status") != "complete" or identity.get("identity_sha256") != digest
            or status.get("identity_sha256") != digest or status.get("record_count") != 192
            or status.get("exit_code") != 0 or status.get("validation_sha256") != sha(cache / "validation.json")
            or validation.get("identity_sha256") != digest or worker.get("identity_sha256") != digest
            or worker.get("status") != "complete" or worker.get("record_count") != 192
            or status.get("training_started") is not False or status.get("eval_accessed") is not False
            or worker.get("training_started") is not False or worker.get("eval_accessed") is not False):
        raise ValueError("Require matching independently completed extraction identity")
    expected_sources = {"config.json": CONFIG,
        "localization_ssl_geometry.py": "src/audio_deepfake/localization_ssl_geometry.py",
        "ssl_extract_worker.py": "scripts/ssl_extract_worker.py", "ssl_extract_job.py": "scripts/ssl_extract_job.py",
        "partial_selection_v1.json": "docs/validation/partial_selection_v1.json",
        "localization_alignment.json": "docs/validation/localization_alignment.json"}
    sources = identity["identity"].get("source_sha256", {})
    if any(sources.get(name) != sha(ROOT / path) for name, path in expected_sources.items()):
        raise ValueError("Extraction configuration, geometry or source differs from frozen comparison")
    if (identity["identity"].get("worker_sha256") != sources["ssl_extract_worker.py"]
            or identity["identity"].get("supervisor_sha256") != sources["ssl_extract_job.py"]
            or identity["identity"].get("device") not in ("cpu", "cuda")
            or not identity["identity"].get("environment")):
        raise ValueError("Missing or inconsistent extraction environment/source provenance")
    model = identity["identity"].get("model_sha256", {})
    if (model.get("pytorch_model.bin") != config["checkpoint"]["sha256"]
            or any(model.get(name) != config["checkpoint"]["metadata_sha256"][name]
                   for name in ("config.json", "preprocessor_config.json"))):
        raise ValueError("Extraction checkpoint differs from frozen SSL configuration")
    receipts = {}
    for split in ("train", "dev"):
        folder = cache / "cache" / split
        if ({p.stem for p in folder.glob("*.npy")} != set(indexed[split])
                or {p.stem for p in folder.glob("*.json")} != set(indexed[split])):
            raise ValueError("Require exactly the frozen 128 train / 64 dev cache IDs")
        for uid, audited in indexed[split].items():
            path = folder / f"{uid}.npy"
            receipt_path = folder / f"{uid}.json"
            receipt = read(receipt_path)
            expected_shape = [(audited["sample_count"] + 319) // 320, 768]
            if (receipt.get("identity_sha256") != digest
                    or receipt.get("status") != "complete" or receipt.get("recording_id") != uid
                    or receipt.get("split") != split
                    or receipt.get("audio_sha256") != audited["audio_sha256"]
                    or receipt.get("features_sha256") != sha(path)
                    or receipt.get("features_bytes") != path.stat().st_size
                    or receipt.get("shape") != expected_shape or receipt.get("dtype") != "float32"
                    or receipt.get("source_frame_count") != 1 + (audited["sample_count"] - 400) // 320):
                raise ValueError(f"Invalid extraction receipt: {split}/{uid}")
            array = np.load(path, allow_pickle=False, mmap_mode="r")
            if list(array.shape) != expected_shape or array.dtype != np.float32 or not np.isfinite(array).all():
                raise ValueError(f"Invalid cached SSL features: {split}/{uid}")
            receipts[f"{split}/{uid}"] = {"features_sha256": receipt["features_sha256"],
                                          "receipt_sha256": sha(receipt_path)}
    expected_validation = [{"split": split, "recording_id": uid, **receipts[f"{split}/{uid}"]}
                           for split in ("train", "dev") for uid in sorted(indexed[split])]
    if validation.get("records") != expected_validation:
        raise ValueError("Independent supervisor validation inventory differs from cache")
    return identity, receipts


def sampling_preflight(replay_lfcc=False):
    """Check real frozen labels; optional LFCC feature-byte replay, never fit."""
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
    import numpy as np
    from audio_deepfake.localization_ssl_head import reference_sampling
    from audio_deepfake.partialspoof import labels_for_audio
    from run_localization_pilot import validate_gate
    config = read(ROOT / CONFIG)
    for path, digest in config["input_sha256"].items():
        if sha(ROOT / path) != digest:
            raise ValueError(f"Frozen input changed: {path}")
    baseline = read(ROOT / "docs/validation/localization_pilot_v1.json")
    if any(version(package) != expected for package, expected in baseline["results"]["training"]["versions"].items()):
        raise ValueError("Head numerical environment differs from frozen LFCC baseline")
    for path in ("src/audio_deepfake/localization_model.py", "src/audio_deepfake/partialspoof.py",
                 "src/audio_deepfake/localization_features.py", "src/audio_deepfake/features.py",
                 "src/audio_deepfake/audio.py", "scripts/run_localization_pilot.py"):
        if sha(ROOT / path) != baseline["config"]["source_sha256"][path]:
            raise ValueError("Frozen LFCC source changed")
    selection = read(ROOT / "docs/validation/partial_selection_v1.json")
    audit = read(ROOT / "docs/validation/localization_alignment.json")
    labels = read(ROOT / "data/partialspoof/selected_labels_v1.json")["splits"]
    indexed = validate_gate(selection, audit,
        config["input_sha256"]["docs/validation/partial_selection_v1.json"],
        config["input_sha256"]["data/partialspoof/selected_labels_v1.json"])
    rows = [{"split": "train", "utterance_id": uid} for uid in sorted(indexed["train"])]
    def truth(row):
        uid = row["utterance_id"]
        return labels_for_audio(labels["train"][uid], sample_count=indexed["train"][uid]["sample_count"])
    result = reference_sampling(rows, truth, include_ids=replay_lfcc)
    if result["frames_retained_per_class"] != 4524:
        raise ValueError("LFCC reference class count changed")
    result.update(status="sampling_preflight_passed", training_started=False, eval_accessed=False,
        baseline_sampler_source_sha256=sha(ROOT / "src/audio_deepfake/localization_model.py"),
        input_sha256=config["input_sha256"], lfcc_feature_replay=False)
    if replay_lfcc:
        from audio_deepfake.audio import load_audio
        from audio_deepfake.localization_features import extract_localization_features
        identities = result.pop("retained_sample_ids")
        values = {}
        for row in rows:
            uid = row["utterance_id"]
            path = ROOT / indexed["train"][uid]["audio_path"]
            if sha(path) != indexed["train"][uid]["audio_sha256"]:
                raise ValueError("Actual train audio differs from alignment audit")
            values[uid] = extract_localization_features(load_audio(path), sample_rate=16000).features
        retained = np.asarray([values[uid][i] for uid, i in identities], dtype=np.float32)
        digest = hashlib.sha256(retained.tobytes()).hexdigest()
        expected = baseline["results"]["training"]["retained_feature_bytes_sha256"]
        if digest != expected:
            raise ValueError("Real LFCC sampled feature bytes differ from frozen baseline")
        result.update(lfcc_feature_replay=True, lfcc_retained_feature_bytes_sha256=digest)
    return result


def run_worker(cache, output):
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
    import numpy as np
    from audio_deepfake.localization_model import TemporalConfig
    from audio_deepfake.localization_ssl_head import reference_sampling, train_ssl_head, score_ssl_head
    from audio_deepfake.localization_timeline import render_timeline_svg
    from audio_deepfake.partialspoof import labels_for_audio
    from audio_deepfake.resources import memory_snapshot
    from run_localization_pilot import validate_gate, duration_eer, spoof_intervals

    started = time.monotonic()
    config = read(ROOT / CONFIG)
    for path, digest in config["input_sha256"].items():
        if sha(ROOT / path) != digest:
            raise ValueError(f"Frozen comparison input changed: {path}")
    head = config["head"]
    if (config["eval_enabled"] is not False or head["seed"] != 45
            or head["frames_per_class_cap"] != 50000 or head["expected_retained_per_class"] != 4524
            or head["C"] != 1.0 or head["solver"] != "lbfgs" or head["max_iter"] != 300
            or head["numerical_threads"] != 1 or config["representation"]["dimensions"] != 768):
        raise ValueError("Frozen head configuration changed")
    selection = read(ROOT / "docs/validation/partial_selection_v1.json")
    audit = read(ROOT / "docs/validation/localization_alignment.json")
    labels = read(ROOT / "data/partialspoof/selected_labels_v1.json")["splits"]
    indexed = validate_gate(selection, audit,
                           config["input_sha256"]["docs/validation/partial_selection_v1.json"],
                           config["input_sha256"]["data/partialspoof/selected_labels_v1.json"])
    baseline = read(ROOT / "docs/validation/localization_pilot_v1.json")
    if any(version(package) != expected for package, expected in baseline["results"]["training"]["versions"].items()):
        raise ValueError("Head numerical environment differs from frozen LFCC baseline")
    baseline_dependencies = ("src/audio_deepfake/localization_model.py", "scripts/run_localization_pilot.py",
                 "src/audio_deepfake/partialspoof.py", "src/audio_deepfake/localization_timeline.py",
                 "src/audio_deepfake/resources.py")
    for path in baseline_dependencies:
        if sha(ROOT / path) != baseline["config"]["source_sha256"][path]:
            raise ValueError(f"Frozen baseline dependency changed: {path}")
    pins = {p: sha(ROOT / p) for p in [CONFIG, "scripts/run_localization_ssl_head.py",
            "src/audio_deepfake/localization_ssl_head.py", "scripts/full_run_job.py",
            "scripts/ssl_extract_worker.py", "scripts/ssl_extract_job.py",
            "src/audio_deepfake/localization_ssl_geometry.py", *baseline_dependencies, *config["input_sha256"]]}
    identity, cache_pins = checked_cache(cache, indexed, config)

    def guard():
        snapshot = memory_snapshot()
        if (snapshot["system_available_ram_bytes"] is None
                or snapshot["system_available_ram_bytes"] < 2 * 1024**3
                or snapshot["peak_rss_bytes"] > 3 * 1024**3
                or shutil.disk_usage(ROOT).free < 20 * 1024**3
                or time.monotonic() - started > 300):
            raise RuntimeError("SSL head resource/time guard failed")
        return snapshot

    def truth(row):
        split, uid = row["split"], row["utterance_id"]
        return labels_for_audio(labels[split][uid], sample_count=indexed[split][uid]["sample_count"])

    def loader(row):
        guard()
        split, uid = row["split"], row["utterance_id"]
        path = cache / "cache" / split / f"{uid}.npy"
        if sha(path) != cache_pins[f"{split}/{uid}"]["features_sha256"]:
            raise ValueError("Cache changed after validation")
        return np.load(path, allow_pickle=False), truth(row)

    rows = [{"split": "train", "utterance_id": uid} for uid in sorted(indexed["train"])]
    reference = reference_sampling(rows, truth)
    if reference["frames_retained_per_class"] != 4524:
        raise ValueError("Reference LFCC sampling count changed")
    guard()
    save(output / "config.json", {"status": "frozen_before_fit", "created_utc": datetime.now(timezone.utc).isoformat(),
         "official_benchmark": False, "eval_enabled": False, "source_sha256": pins,
         "cache_identity": identity, "cache_receipts": cache_pins, "lfcc_sampling_reference": reference,
         "selected_ids": {s: sorted(indexed[s]) for s in indexed}, "head": head,
         "timeline_ids": selection["splits"]["dev"]["timeline_ids"], "preflight": guard()})
    loaded_count = 0
    fitting_started = False
    def train_loader(row):
        nonlocal loaded_count
        values = loader(row)
        loaded_count += 1
        return values
    def fit_guard():
        nonlocal fitting_started
        snapshot = guard()
        if loaded_count == len(rows) and not fitting_started:
            fitting_started = True
            worker_state(output, phase="fitting", training_started=True, resources_before_fit=snapshot)
    model = train_ssl_head(rows, train_loader, expected_retained_per_class=4524,
                           expected_sample_digest=reference["retained_sample_ids_sha256"], guard=fit_guard)
    worker_state(output, phase="scoring_dev", training_started=True)
    scores_by_id, all_truth, all_scores, all_widths, diagnostics = {}, [], [], [], []
    for uid in sorted(indexed["dev"]):
        features, native = loader({"split": "dev", "utterance_id": uid})
        scores = score_ssl_head(model, features)
        n = indexed["dev"][uid]["sample_count"]
        starts = np.arange(len(scores)) * 320 / 16000
        ends = np.minimum((np.arange(len(scores)) + 1) * 320, n) / 16000
        widths = ends - starts
        scores_by_id[uid] = {"scores": scores.tolist(), "labels_20ms": native.tolist(),
                            "bin_starts": starts.tolist(), "bin_ends": ends.tolist()}
        all_truth.extend(native.tolist()); all_scores.extend(scores.tolist()); all_widths.extend(widths.tolist())
        predicted = scores < 0
        intervals = spoof_intervals(np.where(predicted, 0, 1), starts, ends)
        diagnostics.append({"utterance_id": uid,
             "missed_spoof_seconds": float(widths[(native == 0) & ~predicted].sum()),
             "false_alarm_seconds": float(widths[(native == 1) & predicted].sum()),
             "predicted_spoof_run_count": len(intervals)})
        if uid in selection["splits"]["dev"]["timeline_ids"]:
            path = output / "timelines" / f"{uid}.svg"
            path.parent.mkdir(exist_ok=True)
            path.write_text(render_timeline_svg(title=f"{uid} — frozen SSL custom dev pilot",
                duration=float(ends[-1]), reference_intervals=spoof_intervals(native, starts, ends),
                scores=scores, threshold=0, include_curve=False), encoding="utf-8")
    if any(sha(ROOT / path) != digest for path, digest in pins.items()):
        raise RuntimeError("Frozen source/input changed during fitting")
    checked_cache(cache, indexed, config)
    with (output / "model.pkl").open("xb") as stream:
        pickle.dump(model, stream, protocol=5)
    save(output / "dev_scores.json", scores_by_id)
    save(output / "results.json", {"status": "complete", "config_sha256": sha(output / "config.json"),
         "model_sha256": sha(output / "model.pkl"), "dev_scores_sha256": sha(output / "dev_scores.json"),
         "training": model.metadata, "custom_duration_eer": duration_eer(all_truth, all_scores, all_widths),
         "display_threshold": 0.0, "per_record": diagnostics, "elapsed_seconds": time.monotonic() - started,
         "memory": guard(), "official_benchmark": False, "eval_enabled": False,
         "limitations": ["Custom repeatedly inspected 64-record dev subset; no official benchmark claim",
             "Offline whole-recording encoder context; no streaming claim",
             "Pretraining recording overlap is unknown; no unseen-corpus claim",
             "Dev EER threshold is diagnostic only; display threshold fixed at zero",
             "Derived-reference range diagnostic must be run separately"]})
    worker_state(output, phase="complete", training_started=True, resources_after_completion=guard())


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--sampling-preflight", action="store_true")
    parser.add_argument("--replay-lfcc", action="store_true", help="Replay train LFCC bytes without fitting")
    parser.add_argument("--cache-run", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.sampling_preflight:
        if args.execute or args.worker:
            parser.error("Sampling preflight cannot fit")
        print(json.dumps(sampling_preflight(args.replay_lfcc), indent=2, allow_nan=False))
        return
    if args.replay_lfcc:
        parser.error("--replay-lfcc requires --sampling-preflight")
    if not args.execute and not args.worker:
        print(json.dumps({"status": "plan_only", "training_started": False, "cache_records_required": 192,
                          "head_seconds": 300, "eval_enabled": False}))
        return
    if args.cache_run is None or args.output_dir is None:
        parser.error("Execution requires --cache-run and a fresh --output-dir")
    cache, output = args.cache_run.resolve(), args.output_dir.resolve()
    if not cache.is_relative_to(ROOT / "outputs") or not output.is_relative_to(ROOT / "outputs"):
        raise ValueError("Cache and outputs must remain inside ignored outputs/")
    if args.worker:
        sys.path.insert(0, str(ROOT / "scripts"))
        from full_run_job import configure_project_environment
        configure_project_environment()
        worker_state(output, phase="validating_inputs", training_started=False)
        try:
            run_worker(cache, output)
        except BaseException as error:
            worker_state(output, phase="failed", error_type=type(error).__name__, error=str(error))
            raise
        return
    if output.exists():
        raise ValueError("Head output must be a fresh directory")
    output.mkdir(parents=True)
    start = time.monotonic()
    command = worker_command(cache, output)
    status = {"status": "stopped", "training_started": False, "timeout_seconds": 300,
              "command": command, "monitor_pid": None, "worker_pid": None, "exit_code": None,
              "peak_rss_bytes": 0, "source_sha256": {
                  "scripts/run_localization_ssl_head.py": sha(__file__),
                  "scripts/full_run_job.py": sha(ROOT / "scripts/full_run_job.py")}}
    try:
        if os.name != "nt":
            raise RuntimeError("Local head external resource monitor currently requires Windows")
        sys.path.insert(0, str(ROOT / "scripts"))
        from full_run_job import windows_memory, stop_owned_child
        if windows_memory()["system_available_ram_bytes"] < 2 * 1024**3 or shutil.disk_usage(ROOT).free < 20 * 1024**3:
            raise RuntimeError("Head pre-launch RAM/disk guard failed")
        with (output / "stdout.log").open("xb") as stdout, (output / "stderr.log").open("xb") as stderr:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                     stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW)
            status["monitor_pid"] = child.pid
            try:
                while child.poll() is None:
                    memory = windows_memory(child.pid, process_handle=child._handle)
                    status["peak_rss_bytes"] = max(status["peak_rss_bytes"], memory["child_peak_rss_bytes"])
                    if (time.monotonic() - start >= 300 or status["peak_rss_bytes"] > 3 * 1024**3
                            or memory["system_available_ram_bytes"] < 2 * 1024**3
                            or shutil.disk_usage(ROOT).free < 20 * 1024**3):
                        raise RuntimeError("External head timeout/RAM/disk guard failed")
                    time.sleep(.1)
                final_memory = windows_memory(child.pid, process_handle=child._handle)
                status["peak_rss_bytes"] = max(status["peak_rss_bytes"], final_memory["child_peak_rss_bytes"])
                if (time.monotonic() - start >= 300 or status["peak_rss_bytes"] > 3 * 1024**3
                        or final_memory["system_available_ram_bytes"] < 2 * 1024**3
                        or shutil.disk_usage(ROOT).free < 20 * 1024**3):
                    raise RuntimeError("Exited head worker exceeded timeout/RAM/disk limits")
            finally:
                stop_owned_child(child)
                status["exit_code"] = child.returncode
                state_path = output / "worker-state.json"
                if state_path.exists():
                    state = read(state_path)
                    status.update(worker_pid=state["worker_pid"], worker_executable=state["worker_executable"],
                                  training_started=state["training_started"], worker_phase=state["phase"],
                                  worker_state_sha256=sha(state_path))
                    if state.get("error_type"):
                        status.update(worker_error_type=state["error_type"], worker_error=state["error"])
                    if state["worker_pid"] != child.pid:
                        raise RuntimeError("Supervisor PID differs from actual Python worker PID")
        if child.returncode != 0 or read(output / "results.json").get("status") != "complete":
            raise RuntimeError("SSL head worker failed; no completed result accepted")
        status.update(status="complete", training_started=True, results_sha256=sha(output / "results.json"))
    except Exception as error:
        status.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        status["elapsed_seconds"] = time.monotonic() - start
        save(output / "status.json", status)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
