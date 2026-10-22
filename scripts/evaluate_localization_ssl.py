"""SSL-only adapter for unchanged frozen error analysis and bounded RangeEER.

No fitting, extraction, download, eval split or threshold tuning. The upstream
RangeEER search remains a derived-reference, inspected-dev diagnostic.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import runpy
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_localization_ranges as frozen


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def worker_command(pilot, output, analysis):
    from full_run_job import base_python
    return [base_python(), "-S", "-u", str(Path(__file__).resolve()), "--worker", "--analysis", analysis,
            "--pilot", str(pilot), "--output-dir", str(output)]


def parse_selected_row(text):
    found = re.search(r"BoundedSearch selected_row=(\[[^\n]+\])", text)
    if found is None:
        raise ValueError("Bounded evaluator did not publish its selected row")
    row = json.loads(found.group(1))
    if (len(row) != 4 or any(isinstance(x, bool) or not isinstance(x, (int, float))
                            or not math.isfinite(x) for x in row)
            or not all(0 <= x <= 1 for x in row[1:])):
        raise ValueError("Invalid bounded RangeEER selected row")
    return row


def validate_scores(pilot, scores, native, durations):
    result, status = read(pilot / "results.json"), read(pilot / "status.json")
    config = read(pilot / "config.json")
    if (status.get("status") != "complete" or result.get("status") != "complete"
            or status.get("results_sha256") != sha(pilot / "results.json")
            or result.get("dev_scores_sha256") != sha(pilot / "dev_scores.json")
            or result.get("config_sha256") != sha(pilot / "config.json")
            or result.get("model_sha256") != sha(pilot / "model.pkl")
            or result.get("display_threshold") != 0 or result.get("eval_enabled") is not False
            or result.get("official_benchmark") is not False):
        raise ValueError("Require completed, hash-matching fixed-zero SSL results")
    training = result["training"]
    if (training.get("implementation") != "frozen-ssl-standardscaler-logistic-v1"
            or training.get("dimensions") != 768 or training.get("seed") != 45
            or training.get("frames_retained_per_class") != 4524
            or training.get("retained_sample_ids_sha256") != config["lfcc_sampling_reference"]["retained_sample_ids_sha256"]):
        raise ValueError("Require the frozen SSL head and matching LFCC frame identities")
    for path, digest in config["source_sha256"].items():
        if sha(ROOT / path) != digest:
            raise ValueError("SSL head source/input changed after fitting")
    if set(scores) != set(durations) or len(scores) != 64:
        raise ValueError("Exact frozen 64 dev IDs required")
    for uid, row in scores.items():
        values = row["scores"]
        expected = [min(native[uid][i:i+2]) for i in range(0, len(native[uid]), 2)]
        if (row["labels_20ms"] != expected or len(values) != len(expected)
                or len(row["bin_starts"]) != len(values) or len(row["bin_ends"]) != len(values)
                or any(not math.isfinite(v) for v in values)):
            raise ValueError("Invalid SSL/native score grid")
        for i, (start, end) in enumerate(zip(row["bin_starts"], row["bin_ends"])):
            if (not math.isfinite(start) or not math.isfinite(end)
                    or abs(start-i*.02) > 1e-9 or abs(end-min((i+1)*.02, durations[uid])) > 1e-9):
                raise ValueError("SSL score time grid differs from actual audio")
        if uid.startswith("CON") and not any(v == 0 for v in native[uid]):
            raise ValueError("Upstream would omit a CON recording without spoof RTTM")
    return result


def bounded(command, directory):
    from full_run_job import windows_memory, stop_owned_child
    from localization_pilot_job import check_resources
    if os.name != "nt":
        raise RuntimeError("This local metric supervisor requires Windows")
    check_resources(windows_memory(), shutil.disk_usage(ROOT).free)
    start, peak, child = time.monotonic(), 0, None
    record = {"status": "stopped", "command": command, "training_started": False,
              "monitor_pid": None, "worker_pid": None, "exit_code": None}
    try:
        with (directory / "stdout.log").open("xb") as stdout, (directory / "stderr.log").open("xb") as stderr:
            child = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                       stdin=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
            record["monitor_pid"] = child.pid
            while child.poll() is None:
                if time.monotonic()-start >= 120:
                    raise RuntimeError("SSL diagnostic worker exceeded 120 seconds")
                memory = windows_memory(child.pid, process_handle=child._handle)
                peak = max(peak, memory["child_peak_rss_bytes"])
                check_resources(memory, shutil.disk_usage(ROOT).free)
                time.sleep(.1)
            memory = windows_memory(child.pid, process_handle=child._handle)
            peak = max(peak, memory["child_peak_rss_bytes"])
            check_resources(memory, shutil.disk_usage(ROOT).free)
            if child.returncode != 0 or time.monotonic()-start >= 120:
                raise RuntimeError("SSL diagnostic worker failed; retain logs")
        process = read(directory / "worker-process.json")
        if process.get("worker_pid") != child.pid:
            raise RuntimeError("Metric supervisor PID differs from actual Python worker PID")
        record["status"] = "complete"
    except BaseException as error:
        record.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if child is not None:
            stop_owned_child(child)
            record["exit_code"] = getattr(child, "returncode", None)
        record.update(elapsed_seconds=time.monotonic()-start, peak_rss_bytes=peak)
        if (directory / "worker-process.json").exists():
            process = read(directory / "worker-process.json")
            record.update(worker_pid=process["worker_pid"], worker_executable=process["worker_executable"],
                          worker_process_sha256=sha(directory / "worker-process.json"))
        for filename in ("stdout.log", "stderr.log"):
            if (directory / filename).exists():
                record[filename.replace(".log", "_sha256")] = sha(directory / filename)
        save(directory / "supervisor.json", record)
    return record


def run(pilot, output):
    from full_run_job import base_python
    prior_range = read(ROOT / "docs/validation/localization_range_results_v1.json")
    prior_errors = read(ROOT / "docs/validation/localization_errors_v1.json")
    for path, digest in prior_range["source_sha256"].items():
        if sha(ROOT / path) != digest:
            raise ValueError(f"Frozen range dependency changed: {path}")
    error_source = "scripts/analyze_localization_errors.py"
    if sha(ROOT / error_source) != prior_errors["source_sha256"][error_source]:
        raise ValueError("Frozen error analyzer changed")
    refs = read(frozen.REF_AUDIT)
    if refs["official_native_reference_gate_passed"] is not False:
        raise ValueError("Adapter requires explicitly derived references")
    for filename, key in (("reference.rttm", "reference_sha256"), ("reco2dur", "reco2dur_sha256")):
        if sha(frozen.REFERENCES / filename) != refs["splits"]["dev"][key]:
            raise ValueError("Derived native references changed")
    scores = read(pilot / "dev_scores.json")
    native = read(ROOT / "data/partialspoof/selected_labels_v1.json")["splits"]["dev"]
    durations = {r["utterance_id"]: r["duration_seconds"] for r in refs["splits"]["dev"]["records"]}
    result = validate_scores(pilot, scores, native, durations)
    pins = dict(prior_range["source_sha256"])
    for path in [error_source, "scripts/evaluate_localization_ssl.py",
                 "docs/validation/localization_range_results_v1.json", "docs/validation/localization_errors_v1.json",
                 "docs/validation/localization_pilot_v1.json"]:
        pins[path] = sha(ROOT / path)
    for path in (pilot / "results.json", pilot / "config.json", pilot / "status.json", pilot / "dev_scores.json",
                 frozen.REFERENCES / "reference.rttm", frozen.REFERENCES / "reco2dur"):
        pins[path.relative_to(ROOT).as_posix()] = sha(path)
    output.mkdir(parents=True)
    audit = {"status": "stopped", "created_utc": datetime.now(timezone.utc).isoformat(),
             "official_benchmark": False, "publisher_reference_gate_passed": False,
             "training_started": False, "eval_enabled": False, "source_sha256": pins,
             "scope": "same inspected 64-dev SSL comparison; fixed-zero errors plus derived-reference RangeEER"}
    try:
        error_dir = output / "errors"
        error_dir.mkdir()
        raw = error_dir / "raw_frozen_analyzer.json"
        audit["error_worker"] = bounded(worker_command(pilot, error_dir, "errors"), error_dir)
        errors = read(raw)
        if errors["status"] != "complete":
            raise ValueError("Error analyzer did not complete")
        range_dir = output / "range"
        range_dir.mkdir()
        audit["range_worker"] = bounded(worker_command(pilot, range_dir, "range"), range_dir)
        text = (range_dir / "stdout.log").read_text(encoding="utf-8")
        row = parse_selected_row(text)
        independent = frozen.direct_rates(scores, native, durations, row[0])
        if max(abs(row[1]-independent["fpr"]), abs(row[2]-independent["fnr"])) > 1e-8:
            raise ValueError("Range rates differ from independent native-bin duration arithmetic")
        packages = read(range_dir / "environment.json")
        if packages != prior_range["variants"]["baseline"]["packages"]:
            raise ValueError("Isolated metric environment differs from frozen baseline")
        lfcc = read(ROOT / "docs/validation/localization_pilot_v1.json")["results"]
        metrics = {"custom_duration_eer_fraction": result["custom_duration_eer"]["eer_fraction"],
                   "derived_reference_range_eer_fraction": (row[1]+row[2])/2,
                   "fixed_zero_pooled20ms": errors["pooled20ms_total"],
                   "fragmentation20ms": errors["fragmentation20ms"]}
        baseline = {"custom_duration_eer_fraction": lfcc["custom_duration_eer"]["eer_fraction"],
                    "derived_reference_range_eer_fraction": prior_range["variants"]["baseline"]["range_eer_fraction"],
                    "fixed_zero_pooled20ms": prior_errors["pooled20ms_total"],
                    "fragmentation20ms": prior_errors["fragmentation20ms"]}
        for path, digest in {**pins, **errors["vad_sha256"]}.items():
            if sha(ROOT / path) != digest:
                raise ValueError("Diagnostic source/input changed during evaluation")
        audit.update(status="complete_derived_reference_diagnostic", ssl=metrics, frozen_lfcc=baseline,
             selected_range_row=row, independent_native_duration_check=independent,
             packages=packages, raw_error_audit_sha256=sha(raw),
             raw_error_limitation_correction="Raw frozen analyzer mentions LFCC; this run uses whole-recording offline SSL encoder context",
             limitations=["Same repeatedly inspected dev subset; no official or held-out benchmark claim",
               "Derived native-bin references, not publisher RTTM; separate from pooled20ms EER",
               "Range evaluator uses score <= threshold; fixed-zero errors and timelines use score < 0",
               "Range/dev EER thresholds are diagnostics, not deployable thresholds",
               "SSL encoder uses future audio from the whole recording; no streaming or unseen-corpus claim"])
    except BaseException as error:
        audit.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        save(output / "results.json", audit)
    print(json.dumps({"status": audit["status"], "ssl": metrics, "frozen_lfcc": baseline}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--analysis", choices=("range", "errors"), default="range", help=argparse.SUPPRESS)
    parser.add_argument("--pilot", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not args.execute and not args.worker:
        print(json.dumps({"status": "plan_only", "training_started": False, "eval_enabled": False,
                          "workers": "fixed-zero error analysis + unchanged isolated bounded RangeEER"}))
        return
    if args.pilot is None or args.output_dir is None:
        parser.error("Require --pilot and fresh --output-dir")
    pilot, output = args.pilot.resolve(), args.output_dir.resolve()
    if not pilot.is_relative_to(ROOT / "outputs") or not output.is_relative_to(ROOT / "outputs"):
        raise ValueError("Require inputs and fresh outputs inside ignored outputs/")
    if args.worker:
        from full_run_job import configure_project_environment
        configure_project_environment()
        save(output / "worker-process.json", {"worker_pid": os.getpid(), "worker_executable": sys.executable,
                                               "training_started": False, "analysis": args.analysis})
        if args.analysis == "errors":
            source = ROOT / "scripts/analyze_localization_errors.py"
            sys.argv = [str(source), "--pilot", str(pilot), "--output", str(output / "raw_frozen_analyzer.json")]
            runpy.run_path(str(source), run_name="__main__")
            return
        # Only route the SSL score file to the existing unchanged evaluator.
        frozen.PILOTS["ssl"] = pilot.relative_to(ROOT).as_posix()
        frozen.metric_worker("ssl", output)
    else:
        if output.exists():
            raise ValueError("Preserve previous diagnostic outputs")
        run(pilot, output)


if __name__ == "__main__":
    main()
