"""Bounded custom PartialSpoof localization pilot; default invocation only plans.

The external supervisor must enforce the independent 300 second worker timeout.
This script never downloads data and never accesses eval.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import sys
import time

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from audio_deepfake.audio import load_audio
from audio_deepfake.features import LFCCConfig
from audio_deepfake.localization_features import extract_localization_features
from audio_deepfake.localization_model import TemporalConfig, train_temporal, score_temporal
from audio_deepfake.localization_timeline import render_timeline_svg
from audio_deepfake.partialspoof import labels_for_audio
from audio_deepfake.resources import memory_snapshot

SELECTION = "docs/validation/partial_selection_v1.json"
LABELS = "data/partialspoof/selected_labels_v1.json"
ALIGNMENT = "docs/validation/localization_alignment.json"
SOURCE_PATHS = ["scripts/run_localization_pilot.py", "scripts/audit_partial_alignment.py"] + [
    f"src/audio_deepfake/{name}.py" for name in
    ("audio", "features", "localization", "localization_features", "localization_model",
     "localization_timeline", "partialspoof", "resources")]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def save_status(output, phase, training_started):
    temporary = output / "status.tmp"
    temporary.write_text(json.dumps({"phase": phase, "training_started": training_started,
                         "updated_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output / "status.json")


def guard(start):
    memory = memory_snapshot()
    if memory["system_available_ram_bytes"] is None:
        raise RuntimeError("Available system RAM must be measurable")
    if memory["system_available_ram_bytes"] < 2 * 1024**3 or memory["peak_rss_bytes"] > 1024**3:
        raise RuntimeError("Localization RAM guard failed")
    if shutil.disk_usage(ROOT).free < 20 * 1024**3:
        raise RuntimeError("Localization disk guard failed")
    if time.monotonic() - start > 300:
        raise RuntimeError("Cooperative localization worker timeout")
    return memory


def validate_gate(selection, audit, selection_sha256, labels_sha256):
    """Reject stale/incomplete audit and any change in the fixed split selection."""
    if selection.get("seed") != 45 or selection.get("dataset") != "PartialSpoof v1.2":
        raise ValueError("Require the seed-45 PartialSpoof v1.2 selection")
    if (audit.get("training_alignment_ready") is not True
            or audit.get("selection_sha256") != selection_sha256
            or audit.get("compact_labels_sha256") != labels_sha256):
        raise ValueError("Successful matching real-audio alignment audit required")
    indexed = {}
    for split, count in (("train", 128), ("dev", 64)):
        selected = selection["splits"][split]["selected_records"]
        expected = [r["utterance_id"] for r in selected]
        actual = audit["splits"][split]["records"]
        found = [r["utterance_id"] for r in actual]
        if len(expected) != count or len(set(expected)) != count or sorted(expected) != sorted(found):
            raise ValueError("Alignment audit does not cover the exact frozen selection")
        if len(set(found)) != len(found):
            raise ValueError("Duplicate audited ID")
        if {r["utterance_class"] for r in selected} != {"bonafide", "spoof"}:
            raise ValueError("Both utterance classes required")
        if sum(r["utterance_class"] == "bonafide" for r in selected) != count // 2:
            raise ValueError("Fixed selection must balance utterance classes")
        for record in actual:
            if (record.get("candidate_match") is not True or record.get("sample_rate") != 16000
                    or record.get("channels") != 1 or not record.get("audio_sha256")
                    or record["native_count_10ms"] != (record["sample_count"] + 159) // 160):
                raise ValueError("Record has not passed the exact native binary alignment contract")
        indexed[split] = {r["utterance_id"]: r for r in actual}
    timeline = selection["splits"]["dev"]["timeline_ids"]
    if len(timeline) != 10 or len(set(timeline)) != 10 or not set(timeline) <= set(indexed["dev"]):
        raise ValueError("Ten unique preselected dev timeline IDs required")
    dev_classes = {r["utterance_id"]: r["utterance_class"]
                   for r in selection["splits"]["dev"]["selected_records"]}
    if sum(dev_classes[uid] == "bonafide" for uid in timeline) != 5:
        raise ValueError("Fixed timeline IDs must contain five recordings per utterance class")
    return indexed


def duration_eer(labels, scores, widths):
    """Custom closest-grid duration EER; tied scores cross together, no interpolation.

    This diagnostic uses 20 ms min-pooled references, with the last bin clipped.
    It is neither official 10 ms SegmentEER nor RangeEER.
    """
    labels, scores, widths = np.asarray(labels), np.asarray(scores), np.asarray(widths)
    if (labels.ndim != 1 or scores.shape != labels.shape or widths.shape != labels.shape
            or not len(labels) or not np.isin(labels, [0, 1]).all()
            or not np.isfinite(scores).all() or not np.isfinite(widths).all() or np.any(widths <= 0)):
        raise ValueError("Invalid weighted binary score vectors")
    totals = [float(widths[labels == k].sum()) for k in (0, 1)]
    if min(totals) <= 0:
        raise ValueError("Both binary classes required for duration EER")
    order = np.argsort(scores, kind="stable")
    values, truth, weights = scores[order], labels[order], widths[order]
    ends = np.r_[np.flatnonzero(values[1:] != values[:-1]) + 1, len(values)]
    spoof = np.r_[0., np.cumsum(weights * (truth == 0))[ends - 1]]
    bona = np.r_[0., np.cumsum(weights * (truth == 1))[ends - 1]]
    missed, false = np.clip(1 - spoof / totals[0], 0, 1), np.clip(bona / totals[1], 0, 1)
    candidates = np.r_[values[0], np.nextafter(values[ends - 1], np.inf)]
    index = int(np.argmin(np.abs(missed - false)))
    threshold = float(candidates[index])
    if not np.isfinite(threshold):
        raise ValueError("Threshold exceeds finite floating-point range")
    return {"eer_fraction": float((missed[index] + false[index]) / 2), "threshold": threshold,
            "missed_spoof_fraction": float(missed[index]), "false_alarm_fraction": float(false[index]),
            "spoof_seconds": totals[0], "bonafide_seconds": totals[1],
            "definition": "custom 20ms min-pooled, clipped-duration weighted closest-grid EER; no interpolation",
            "official_segment_eer": False, "official_range_eer": False,
            "threshold_scope": "selected on the same dev subset; optimistic diagnostic, not held-out evaluation",
            "tie_rule": "score < threshold predicts spoof; equality bona fide"}


def spoof_intervals(labels, starts, ends):
    result = []
    for label, start, end in zip(labels, starts, ends):
        if label != 0:
            continue
        if result and abs(result[-1][1] - start) < 1e-9:
            result[-1] = (result[-1][0], float(end))
        else:
            result.append((float(start), float(end)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--alignment", type=Path, default=ROOT / ALIGNMENT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/localization/pilot-v1")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"status": "plan_only", "requires": ALIGNMENT,
                          "train": 128, "dev": 64, "seed": 45, "worker_seconds": 300,
                          "eval_enabled": False, "training_started": False}, indent=2))
        return
    output = args.output_dir.resolve()
    if (not any(output.is_relative_to(ROOT / p) for p in ("outputs/localization", "outputs/jobs"))
            or output.exists()):
        raise ValueError("Require a fresh directory inside outputs/localization or outputs/jobs")
    alignment_path = args.alignment.resolve()
    if not alignment_path.is_relative_to(ROOT):
        raise ValueError("Alignment audit must remain in this repository")
    start = time.monotonic()
    preflight = guard(start)
    selection = json.loads((ROOT / SELECTION).read_text(encoding="utf-8"))
    audit = json.loads(alignment_path.read_text(encoding="utf-8"))
    labels = json.loads((ROOT / LABELS).read_text(encoding="utf-8"))["splits"]
    indexed = validate_gate(selection, audit, sha(ROOT / SELECTION), sha(ROOT / LABELS))
    if audit.get("alignment_source_sha256") != sha(ROOT / "scripts/audit_partial_alignment.py"):
        raise ValueError("Alignment source changed after audit")
    pins = {p: sha(ROOT / p) for p in SOURCE_PATHS + [SELECTION, LABELS, alignment_path.relative_to(ROOT).as_posix()]}
    config = {"status": "frozen_before_fit", "created_utc": datetime.now(timezone.utc).isoformat(),
              "dataset": "PartialSpoof v1.2", "scope": "custom 128train/64dev localization pilot",
              "official_benchmark": False, "eval_enabled": False, "model": asdict(TemporalConfig()),
              "lfcc": asdict(LFCCConfig()), "source_sha256": pins,
              "selected_ids": {s: sorted(indexed[s]) for s in indexed},
              "timeline_ids": selection["splits"]["dev"]["timeline_ids"],
              "alignment": "exact ceil(N/160) native binary 10ms; minimum over adjacent pairs; final clipped 20ms bin",
              "masking": "none; keep publisher binary labels including nonspeech",
              "features": "nearest LFCC window midpoint to output-bin center; ties earlier; offline future context",
              "threshold": "fixed decision_function threshold 0 for timelines/duration diagnostics; EER sweep separate",
              "limits": {"worker_seconds": 300, "numerical_threads": 1, "peak_rss_bytes": 1024**3,
                         "minimum_available_ram_bytes": 2 * 1024**3}, "preflight": preflight}
    save_new(output / "config.json", config)
    save_status(output, "extracting_train_features", False)

    def check_pins():
        if any(sha(ROOT / p) != digest for p, digest in pins.items()):
            raise RuntimeError("Frozen source/data audit changed during pilot")

    def loader(record):
        guard(start)
        split, uid = record["split"], record["utterance_id"]
        verified = indexed[split][uid]
        path = ROOT / f"data/partialspoof/database/{split}/con_wav/{uid}.wav"
        if sha(path) != verified["audio_sha256"]:
            raise ValueError("Audio changed after alignment audit")
        waveform = load_audio(path)
        if len(waveform) != verified["sample_count"] or len(waveform) > 60 * 16000:
            raise ValueError("Audio duration differs from checked pilot bounds")
        features = extract_localization_features(waveform, sample_rate=16000)
        truth = labels_for_audio(labels[split][uid], sample_count=len(waveform))
        if len(features.features) != len(truth):
            raise ValueError("Feature/binary-reference grids differ")
        return features, truth

    rows = [{"split": "train", "utterance_id": uid} for uid in sorted(indexed["train"])]
    def train_loader(record):
        features, truth = loader(record)
        return features.features, truth
    check_pins()
    # The model API guards once before fitting, after its per-record feature loads.
    # Count completed loader calls so the phase marker reflects the actual fit boundary.
    loaded_count = 0
    def tracked_loader(record):
        nonlocal loaded_count
        values = train_loader(record)
        loaded_count += 1
        return values
    def fit_guard():
        guard(start)
        if loaded_count == len(rows):
            save_status(output, "fitting", True)
    model = train_temporal(rows, tracked_loader, guard=fit_guard)
    if model.metadata["warnings"]:
        save_new(output / "fit_warnings.json", model.metadata)
        raise RuntimeError("Fit emitted warnings; no localization results accepted")
    check_pins()
    save_status(output, "scoring_dev", True)
    score_rows, all_labels, all_scores, all_widths = {}, [], [], []
    for uid in sorted(indexed["dev"]):
        features, truth = loader({"split": "dev", "utterance_id": uid})
        scores = score_temporal(model, features.features)
        widths = features.bin_ends - features.bin_starts
        score_rows[uid] = {"scores": scores.tolist(), "labels_20ms": truth.tolist(),
                           "bin_starts": features.bin_starts.tolist(), "bin_ends": features.bin_ends.tolist()}
        all_labels.extend(truth.tolist()); all_scores.extend(scores.tolist()); all_widths.extend(widths.tolist())
    metric = duration_eer(all_labels, all_scores, all_widths)
    threshold = 0.0
    diagnostics = []
    for uid, row in score_rows.items():
        scores, truth = np.asarray(row["scores"]), np.asarray(row["labels_20ms"])
        starts, ends = np.asarray(row["bin_starts"]), np.asarray(row["bin_ends"])
        widths, predicted = ends - starts, scores < threshold
        diagnostics.append({"utterance_id": uid,
                            "missed_spoof_seconds": float(widths[(truth == 0) & ~predicted].sum()),
                            "false_alarm_seconds": float(widths[(truth == 1) & predicted].sum())})
        if uid in config["timeline_ids"]:
            path = output / "timelines" / f"{uid}.svg"
            path.parent.mkdir(exist_ok=True)
            path.write_text(render_timeline_svg(title=f"{uid} — custom dev pilot", duration=float(ends[-1]),
                            reference_intervals=spoof_intervals(truth, starts, ends), scores=scores,
                            threshold=threshold, include_curve=False), encoding="utf-8")
    check_pins(); guard(start)
    with (output / "model.pkl").open("xb") as handle:
        pickle.dump(model, handle, protocol=5)
    save_new(output / "dev_scores.json", score_rows)
    save_new(output / "results.json", {"status": "complete", "config_sha256": sha(output / "config.json"),
             "model_sha256": sha(output / "model.pkl"), "dev_scores_sha256": sha(output / "dev_scores.json"),
             "training": model.metadata, "custom_duration_eer": metric,
             "display_threshold": threshold, "display_threshold_selection": "fixed before fitting; classifier boundary zero",
             "per_record": diagnostics,
             "elapsed_seconds": time.monotonic() - start, "memory": memory_snapshot(),
             "limitations": ["Small balanced utterance subset; not an official benchmark result",
                             "EER sweeps the same dev subset; fixed-zero displayed diagnostics are separate",
                             "No native RTTM RangeEER or official SegmentEER reported",
                             "LFCC delta context uses future audio; not streaming"]})
    print(json.dumps({"status": "complete", "output_dir": str(output), "metric": metric}, indent=2))
    save_status(output, "complete", True)


if __name__ == "__main__":
    main()
