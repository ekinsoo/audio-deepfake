"""Read-only fixed-zero pilot error analysis; no fitting or threshold search.

Duration groups are fixed before analysis: native spoof runs <=100 ms,
(100,500] ms and >500 ms. These are descriptive groups, not tuned bins.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PILOT = ROOT / "outputs/jobs/localization-pilot-20261010T163921940Z/pilot"
SELECTION = ROOT / "docs/validation/partial_selection_v1.json"
ALIGNMENT = ROOT / "docs/validation/localization_alignment.json"
LABELS = ROOT / "data/partialspoof/selected_labels_v1.json"
SHORT_GROUPS = ("up_to_100ms", "over_100_to_500ms", "over_500ms")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def overlap(a, b):
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def runs(labels, starts, ends, target=0):
    result = []
    for label, start, end in zip(labels, starts, ends):
        if label != target:
            continue
        if result and abs(result[-1][1] - start) < 1e-9:
            result[-1] = (result[-1][0], end)
        else:
            result.append((start, end))
    return result


def empty():
    return {"spoof_seconds": 0.0, "bonafide_seconds": 0.0,
            "missed_spoof_seconds": 0.0, "false_alarm_seconds": 0.0}


def add(stats, label, predicted_spoof, seconds):
    if seconds < 0 or label not in (0, 1):
        raise ValueError("Invalid binary duration")
    stats["spoof_seconds" if label == 0 else "bonafide_seconds"] += seconds
    if label == 0 and not predicted_spoof:
        stats["missed_spoof_seconds"] += seconds
    if label == 1 and predicted_spoof:
        stats["false_alarm_seconds"] += seconds


def finish(stats):
    result = dict(stats)
    for label, numerator, denominator in (
        ("miss_fraction", "missed_spoof_seconds", "spoof_seconds"),
        ("false_alarm_fraction", "false_alarm_seconds", "bonafide_seconds")):
        result[label] = stats[numerator] / stats[denominator] if stats[denominator] else None
    return result


def native_vad_stats(row, native, vad, duration):
    """Exact intersection of predicted 20ms, native 10ms and VAD intervals.

    VAD=0/1 are grouped as speech, VAD=2 as nonspeech; VAD never supplies
    binary truth. Preserve audio outside printed VAD coverage separately.
    """
    if len(native) != math.ceil(duration * 100 - 1e-9):
        raise ValueError("Native label count differs from actual duration")
    previous = 0.0
    for start, end, state in vad:
        if abs(start - previous) > 1e-9 or end <= start or state not in (0, 1, 2):
            raise ValueError("Invalid VAD coverage")
        previous = end
    groups = {name: empty() for name in ("speech", "nonspeech", "vad_uncovered")}
    for i, label in enumerate(native):
        interval = (i / 100, min((i + 1) / 100, duration))
        # Only two nearby prediction bins can intersect a native bin.
        for j in range(max(0, i // 2 - 1), min(len(row["scores"]), i // 2 + 2)):
            start = max(interval[0], row["bin_starts"][j])
            end = min(interval[1], row["bin_ends"][j])
            if end <= start:
                continue
            covered = 0.0
            for vs, ve, state in vad:
                width = overlap((start, end), (vs, ve))
                if width:
                    add(groups["nonspeech" if state == 2 else "speech"], label, row["scores"][j] < 0, width)
                    covered += width
            rest = end - start - covered
            if rest < -1e-8:
                raise ValueError("Overlapping VAD annotations")
            if rest > 1e-12:
                add(groups["vad_uncovered"], label, row["scores"][j] < 0, rest)
    return groups


def duration_group(seconds):
    return SHORT_GROUPS[0] if seconds <= .1 + 1e-9 else SHORT_GROUPS[1] if seconds <= .5 + 1e-9 else SHORT_GROUPS[2]


def fragment_stats(reference, predicted):
    counts = [sum(overlap(r, p) > 1e-10 for p in predicted) for r in reference]
    return {"reference_runs": len(reference), "predicted_runs": len(predicted),
            "reference_runs_with_no_overlap": counts.count(0),
            "reference_runs_with_one_prediction_run": counts.count(1),
            "reference_runs_split_over_multiple_predictions": sum(c > 1 for c in counts),
            "predicted_runs_with_no_reference_overlap": sum(not any(overlap(r, p) > 1e-10 for r in reference) for p in predicted)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/validation/localization_errors_v1.json")
    args = parser.parse_args()
    score_path = args.pilot / "dev_scores.json"
    scores = json.loads(score_path.read_text())
    result = json.loads((args.pilot / "results.json").read_text())
    config = json.loads((args.pilot / "config.json").read_text())
    selection = json.loads(SELECTION.read_text())
    alignment = json.loads(ALIGNMENT.read_text())
    native = json.loads(LABELS.read_text())["splits"]["dev"]
    selected = {r["utterance_id"]: r for r in selection["splits"]["dev"]["selected_records"]}
    headers = {r["utterance_id"]: r for r in alignment["splits"]["dev"]["records"]}
    if set(scores) != set(selected) or sha(score_path) != result["dev_scores_sha256"]:
        raise ValueError("Require exact frozen dev scores/selection")
    if result["display_threshold"] != 0 or result["status"] != "complete":
        raise ValueError("Require completed fixed-zero pilot")
    for path, digest in config["source_sha256"].items():
        if sha(ROOT / path) != digest:
            raise ValueError(f"Frozen input changed: {path}")
    totals, by_class = empty(), {k: empty() for k in ("bonafide", "spoof")}
    by_vad = {k: empty() for k in ("speech", "nonspeech", "vad_uncovered")}
    short = {k: {"runs": 0, "spoof_seconds": 0.0, "missed_spoof_seconds": 0.0} for k in SHORT_GROUPS}
    records, vad_pins, onset_errors, offset_errors, predicted_widths = [], {}, [], [], []
    fragments = Counter()
    for uid, row in sorted(scores.items()):
        duration = headers[uid]["sample_count"] / 16000
        starts, ends = row["bin_starts"], row["bin_ends"]
        labels, values = row["labels_20ms"], row["scores"]
        if len(labels) != len(values) or len(starts) != len(values) or len(ends) != len(values):
            raise ValueError("Score dimensions differ")
        if any(not math.isfinite(v) for v in values) or abs(ends[-1] - duration) > 1e-9:
            raise ValueError("Invalid scores/duration")
        if labels != [min(native[uid][i:i+2]) for i in range(0, len(native[uid]), 2)]:
            raise ValueError("Score references differ from native labels")
        record_stats = empty()
        for i, (label, score, start, end) in enumerate(zip(labels, values, starts, ends)):
            if abs(start - i * .02) > 1e-9 or abs(end - min((i+1)*.02, duration)) > 1e-9:
                raise ValueError("Unexpected prediction grid")
            for bucket in (record_stats, totals, by_class[selected[uid]["utterance_class"]]):
                add(bucket, label, score < 0, end - start)
        vad_path = ROOT / "data/partialspoof/verified_train_dev" / selected[uid]["vad_member"]
        vad_pins[vad_path.relative_to(ROOT).as_posix()] = sha(vad_path)
        if sha(vad_path) != selected[uid]["vad_sha256"]:
            raise ValueError("VAD changed")
        vad = [(float(a), float(b), int(c)) for a, b, c in (line.split() for line in vad_path.read_text().splitlines() if line.strip())]
        intersections = native_vad_stats(row, native[uid], vad, duration)
        covered_duration = sum(v["spoof_seconds"] + v["bonafide_seconds"] for v in intersections.values())
        if abs(covered_duration - duration) > 1e-8:
            raise ValueError("Native/VAD intersections do not partition actual audio")
        for key, stats in intersections.items():
            for name, value in stats.items():
                by_vad[key][name] += value
        reference = runs(labels, starts, ends)
        predicted = runs([0 if v < 0 else 1 for v in values], starts, ends)
        frag = fragment_stats(reference, predicted)
        fragments.update(frag)
        predicted_widths.extend(b - a for a, b in predicted)
        for index, errors in ((0, onset_errors), (1, offset_errors)):
            if predicted:
                errors.extend(min(abs(r[index] - p[index]) for p in predicted) for r in reference)
        native_starts = [i / 100 for i in range(len(native[uid]))]
        native_ends = [min((i+1) / 100, duration) for i in range(len(native[uid]))]
        for interval in runs(native[uid], native_starts, native_ends):
            bucket = short[duration_group(interval[1] - interval[0])]
            detected = sum(overlap(interval, p) for p in predicted)
            bucket["runs"] += 1
            bucket["spoof_seconds"] += interval[1] - interval[0]
            bucket["missed_spoof_seconds"] += interval[1] - interval[0] - detected
        records.append({"utterance_id": uid, "utterance_class": selected[uid]["utterance_class"],
                        "duration_seconds": duration, "pooled20ms": finish(record_stats),
                        "native10ms_by_vad": {k: finish(v) for k, v in intersections.items()},
                        "fragmentation20ms": frag,
                        "reference20ms_spoof_intervals": reference if uid in config["timeline_ids"] else None,
                        "predicted_spoof_intervals": predicted if uid in config["timeline_ids"] else None})
    def boundaries(errors):
        return {"count_with_predictions": len(errors),
                "median_nearest_same_type_error_seconds": statistics.median(errors) if errors else None,
                "within_20ms": sum(e <= .02 + 1e-9 for e in errors),
                "within_100ms": sum(e <= .1 + 1e-9 for e in errors)}
    native_total = {key: sum(stats[key] for stats in by_vad.values()) for key in empty()}
    for key in ("spoof_seconds", "missed_spoof_seconds"):
        if abs(sum(group[key] for group in short.values()) - native_total[key]) > 1e-8:
            raise ValueError("Native run duration groups fail reconciliation")
    output = {"version": 1, "status": "complete", "scope": "custom fixed 64-dev descriptive error analysis; no training/eval/tuning",
              "threshold": 0, "tie_rule": "score < 0 predicts spoof; equality bona fide",
              "source_sha256": {"scripts/analyze_localization_errors.py": sha(__file__),
                  str(score_path.relative_to(ROOT).as_posix()): sha(score_path),
                  str((args.pilot / "results.json").relative_to(ROOT).as_posix()): sha(args.pilot / "results.json"),
                  str((args.pilot / "config.json").relative_to(ROOT).as_posix()): sha(args.pilot / "config.json"),
                  SELECTION.relative_to(ROOT).as_posix(): sha(SELECTION),
                  ALIGNMENT.relative_to(ROOT).as_posix(): sha(ALIGNMENT), LABELS.relative_to(ROOT).as_posix(): sha(LABELS)},
              "vad_sha256": vad_pins,
              "pooled20ms_total": finish(totals), "pooled20ms_by_utterance_class": {k: finish(v) for k, v in by_class.items()},
              "native10ms_by_vad": {k: finish(v) for k, v in by_vad.items()},
              "native10ms_total": finish(native_total),
              "native_spoof_run_duration_groups": {k: {**v, "miss_fraction": v["missed_spoof_seconds"] / v["spoof_seconds"] if v["spoof_seconds"] else None} for k,v in short.items()},
              "fragmentation20ms": dict(fragments),
              "predicted_run_duration_median_seconds": statistics.median(predicted_widths),
              "boundaries20ms": {"onset": boundaries(onset_errors), "offset": boundaries(offset_errors),
                  "definition": "Each reference boundary chooses nearest predicted boundary of the same type within its recording; no one-to-one matching, so not detection precision/recall. Missing prediction runs excluded from distance and counted by fragmentation."},
              "preselected_timeline_ids": config["timeline_ids"], "records": records,
              "limitations": ["Same 64 dev recordings as pilot; descriptive post-hoc analysis, not independent validation",
                  "Fixed 0 classifier threshold; no threshold or smoothing tuning",
                  "20ms pooled truth differs from native10ms truth near transitions; totals must not be mixed",
                  "VAD=2 can have native label0 or1; printed VAD uncovered audio retained separately",
                  "Prediction boundary precision is 20ms with clipped final interval; no sub-bin localization claim",
                  "LFCC uses future context; no streaming claim; PartialSpoof shares source with ASVspoof"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: output[k] for k in ("pooled20ms_total", "native10ms_by_vad", "native_spoof_run_duration_groups", "fragmentation20ms", "boundaries20ms")}, indent=2))


if __name__ == "__main__":
    main()
