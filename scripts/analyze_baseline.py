"""Export a public baseline audit with literal-threshold failure cases."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_audit(run_dir: Path, destination: Path) -> dict:
    if destination.exists():
        raise ValueError("Audit output already exists; choose a new destination")
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["status"] != "ok" or summary["scope"] not in {"smoke", "pilot"}:
        raise ValueError("Expected a completed smoke or pilot run")
    scores_path = run_dir / "scores.csv"
    if digest(scores_path) != summary["output_sha256"]["scores.csv"]:
        raise ValueError("Scores differ from the completed run hash")
    with scores_path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    ids = [row["utterance_id"] for row in rows]
    if len(ids) != len(set(ids)) or len(ids) != summary["splits"]["dev"]["count"]:
        raise ValueError("Scores must contain unique, complete dev IDs")
    if set(ids) & set(summary["model_metadata"]["training_ids"]):
        raise ValueError("Training IDs appear in dev scores")
    threshold = float(summary["metrics"]["eer_threshold"])
    if not math.isfinite(threshold):
        raise ValueError("Non-finite diagnostic threshold")
    counts = {"bonafide": 0, "spoof": 0}
    by_attack, failures = {}, []
    for row in rows:
        label, attack, score = row["label"], row["attack_id"], float(row["score"])
        if row["split"] != "dev" or label not in counts or not math.isfinite(score):
            raise ValueError("Expected finite dev scores with known labels")
        if (label == "bonafide" and attack != "-") or (label == "spoof" and attack not in {f"A{i:02d}" for i in range(1, 7)}):
            raise ValueError("Invalid dev attack/label pair")
        counts[label] += 1
        wrong = score < threshold if label == "bonafide" else score >= threshold
        group = by_attack.setdefault(attack, {"count": 0, "errors": 0})
        group["count"] += 1
        group["errors"] += int(wrong)
        if wrong:
            failures.append({**row, "score": score, "error":
                             "bonafide_rejected" if label == "bonafide" else "spoof_accepted"})
    if counts != summary["metrics"]["cm_counts"]:
        raise ValueError("Score class counts differ from run metrics")
    if {key: value["count"] for key, value in by_attack.items()} != summary["splits"]["dev"]["attacks"]:
        raise ValueError("Score attacks differ from the recorded dev selection")
    summary["error_analysis"] = {
        "threshold_source": "same-dev reference EER threshold; diagnostic only, not held-out evaluation",
        "threshold": threshold, "decision": "bonafide if score >= threshold",
        "by_attack": dict(sorted(by_attack.items())), "failure_cases": failures,
        "note": "Literal operational counts can differ from the official EER grid, including at the threshold boundary. Do not treat small per-attack counts as generalization evidence.",
    }
    summary["audit_export"] = {"source_summary_sha256": digest(summary_path),
                               "analysis_source_sha256": digest(Path(__file__))}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as target:
        target.write(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = export_audit(args.run_dir, args.output)
    except (OSError, KeyError, ValueError, TypeError) as exc:
        parser.exit(1, f"Audit error: {exc}\n")
    print(json.dumps({"scope": result["scope"], "metrics": result["metrics"],
                      "by_attack": result["error_analysis"]["by_attack"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
