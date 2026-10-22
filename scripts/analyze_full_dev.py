"""Export compact literal-threshold error counts from a completed full dev run."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "data/metadata/asvspoof2019/ASVspoof2019.LA.cm.dev.trl.txt"
PROTOCOL_SHA256 = "2fedaebf1ba3ae1a9bb7d3ec916804ac237b4eee063a27fa92e6549f03d129c7"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def literal_errors(rows, threshold):
    if not math.isfinite(threshold):
        raise ValueError("Threshold must be finite")
    by_attack, failures, ties = {}, [], 0
    for row in rows:
        label, attack, score = row["label"], row["attack_id"], float(row["score"])
        if (label not in {"bonafide", "spoof"} or not math.isfinite(score)
                or row["split"] != "dev"
                or (label == "bonafide" and attack != "-")
                or (label == "spoof" and attack not in {f"A{i:02d}" for i in range(1, 7)})):
            raise ValueError("Invalid dev score/metadata")
        wrong = score < threshold if label == "bonafide" else score >= threshold
        group = by_attack.setdefault(attack, {"count": 0, "errors": 0})
        group["count"] += 1
        group["errors"] += int(wrong)
        ties += int(score == threshold)
        if wrong:
            failures.append({**row, "score": score,
                             "error": "bonafide_rejected" if label == "bonafide" else "spoof_accepted"})
    for group in by_attack.values():
        group["error_rate"] = group["errors"] / group["count"]
    ordered = sorted(failures, key=lambda row: (-abs(row["score"] - threshold), row["utterance_id"]))
    selected = {}
    for attack in sorted(by_attack):
        representative = next((row for row in ordered if row["attack_id"] == attack), None)
        if representative is not None:
            selected[representative["utterance_id"]] = representative
    for row in ordered:
        if len(selected) >= 10:
            break
        selected[row["utterance_id"]] = row
    return {"threshold": threshold, "decision": "bonafide iff score >= threshold",
            "threshold_source": "Reference EER threshold selected on the same dev labels; diagnostic only.",
            "threshold_ties": ties, "by_attack": dict(sorted(by_attack.items())),
            "total_errors": len(failures), "total_records": len(rows),
            "failure_selection": "Largest absolute wrong margin per attack, then fill to ten by absolute wrong margin; ties by ID. Illustrative, not a random sample.",
            "failure_examples": list(selected.values()),
            "limitation": "Literal threshold counts can differ from the reference EER grid at ties. No unseen-attack or deployment claim."}


def export(run_dir, destination):
    if destination.exists():
        raise ValueError("Preserve existing analysis output")
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "ok" or summary.get("scope") not in {
            "full-official-train-dev-adaptation", "full-official-train-dev-adaptation-scoring-recovery"}:
        raise ValueError("Requires a completed full dev run")
    if summary["configuration"]["data"]["eval_enabled"] is not False:
        raise ValueError("Eval must remain closed")
    if digest(run_dir / "scores.csv") != summary["output_sha256"]["scores.csv"] or digest(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("Score or protocol hash differs")
    expected = [line.split() for line in PROTOCOL.read_text().splitlines() if line.strip()]
    with (run_dir / "scores.csv").open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != ["utterance_id", "split", "label", "attack_id", "score"]:
            raise ValueError("Invalid score header")
        rows = list(reader)
    if len(rows) != 24844 or len({row["utterance_id"] for row in rows}) != 24844:
        raise ValueError("Requires complete unique dev scores")
    for row, fields in zip(rows, expected):
        if (row["utterance_id"], row["attack_id"], row["label"]) != (fields[1], fields[3], fields[4]):
            raise ValueError("Score order/metadata differs from official dev")
    if {row["utterance_id"] for row in rows} & set(summary["model_metadata"]["training_ids"]):
        raise ValueError("Train IDs overlap dev")
    result = {"status": "ok", "scope": "full-dev-literal-threshold-diagnostic",
              "seed": summary["experiment"]["seed"], "eval_enabled": False,
              "source_summary_sha256": digest(run_dir / "summary.json"),
              "scores_sha256": digest(run_dir / "scores.csv"),
              "protocol_sha256": PROTOCOL_SHA256, "analysis_source_sha256": digest(Path(__file__)),
              **literal_errors(rows, float(summary["metrics"]["eer_threshold"]))}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as target:
        json.dump(result, target, indent=2, allow_nan=False)
        target.write("\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = export(args.run_dir, args.output)
    print(json.dumps({"status": result["status"], "by_attack": result["by_attack"]}, indent=2))
