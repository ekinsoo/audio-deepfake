"""Validate and summarize all three frozen full-dev narrowband score sets."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_full_baseline as full
from recover_full_scoring import validate_scores
from run_robustness import select_failure_examples
from audio_deepfake.metrics import evaluate_scores
from audio_deepfake.reliability import evaluate_error_ranking

ROOT = full.ROOT
AUDIT = ROOT / "docs/validation/full_robustness_results_v1.json"
SEEDS = (42, 43, 44)
METADATA = ("utterance_id", "split", "speaker_id", "label", "attack_id")
NUMERIC = ("s_clean", "s_narrow", "s_average", "delta", "abs_delta")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_paired_rows(records, rows, clean, calibration_ids):
    """Require ordered exact joins and literal frozen arithmetic, never repair rows."""
    if len(records) != len(rows):
        raise ValueError("Paired row count differs from official dev")
    expected_fields = set(METADATA + NUMERIC + ("group",))
    for record, row in zip(records, rows):
        if set(record) != expected_fields:
            raise ValueError("Unexpected paired score fields")
        if any(record[name] != getattr(row, name) for name in METADATA):
            raise ValueError("Paired score IDs/order/metadata differ from official dev")
        for name in NUMERIC:
            value = record[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("Paired scores must be finite numbers")
        original, narrow = record["s_clean"], record["s_narrow"]
        expected = {"s_clean": clean[row.utterance_id], "s_average": original / 2 + narrow / 2,
                    "delta": narrow - original, "abs_delta": abs(narrow - original),
                    "group": "calibration" if row.utterance_id in calibration_ids else "analysis"}
        if any(record[name] != value for name, value in expected.items()):
            raise ValueError("Paired score clean value, arithmetic or group differs")
    return records


def aggregate_values(values):
    """Undefined seeds stay visible and make the three-seed aggregate incomplete."""
    if len(values) != 3:
        raise ValueError("Aggregate requires all three fixed seeds")
    defined = [value for value in values if value is not None]
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
           for value in defined):
        raise ValueError("Invalid aggregate metric")
    return {"mean": statistics.mean(defined) if defined else None,
            "std": statistics.pstdev(defined) if defined else None,
            "std_ddof": 0, "defined_seed_count": len(defined),
            "expected_seed_count": 3, "complete": len(defined) == 3}


def collect_chunks(output, seed, rows, clean, calibration_ids, preflight, contract):
    chunks = []
    for path in (output / "chunks").glob(f"seed{seed}_*.json"):
        chunk = read(path)
        if type(chunk.get("start")) is not int or type(chunk.get("stop")) is not int:
            raise ValueError("Chunk bounds must be integers")
        if path.name != f"seed{seed}_{chunk['start']:05d}_{chunk['stop']:05d}.json":
            raise ValueError("Chunk filename differs from bounds")
        chunks.append((chunk["start"], path, chunk))
    chunks.sort(key=lambda item: item[0])
    position, records, provenance = 0, [], []
    for start, path, chunk in chunks:
        expected = {"status": "ok", "protocol": contract["protocol"], "seed": seed,
                    "contract_sha256": preflight["contract_sha256"],
                    "worker_source_sha256": preflight["worker_source_sha256"],
                    "preflight_sha256": full.digest(output / "preflight.json"),
                    "execution_sha256": preflight["execution_sha256"],
                    "input_sha256": preflight["input_sha256"],
                    "model_sha256": contract["seeds"][str(seed)]["model_sha256"]}
        if any(chunk.get(key) != value for key, value in expected.items()):
            raise ValueError("Chunk provenance differs from frozen run")
        stop = chunk["stop"]
        if (start != position or not start < stop <= len(rows) or stop - start > 2000
                or len(chunk["rows"]) != stop - start):
            raise ValueError("Missing, duplicate, overlapping or invalid chunk bounds")
        elapsed = chunk.get("processing_seconds")
        if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Invalid chunk processing time")
        records.extend(chunk["rows"])
        provenance.append({"file": path.relative_to(output).as_posix(), "sha256": full.digest(path),
                           "start": start, "stop": stop, "processing_seconds": elapsed,
                           "initial_memory": chunk.get("initial_memory"), "final_memory": chunk.get("final_memory")})
        position = stop
    if position != len(rows):
        raise ValueError("Incomplete full-dev chunk population")
    if sum(item["processing_seconds"] for item in provenance) > contract["resources"]["max_processing_seconds_per_seed"]:
        raise ValueError("Per-seed processing budget exceeded")
    validate_paired_rows(records, rows, clean, calibration_ids)
    return records, provenance


def validate_supervision(supervision, preflight, chunks, smoke_seconds):
    if (supervision.get("status") != "scoring_complete"
            or supervision.get("source_sha256") != preflight["execution_source_sha256"]
            or set(supervision.get("seed_wall_seconds", {})) != {str(seed) for seed in SEEDS}
            or supervision.get("seed_smoke_seconds") != smoke_seconds):
        raise ValueError("Invalid supervision pins or population")
    expected_names = ["preflight", "smoke"] + [Path(item["file"]).stem for seed in SEEDS for item in chunks[str(seed)]]
    stages = supervision.get("stages", [])
    if [item.get("name") for item in stages] != expected_names:
        raise ValueError("Supervised stages differ from complete chunk set")
    for stage in stages:
        elapsed, budget, peak = (stage.get(key) for key in ("elapsed_seconds", "budget_seconds", "peak_rss_bytes"))
        if (stage.get("status") != "complete" or stage.get("exit_code") != 0
                or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                       for value in (elapsed, budget, peak))
                or not 0 <= elapsed <= budget <= 1800 or not 0 <= peak <= 2 * 1024**3):
            raise ValueError("Supervised stage failed or exceeded resource budget")
        memory = stage.get("last_memory", {})
        if (memory.get("system_available_ram_bytes", 0) < 1024**3
                or memory.get("child_peak_rss_bytes", 0) > 2 * 1024**3):
            raise ValueError("Supervised stage memory guard differs")
    for seed in SEEDS:
        elapsed = sum(stage["elapsed_seconds"] for stage in stages if stage["name"].startswith(f"seed{seed}_")) + smoke_seconds[str(seed)]
        # Supervisor stores 1800 - (1800 - sum), which can round at one ULP.
        measured = supervision["seed_wall_seconds"][str(seed)]
        if (isinstance(measured, bool) or not isinstance(measured, (int, float))
                or not math.isfinite(measured) or not 0 <= measured <= 1800
                or not math.isclose(measured, elapsed, rel_tol=0, abs_tol=1e-9)):
            raise ValueError("Measured per-seed wall time differs from stages")


def summarize_seed(seed, records, rows, analysis, contract, plan):
    pairs = {name: [(record["utterance_id"], record[column]) for record in records]
             for name, column in (("clean", "s_clean"), ("narrowband", "s_narrow"), ("average", "s_average"))}
    metrics = {name: evaluate_scores(rows, values, plan["asv"], plan["reference"], evaluation_scope="full")
               for name, values in pairs.items()}
    clean_path = ROOT / contract["seeds"][str(seed)]["clean_scores_path"]
    parent = read(clean_path.parent / "summary.json")
    if metrics["clean"] != parent["metrics"]:
        raise ValueError("Recomputed clean metrics differ from parent result")
    ids = {row.utterance_id for row in analysis}
    ranking = evaluate_error_ranking(analysis, [pair for pair in pairs["clean"] if pair[0] in ids],
                                     [pair for pair in pairs["narrowband"] if pair[0] in ids],
                                     contract["seeds"][str(seed)]["calibration"]["threshold"])
    return {"seed": seed, "metrics": metrics, "reliability": ranking,
            "failure_examples": select_failure_examples(ranking),
            "model_metadata": parent["model_metadata"],
            "detection_scopes": {"clean": "official dev Python adaptation",
                                 "narrowband": "custom bandwidth stress; unchanged official ASV reference",
                                 "average": "custom fixed score average; unchanged official ASV reference"}}


def finalize(output):
    import full_robustness_worker as worker
    plan, contract, config, groups, calibration, analysis = worker.frozen_inputs(output)
    preflight, smoke = read(output / "preflight.json"), read(output / "smoke.json")
    if (smoke.get("status") != "ok" or smoke.get("protocol") != contract["protocol"]
            or smoke.get("no_metric_tuning") is not True
            or [item.get("seed") for item in smoke.get("seeds", [])] != list(SEEDS)):
        raise ValueError("Successful bounded smoke check required")
    for item in smoke["seeds"]:
        if (item.get("status") != "ok" or item.get("contract_sha256") != preflight["contract_sha256"]
                or item.get("execution_sha256") != preflight["execution_sha256"]
                or item.get("preflight_sha256") != full.digest(output / "preflight.json")
                or item.get("start") != 0 or item.get("stop") != 20
                or len(item.get("clean_reproduction", [])) != 20):
            raise ValueError("Smoke provenance or bounded population differs")
        seed = str(item["seed"])
        clean = dict(validate_scores(ROOT / contract["seeds"][seed]["clean_scores_path"], groups["dev"]))
        validate_paired_rows(item["rows"], groups["dev"][:20], clean, {row.utterance_id for row in calibration})
        for check, row in zip(item["clean_reproduction"], groups["dev"][:20]):
            if (check.get("utterance_id") != row.utterance_id or check.get("saved_score") != clean[row.utterance_id]
                    or not math.isfinite(check.get("reproduced_score", float("nan")))
                    or abs(check["reproduced_score"] - clean[row.utterance_id]) > 1e-10
                    or check.get("absolute_difference") != abs(check["reproduced_score"] - clean[row.utterance_id])):
                raise ValueError("Smoke clean reproduction differs")
    if (output / "summary.json").exists() or AUDIT.exists():
        raise ValueError("Preserve completed results; use a new experiment version")
    for name, expected in preflight["execution_source_sha256"].items():
        if full.digest(ROOT / name) != expected:
            raise ValueError("Execution source changed before finalization")
    chunk_paths = list((output / "chunks").glob("*.json"))
    if any(not any(path.name.startswith(f"seed{seed}_") for seed in SEEDS) for path in chunk_paths):
        raise ValueError("Unexpected seed chunk")
    results, tables, chunks = {}, {}, {}
    for seed in SEEDS:
        clean = dict(validate_scores(ROOT / contract["seeds"][str(seed)]["clean_scores_path"], groups["dev"]))
        records, provenance = collect_chunks(output, seed, groups["dev"], clean,
                                             {row.utterance_id for row in calibration}, preflight, contract)
        tables[str(seed)] = records
        metric_plan = {"asv": ROOT / config["paths"].get("asv_scores", full.DEFAULT_ASV),
                       "reference": ROOT / config["paths"].get("reference_evaluator", full.DEFAULT_REFERENCE)}
        results[str(seed)] = summarize_seed(seed, records, groups["dev"], analysis, contract, metric_plan)
        chunks[str(seed)] = provenance
    supervision = read(output / "supervision.json")
    validate_supervision(supervision, preflight, chunks,
                         {str(item["seed"]): item["processing_seconds"] for item in smoke["seeds"]})
    # One final complete audio hash check prevents post-preflight input drift.
    checked = full.preflight(config, output, owned_output=True)
    if checked["input_sha256"] != preflight["input_sha256"]:
        raise ValueError("Full audio verification inputs changed")
    # Nothing is published until every seed passes exact joins and metric checks.
    runs = []
    for seed in SEEDS:
        key = str(seed)
        directory = output / f"seed{seed}"
        directory.mkdir(exist_ok=True)
        table = directory / "paired_scores.csv"
        if table.exists():
            raise ValueError("Preserve existing paired table")
        with table.open("x", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=METADATA + NUMERIC + ("group",), lineterminator="\n")
            writer.writeheader()
            writer.writerows(tables[key])
        full.atomic_json(directory / "analysis.json", results[key])
        compact = {name: value for name, value in results[key].items() if name != "model_metadata"}
        compact["reliability"] = {name: value for name, value in compact["reliability"].items() if name != "records"}
        compact.update(pins=contract["seeds"][key], output_sha256={"paired_scores.csv": full.digest(table),
                       "analysis.json": full.digest(directory / "analysis.json")},
                       chunks=chunks[key], processing_seconds=sum(item["processing_seconds"] for item in chunks[key]))
        runs.append(compact)
    aggregate = {condition: {metric: aggregate_values([run["metrics"][condition][metric] for run in runs])
                            for metric in ("eer", "min_tDCF")}
                 for condition in ("clean", "narrowband", "average")}
    ranking = {name: {metric: aggregate_values([run["reliability"]["rankings"][name][metric] for run in runs])
                      for metric in ("auroc", "average_precision")}
               for name in ("abs_delta", "negative_margin")}
    result = {"status": "ok", "protocol": contract["protocol"], "finished_utc": full.timestamp(),
              "eval_enabled": False, "retraining": False, "dataset": config["data"],
              "contract_sha256": preflight["contract_sha256"], "input_sha256": preflight["input_sha256"],
              "execution_source_sha256": preflight["execution_source_sha256"],
              "frozen_source_sha256": contract["source_sha256"], "transform": contract["transform"],
              "partition": contract["partition"], "rate_units": "fraction", "std_ddof": 0,
              "seeds": list(SEEDS), "runs": runs, "detection_aggregate": aggregate,
              "ranking_aggregate": ranking, "error_prevalence_aggregate": aggregate_values([run["reliability"]["error_prevalence"] for run in runs]),
              "limitations": contract["limitations"],
              "preflight_sha256": full.digest(output / "preflight.json"), "smoke_sha256": full.digest(output / "smoke.json"),
              "preflight_seconds": preflight["preflight_seconds"]}
    result.update(supervision=supervision, supervision_sha256=full.digest(output / "supervision.json"),
                  final_audio_records_verified=50224)
    full.atomic_json(output / "summary.json", result)
    public = {**result, "runs": [{key: value for key, value in run.items() if key != "chunks"} for run in runs],
              "local_summary_sha256": full.digest(output / "summary.json")}
    full.atomic_json(AUDIT, public)
    report_path = ROOT / "reports.md"
    with report_path.open("a", encoding="utf-8") as target:
        target.write(f"\n\n## Full-dev narrowband experiment completed\n\nCompleted {result['finished_utc']}. All 24,844 dev recordings were scored with each fixed model seed (42, 43, 44). "
                     "Clean-score joins and official metric recomputation passed. Narrowband and fixed averaging are custom stress diagnostics with unchanged ASV scores. "
                     "Error rankings use the frozen clean calibration thresholds and analysis speakers. "
                     "This reuses inspected dev data and is not independent validation. "
                     "See [the complete audit](docs/validation/full_robustness_results_v1.json) for each seed, population aggregates, failure examples and provenance.\n")
    backlog = ROOT / "backlog.md"
    previous = backlog.read_text(encoding="utf-8")
    backlog.write_text("# Latest milestone\n\nFull-dev narrowband scoring completed for seeds 42, 43 and 44. "
                       "Review [the results](docs/validation/full_robustness_results_v1.json), compare the two fixed error rankings and update the project narrative before choosing another experiment. Eval remains closed.\n\n" + previous, encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = finalize(args.output_dir.resolve())
    print(json.dumps({"status": result["status"], "detection_aggregate": result["detection_aggregate"],
                      "ranking_aggregate": result["ranking_aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
