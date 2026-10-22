"""Pure helpers for the frozen narrowband reliability analysis protocol."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from typing import Iterable, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from audio_deepfake.manifests import ManifestRow, summarize, validate_splits


PROTOCOL = "narrowband-reliability-v1"
ATTACKS = {f"A{index:02d}" for index in range(1, 7)}


def _dev_rows(rows: Sequence[ManifestRow]) -> list[ManifestRow]:
    rows = list(rows)
    if not rows:
        raise ValueError("Expected nonempty dev rows")
    validate_splits({"dev": rows})
    return sorted(rows, key=lambda row: row.utterance_id)


def _manifest_hash(rows: Sequence[ManifestRow]) -> str:
    content = json.dumps([asdict(row) for row in rows], sort_keys=True,
                         separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def partition_dev(train_rows: Sequence[ManifestRow], dev_rows: Sequence[ManifestRow]) -> tuple[list[ManifestRow], list[ManifestRow], dict]:
    """Split speakers once by the fixed hash; fail rather than search seeds."""
    train, dev = list(train_rows), list(dev_rows)
    if not train or not dev:
        raise ValueError("Expected nonempty train and dev rows")
    validate_splits({"train": train, "dev": dev})
    rank = lambda speaker: (hashlib.sha256(f"42\0{PROTOCOL}\0{speaker}".encode("utf-8")).hexdigest(), speaker)
    speakers = sorted({row.speaker_id for row in dev}, key=rank)
    calibration_speakers = set(speakers[:2 * len(speakers) // 5])
    calibration = sorted((row for row in dev if row.speaker_id in calibration_speakers), key=lambda row: row.utterance_id)
    analysis = sorted((row for row in dev if row.speaker_id not in calibration_speakers), key=lambda row: row.utterance_id)
    for name, rows in (("calibration", calibration), ("analysis", analysis)):
        if {row.label for row in rows} != {"bonafide", "spoof"}:
            raise ValueError(f"{name} requires both labels; no alternate partition is allowed")
        if {row.attack_id for row in rows if row.label == "spoof"} != ATTACKS:
            raise ValueError(f"{name} requires every attack A01-A06; no alternate partition is allowed")
    metadata = {"protocol": PROTOCOL, "seed": 42,
                "speaker_rank": "SHA256(UTF8('42\\0narrowband-reliability-v1\\0{speaker_id}'))",
                "calibration_fraction": "floor(2 * n_speakers / 5)",
                "manifest_hash_encoding": "SHA256 of ID-sorted full-row JSON; sorted keys, compact separators, ASCII",
                "train_manifest_sha256": _manifest_hash(sorted(train, key=lambda row: row.utterance_id))}
    for name, rows in (("calibration", calibration), ("analysis", analysis)):
        metadata[name] = {**summarize(rows), "speaker_ids": sorted({row.speaker_id for row in rows}),
                          "utterance_ids": [row.utterance_id for row in rows],
                          "manifest_sha256": _manifest_hash(rows)}
    return calibration, analysis, metadata


def _join(rows: Sequence[ManifestRow], pairs: Iterable[tuple[str, float]]) -> dict[str, float]:
    expected = {row.utterance_id for row in rows}
    result = {}
    for utterance, score in pairs:
        if utterance in result:
            raise ValueError(f"Duplicate score ID: {utterance}")
        if isinstance(score, (bool, np.bool_)):
            raise ValueError("Scores must be finite numbers")
        try:
            value = float(score)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("Scores must be finite numbers") from error
        if not math.isfinite(value):
            raise ValueError("Scores must be finite numbers")
        result[utterance] = value
    if set(result) != expected:
        raise ValueError("Score IDs must exactly match manifest IDs")
    return result


def calibration_threshold(rows: Sequence[ManifestRow], pairs: Iterable[tuple[str, float]]) -> float:
    """Select clean-score BF acceptance >=95%; ties use bonafide score >= t."""
    rows = _dev_rows(rows)
    if {row.label for row in rows} != {"bonafide", "spoof"}:
        raise ValueError("Calibration requires both labels")
    scores = _join(rows, pairs)
    bonafide = sorted(scores[row.utterance_id] for row in rows if row.label == "bonafide")
    return bonafide[len(bonafide) // 20]


def _ranking(errors: list[bool], values: list[float]) -> dict:
    if len(set(errors)) < 2:
        return {"auroc": None, "average_precision": None,
                "undefined_reason": "No errors" if not any(errors) else "No correct predictions"}
    return {"auroc": float(roc_auc_score(errors, values)),
            "average_precision": float(average_precision_score(errors, values)),
            "undefined_reason": None}


def evaluate_error_ranking(analysisrows: Sequence[ManifestRow], clean_pairs: Iterable[tuple[str, float]],
                           narrow_pairs: Iterable[tuple[str, float]], threshold: float) -> dict:
    """Rank clean-prediction errors using two untrained, fixed diagnostics."""
    rows = _dev_rows(analysisrows)
    if isinstance(threshold, (bool, np.bool_)) or not math.isfinite(float(threshold)):
        raise ValueError("Threshold must be finite")
    threshold = float(threshold)
    clean, narrow = _join(rows, clean_pairs), _join(rows, narrow_pairs)
    records = []
    for row in rows:
        original, transformed = clean[row.utterance_id], narrow[row.utterance_id]
        delta = transformed - original
        margin = -abs(original - threshold)
        if not math.isfinite(delta) or not math.isfinite(margin):
            raise ValueError("Derived score difference or margin is non-finite")
        predicted = "bonafide" if original >= threshold else "spoof"
        records.append({"utterance_id": row.utterance_id, "label": row.label, "attack_id": row.attack_id,
                        "predicted_label": predicted, "error": predicted != row.label,
                        "delta": delta, "abs_delta": abs(delta), "negative_margin": margin})
    errors = [record["error"] for record in records]
    output = {"count": len(records), "errors": sum(errors), "correct": len(records) - sum(errors),
              "error_prevalence": sum(errors) / len(records), "threshold": threshold,
              "prediction_rule": "bonafide iff clean_score >= threshold", "records": records,
              "rankings": {key: _ranking(errors, [record[key] for record in records])
                           for key in ("abs_delta", "negative_margin")}}
    for dimension in ("label", "attack_id"):
        output["by_" + dimension] = {}
        for key in sorted({record[dimension] for record in records}):
            group = [record for record in records if record[dimension] == key]
            count = sum(record["error"] for record in group)
            output["by_" + dimension][key] = {"count": len(group), "errors": count, "correct": len(group) - count}
    return output
