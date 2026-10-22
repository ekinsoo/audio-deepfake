"""Validated adapter for the external official ASVspoof 2019 v1 evaluator.

The evaluator source is CC BY-NC-SA 4.0 and stays outside Git with its license.
This module loads the hash-pinned original; it does not implement its formulas.
See docs/METRICS.md for reference tie behavior and benchmark limitations.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from audio_deepfake.manifests import ManifestRow


REFERENCE_SHA256 = "93ea505282d329ed14537d977a74b2d1a25d161ac6767a16e0216a1a18391049"
DEFAULT_REFERENCE = Path(__file__).resolve().parents[2] / "data/metadata/asvspoof2019/metrics/eval_metrics.py"


def _reference(reference_path: str | Path | None):
    path = Path(reference_path) if reference_path is not None else DEFAULT_REFERENCE
    if not path.is_file():
        raise ValueError("Official 2019 evaluator is missing; supply the external eval_metrics.py")
    if hashlib.sha256(path.read_bytes()).hexdigest() != REFERENCE_SHA256:
        raise ValueError("Official evaluator SHA256 does not match the pinned 2019 v1 reference")
    spec = importlib.util.spec_from_file_location("_official_asvspoof2019_v1", path)
    if spec is None or spec.loader is None:
        raise ValueError("Could not load official evaluator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _finite_scores(values, name: str) -> np.ndarray:
    try:
        scores = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric scores") from exc
    if scores.ndim != 1 or scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError(f"{name} must contain non-empty finite one-dimensional scores")
    return scores


def evaluate_eer(bonafide_scores, spoof_scores, reference_path: str | Path | None = None) -> dict:
    """Return reference-grid EER as a fraction, including its original tie behavior."""
    bona = _finite_scores(bonafide_scores, "Bonafide class")
    spoof = _finite_scores(spoof_scores, "Spoof class")
    eer, threshold = _reference(reference_path).compute_eer(bona, spoof)
    return {"eer": float(eer), "threshold": float(threshold),
            "reference_version": "ASVspoof2019 tDCF_python_v1", "reference_sha256": REFERENCE_SHA256}


def _asv_scores(path: str | Path) -> tuple[dict[str, np.ndarray], str]:
    path = Path(path)
    contents = path.read_bytes()
    groups: dict[str, list[float]] = {"target": [], "nontarget": [], "spoof": []}
    for number, line in enumerate(contents.decode("utf-8-sig").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 3:
            raise ValueError(f"ASV line {number} must have three fields")
        _, label, value = fields
        if label not in groups:
            raise ValueError(f"Unknown ASV label on line {number}")
        try:
            score = float(value)
        except ValueError as exc:
            raise ValueError(f"Invalid ASV score on line {number}") from exc
        if not np.isfinite(score):
            raise ValueError(f"Non-finite ASV score on line {number}")
        groups[label].append(score)
    return ({key: _finite_scores(values, f"ASV {key}") for key, values in groups.items()},
            hashlib.sha256(contents).hexdigest())


def evaluate_scores(
    rows: Sequence[ManifestRow],
    score_rows: Iterable[tuple[str, float]],
    asv_path: str | Path,
    reference_path: str | Path | None = None,
    *,
    evaluation_scope: str = "pilot",
) -> dict:
    """Join complete CM scores by ID and run the unchanged 2019 v1 metrics.

    Supply pairs rather than a mapping so duplicate score IDs remain detectable.
    Scope labels the caller's manifest, not independent benchmark certification.
    All values are fractions (not percentages); higher scores must mean bona fide.
    The caller must supply the matching LA split's official ASV file. This adapter
    records its hash but does not guarantee a filename/hash match to the split;
    custom numerical fixtures can be supplied for reference validation.
    """
    if evaluation_scope not in {"pilot", "full"}:
        raise ValueError("evaluation_scope must be pilot or full")
    if not rows:
        raise ValueError("Manifest must not be empty")
    ids = [row.utterance_id for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate manifest utterance ID")
    if any(not isinstance(utterance, str) or not utterance for utterance in ids):
        raise ValueError("Manifest IDs must be non-empty strings")
    labels = np.asarray([row.label for row in rows])
    if set(labels) != {"bonafide", "spoof"}:
        raise ValueError("Manifest must contain both bonafide and spoof classes and no unknown labels")
    splits = {row.split for row in rows}
    if len(splits) != 1 or not splits <= {"train", "dev", "eval"}:
        raise ValueError("Evaluate exactly one known manifest split")
    scores_by_id = {}
    for item in score_rows:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError("Each CM score must be an (utterance_id, score) pair")
        utterance, score = item
        if not isinstance(utterance, str) or not utterance:
            raise ValueError("Score IDs must be non-empty strings")
        if utterance in scores_by_id:
            raise ValueError("Duplicate CM score utterance ID")
        scores_by_id[utterance] = score
    if set(scores_by_id) != set(ids):
        missing = len(set(ids) - set(scores_by_id))
        extra = len(set(scores_by_id) - set(ids))
        raise ValueError(f"CM score IDs do not match manifest: {missing} missing, {extra} extra")
    scores = _finite_scores([scores_by_id[utterance] for utterance in ids], "CM")
    if np.unique(scores).size < 3:
        raise ValueError("2019 t-DCF requires at least three unique soft CM scores")
    asv, asv_sha256 = _asv_scores(asv_path)
    reference = _reference(reference_path)
    asv_eer, asv_threshold = reference.compute_eer(asv["target"], asv["nontarget"])
    pfa, pmiss, pmiss_spoof = reference.obtain_asv_error_rates(
        asv["target"], asv["nontarget"], asv["spoof"], asv_threshold)
    costs = dict(Pspoof=0.05, Ptar=0.9405, Pnon=0.0095,
                 Cmiss_asv=1, Cfa_asv=10, Cmiss_cm=1, Cfa_cm=10)
    bona, spoof = scores[labels == "bonafide"], scores[labels == "spoof"]
    try:
        with np.errstate(divide="ignore", invalid="ignore"):
            curve, thresholds = reference.compute_tDCF(bona, spoof, pfa, pmiss, pmiss_spoof, costs, False)
    except SystemExit as exc:
        raise ValueError("Official evaluator rejected the ASV operating point or score inputs") from exc
    if not np.isfinite(curve).all():
        raise ValueError("Official t-DCF is undefined for these ASV rates (zero normalizer)")
    eer, eer_threshold = reference.compute_eer(bona, spoof)
    best = int(np.argmin(curve))
    return {
        "evaluation_scope": evaluation_scope, "split": next(iter(splits)),
        "scope_note": "Caller-supplied manifest; pilot/full label does not certify official benchmark completeness.",
        "reference_version": "ASVspoof2019 tDCF_python_v1", "reference_sha256": REFERENCE_SHA256,
        "asv_score_sha256": asv_sha256, "score_direction": "higher_is_bonafide", "rate_units": "fraction",
        "cm_counts": {"bonafide": len(bona), "spoof": len(spoof)},
        "eer": float(eer), "eer_threshold": float(eer_threshold),
        "min_tDCF": float(curve[best]), "min_tDCF_threshold": float(thresholds[best]),
        "asv": {"eer": float(asv_eer), "eer_threshold": float(asv_threshold),
                "Pfa": float(pfa), "Pmiss": float(pmiss), "Pmiss_spoof": float(pmiss_spoof),
                "counts": {key: len(values) for key, values in asv.items()}},
        "cost_model": costs,
    }
