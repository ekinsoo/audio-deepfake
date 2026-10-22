"""Dimension-aware frozen SSL head, matching the hash-frozen LFCC sampler.

No encoder, dataset access or implicit training. Native labels are 0=spoof,
1=bona fide. Sampling identities use canonical JSON [ID, bin] pairs in fitted
class-0 then class-1 priority order; their digest is feature-independent.
"""

from dataclasses import dataclass
import hashlib
from importlib.metadata import version
import json
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .localization_model import TemporalConfig


def sample_digest(identities):
    payload = json.dumps(identities, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def matrix(value, dimensions):
    value = np.asarray(value)
    if (value.ndim != 2 or not len(value) or value.shape[1] != dimensions
            or value.dtype != np.float32 or not np.isfinite(value).all()):
        raise ValueError(f"Expected finite float32 frames x {dimensions} features")
    return value


def train_rows(records):
    records = list(records)
    ids = [r["utterance_id"] for r in records]
    if (not records or any(r["split"] != "train" for r in records)
            or any(not isinstance(uid, str) or not uid for uid in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("Require unique nonempty train IDs exclusively")
    return sorted(records, key=lambda r: r["utterance_id"])


def binary_labels(value, count):
    value = np.asarray(value)
    if (value.ndim != 1 or len(value) != count or value.dtype.kind not in "iuf"
            or not np.isin(value, [0, 1]).all()):
        raise ValueError("Require one native binary label per feature bin")
    return value


def reference_sampling(records, label_loader, *, config=TemporalConfig(), include_ids=False):
    """Independent full-priority reference for the LFCC algorithm's frame IDs.

    Unlike the bounded implementation, collect all priorities before sorting.
    Only labels and IDs enter this comparison, never dev features or scores.
    """
    rngs = [np.random.default_rng(config.seed + i) for i in (0, 1)]
    pairs, priorities = [[], []], [[], []]
    for row in train_rows(records):
        labels = np.asarray(label_loader(row))
        labels = binary_labels(labels, len(labels))
        for label in (0, 1):
            bins = np.flatnonzero(labels == label)
            pairs[label].extend([[row["utterance_id"], int(i)] for i in bins])
            priorities[label].extend(rngs[label].random(len(bins)).tolist())
    count = min(config.frames_per_class, *(len(p) for p in pairs))
    if count < 2:
        raise ValueError("Need at least two labelled frames from each class")
    identities = []
    for label in (0, 1):
        order = np.argsort(priorities[label], kind="stable")[:count]
        identities.extend(pairs[label][int(i)] for i in order)
    result = {"retained_sample_ids_sha256": sample_digest(identities),
            "frames_retained_per_class": count,
            "frames_seen_by_native_class": list(map(len, pairs))}
    if include_ids:
        result["retained_sample_ids"] = identities
    return result


@dataclass
class SSLHead:
    scaler: StandardScaler
    classifier: LogisticRegression
    metadata: dict


def train_ssl_head(records, loader, *, dimensions=768, config=TemporalConfig(),
                   expected_retained_per_class=None, expected_sample_digest=None,
                   guard=lambda: None):
    if isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions < 1:
        raise ValueError("Dimensions must be a positive integer")
    records = train_rows(records)
    rngs = [np.random.default_rng(config.seed + i) for i in (0, 1)]
    retained = [np.empty((0, dimensions), dtype=np.float32) for _ in (0, 1)]
    priorities = [np.empty(0) for _ in (0, 1)]
    identities, seen = [[], []], [0, 0]
    for row in records:
        guard()
        features, labels = loader(row)
        features = matrix(features, dimensions)
        labels = binary_labels(labels, len(features))
        for label in (0, 1):
            bins = np.flatnonzero(labels == label)
            seen[label] += len(bins)
            values = np.concatenate((retained[label], features[bins]))
            keys = np.r_[priorities[label], rngs[label].random(len(bins))]
            pairs = identities[label] + [[row["utterance_id"], int(i)] for i in bins]
            if len(keys) > config.frames_per_class:
                keep = np.argpartition(keys, config.frames_per_class - 1)[:config.frames_per_class]
                values, keys = values[keep], keys[keep]
                pairs = [pairs[int(i)] for i in keep]
            order = np.argsort(keys, kind="stable")
            retained[label], priorities[label] = values[order], keys[order]
            identities[label] = [pairs[int(i)] for i in order]
    count = min(map(len, retained))
    if count < 2 or (expected_retained_per_class is not None and count != expected_retained_per_class):
        raise ValueError("Retained class count differs from the frozen comparison")
    selected_ids = identities[0][:count] + identities[1][:count]
    digest = sample_digest(selected_ids)
    if expected_sample_digest is not None and digest != expected_sample_digest:
        raise ValueError("SSL sampling differs from the LFCC reference frame identities")
    x = np.concatenate([values[:count] for values in retained])
    y = np.repeat([0, 1], count)
    scaler = StandardScaler()
    classifier = LogisticRegression(C=config.regularization_c, solver="lbfgs",
                                    max_iter=config.max_iter, random_state=config.seed)
    guard()
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.simplefilter("error", ConvergenceWarning)
        classifier.fit(scaler.fit_transform(x), y)
    guard()
    if caught:
        raise RuntimeError("Fit emitted warnings; no SSL results accepted")
    metadata = {"implementation": "frozen-ssl-standardscaler-logistic-v1",
                "dimensions": dimensions, "training_ids": [r["utterance_id"] for r in records],
                "seed": config.seed, "frames_seen_by_native_class": seen,
                "frames_retained_per_class": count, "frame_cap_per_class": config.frames_per_class,
                "retained_sample_ids_sha256": digest, "retained_sample_ids": selected_ids,
                "sample_digest_encoding": "UTF-8 compact ASCII JSON [ID,bin] pairs; class0 then class1 priority order",
                "retained_feature_bytes_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
                "normalization": "StandardScaler fitted only on retained training frames",
                "score_direction": "higher supports bona fide; uncalibrated decision_function",
                "C": config.regularization_c, "solver": "lbfgs", "max_iter": config.max_iter,
                "numerical_threads": 1, "iterations": classifier.n_iter_.tolist(), "warnings": [],
                "versions": {p: version(p) for p in ("numpy", "scikit-learn", "scipy", "threadpoolctl")}}
    return SSLHead(scaler, classifier, metadata)


def score_ssl_head(model, features):
    features = matrix(features, model.metadata["dimensions"])
    if not np.array_equal(model.classifier.classes_, [0, 1]):
        raise ValueError("Unexpected native classifier class order")
    with threadpool_limits(limits=1):
        scores = model.classifier.decision_function(model.scaler.transform(features))
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite SSL scores")
    return scores
