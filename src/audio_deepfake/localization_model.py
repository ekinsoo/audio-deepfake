"""Small supervised frame classifier; no dataset loading or automatic training."""

from dataclasses import asdict, dataclass
from importlib.metadata import version
import hashlib
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


@dataclass(frozen=True)
class TemporalConfig:
    seed: int = 45
    frames_per_class: int = 50_000
    max_iter: int = 300
    regularization_c: float = 1.0

    def __post_init__(self):
        for name in ("seed", "frames_per_class", "max_iter"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not 0 <= self.seed < 2**32 - 1 or min(self.frames_per_class, self.max_iter) < 1:
            raise ValueError("Invalid seed, frame budget or iteration budget")
        if (isinstance(self.regularization_c, bool)
                or not np.isfinite(self.regularization_c) or self.regularization_c <= 0):
            raise ValueError("C must be finite and positive")


@dataclass
class TemporalModel:
    scaler: StandardScaler
    classifier: LogisticRegression
    metadata: dict


def _matrix(value):
    x = np.asarray(value)
    if (x.ndim != 2 or not len(x) or x.shape[1] != 60
            or x.dtype.kind != "f" or not np.isfinite(x).all()):
        raise ValueError("Expected finite floating-point frames x 60 features")
    return x


def train_temporal(records, loader, *, config=TemporalConfig(), guard=lambda: None):
    """Fit on train only, from a caller-validated feature/label loader.

    Records need split and utterance_id fields. loader(record) returns aligned
    20 ms features and binary labels (0=spoof, 1=bona fide). The caller must
    resolve native annotation alignment and nonspeech before using this API.
    No dev data is requested or used to fit normalization. Uniform random
    priority sampling caps each class, then balances retained counts.
    """
    records = list(records)
    ids = [r["utterance_id"] for r in records]
    if (not records or any(r["split"] != "train" for r in records)
            or any(not isinstance(uid, str) or not uid for uid in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("Require unique nonempty train IDs exclusively")
    rngs = [np.random.default_rng(config.seed + i) for i in (0, 1)]
    retained = [np.empty((0, 60), dtype=np.float32) for _ in (0, 1)]
    priorities = [np.empty(0) for _ in (0, 1)]
    seen = [0, 0]
    for row in sorted(records, key=lambda r: r["utterance_id"]):
        guard()
        features, labels = loader(row)
        features = _matrix(features)
        labels = np.asarray(labels)
        if (labels.ndim != 1 or len(labels) != len(features)
                or labels.dtype.kind not in "iuf" or not np.isin(labels, [0, 1]).all()):
            raise ValueError("Require one native binary label per feature bin")
        for label in (0, 1):
            frames = features[labels == label]
            seen[label] += len(frames)
            with np.errstate(over="ignore"):
                frames = frames.astype(np.float32)
            if not np.isfinite(frames).all():
                raise ValueError("Features exceed the finite float32 sampling range")
            values = np.concatenate((retained[label], frames))
            keys = np.r_[priorities[label], rngs[label].random(len(frames))]
            if len(keys) > config.frames_per_class:
                keep = np.argpartition(keys, config.frames_per_class - 1)[:config.frames_per_class]
                values, keys = values[keep], keys[keep]
            order = np.argsort(keys, kind="stable")
            retained[label], priorities[label] = values[order], keys[order]
    count = min(map(len, retained))
    if count < 2:
        raise ValueError("Need at least two labelled frames from each class")
    x = np.concatenate([values[:count] for values in retained])
    y = np.repeat([0, 1], count)
    guard()
    scaler = StandardScaler()
    classifier = LogisticRegression(C=config.regularization_c, solver="lbfgs",
                                    max_iter=config.max_iter, random_state=config.seed)
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        classifier.fit(scaler.fit_transform(x), y)
    guard()
    metadata = {"implementation": "lfcc-standardscaler-logistic-temporal-v1",
                "training_ids": sorted(ids), "seed": config.seed, "config": asdict(config),
                "frames_seen_by_native_class": seen, "frames_retained_per_class": count,
                "retained_feature_bytes_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
                "frame_cap_per_class": config.frames_per_class,
                "sampling": "uniform random priority, sorted IDs; balance to smaller retained class",
                "normalization": "StandardScaler fitted only on retained training frames",
                "score_direction": "higher supports bona fide; uncalibrated decision_function",
                "numerical_threads": 1, "iterations": classifier.n_iter_.tolist(),
                "warnings": [str(w.message) for w in caught],
                "versions": {p: version(p) for p in ("numpy", "scikit-learn", "scipy", "threadpoolctl")}}
    return TemporalModel(scaler, classifier, metadata)


def score_temporal(model, features):
    """Return one uncalibrated bona fide-directed score per aligned bin."""
    features = _matrix(features)
    if not np.array_equal(model.classifier.classes_, [0, 1]):
        raise ValueError("Unexpected native classifier class order")
    with threadpool_limits(limits=1):
        scores = model.classifier.decision_function(model.scaler.transform(features))
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite temporal scores")
    return scores
