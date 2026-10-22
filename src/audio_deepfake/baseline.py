"""Bounded-memory LFCC-GMM pilot, with an explicit training split boundary."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Callable, Sequence
import warnings

import joblib
import numpy as np
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits

from .audio import load_audio
from .features import LFCCConfig, extract_lfcc
from .manifests import ManifestRow, check_audio_files, validate_splits


@dataclass(frozen=True)
class GMMConfig:
    n_components: int = 32
    seed: int = 42
    frames_per_class: int = 50_000
    reg_covar: float = 1e-6
    tol: float = 1e-3
    max_iter: int = 50

    def __post_init__(self):
        for name in ("n_components", "frames_per_class", "max_iter"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.frames_per_class < max(2, self.n_components):
            raise ValueError("Frame budget must be at least two and >= n_components")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        for name in ("reg_covar", "tol"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass
class BaselineModel:
    bonafide: GaussianMixture
    spoof: GaussianMixture
    lfcc_config: LFCCConfig
    gmm_config: GMMConfig
    metadata: dict


def _features(value: np.ndarray, dimension: int) -> np.ndarray:
    matrix = np.asarray(value)
    if matrix.ndim != 2 or matrix.shape[1] != dimension or len(matrix) == 0:
        raise ValueError(f"Expected non-empty frames x {dimension} features")
    if not np.issubdtype(matrix.dtype, np.floating) or not np.isfinite(matrix).all():
        raise ValueError("Features must contain finite floating-point values")
    return matrix


def _validate_rows(rows: Sequence[ManifestRow], *, train: bool, allow_eval: bool = False):
    if not rows:
        raise ValueError("Manifest is empty")
    splits = {row.split for row in rows}
    if train and splits != {"train"}:
        raise ValueError("Training accepts only train rows")
    if not train and "eval" in splits and not allow_eval:
        raise ValueError("Eval scoring requires allow_eval=True")
    validate_splits({split: [row for row in rows if row.split == split] for split in splits})
    if train and {row.label for row in rows} != {"bonafide", "spoof"}:
        raise ValueError("Training requires both bonafide and spoof records")


def train_feature_stream(
    rows: Sequence[ManifestRow], feature_loader: Callable[[ManifestRow], np.ndarray], *,
    lfcc_config: LFCCConfig = LFCCConfig(), gmm_config: GMMConfig = GMMConfig(),
) -> BaselineModel:
    """Train from one recording at a time, retaining <= frame budget per class.

    Uniform random frame priorities implement a reservoir over all class frames;
    longer files contribute proportionally more frames. Sorted utterance IDs and
    fixed class RNGs make the selection independent of manifest input order.
    No standardization or CMVN is fitted or applied.
    """
    _validate_rows(rows, train=True)
    started = perf_counter()
    dimension = lfcc_config.feature_dim
    rngs = {label: np.random.default_rng(gmm_config.seed + offset)
            for label, offset in (("bonafide", 0), ("spoof", 1))}
    reservoirs = {label: np.empty((0, dimension), dtype=np.float32) for label in rngs}
    priorities = {label: np.empty(0) for label in rngs}
    seen = dict.fromkeys(rngs, 0)
    for row in sorted(rows, key=lambda row: row.utterance_id):
        frames = _features(feature_loader(row), dimension)
        label = row.label
        seen[label] += len(frames)
        candidates = np.concatenate((reservoirs[label], frames.astype(np.float32)))
        keys = np.concatenate((priorities[label], rngs[label].random(len(frames))))
        if len(keys) > gmm_config.frames_per_class:
            selected = np.argpartition(keys, gmm_config.frames_per_class - 1)[:gmm_config.frames_per_class]
            # Stable priority ordering also fixes the GMM input order.
            selected = selected[np.argsort(keys[selected], kind="stable")]
            candidates, keys = candidates[selected], keys[selected]
        reservoirs[label], priorities[label] = candidates, keys
    extraction_seconds = perf_counter() - started
    models, class_metadata = {}, {}
    for label in rngs:
        frames = reservoirs[label]
        if len(frames) < max(2, gmm_config.n_components):
            raise ValueError(f"{label} has {len(frames)} frames; need >= max(2, n_components)")
        model = GaussianMixture(
            n_components=gmm_config.n_components, covariance_type="diag",
            random_state=gmm_config.seed, reg_covar=gmm_config.reg_covar,
            tol=gmm_config.tol, max_iter=gmm_config.max_iter, n_init=1, init_params="kmeans",
        )
        fit_started = perf_counter()
        with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
            warnings.simplefilter("always")
            model.fit(frames)
        with threadpool_limits(limits=1):
            if not np.isfinite(model.score_samples(frames)).all():
                raise ValueError(f"Non-finite fitted likelihoods for {label}")
        models[label] = model
        class_metadata[label] = {
            "frames_seen": seen[label], "frames_sampled": len(frames),
            "iterations": int(model.n_iter_), "converged": bool(model.converged_),
            "lower_bound": float(model.lower_bound_),
            "fit_seconds": perf_counter() - fit_started,
            "warnings": [str(item.message) for item in caught],
        }
    metadata = {
        "format_version": 1, "implementation": "sklearn-diagonal-lfcc-gmm-pilot",
        "lfcc_config": asdict(lfcc_config), "gmm_config": asdict(gmm_config),
        "normalization": "none", "covariance_type": "diag", "n_init": 1,
        "init_params": "kmeans", "fit_threads": 1,
        "frame_sampling": "uniform random priority per class, sorted utterance IDs",
        "training_ids": sorted(row.utterance_id for row in rows),
        "classes": class_metadata, "extraction_sampling_seconds": extraction_seconds,
        "total_seconds": perf_counter() - started,
        "versions": {package: version(package) for package in
                     ("numpy", "scipy", "scikit-learn", "joblib", "soundfile")},
        "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in ("baseline.py", "features.py")},
    }
    return BaselineModel(models["bonafide"], models["spoof"], lfcc_config, gmm_config, metadata)


def _audio_loader(data_root: str | Path, config: LFCCConfig):
    root = Path(data_root).resolve()
    def load(row: ManifestRow):
        missing = check_audio_files([row], root)
        if missing:
            raise FileNotFoundError(missing[0])
        return extract_lfcc(load_audio(root / row.audio_path, sample_rate=config.sample_rate), config)
    return load


def train_baseline(rows: Sequence[ManifestRow], data_root: str | Path, *,
                   lfcc_config: LFCCConfig = LFCCConfig(),
                   gmm_config: GMMConfig = GMMConfig()) -> BaselineModel:
    """Train from caller-verified local audio; source/hash journal checks belong to caller."""
    return train_feature_stream(rows, _audio_loader(data_root, lfcc_config),
                                lfcc_config=lfcc_config, gmm_config=gmm_config)


def score_features(model: BaselineModel, features: np.ndarray) -> float:
    """Mean frame log p(bonafide) - log p(spoof); higher is more bonafide."""
    frames = _features(features, model.lfcc_config.feature_dim)
    with threadpool_limits(limits=1):
        difference = model.bonafide.score_samples(frames) - model.spoof.score_samples(frames)
    if not np.isfinite(difference).all():
        raise ValueError("Model produced non-finite likelihood ratios")
    score = float(np.mean(difference))
    if not np.isfinite(score):
        raise ValueError("Model produced a non-finite recording score")
    return score


def score_rows(model: BaselineModel, rows: Sequence[ManifestRow], data_root: str | Path,
               *, allow_eval: bool = False) -> list[tuple[str, float]]:
    """Score train/dev records, or explicitly authorized eval, in input order."""
    _validate_rows(rows, train=False, allow_eval=allow_eval)
    loader = _audio_loader(data_root, model.lfcc_config)
    return [(row.utterance_id, score_features(model, loader(row))) for row in rows]


def save_model(model: BaselineModel, path: str | Path) -> None:
    """Write a local joblib artifact. Only load files you created and trust."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination)


def load_model(path: str | Path) -> BaselineModel:
    """Load a TRUSTED local artifact; joblib can execute code before validation."""
    model = joblib.load(path)
    if not isinstance(model, BaselineModel) or model.metadata.get("format_version") != 1:
        raise ValueError("Unsupported baseline artifact")
    installed = {package: version(package) for package in model.metadata["versions"]}
    if installed != model.metadata["versions"]:
        warnings.warn("Model package versions differ from the training environment", RuntimeWarning)
    return model
