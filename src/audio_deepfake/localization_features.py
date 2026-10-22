"""Offline LFCC features on the project's explicit 20 ms localization grid.

This grid is not an assertion about official PartialSpoof annotation lengths.
For each output bin, choose the nearest 20 ms LFCC window center; exact ties
choose the earlier frame. No interpolation, pooling or implicit resampling.
"""

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from .features import LFCCConfig, extract_lfcc


@dataclass(frozen=True)
class LocalizationFeatures:
    """Aligned bin boundaries in seconds, features, and source frame indices."""

    bin_starts: np.ndarray
    bin_ends: np.ndarray
    features: np.ndarray
    frame_indices: np.ndarray


def _nearest_frame_indices(bin_centers_twice, frame_centers_twice):
    """Map integer doubled-sample centers; ties choose the earlier frame."""
    right = np.searchsorted(frame_centers_twice, bin_centers_twice, side="left")
    right = np.minimum(right, len(frame_centers_twice) - 1)
    left = np.maximum(right - 1, 0)
    left_distance = np.abs(bin_centers_twice - frame_centers_twice[left])
    right_distance = np.abs(frame_centers_twice[right] - bin_centers_twice)
    return np.where(left_distance <= right_distance, left, right)


def extract_localization_features(waveform, *, sample_rate,
                                  config: LFCCConfig = LFCCConfig()):
    """Return one LFCC vector per 20 ms bin, with a clipped final bin.

    Require explicit 16 kHz mono floating-point input and 20 ms window/10 ms
    hop LFCC geometry. Default LFCC configuration gives 60 dimensions. Actual
    extraction count is checked against the frozen extractor's padding rule.
    Frame centers use window midpoints, including right-zero-padded frames;
    they may extend past the audio end for very short recordings. Bin centers
    use their actual clipped width. Nearest-frame ties choose the earlier one.

    This is offline: delta/double-delta features use future neighbors (up to
    two hops for default double deltas), and the selected window can overlap
    adjacent bins. Pre-emphasis also uses the preceding waveform sample. A
    label's 20 ms width therefore does not imply 20 ms acoustic context or
    streaming latency. No data labels, training or detector scores are used.
    """
    if not isinstance(config, LFCCConfig):
        raise ValueError("config must be an LFCCConfig")
    if (isinstance(sample_rate, (bool, np.bool_))
            or not isinstance(sample_rate, Integral) or sample_rate != 16_000
            or config.sample_rate != sample_rate):
        raise ValueError("Explicit input and LFCC sample rates must both be 16000")
    if config.window_samples != 320 or config.hop_samples != 160:
        raise ValueError("Localization requires 20 ms windows and 10 ms hops")
    samples = np.asarray(waveform)
    if (samples.ndim != 1 or not samples.size
            or not np.issubdtype(samples.dtype, np.floating)
            or not np.all(np.isfinite(samples))):
        raise ValueError("Waveform must contain non-empty finite mono floating-point samples")
    sample_count = len(samples)
    bin_count = (sample_count + 319) // 320
    if bin_count > 1_000_000:
        raise ValueError("Timeline exceeds the one-million-bin utility limit")
    features = extract_lfcc(samples, config)
    expected_frames = 1 + max(0, (sample_count - 320 + 159) // 160)
    if (features.shape != (expected_frames, config.feature_dim)
            or not np.all(np.isfinite(features))):
        raise ValueError("LFCC output does not match the expected finite frame geometry")

    starts = np.arange(bin_count, dtype=np.int64) * 320
    ends = np.minimum(starts + 320, sample_count)
    # Twice the sample coordinate represents half-sample centers exactly.
    bin_centers_twice = starts + ends
    frame_centers_twice = np.arange(expected_frames, dtype=np.int64) * 320 + 320
    indices = _nearest_frame_indices(bin_centers_twice, frame_centers_twice)
    return LocalizationFeatures(
        bin_starts=starts.astype(np.float64) / sample_rate,
        bin_ends=ends.astype(np.float64) / sample_rate,
        features=features[indices].copy(),
        frame_indices=indices,
    )
