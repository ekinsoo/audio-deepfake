"""Small, explicit CPU LFCC adaptation; see docs/LFCC.md for conventions."""

from dataclasses import dataclass

import numpy as np
from scipy.fft import dct, rfft
from scipy.signal import lfilter


@dataclass(frozen=True)
class LFCCConfig:
    sample_rate: int = 16_000
    window_ms: float = 20.0
    hop_ms: float = 10.0
    n_fft: int = 512
    n_filters: int = 20
    n_coeffs: int = 20
    pre_emphasis: float = 0.97
    delta_order: int = 2

    def __post_init__(self):
        integers = (self.sample_rate, self.n_fft, self.n_filters,
                    self.n_coeffs, self.delta_order)
        if any(isinstance(v, bool) or not isinstance(v, int) for v in integers):
            raise ValueError("Sample rate, FFT, filter, coefficient and delta counts must be integers")
        if self.sample_rate <= 0 or self.n_fft < 2 or self.n_fft % 2:
            raise ValueError("Sample rate must be positive and n_fft must be positive and even")
        if not 1 <= self.n_coeffs <= self.n_filters:
            raise ValueError("Require 1 <= n_coeffs <= n_filters")
        if self.delta_order not in (0, 1, 2):
            raise ValueError("delta_order must be 0, 1 or 2")
        if not np.isfinite(self.pre_emphasis) or not 0 <= self.pre_emphasis < 1:
            raise ValueError("pre_emphasis must be finite and in [0, 1)")
        for name in ("window_ms", "hop_ms"):
            value = getattr(self, name) * self.sample_rate / 1000
            if not np.isfinite(value) or value < 1 or not np.isclose(value, round(value)):
                raise ValueError(f"{name} must represent a positive whole number of samples")
        if self.hop_samples > self.window_samples or self.window_samples > self.n_fft:
            raise ValueError("Require hop_samples <= window_samples <= n_fft")
        # A triangle with no FFT-bin support makes log energies uninformative.
        if np.any(linear_filterbank(self).sum(axis=1) == 0):
            raise ValueError("Too many filters for n_fft; a filter has no FFT-bin support")

    @property
    def window_samples(self) -> int:
        return round(self.window_ms * self.sample_rate / 1000)

    @property
    def hop_samples(self) -> int:
        return round(self.hop_ms * self.sample_rate / 1000)

    @property
    def feature_dim(self) -> int:
        return self.n_coeffs * (1 + self.delta_order)


def linear_filterbank(config: LFCCConfig) -> np.ndarray:
    """Return unnormalized triangular weights, filters x one-sided FFT bins."""
    frequencies = np.linspace(0, config.sample_rate / 2, config.n_fft // 2 + 1)
    edges = np.linspace(0, config.sample_rate / 2, config.n_filters + 2)
    left = (frequencies[None, :] - edges[:-2, None]) / (edges[1:-1] - edges[:-2])[:, None]
    right = (edges[2:, None] - frequencies[None, :]) / (edges[2:] - edges[1:-1])[:, None]
    return np.maximum(0, np.minimum(left, right))


def temporal_delta(features: np.ndarray) -> np.ndarray:
    """Unnormalized three-frame slope with endpoint replication (2019 code)."""
    if features.ndim != 2 or len(features) == 0:
        raise ValueError("Delta input must be non-empty frames x coefficients")
    padded = np.pad(features, ((1, 1), (0, 0)), mode="edge")
    return padded[2:] - padded[:-2]


def extract_lfcc(waveform: np.ndarray, config: LFCCConfig = LFCCConfig()) -> np.ndarray:
    """Extract frames x coefficients; caller must supply mono at config.sample_rate.

    A short non-empty clip produces one right-zero-padded frame. A final partial
    frame is padded, without adding a redundant frame after a full final frame.
    No resampling, waveform normalization, VAD, CMVN or duration cap is applied.
    """
    samples = np.asarray(waveform)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("Waveform must be non-empty mono (one-dimensional)")
    if not np.issubdtype(samples.dtype, np.floating) or not np.all(np.isfinite(samples)):
        raise ValueError("Waveform must contain finite floating-point samples")
    samples = lfilter([1.0, -config.pre_emphasis], [1.0], samples.astype(np.float64))
    window, hop = config.window_samples, config.hop_samples
    frame_count = 1 + max(0, (len(samples) - window + hop - 1) // hop)
    required = (frame_count - 1) * hop + window
    samples = np.pad(samples, (0, required - len(samples)))
    frames = np.lib.stride_tricks.sliding_window_view(samples, window)[::hop]
    spectrum = rfft(frames * np.hamming(window), n=config.n_fft, axis=1)
    energies = (spectrum.real ** 2 + spectrum.imag ** 2) @ linear_filterbank(config).T
    static = dct(np.log10(energies + np.finfo(np.float64).eps), type=2, norm="ortho", axis=1)
    static = static[:, :config.n_coeffs]
    parts = [static]
    for _ in range(config.delta_order):
        parts.append(temporal_delta(parts[-1]))
    return np.concatenate(parts, axis=1).astype(np.float32)
