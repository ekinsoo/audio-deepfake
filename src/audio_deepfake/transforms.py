"""The fixed bandwidth stress test; see docs/ROBUSTNESS_PLAN.md."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import resample_poly


# Recorded verbatim with an experiment. This describes one frozen transform.
TRANSFORM_SPEC = {
    "name": "narrowband-roundtrip-v1",
    "input_sample_rate": 16000,
    "intermediate_sample_rate": 8000,
    "output_sample_rate": 16000,
    "method": "scipy.signal.resample_poly",
    "down_up": 1,
    "down_down": 2,
    "restore_up": 2,
    "restore_down": 1,
    "window": ["kaiser", 5.0],
    "padtype": "constant",
    "cval": 0.0,
    "processing_dtype": "float64",
    "output_dtype": "float32",
    "length_policy": "trim-to-original-sample-count",
    "normalization": "none",
    "clipping": False,
    "codec": "none",
}


def narrowband_roundtrip(waveform: ArrayLike, sample_rate: int = 16000) -> NDArray[np.float32]:
    """Return a 16→8→16 kHz waveform without changing gain or the input array.

    This approximates reduced bandwidth, not a telephone codec. The input must
    already be finite, floating-point mono audio at 16 kHz. Output has exactly
    the original length. Values outside [-1, 1] are retained rather than clipped.
    """
    if isinstance(sample_rate, (bool, np.bool_)) or not isinstance(sample_rate, (int, np.integer)) or sample_rate != 16000:
        raise ValueError("Expected integer sample_rate=16000")
    samples = np.asarray(waveform)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("Expected nonempty mono waveform")
    if not np.issubdtype(samples.dtype, np.floating):
        raise ValueError("Expected floating-point waveform")
    if not np.isfinite(samples).all():
        raise ValueError("Waveform contains non-finite samples")
    # A copy also protects read-only arrays and views from accidental mutation.
    original = samples.astype(np.float64, copy=True)
    down = resample_poly(original, 1, 2, window=("kaiser", 5.0), padtype="constant", cval=0)
    restored = resample_poly(down, 2, 1, window=("kaiser", 5.0), padtype="constant", cval=0)
    with np.errstate(over="ignore", invalid="ignore"):
        result = restored[:samples.size].astype(np.float32)
    if result.size != samples.size or not np.isfinite(result).all():
        raise ValueError("Transformation cannot produce finite float32 audio")
    return result
