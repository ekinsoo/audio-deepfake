"""Fixed offline per-recording robust context transform for localization.

This deliberately uses the whole unlabelled recording, including future bins.
It is unsuitable for a streaming latency claim. No labels enter this API.
"""

import numpy as np

MAD_SCALE = 1.4826
SCALE_FLOOR = 1e-3


def recording_context(features):
    """Normalize every LFCC/static/delta/double-delta dimension independently.

    Input is the complete aligned 20 ms feature matrix for one recording.
    Across all bins, center_j = median(x[:, j]), MAD_j =
    median(abs(x[:, j] - center_j)); output = (x - center) /
    maximum(1.4826 * MAD, 0.001). Partial final bins have equal weight.
    Computation and output use float64. No clipping or parameter estimation
    across recordings is performed. Train-only StandardScaler follows this.
    """
    values = np.asarray(features)
    if (values.ndim != 2 or not len(values) or values.shape[1] != 60
            or values.dtype.kind != "f" or not np.isfinite(values).all()):
        raise ValueError("Require one complete finite floating-point bins x 60 matrix")
    values = values.astype(np.float64)
    center = np.median(values, axis=0)
    deviation = np.median(np.abs(values - center), axis=0)
    scale = np.maximum(MAD_SCALE * deviation, SCALE_FLOOR)
    result = (values - center) / scale
    if not np.isfinite(result).all():
        raise ValueError("Context transform exceeds finite floating-point range")
    return result
