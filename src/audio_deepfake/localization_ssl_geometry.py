"""Map unpadded wav2vec 2.0 frames to the existing clipped 20 ms grid.

This utility loads no model and reads no labels. Transformer states have
whole-recording context; convolution centers are alignment anchors only.
"""

from numbers import Integral

import numpy as np


def convolution_geometry(kernels, strides):
    if len(kernels) != len(strides) or not kernels:
        raise ValueError("Require nonempty matching convolution lists")
    receptive, hop = 1, 1
    for kernel, stride in zip(kernels, strides):
        if any(isinstance(x, bool) or not isinstance(x, Integral) or x < 1
               for x in (kernel, stride)):
            raise ValueError("Convolution dimensions must be positive integers")
        receptive += (int(kernel) - 1) * hop
        hop *= int(stride)
    return receptive, hop


def frame_mapping(sample_count, *, receptive_samples=400, hop_samples=320):
    """Return nearest frame indices, clipping only the last label bin.

    No waveform padding, cropping or interpolation. Inputs shorter than one
    receptive field are rejected. Exact center ties select the earlier frame.
    """
    for value in (sample_count, receptive_samples, hop_samples):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError("Sample counts must be positive integers")
    if sample_count < receptive_samples:
        raise ValueError("Recording is shorter than one convolution frame")
    count = 1 + (sample_count - receptive_samples) // hop_samples
    bins = (sample_count + 319) // 320
    if max(count, bins) > 1_000_000:
        raise ValueError("Frame geometry exceeds the utility limit")
    starts = np.arange(bins, dtype=np.int64) * 320
    ends = np.minimum(starts + 320, sample_count)
    centers = np.arange(count, dtype=np.int64) * (2 * hop_samples) + receptive_samples
    targets = starts + ends
    right = np.minimum(np.searchsorted(centers, targets), count - 1)
    left = np.maximum(right - 1, 0)
    indices = np.where(abs(targets - centers[left]) <= abs(targets - centers[right]), left, right)
    return {"source_frame_count": count, "frame_indices": indices,
            "bin_starts_samples": starts, "bin_ends_samples": ends}
