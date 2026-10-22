"""Explicit timeline utilities for a future localization pilot.

These are project conventions, not an implementation of PartialSpoof's stored
labels or official RangeEER. Intervals are half-open seconds [start, end).
Boolean masks use True for spoof; the native-label adapter instead accepts
only numeric 0=spoof and 1=bona fide. No LFCC frame alignment is assumed.
"""

from fractions import Fraction
import math
from numbers import Real

import numpy as np


def _positive_seconds(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive finite number of seconds")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number of seconds")
    return value


def time_grid(*, duration, step):
    """Return Nx2 half-open bins, clipping the last bin to duration.

    Count is ceil(duration/step), using decimal representations of supplied
    floats to avoid division artifacts (e.g. 0.14/0.02). This is a project
    grid; official annotation lengths must be checked before adaptation.
    """
    duration = _positive_seconds(duration, "duration")
    step = _positive_seconds(step, "step")
    length = Fraction(str(duration)) / Fraction(str(step))
    count = math.ceil(length)
    if count > 1_000_000:
        raise ValueError("Timeline exceeds the one-million-bin utility limit")
    decimal_step = Fraction(str(step))
    boundaries = np.array([float(i * decimal_step) for i in range(count + 1)])
    boundaries[-1] = duration
    if np.any(np.diff(boundaries) <= 0):
        raise ValueError("Timeline bins are not representable at this precision")
    return np.column_stack((boundaries[:-1], boundaries[1:]))


def validate_intervals(intervals, *, duration):
    """Validate ordered, non-overlapping ranges without sorting or clipping."""
    duration = _positive_seconds(duration, "duration")
    result = []
    previous_end = 0.0
    for interval in intervals:
        try:
            start, end = interval
        except (TypeError, ValueError) as exc:
            raise ValueError("Each interval must contain exactly start and end") from exc
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real)
               for v in (start, end)):
            raise ValueError("Interval endpoints must be finite numbers")
        start, end = float(start), float(end)
        if not math.isfinite(start) or not math.isfinite(end):
            raise ValueError("Interval endpoints must be finite numbers")
        if not 0 <= start < end <= duration:
            raise ValueError("Require 0 <= start < end <= duration")
        if start < previous_end:
            raise ValueError("Intervals must be ordered and non-overlapping")
        result.append((start, end))
        previous_end = end
    return tuple(result)


def spoof_overlap_fraction(intervals, *, duration, step):
    """Return covered fraction of each bin, including a shortened final bin."""
    grid = time_grid(duration=duration, step=step)
    ranges = validate_intervals(intervals, duration=duration)
    covered = np.zeros(len(grid), dtype=np.float64)
    for start, end in ranges:
        covered += np.maximum(0, np.minimum(grid[:, 1], end)
                              - np.maximum(grid[:, 0], start))
    return np.clip(covered / (grid[:, 1] - grid[:, 0]), 0, 1)


def intervals_to_spoof_mask(intervals, *, duration, step, policy):
    """Rasterize with an explicit policy; majority ties are bona fide.

    Policies: any_overlap, majority (>50% bin coverage), midpoint (point lies
    in a spoof interval). The policy is required because they differ at edges.
    """
    ranges = validate_intervals(intervals, duration=duration)
    if policy == "any_overlap":
        return spoof_overlap_fraction(ranges, duration=duration, step=step) > 0
    if policy == "majority":
        fractions = spoof_overlap_fraction(ranges, duration=duration, step=step)
        mask = fractions > 0.5
        # A decimal half-bin boundary can round just above 0.5 in binary.
        # Resolve only close cases exactly, rather than changing the threshold.
        grid = time_grid(duration=duration, step=step)
        decimal_ranges = [(Fraction(str(a)), Fraction(str(b))) for a, b in ranges]
        for index in np.flatnonzero(np.abs(fractions - .5) < 1e-12):
            start, end = (Fraction(str(v)) for v in grid[index])
            overlap = sum((max(Fraction(0), min(end, b) - max(start, a))
                           for a, b in decimal_ranges), Fraction(0))
            mask[index] = 2 * overlap > end - start
        return mask
    if policy == "midpoint":
        grid = time_grid(duration=duration, step=step)
        centers = grid.mean(axis=1)
        mask = np.zeros(len(grid), dtype=bool)
        for start, end in ranges:
            mask |= (centers >= start) & (centers < end)
        return mask
    raise ValueError("policy must be any_overlap, majority or midpoint")


def _aligned_vector(values, *, duration, step):
    vector = np.asarray(values)
    grid = time_grid(duration=duration, step=step)
    if vector.ndim != 1 or len(vector) != len(grid):
        raise ValueError("Vector must be 1D with exactly ceil(duration/step) bins")
    return vector, grid


def native_labels_to_spoof_mask(labels, *, duration, step):
    """Convert numeric 0=spoof/1=bona fide; reject VAD=2 or unknown codes.

    Does not read a dataset file. Callers must establish annotation step,
    duration and final-frame convention from the actual distribution first.
    """
    labels, _ = _aligned_vector(labels, duration=duration, step=step)
    if labels.dtype.kind not in "iuf" or not np.all(np.isin(labels, [0, 1])):
        raise ValueError("Native labels must be numeric 0=spoof or 1=bona fide")
    return labels == 0


def spoof_mask_to_intervals(mask, *, duration, step):
    """Merge consecutive True bins into half-open spoof ranges."""
    mask, grid = _aligned_vector(mask, duration=duration, step=step)
    if mask.dtype.kind != "b":
        raise ValueError("Spoof mask must contain booleans, with True=spoof")
    transitions = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts = np.flatnonzero(transitions == 1)
    stops = np.flatnonzero(transitions == -1)
    return tuple((float(grid[start, 0]), float(grid[stop - 1, 1]))
                 for start, stop in zip(starts, stops))


def scores_to_spoof_intervals(scores, *, threshold, duration, step):
    """Threshold bin scores where higher supports bona fide; equality is BF.

    Scores must already be aligned to this grid. This function neither trains
    a detector nor converts overlapping feature windows to annotation bins.
    This tie rule is a project convention: an official evaluator using
    score <= threshold for spoof requires a separate, explicit adapter.
    """
    scores, _ = _aligned_vector(scores, duration=duration, step=step)
    if scores.dtype.kind not in "iuf" or not np.all(np.isfinite(scores)):
        raise ValueError("Scores must contain finite numbers")
    if (isinstance(threshold, (bool, np.bool_))
            or not isinstance(threshold, Real) or not math.isfinite(threshold)):
        raise ValueError("Threshold must be finite")
    return spoof_mask_to_intervals(scores < threshold, duration=duration, step=step)
