"""Explicit adapters for checked PartialSpoof v1.2 metadata.

File acquisition and publisher checksum verification belong to preparation.
These functions never fetch data or load pickle. Native binary arrays may
contain the literal strings '0'/'1'; VAD's third state is kept separate.
"""

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from .localization import validate_intervals


def binary_labels(values):
    """Convert only published binary codes; never coerce nonspeech to spoof."""
    labels = np.asarray(values)
    if labels.ndim != 1 or not len(labels):
        raise ValueError("Expected nonempty native binary label vector")
    if labels.dtype.kind in "US":
        # Bytes are decoded explicitly; no permissive int('1.0') coercion.
        strings = labels.astype("U")
        if not np.isin(strings, ["0", "1"]).all():
            raise ValueError("Expected literal native labels '0' or '1'")
        return strings.astype(np.int8)
    if labels.dtype.kind not in "iuf" or not np.isin(labels, [0, 1]).all():
        raise ValueError("Expected numeric native labels 0 or 1")
    return labels.astype(np.int8)


def coarse_binary_labels(values, *, factor):
    """Published minimum-over-groups rule, keeping a final partial group."""
    if isinstance(factor, (bool, np.bool_)) or not isinstance(factor, Integral) or factor < 1:
        raise ValueError("Coarsening factor must be a positive integer")
    labels = binary_labels(values)
    return np.minimum.reduceat(labels, np.arange(0, len(labels), factor))


def labels_for_audio(values, *, sample_count):
    """Require ceil(N/160) native 10 ms labels; return 20 ms min-pooled labels.

    This is an explicit candidate alignment contract, to be verified against
    selected real audio. A mismatch raises; labels are never trimmed or padded.
    """
    if (isinstance(sample_count, (bool, np.bool_))
            or not isinstance(sample_count, Integral) or sample_count < 1):
        raise ValueError("Require a positive actual 16 kHz sample count")
    labels = binary_labels(values)
    expected = (sample_count + 159) // 160
    if len(labels) != expected:
        raise ValueError(f"Native 10 ms label count {len(labels)} differs from ceil(N/160)={expected}")
    result = coarse_binary_labels(labels, factor=2)
    if len(result) != (sample_count + 319) // 320:
        raise ValueError("Coarse labels differ from the feature output grid")
    return result


@dataclass(frozen=True)
class VadSegment:
    start: float
    end: float
    native_label: int


def parse_vad(text, *, duration):
    """Read start/end seconds and 0/1/2 state, requiring complete coverage.

    Duration must come from an independently checked audio header. Label 2
    remains nonspeech. A 1 ns tolerance accommodates printed decimal seconds;
    it does not allow dropping an unlabelled tail or filling a gap.
    """
    if not isinstance(text, str):
        raise ValueError("VAD input must be text")
    segments = []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 3 or fields[2] not in ("0", "1", "2"):
            raise ValueError("Expected VAD start end label with label 0/1/2")
        try:
            start, end = map(float, fields[:2])
        except ValueError as exc:
            raise ValueError("Invalid VAD time") from exc
        segments.append(VadSegment(start, end, int(fields[2])))
    if not segments:
        raise ValueError("VAD file is empty")
    validate_intervals([(s.start, s.end) for s in segments], duration=duration)
    previous = 0.0
    for segment in segments:
        if abs(segment.start - previous) > 1e-9:
            raise ValueError("VAD does not cover audio contiguously")
        previous = segment.end
    if abs(previous - duration) > 1e-9:
        raise ValueError("VAD duration differs from actual audio duration")
    return tuple(segments)


def vad_spoof_intervals(segments):
    """Keep VAD=0 for annotation diagnostics, not binary ground truth.

    Real v1.2 metadata shows that nonspeech (VAD=2) may be native binary 0 or
    1. This function cannot produce the official binary RangeEER reference.
    """
    return tuple((s.start, s.end) for s in segments if s.native_label == 0)
