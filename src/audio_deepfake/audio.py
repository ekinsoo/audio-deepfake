"""Strict audio loading for the initial 16 kHz mono experiments."""

from pathlib import Path

import numpy as np
import soundfile as sf
from numpy.typing import NDArray


def load_audio(path: str | Path, *, sample_rate: int = 16_000) -> NDArray[np.float32]:
    """Load one recording without silently resampling or mixing channels.

    Keeping input changes explicit matters when comparing clean and stressed audio.
    """
    with sf.SoundFile(path) as audio:
        if audio.samplerate != sample_rate:
            raise ValueError(f"Expected {sample_rate} Hz, got {audio.samplerate} Hz")
        if audio.channels != 1:
            raise ValueError(f"Expected mono audio, got {audio.channels} channels")
        if audio.frames == 0:
            raise ValueError("Audio is empty")
        waveform = audio.read(dtype="float32")
    if not np.isfinite(waveform).all():
        raise ValueError("Audio contains non-finite samples")
    return waveform
