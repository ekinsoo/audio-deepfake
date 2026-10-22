# LFCC conventions

This is a small NumPy/SciPy adaptation for the CPU pilot, not a verified exact
reproduction of the official MATLAB system. It does not train a model.

## Source checked

We inspected the official `ASVspoof_2019_baseline_CM_v1.zip` already stored in
ignored `data/metadata/asvspoof2019/`, specifically `ASVspoof2019_baseline_CM.m`,
`LFCC/extract_lfcc.m` and `LFCC/deltas.m`. The download entry is on the
[official 2019 page](https://www.asvspoof.org/index2019.html).

The MATLAB driver calls `extract_lfcc(x, fs, 20, 512, 20)` and concatenates static,
delta and double-delta features. The feature function uses pre-emphasis 0.97,
20 ms frames with half-frame overlap, symmetric Hamming windows, a 512-point FFT,
20 linear triangular filters from 0 to Nyquist, raw squared FFT magnitude,
`log10(energy + eps)` and MATLAB's orthonormal DCT-II. It keeps all 20 coefficients,
including c0. There is no active dither, CMVN, VAD or signal scaling.

One easily missed detail: the delta function's window argument is **3**, and it
does **not** normalize the slope. Delta is `x[t+1] - x[t-1]`, with repeated endpoints.
Double-delta applies the same operation to delta. A usual regression delta
divided by two would differ from this code.

## Our explicit settings

`extract_lfcc(waveform, config=LFCCConfig())` accepts non-empty finite floating-point
mono audio and returns float32 `[frames, 60]` by default. The caller must verify
16 kHz input. The function cannot infer a sample rate from a bare array and does
not resample. `LFCCConfig` records every configurable processing parameter.

Defaults: 16,000 Hz; 320-sample window; 160-sample hop; FFT 512; 20 filters;
20 coefficients; pre-emphasis 0.97; delta order 2. Filters have unit triangle
height and no area normalization. Calculations use float64, with float64 epsilon
added before log10; only the final feature matrix is converted to float32.

We start frames at sample zero. Short non-empty audio is zero-padded to one frame.
We pad the final partial frame on the right, but do not append another frame if the
last full frame ends exactly at the end of the recording. For N samples the count
is `1 + max(0, ceil((N - 320) / 160))`. We do not trim, cap duration, remove silence,
normalize amplitude or mutate the caller's waveform. The original audio stays intact.

## Limits and verification

The default settings follow the inspected baseline, including its unusual delta
scaling. Our framing policy is explicit and intended to match MATLAB
`buffer(..., 'nodelay')`, but MATLAB output has not been run against this code.
FFT/DCT implementations and the float32 output can also introduce numerical
differences. Do not label results an exact baseline reproduction until a shared
audio fixture has been compared with MATLAB. GMM training and scoring will need
their own settings and equivalence checks.

Unit tests cover silence, triangle partition and frequency selectivity, frame
boundaries, delta slope/edges, amplitude-gain behavior, deterministic output,
input preservation and invalid configurations. These are signal-processing
checks; they do not prove benchmark accuracy or MATLAB parity.
