# Audio Deepfake Detection

Experiments in detecting synthetic speech and locating manipulated segments,
by Özgür Ekin Sönmez.

I started with an LFCC-GMM detector and tested whether a bandwidth change
could help identify its mistakes. It did not: distance from the decision
threshold was a better error signal. I then compared LFCC and frozen
wav2vec2 features for partial-spoof localization.

[Read the report](docs/report/audio_deepfake_report.pdf) ·
[Detection results](docs/RESULTS.md) ·
[Localization results](docs/LOCALIZATION_SSL_RESULTS.md)

## Results

On ASVspoof 2019 LA development audio, across three fixed seeds:

| Scoring condition | Mean EER ↓ | Mean min. normalized t-DCF ↓ |
| --- | ---: | ---: |
| Original audio | **6.34%** | **0.1724** |
| Narrowband audio (16 → 8 → 16 kHz) | 22.17% | 0.5298 |
| Fixed average of both scores | 9.03% | 0.2012 |

The models used all 25,380 official train recordings, with a cap of 50,000
training frames per class, and scored 24,844 dev recordings. Absolute score
change ranked clean prediction errors poorly (AUROC 0.385); decision margin
reached 0.936 on the same speaker-separated analysis partition.

![Detection results](docs/figures/detection_comparison.svg)

For localization, I used a fixed PartialSpoof v1.2 subset of 128 train and
64 dev recordings:

| Model | Custom duration-weighted 20 ms EER ↓ |
| --- | ---: |
| LFCC + logistic regression | 39.50% |
| LFCC with recording-context normalization | 42.92% |
| Frozen wav2vec2 + logistic regression, convergence follow-up | **28.57%** |

The original 300-iteration SSL fit did not converge. A separately recorded
follow-up allowed 3,000 iterations and converged at 303, before dev scoring.
This differs from the LFCC optimization budget. At the fixed zero threshold,
SSL reduced false-alarm duration but increased missed spoof duration.
[Protocol and limitations](docs/LOCALIZATION_SSL_CONVERGENCE_PLAN.md).

All figures above are **development results**. Eval remains closed. The
localization subset has been inspected repeatedly, the encoder uses the whole
recording, and pretraining overlap is unknown. These are neither official
localization benchmark results nor evidence of production accuracy.

## Install

Python 3.11+; experiments were run on Windows with Python 3.12.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -e .
python -m audio_deepfake doctor
python -m audio_deepfake smoke
python scripts/run_tests.py
```

The smoke check uses a generated tone; it downloads no dataset and trains no
model. The default tests use synthetic or temporary fixtures. External
RangeEER conformance and the Windows process-monitor probe are separate
integration checks; see [testing](docs/DEVELOPMENT.md#testing).

The recorded experiment environment is in
[requirements/windows-py312.txt](requirements/windows-py312.txt). SSL extraction,
reporting and external metrics use separate environments.

## Local demo

The demo scores a mono 16 kHz WAV/FLAC file, up to 60 seconds and 8 MiB:

```bash
python -m audio_deepfake.demo score path/to/recording.wav
python -m audio_deepfake.demo serve
```

The browser interface runs at `http://127.0.0.1:8765`. It shows an uncalibrated
score and processing time; higher scores support bona fide speech. Files stay
on the local machine.

**The audited model and dataset manifests are not included in this repository.**
The demo requires the matching seed-43 artifacts from a reproduced run; a
fresh checkout supports the smoke check and unit tests without them.
See [the full-run guide](docs/FULL_RUN.md) and
[demo details](docs/RESULTS.md#local-demo-and-cpu-measurement).

## Reproducing the experiments

- [Data sources and preparation](docs/DATA_PROTOCOL.md)
- [Full LFCC-GMM training](docs/FULL_RUN.md)
- [Bandwidth experiment](docs/FULL_ROBUSTNESS_RUN.md)
- [PartialSpoof labels and protocol](docs/PARTIALSPOOF_PROTOCOL.md)
- [Localization training](docs/LOCALIZATION_TRAINING.md)
- [Frozen SSL experiment](docs/LOCALIZATION_SSL_PLAN.md)
- [Report sources and build](docs/report/README.md)

Code lives in `src/`, experiment entry points in `scripts/`, settings in
`configs/`, and numerical checks in `tests/`. Compact experiment evidence is
kept in `docs/validation/`; run logs and earlier plans remain as background
rather than the entry point. See [development notes](docs/DEVELOPMENT.md) for
making and checking future changes.

Audio, dataset copies, caches and model weights stay outside Git. Dataset,
checkpoint and external evaluator licenses are described in the corresponding
protocol documents. A model score is not proof that a recording is authentic.
