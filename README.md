# Audio Deepfake Detection

A project on detecting synthetic speech and finding manipulated segments in
recordings, by Özgür Ekin Sönmez.

I built an LFCC-GMM baseline, tested how bandwidth changes affect its errors,
and compared LFCC with frozen wav2vec2 features for partial-spoof localization.
The bandwidth-based error signal did not work well; decision margin was better.

**[Report (PDF)](docs/report/audio_deepfake_report.pdf)** ·
[Report (Markdown)](reports.md) · [Experiment details](docs/README.md)

## Results

| Experiment | Development result |
| --- | --- |
| LFCC-GMM, original audio | 6.34% mean EER across three seeds |
| After a 16 → 8 → 16 kHz bandwidth change | 22.17% mean EER |
| Fixed average of original and transformed scores | 9.03% mean EER |
| PartialSpoof localization, LFCC | 39.50% custom duration EER |
| PartialSpoof localization, frozen wav2vec2 follow-up | 28.57% custom duration EER |

Detection used ASVspoof 2019 LA train/dev. Localization used a fixed 128-train /
64-dev PartialSpoof subset. The SSL follow-up allowed more optimizer iterations
than LFCC after the original SSL fit failed to converge. It reduced false alarms
but missed more spoof duration at the fixed zero threshold.

These are development experiments, not official localization benchmark results.
Eval remains closed; the dev subset was reused, the encoder sees the whole
recording, and pretraining overlap is unknown. See the report for methods and limits.

## Run

Use Python 3.11+; the recorded experiment environment was Windows/Python 3.12.

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -e .
python -m audio_deepfake smoke
python scripts/run_tests.py
```

The smoke check and unit tests need no dataset or trained model.

## Local demo

With the matching seed-43 model and manifests from a reproduced run:

```bash
python -m audio_deepfake.demo serve
```

Open `http://127.0.0.1:8765` to score a mono 16 kHz WAV/FLAC file. The score
is not a probability or proof of authenticity. The model and audio are not
included in Git; [setup details](docs/FULL_RUN.md) explain the required artifacts.

## Files

- `src/`: audio processing, models and demo.
- `scripts/`: training, evaluation and report tools.
- `configs/` and `requirements/`: experiment settings and environments.
- `tests/`: numerical, input and pipeline checks.
- `docs/`: report, results and data sources. Longer guides are in `docs/details/`.

Dataset copies, audio, caches and model weights stay outside Git.
[Development notes](docs/details/DEVELOPMENT.md) cover tests and future experiments.
