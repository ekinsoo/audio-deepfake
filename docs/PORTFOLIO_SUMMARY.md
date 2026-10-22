# Project summary

## GitHub description

Audio deepfake detection and localization experiments with reproducible
baselines, a local demo, prediction timelines and error analysis.

## CV wording

- Built reproducible audio deepfake detection and segment-localization pipelines
  using LFCC-GMM and logistic baselines, official train/dev splits and guarded runs.
- Tested bandwidth-based error ranking and recording-context normalization;
  documented negative results, reference-alignment issues and prediction failures
  with a local demo, visual timelines and an English technical report.

The detection experiment reached 6.34% mean EER on ASVspoof 2019 LA dev across
three seeds. This is a development result from a Python baseline adaptation,
not an official leaderboard submission. Localization remains a weak small
pilot. Avoid calling the project state of the art, a production detector,
an unseen-generator validation or a new-method contribution.

Evidence: [report](report/audio_deepfake_report.pdf),
[detection results](RESULTS.md), [localization results](LOCALIZATION_RESULTS.md).
