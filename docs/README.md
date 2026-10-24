# Project documentation

Start with the [PDF report](report/audio_deepfake_report.pdf) or the
[same report in Markdown](../reports.md).

## Results

- [Detection and bandwidth experiments](RESULTS.md)
- [SSL localization comparison and limitations](LOCALIZATION_SSL_RESULTS.md)
- [LFCC localization baseline](details/LOCALIZATION_RESULTS.md)

## Reproduce

- [ASVspoof data and licenses](DATA_PROTOCOL.md)
- [LFCC-GMM full run](FULL_RUN.md)
- [Bandwidth experiment](details/FULL_ROBUSTNESS_RUN.md)
- [PartialSpoof data and labels](PARTIALSPOOF_PROTOCOL.md)
- [Localization training](details/LOCALIZATION_TRAINING.md)
- [SSL feature extraction and runs](details/LOCALIZATION_SSL_RUN.md)
- [Tests and development](details/DEVELOPMENT.md)
- [Report build](report/README.md)

`details/` holds longer implementation and run guides. The remaining experiment
plans retain their original paths because completed runs record their hashes.
`validation/` contains the corresponding numerical records; it is supporting
evidence rather than required reading. Figures are in `figures/`.
