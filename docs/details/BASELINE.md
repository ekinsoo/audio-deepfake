# LFCC-GMM pilot

This is a small CPU baseline for the first experiments. It uses our LFCC
extractor and two scikit-learn diagonal Gaussian mixtures. It is not an exact
reproduction of the official MATLAB implementation or its 512-component GMMs.
See [LFCC.md](LFCC.md) for the feature conventions and differences.

## Training

Run the frozen 50-train / 50-dev mechanical check from the repository root:

```powershell
.venv\Scripts\python.exe scripts/run_baseline.py --config configs/smoke.toml --output-dir outputs/baseline-smoke100-seed42
```

The runner requires the downloaded selection and its hash journal, plus the
external evaluator and official dev ASV scores described in [METRICS.md](../METRICS.md).
It checks selection/audio/reference hashes and train/dev speaker separation
before fitting. Existing output directories are protected; use a new directory
for another run. It writes `model.joblib`, `scores.csv`, `config.json` and
`summary.json` under ignored outputs.

The smoke config uses 8 components and at most 10,000 frames per class. This is
a pipeline check, not a tuned model or the 32-component, 2,000/1,000-record
pilot.

For the full pilot, first prepare and verify the fixed audio selection as
described in [PILOT_DATA.md](PILOT_DATA.md). Prepare the external evaluator and
official dev ASV scores with `scripts/fetch_metric_reference.py`; see
[METRICS.md](../METRICS.md). Then run from the repository root:

```powershell
.venv\Scripts\python.exe scripts/build_pilot_selection.py
.venv\Scripts\python.exe scripts/run_baseline.py --config configs/pilot.toml --output-dir outputs/baseline-pilot-seed42
```

The exporter verifies both original manifest checksums, preserves every field
and the original train-then-dev row order, and writes a combined UTF-8 CSV with
LF line endings. Its SHA-256 is pinned in `configs/pilot.toml`. Repeating the
export is safe when the existing file is identical; changed outputs are refused.
The runner requires all 3,000 selected recordings and their preparation journal.
It does not silently fall back to the smaller smoke selection. The pilot uses
32 components, at most 100,000 training frames, seed 42, 50 EM iterations and one
numerical thread. Eval remains excluded; this balanced subset is still a custom
pilot rather than a full official benchmark.

- Only official train rows are accepted. Both classes are required; duplicate
  recording IDs and invalid protocol rows are rejected before loading audio.
- Default: 32 components per class, seed 42, diagonal covariance, regularization
  `1e-6`, tolerance `1e-3`, 50 EM iterations, one k-means initialization.
- At most 50,000 frames are retained per class (100,000 total). Random priorities
  sample uniformly from all class frames. Longer recordings contribute more
  frames. IDs are sorted before sampling; manifest input order does not change
  the sample. All frames are retained when fewer than the budget exist.
- Audio is loaded one recording at a time. Memory contains class reservoirs and
  temporary arrays for the current recording, rather than the full feature set.
- There is no CMVN, standardization, VAD or feature scaling. The two classes use
  the same fixed feature configuration. Fits and likelihood scoring use one
  native numerical thread to keep the pilot resource budget predictable.

`train_baseline(rows, data_root, lfcc_config=..., gmm_config=...)` returns a
`BaselineModel`. The caller must verify audio provenance and the hash journal
before training. The lower-level `train_feature_stream` accepts a feature loader
for tests and controlled experiments.

Metadata records configuration, retained/seen frame counts, iterations,
convergence, warnings, elapsed times, training IDs, package versions and SHA-256
hashes of the baseline and feature source files. Non-convergence is recorded;
it must be reported when interpreting a pilot. A pilot result is not a full
official benchmark result.

## Measured pilot result

The seed-42 run on 2026-10-09 used all 2,000 selected train recordings and all
1,000 selected dev recordings. Each split was label-balanced. The complete
archive passed its published size and MD5 checks, and selected audio matched
the preparation journal. Eval audio was not extracted or scored.

| Clean dev pilot | EER | Minimum normalized t-DCF |
| --- | ---: | ---: |
| 32-component LFCC-GMM | 8.8% | 0.2006 |

Training took 39.22 seconds and clean dev scoring took 8.77 seconds on the local
CPU with one numerical thread. Peak process memory was 189.93 MiB. Both class
models converged without warnings: 43 iterations for bonafide and 30 for spoof.
The retained training budget was 50,000 frames per class. Peak memory includes
imports; filesystem cache was not controlled. See the
[saved pilot report](../validation/baseline_pilot_seed42.json) for exact metrics,
configuration, hashes, environment and resource measurements.

## Scoring

For each recording, the score is the mean frame log likelihood under the
bonafide mixture minus the mean under the spoof mixture. Higher scores indicate
bonafide speech. No threshold is fitted inside this module.

Score one local recording with a saved model:

```powershell
.venv\Scripts\python.exe -m audio_deepfake score path/to/recording.flac --model outputs/baseline-pilot-seed42/model.joblib
```

Use only local joblib models you created and trust. Loading a joblib file can
execute code before validation. `--model` is required; there is no default model.
The recording must be finite, non-empty mono audio at the model's sample rate;
the command does not resample or downmix it.

JSON output contains the recording score, score direction, sample count,
duration, feature/model settings and model SHA-256. It does not expose the
recording's path, turn the score into a probability, or apply a decision
threshold. This is a single-recording diagnostic, not a benchmark evaluation
or a claim that an arbitrary recording is authentic.

`score_rows(model, rows, data_root)` returns `(utterance_id, score)` pairs in
manifest order. Train scores are diagnostics; dev scores are the development
result. Eval scoring is blocked unless the caller passes `allow_eval=True`.
Do not use eval to tune components, frame budgets, transforms or thresholds.

## Narrowband experiment

Run this only after the frozen full pilot has completed successfully. The
smoke model cannot be used for this experiment.

```powershell
.venv\Scripts\python.exe scripts/run_robustness.py --baseline-run-dir outputs/baseline-pilot-seed42 --config configs/pilot.toml --output-dir outputs/robustness-pilot-seed42
```

Use a new output directory. The runner verifies the baseline model and score
file hashes, configuration, training IDs, audio journal and selected dev rows.
Only use a baseline directory you created and trust: verifying a hash against
its accompanying summary does not make a third-party joblib model safe.
Changes to the frozen partition, protocol or reliability helper stop the run.
The default checks use `docs/validation/reliability_partition_seed42.json` and
`docs/validation/robustness_protocol_v1.md`.

The runner reuses the saved clean scores. It transforms each selected dev
recording from 16 to 8 to 16 kHz in memory, extracts the same LFCCs, and scores
it with the same frozen model using one numerical thread. It does not retrain,
change gain, clip the waveform or simulate a speech codec. Labels and attack
IDs are used for evaluation only.

The reliability threshold comes from clean calibration scores: accept at
least 95% of calibration bonafide recordings. `calibration.json` records this
threshold, calibration IDs and manifest hash before transformed scoring or
analysis error ranking. The held-apart analysis speakers are used to compare
absolute score change against distance from the original threshold. AUROC and
average precision are undefined when every analysis prediction is correct or
every prediction is wrong. See [ROBUSTNESS_PLAN.md](ROBUSTNESS_PLAN.md) for the
exact split, ties and metric rules.

Outputs stay under the ignored run directory:

- `paired_scores.csv`: clean, narrowband, average, signed change and absolute
  change for every dev ID, with its calibration/analysis group.
- `calibration.json` and `partition.json`: the fixed operating rule and groups.
- `recording_timings.json`: per-recording load, transform, feature and score time.
- `summary.json`: metrics, error rankings, selected failure examples, resource
  measurements, environment and hashes.

Clean pilot metrics and custom narrowband/average metrics are reported
separately. Official dev ASV scores stay fixed, so this does not model a
transformation of both ASV and the countermeasure. Full-dev EER and minimum
t-DCF choose thresholds on dev and are exploratory. The separate reliability
analysis avoids reusing calibration speakers, but remains in the same dev
domain. These results do not establish unseen-attack or deployment performance.

The completed custom stress test on the same 1,000 dev recordings gave:

| Custom diagnostic | EER | Minimum normalized t-DCF |
| --- | ---: | ---: |
| Narrowband score | 21.0% | 0.4962 |
| Average of clean and narrowband scores | 8.4% | 0.1913 |

The average's small difference from clean scores is exploratory; no statistical
test or independent evaluation establishes an improvement. Narrowband-only
scoring substantially increased errors at the EER operating point. The saved
ASV scores were unchanged throughout.

For reliability, calibration used 371 recordings from 8 speakers and analysis
used 629 recordings from the other 12 speakers. The fixed clean threshold was
0.7592377067. Analysis contained 63 errors and 566 correct predictions, giving
an error prevalence of 0.1002.

| Fixed error-ranking score | AUROC | Average precision |
| --- | ---: | ---: |
| Absolute score change | 0.4200 | 0.0802 |
| Negative absolute clean-score margin | 0.9082 | 0.5086 |

Absolute score change failed to rank errors well in this experiment; the
original margin was much stronger. Average precision for score change was also
below the 0.1002 error-prevalence reference. This is a negative result for the
frozen hypothesis on this dev subset, not a reason to reverse the score direction
after looking at the labels. Transformed scoring took 10.28 seconds, with
145.19 MiB peak process memory. The
[saved robustness report](../validation/robustness_pilot_seed42.json) records the
frozen partition, calibration, exact results, failure cases and limitations.

## Local artifacts

`save_model(model, path)` stores a joblib file including metadata. Load it with
`load_model(path)` only when it is a local artifact you created and trust.
Joblib can execute arbitrary code during loading, before type validation;
never load models from unknown contributors. Loading warns on package-version
differences. Keep model files out of Git and keep the environment lock with the
experiment report.
