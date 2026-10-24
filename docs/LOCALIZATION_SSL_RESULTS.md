# Frozen SSL feature experiment

The public PartialSpoof v1.2 selection is fixed at 128 official train and
64 official dev recordings, seed 45. Eval remains closed. This is an exploratory
development pilot; repeated inspection and unknown recording overlap with
encoder pretraining limit generalization claims.

## Separate convergence follow-up completed

Before any SSL dev scores, Ekin approved a separate train-only convergence
budget. [The preregistered protocol](LOCALIZATION_SSL_CONVERGENCE_PLAN.md)
changed only max_iter from 300 to 3000; the original failed experiment below
remains unchanged. The follow-up converged in 303 iterations with no warnings.
Total supervised train stage: 7.672 s, actual process peak 332,316,672 bytes
(316.9 MiB). Ordered train feature bytes, sample identities, scaler mean/variance,
model parameters, source/cache pins and actual worker PID were independently
checked before dev scoring. No dev scores existed at that verification point.

| Same fixed 64-dev diagnostic | Frozen LFCC max 300 | SSL max 3000 follow-up |
| --- | ---: | ---: |
| Custom pooled 20 ms duration EER |39.50%|28.57%|
| Derived native 10 ms RangeEER |39.38%|28.75%|
| Missed spoof at fixed zero |18.29 s|23.09 s|
| False alarm at fixed zero |84.20 s|32.87 s|
| Predicted spoof runs |1349|720|
| Predictions without reference overlap |989|423|

The EER diagnostic decreased 10.93 percentage points, while the fixed-zero
boundary traded lower false alarms for more missed spoof. This is a separate
exploratory follow-up with a different optimizer budget from LFCC 300, not an
equal-training-budget comparison. No threshold was adopted from either EER
sweep. The same repeatedly inspected 64-dev subset, one seed, correlated frames,
whole-recording future context and unknown pretraining overlap limit claims.
RangeEER uses derived native labels and a bounded patched evaluator, without
publisher RTTM parity. Eval stayed closed; no further optimizer/seed search.

The score-only stage completed in 2.515 s with 127.1 MiB peak, no fit. Every saved
score was reproduced exactly; an independent weighted sklearn ROC reproduced
custom EER; fixed-zero durations and all 10 timeline spans also passed replay.
Ten SVGs are in [the figure directory](figures/ssl_convergence_v1).
The [PDF report](report/audio_deepfake_report.pdf) includes separate SSL pages.

Evidence: [train verification](validation/localization_ssl_convergence_train_v1.json),
[dev/diagnostic verification](validation/localization_ssl_convergence_dev_v1.json),
[pre-execution review](validation/localization_ssl_convergence_protocol_review_v1.json).

## Completed extraction

The pinned wav2vec2-base encoder produced 32,710 aligned 20 ms bins with
768 finite float32 dimensions. The bounded T4 extraction completed all 192
recordings in 18.2717 seconds, with external peak RSS 1.394 GiB and exit zero.
Archive SHA-256/CRC, source/model/environment identity and every cache receipt
passed local verification and independent review. Features use whole-recording
context and therefore do not support a streaming claim.

## Fixed 300-iteration head did not complete

The first local invocation stopped before worker launch at the 2 GiB available
RAM gate. After a fresh probe found 5.27 GB available, a separately named second
attempt started fitting. The same train sampler reference retained 4,524 frames
per native class with digest
`0fb814aa3cf8be436f199a2f971e1ccc31eb616a617b097767e93730aeabc82a`.
StandardScaler and logistic regression used the fixed seed 45, C=1, lbfgs,
max_iter 300 and one numerical thread. A ConvergenceWarning was fatal as planned.

In that original max 300 attempt, no model, dev score, SSL EER, range diagnostic
or timeline was accepted. This
establishes failure to finish the fixed optimization budget; it does not
establish the accuracy of these features. Do not put an inferred SSL accuracy
beside the measured LFCC 39.50% or context 42.92% custom duration EER.

The outer monitor's 4,648,960-byte peak in the second attempt measured the Windows
venv redirector rather than its numerical child. It is invalid as training RAM
or a strict resource-gate pass. The separate SSL launchers were repaired with direct base-interpreter
bootstrap and a real 64 MiB same-process probe before the follow-up. The original
attempt logs, config and source copies are retained. Frozen LFCC and upstream
metric sources remain untouched.

The separately preregistered follow-up above preserves this original failure.
Its successful fit is not a retrospective pass of the original experiment.

Evidence: [extraction](validation/localization_ssl_extraction_v1.json),
[sampling preflight](validation/localization_ssl_sampling_preflight_v1.json),
[first pre-launch stop](validation/localization_ssl_head_prelaunch_v1.json),
[fitting failure](validation/localization_ssl_head_convergence_v2.json),
[fixed plan](LOCALIZATION_SSL_PLAN.md), [execution state](details/LOCALIZATION_SSL_RUN.md).
