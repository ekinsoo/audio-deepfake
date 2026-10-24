---
title: Audio Deepfake project brief
created: 2026-10-09
modified: 2026-10-10
type: project
status: localization-comparison-complete
tags: [audio, deepfake, machine-learning, portfolio]
---

# Audio Deepfake

## Goal

Build a small, reproducible project for detecting fake speech. Later, extend it to find manipulated time ranges inside a recording. The aim is to show careful ML work: a working baseline, a clear experiment, honest error analysis, and a usable local demo.

This is an independent project using public datasets. Company data, internal results, private recordings, and credentials must stay out of the repository. Code and project documents belong here; downloaded data and large model artifacts stay outside Git.

## Selected first scope

Start with full-recording detection on ASVspoof 2019 LA. Use an LFCC-GMM baseline, then test one question: does the change in score after a narrowband transformation help identify unreliable predictions? CPU execution is a practical local implementation choice, not the goal of the project.

Compare the original score, the average of original and transformed scores, and a simple use of their difference. Keep the clean benchmark and custom stress test separate. A successful project does not require a higher score: a well-tested negative result and clear failure cases are useful too.

Partial manipulation localization is the next milestone. Other codecs, noise, unseen synthesis methods, and abstention can follow after the first experiment. Do not add all of them to the first iteration.

These ideas are research questions, not claims of novelty or better performance. A Python implementation must not be called an exact reproduction of the official baseline before feature extraction and scoring differences are checked.

## Starting sources

- [ASVspoof 2019 LA](https://zenodo.org/records/6906306): 7.6 GB archive; selected starting dataset. The main metric is minimum normalized t-DCF, with EER as a secondary metric. See the [official evaluation plan](https://www.asvspoof.org/asvspoof2019/asvspoof2019_evaluation_plan.pdf).
- [ASVspoof 5 data](https://zenodo.org/records/14498691): about 142.3 GB for the full distribution; deferred because of storage and compute needs. [Official baseline code](https://github.com/asvspoof-challenge/asvspoof5) includes RawNet2 and AASIST.
- [PartialSpoof code](https://github.com/nii-yamagishilab/PartialSpoof) and [data](https://zenodo.org/records/5766198): candidate for finding fake segments inside recordings. RangeEER is the proposed main localization metric; label version and time alignment need checking.
- [LlamaPartialSpoof](https://github.com/hieuthi/LlamaPartialSpoof): candidate for later external evaluation. Confirm its version, license, evaluation-only protocol, and possible source overlap before use.

This brief is a starting snapshot. Verify licenses, distribution contents, protocols, and current code before relying on them. Detailed source checks and initial hardware measurements are in [reports.md](../reports.md).

## Experiment rules

- Preserve official train, development, and evaluation splits. Keep evaluation closed until model and experiment choices are fixed.
- Store pilot subsets separately with a seed and record IDs. Pilot results are not full benchmark results.
- Record dataset version/checksum, code revision, environment, configuration, seeds, score direction, metrics, runtime, memory, and failure cases.
- Measure free disk, RAM, and available GPU before large downloads or long training. Start with a small measured pilot; do not assume GPU availability.
- Keep custom robustness tests separate from official benchmark results. Explain leakage risks and dataset limits.
- ASVspoof 2019 and PartialSpoof share source material. Testing between them is not independent evidence of generalization to a new dataset.

## Expected output

Reproducible code, a comparison table, error analysis, a short report, and a local demo that returns a recording score. Add a manipulation timeline when localization is ready. Report CPU latency or ONNX behavior only after measuring it. The demo must not promise that it can prove a recording is authentic.

The implementation in this repository is the source of truth. At useful milestones, prepare a short summary with source links that can be copied into EkinOS. Do not assume access to personal notes or write to them without authorization.

## Current work

Full train/dev models and narrowband scoring are complete for seeds 42, 43
and 44. Mean EER is 6.34% clean, 22.17% narrowband and 9.03% for the fixed
average. Absolute score change ranked errors poorly (mean AUROC 0.385), behind
the margin baseline (0.936). Independent checks reproduced the score joins,
metrics, thresholds and ranking results. The [report](report/audio_deepfake_report.pdf),
[results](RESULTS.md) and [audit](validation/full_robustness_results_v1.json)
document the negative result, recovery history and limitations. Eval remains
closed. A local CLI/browser scoring demo now reuses the fixed seed 43 model.
Its warm-cache CPU measurement on 20 fixed bona fide dev recordings repeated
three times gave 12.40 ms mean and 16.92 ms p95 decode/scoring time. This
single-speaker check is separate from accuracy evaluation and excludes
model load and browser overhead. See the [timing audit](validation/demo_latency.json).
The first PartialSpoof localization pilot is complete. All 192 selected
recordings passed real audio/native-label alignment; seed 45 LFCC/logistic
trained on 128 and scored 64 dev recordings. Custom duration-weighted 20 ms
EER was 39.50%, with substantial false alarms at fixed threshold zero.
This is not official SegmentEER or RangeEER. One fixed recording median/MAD
ablation worsened this diagnostic to 42.92%. The checked distribution has no
publisher RTTM reference; explicitly derived native-label RangeEER diagnostics
were 39.38% baseline and 42.97% context. Error analysis and the extended
PDF/LaTeX report are complete. Publisher-reference parity remains unverified.
See [results](details/LOCALIZATION_RESULTS.md), [metric details](details/LOCALIZATION_METRICS.md),
[pilot plan](LOCALIZATION_PILOT_PLAN.md) and [source review](PARTIALSPOOF_PROTOCOL.md).

The earlier balanced pilot is preserved separately. Its clean EER was 8.8%
and fixed-average EER 8.4%; that small benefit did not carry to the full run.
Both used the same 50,000-frame-per-class cap, but their training recordings,
sampled frames, dev populations and class balance differ. See the
[pilot audit](validation/robustness_pilot_seed42.json), [starting plan](details/START_PLAN.md),
[reports](../reports.md) and [backlog](../backlog.md).


## SSL continuation (2026-10-11)

Pinned wav2vec2-base extraction and a separately preregistered convergence
follow-up are complete. Original300-iteration SSL failure is retained; follow-up
max 3000 converged in 303 iterations, independently verified before fixed 64-dev
scoring. Custom duration EER 28.57% vs LFCC 39.50%; fixed-zero false alarm improved
while missed spoof worsened. Different optimizer budgets, repeated dev inspection
and unknown encoder overlap remain limits. See [SSL results](LOCALIZATION_SSL_RESULTS.md)
and the updated report. Eval remains closed.
