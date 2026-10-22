# Current step - SSL convergence follow-up verified (2026-10-11)

The original fixed 300 SSL fit remains a failed convergence experiment. A
separately preregistered train-only max 3000 follow-up converged in 303 iterations,
7.67 s external duration and 316.9 MiB actual worker peak. Independent retained
feature-byte/scaler/PID/resource verification passed before any dev scoring.
The repaired launchers use direct base Python;36 SSL tests passed.

Separate no-fit scoring completed the same fixed 64 dev IDs and 10 timelines.
Custom pooled20ms duration EER: SSL 28.57%, frozen LFCC 39.50%; derived native
RangeEER diagnostic:28.75% vs39.38%. At fixed zero, false alarms decreased
84.20 s ->32.87 s but missed spoof increased18.29 s ->23.09 s. Different optimizer
budget, repeated dev inspection, whole-recording context and unknown encoder
pretraining overlap prevent equal-budget or unseen-domain claims. Eval closed.

Evidence: docs/validation/localization_ssl_convergence_train_v1.json and
localization_ssl_convergence_dev_v1.json; docs/LOCALIZATION_SSL_RESULTS.md.
Report updated with separate protocol/failure/results/timelines. Next work
requires a separately fixed independent evaluation protocol; no adaptive solver,
seed or threshold search. Older entries below are historical.

# Bounded SSL comparison continuation

The [fresh supervisor v2 smoke](validation/localization_ssl_colab_smoke_v2.json)
passed. The next stage uses exactly the frozen 128 train / 64 dev recordings,
wav2vec2-base revision, float32 last-layer representation, seed 45 and frame
mapping in [the frozen plan](LOCALIZATION_SSL_PLAN.md). This document is an
execution guide, not evidence that extraction or training has completed.

Current result: [192-cache extraction verified](validation/localization_ssl_extraction_v1.json).
The [local head stopped before launch](validation/localization_ssl_head_prelaunch_v1.json)
at the unchanged RAM gate. Reuse the verified cache; preserve the stopped head
directory and inspect resources before an explicit fresh output attempt.

## Inputs and transfer

`scripts/build_ssl_extraction_bundle.py` creates an ignored ZIP with the 192
previously audited public WAV files, extraction sources, the pinned config,
selection/alignment audits, smoke evidence and dataset attribution. No new
dataset download is needed. No frame-label arrays, eval audio, private audio,
weights or credentials are included. Selection/alignment metadata does contain
utterance classes and label-derived summaries; these are not training targets
for the frozen encoder. Native frame-label arrays stay local for head fitting.

The user explicitly approved the 192-record public package transfer and bounded
Colab execution. The final reviewed source package is v2, 14,698,353 bytes,
SHA-256 `b622365cdf58773045454cfa9fb27a545a0911ba2ee9546eb1225bda4f7ac03d`.
See `docs/validation/localization_ssl_extraction_bundle_v2.json` (preparation
snapshot) and the completed extraction audit. V1 was not executed.

Recheck the current Colab T4, RAM, disk, environment and model paths. The
existing model is `/content/ssl-smoke-bundle/model`; encoder Python is
`/content/ssl-isolated-v1/bin/python`. Do not silently replace an expired VM,
switch packages, use a CPU fallback, or recreate a stopped attempt. Verify
the uploaded ZIP hash, require a fresh extraction directory, bound expanded
bytes, reject paths escaping the directory and verify every manifest file.

## Extraction and recovery

Use the new extraction supervisor, not the old single-recording worker:

```text
/content/ssl-isolated-v1/bin/python /content/ssl-extraction-inputs-v2/ssl_extract_job.py --execute --bundle-dir /content/ssl-extraction-inputs-v2 --model-dir /content/ssl-smoke-bundle/model --output-dir /content/ssl-extraction-v1 --device cuda
```

The external supervisor enforces 1,800 cumulative seconds, a 4 GiB start and
immediate pre-load available-RAM gate, 2 GiB running available RAM, 3 GiB
worker peak, 20 GiB free disk and a 1 GiB feature-cache cap. Do not relax them.
Check actual parent completion, exit code and final validation, not just
per-record worker receipts. Save stdout/stderr, attempts, environment, source
pins, manifest, identity, result and every cache/receipt outside Git.

Caches live under `cache/{train,dev}/{ID}.npy` with matching JSON receipts.
Validate audio, model, source, environment, device, checksum, geometry,
float32 dtype and finite values before reuse. A stopped attempt requires
inspection and an explicit `--resume` invocation; prior charged time counts
against the same 1,800-second total. Partial or inconsistent pairs require
investigation. Never take over an active/stale lock automatically. Preserve
all failed attempts and their logs.

Recover the complete result archive locally and independently verify it
before allowing the head to fit. A transport failure must not cause model
extraction to rerun. Preserve CPU versus CUDA provenance and the measured
smoke numerical differences; there is no result-dependent tolerance tuning.

## Local head and comparison

The new dimension-aware head leaves the frozen 60-dimensional LFCC model
untouched. Native labels and train-only random-priority sampling must select
the same `(recording ID, bin index)` frames as the baseline: 4,524 per class,
seed 45, sorted IDs. Record the canonical sample-ID digest before fitting.
Fit the scaler on retained train frames only and use the fixed logistic
regression settings. External fitting limit is 300 seconds. Stop on a
convergence warning; do not extend iterations after seeing dev scores.

Compare custom duration-weighted 20 ms EER, fixed-zero missed-spoof and false
alarm seconds, fragmentation and the same ten dev timelines. The existing
derived-reference RangeEER diagnostic remains separate, with its missing
publisher-reference caveat. Eval stays closed. Repeated inspection of this
small dev subset and unverified pretraining recording overlap limit claims.
Update results/report only after independent verification of completed work.
