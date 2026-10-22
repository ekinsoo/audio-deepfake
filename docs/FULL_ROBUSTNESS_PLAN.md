# Full-dev narrowband experiment

Written: 2026-10-10. This step freezes the protocol and clean-score calibration
artifacts only. It does not start transformed scoring, training or downloads.
The full experiment contract ID is `full-narrowband-confirmation-v1`; the
older helper protocol ID below remains fixed for speaker partitioning only.

## Question and scope

Does a fixed bandwidth change help identify incorrect predictions from our
small speech detector? Compare the original score, the transformed score and
their fixed average. Then compare absolute score change with the simple
clean-score margin as ways to rank clean prediction errors.

Use all **24,844 official ASVspoof 2019 LA dev recordings**: 2,548 bona fide
and 22,296 spoof recordings, including 3,716 recordings for each attack A01-A06.
Keep official IDs, labels, speakers, attacks and natural class counts. Join
the exact official protocol; exclude the 142 unkeyed dev archive members.
The full train split stays fixed at 25,380 recordings. Eval remains closed.

Use all three completed LFCC-GMM models, seeds **42, 43 and 44**, with their
saved configurations and exact model hashes. Each has 32 diagonal mixtures
per class and a cap of 50,000 retained training frames per class. Reuse their
complete, hash-verified clean dev scores. No retraining, new seed selection,
feature changes or score normalization is allowed in this experiment.

The full clean dev results and labels, and the earlier narrowband pilot
results, are already known. The pilot is a subset of this dev population.
This is a **pre-transform protocol freeze**, not a blind test, independent
confirmation or preregistration before looking at the data. The same dev
population is reused by all three seeds. Three seeds describe model variation;
they are not three independent datasets or a significance test.

## Fixed transformation and scores

Use the exact `TRANSFORM_SPEC` and `narrowband_roundtrip` implementation in
[transforms.py](../src/audio_deepfake/transforms.py). Convert finite mono 16 kHz
input to float64, use `scipy.signal.resample_poly` for 16 to 8 to 16 kHz,
with `window=("kaiser", 5.0)`, constant padding and `cval=0`, then trim to the
original sample count and return float32. Preserve odd-length handling.
No clipping, gain change, normalization, extra filter, noise or codec is added.
Record the exact spec, source hash and SciPy version with the run.

This is an approximate bandwidth stress test. It removes much of the content
above 4 kHz, but does not reproduce a telephone channel or speech codec.
Keep original audio unchanged; transformed waveforms can stay in memory.

For each seed and recording, save the finite paired scores:

- `s_clean`: saved original score; larger means bona fide.
- `s_narrow`: score from the same model after the fixed transform.
- `s_average = s_clean / 2 + s_narrow / 2`, using the same arithmetic as the pilot.
- `delta = s_narrow - s_clean` and `abs_delta = abs(delta)`.

Audio and the fixed model determine inference. Labels, attack IDs and speaker
IDs must not affect transformation, scoring or averaging. Require an exact
one-to-one join with every official dev ID; no missing, duplicate or skipped
scores are accepted.

## Detection metrics

Keep clean official-dev adaptation results separate from custom narrowband
and average-score diagnostics. For each seed, report EER and minimum normalized
t-DCF using the same hash-pinned official ASVspoof 2019 v1 evaluator and matching
dev ASV scores used for the clean runs. Keep the ASV scores unchanged.
The custom t-DCF therefore measures a changed countermeasure against a fixed
ASV reference, rather than both systems receiving a changed channel.

Report all three seeds individually, then their mean and population standard
deviation (`ddof=0`) for each condition. Preserve failures and undefined values;
do not replace or hide an unfavorable seed. EER and minimum t-DCF use dev labels
to choose diagnostic thresholds. These are development metrics, not held-out
eval results or deployment operating guarantees. Keep the Python adaptation
label: feature parity with the official MATLAB baseline has not been shown.

## Speaker partition and clean calibration

Reuse `partition_dev` from [reliability.py](../src/audio_deepfake/reliability.py)
with `PROTOCOL="narrowband-reliability-v1"`. Rank dev speakers by SHA-256 of
UTF-8 `42\0narrowband-reliability-v1\0{speaker_id}`, with speaker ID as a
tie-breaker. Assign the first `floor(2 * n_speakers / 5)` to calibration and the
rest to analysis. For the 20 official dev speakers this is **8 calibration and
12 analysis speakers**. This hash seed is fixed and does not change with the
model seed.

Generate a **new full-ID audit** from the complete official manifests. Record
each group's exact recording IDs, speaker IDs, class and attack counts, and
manifest hashes. Do not reuse the pilot's 371/629 recording counts or its
subset hashes. Require disjoint calibration/analysis speakers and IDs,
disjointness from train, both labels and all A01-A06 attacks in both groups.
If coverage fails, stop and report it; do not search for another partition.

Freeze one threshold per model before computing full transformed scores or
analysis error rankings. Using only clean calibration bona fide scores, sort
them ascending as `b` and choose **`threshold = b[n // 20]`**. Accept bona fide
when `s_clean >= threshold`. This makes at most `n // 20` clean calibration
bona fide scores fall strictly below the threshold; ties can reduce rejection.
Save the model seed/hash, clean-score hash, calibration membership/hash,
number of bona fide scores, selected index, threshold and literal acceptance
counts. Do not substitute a full-dev EER threshold.

Create these artifacts with
[freeze_full_robustness.py](../scripts/freeze_full_robustness.py), which defaults
to a read-only plan and requires `--execute` to write the freeze:

- [Public contract](validation/full_robustness_protocol_v1.json): compact
  partition counts/speakers/hashes, all three thresholds and bona fide
  acceptance counts, model/score pins, source pins and protocol pins.
- `outputs/full-robustness-protocol-v1/partition.json`: full ID metadata.
- `outputs/full-robustness-protocol-v1/calibration.csv` and `analysis.csv`:
  complete selected rows. These local artifacts stay outside Git.

Speaker separation reduces calibration reuse in the diagnostic. It does not
erase the fact that the full dev results and pilot outcomes have already been
seen, and it does not create an independent external test set.

## Reliability and failure analysis

On analysis speakers, an error is disagreement between the clean prediction
at that model's frozen threshold and the true label. Compare these fixed
rankings, where a larger value predicts an error:

- `abs(s_narrow - s_clean)`.
- `-abs(s_clean - threshold)`.

For each seed report AUROC, non-interpolated average precision, error
prevalence, and correct/error counts by label and attack. Use
`evaluate_error_ranking` without changing its ranking directions. If a ranking
population has only errors or only correct predictions, its AUROC and average
precision are null with the reason; report prevalence and counts normally.
For grouped metrics, apply the same one-class rule if they are calculated.

Report every seed and mean/population standard deviation of defined ranking
metrics, with the number of defined seeds. If any seed is undefined, mark the
three-seed aggregate incomplete rather than presenting a reduced set as a
complete aggregate. Do not tune an average weight, invert the delta ranking,
learn a rejection threshold or select a favorable seed after viewing results.

Use the existing `select_failure_examples` rule in
[run_robustness.py](../scripts/run_robustness.py) for each seed. Select the first
ID in each delta-sign and top-quarter-disagreement error stratum, deduplicate,
then fill to ten by descending absolute delta and ID. Top quarters use
`ceil(N / 4)` of all analysis records with ID tie-breaking. If fewer than ten
errors exist, report all available errors and missing strata. These examples
illustrate behavior; they are not a random sample or a performance estimate.

## Execution gate and first deliverable

This document does not authorize relaxing the pilot runner's input checks.
Implement and test a separate guarded full-dev runner before execution. It
must validate the frozen contract, official manifest/reference hashes, model
and clean-score hashes, exact score joins and the new partition/threshold
artifacts. Recompute and require exact partition/threshold equality with the
contract, and recheck source and freeze hashes before transformed scoring.
Save its source/configuration hashes and package versions. If any input or
protocol changes, stop and record a new protocol version before continuing.

Use cached public audio with no download and no training. Inspect GPU, RAM and
disk again before launching. Proposed initial guards are one numerical CPU
thread, at least 3.5 GiB available RAM to start, at least 1 GiB available RAM
while running, 20 GiB free disk, 2 GiB worker peak memory and a 30-minute
processing budget per seed. These are engineering review budgets, not measured
requirements. Measure a small bounded transformed-scoring sample first if
needed; it must not tune the transform or metric choices. Run seeds sequentially
and use fresh bounded worker chunks with recoverable score files and strict
joins. Preserve partial results and stop reasons if a guard triggers.

The first executed deliverable will contain the paired score table for every
seed, clean/custom detection tables, the full-ID speaker audit, frozen
thresholds, ranking comparison and deterministic failure examples. Include
runtime, peak memory, model/configuration/source/environment hashes, dataset
version, input/output hashes, convergence provenance and all limitations.
Keep audio and model artifacts out of Git. Record progress in
[reports.md](../reports.md) and [backlog.md](../backlog.md).

Sources: [ASVspoof 2019 LA v1](https://zenodo.org/records/6906306),
[official evaluation plan](https://www.asvspoof.org/asvspoof2019/asvspoof2019_evaluation_plan.pdf),
[local data protocol](DATA_PROTOCOL.md),
[clean three-seed audit](validation/full_multiseed_summary.json),
[pilot protocol](validation/robustness_protocol_v1.md) and
[pilot results](validation/robustness_pilot_seed42.json).
