# Partial manipulation localization: first pilot

Status: the first bounded pilot is complete. Real audio alignment passed,
configuration was frozen before fitting, and ten dev timelines were saved.
See [results and limitations](details/LOCALIZATION_RESULTS.md). The completed
full-recording report remains unchanged. The plan below records the original
gates; final configuration and measurements are in the linked result audit.

## Question and first delivery

Can a small temporal classifier locate fake speech inside a recording?
The first delivery is a checked data manifest, reference/predicted timelines
for ten fixed dev recordings, and a small table of localization errors.
The timeline will show where a model places suspicious audio, with its time
resolution stated explicitly. It cannot prove that a recording is authentic.

Use public PartialSpoof v1.2 from [Zenodo](https://zenodo.org/records/5766198).
Keep official train/dev/eval assignments. Eval stays unopened. The v1.3
detailed-label link is a separate release path and must not be mixed into
the v1.2 experiment. Source checks are in [the protocol review](PARTIALSPOOF_PROTOCOL.md).

## Small scope

1. Fetch only the README and protocol metadata first, with checksums and
   an explicit byte limit. Inspect archive entries safely before extraction.
2. Check eight train and eight dev recordings with labels, exact audio
   durations, and boundary cases. This is an alignment smoke check.
3. Freeze a reproducible selection of 128 train and 64 dev recordings with
   seed 45, sampled separately within each official split and utterance class.
   Save IDs, source hashes, class counts, speakers, durations and spoof-duration
   distributions before fitting. If metadata cannot support this rule,
   revise the plan before training, never after seeing scores.
4. Fit one LFCC plus logistic-regression temporal baseline. Fit feature
   normalization on train only; use a capped, class-balanced reservoir of at
   most 100,000 labelled frames in total. Start with the existing 60-dimensional
   LFCC representation and target 20 ms output intervals. This is a new small
   baseline, not a reproduction of the authors' SSL model.
5. Save scores and examine ten dev examples chosen by ID before scoring.
   Keep threshold sweeps for evaluation separate from any displayed demo
   threshold. The latter is selected on dev and remains a dev diagnostic.

The subset sizes are deliberately smaller than the original 2,000/1,000
scope sketch. Increase them only after timing the alignment smoke check and
one bounded fit. Do not reuse the ASVspoof detector's dev scores as independent
validation: these datasets share source material.

## Alignment gate

The existing LFCC uses a 20 ms window and a 10 ms hop with end padding.
Its frame count is not the count of 20 ms annotation bins. Before a real
pilot, specify and test frame support times, pooling, end padding, short
recordings and final partial bins against real labels. Never trim arrays
silently to their common length. Keep intervals half-open in seconds inside
our own utilities; verify conversion against the native evaluator.

Binary source labels use 0 for spoof and 1 for bona fide, stored as literal
strings in the source arrays. VAD has a separate nonspeech state. Local
inspection found that VAD nonspeech can have either binary label; preserve
the published binary vectors. VAD label 0 alone is not a binary reference.
Any train masking rule must be written into the frozen configuration.

Metadata is now verified: 92,230,352 bytes, all published MD5 checks passed.
The [selection](validation/partial_selection_v1.json) fixes 128 train and 64
dev IDs with equal utterance classes, plus 8+8 smoke and balanced ten dev
timeline IDs. In 11 train and four dev records, native label count differs
from ceil(VAD duration / 10 ms). These are metadata findings; actual audio
headers are needed before deciding how terminal labels align. The new
`labels_for_audio` helper is a candidate strict contract which deliberately
raises on mismatch; it has not passed the real-audio gate.

## Metrics and evaluator gate

Use the authors' [metric implementation](https://github.com/nii-yamagishilab/PartialSpoof/tree/main/metric):
RangeEER is the intended main localization metric; 10 ms reference SegmentEER
and utterance EER are secondary. Report prediction and reference resolutions.
Do not rename a custom overlap score RangeEER. Missed spoof duration and false
alarm duration at the fixed dev threshold are useful readable diagnostics.

Pin the evaluator commit, inspect dependencies and run synthetic conformance
cases in a separate environment before processing real predictions. Check
score direction, threshold ties, duration clipping and scale units. Record
upstream implementation issues explicitly. The [synthetic conformance review](RANGE_EER_CONFORMANCE.md) documents executed
upstream failures and a separately named bounded patched upstream variant.
Its duration formulas and closest-observed selection are retained; the search
has explicit stagnation and iteration guards. It is not an unmodified official
evaluator or a paper reproduction. No real-data RangeEER has run yet.

## Compute and download gates

The [local preflight](validation/localization_preflight.json) records current
RAM and free disk. CPU and GPU names there come from the earlier hardware
audit. No GPU is needed for the planned logistic baseline.

Audio preparation has started in a separate bounded process. The two official
archives total exactly 4,055,073,906 compressed bytes. A three-byte HTTP probe
verified first-byte range responses for both website URLs; arbitrary resume
offsets must still pass response validation. The archives must pass MD5 before
inspection, and only 192 selected WAV files are extracted. Selecting 192
recordings does not imply a small download.
Budget compressed and extracted bytes together, record exact sizes, preserve
at least 20 GiB free disk, and start only after a fresh resource check.
Do not download eval. Do not install evaluator dependencies into the pinned
detection environment. Initial fit cap: 5 minutes, one numerical thread,
process peak 1 GiB, available system RAM floor 2 GiB; stop and preserve logs
when a guard fails. Refine these caps from the smoke measurement before freeze.

## One later ablation

After the baseline works, compare ordinary frame features with recording-level
median/MAD context normalization. The idea is to reduce dependence on the
speaker/channel by comparing a frame with its surrounding recording. It is
an exploratory offline idea, with no novelty claim. Whole-recording context
uses future audio and must not be described as streaming. Freeze the rule
before scoring; do not add this ablation to the first fit.

## Remaining gates

- Complete: metadata hashes, source formats and fixed subset IDs verified.
- Complete: limited synthetic compatibility checks; native binary RTTM still pending.
- Complete: real audio/label alignment and deterministic subset IDs.
- Complete: configuration, sources and pilot IDs frozen before fitting.
- Complete: bounded pilot and ten prediction timelines.
- Complete: checked archives for native binary RTTM; absent in the checked distribution.
- Complete: separately named native-label-derived range diagnostic; publisher parity unverified.
- Complete: one fixed median/MAD context ablation, negative result.
- Complete: error analysis and extended PDF/LaTeX report.

The result is a custom small dev pilot. Official RangeEER and 10 ms SegmentEER
remain unreported. Follow the [results](details/LOCALIZATION_RESULTS.md),
[training sequence](details/LOCALIZATION_TRAINING.md) and [resume guide](LOCALIZATION_RUN.md).
