# Detection and reliability results

I started with an LFCC-GMM detector, then tested a simple idea: a prediction
might be unreliable if its score changes a lot after reducing the audio
bandwidth. The larger experiment did not support that idea. The clean detector
also performed better than a fixed average of clean and narrowband scores.

The aim is to study fake-speech detection and errors. Running this baseline
on the CPU is a practical implementation choice. It does not define the
research contribution.

All reported detection results use ASVspoof 2019 LA development audio. Official
eval audio remains closed. The [PDF report](report/audio_deepfake_report.pdf)
and [LaTeX source](report/audio_deepfake_report.tex)
give the methods and discussion in one document; the
[complete audit](validation/full_robustness_results_v1.json) records exact values,
file hashes, configuration, failure examples and recovery history.

## Clean speech detection

The seed 42 LFCC-GMM model was trained from all 25,380 official ASVspoof 2019 LA
train recordings and scored all 24,844 official dev recordings. It uses 32
diagonal mixtures per class and at most 50,000 training frames per class. Both
mixtures converged, in 47 and 35 iterations, with no warnings.

| Run | Train recordings | Dev recordings | EER | Minimum normalized t-DCF |
| --- | ---: | ---: | ---: | ---: |
| Balanced pilot, seed 42 | 2,000 | 1,000 | 8.80% | 0.2006 |
| Full train/dev, seed 42 | 25,380 | 24,844 | 5.89% | 0.1527 |
| Full train/dev, seed 43 | 25,380 | 24,844 | 6.67% | 0.1794 |
| Full train/dev, seed 44 | 25,380 | 24,844 | 6.47% | 0.1850 |

Across the three fixed seeds, mean EER is **6.34%**, with population standard
deviation **0.33 percentage points**. Mean minimum normalized t-DCF is
**0.1724**, with population standard deviation **0.0141** (`ddof=0`). All six
class mixtures converged without warnings. Three seeds describe variation;
they are not a significance test. No seed was selected or replaced for its score.

Pilot and full runs use different training and dev populations, class balance
and sampled training frames, with the same 100,000-frame combined cap. The table
does not isolate the effect of additional training data. Full dev keeps its natural
2,548 bona fide / 22,296 spoof counts. These are development results for a Python
adaptation, not held-out evaluation or an exact MATLAB baseline reproduction.
The official eval audio remains closed. Seeds 43 and 44 finished successfully
at 04:44 Istanbul time on 2026-10-10. Each scored all 24,844 dev recordings;
their official-reference metrics and aggregate were rechecked after completion.
See the [three-seed audit](validation/full_multiseed_summary.json) and
[post-run verification](validation/full_multiseed_review.json).

The first full run saved its model but stopped during scoring because our
Windows memory probe created and retained ctypes pointer types on every call.
After fixing the probe, a scoring-only recovery reused the exact model and
original score prefix. Three reproduced scores matched exactly. Recovery
finished at 03:44 Istanbul time on 2026-10-10, with a maximum supervised worker
peak of about 212 MiB. Recovery timings exclude the original training/scoring
work; they are not an uninterrupted end-to-end benchmark.

Evidence: [compact seed 42 audit](validation/full_baseline_seed42.json),
[independent review](validation/full_independent_review.json),
[recovery method](SCORING_RECOVERY.md).

## Full-dev bandwidth stress test

Each saved model scored all 24,844 official dev recordings after a
16 → 8 → 16 kHz polyphase resampling round trip. The transformation preserves
the original length and applies no gain normalization or codec. Higher scores
mean more evidence for bona fide speech. The average is fixed at
`clean_score / 2 + narrowband_score / 2`; no fusion weight was fitted.

| Seed | Clean EER | Narrowband EER | Average EER |
| --- | ---: | ---: | ---: |
| 42 | 5.89% | 18.56% | 8.59% |
| 43 | 6.67% | 24.96% | 9.30% |
| 44 | 6.47% | 22.99% | 9.19% |
| Mean | **6.34%** | **22.17%** | **9.03%** |
| Population SD, percentage points | 0.33 | 2.68 | 0.31 |

| Score | Mean minimum normalized t-DCF | Population SD |
| --- | ---: | ---: |
| Clean | 0.1724 | 0.0141 |
| Narrowband | 0.5298 | 0.0428 |
| Average | 0.2012 | 0.0078 |

![EER and minimum normalized t-DCF for three fixed seeds](figures/detection_comparison.svg)

Figure bars show three-seed means only. The tables retain each seed and
population SD; these are measures of model variation on the same dev data.

Narrowband and average results are custom stress diagnostics. Their t-DCF uses
the unchanged official dev ASV scores; the ASV system was not rerun on
transformed audio. This measures the countermeasure score under bandwidth
reduction, rather than the whole recognition system under a telephone channel.

The average worsened EER and minimum normalized t-DCF in every seed. The small
apparent EER benefit in the earlier pilot did not carry over to this comparison.
I did not tune another weight after seeing the full results.

## Can score changes identify errors?

The dev speakers were split deterministically into eight calibration speakers
(9,609 recordings) and 12 analysis speakers (15,235 recordings). Each model's
threshold was set from clean bona fide calibration scores to accept about 95%
of those recordings. An analysis error means an incorrect clean prediction at
that threshold; it is a different operating point from the EER diagnostic below.

I compared two fixed error-ranking scores: absolute clean/narrowband score
change, `abs(narrowband_score - clean_score)`, and proximity to the clean
threshold, `-abs(clean_score - threshold)`. Higher values rank a recording as
more likely to be an error.

| Error-ranking score | Mean AUROC ↑ | Population SD | Mean average precision ↑ | Population SD |
| --- | ---: | ---: | ---: | ---: |
| Absolute score change | 0.3851 | 0.0844 | 0.0407 | 0.0041 |
| Clean decision margin | **0.9362** | 0.0017 | **0.3834** | 0.0098 |

![Error ranking comparison with error prevalence](figures/reliability_comparison.svg)

There were 774, 844 and 853 analysis errors for seeds 42, 43 and 44,
respectively. Mean error prevalence was 5.41% (population SD 0.23 percentage
points), which is the relevant reference level for average precision. Absolute
score change had lower mean average precision than this prevalence, and was
well behind the clean margin in both metrics. It was not a useful error signal
under this protocol. I retained the original score direction rather than
inverting it after inspecting the results.

All aggregates use the same three fixed seeds and population standard deviation
(`ddof=0`). The seeds share recordings and are not independent validation sets.

## Where the clean detector fails: seed 42 diagnostic

At the same-dev reference EER threshold of 0.6700673, a recording is accepted as
bona fide when its score is greater than or equal to the threshold. This is a
diagnostic threshold chosen using dev labels, not a deployment operating point.

| Dev group | Recordings | Errors | Error rate at this threshold |
| --- | ---: | ---: | ---: |
| Bona fide | 2,548 | 150 rejected | 5.89% |
| A01 | 3,716 | 0 accepted | 0.00% |
| A02 | 3,716 | 0 accepted | 0.00% |
| A03 | 3,716 | 0 accepted | 0.00% |
| A04 | 3,716 | 366 accepted | 9.85% |
| A05 | 3,716 | 0 accepted | 0.00% |
| A06 | 3,716 | 949 accepted | 25.54% |

The observed spoof errors concentrate in A04 and A06. Zero errors in the other
groups do not establish generalization to new generators or recording conditions.
Literal threshold counts may differ from the official EER grid at ties. Ten
illustrative failure examples and the exact selection rule are saved in the
[error audit](validation/full_seed42_errors.json).

Recreate the diagnostic in a new output file:

```powershell
& .\.venv\Scripts\python.exe scripts/analyze_full_dev.py `
  --run-dir outputs/scoring-recovery-seed42 `
  --output outputs/full-seed42-errors-repeat.json
```

## Narrowband pilot

On the earlier balanced 1,000-record dev pilot, the narrowband-only score gave
21.0% EER and the fixed clean/narrowband average gave 8.4%. This was a custom
bandwidth stress test with a fixed ASV reference, not a telephone codec test.

On 629 analysis recordings from speakers separate from calibration, absolute
score change ranked errors worse than the clean-score margin baseline:
AUROC 0.420 versus 0.908. This negative result is retained. No inversion,
fusion-weight search or favorable seed selection was performed after viewing it.
The full-dev repetition is reported above. Pilot and full comparisons differ
in data population and class balance; neither isolates a single cause of the
change in results. See the [pilot audit](validation/robustness_pilot_seed42.json) and
[frozen pilot protocol](validation/robustness_protocol_v1.md).

## Local demo and CPU measurement

The local CLI and browser demo reuse the fixed seed 43 detector, checking
its saved hash before loading. They report a raw model score and processing
time. The input is a mono 16 kHz WAV or FLAC file, up to 60 seconds and
8 MiB. No calibrated probability or universal real/fake verdict is produced.
The browser server binds to this computer only; uploads stay in memory.
See the [demo commands](../README.md#local-demo).

The CPU measurement used the first 20 official dev rows, selected before
timing, with three repeated passes and one numerical thread on an Intel
Core i7-1195G7 (4 physical cores, 8 logical threads). These are all bona fide
recordings from speaker LA_0069, lasting 2.555–5.927 seconds. All 60 observed
scores exactly matched the saved seed 43 clean scores.

| Measurement | Result |
| --- | ---: |
| Mean decode + scoring wall time | 12.40 ms/file |
| Median wall time | 11.64 ms/file |
| 95th percentile wall time | 16.92 ms/file |
| Total measured time / total audio duration | 0.003116 |
| Model verification and loading, measured separately | 0.211 s |
| Peak process memory | 189 MiB |

The file cache and process were warm. Per-record wall time excludes model
loading and browser/HTTP overhead. Repeated observations are dependent,
and this single-speaker bona fide sample does not cover attack diversity,
long recordings or other hardware. This checks that the saved detector can
be used locally; it is not an accuracy experiment or a production benchmark.
The [timing audit](validation/demo_latency.json) and
[fixed plan](validation/demo_latency_plan.json) retain the inputs, model,
environment, source hashes and individual observations. Running
`python scripts/benchmark_demo.py` prints a read-only plan; execution refuses
to overwrite the existing frozen plan or result.

## Limits and next work

The full robustness protocol was frozen before transformed scoring, but after
the clean dev labels/results and pilot stress results had been inspected. Full
dev contains the earlier pilot subset. This is a documented follow-up
experiment, not blind preregistration or independent validation. Calibration
and analysis have separate speakers within dev, while attack types A01–A06 are
also represented in training. The results do not establish performance on new
generators, other datasets or real recordings.

The implementation is a Python LFCC-GMM adaptation, not an exact reproduction
of the official MATLAB baseline. EER and minimum t-DCF summarize dev scores
over thresholds; they do not provide calibrated authenticity probabilities.
The detailed audits retain hash checks and unsuccessful run history. Resource
recovery reused fixed models and verified completed score chunks; no seed was
replaced or retrained to improve its result.

The local demo and small CPU measurement are complete. My next research
step is a measured PartialSpoof localization pilot after checking its
current labels, license, split overlap and evaluation protocol. No
localization results are claimed here.

Dataset and metric sources: [ASVspoof 2019 LA](https://zenodo.org/records/6906306)
and [official evaluation plan](https://www.asvspoof.org/asvspoof2019/asvspoof2019_evaluation_plan.pdf).
