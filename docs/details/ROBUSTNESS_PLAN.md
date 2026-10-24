# Narrowband score stability

Status: first pilot experiment completed. The unchanged pre-result protocol is
preserved in [the frozen snapshot](../validation/robustness_protocol_v1.md).
The [run audit](../validation/robustness_pilot_seed42.json) records the outcomes:
absolute score change did not beat the simple margin baseline for ranking
errors. This is one small experiment, not a claim of a new method. The rules
below describe the original plan; they were not revised after viewing results.

## Question

Does the change in an LFCC-GMM score after a narrowband transformation help rank
incorrect predictions? We will also compare the original score with a fixed
average of original and transformed scores. A negative result is useful: the
transformation may hurt detection or add no information beyond the original
score's distance from the decision threshold.

Keep the full-pilot model fixed: 2,000 official train recordings, 32 diagonal
mixtures per class, seed 42 and at most 100,000 retained training frames. Use the
same frozen model and LFCC settings for both versions of each dev recording.
No transformed-data retraining or parameter search belongs in this experiment.

## Fixed transformation

Use `scipy.signal.resample_poly` twice:

```python
down = resample_poly(x, 1, 2, window=("kaiser", 5.0), padtype="constant", cval=0)
restored = resample_poly(down, 2, 1, window=("kaiser", 5.0), padtype="constant", cval=0)
y = restored[:len(x)].astype("float32")
```

Inputs must be finite, nonempty, mono audio at 16 kHz. Return 16 kHz audio with
exactly the original sample count; trimming removes the possible extra sample
for odd-length inputs. Convert the input to float64 before the two resampling
steps; convert the final waveform to float32 for the LFCC extractor. Record
these dtypes. Do not normalize, clip, add noise, change gain,
apply a separate high-pass filter or encode the output with a speech codec.
Never overwrite source files. Transformed waveforms can stay in memory.

This removes much of the content above 4 kHz through antialias filtering. It is
an approximate bandwidth stress test, not a real telephone channel or codec.
Boundary padding and filter transients are part of the recorded transformation.
The installed SciPy 1.18.1 signature was checked locally:
`resample_poly(x, up, down, axis=0, window=('kaiser', 5.0), padtype='constant', cval=None)`.
Pin the SciPy version and all explicit arguments in the run metadata.

For each recording, save:

- `s_clean`: original score; higher means bonafide.
- `s_narrow`: transformed score from the same model.
- `s_average = (s_clean + s_narrow) / 2`: fixed, untrained fusion.
- `delta = s_narrow - s_clean` and `abs_delta = abs(delta)`.

The inference function receives audio and the frozen model only. Labels,
speaker IDs and attack IDs are used later for evaluation and grouping; they
must not affect transformation, scoring or fusion.

## Detection and stress-test reporting

Report the complete selected 1,000-record dev set in two separate tables:
clean pilot detection, then custom narrowband/average stress-test diagnostics.
Use the existing hash-pinned official 2019 reference evaluator and official LA
dev ASV scores for EER and minimum normalized t-DCF. Record score direction and
fraction units. All selected IDs need finite scores; no silent skipping.

The ASV scores remain unchanged. Transformed CM results therefore describe
changes to the countermeasure with a fixed ASV reference, not a simulation in
which both ASV and CM receive the transformed channel. These are custom pilot
results. Minimum t-DCF and EER select thresholds from the scored dev labels;
they do not establish an operating threshold for new data. The full-dev
comparisons are exploratory development results, not held-out eval evidence.

## Reliability analysis without reusing its evaluation labels

Before inspecting errors or transformed scores, create a deterministic split
of the selected dev speakers. Sort unique speaker IDs by the hexadecimal
SHA-256 of UTF-8 `42\0narrowband-reliability-v1\0{speaker_id}`; use the first
`floor(0.4 * number_of_speakers)` speakers for calibration and the rest for
analysis. With the current 20 dev speakers this is 8 calibration / 12 analysis.
Save the exact IDs, split hashes and counts. The train model remains unchanged.

Require both labels and every selected spoof attack A01-A06 in both groups;
validate speaker disjointness and report label, speaker, attack and recording
counts. Also check disjointness from train. If these checks fail, stop the
reliability experiment and report the limitation. Do not try alternate seeds
until favorable coverage or results appear. A revised partition would require
a documented new protocol before using its analysis outcomes.

Use one predefined operating criterion: accept at least 95% of calibration
bonafide recordings using the **clean** score. For ascending calibration
bonafide scores `b` with length `n`, choose
`threshold = b[floor(0.05 * n)]`, and predict bonafide when
`s_clean >= threshold`. The number of calibration bonafide scores strictly
below this threshold is at most `floor(0.05 * n)`; ties can make the rejection
rate lower. This convention is deterministic and does not interpolate a
quantile. It is an experimental operating point, not a deployment guarantee.
Freeze this threshold before viewing analysis labels or errors.

On the analysis group, define an error as disagreement between that clean-score
prediction and the true label. Compare two fixed error-ranking scores:

- `abs_delta`: larger changes suggest a less reliable prediction.
- `-abs(s_clean - threshold)`: predictions closer to the original threshold
  form the simple margin baseline.

Report error-prediction AUROC and average precision (AUPRC summarized as
non-interpolated average precision), with error prevalence as the reference
level. Give correct/error counts separately for bonafide and spoof recordings,
and attack-wise error counts. For any overall or grouped result lacking either
errors or correct predictions, return undefined/null with the reason; do not
invent AUROC 0.5 or present a one-class ranking as evidence. Small counts limit
interpretation even when a metric is defined.

Do not learn a fusion model, tune the average weight, select a delta threshold,
or optimize a rejection fraction on this analysis group. An EER threshold
fitted on all dev labels would leak into this error analysis. Quantiles chosen
after inspecting errors have the same problem. If a later experiment learns
fusion or abstention, it needs a separate calibration protocol and evaluation.

## First deliverable and checks

The first deliverable is one reproducible score table, the clean/custom metric
tables, the calibration/analysis manifest and threshold, an error-ranking
comparison, and ten transparent failure examples selected by a documented rule.
Include both directions of score change and cases where margin and delta
disagree. Record model, input manifest, source code and environment hashes;
transformation parameters; seed; timing; memory; counts; and undefined metrics.
Keep audio and model artifacts out of Git.

Before a real run, check odd/even sample counts, finite/mono input validation,
determinism, absence of input mutation, preservation of a 1 kHz test tone,
attenuation of a 6 kHz tone, exact ID pairing and exclusion of eval. Check split
and threshold conventions with ties. Measure transformed scoring separately
from clean scoring. Reuse the frozen full-pilot artifacts rather than fitting
a second model.

Interpret improvements cautiously. The selected data are balanced rather than
representative of deployment prevalence, attacks overlap the train taxonomy,
and the reliability analysis still comes from the official dev domain. This
does not demonstrate unseen-attack or cross-dataset generalization. The
resampling method is standard SciPy signal processing; score averaging and
margin/error-ranking diagnostics are established baselines. The portfolio
value is a clear question, reproducible comparison and honest failure analysis.
