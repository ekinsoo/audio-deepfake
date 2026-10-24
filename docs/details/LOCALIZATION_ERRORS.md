# What the first localization model gets wrong

This analysis uses the same fixed 64 official dev recordings as the first
pilot (32 bona fide, 32 partially spoofed). The classifier threshold remains
zero, chosen before fitting: scores below zero predict spoof. No training,
smoothing, threshold search or new data selection was done here. This is a
custom diagnostic, not an official benchmark or an independent validation.

## Fixed-zero errors

The original pilot uses minimum-pooled 20 ms native binary references and
clips the last bin to the actual audio duration.

| Recording class | Spoof duration | Missed spoof | Bona fide duration | False alarm |
| --- | ---: | ---: | ---: | ---: |
| Bona fide recordings | 0 s | — | 122.91 s | 58.94 s (47.96%) |
| Partially spoofed recordings | 58.05 s | 18.29 s (31.50%) | 52.18 s | 25.25 s (48.40%) |
| All 64 | 58.05 s | 18.29 s (31.50%) | 175.09 s | 84.20 s (48.09%) |

False alarms are common even in entirely bona fide recordings. They are
not limited to clean spans next to a manipulation.

## Speech and nonspeech

For this section, each prediction interval is intersected with the original
10 ms binary reference intervals and printed VAD intervals. VAD states 0/1
mean speech here; state 2 means nonspeech. Binary truth still comes from
the native binary labels: nonspeech can be either bona fide or spoof.
Audio outside printed VAD coverage stays in a separate group. Nothing is
trimmed, padded, or silently labelled using VAD.

| VAD group | Native spoof duration | Missed spoof | Native bona fide duration | False alarm |
| --- | ---: | ---: | ---: | ---: |
| Speech | 33.66 s | 12.05 s (35.80%) | 115.26 s | 52.17 s (45.26%) |
| Nonspeech | 23.53 s | 5.88 s (24.99%) | 60.66 s | 32.52 s (53.61%) |
| Outside printed VAD | 12.06 ms | 3.69 ms | 15.81 ms | 5.62 ms |

False alarms are higher in nonspeech, but speech itself also has a large
false-alarm rate. A speech mask alone would not address the full problem,
and dropping nonspeech would change the evaluation target.

These native 10 ms totals differ from the pooled 20 ms totals above: minimum
pooling expands a spoof region to its containing 20 ms bins near transitions.
The two reference definitions must not be mixed in one rate calculation.

## Short manipulations and fragmented predictions

Duration groups were defined in the analysis source before producing the
output. Each group contains contiguous native binary spoof runs, not whole
recordings. A recording can contribute runs to more than one group.

| Native spoof run duration | Runs | Spoof duration | Missed duration fraction |
| --- | ---: | ---: | ---: |
| Up to 100 ms | 38 | 0.89 s | 42.70% |
| Over 100 through 500 ms | 33 | 8.08 s | 30.54% |
| Over 500 ms | 38 | 48.23 s | 31.28% |

The shortest runs have a higher missed-duration fraction in this sample.
They contribute less than one second of reference audio, so this does not
establish a general length effect. Runs from one recording are correlated.

Using the original 20 ms reference grid, the model produces **1,349 spoof
runs for 109 reference runs**. The median predicted run is only **60 ms**.
Of the reference runs, 15 have no predicted overlap, 35 overlap one predicted
run, and 59 are split across multiple predicted runs. Of the predicted runs,
989 have no reference overlap. These are overlap counts, not event
precision/recall under an official matching protocol.

The median distance to the nearest predicted onset and offset is 20 ms.
This seemingly small number should not be read as good boundary detection:
the model generates many boundaries, and this descriptive calculation allows
one predicted boundary to be nearest to several reference boundaries. It
does not require one-to-one matches or penalize every extra boundary.
Predictions have a 20 ms grid with a clipped terminal interval; no finer
localization precision is claimed.

## The ten preselected timelines

These are all the IDs fixed before scoring, including the five bona fide
examples. The values use the original pooled 20 ms reference at threshold
zero; no examples were selected after inspecting errors.

| ID | Missed spoof | False alarm | Predicted spoof runs |
| --- | ---: | ---: | ---: |
| CON_D_0000586 | 0.860 s | 0.400 s | 17 |
| CON_D_0000734 | 0.660 s | 0.654 s | 20 |
| CON_D_0002058 | 0.525 s | 0.960 s | 21 |
| CON_D_0003090 | 0.060 s | 0.420 s | 6 |
| CON_D_0003229 | 0.340 s | 0.520 s | 20 |
| LA_D_1161800 | — | 1.412 s | 24 |
| LA_D_1323454 | — | 3.746 s | 57 |
| LA_D_1469404 | — | 1.760 s | 17 |
| LA_D_1494418 | — | 1.320 s | 19 |
| LA_D_1796445 | — | 1.548 s | 22 |

The complete interval coordinates for these examples are in the audit;
the [pilot results](LOCALIZATION_RESULTS.md) link the SVG timelines.
CON_D_0003090 remains a useful annotation check: VAD has no speech labelled
spoof, while the publisher's binary reference includes spoof in nonspeech.
It must not be converted into an entirely bona fide reference.

## What to try next

The [reference and bounded RangeEER diagnostic](LOCALIZATION_METRICS.md) is
now complete, with publisher-reference parity explicitly unverified. One
separately frozen recording median/MAD comparison also completed and worsened
custom EER to42.92%. The original baseline is retained. High false alarms
justify investigating a stronger representation but do not prove their
cause or predict an improvement. Freeze one change before fitting. Temporal smoothing would
trade short-spoof sensitivity against fragmentation and needs its own
predefined rule and comparison rather than repeated dev tuning.

PartialSpoof shares source material with ASVspoof. These 64 recordings are
small and repeatedly inspected; their errors guide development, not a claim
of generalization to unseen attacks or datasets. The LFCC delta features
also use future context, so these results do not describe streaming behavior.

Evidence: [machine-readable analysis](../validation/localization_errors_v1.json),
[pilot configuration and results](../validation/localization_pilot_v1.json),
[alignment audit](../validation/localization_alignment.json),
[analysis script](../../scripts/analyze_localization_errors.py).

Reproduce from the retained local pilot artifacts:

```powershell
.venv/Scripts/python.exe scripts/analyze_localization_errors.py
.venv/Scripts/python.exe -m unittest discover -s tests -p test_localization_errors.py -v
```

The score file, configuration, original results, selected labels, alignment,
selection, VAD files and analysis source are SHA-256 pinned in the audit.
Downloaded audio and detailed scores remain outside Git.
