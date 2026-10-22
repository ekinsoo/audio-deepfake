# Range evaluation and the missing publisher reference

The verified PartialSpoof v1.2 train/dev audio archives were streamed without
extracting payloads to inspect member names. Neither contains RTTM or
`reco2dur`. The earlier checked protocol/label/VAD archive inventories also
contain no RTTM member names. No eval payload was interpreted.

The pinned upstream [metric instructions](https://github.com/nii-yamagishilab/PartialSpoof/blob/847347aaec6f65c3c6d2f17c63515b826b94feb3/metric/README.md)
expect `con_data/rttm_2cls_0sil_spoof`. The checked `rttm_tool.py` converts
RTTM to label vectors; it does not establish an exact inverse for the labels
we have. Publisher silence treatment and sub-10 ms timestamps remain unknown.
The publisher-reference gate therefore remains **unverified**.

## A separately named diagnostic

For a useful, reproducible range check, I converted contiguous zeros in the
unchanged publisher 10 ms binary vectors into spoof intervals. The final
interval clips to the actual WAV duration. VAD is never substituted for
binary truth. Selected train/dev references and exact durations stay under
ignored `data/partialspoof/selected_references_v1/`.

The actual hash-pinned **bounded patched upstream RangeEER** runs on these
derived references in the separate metric environment. Prediction resolution
is 20 ms (`--scale 2`), reference resolution 10 ms. Each worker has an external
120-second timeout, 1 GiB peak RAM limit and 2 GiB available system RAM floor.
Both model runs stopped their search on a repeated complete state after 55
iterations. Independent interval arithmetic reproduced FPR/FNR at each
selected threshold to within 1e-8. This is not exhaustive threshold search
or interpolated EER. [Compatibility findings](RANGE_EER_CONFORMANCE.md)
document the upstream errors and exact patches.

| Custom pilot diagnostic | LFCC baseline | Recording median/MAD |
| --- | ---: | ---: |
| Pooled 20 ms duration EER | 39.50% | 42.92% |
| Derived native-10 ms reference RangeEER | 39.38% | 42.97% |

These rows have different references and threshold selection rules; they
must not be interchanged. The range evaluator treats score equal to threshold
as spoof, while the project timelines use strict score < threshold. Both
range results use the same fixed 64 dev recordings. No official benchmark,
native-RTTM parity, paper reproduction or independent generalization is claimed.

The derived native reference contains 57.2025 s of spoof and 175.9401875 s
of bona fide duration. Minimum pooling to the pilot's 20 ms grid expands
spoof boundaries, producing 58.0525 / 175.0901875 s instead. This explains
why even the class-duration totals differ between the two diagnostics.

The three small upstream source files were fetched at the pinned commit and
their hashes verified; no new audio was downloaded. The pinned dependency
environment retains deprecation warnings from `pkg_resources` and pandas.
Those warnings are logged and are unrelated to classifier convergence.

## Reproduction

```powershell
# Default invocations print plans only.
.venv/Scripts/python.exe scripts/prepare_partial_references.py
.venv/Scripts/python.exe scripts/evaluate_localization_ranges.py
```

Completed audits are preserved; these scripts refuse to overwrite an existing
version. Reproduction of a new run needs separately versioned output paths
and recorded source/configuration changes. The metric runner uses the actual
base Python executable and explicitly selects the isolated metric packages,
so the original model environment is not changed.

Evidence: [reference audit](validation/localization_references_v1.json),
[range results and duration check](validation/localization_range_results_v1.json),
[first pilot](LOCALIZATION_RESULTS.md), [context plan](LOCALIZATION_CONTEXT_PLAN.md).
Cite [Zhang et al. (2023)](https://www.isca-archive.org/interspeech_2023/zhang23v_interspeech.html)
and [pyannote.metrics](https://pyannote.github.io/pyannote-metrics/) when discussing
the range evaluator.
