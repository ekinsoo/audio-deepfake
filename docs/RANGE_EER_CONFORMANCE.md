# RangeEER compatibility and conformance

Reviewed 10 October 2026. These are synthetic software checks, not localization
accuracy results. No audio, training data or eval labels are used.

The evaluator is pinned to PartialSpoof commit
`847347aaec6f65c3c6d2f17c63515b826b94feb3`. The exact original
[RangeEER.py](https://github.com/nii-yamagishilab/PartialSpoof/blob/847347aaec6f65c3c6d2f17c63515b826b94feb3/metric/RangeEER.py),
its full `sandbox.eval_asvspoof` dependency and the latter's `io_tools` import
are retained in ignored `.cache/partial-metric-env/upstream`. Their downloaded
bytes and SHA256 hashes, the separately generated compatibility diff, package
versions and executed cases are recorded in the
[audit](validation/range_eer_conformance.json). No metric function is
reimplemented or replaced by our interval diagnostic.

## Executed findings

The original code raises `NameError: name 'frr' is not defined` in
`cal_eer_by_scolab`. It also raises `UnboundLocalError` for `eer_ary` when the
initial threshold already converges and no search rows were written.
The first compatibility patch changes the three mistaken `frr` references to
the existing `fnr` variable and initializes `eer_ary` from the already collected
rows when no saved curve exists. The percentile search is unchanged in that
variant.

The actual upstream duration functions pass nine cases: perfect, inverted,
mixed, no spoof, false alarm with no spoof, all spoof, threshold ties, final
duration clipping and 20 ms scale conversion. The patched actual `main()`
also returns the expected values for four immediately terminating examples:
0, 1, 0.5 and 0.5 for perfect, inverted, mixed and tied predictions.

A harder mixed example does **not** finish within an external eight-second
timeout. Its last saved rows repeat threshold 0.4, FPR 0.75 and FNR 2/3.
The nonzero gap prevents the stopping condition from succeeding even after
the floating-point percentile search stops making progress. The exact
fixture and final rows are in the audit.

A second, explicitly named **bounded patched upstream variant** tracks the
complete six-value search state, not just the threshold. It stops only on a
repeated complete state or after 128 iterations, and prints the reason. A
repeated threshold while percentile bounds still change does not stop it.
The same fixture finishes after 54 iterations with `repeated_full_state`.
It returns the closest observed gap using the original selection expression:
FPR 3/4, FNR 2/3, averaged error 0.708333. Those duration counts also follow
directly from the three spoof bins and four bona fide bins in the fixture.
The selection includes the initial in-memory row, fixing upstream's omission
when only later iterations had been written to disk. It adds no threshold
interpolation or alternative metric formula. Original, typo-only and bounded
sources/diffs/hashes remain separate. The audit records the typo-only timeout
even though the bounded variant passes.

This clears a limited synthetic compatibility check, not paper reproduction
or native reference alignment. Real-data reporting must name this variant
and retain a process timeout. The upstream printed threshold is the last
visited threshold; use the explicitly printed `BoundedSearch selected_row`
when the selected threshold is needed.

## Interface rules

- Higher `Predict_Score_1` supports bona fide. Scores equal to the threshold
  predict spoof. Discrete tied examples can return an averaged error of 0.5
  even though FPR and FNR are not equal; no interpolation has been added.
- `--scale 2` means 20 ms: the CLI multiplies by 10 and the segment converter
  converts milliseconds to seconds. A final predicted segment is clipped to
  the `reco2dur` evaluation interval by the actual pyannote metric.
- Three-column score rows load without reference classes when
  `pred_cls=False`; segment-EER initialization needs four columns including
  native `True_Class` (0 spoof, 1 bona fide). Use four-column locally generated
  trusted NumPy files for the pilot. Upstream uses `allow_pickle=True`.
- Zero-denominator duration rates follow upstream conventions. A single-class
  fixture checks those conventions; it does not establish a meaningful EER.
- `--hyp_rttm` is still unimplemented. No alternative branch is silently used.

Real audio/native-label alignment now passes all 192 selected recordings;
see the [alignment audit](validation/localization_alignment.json). VAD label
0 remains unsuitable as a binary reference. The checked distribution has no
publisher RTTM, so native publisher-reference parity remains unverified.
The later [derived-reference diagnostic](details/LOCALIZATION_METRICS.md) evaluates
the actual 20 ms predictions against explicitly reconstructed native 10 ms
spoof ranges. It is separate from these synthetic checks and is not official
benchmark certification.

## Isolated reproduction

Use a separate Python 3.12 environment; never install these packages into the
pinned recording-detection environment. The principal compatible versions are
NumPy 1.26.4, pandas 2.2.3, SciPy 1.13.1, pyannote.metrics 3.2.1,
pyannote.core 5.0.0, pyannote.database 5.1.0, tqdm 4.67.1,
**python-rle 0.0.3**, and setuptools 80.9.0. The PyPI package named `RLE`
does not provide the required lowercase module and was removed. Setuptools
is pinned below 81 because this pyannote.database version imports the
deprecated `pkg_resources` API. pandas' whitespace-reader deprecation warning
is retained, not patched away. Full installed versions are in the audit.

```powershell
.cache/partial-metric-env/venv/Scripts/python.exe scripts/check_range_eer.py --fetch --write-audit
.cache/partial-metric-env/venv/Scripts/python.exe -m unittest discover -s tests -p test_range_eer.py -v
```

`--fetch` retrieves only the three pinned source files with a 5 MiB total
budget and 30-second per-file timeout. The tests need no network once sources
and dependencies exist. Dependency archives total about 113 MB; the isolated
installed environment is about 0.5 GB. No PyTorch or pyannote.audio is installed.

The code is mainly BSD-3-Clause; `eval_asvspoof` includes its own
CC BY-NC-SA 4.0 notice, retained with the original source. Cite both
[Zhang et al., Range-Based Equal Error Rate for Spoof Localization (2023)](https://www.isca-archive.org/interspeech_2023/zhang23v_interspeech.html)
and [Bredin, pyannote.metrics (2017)](https://pyannote.github.io/pyannote-metrics/)
when reporting this evaluation.
