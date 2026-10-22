# ASVspoof 2019 metrics

Research date: 2026-10-09. The external-reference adapter is implemented in
`src/audio_deepfake/metrics.py`. It validates CM score IDs and delegates to the
hash-pinned official 2019 evaluator. Reference parity tests pass. The first
50-train / 50-dev model smoke run is recorded in
[baseline_smoke100.json](validation/baseline_smoke100.json); it is not a full benchmark.

## Primary references

The [official 2019 challenge page](https://www.asvspoof.org/index2019.html)
links [tDCF_python_v1.zip](https://www.asvspoof.org/asvspoof2019/tDCF_python_v1.zip).
We downloaded this small package and inspected `eval_metrics.py`,
`evaluate_tDCF_asvspoof19.py`, `README.txt` and `LICENSE`. The version is 1.0.
Use this 2019 implementation, not a later 2021/2024 formula. Also see the
[2019 evaluation plan](https://www.asvspoof.org/asvspoof2019/asvspoof2019_evaluation_plan.pdf).

Reference files stay under ignored `data/metadata/asvspoof2019/metrics/`.
No reference source was copied into our Python package. The evaluator's supplied
license is **CC BY-NC-SA 4.0**, distinct from the dataset license. Preserve its
notice when sharing reference material; do not describe it as MIT code.

| Reference | Observed SHA256 |
| --- | --- |
| Official Python ZIP | `0e60f66ac297cf6bd49539af5e7650d799e1aedb7b50496450aa12d4eb4adbaa` |
| `eval_metrics.py` | `93ea505282d329ed14537d977a74b2d1a25d161ac6767a16e0216a1a18391049` |
| `evaluate_tDCF_asvspoof19.py` | `642d69062771869ec0028578a607027cdf755499f831ab24f07bdab6f8195b17` |
| Package LICENSE | `3c2ac94fd3175166a850a52da4fe67b6a5a17f57151c44c29fabb81a644a2211` |
| Official LA dev ASV scores | `fe8114cf608c3062c141df5979892eba9664aa37a8c99e97157f129d679d3389` |

These hashes identify the observed downloads; they are not publisher-supplied
checksums and do not verify the full LA archive.

## Set up the external reference

After installing this project's Python dependencies, run:

```powershell
.venv\Scripts\python.exe scripts/fetch_metric_reference.py
.venv\Scripts\python.exe scripts/fetch_metric_reference.py --check-only
```

The setup downloads the 252,290-byte official evaluator ZIP and the 632,703-byte
official LA dev ASV score file, about 0.84 MiB combined. It does not download
audio or eval scores. The ZIP is pinned by size and SHA-256 before extraction.
Only `LICENSE`, `README.txt`, `eval_metrics.py`, and the two bundled example
score files under `scores/` are extracted, with individual size and SHA-256
checks. The license notice is installed before the evaluator. The package's
bytecode and old plotting driver are not needed.

All files stay in the ignored reference directory. Their source paths, exact
sizes and observed hashes are pinned in `scripts/fetch_metric_reference.py`.
Archive requests and redirects accept only the original ASVspoof HTTPS host
and archive path; the official dev ASV file uses the existing bounded Zenodo
metadata downloader. Each download is verified before replacing its destination.

Matching cached files skip the network. A changed cached file stops setup;
use `--overwrite` explicitly to repair it. `--check-only` makes no network
requests or writes and fails if any required file is missing or changed.
Fresh clones can run the input-contract tests without these external files;
the reference-dependent integration tests skip until setup is complete.
Keep the supplied CC BY-NC-SA 4.0 notice with the external evaluator and fixtures.

## EER and tied scores

Higher CM scores mean bona fide. `compute_det_curve` concatenates bona fide
scores before spoof scores, then uses a stable ascending mergesort. It starts
with miss rate 0, false acceptance rate 1, and threshold `minimum_score - 0.001`.
Each subsequent entry processes one sorted trial. Thresholds include duplicate
scores; the reference does not group ties.

EER selects the first index minimizing `abs(miss - false_accept)` and averages
the two rates at that index. It does not interpolate a crossing. Rates are
fractions; multiply by 100 only for percent display.

Tie handling matters: a single bona fide score 0 and a single spoof score 0
produce reference EER **1.0**, rather than 0.5. Intermediate tied-score entries
are not realizable by a single ordinary threshold. Keep this behavior explicit
for reference parity; do not silently replace it with another library's ROC/EER.
Operational threshold selection must be checked separately on dev data.

## ASV operating point and required files

The evaluator fixes ASV at its target-versus-nontarget EER threshold. It then
counts nontarget acceptance with `score >= threshold`, target rejection with
`score < threshold`, and spoof rejection with `score < threshold`. This literal
comparison can disagree with the EER grid at tied scores.

The matching files are documented by the original
[LA README](https://zenodo.org/api/records/6906306/files/LA.zip/container/LA/README.LA.txt):

| Split | Archive member |
| --- | --- |
| Dev | `LA/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.dev.gi.trl.scores.txt` |
| Eval | `LA/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt` |

Append the member path to
`https://zenodo.org/api/records/6906306/files/LA.zip/container/` for an individual
member endpoint. We fetched and inspected only the dev ASV file, using the
[direct dev endpoint](https://zenodo.org/api/records/6906306/files/LA.zip/container/LA/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.dev.gi.trl.scores.txt).
The eval path is README-derived and has not been fetched or independently tested.

ASV rows contain `source key score`, with key target/nontarget/spoof; these are
ASV trials, not one-to-one CM rows. The dev file has 29,548 rows: 1,484 target,
5,768 nontarget, 22,296 spoof. Its reference ASV EER is 0.02426530238400544,
threshold -3.548998. The resulting rates are Pfa=0.02444521497919556,
Pmiss=0.02425876010781671, Pmiss_spoof=0.23972909939002512.

The package's bundled example ASV uses older `SS_1`/`VC_1` labels. It is useful
as a numerical fixture, **not** as a substitute for the LA dev/eval ASV file.
The driver expects CM rows `utterance_id source bonafide_or_spoof score`.
It does not verify completeness or align scores by IDs, so our wrapper must.

## t-DCF v1 settings

Pspoof=0.05, Ptar=0.9405, Pnon=0.0095. Cmiss_asv=Cmiss_cm=1;
Cfa_asv=Cfa_cm=10. Using ASV rates as fractions:

```text
C1 = Ptar * (Cmiss_cm - Cmiss_asv * Pmiss_asv)
     - Pnon * Cfa_asv * Pfa_asv
C2 = Cfa_cm * Pspoof * (1 - Pmiss_spoof_asv)
normalized_tDCF(s) = (C1 * Pmiss_cm(s) + C2 * Pfa_cm(s)) / min(C1, C2)
min_tDCF = min(normalized_tDCF)
```

There is no additive C0 term in this downloaded v1 code. Later formulations
must not be swapped in without changing the reported metric. The reference
rejects non-finite CM scores, fewer than three unique CM scores, missing spoof
ASV rates and negative C1/C2. It does not robustly guard empty classes or a zero
normalizer; our adapter rejects these explicitly.

## Validation plan

1. The adapter calls the unmodified reference rather than implementing another
   formula. Tests cover its tied-score behavior and invalid inputs. Future
   alternative implementations must compare complete curves and thresholds too.
2. Use the bundled example as a golden fixture: EER=0.046701084744563005,
   min_tDCF=0.1104983661284351. These are reference-example checks, not model results.
3. Check official LA dev ASV rates against the values above. Require complete,
   unique finite CM scores joined to the selected protocol by utterance ID.
4. Test rejected inputs and score direction. Keep pilot-subset EER/t-DCF labeled
   as pilot diagnostics; a subset is not an official benchmark result.
5. Freeze experiment choices before eval. Use the matching eval ASV file only
   when final evaluation is authorized by the experiment plan.

Nine adapter tests passed locally, including the bundled numerical fixture,
score reordering, strict ID matching, invalid classes/scores, hash-pinned source
loading, tied EER and undefined t-DCF. Reference-dependent integration tests
skip when the external source or fixtures are absent; input-contract tests do
not require them. ASV file/split matching remains the caller's responsibility;
the adapter records its hash and split but does not certify their pairing.
The train/dev runner in `scripts/run_baseline.py` additionally pins the official
LA dev ASV file hash before fitting; it does not accept eval inputs.

The old plotting driver uses removed `np.float` and imports matplotlib. We ran
only its unmodified metric module directly with current NumPy; no dependency
change was needed. Observed checks are saved in ignored `reference_checks.json`.
