# PartialSpoof protocol review

Reviewed 10 October 2026. This is preparation for a localization pilot, not a benchmark result. No audio download or training was performed for this review.

## Dataset choice and attribution

Use **PartialSpoof v1.2**, Zenodo record **5766198**, DOI **10.5281/zenodo.5766198**. The record still describes v1.3 detailed timestamps as a separate Google Drive download with the official release in preparation. Do not combine those labels with v1.2 without a separate version audit. [Dataset record](https://zenodo.org/records/5766198)

| Archive | Advertised size | Published MD5 |
| --- | ---: | --- |
| database_train.tar.gz | 2.0 GB | c4853ddd831e8e96b0e279fc0a512e7e |
| database_dev.tar.gz | 2.0 GB | ddd4cd3221b7210ac879f67452fb209e |
| database_eval.tar.gz | 5.8 GB | 79c7c834d0d9979ecd374a98a059ea19 |
| database_protocols.tar.gz | 5.4 MB | 699d81f020e4b7fa8f33747010e1cba8 |
| database_segment_labels_v1.2.tar.gz | 76.6 MB | c2bf6638e59ec7a5cf93c4a510fe4efe |
| database_vad.tar.gz | 10.2 MB | 95e77e19a1bb4f2e79ed138fd35621ad |

Sizes are rounded publisher values, not measured transfer sizes. [File listing](https://zenodo.org/records/5766198)

The dataset README specifies **CC BY 4.0** and also acknowledges **ODC-By** for the source ASVspoof 2019 database. Retain both attributions and cite the dataset papers. Original bona fide recordings are normalized ASVspoof 2019 audio. [Dataset README](https://zenodo.org/records/5766198/files/README_v1.2?download=1)

The paper's counts are 25,380 train (2,580 bona fide / 22,800 spoof), 24,844 dev (2,548 / 22,296), and 71,237 eval (7,355 / 63,882). Each PartialSpoof split was constructed from the corresponding ASVspoof 2019 LA split. Existing ASVspoof models therefore do **not** provide independent cross-dataset evidence here. Preserve official splits and keep eval closed. These are publication counts until checked against downloaded protocol files. The data's random substitutions do not establish realistic semantic manipulation or Turkish-language performance. [Database paper, Sections III-A–D](https://arxiv.org/html/2204.05177v3)

## Labels and time

The v1.2 segment arrays are dictionaries keyed by utterance ID; native labels are **0 = spoof, 1 = bona fide**. Base resolution is 10 ms. Coarser labels take the minimum over each group, including a remaining partial group. Timestamp VAD files contain `start_seconds end_seconds label`, with **2 = nonspeech** as an additional label. Nonspeech must not silently become a third class or a spoof target in a binary classifier. [Dataset README](https://zenodo.org/records/5766198/files/README_v1.2?download=1)

Project interval utilities use half-open `[start, end)` seconds, checked positive durations, and an explicit final-bin clipping rule. This is our interface contract, not a verified assertion about every stored label vector. Check native vector lengths against audio duration before selecting a floor/ceil rule. The upstream RTTM converter defaults to dropping the final incomplete frame (`skip_last=True`), so rounding must be documented. [Pinned RTTM helper](https://github.com/nii-yamagishilab/PartialSpoof/blob/847347aaec6f65c3c6d2f17c63515b826b94feb3/metric/rttm_tool.py)

## Evaluation interface

The authors recommend **RangeEER** for localization and utterance EER for recording detection. Also report point-based SegmentEER at a finer reference resolution than the model output where available. Thresholds chosen by EER sweeps are evaluation thresholds; choose the demo operating threshold on dev only. [Metric guidance](https://github.com/nii-yamagishilab/PartialSpoof/tree/847347aaec6f65c3c6d2f17c63515b826b94feb3/metric)

Live `git ls-remote` resolved upstream main to **847347aaec6f65c3c6d2f17c63515b826b94feb3**. The pinned `RangeEER.py` was fetched in memory; byte SHA256 is **8b9c65c20c89443dc36aa77c4862ce9e89f62a2fb48227096f37d44e22dee194**.

The source reads score arrays with `numpy.load(..., allow_pickle=True)`. Prefer per-recording rows `[uttid, score_0, score_1, true_class]` with binary native labels; three-column rows load, but default threshold initialization requires reference labels. Preserve frame order. **Higher score_1 supports bona fide**; threshold ties predict spoof. `--scale` uses **10 ms units**: **20 ms output requires `--scale 2`**. Supply spoof-only reference RTTM and `reco2dur`; default evaluation spans whole recordings. Only load locally generated trusted files.

Imports require NumPy, pandas, tqdm, SciPy, `rle`, pyannote.metrics/core/database, and the repository's `sandbox.eval_asvspoof` on PYTHONPATH. Source inspection found undefined `frr` in `cal_eer_by_scolab`, potentially undefined `eer_ary` when no search row is saved, and an unimplemented `--hyp_rttm` branch. These are inspection findings, not executed failures. A reviewed compatibility patch and synthetic conformance run are required before reporting official RangeEER. Keep the original source/hash and patch separately. [Pinned evaluator](https://github.com/nii-yamagishilab/PartialSpoof/blob/847347aaec6f65c3c6d2f17c63515b826b94feb3/metric/RangeEER.py)

Upstream code is mainly BSD-3-Clause; submodules have separate licenses. The original multi-resolution network is a different model from our proposed pilot. [Repository license guidance](https://github.com/nii-yamagishilab/PartialSpoof/blob/847347aaec6f65c3c6d2f17c63515b826b94feb3/README.md)

## Bounded next steps

1. Fetch only the small README/protocol metadata first, verify checksums, inspect names and split counts. Avoid the upstream downloader: it requests train, dev **and eval**. [Upstream downloader](https://github.com/nii-yamagishilab/PartialSpoof/blob/847347aaec6f65c3c6d2f17c63515b826b94feb3/01_download_database.sh)
2. Establish VAD-to-binary-reference conversion and final-frame alignment with synthetic cases: exact boundaries, incomplete last bin, no spoof, all spoof, missing IDs, and nonspeech. An interval confusion calculation is an engineering diagnostic until official conformance is established; do not label it RangeEER.
3. Inspect gzip archive member order and measure bytes required to reach selected audio. A small sample from a `.tar.gz` does not imply a small transfer. Freeze a transfer budget before any audio fetch; stop if it is exceeded. Record bytes and duration even after interruption.
4. Start with eight train / eight dev recordings exclusively as a loader/alignment smoke test. Freeze IDs before examining scores. Eval audio/labels remain unopened.
5. Follow the [pilot plan](LOCALIZATION_PILOT_PLAN.md): seed 45, 128 train / 64 dev recordings, one LFCC plus logistic-regression baseline, and at most 100,000 labelled training frames total. Validate the 20 ms output mapping before fitting. Freeze the source IDs, configuration, operating threshold policy and compute budget. Retain the existing full-recording experiment unchanged. A later context-normalization ablation is a separate experiment.

Open checks: actual archive layout and byte accessibility; native terminal-frame handling; upstream `rttm_2cls_0sil_spoof` construction/nonspeech treatment; a compatible pinned metric environment; reference/adapter conformance; source recording overlap beyond the known construction dependency. No localization performance is claimed yet.


## Local verification update

The [metadata audit](validation/partial_metadata.json) verifies all four
publisher MD5 checks: exactly 92,230,352 transferred bytes, no retry.
Actual official protocol counts agree with 25,380 train / 24,844 dev.
The [fixed selection](validation/partial_selection_v1.json) contains 128/64
balanced train/dev IDs, real protocol speaker IDs, 8+8 smoke IDs and ten
balanced dev timeline IDs. Selected labels are exported to a small ignored
JSON without pickle; export size and SHA256 are in the audit.

Do not derive binary references from VAD label 0. Native arrays are Unicode
'0'/'1', and VAD nonspeech can be either binary label. In CON_D_0003090,
VAD spoof duration is zero while the native vector has 77 spoof bins.
The [metadata figure](figures/partial_label_alignment.svg) shows this discrepancy;
it contains annotations, not model predictions. Of 192 selected recordings,
11 train and four dev have label counts different from ceil(VAD duration /
10 ms). Real audio headers and upstream binary RTTM remain necessary.

The [conformance review](RANGE_EER_CONFORMANCE.md) supersedes the earlier
source-only findings with executed failures and documented compatibility
variants. A bounded patched upstream search passed synthetic checks; this
is neither unmodified official evaluation nor real-data performance.
Eval metadata members were not extracted, deserialized or interpreted.
Published containers include them, so gzip scans may decompress their bytes
while skipping members. Eval audio is neither downloaded nor inspected.
