# Research and progress reports

## 2026-10-09 — Initial scope research

Two parallel research agents reviewed full-recording detection and partial manipulation localization. The findings below are source checks and experiment plans, not benchmark results. No dataset or model download, training, or benchmark was run during this research stage. Scope A has since been selected and local setup is complete; validation is recorded at the end of this report.

### Decision

Start with **A: ASVspoof 2019 LA + CPU LFCC-GMM + one narrowband robustness experiment**. It fits the local hardware and keeps the first data and evaluation pipeline small. Move to **B: PartialSpoof + a small temporal classifier** after that. B offers a useful timeline demo but needs more work on label alignment and metrics. Do not train both scopes at once.

The six-step plan is in [docs/START_PLAN.md](docs/START_PLAN.md). Colab is an option for later GPU models; resource availability varies, so measure the session hardware and support resume before a long run. [Official FAQ](https://research.google.com/colaboratory/faq.html).

### Initial hardware check

Measured with Windows hardware queries and `GlobalMemoryStatusEx` on 2026-10-09. Available memory and disk space are snapshots, not permanent limits.

| Resource | Observation | Effect |
|---|---|---|
| CPU | Intel Core i7-1195G7 @ 2.90 GHz; 4 physical / 8 logical cores | Start with CPU features and a small classical model |
| GPU | Intel Iris Xe Graphics; driver 32.0.101.7085; no NVIDIA device observed | Do not plan local CUDA training |
| Reported GPU memory | Windows AdapterRAM reports about 2 GiB | This is not a reliable measure of dedicated training VRAM on an integrated GPU |
| RAM | 15.71 GiB total; 2.82 GiB available, 82% load | Process files one at a time; recheck before the pilot |
| System drive | 80.32 GiB free, about 458.3 GiB total | The full ASVspoof 5 distribution does not fit |
| Secondary drive | 76.30 GiB free reported | Physical independence was not checked; do not add capacities together |
| Python, initial check | The default command failed; another working installation had not yet been checked | Resolve during setup |
| Repository, initial check | Only planning documents existed; the folder was not a Git repository | Add ignore rules before Git initialization |

The Windows GPU inventory was used rather than treating a missing `nvidia-smi` command as proof of no GPU. GB denotes decimal archive size; GiB denotes binary local measurements.

### Parallel research: full-recording detection

The full-detection agent checked datasets, licenses, official protocols, and baseline requirements.

**ASVspoof 2019 LA:** Zenodo v1 lists `LA.zip` at **7.6 GB**, MD5 `30c98f11d8b2bc21f2c257bfd78bb5c5`. The distribution README states ODC-By; its LICENSE distinguishes database rights from individual content rights. Preserve official train/dev/eval. The README lists 20/10/48 speakers. Later direct protocol inspection clarified this: these are the spoof-side speaker counts; all CM rows contain 20/20/67 unique speaker IDs. The initial apparent count conflict was a difference in populations being counted, not a reason to change the split. [Data](https://zenodo.org/records/6906306), [README](https://zenodo.org/records/6906306/files/README.txt?download=1), [license](https://zenodo.org/records/6906306/files/LICENSE_text.txt?download=1). See the metadata milestone below.

One score is required per recording, with higher scores meaning bona fide. The main metric is **minimum normalized t-DCF**, with **EER** secondary. The official LFCC-GMM/CQCC-GMM baselines use MATLAB. Train has 25,380 recordings and dev has 24,844; eval stays out of development. Adding an external pretrained speech model changes the original condition. Check feature and scoring equivalence before describing a Python adaptation as an exact reproduction. [Evaluation plan](https://www.asvspoof.org/asvspoof2019/asvspoof2019_evaluation_plan.pdf), [official baseline links](https://www.asvspoof.org/index2019.html).

**ASVspoof 5:** Zenodo v0.1 totals **142.3 GB**: about 37.5 GB train, 20 GB dev, and 84.8 GB eval. Counts are 182,357 train, 140,950 Track 1 dev, and 680,774 eval recordings. Data is ODC-By, with bona fide content also under CC BY 4.0. Protocol files have a `.tsv` extension but space-separated fields; the README column count is also inconsistent. Validate the parser against the files. An archive part is not a complete split. [Data](https://zenodo.org/records/14498691), [README and license](https://zenodo.org/records/14498691/files/README.txt?download=1).

ASVspoof 5 Track 1 uses **minDCF** as its main metric, with actDCF/Cllr/EER secondary; spoof prior 0.05, miss cost 1, false-alarm cost 10. Do not merge train and dev; dev can be used for calibration. The open condition restricts MLS English, LibriLight, MUSAN speech, and pretrained sources with uncertain overlap. Testing a model trained only on 2019 is not a closed Track 1 result. [Phase 2 plan v0.6](https://www.asvspoof.org/file/ASVspoof5___Evaluation_Plan_Phase2.pdf), [official scoring package](https://github.com/asvspoof-challenge/asvspoof5/tree/main/evaluation-package).

The ASVspoof 5 AASIST example uses Python 3.9/PyTorch 1.13.1/CUDA 11.7, batch 24, 100 epochs, and 64,600 samples (about 4.04 s) per input. RawNet2 recommends an older Python 3.7/PyTorch 1.8 setup. These are not minimum VRAM or runtime guarantees. An explicit root LICENSE was not confirmed; check individual code/subdirectory licenses before reuse. Both models are deferred. [AASIST](https://github.com/asvspoof-challenge/asvspoof5/blob/main/Baseline-AASIST/README.md), [config](https://github.com/asvspoof-challenge/asvspoof5/blob/main/Baseline-AASIST/config/AASIST_ASVspoof5.conf), [RawNet2](https://github.com/asvspoof-challenge/asvspoof5/blob/main/Baseline-RawNet2/README.md).

### Parallel research: partial localization

The localization agent checked PartialSpoof, LlamaPartialSpoof, labels, metrics, and baseline dependencies.

**PartialSpoof v1.2:** about **10 GB** compressed: 2.0/2.0/5.8 GB train/dev/eval. Data is CC BY 4.0 and also notes ODC-By attribution for source ASVspoof 2019 material. Base labels are 10 ms; `0=spoof`, `1=bona fide`, with a separate non-speech VAD label. A coarse frame is spoof if any part is spoof. Do not silently mix v1.3 detailed labels with v1.2. [Data](https://zenodo.org/records/5766198), [README_v1.2](https://zenodo.org/records/5766198/files/README_v1.2?download=1).

Split sizes are 25,380/24,844/71,237 recordings, about 24.25/24.34/67.68 hours. The data joins VAD chunks from ASVspoof 2019; it does not cover all meaningful word replacements in real speech. Check source-recording overlap before carrying models between A and B. Their shared source means this is not independent external generalization. [Dataset paper](https://arxiv.org/html/2204.05177v3).

The authors recommend **Utterance EER** for detection and **RangeEER** for localization. SegmentEER is also available. Report the 20 ms output resolution and, where possible, a finer reference. RangeEER needs reference RTTM, recording duration, and time-aligned scores; follow the official evaluator's `scale` units. EER threshold sweeps are evaluation tools; select demo thresholds on dev only. [Metric README](https://github.com/nii-yamagishilab/PartialSpoof/tree/main/metric).

The main code license is BSD-3-Clause; submodules have separate licenses. The original baseline needs s3prl/SSL, with an example training limit of 5,000 epochs. MIT-licensed `MultiResoModel-Simple` is an alternative but states that it is not a full replication; old fairseq/torch versions may complicate Windows setup. The planned first B model will not be described as a reproduction of these baselines. [Original repo](https://github.com/nii-yamagishilab/PartialSpoof), [baseline](https://github.com/nii-yamagishilab/PartialSpoof/tree/main/03multireso), [alternative](https://github.com/hieuthi/MultiResoModel-Simple), [dependencies](https://raw.githubusercontent.com/hieuthi/MultiResoModel-Simple/main/requirements.txt).

**LlamaPartialSpoof v1.0.b:** **28.2 GB**, CC BY 4.0; 15.4 GB crossfade collection plus a 12.8 GB cut/paste-overlap/add collection. Originally eval-only; a later continual-learning speaker split must not be confused with that external evaluation protocol. The original paper's 130 hours and counts are not the sum of both archives. Its LibriTTS-based English data does not establish Turkish generalization. No download is planned for the first milestone. [Data](https://zenodo.org/records/14214149), [FAQ/project](https://github.com/hieuthi/LlamaPartialSpoof), [paper](https://arxiv.org/html/2409.14743v2). Millisecond EER is different from RangeEER; name the evaluator and metric explicitly in external tests. [Metric repository](https://github.com/hieuthi/partialspoof-metrics).

### Scope comparison

These are **pilot limits**, not measured requirements or runtime/performance predictions. Pilot results will not be presented as full official results.

| Item | A — full-recording detection | B — partial localization |
|---|---|---|
| Data | 2019 LA, 7.6 GB archive; fixed 2,000 train / 1,000 dev pilot | PartialSpoof v1.2, about 10 GB total; fixed 2,000 train / 1,000 dev pilot |
| First model | LFCC + two diagonal GMMs; start with 32 mixtures | LFCC at 20 ms steps + frame logistic regression; then a small 1D TCN |
| Training | At most 100,000 class-balanced train frames; mean log-likelihood ratio per recording | Sample frames from train only; if trying TCN, 2 s windows, batch 4, at most 5 pilot epochs |
| Compute | CPU, 2–4 workers, target peak process RAM ≤2 GiB; measure first 100 recordings | CPU, 2 workers, target peak RAM ≤2 GiB; measure 100 recordings and label alignment |
| Disk budget | 25 GiB for archive, extraction, and cache | Start with train/dev; 35 GiB planning budget for full distribution |
| Metrics | Clean eval: min t-DCF + EER; separate stress: EER and bona fide false rejection at fixed dev threshold | Main: RangeEER; secondary: 20 ms SegmentEER and Utterance EER; missed spoof duration / false-alarm duration at dev threshold |
| First delivery | Manifests, features, baseline scores, scorer check, clean/narrowband table, 10 error cases | Manifests, RTTM/label reader, score-time adapter, RangeEER, reference/predicted timelines for 10 recordings |
| Main risk | Overstating generalization from older synthetic speech to current real channels | Learning splice boundaries, missing short spoof spans, alignment mistakes, synthetic splice bias |

Extracted sizes are unverified: 25/35 GiB are budgets, not guarantees. Stop and revise if exceeded. A small training subset does not guarantee a small download; check selective archive access. Aim for about 5 GiB available RAM before the pilot, or reduce samples and use streaming. Measure extraction runtime, peak RAM, feature storage per audio hour, and disk cost on 100 recordings before estimating a full run. Do not load all audio/features into memory.

### Small experiment ideas

**A1 — score consistency under a transformation (selected first experiment).** Score the original audio and its 16→8→16 kHz version with the same model. Hypothesis: score differences help flag fragile predictions. Compare original score / average score / dev calibration using average and absolute difference. Start with score analysis without retraining. If abstention is added, compare error at equal coverage and plot risk–coverage; removing examples and reporting lower EER is insufficient. The transformation may remove spoof cues, so examine bona fide false alarms. This is a custom experiment separate from the clean official benchmark. One transformation cannot support a broad robustness claim.

**B1 — recording context.** Add each frame's difference from the recording median and MAD to local LFCC features. Compare local / local+median / local+median+MAD. Hypothesis: this reduces shared speaker/channel effects and exposes short local changes. It uses the full recording, so the demo is offline. A spoof-majority recording may give a misleading median; split results by spoof fraction and segment duration. Use the same recording-level feature construction during training.

**B2 — duration-aware postprocessing (after B1).** Compare raw scores / median smoothing / a pair of dev-selected hysteresis thresholds. No new model is needed. Measure removal of short fake spans and incorrect merging of nearby spans. Test B1 and B2 separately so their effects are clear.

These are practical hypotheses. Novelty and performance gains are unverified; any contribution claim needs controlled ablations and a literature check.

### Experiment rules and open work

Keep official train/dev/eval unchanged. Save pilot manifests with class/attack distributions and seed. Open eval only after model, transformation, and threshold choices are fixed. Fit normalization and select GMM frames on train; windows and derived versions of a recording must stay in its split. Proposed seeds are 42 for the pilot and 42/43/44 for the main comparison. Record data version/checksum, code commit, environment, config, score direction, metric version, transformation parameters, runtime/RAM, and errors. Keep public data and large weights outside Git; private audio and company data are excluded.

Open work and restart order are in [backlog.md](backlog.md). ONNX, an API, and full SSL fine-tuning are deferred. Performance, demo latency, Turkish generalization, and forensic reliability have not been measured.

### Milestone summary

2026-10-09: Initial research completed. Local hardware is i7-1195G7, Iris Xe, 15.7 GiB RAM, and about 80.3 GiB free system-drive space. A CPU pilot was selected: ASVspoof 2019 LA, LFCC-GMM, and a narrowband score-consistency ablation. PartialSpoof v1.2 localization with RangeEER is the next milestone. The 142.3 GB ASVspoof 5 distribution is deferred. No dataset download or training has started. Sources: [2019 data](https://zenodo.org/records/6906306), [PartialSpoof metrics](https://github.com/nii-yamagishilab/PartialSpoof/tree/main/metric). This repository is the implementation source; open work is in [backlog.md](backlog.md). This summary has not been written to external notes.

## 2026-10-09 — Step 1: local setup (complete)

Scope A is confirmed. The first implementation adds a Python package, a TOML pilot plan, and CLI commands for environment inspection, audio inspection, and a temporary FLAC round-trip check. Runtime dependencies are NumPy, SciPy, SoundFile, and scikit-learn. No GPU framework is needed yet. Public documentation uses simple English; personal biography and private paths were removed. A local Git repository on `main` was initialized; nothing was published.

Validation completed in an isolated environment:

- Python 3.12.14 on Windows 11; NumPy 2.5.3, SciPy 1.18.1, SoundFile 0.13.1, scikit-learn 1.9.1, libsndfile 1.2.2.
- `python -m audio_deepfake smoke`: passed. A generated 4,000-sample mono tone at 16 kHz survived a PCM16 FLAC round trip. Maximum absolute error was 0.0000152587890625, within the quantization tolerance.
- `python -m unittest discover -s tests -v`: 2 tests passed, including four rejection cases: stereo, wrong sample rate, empty audio, and non-finite samples.
- `python -m pip check`: no broken requirements.
- Git ignore checks passed for the virtual environment, local cache, dataset audio, model artifacts and `.env`.
- A separate agent reviewed the code, config and README; no actionable step-1 issue was found.

Exact resolved runtime dependencies are saved in [requirements/windows-py312.txt](requirements/windows-py312.txt). This is a local snapshot, not a verified Colab lock. The environment uses an app-provided Python installation as its base; on another computer, create a new environment from a normal Python installation rather than copying `.venv`.

Dataset manifests, LFCC features, model training and benchmark scoring remain future steps; no performance result is claimed. No dataset or pretrained model was downloaded. Next: verify protocol files and build split manifests before the measured data pilot.

Milestone summary: the local project now installs and passes FLAC and strict audio-input checks. The first research question remains narrowband score consistency on [ASVspoof 2019 LA](https://zenodo.org/records/6906306). This completes setup, not the detector. No external notes were updated.

## 2026-10-09 — Step 2: protocol manifests and pilot lists

Three agents split the work: primary-source protocol research, manifest implementation/tests, and an independent leakage review. The lead integrated the CLI, tested the real metadata and fixed an output-overwrite edge case from review. Public explanations remain simple English.

### Retrieved and checked

The three original CM key files were fetched as individual members of the [official Zenodo LA archive](https://zenodo.org/records/6906306), with three README/license text files. Their total size is 4,058,293 bytes. A small official baseline source archive was also inspected; it contains no full CM keys. No audio, full `LA.zip`, or pretrained model was downloaded and no training ran.

Source URLs, byte sizes, observed SHA256 hashes and expected counts are pinned in [the dataset specification](configs/datasets/asvspoof2019_la.json). [Data protocol notes](docs/DATA_PROTOCOL.md) explain the schema and licenses. Individual member hashes do not establish that the entire archive passed its published MD5 check.

| Split | Full records | Bona fide | Spoof | Total unique speaker IDs | Spoof speaker IDs |
|---|---:|---:|---:|---:|---:|
| Train | 25,380 | 2,580 | 22,800 | 20 | 20 |
| Dev | 24,844 | 2,548 | 22,296 | 20 | 10 |
| Eval | 71,237 | 7,355 | 63,882 | 67 | 48 |

All 121,461 rows passed schema, label/attack and duplicate-ID validation. Cross-split recording and speaker intersections were empty. Measured file hashes, sizes, counts and class-specific speaker counts matched the pinned specification. The small audit is saved in [docs/validation/asvspoof2019_metadata.json](docs/validation/asvspoof2019_metadata.json); full CSVs stay in ignored `outputs/manifests/asvspoof2019-la-seed42/`.

Speaker-count clarification: the README's 20/10/48 summary matches the spoof-side population. Counting all CM rows gives 20/20/67. This resolves the apparent discrepancy reported during initial research; definitions must accompany these numbers.

### Pilot policy

Full manifests preserve official protocol order. Separate pilots contain 1,000 bona fide + 1,000 spoof train rows, and 500 + 500 dev rows. Spoof quotas are proportional by attack A01–A06, using largest remainders with attack-ID tie breaking. SHA256 ranks the seed, split and recording ID, so input row reordering does not change selected IDs. Seed is 42. Exact manifest hashes and distributions are saved in the audit.

The pilot deliberately changes class prevalence and is not a full benchmark or deployment-prevalence estimate. It gives more bona fide examples for early failure analysis. Eval is never eligible for pilot selection; its metadata was inspected only for split integrity. No attack or speaker label will be a model input.

### Validation and limits

Final verification: all **16 unittest tests passed**. Offline downloader tests cover matching caches, checksum/size failures, protected existing files, forbidden URLs/redirects and network-free checks. `scripts/fetch_protocols.py --check-only` verified all six real cached text files. Exported pilot CSVs also passed record counts, split membership and train/dev recording/speaker disjointness checks. Git ignore rules exclude raw metadata and generated manifests.

The manifest module rejects malformed protocol rows, invalid split/attack pairs, duplicate IDs and cross-split speakers. It exports only canonical relative audio paths. The CLI validates before export, protects existing outputs and prevents stale eval metadata from being left after a train/dev-only overwrite. Optional audio checks resolve paths under the provided root and test file existence only.

The independent reviewer found no further blocking correctness issue. Full-split `--audio-root` checking is intentionally different from selective pilot audio readiness; the latter remains a follow-up. No recording is currently claimed available or audio-valid. `official_distribution_verified` stays false in the audit even though metadata checks pass.

Next: choose bounded pilot audio access after a fresh RAM/disk check, then measure 100 train/dev recordings. LFCC, EER/min t-DCF integration, training and robustness results remain open in [backlog.md](backlog.md).

Milestone summary: ASVspoof 2019 LA protocol metadata and balanced train/dev pilot lists are ready. All 121,461 official rows were audited without audio or training; split IDs and speakers are disjoint. Sources: [official dataset](https://zenodo.org/records/6906306), [original LA README](https://zenodo.org/api/records/6906306/files/LA.zip/container/LA/README.LA.txt). Code and validation are in this repository; no external notes were updated.

## 2026-10-09 — Step 3: 100-record audio and resource pilot

The audio, feature and review agents split the work; the lead integrated the profiler and checked the results. When the account limit interrupted agent supervision, the bounded downloader finished in the background. On resuming, all files and the journal were audited before profiling. No recording was replaced or selected based on its content.

### Audio and resource checks

Before download, available RAM was about 2.94 GiB and free system disk about 78.94 GiB. We chose small archive-member requests, not the 7.6 GB archive. The frozen sample contains 50 train and 50 dev recordings, each with 25 bona fide and 25 spoof. It is a subset of the existing seed-42 pilots. All 100 files passed 16 kHz mono PCM16 FLAC, non-empty/finite audio, size and observed SHA256 checks. Saved audio is exactly **6,654,382 bytes (6.35 MiB)**; no partial files remain. Eval audio was not fetched.

The downloader defaults to one worker and supports at most two, a 2 MiB file limit and a shared 32 MiB recorded transfer budget. The first two files confirmed access before the rest. High server latency led to two workers, with bounded batches and a locked byte counter. Initial sandbox DNS failures were retained in the journal; there were no source-file validation failures. A deliberately interrupted earlier process did not save all elapsed time, so recorded download duration is a lower bound and transfer accounting may omit unjournaled partial traffic. Saved file sizes and hashes were rechecked exactly. [Download audit](docs/validation/audio_download_100.json).

LFCC uses 20 ms frames, 10 ms hop, FFT 512, 20 linear filters, c0–c19, pre-emphasis 0.97 and the reference's unnormalized three-frame delta/double-delta, producing 60 float32 values per frame. Original audio is unchanged. Short input and final-frame padding are explicit. [Feature conventions and limits](docs/LFCC.md). MATLAB output parity has not been established; this remains a reference-inspired adaptation.

### Measured CPU run

Run: 2026-10-09 at 21:00 Europe/Istanbul. Fresh process on i7-1195G7, one worker and one native thread. Hash checks happen before the processing timer; OS file cache was not controlled.

| Measurement | Observed value |
|---|---:|
| Recordings | 100 |
| Total audio duration | 366.993 s |
| LFCC computation | 0.260829 s |
| Audio loading | 0.093163 s |
| Processing loop wall time | 0.383299 s |
| LFCC time / audio duration | 0.000711 |
| Per-file LFCC p50 / p95 | 2.30 / 5.25 ms |
| Peak process working set | 118.125 MiB |
| Feature arrays if cached | 8.39 MiB |

Peak memory is a process-lifetime high-water mark including imports and preflight, not a measurement of GMM training memory. Feature arrays were discarded after each recording. Neither inference speed nor detection accuracy was measured.

The [machine-readable run](docs/validation/audio_pilot_100.json) records CPU/environment, native thread counts, source/config/selection hashes, timestamps and limitations. The [exact measured config](docs/validation/pilot_profile_100.toml) preserves its recorded hash; the current pilot config only has an updated status comment. Per-record details remain in ignored `outputs/profile-audio-100/`.

Split-weighted sample means suggest roughly **7.83 s of LFCC computation**, **253 MiB of float32 features** and **192 MiB of compressed audio** for the 2,000 train + 1,000 dev pilot. These are rough extrapolations from 50 files per split, not measured full-pilot costs. They exclude download latency, cache file overhead, training and scoring. Do not scale streaming peak memory linearly by file count.

Decision: keep the first LFCC-GMM pilot on local CPU. Colab GPU is unnecessary for this feature stage. The main access bottleneck was individual-member network latency; select a measured strategy for the remaining 2,900 files before expanding the downloader limits. No full archive or training was started.

### Evaluation reference and checks

The metric agent verified the small official 2019 `tDCF_python_v1` source and LA dev ASV file, and implemented a strict score-ID adapter around the external reference. Its source is **CC BY-NC-SA 4.0**, remains ignored with its license and was not copied into our package. The adapter rejects missing/extra/duplicate CM IDs and invalid scores, and records ASV/reference hashes. Matching the ASV file to the evaluated split remains the caller's responsibility. [Metric reference and setup notes](docs/METRICS.md).

Reference tests cover the original grid-based EER, surprising tied-score behavior, the v1 t-DCF formula without C0, and the bundled numerical fixture. The bundled fixture's EER/min t-DCF are reference-validation numbers, not results from our detector. Some old fixture IDs repeat; the adapter test assigns unique synthetic IDs without changing scores or labels.

Final verification: **43 tests passed**, including all nine metric-adapter tests; `pip check` reported no broken requirements. A separate review found no blocking issue. Dataset audio, reference sources, journals and detailed measurements remain outside Git. Small public audits and simple English explanations are committed-ready project documents. No GitHub publication or external-note write occurred.

Milestone summary: the real 100-record CPU pilot is validated; LFCC and reference metric checks are ready. Feature extraction fits the laptop comfortably in this measured run. Next: expand access only to the fixed pilot audio, build bounded-frame LFCC-GMM training, then evaluate on dev. Sources: [ASVspoof 2019 LA](https://zenodo.org/records/6906306), [official evaluator](https://www.asvspoof.org/asvspoof2019/tDCF_python_v1.zip).

## 2026-10-09 — First trained model and pilot access plan

### Parallel work and decisions

Three agents worked on separate parts, with integration and the real training run handled by the lead agent:

- **Data access:** verified the official archive API size (7,640,952,520 bytes) and published MD5 (`30c98f11d8b2bc21f2c257bfd78bb5c5`); implemented a resumable downloader and selective pilot extraction. An 8 MiB HTTP range probe took 6.33 seconds and was discarded. The roughly 96-minute linear extrapolation includes connection overhead and is not a reliable sustained download estimate. No full archive was started.
- **Baseline implementation:** built the two diagonal GMMs, deterministic train-only frame reservoir, guarded experiment runner, serialization and tests. Config, input/source hashes, package versions, convergence and resource measurements are saved per run.
- **Independent review:** checked leakage boundaries, score direction and the official MATLAB baseline. The reference uses 512 mixtures per class, 10 maximum iterations and all train frames; our smaller sklearn model is an adaptation. After training, this agent recomputed metrics from saved scores and verified artifact hashes and complete dev IDs.

Fresh resource check: about 75.9 GiB free disk and 2.32 GiB available RAM out of 15.71 GiB. The archive preparation plan requires roughly 29.1 GiB free, including a 20 GiB reserve and 2 GiB extraction allowance. It passed in plan mode. The implementation uses local CPU, sequential class fits and one native numerical thread. No GPU session or long training was started.

The separate preparation command preserves the existing smoke downloader's 100-record / 32 MiB caps. It pins the two pilot CSV hashes, checks their class balance, resumes exact byte ranges, verifies published archive MD5 and extracts only the fixed 2,000 train / 1,000 dev files. The archive contains eval bytes, but eval is not extracted for modeling. Existing cached audio must match its journal and archive member. See [pilot data preparation](docs/PILOT_DATA.md).

### Measured smoke result

The run used [configs/smoke.toml](configs/smoke.toml), seed 42, 8 diagonal components per class and at most 10,000 frames per class. Actual training input was **8,848 bona fide + 9,704 spoof frames** from 50 train recordings. All 50 dev recordings were scored. Each split has 25 bona fide and 25 spoof recordings; no train/dev speaker overlap is allowed. There is no feature normalization, dev fitting or attack/speaker feature input.

| Measurement | Observed value |
| --- | ---: |
| Train / dev recordings | 50 / 50 |
| Dev EER | 12% |
| Dev minimum normalized t-DCF v1 | 0.16 |
| Training, including feature extraction and sampling | 2.704 s |
| Dev scoring, including feature extraction | 1.845 s |
| Metric computation | 0.176 s |
| Total processing wall time | 4.725 s |
| Peak process working set | 139.49 MiB |
| Bona fide / spoof EM iterations | 18 / 23 |
| Convergence / warnings | Both converged / none |

These are **small smoke diagnostics, not official benchmark results**. Thresholds for EER and min t-DCF are selected on this same small dev set. The counts per attack are too small to support generalization claims. The run is not a modern-generator, deployment or localization evaluation.

Processing time excludes imports, preflight hash reads and artifact writing; filesystem cache was not controlled. Peak memory is the process-lifetime high-water mark, not an isolated allocation measurement. This run cannot determine full-pilot training cost. Individual audio hashes are observed download hashes; the full archive MD5 remains unverified locally.

At the literal dev EER threshold (`score >= 0.831485` means bona fide), there were 3 rejected bona fide recordings and 4 accepted spoof recordings: three A04 and one A06. One spoof lies exactly at the reference threshold. The official EER grid processes individual sorted trials, so literal operational counts at its threshold need not equal the grid EER. IDs and exact scores are in the [public run audit and failure cases](docs/validation/baseline_smoke100.json). This is a diagnostic threshold, not a selected deployment rule.

The local artifacts are under ignored `outputs/baseline-smoke100-seed42/`: trusted local `model.joblib`, `scores.csv`, `config.json` and `summary.json`. Public audit includes configuration, seed, environment, source/input/output hashes and limitations. Source hashes identify this uncommitted implementation; there is no experiment Git commit yet. No audio, model weights or external evaluator source is included in public documents.

### Validation and next step

All **66 tests passed**, including training boundaries, deterministic sampling, score direction, artifact round trip, archive resume/checksum protection and experiment preflight. `pip check` found no broken requirements. Independent recomputation exactly matched both saved metrics; all 50 scores were unique by ID, finite, complete and disjoint from training IDs. Git ignore checks passed for audio, archives, local models and environment files.

Next: prepare the remaining 2,900 audio files using the explicit archive workflow, then export/pin a combined 3,000-record selection and wire its journal into `configs/pilot.toml`. Run the planned 32-component / 100k-frame pilot before comparing original and 16→8→16 kHz scores. The research contribution remains the transformation-score stability question; this smoke result alone is not the portfolio's final claim.

Milestone summary for EkinOS: the first public-data LFCC-GMM runs end to end on CPU with reproducible artifacts and independent metric verification. A tiny 50/50 train/dev check produced EER 12% and min t-DCF 0.16; it is not a benchmark result. Training took 2.70 s, with 139.49 MiB peak process memory. The full pilot audio and robustness experiment are still pending. Sources: [ASVspoof 2019 LA](https://zenodo.org/records/6906306), [official 2019 evaluator](https://www.asvspoof.org/asvspoof2019/tDCF_python_v1.zip), [project run audit](docs/validation/baseline_smoke100.json). No personal vault or GitHub publication was performed.

## 2026-10-09 — Complete pilot and first narrowband result

### Parallel work

The lead agent integrated the changes and ran data preparation, training and the experiment. Three agents handled separate tasks:

- **Data access and integrity:** hardened interrupted extraction recovery, added bounded parallel archive ranges and official mirror support, prepared the external metric setup command, and independently audited the completed archive and selected audio.
- **Experiment implementation and documentation:** wired the pinned 3,000-record selection, implemented the fixed narrowband transform and speaker-separated reliability helpers, built the guarded experiment runner, and documented reproduction commands.
- **Independent review:** checked implementation boundaries, then independently recomputed actual metrics, partition, threshold, error rankings and failure selection. It also rescored three actual recordings in both conditions with the saved model.

The final implementation suite passed **128 tests in 25.815 seconds**; `pip check` found no broken requirements. No source changes affecting inference or evaluation followed that check. Documentation, dataset local-state metadata and a Git ignore exception were updated afterward. The exception makes the small `configs/datasets/*.json` source metadata available to Git; it was previously hidden by the general `datasets/` rule. Audio, archives, model files, evaluator source, environments and caches remain ignored.

### Data preparation and transport

Before downloading, the laptop had about 75.9 GiB free disk and 1.89 GiB available RAM out of 15.71 GiB. The earlier hardware inspection found an Intel Iris Xe, with no NVIDIA GPU. A new management-interface hardware query was denied, so we used the working Python/Windows memory probe and did not infer a new GPU measurement. Before training, available RAM was about 3.95 GiB and free disk about 61.5 GiB. Training and scoring used CPU with one native numerical thread.

The original Zenodo transfer and a parallel Zenodo trial were slow. Small range probes did not predict sustained speed. We switched to the [official Edinburgh DataShare release](https://doi.org/10.7488/ds/2555), linked by the [challenge page](https://www.asvspoof.org/index2019.html). A four-connection downloader with 8 MiB ranges reused a 639,631,360-byte contiguous prefix and completed the transfer. The prefix includes bytes from both official sources; the last selected URL does not identify every byte's origin. Progress is counted from completed chunks, not total network traffic. Abandoned trial chunks remain ignored locally.

The completed archive is **7,640,952,520 bytes**, and its observed MD5 matches the published `30c98f11d8b2bc21f2c257bfd78bb5c5`. The downloader, preparation command and independent audit checked the complete archive. Only the fixed pilot was extracted: **2,000 official train + 1,000 official dev recordings**, class-balanced within each split, seed 42. All 3,000 selected FLAC files passed exact selection/path matching, journal and archive-member SHA-256 checks, 16 kHz mono PCM16, finite/nonempty decoding and train/dev speaker separation. Audio totals **187,197,762 bytes and 10,434.510625 seconds**. Eval bytes are retained inside the distribution archive; eval audio was not extracted, decoded or scored. Full-split audio has not been semantically audited.

See the [data audit](docs/validation/pilot_data_seed42.json), [data protocol](docs/DATA_PROTOCOL.md) and [preparation guide](docs/PILOT_DATA.md). Individual member SHA-256 values are locally observed; the archive MD5 is the publisher's checksum. Data remains ASVspoof 2019 LA v1 under ODC-By-1.0, with attribution links to the [dataset](https://zenodo.org/records/6906306) and [license](https://opendatacommons.org/licenses/by/1.0/).

### Clean pilot detection

The model used [configs/pilot.toml](configs/pilot.toml): seed 42, 32 diagonal mixtures per class, 60-dimensional LFCCs, no feature normalization, maximum 50 EM iterations, and at most 50,000 retained frames per class. Training saw 343,761 bona fide and 346,389 spoof frames; exactly 50,000 per class were retained with the deterministic train-only sampler. Longer recordings contribute more candidate frames. All 1,000 dev IDs received finite scores. No dev features fitted the model, and speaker/attack labels were not model inputs.

| Clean pilot measurement | Value |
| --- | ---: |
| Train / dev recordings | 2,000 / 1,000 |
| Dev EER | 8.8% |
| Dev minimum normalized t-DCF v1 | 0.2006077385 |
| Training, including LFCC extraction and sampling | 39.223 s |
| Dev scoring, including LFCC extraction | 8.766 s |
| Metric computation | 0.055 s |
| Total processing wall time | 48.044 s |
| Peak process working set | 189.93 MiB |
| Bona fide / spoof EM iterations | 43 / 30 |
| Convergence / warnings | Both converged / none |

The [public baseline audit](docs/validation/baseline_pilot_seed42.json) contains exact source/input/output hashes, environment, configuration and failures. The code is an sklearn adaptation, not an exact official 512-mixture MATLAB reproduction; feature parity remains open. Source hashes identify the implementation because the repository has no experiment commit yet. Local artifacts are in ignored `outputs/baseline-pilot-seed42/`.

At the same-dev EER threshold 0.7139056325, using `score >= threshold` for bona fide, 44 of 500 bona fide recordings were rejected and 45 of 500 spoof recordings accepted: 12 A04 and 33 A06. Literal counts differ by one trial from the official EER grid at the threshold boundary. This threshold is diagnostic, not a deployment rule. Per-attack results are small same-domain counts.

### Custom bandwidth stress test

The model stayed fixed. Each dev waveform was resampled 16→8→16 kHz with the exact Kaiser/polyphase parameters, original sample count and no normalization or codec. Clean scores were reused; transformed scores and the fixed 50/50 arithmetic average were computed without retraining. The [protocol snapshot](docs/validation/robustness_protocol_v1.md), helper source hash and [speaker partition audit](docs/validation/reliability_partition_seed42.json) were frozen before the full-pilot model/results. The snapshot SHA-256 is `c0d0cb9e240a597671c072053a14fd184695c7a69d293258cc20638abcb4d50a`.

| Custom condition on the same 1,000 dev records | EER | Minimum normalized t-DCF v1 |
| --- | ---: | ---: |
| Narrowband only | 21.0% | 0.4961597988 |
| Fixed clean/narrowband score average | 8.4% | 0.1912639195 |

Bandwidth reduction hurts this model when used alone. Averaging gives a small numerical difference from clean scoring, but a single dev pilot with no uncertainty estimate does not establish improvement or superiority. These are exploratory custom diagnostics, separate from clean detection. Official dev ASV scores remain fixed: the experiment does not transform both ASV and CM channels. The transformation is not a real telephone channel or speech codec.

Transformed scoring took 10.276 seconds, with 10.421 seconds total processing and 145.19 MiB peak process memory. Processing timings exclude imports, preflight hashing and artifact output. Peaks include imports and filesystem cache was uncontrolled. The CLI successfully scored a trusted local pilot model and exactly matched the stored dev score for `LA_D_1135743` (-3.4187221527); this consistency check is not a measured demo-latency benchmark.

### Reliability finding and failures

The deterministic dev-speaker split has 8 calibration speakers / 371 records and 12 analysis speakers / 629 records. All A01–A06 attacks occur in both groups, and both are disjoint from train speakers. The clean-score threshold was selected only from the **167 calibration bona fide scores**, using ascending index 8, then saved before transformed scoring and analysis. Threshold 0.7592377067 accepts 159/167 calibration bona fide records (95.21%). Analysis has 63 clean prediction errors out of 629 (10.02%): 41/333 bona fide rejections and 22/296 spoof acceptances. The spoof errors are 4 A04 and 18 A06.

| Error-ranking score on analysis speakers | AUROC | Average precision |
| --- | ---: | ---: |
| Absolute clean/narrowband score change | 0.419962 | 0.080207 |
| Negative distance from the clean threshold | 0.908183 | 0.508570 |

The predefined **large score change means unreliable** hypothesis fails in this pilot. The simple margin baseline ranks errors much better; absolute-change average precision is below the 0.100159 error-prevalence reference. We did not reverse the score, search new seeds, tune a fusion or rewrite the hypothesis after seeing results. Analysis speakers differ from calibration, but remain within the same dev domain and known attack taxonomy. No significance or unseen-domain claim follows.

The [robustness audit](docs/validation/robustness_pilot_seed42.json) stores all analysis records and ten deterministically selected illustrative errors. For example, rejected bona fide `LA_D_1118839` has delta -1.2135; accepted A06 `LA_D_9926699` has delta -1.8074. Accepted A06 `LA_D_1135973` is only 0.00278 from the clean decision boundary, illustrating why margin can help. All 63 analysis errors had negative delta, so a positive-delta error example is explicitly unavailable. These examples were selected from errors, not sampled as representative data.

### Independent validation and next decision

The [independent review](docs/validation/pilot_independent_review.json) passed: all 3,000 audio hashes, exact IDs and splits, model/config/source/artifact hashes, official baseline and three robustness metric dictionaries, independently rebuilt speaker partition, BF-only calibration, error labels, AUROC/AP, failure selection and six rescored clean/transformed values matched. The official [2019 evaluator](https://www.asvspoof.org/asvspoof2019/tDCF_python_v1.zip) remains outside Git under its separate license; the new small setup helper verifies pinned source/ASV files on a fresh clone, and the real local cache passed `--check-only`.

Next, define a bounded full official train/dev confirmation run with seeds 42/43/44 and a locked model/configuration before opening eval. Do not chase the weak delta result with post-hoc tuning. MATLAB feature parity, full benchmark evidence, measured demo latency, the project code license and later localization remain open in [backlog.md](backlog.md). The CPU pilot does not need Colab; reassess Colab for a later neural/localization model after measuring its requirements. No additional full-data extraction, long training, GitHub publication or personal-vault write was started.

Milestone summary for EkinOS: the verified 3,000-record public-data pilot runs on CPU in 48.0 seconds processing time with about 190 MiB peak process memory. Clean EER is 8.8% and min t-DCF 0.2006. The first frozen narrowband experiment found that absolute score change is a poor error indicator (AUROC 0.420), behind a simple margin baseline (0.908). This is a reproducible negative pilot finding, not a full benchmark or novelty claim. Sources: [ASVspoof 2019 LA](https://zenodo.org/records/6906306), [official evaluation plan](https://www.asvspoof.org/asvspoof2019/asvspoof2019_evaluation_plan.pdf), [run audit](docs/validation/robustness_pilot_seed42.json), [independent review](docs/validation/pilot_independent_review.json).

## 2026-10-10 — Resume check and full train/dev plan

The resumed review confirmed the completed pilot artifacts and frozen protocol.
An independent agent checked the README, baseline notes and final report against
the actual JSON results. Values and claims matched; one stale backlog statement
about unmeasured pilot training was corrected. Source/inference code and saved
results were not changed, so the 128-test validation remains the implementation
check for this milestone.

A second agent inventoried the cached ZIP central directory against official CM
keys, without decoding or extracting any new audio. All 25,380 train and 24,844
dev keyed members exist with no missing/duplicate keyed paths. **The dev directory
also contains 142 unkeyed `LA_D_A*.flac` members.** They must be excluded using
exact protocol joins; a folder glob would create the wrong evaluation set.
Keyed train/dev FLAC bytes total 3,159,565,801. Available disk was about 60.5 GiB
and available RAM about 3.64 GiB. Even a separate train/dev copy fits the disk
reserve. The [public inventory](docs/validation/full_train_dev_inventory.json)
records the exact measurements and their limits. ZIP encoded sizes do not prove
decoded waveform validity or memory requirements.

The [full confirmation plan](docs/FULL_CONFIRMATION_PLAN.md) keeps the current
32-component model and 100k-frame cap, preserves natural official class proportions,
and measures seed 42 before sequential seeds 43/44. A count-based projection is
about 12 minutes training plus clean dev scoring per seed; this is only an estimate.
The first engineering review budget is 30 minutes processing and 2 GiB process
peak memory. Extraction/validation time is separate. These budgets need explicit
monitoring in the new path; they are not enforced by the old runner.

The existing runner intentionally requires balanced pilot inputs, so a separate
guarded full-split preparation/runner is the next implementation task. Do not
change the frozen pilot config to bypass those checks. Clean full-dev comes first;
eval remains closed, and any expanded robustness analysis needs a separate protocol.
No additional download, extraction or model training was started during this
resume check. `notes.md` remains available for Ekin's rough notes.

Milestone summary for EkinOS: pilot reporting and independent checks are complete.
Full train/dev confirmation is now specified, with enough local storage and a
measured CPU starting point. The inventory caught 142 extra unkeyed dev files,
which will be excluded by official ID joins. Sources: [ASVspoof 2019 LA](https://zenodo.org/records/6906306),
[inventory](docs/validation/full_train_dev_inventory.json), [next plan](docs/FULL_CONFIRMATION_PLAN.md).

## 2026-10-10 — Full train/dev implementation and detached seed 42 launch

Three parallel agents implemented the separate preparation, full runner and
Windows supervisor. The lead agent reviewed interfaces, integrated the changes,
ran validation and launched the job. A cross-review found no material data-split
or completion-status issue in preparation/runner interoperation.

- `scripts/prepare_full_data.py` verifies the cached archive and pinned official
  protocols, exports the exact 50,224 train/dev rows and extracts only keyed
  members. It preserves natural class counts and excludes 142 unkeyed dev members.
  A separate journal supports verified interruption recovery and protects the
  pilot journal. Atomic checkpoints are batched every 250 files. Plan mode is
  read-only; no network fallback exists.
- `scripts/run_full_baseline.py` requires the full canonical manifest SHA-256
  `37da16080ddbf0c0253b1f093a11917e1c463f12b2f88d322c270587abf17458`, exact official
  metadata and complete verified audio. It keeps the 32-component/100k-frame
  settings fixed, saves the model before dev scoring, and persists incremental
  scores and failure/progress state. `configs/full_seed42.toml` keeps eval disabled.
- `scripts/start_full_run.ps1` and `scripts/full_run_job.py` run preparation then
  seed 42 only, without an active chat or OpenAI API calls. Logs, source/config
  hashes, actual process IDs, exit codes and stop reasons are recorded. Preparation
  has a 20-minute budget; the full runner, including its preflight, has 30 minutes.
  The supervisor stops its own worker at 2 GiB process peak or below 1 GiB available
  system RAM. Startup requires 3 GiB available RAM and 20 GiB free disk; preparation
  also reserves its missing audio requirement. Complete status requires verified
  preparation and all final model/score/config artifacts and hashes.

Validation passed: the full suite ran **155 tests in 124.390 seconds**, followed
by final targeted runner checks (**9 tests**) and supervisor checks (**10 tests**).
Across those runs, **159 distinct test cases** passed; overlapping tests are not
counted twice. Dependency checks reported no broken requirements. PowerShell syntax
passed. Existing inference/feature/pilot source files and pilot artifacts were
not changed.

Launch verification caught a real Windows issue: venv `python.exe` is a redirector
which spawns the actual interpreter. The initial supervisor was measuring that
small redirector rather than the worker. We identified the exact owned preparation
process tree and stopped its actual worker before training; its failed run logs
were preserved. An earlier sandbox-contained launch did not establish a persistent
supervisor. No failed startup was reported as successful training.

The fix launches the actual base interpreter with global site initialization
disabled, then explicitly prioritizes the project venv packages. A new real Windows
regression checks worker `os.getpid()` equals the retained subprocess PID, queries
that process handle and verifies actual NumPy/SciPy/sklearn/soundfile/joblib imports
match the venv versions. Final targeted tests passed after this change. The actual
background launch uses a local execution context that survives the tool session.

The corrected job started **2026-10-10 02:27 Istanbul time**. Its authoritative
status is `outputs/jobs/full-seed42-20261009T232740327Z/status.json`; supervisor PID
is 11404 and the preparation worker is 16532. Launch verification observed
`running`, stage `prepare`, and about 150 MiB real worker peak memory after 75
seconds. This is preparation, not a completed training result. The start measurement
recorded 3,351,379,968 bytes available RAM and 64,997,113,856 bytes free disk.
GPU management queries were unavailable; this job uses CPU only, consistent with
the previously observed Intel integrated GPU setup.

After preparation succeeds, seed 42 training and clean scoring start automatically.
No seeds 43/44, robustness extension, eval scoring, automatic notification or
GitHub publication are scheduled. The assistant ends the turn instead of polling
through training. On the next result-check request, read supervisor status and
logs first, then independently audit any completed output. Keep the computer
awake; a stopped run needs review before recovery. See [FULL_RUN.md](docs/FULL_RUN.md)
and [backlog.md](backlog.md).

Milestone summary for EkinOS: full official train/dev preparation and CPU runner
are implemented, tested and launched as one detached seed 42 job. The exact
official IDs and original class proportions are enforced; eval remains closed.
Full-set metrics are pending. Sources: [ASVspoof 2019 LA](https://zenodo.org/records/6906306),
[fixed plan](docs/FULL_CONFIRMATION_PLAN.md), [run guide](docs/FULL_RUN.md).

## 2026-10-10 — Preparation timeout diagnosis

The corrected detached job stopped at 02:47 Istanbul time because its preparation
stage reached the configured 1,200-second budget. The saved journal contains
47,750 / 50,224 verified records (95.1%) and no preparation errors. Training and
dev scoring never started. The real worker peak was 262,709,248 bytes (about
250.5 MiB), with about 2.93 GiB available system RAM at the last observation;
the stop reason was time, not memory. The initial 20-minute preparation budget
was too short for the measured local workload.

Files and the separate journal are preserved. The journal's `running` status is
the last checkpoint, not a live process indication; the supervisor says `stopped`.
A restart must review the owned PID and any stale preparation lock, then use a
documented larger preparation budget. The current preparation resume validates
existing files again, so remaining-record count alone understates restart cost.
No restart was performed during diagnosis.

Recommendation: finish this CPU LFCC-GMM locally after correcting the preparation
budget. Colab would require environment/data setup and does not automatically
accelerate this sklearn CPU model with a GPU. Keep Colab for a later neural model
or a deliberate portable CPU run. Its resource availability and runtime limits
are variable; see the [official Colab FAQ](https://research.google.com/colaboratory/faq.html).
No Colab session, upload or new download was started.

## 2026-10-10 — Resume with a larger preparation budget

After Ekin authorized recovery, the preparation budget was increased from 20 to
60 minutes. Resume revalidates existing files, so the previous 95.1% checkpoint
does not imply only a short remaining pass. The seed 42 model, frame cap,
30-minute runner budget and closed eval gate are unchanged.

The supervisor now waits up to 15 minutes before each stage for two consecutive
3.5 GiB available-RAM readings, five seconds apart. Workers still enforce their
3 GiB startup gate, 1 GiB running floor and 2 GiB process peak budget. The job
lock stays held while waiting. Resource waits have separate deadlines and do not
consume worker budgets; insufficient disk stops immediately. This is bounded
resource waiting, not an automatic retry after failure.

All 13 supervisor tests passed, including mocked resource recovery, deadline
expiry, disk rejection and the real Windows worker PID/package regression.
A parallel agent reviewed the resource-wait integration and found no material
correctness issue. It confirmed that the two per-stage waits can total 30 minutes.
The native process query found old worker PID 16532 absent; only its stale
preparation lock was removed. Existing journals, audio and old job logs remain.

The new detached job is `outputs/jobs/full-seed42-20261009T235915974Z/`, launched
at 02:59 Istanbul time with supervisor PID 19124. A separate native query proved
it survived the launch shell. First status: `waiting_resources`, no worker,
3,318,919,168 bytes available RAM and 63,854,522,368 bytes free disk. Its startup
deadline is 03:14 Istanbul time. Intel Iris Xe was reconfirmed; this is a CPU job.
Training has not started at this observation. No download or Colab run was added.

Milestone summary for EkinOS: the local recovery path is tested and independently
reviewed. A detached seed 42 job is waiting for enough RAM; full-set metrics remain
pending. See [run instructions](docs/FULL_RUN.md), [fixed experiment plan](docs/FULL_CONFIRMATION_PLAN.md)
and [resume state](backlog.md). Dataset source: [ASVspoof 2019 LA](https://zenodo.org/records/6906306).

## 2026-10-10 — Model saved; scoring hit the memory budget

The latest status check confirms that preparation finished successfully in
708.906 seconds and verified all 50,224 official train/dev records. The runner
processed all 25,380 train recordings and saved the fitted model at 03:21:48
Istanbul time. Its actual SHA-256 matches the saved run state:
`701fc3d8107cc8b2696b5552dd9e1b4751e44d2d197cfd5d1dfe7e4814add101`.

At 03:26:14 the runner and supervisor recorded the configured 2 GiB process-peak
budget crossing (2,147,807,232 bytes observed by the supervisor). The runner's
error was `Resource stop: process peak exceeds 2 GiB`, during dev scoring.
Available system RAM was still about 2.91 GiB at that observation. This was
neither the runtime timeout nor evidence that closing Codex stopped the job.
Native process queries found the recorded supervisor and workers absent.

The partial score file has 22,815 unique dev IDs with finite scores (91.8% of
24,844). No successful summary exists, so full dev metrics remain pending.
The saved model permits a scoring-only recovery; validate provenance and the
official score prefix, investigate memory growth, and preserve the failed run
before implementing that recovery. This status check did not launch another job.
See [resume state](backlog.md) and [run guide](docs/FULL_RUN.md).

## 2026-10-10 — Memory probe fix and scoring-only recovery

A parallel investigation found the cause of the process growth in the Windows
resource probe. Each call defined new ctypes structures and pointer types;
Python's pointer-type cache retained them. In a 2,000-call diagnostic, the cache
grew from 22 to 4,024 entries and RSS increased by 31,891,456 bytes. The original
run made at least about 119,205 worker probes by its scoring checkpoint, which
explains growth near 2 GiB without a retained feature cache.

The worker probe and supervisor probe now cache their Windows types and API
bindings. A separate 2,000-call diagnostic after the worker fix kept the cache at
22 entries; observed RSS growth was 671,744 bytes. These are small diagnostic
measurements, not guaranteed full-run memory bounds. Model fitting, LFCC and
score mathematics are unchanged. The original resource-source hash stays in
the failed job; recovery records the new instrumentation hash separately.

All 14 supervisor tests, 3 native resource tests and 3 recovery-orchestrator
tests passed. The memory agent also ran all 9 existing full-runner tests.
A second agent reviewed process ownership, recovery ordering, completion checks
and public audit output. A real checkpoint smoke check reproduced the first
stored dev score exactly (absolute difference 0), then scored the first missing
record successfully. Peak process memory in that two-record check was
152,326,144 bytes; this is not the complete recovery peak.

The recovery pipeline checks original model/config/source/package provenance,
validates the official score prefix, scores the remaining 2,029 recordings in
three fresh processes, and calculates complete official dev diagnostics. It
preserves the failed original run and its saved model. Final scores are checked
independently by the supervisor before the compact public audit and measured
metrics are written. No retraining, dataset download or eval access is involved.
See [the recovery guide](docs/SCORING_RECOVERY.md).

### Recovery launch and final validation

The recovery worker's 12 tests passed, including successful atomic chunk commit,
complete final score assembly, missing/overlapping/corrupt input rejection and
unpickle hash protection. The real checkpoint validation passed all 25,380
training IDs and 22,815 original score rows. All three overlap checks had zero
absolute score difference. Supervisor, resource and orchestration checks also
passed; the parallel reviews found no remaining blocker.

The independent recovery job started at 03:41 Istanbul time:
`outputs/jobs/scoring-recovery-seed42-20261010T004116241Z/`, supervisor PID 28900.
A native process query confirmed that it survived the launcher shell. First
observed status: running preflight, worker PID 19432, worker peak 199,077,888
bytes. The ready gate recorded 5,094,617,088 bytes available RAM and
63,709,179,904 bytes free disk. These are launch observations, not final results.

The job will compute the remaining scores, full dev metrics and a compact public
audit sequentially without another user confirmation. It appends a separate
completed-results entry only after successful independent artifact validation.
No agent polling loop or scheduled wakeup was added. See
[recovery instructions](docs/SCORING_RECOVERY.md) and [resume state](backlog.md).


## Completed full dev scoring recovery

Finished: 2026-10-10T00:44:03.079639+00:00. The saved seed 42 model now has all 24,844 official dev scores. No retraining or eval scoring was performed. Dev EER: 5.8902%; minimum normalized t-DCF: 0.152678. These are development results for the Python LFCC-GMM adaptation, not held-out benchmark claims.

The supervisor independently checked the complete score artifact against the pinned official dev protocol. Recovery kept the failed original run and model unchanged. Timings describe recovery stages only. See [the compact audit](docs/validation/full_baseline_seed42.json) and [recovery guide](docs/SCORING_RECOVERY.md).

## 2026-10-10 — Full seed 42 results and literal error analysis

The scoring recovery completed at 03:44 Istanbul time. Complete dev EER is
5.8902% and minimum normalized t-DCF is 0.152678. Both mixtures converged
without warnings, at 47/35 iterations. Recovery's maximum supervised worker
peak was 222,257,152 bytes (about 212 MiB). Original training/scoring time is
not reconstructed from recovery timings. Eval remains closed.

A compact, reproducible literal-threshold diagnostic uses the same-dev EER
threshold 0.670067310333252, accepting bona fide on equality. It rejects
150/2,548 bona fide recordings and accepts 366/3,716 A04 and 949/3,716 A06
spoof recordings. A01/A02/A03/A05 have zero observed errors at this threshold;
that is not a generalization guarantee. There is one threshold tie. The counts
are diagnostic and may differ from the official EER grid at a boundary.
Two new tests cover tie decisions and invalid/nonfinite metadata. The exporter
checks exact official dev ordering, score hashes and train/dev ID separation.

See [results](docs/RESULTS.md), [error audit](docs/validation/full_seed42_errors.json)
and [seed 42 audit](docs/validation/full_baseline_seed42.json). Independent
review and the remaining preplanned seeds 43/44 are the next confirmation gate;
no parameter retuning or new robustness protocol is being mixed into this run.


## Independent full train/dev recovery review (2026-10-10)

The completed seed-42 recovery passed an independent audit. All 25,380 official training IDs and 24,844 development score IDs match the original protocols; train/dev speakers and IDs are separate. All 50,224 local audio files match the frozen journal hashes. The original model, configuration, run state and 22,815-score prefix remain unchanged. Model/features source and package versions match the saved checkpoint; the recorded resource instrumentation fix is the only original core-source change. Six waveform score checks reproduce exactly, including recovered rows. Direct calls to the pinned official 2019 v1 evaluator reproduce EER **5.890201986%** and minimum normalized t-DCF **0.1526784046**, with exact threshold and ASV-rate agreement.

Evidence: [independent review](docs/validation/full_independent_review.json), [recovery implementation](scripts/recover_full_scoring.py), [metric protocol](docs/METRICS.md). This is a complete official development-split result for the fixed sklearn adaptation. Eval remains closed; this is not a held-out benchmark or deployment claim.

## 2026-10-10 — Remaining fixed seeds launched

The independent seed 42 review passed: all 50,224 audio hashes, exact protocols,
model/source/package provenance, unchanged original score prefix, six exact
waveform score reproductions and direct official-reference metrics. The actual
prior-result gate passed before another worker was started.

A parallel implementation added guarded sequential seeds 43 then 44, changing
only experiment.seed in the fixed configuration. All 11 tests passed, covering
conflicting outputs/locks, missing or mismatched prior audit, stopping before a
second seed after failure, metric protocol drift, population standard deviation,
retained nonconvergence and removal of absolute commands/paths from public audit.
A separate agent reviewed the launch and completion checks; its one public-path
issue was fixed before launch. No learning or inference definition changed.

Detached job: `outputs/jobs/full-multiseed-20261010T012814223Z/`, launched at
04:28 Istanbul time, supervisor PID 19852. First check: running seed43 worker
PID 10992; its input preflight precedes fitting. A native process query confirmed
survival after the launch shell. The RAM gate measured 5,078,228,992 available
bytes and disk 63,663,435,776 free bytes. Intel Iris Xe was reconfirmed; fits use
CPU with one numerical thread. Each seed has the existing 30-minute/2 GiB limits.

After both seeds complete, the job validates exact dev score order, all official
input hashes and model configuration, then writes per-seed metrics and mean /
population standard deviation (`ddof=0`) to the compact public multiseed audit
and appends a measured report. No automatic retry, new seed, dataset download,
eval run or scheduled agent wakeup is enabled. Full-set custom robustness still
needs its separate frozen protocol and partition audit.

Milestone summary for EkinOS: full seed 42 is independently verified at EER
5.89% and min t-DCF 0.1527; the fixed three-seed confirmation is now running
sequentially. These are dev results for a Python adaptation, not held-out eval
claims. See [results](docs/RESULTS.md), [independent review](docs/validation/full_independent_review.json),
[multiseed guide](docs/MULTISEED_RUN.md) and [resume state](backlog.md).


## Full dev confirmation across three seeds

Completed 2026-10-10T01:44:56.812975+00:00. Fixed seeds 42, 43 and 44: mean dev EER 6.3441% (population standard deviation 0.3316 percentage points); mean minimum t-DCF 0.172367 (population standard deviation 0.014112). Seeds with convergence warnings/failure: []. Eval stayed closed. These are development results for the Python adaptation. Seed 42 timing covers recovery only. See [the measured audit](docs/validation/full_multiseed_summary.json).

## 2026-10-10 — Morning verification: all three seeds complete

The sequential seed 43/44 job completed at 04:44:56 Istanbul time without errors.
Each seed scored all 24,844 official dev recordings; all class mixtures converged
without warnings. Seed43 EER/min t-DCF: 6.6729% / 0.179381. Seed44: 6.4694% /
0.185041. Across fixed seeds42/43/44, mean EER is 6.3441% with population std
0.3316 percentage points; mean min t-DCF is 0.172367 with population std 0.014112.

The morning check revalidated saved artifacts, configuration and official input
provenance, all training IDs, complete dev score ordering and direct pinned
reference metrics for both new seeds. Recomputed metrics and aggregate matched
exactly. Worker elapsed times including preflight were 488.984 and 503.078
seconds; peaks were about 210.2 and 212.4 MiB. No further job was started.

These are development diagnostics for the Python adaptation; eval remains
closed. Next is a separately frozen full-dev narrowband protocol and partition,
followed by measured file-scoring demo latency. See
[three-seed results](docs/RESULTS.md), [measured audit](docs/validation/full_multiseed_summary.json),
[post-run verification](docs/validation/full_multiseed_review.json) and [backlog](backlog.md).

## 2026-10-10 — Full-dev narrowband protocol frozen before transformed scoring

The next experiment is fixed in [FULL_ROBUSTNESS_PLAN.md](docs/FULL_ROBUSTNESS_PLAN.md).
It uses all three saved models and the complete official dev population, with the
unchanged narrowband transformation, clean/narrow/fixed-average comparisons and
absolute-delta versus negative-margin error rankings. There is no retraining,
parameter search or favorable seed selection. A parallel agent drafted and
reviewed the protocol; explicit validation-script hashes were added before freeze.

The new full-ID speaker partition has 9,609 calibration records from 8 speakers
(945 bona fide / 8,664 spoof) and 15,235 analysis records from 12 speakers
(1,603 bona fide / 13,632 spoof). Both groups contain all six attacks and are
speaker-disjoint from each other and train. The old pilot partition files remain
unchanged; the new complete ID metadata and CSVs stay in ignored outputs.

Using only clean calibration bona fide scores, ascending index47 of945 freezes
thresholds 0.7167018055915833, 0.7260986566543579 and 0.7200403809547424 for seeds
42/43/44. Each accepts898/945 (95.026%) at score>=threshold. No full-dev EER
threshold is reused for this error-ranking analysis. Three model thresholds,
input/model/source/package hashes, partition hashes and reporting rules are in
[the immutable contract](docs/validation/full_robustness_protocol_v1.json), SHA-256
`806fd09f488449d6e276e4b36b60ddc016441be2b62e7a223d3bc74d7ea6d913`.

Five tests passed: calibration-only thresholding, order invariance, boundary ties,
missing/nonfinite input rejection, overwrite prevention and serialized partition
hashes. After writing, rebuilding the contract and validating all saved file
hashes reproduced the freeze exactly. No transformed score, audio processing,
training or download was started. A separate guarded full-dev runner is the next
implementation step; the pilot runner's scope restrictions remain intact.

The clean dev results/labels and pilot stress results were already known. This
is a pre-transform protocol freeze, not blind preregistration or independent
validation. The pilot is a subset of full dev, the attack taxonomy overlaps train,
and all three seeds reuse the same population. Eval remains closed. Undefined
ranking metrics must retain their reason; partially defined seed aggregates must
be labeled incomplete.

Milestone summary for EkinOS: the three-seed clean baseline is complete (mean dev
EER6.34%, mean min t-DCF0.1724). Full-dev robustness now has a reproducible frozen
protocol and new speaker-separated calibration/analysis artifacts; execution is
pending. Sources: [clean results](docs/RESULTS.md), [protocol](docs/FULL_ROBUSTNESS_PLAN.md),
[contract](docs/validation/full_robustness_protocol_v1.json), [resume state](backlog.md).

## Full-dev narrowband runner and launch — 2026-10-10

Parallel workers implemented bounded scoring and independent finalization; a third agent reviewed their integration. Initial smoke-schema and metric-path mismatches were fixed before launch. Twenty focused tests passed. Real read-only checks rebuilt the frozen contract and verified all three model metadata sets. The 16 frozen implementation sources remain unchanged; four new execution files have separate launch pins.

Fresh hardware check: Intel Iris Xe, 16,872,415,232 bytes total RAM, 4,995,252,224 bytes available RAM and 64,082,100,224 bytes free disk. This job uses the CPU. The resource gate observed 4,962,418,688 bytes available RAM and 63,711,539,200 bytes free disk at start.

Started at 09:51:34 UTC / 12:51 Istanbul. Job: `outputs/jobs/full-robustness-20261010T095134725Z/status.json`; supervisor PID 30012. An independent native process check after the launcher exited confirmed the supervisor alive and preflight worker 18740 running. No training, download or eval decoding is requested. Preflight checks all cached train/dev audio; a fixed first-20-dev smoke for each model must pass before full transformed scoring. Full scoring uses fresh chunks and sequential seeds 42/43/44, a single numerical thread and a shared 30-minute processing budget per seed including startup and measured smoke. Partial files and stop reasons are preserved.

The finalizer requires all ordered paired scores, recomputes clean reference metrics, checks supervision budgets and hashes all audio again before publishing the compact public audit. Completion is not claimed by this launch note. Read the current status before continuing. See [the run guide](docs/FULL_ROBUSTNESS_RUN.md), [frozen protocol](docs/FULL_ROBUSTNESS_PLAN.md) and [backlog](backlog.md). Clean and pilot dev results were previously inspected; this is a custom stress diagnostic, not independent validation.

## Narrowband RAM stop and explicit recovery — 2026-10-10

The first scoring job stopped at 10:33:13 UTC / 13:33:13 Istanbul in seed 44's first chunk. The worker reported insufficient available system RAM, below the unchanged 1 GiB running floor. Its peak process memory was 217,456,640 bytes (about 207 MiB); this was a system available-memory guard, not a reported worker out-of-memory exception. Models were not retrained. All 26 completed chunks for seeds 42 and 43 passed exact joins, hashes and provenance checks. Original frozen and execution sources remain unchanged.

Parallel work produced a separate recovery audit and an independent review. Eight focused recovery tests passed, including failed-attempt budget accounting, source/evidence pin mutation rejection and finalizer-hook recursion prevention. The broader worker/supervisor/finalizer test set passed before the last evidence-pin hardening; those new pin tests passed separately. The real resume check and actual base-Python bootstrap also passed. Recovery keeps the original failed attempt's 90.844 seconds and seed 44's 3.1026651-second smoke inside the same 1,800-second budget. It verifies audio again, scores only seed 44 and publishes the failure history alongside successful-stage supervision. Original chunks and logs are retained.

Recovery launched at 10:41:30 UTC / 13:41:30 Istanbul, supervisor PID 10840. Current status: `outputs/jobs/full-robustness-recovery-20261010T104130247Z/status.json`. Available RAM before launch was 5,501,374,464 bytes; the fresh disk check showed 63,074,099,200 bytes free. GPU: Intel Iris Xe; inference stays on CPU with one numerical thread. No guard was relaxed. This entry records launch, not completed results. See [the recovery guide](docs/FULL_ROBUSTNESS_RUN.md) and [backlog](backlog.md).


## Full-dev narrowband experiment completed

Completed 2026-10-10T10:54:31.684202+00:00. All 24,844 dev recordings were scored with each fixed model seed (42, 43, 44). Clean-score joins and official metric recomputation passed. Narrowband and fixed averaging are custom stress diagnostics with unchanged ASV scores. Error rankings use the frozen clean calibration thresholds and analysis speakers. This reuses inspected dev data and is not independent validation. See [the complete audit](docs/validation/full_robustness_results_v1.json) for each seed, population aggregates, failure examples and provenance.

## 2026-10-10 — Project report and GitHub results ready

The project now has a six-page [PDF report](docs/report/audio_deepfake_report.pdf),
a standalone [LaTeX source](docs/report/audio_deepfake_report.tex), and updated
[README](README.md) and [results](docs/RESULTS.md). The writing uses simple
English and focuses on the question, method, evidence and remaining limits.
There are no novelty claims or invented achievements. The negative result is
kept: mean clean, narrowband and fixed-average EER were 6.34%, 22.17% and 9.03%;
absolute score change ranked errors worse than the clean decision margin.

Three figure sets cover detection, reliability and seed 42 attack errors, with
SVG files for GitHub and PNG/PDF versions for the report. The figures are
recreated from checked public audits rather than copied from screenshots. The
[report build guide](docs/report/README.md) records the build commands and
reporting dependencies. The PDF was exported with ReportLab using the same
report content and data; the standalone LaTeX source also compiled successfully,
including the final revision. These are separate export paths, as documented.

Parallel work was divided into three roles: one agent checked the numerical
claims and evidence joins, one updated the GitHub-facing text, and one reviewed
the report and figure artifacts. The project lead assembled the report,
generated the figures and coordinated final checks. The independent
[evidence review](docs/REPORT_EVIDENCE.md) and
[machine-readable review](docs/validation/report_evidence_review.json) passed
all nine metric comparisons and joins. The
[artifact review](docs/validation/report_artifact_review.json) records a
repeatable rebuild with matching hashes and visual inspection of all six pages.
Table spacing was corrected during review.

The report distinguishes the official dev Python adaptation from custom
bandwidth stress diagnostics, keeps the fixed ASV reference limitation, and
explains that dev and pilot results were already inspected. Three seeds reuse
the same recordings. Eval remains closed. No training, new experiment, audio
download or GitHub publication was started for this milestone.

The next implementation step is a local file-scoring demo with input checks,
a model score and measured CPU latency. It will reuse an existing verified
model and will not label the raw score as an authenticity probability.
PartialSpoof localization remains a later pilot.

Milestone summary for EkinOS: the three-seed detection and robustness results
are now packaged in a checked English report with reproducible figures. The
score-change idea did not help error ranking under the frozen dev protocol.
The [source-linked EkinOS summary](docs/EKINOS_SUMMARY.md) is ready to reuse;
[backlog](backlog.md) records the next local demo step.

## 2026-10-10 — Simpler report, local demo and measured latency

The report now credits **Özgür Ekin Sönmez** and acknowledges AI assistance
with implementation, debugging, tests, parallel review, documentation and
report editing. The layout is simpler. Grayscale mean bars replace the
earlier whiskers and seed dots; reference-line overlays were removed.
Individual seed values and population SD remain in the tables. Sources and
licenses are cited without claiming an exact official baseline reproduction.
The detection and robustness results are unchanged. The project studies
fake-speech detection and model errors; CPU execution is a practical choice.

Parallel implementation was divided between the local CLI/browser demo,
the frozen latency measurement, and figure/document updates. The project
lead integrated the report, sources and reviews. The demo reuses the fixed
seed 43 model after hash and metadata checks; it does not retrain or select
the best seed. It accepts finite mono 16 kHz WAV/FLAC files up to 60 seconds
and 8 MiB, keeps browser uploads in memory, and returns a raw score rather
than an authenticity probability. Model artifacts stay ignored by Git, so
a fresh clone needs the matching local model before the demo works.

The [timing plan](docs/validation/demo_latency_plan.json) fixed the first
20 official dev rows and three passes before measurement. On an Intel
Core i7-1195G7 with one numerical thread, the 60 observations had mean
12.3973 ms, median 11.6353 ms and p95 16.9184 ms decode/scoring wall time.
Total processing time divided by audio duration was 0.00311649. Model
verification/loading took 0.211337 seconds separately; process peak was
about 189 MiB. Every observed score exactly matched its saved seed 43 clean
score. The [audit](docs/validation/demo_latency.json) records each observation,
source/model hashes, environment and resources.

The sample contains only bona fide recordings from speaker LA_0069;
durations are 2.555–5.92725 seconds. Cache and process were warm, repeated
observations are dependent, and browser/HTTP overhead was excluded. These
numbers check local usability, not deployment performance or accuracy.
No training, dataset download or eval decoding was started. The next
research milestone is a bounded PartialSpoof localization pilot after
checking current labels, licenses, splits, source overlap and metrics.

Milestone summary for EkinOS: the report and GitHub text are simpler and
include full author credit, sources and an honest AI acknowledgment. The
local demo works with the saved detector; a small fixed CPU check reproduces
its scores and records processing time. See the [report](docs/report/audio_deepfake_report.pdf),
[demo commands](README.md#local-demo), [results](docs/RESULTS.md),
[timing audit](docs/validation/demo_latency.json) and [next steps](backlog.md).

Final artifact review: all eight PDF pages were rendered and visually inspected; the LaTeX source compiled successfully. A repeated report build produced identical hashes. Independent final checks confirmed 16 frozen protocol sources, four execution sources, six timing sources and 34 report numeric values with no mismatch. The 13 focused demo/timing tests passed, and the live loopback server returned HTTP 200 after its launcher exited. See [artifact review](docs/validation/report_artifact_review.json).


## 2026-10-10 - PartialSpoof localization preparation

Two parallel agents reviewed the official dataset/evaluator and implemented
pure timeline utilities. The [protocol review](docs/PARTIALSPOOF_PROTOCOL.md)
pins the upstream commit, dataset versions, published checksums, attribution,
score direction and time scale. RangeEER implementation issues were found
by source inspection; they have not been reported as executed failures.
A separate evaluator environment and synthetic conformance tests are still needed.

The [pilot draft](docs/LOCALIZATION_PILOT_PLAN.md) proposes eight records per
split for alignment checks, then 128 train / 64 dev recordings with seed 45
and one LFCC/logistic temporal baseline. IDs and configuration are not frozen
yet. Real archive layouts, last-bin semantics and nonspeech handling must pass
before any fit. Small subset sizes do not reduce the published archive sizes.

The new [timeline module](src/audio_deepfake/localization.py) validates
half-open intervals, explicit rasterization policies, binary native-label
conversion and score-to-interval conversion. Eight focused tests passed,
including partial final bins, invalid labels and threshold ties. A decimal
half-bin rounding failure was caught and corrected. This is infrastructure,
not an official label parser, RangeEER adapter or trained localization model.
Its threshold equality convention differs from the official RangeEER source
and is documented; a future adapter must handle that explicitly.

Current preflight recorded about 3.55 GiB available RAM and 57.52 GiB free
disk. CPU/GPU identity comes from the previous hardware audit. No audio was
downloaded, no training started and no new dependencies were installed.
Frozen detection and demo source hashes are unchanged. See the
[preparation audit](docs/validation/localization_preparation.json) and
[resource snapshot](docs/validation/localization_preflight.json).


## 2026-10-10 - Verified localization metadata and bounded audio preparation

Parallel work covered metadata acquisition, upstream metric conformance,
feature mapping, timeline rendering and resumable audio preparation. The lead
added the temporal model API, explicit native adapters, a metadata diagnostic
figure, integration checks and a detached acquisition supervisor.

All four metadata artifacts passed published MD5 checks: 92,230,352 bytes,
no retry. The true train/dev protocol counts are 25,380 / 24,844. The fixed
seed 45 selection is 128 train / 64 dev with equal utterance classes, real speaker
IDs, 8+8 smoke IDs and five bona fide+five spoof dev timeline IDs. The selected
binary export is 134,126 bytes and requires no pickle. Native source arrays
are Unicode 0/1. Eval member payloads were neither extracted nor interpreted;
gzip scans may decompress skipped members in the published mixed containers.

Real metadata disproved a convenient reference assumption: nonspeech VAD = 2
can be native binary 0 or 1. CON_D_0003090 has 77 native spoof bins despite zero
VAD spoof duration. The [diagnostic figure](docs/figures/partial_label_alignment.svg)
is annotation-only, not a prediction or accuracy result. 11 train / four dev
selected native label vectors differ from ceil(VAD duration / 10 ms). Do not
silently trim them. Actual sample counts and upstream binary RTTM are next.
See the [metadata audit](docs/validation/partial_metadata.json).

The actual pinned upstream metric raised NameError and UnboundLocalError;
the first narrow compatibility variant also stalled on a discrete mixed
fixture. A separately named bounded patched upstream variant detects repeated
six-value search states or 128 iterations. It retains duration rates and the
original closest-observed selection, including the initial row. The fixture
terminates in 54 iterations with FPR 3/4, FNR 2/3, averaged error 0.708333, matching
hand-counted durations. No threshold interpolation is added. The [conformance
audit](docs/validation/range_eer_conformance.json) preserves originals,
patches, hashes, timeout evidence and the isolated environment. This is
synthetic compatibility, not native-data alignment or paper reproduction.

The feature mapping uses integer sample coordinates, 20 ms output bins and
nearest LFCC window centers, with earlier-frame ties. Future-frame deltas
make it offline. The model API caps retained frames per class, balances
counts, fits normalization only on retained train frames and records config,
versions and sampled-feature hash. It has only synthetic training tests.
The timeline renderer returns SVG and has no real dataset predictions yet.
40 focused project tests and three separate metric tests passed in final
checks. Frozen detector/demo source hashes remain unchanged.

Website transfer probes returned exact 206 first-byte responses for both
2 GB archives, reading only three body bytes in total; future offsets still
require validation. Fresh hardware/resource inspection found Intel Iris Xe,
3,650,695,168 bytes available RAM and 63,515,074,560 bytes free disk. CPU/GPU
are unused for acquisition. The detached audio supervisor was launched at
18:08 Istanbul, PID 16136, initial worker 22720. Its job folder is
`outputs/jobs/partial-audio-20261010T150830073Z`. It pins the downloader,
supervisor, metadata and selection. Train/dev total 4,055,073,906 bytes;
cumulative payload cap 4.5 GB, extraction cap 256 MiB, RAM/disk guards and
30-minute per-split cooperative budgets plus 1,865-second external timeouts.
Partial files and budgets survive stops. No training is launched by this job.

See [current run and resume instructions](docs/LOCALIZATION_RUN.md),
[live transfer audit](docs/validation/partial_audio.json),
[pilot plan](docs/LOCALIZATION_PILOT_PLAN.md) and [backlog](backlog.md).
The completed detection report remains unchanged. Real audio alignment,
native binary references, frozen model configuration and the actual pilot
fit remain open. No real-data localization metric is claimed.


## 2026-10-10 - Resume after network change

The first audio transfer stopped with `Premature response EOF` at 18:37:47
Istanbul. Its 889,711,396 partial bytes and cumulative charged-byte history
were preserved. Both old processes had exited and no lock remained. Fresh
resources showed 4,534,132,736 bytes available RAM and 62,622,879,744 bytes
free disk. A new hidden job was launched at 18:38 using the same sources,
metadata, selection and budget. Status:
`outputs/jobs/partial-audio-20261010T153826745Z/status.json`; supervisor
PID 12592, initial worker PID 14892. The script must validate exact HTTP
Content-Range before appending; no full restart or training was requested.


## 2026-10-10 - Audio RAM-guard stop and unchanged-limit resume

The second job stopped at 18:45:12 Istanbul with `Audio RAM guard failed`.
Both processes had exited and no lock remained. The partial archive retained
1,461,185,316 bytes. A fresh probe showed 2,513,739,776 bytes available RAM
and 62,040,276,992 bytes free disk, passing the unchanged resource gates.
A new detached job was launched at 18:51, supervisor PID 21944, status
`outputs/jobs/partial-audio-20261010T155124521Z/status.json`. Existing source
pins, selection, partial archive and cumulative transfer budget were kept.
No model training was started and no guard was relaxed.

## 2026-10-10 - Repeated RAM stop confirmed

The third audio job stopped at 18:52:09 Istanbul with `Audio RAM guard failed`.
Supervisor 21944 and worker 19316 have exited; the exclusive lock is absent.
Train partial size is 1,527,245,604 bytes (75.34% of the published archive);
cumulative charged payload is 1,528,823,808 bytes. A fresh Windows
GlobalMemoryStatusEx probe found 1,245,564,928 bytes available system RAM,
below the unchanged 2 GiB gate. Free disk was about 57.72 GiB. No restart
was attempted with insufficient RAM. Partial files, budgets and source pins
remain preserved. Dev acquisition and localization training remain pending.

## 2026-10-10 - Resume with recovered RAM at 19:06 Istanbul

Fresh available RAM was 6,542,790,656 bytes and free disk 62,062,759,936 bytes.
No old worker or lock remained. Started the unchanged hidden audio job:
`outputs/jobs/partial-audio-20261010T160629331Z/status.json`, supervisor 13608,
worker 11120. Running status and actual train partial growth from
1,527,245,604 to 1,530,391,332 bytes confirmed. Existing source pins,
resource limits and cumulative charge history were retained. This job only
prepares train/dev audio; it does not start training.

## 2026-10-10 - Train archive verified; dev transfer active

At about 19:15 Istanbul, the latest audio job reports train complete and dev
running. Train MD5 matches the publisher: `c4853ddd831e8e96b0e279fc0a512e7e`;
archive SHA256 is `d34d99e0eb59707ab23195bb87fc8a9251373257fae6319119c826f0cd030f71`.
Selected train WAVs have individual audio/header checks and hashes in the
[transfer audit](docs/validation/partial_audio.json). Dev partial growth
from 537,919,488 to 542,113,792 bytes confirmed (26.7%). Available RAM was
6,414,098,432 bytes; stderr was empty. Native-label alignment and training
remain pending after both splits finish.

## 2026-10-10 - Audio preparation complete and pilot queue implemented

The earlier audio supervisor/worker disappeared while its status remained
running, leaving a stale lock at 1,959,788,544 dev bytes. Elevated process
inspection confirmed both PIDs absent; the lock PID matched the old worker.
The review was saved in the old job folder before removing only that lock.
The shutdown cause was not recorded. A new job resumed with unchanged limits,
source pins and cumulative byte history. It completed at 19:35:35 Istanbul:
`outputs/jobs/partial-audio-20261010T163421519Z/status.json`.
Both archives passed published MD5 checks; 128 train and 64 dev WAVs were
extracted. Total received audio archive bytes are 4,055,073,906; cumulative
charged payload is 4,056,652,110 bytes, below the original 4.5 GB cap.

Three parallel agents implemented a real alignment audit, the bounded pilot
runner and the supervised queue in separate files. Root review caught a
worker argument-forwarding issue before execution; a regression check now
requires execute/alignment arguments to survive bootstrap. Parent completion
also requires actual fitting and complete result files, rather than a zero
process exit alone. Fifteen runner/queue checks and five alignment checks
passed after the final fixes. No real model fit is claimed in this entry.
See [training sequence](docs/LOCALIZATION_TRAINING.md).

## 2026-10-10 - First localization pilot completed and reviewed

All 192 native label vectors match ceil(actual sample count / 160); no labels
were trimmed or padded. VAD can end 0–15 samples early in train, 0–14 in dev.
The final [alignment audit](docs/validation/localization_alignment.json)
matches its source. Native binary RTTM remains unvalidated. Three checked
metadata archive inventories have no RTTM-named entries; audio archive
reference members remain to be inspected.

The supervised job completed at 19:39:25 Istanbul:
`outputs/jobs/localization-pilot-20261010T163921940Z/status.json`.
Seed45 LFCC/StandardScaler/logistic used 128 train records, retained 4,524
frames per native class and scored 64 dev records. It converged in 14 solver
iterations without warnings. Worker computation took 1.281 seconds; lifetime
peak RAM was 143,302,656 bytes. External stage including startup took about
three seconds. Model, configuration and score hashes were retained.

Custom duration-weighted 20 ms EER is 39.50%. At fixed zero threshold,
missed spoof duration is 18.29 / 58.05 seconds and false alarms 84.20 / 175.09
seconds. These weak results are a small baseline, not official RangeEER,
SegmentEER or full benchmark results. An independent agent recomputed the
diagnostics and checked ten preselected SVG geometries, IDs and hashes:
[16-check review](docs/validation/localization_pilot_review_v1.json).
The [compact audit](docs/validation/localization_pilot_v1.json) and
[results with timelines](docs/LOCALIZATION_RESULTS.md) are public artifacts;
audio/model/detailed scores remain ignored. Detection PDF was not rebuilt.
Next: native RTTM adapter, metric compatibility and false-alarm analysis.

## 2026-10-10 - Range diagnostic, error review and one negative ablation

Parallel agents handled native-reference investigation, error analysis and
the preplanned recording-context experiment. The agent usage limit interrupted
final reference/metric work; root completed that adapter and evaluation from
the saved files. Completed training, failures and reviews were preserved.

The checked train/dev audio archives contain no RTTM/reco2dur members.
Earlier protocol/label/VAD inventories likewise contain no RTTM-named members.
Pinned upstream metric instructions expect con_data/rttm_2cls_0sil_spoof;
the checked rttm_tool converts RTTM to labels and does not establish its
inverse. Three small official source files were fetched and hash-verified;
no new audio download occurred. Publisher silence/sub-bin semantics remain
unverified. The [reference audit](docs/validation/localization_references_v1.json)
names the fallback explicitly: contiguous native 10 ms binary 0 runs, terminal
interval clipped to actual WAV duration; never VAD-derived truth.

The actual bounded patched upstream evaluator ran in the isolated metric
environment, external 120 s limit per model, scale 2 for 20 ms predictions.
Baseline derived-reference RangeEER is 39.3799%; context 42.9697%. Both searches
stop after 55 iterations on a repeated full state. Independent native-duration
arithmetic reproduces selected FPR/FNR within 1e-8. [Results and package/source
hashes](docs/validation/localization_range_results_v1.json) keep this custom
derived-reference diagnostic separate from the earlier pooled 20 ms EER and
from any official benchmark claim.

Error analysis at a fixed zero threshold found 1,349 predicted runs versus 109 reference
runs, with 989 predictions having zero reference overlap; median predicted
run was 60 ms. Native-reference speech false alarms were 45.26%, nonspeech false alarms were 53.61%.
The VAD-uncovered 27.875 ms tail remains separate. Short-run findings have
less than 1 second of reference duration and do not support a general claim.
Five edge-case tests and duration partition assertions passed. See
[error analysis](docs/LOCALIZATION_ERRORS.md).

The single frozen whole-recording median/MAD transform was tested with the
same seed 45, 128 train / 64 dev recordings and 9,048 retained train frames. It worsened custom
20 ms EER from 39.5002% to 42.9238% (+3.4236 percentage points), with more missed
spoof and false-alarm duration at zero. There was no parameter search. The
first attempt stopped on Windows status-file replacement after fitting;
an infrastructure-only bounded-I/O retry completed under a new prefit freeze.
Failed logs and both plans remain. Twenty independent context checks passed;
the review notes that original runner source bytes were not archived, so an
exact historical implementation diff cannot be reconstructed from hashes.
See [context audit](docs/validation/localization_context_v1.json) and
[review](docs/validation/localization_context_review_v1.json).

Root ran 50 focused localization tests and 26 partial-data tests successfully.
The report now includes localization methods, two preselected timelines,
fragmentation, the negative context comparison and explicit metric/reference
limitations. English prose stays simple; author and AI acknowledgment remain.
PDF/LaTeX and figure manifests record new inputs and outputs. Next work is a
separately frozen stronger-representation comparison, not more tuning of the
same median/MAD idea on this dev subset.

## 2026-10-10 - First frozen speech-representation comparison planned

An independent research agent compared official WavLM Base+ and wav2vec2
Base cards, processor configurations, papers and license links. It preferred
WavLM's middle layer as a research hypothesis. Root selected wav2vec2-base
final states as a simpler first control with an explicit Apache-2.0 model
card declaration. This is an operational choice, not evidence of better
localization accuracy. WavLM's card links a UniSpeech license; its code-tree
license must not be substituted for the checkpoint license.

Sources: [wav2vec2 card](https://huggingface.co/facebook/wav2vec2-base),
[paper](https://arxiv.org/abs/2006.11477),
[WavLM card](https://huggingface.co/microsoft/wavlm-base-plus),
[linked license](https://github.com/microsoft/UniSpeech/blob/main/LICENSE).

Only metadata was fetched: the official model API and three small pinned
files (README, config and processor). Revision, metadata SHA-256 values,
380,267,417-byte weight size and weight SHA-256 are recorded in
[configuration](configs/localization_ssl_v1.json). No weights, optional
environment packages, new audio or training were started.

The [plan](docs/LOCALIZATION_SSL_PLAN.md) fixes final 768-dimensional states,
per-recording processor normalization, nearest-center mapping onto the
unchanged clipped 20 ms grid, train-only head fitting, comparison metrics,
stop budgets and the first smoke ID before any model result. Five geometry
tests passed. The [offline preflight](docs/validation/localization_ssl_plan_v1.json)
verified all 192 audited sample lengths against the derived 400-sample
receptive field / 320-sample stride and preserved 32,710 target bins. Raw
aligned float32 cache size is about 96 MiB. Actual encoder output, peak RAM
and runtime still require the supervised smoke.

The Windows API probe measured 15.71 GiB total RAM and 4.12 GiB available;
disk was 57.51 GiB free. CIM was denied and nvidia-smi unavailable, so no
fresh GPU inventory is claimed; earlier Intel Iris Xe evidence remains.
Start requires 4 GiB available RAM and 20 GiB disk. No job was queued.
The baseline/report are unchanged. The next implementation is isolated
environment locking, restricted weight loading and a bounded train-only
resource smoke. Pretraining overlap and repeated dev use remain explicit
limitations; no unseen-corpus claim is supported.

The independent agent reviewed the final plan/config/preflight and brute-forced
sample lengths 400 through 9,999 against explicit nearest-center distances.
It found no blocking issue. Both memory limits must be enforced: available
system RAM may reach its stop boundary before worker peak RSS reaches its
separate cap. Root rechecked all plan source hashes; see the
[plan review](docs/validation/localization_ssl_plan_review_v1.json).

## 2026-10-10 UTC - Encoder setup, local smoke and Colab preparation

The optional Windows SSL environment is installed separately and hash-locked.
Imports, restricted `weights_only` loading and pip dependency checks passed.
Torch 2.10.0+cpu was selected after checking the publisher's
[patched-version advisory](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p).
The official 380 MB checkpoint downloaded in 45.44 seconds and passed exact
size/hash verification. The initial venv bootstrap needed a workspace TEMP
directory; a UTF-8 metadata read was corrected before package installation.

A worker agent implemented the local one-recording runner. Root implemented
the external Windows supervisor and verified seven focused tests, including
termination of the owned child on memory pressure. A second independent
agent reviewed model loading, geometry, source provenance and resource gates.

The preliminary local smoke completed: 7.3787 seconds of train audio,
1.0589 seconds extraction, 813.77 MiB peak RSS, 368 encoder frames mapped to
369 x 768 finite features. The independent review caught that the 4 GiB
available-RAM check was not repeated immediately before model loading. Root
retained the first sources/results, added the check and froze a second
attempt. The corrected attempt stopped before model loading because the
available-RAM check failed. It was not automatically retried. No dev audio,
label payload, eval or classifier fit was accessed. No full extraction began.
Both attempts are in the [smoke audit](docs/validation/localization_ssl_smoke_v1.json).

Following Ekin's authorized Colab fallback, root prepared a notebook and the
agent prepared a Linux/CUDA supervisor. Only the exact weights, metadata,
one public train recording, preliminary CPU reference and required small
source files enter the 382 MB ignored upload bundle. All archive payload
hashes were checked. Five notebook code cells compile; actual CUDA execution
has not occurred. CPU/GPU numerical differences will be reported rather
than silently mixing caches.

The Colab account was open in the in-app browser, but automatic approval
review rejected uploading the local notebook. Its stated reason was that
general authorization to use Colab did not specifically authorize sending
local project implementation details to that external destination. No upload
or cloud runtime was started. The next action requires explicit approval
for the named notebook and bundle; no workaround was attempted.
See [run state](docs/LOCALIZATION_SSL_RUN.md) and
[bundle inventory](docs/validation/localization_ssl_colab_bundle_v1.json).

## 2026-10-11 - Colab environment checks and transport fallback

Ekin explicitly approved uploading the named notebook and bundle, resolving
the earlier approval gate. The notebook is saved in a
[Colab session](https://colab.research.google.com/drive/1TP5Pyib95cAlYdLHTamyK5ZS7-uIhZ1r).
A free Tesla T4 connected with Python 3.13.15 and Torch 2.11.0+cu130.
The recorded snapshot had 11,790,440 kB available RAM and about 75.47 GB free
disk. No paid compute was selected.

An independent agent reviewed the actual runtime compatibility and warned
about global Colab dependency conflicts and optional audio/vision imports.
The original notebook's global encoder installation then failed `pip check`
with six conflicts. The output and installed environment were preserved.
Root prepared an isolated venv, retaining runtime GPU Torch and installing
the same hash-locked encoder dependencies. The agent reviewed that helper
and caught an incorrect CUDA matrix-sum expectation: 16 instead of 8.
The first failed probe was retained. The corrected probe and isolated
dependency check passed, including CUDA and Wav2Vec2Model imports. These
checks do not establish encoder extraction speed or localization quality.

The agent also prepared a separate clean
[v2 notebook](notebooks/ssl_colab_smoke_v2.ipynb), leaving v1 unchanged.
It resolves dependencies globally but installs encoder packages only into
the isolated environment. Its six code cells and embedded helper passed AST
checks. It is an unexecuted fresh-runtime recipe, not a reconstruction of
the actual session's failed global installation and first probe. The original
v1 has five code cells; the earlier six-cell statement was corrected.
Source hashes are in the [v2 audit](docs/validation/localization_ssl_colab_notebook_v2.json).

The embedded upload chooser failed three times. A Files-panel upload started
but was slow; an observed 18,874,368-byte partial file is not a complete
bundle. Reloading stopped the client transfer. Root prepared a smaller
1,214,355-byte payload ZIP with SHA-256
`44bf452f97f4ce8bc0ccc2d9741dd9ca50704e2df1bca136d0ae8ee480dbcd2b`.
The fallback [payload helper](scripts/prepare_colab_ssl_payload.py) verifies
this ZIP, then retrieves the unchanged checkpoint from the exact official
Hugging Face revision. It checks the frozen 380,267,417-byte size and SHA-256,
has a 180-second transfer budget and does not retry automatically or execute
the model. Audio, model and feature choices remain fixed.

After reconnection, the small payload uploaded successfully. The exact
380,267,417-byte official checkpoint downloaded in 4.6465 seconds and passed
the frozen SHA-256 check. The bounded GPU smoke has started with the isolated
Python and unchanged model runner. Completion is not yet established. No
192-recording extraction, classifier fit or training has started. Root will
update the final state after the bounded smoke.
See [run state](docs/LOCALIZATION_SSL_RUN.md) and
[session audit](docs/validation/localization_ssl_colab_session_v1.json).

GPU smoke update: the worker produced 369 x 768 finite features in 2.1058 s,
but the parent supervisor stopped during exit. The complete archive and
features were recovered and verified locally. This is not an accepted
resource-gate pass. No classifier fit or full extraction started.
See [GPU audit](docs/validation/localization_ssl_colab_smoke_v1.json).

The separate supervisor v2 is now reviewed and source-frozen; nine stdlib
regression tests passed without installing packages. It pins the original
worker and manifest and keeps all resource limits. Transfer to the existing
Colab notebook was rejected by browser security policy as permission denied.
No v2 cloud execution started and no workaround was attempted. Restore
browser access before the fresh bounded attempt; preserve v1 as stopped.
Evidence: [supervisor review](docs/validation/localization_ssl_colab_supervisor_v2.json).
# EkinOS milestone — 2026-10-11: supervised SSL smoke passed

Fresh Colab T4 execution of the frozen supervisor v2 passed the strict
single-recording gate on LA_T_1963829. Encoder extraction took 1.9117 s;
supervised duration was 12.6383 s; peak RSS about 1.263 GiB and GPU allocated/
reserved peak 453/498 MiB. Local verification checked archive/source hashes,
369 x 768 finite float32 features and independently reproduced CPU/CUDA
max absolute difference 0.0001273155 and RMS difference 0.000001425069.
This is a resource/numerical smoke, with no SSL classifier training, dev
evaluation, eval access or full 192-recording extraction. Previous stopped
attempts remain preserved. Next: review bounded resumable feature extraction.
Sources: [audit](docs/validation/localization_ssl_colab_smoke_v2.json),
[run guide](docs/LOCALIZATION_SSL_RUN.md), [frozen plan](docs/LOCALIZATION_SSL_PLAN.md).

# EkinOS milestone — 2026-10-11: 192 frozen SSL feature caches verified

The explicitly approved 128 train / 64 dev public PartialSpoof v1.2 bundle
completed bounded T4 extraction: parent exit zero, 18.2717 seconds, 1.394 GiB
external peak RSS. All 192 caches (32,710 aligned bins x 768 dimensions) and
their provenance, finite float32 arrays, archive SHA-256/CRC and resource
receipts passed local validation and independent agent review. No eval or
native frame-label arrays were opened. This is feature extraction, with no
classifier or performance result. The separate local head stopped before
worker launch because available RAM was below its unchanged 2 GiB gate;
the stopped attempt is preserved. Next is an explicit fresh bounded head
attempt after resource inspection, using these caches. 35 SSL tests passed;
frozen baseline/metric sources remain unchanged.
Sources: [extraction audit](docs/validation/localization_ssl_extraction_v1.json),
[head stop audit](docs/validation/localization_ssl_head_prelaunch_v1.json),
[run state](docs/LOCALIZATION_SSL_RUN.md).



## EkinOS milestone - verified SSL follow-up (2026-10-11)

Public PartialSpoof pilot:192 pinned wav2vec2-base caches verified. Original
max 300 SSL fit failed convergence and remains failed. The separately approved,
preregistered max 3000 train-only follow-up converged in 303 iterations ; 7.67 s,
316.9 MiB actual worker peak. Independent feature/scaler/source/PID verification
preceded fixed 64-dev scoring. Custom duration EER 28.57% (LFCC 39.50%); derived
RangeEER 28.75% (LFCC 39.38%). Fixed-zero false alarms 32.87 s (84.20 s), missed
spoof 23.09 s (18.29 s): a trade-off, not uniform improvement. Different optimizer
budgets, repeatedly inspected dev and whole-recording context limit claims.
Eval closed; no additional tuning. [Results](docs/LOCALIZATION_SSL_RESULTS.md),
[train audit](docs/validation/localization_ssl_convergence_train_v1.json),
[dev audit](docs/validation/localization_ssl_convergence_dev_v1.json),
[report](docs/report/audio_deepfake_report.pdf).
