# Audio Deepfake milestone

Özgür Ekin Sönmez — 2026-10-10

I finished the first detection experiment on public ASVspoof 2019 LA data.
Three CPU LFCC-GMM models each scored all 24,844 official dev recordings.
Mean clean EER was 6.34%, compared with 22.17% after a fixed bandwidth change
and 9.03% for the fixed score average. Absolute score change was also a weaker
error-ranking signal than decision margin: mean AUROC 0.385 versus 0.936.

The result did not support my original diagnostic idea. I kept the negative
result, checked all three seeds and documented the failures. These are dev
results, not independent validation; eval remains closed. No localization or
production accuracy is claimed.

The revised report has a simpler layout and grayscale mean-only figures;
per-seed numbers and SD remain in tables. It includes sources and an
acknowledgment of AI assistance with implementation, debugging, tests,
review and writing. A local CLI/browser demo now reports the fixed seed 43
model's raw score. CPU execution is a practical choice; the main question
remains fake-speech detection and error analysis.

A one-thread CPU check on 20 fixed bona fide dev recordings, repeated three
times, measured 12.40 ms mean, 11.64 ms median and 16.92 ms p95 decode/scoring
time. All 60 scores exactly matched saved scores. The process and file cache
were warm; model loading and browser overhead were excluded. The files are
from one speaker, so this is a limited local speed check. Next: check the
PartialSpoof data, labels, split overlap and official localization protocol
before a measured pilot.

Evidence: [report](report/audio_deepfake_report.pdf),
[results](RESULTS.md), [independent check](REPORT_EVIDENCE.md),
[audit](validation/full_robustness_results_v1.json),
[local demo](../README.md#local-demo),
[timing audit](validation/demo_latency.json),
[dataset](https://zenodo.org/records/6906306).


I also finished a first PartialSpoof v1.2 localization pilot. LFCC/logistic
trained on 128 fixed train recordings and scored 64 dev recordings. All 192
actual audio lengths match native binary labels, and ten prediction timelines
are saved. Custom 20 ms duration EER was 39.50%; false alarms were frequent.
This is a starting baseline, not official RangeEER or SegmentEER. Native RTTM
validation and error analysis come next. Evidence:
[results and timelines](LOCALIZATION_RESULTS.md),
[frozen audit](validation/localization_pilot_v1.json),
[independent review](validation/localization_pilot_review_v1.json),
[alignment checks](validation/localization_alignment.json).

I completed one fixed recording-context normalization experiment. It made
custom 20 ms EER worse, from 39.50% to 42.92%, so I kept the original baseline.
Error analysis found many fragmented false alarms. The available distribution
has no publisher RTTM reference; explicitly derived native-label range
diagnostics were 39.38% baseline and 42.97% context. These are custom small-dev
results, not official benchmark numbers. The PDF/LaTeX report now includes
the methods, timelines, failures and reference limitations.
Evidence: [updated report](report/audio_deepfake_report.pdf),
[comparison](LOCALIZATION_RESULTS.md), [metric details](LOCALIZATION_METRICS.md),
[independent context review](validation/localization_context_review_v1.json).

I planned the next comparison using frozen wav2vec 2.0 features and the same
small train-only classifier setup. The checkpoint version, license, feature
alignment and resource limits are fixed before downloading model weights.
Five geometry tests and the offline checks for all 192 recordings passed.
No new training result exists yet; the next step is a measured extraction
smoke on one fixed train recording.
Evidence: [experiment plan](LOCALIZATION_SSL_PLAN.md),
[configuration](../configs/localization_ssl_v1.json),
[preflight](validation/localization_ssl_plan_v1.json),
[official model card](https://huggingface.co/facebook/wav2vec2-base).

The separate encoder environment and verified checkpoint are ready. A
preliminary CPU measurement extracted a 7.38-second public train recording
in 1.06 seconds with about 814 MiB worker peak memory. It is not the accepted
resource gate: an independent review added a missing RAM check immediately
before loading, and the corrected attempt stopped at that check. Full
extraction and training have not started. A one-recording Colab notebook
and upload bundle were prepared locally; upload permission was still pending
at that earlier milestone.
Evidence: [run state](LOCALIZATION_SSL_RUN.md),
[local smoke audit](validation/localization_ssl_smoke_v1.json),
[upload inventory](validation/localization_ssl_colab_bundle_v1.json).

On 2026-10-11, I approved the notebook and bundle upload and connected a free
Tesla T4 in Colab. The runtime reported Python 3.13.15 and Torch 2.11.0+cu130.
Global Colab package conflicts blocked the first environment check. A separate
encoder environment now passes dependency, CUDA and model-import checks.
An independent review also caught an incorrect expected value in the first
CUDA probe; that failed attempt was kept before the corrected probe passed.

The large browser upload was slow and incomplete. A small approved payload
and a bounded direct fetch of the same pinned public checkpoint are prepared
as a transport fallback. The small upload completed, and the official
380,267,417-byte checkpoint downloaded in 4.65 seconds with the exact expected
hash. The bounded one-recording GPU smoke has started in the isolated
environment; its completion is still pending. Full extraction and classifier
training have not started. A new v2 notebook gives a clean isolated setup
recipe, but has not been run end-to-end.
Evidence: [current run state](LOCALIZATION_SSL_RUN.md),
[session audit](validation/localization_ssl_colab_session_v1.json),
[v2 notebook audit](validation/localization_ssl_colab_notebook_v2.json),
[official pinned checkpoint](https://huggingface.co/facebook/wav2vec2-base/blob/0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8/pytorch_model.bin).

GPU smoke update: the worker produced 369 x 768 finite features in 2.1058 s,
but the parent supervisor stopped during exit. The complete archive and
features were recovered and verified locally. This is not an accepted
resource-gate pass. No classifier fit or full extraction started.
See [GPU audit](validation/localization_ssl_colab_smoke_v1.json).

The separate supervisor v2 is now reviewed and source-frozen; nine stdlib
regression tests passed without installing packages. It pins the original
worker and manifest and keeps all resource limits. Transfer to the existing
Colab notebook was rejected by browser security policy as permission denied.
No v2 cloud execution started and no workaround was attempted. Restore
browser access before the fresh bounded attempt; preserve v1 as stopped.
Evidence: [supervisor review](validation/localization_ssl_colab_supervisor_v2.json).
