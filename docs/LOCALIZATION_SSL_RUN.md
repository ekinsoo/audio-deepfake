# Current step - SSL convergence follow-up verified (2026-10-11)

The original fixed 300 SSL fit remains a failed convergence experiment. A
separately preregistered train-only max 3000 follow-up converged in 303 iterations,
7.67 s external duration and 316.9 MiB actual worker peak. Independent retained
feature-byte/scaler/PID/resource verification passed before any dev scoring.
The repaired launchers use direct base Python;36 SSL tests passed.

Separate no-fit scoring completed the same fixed 64 dev IDs and 10 timelines.
Custom pooled20ms duration EER: SSL 28.57%, frozen LFCC 39.50%; derived native
RangeEER diagnostic:28.75% vs39.38%. At fixed zero, false alarms decreased
84.20 s ->32.87 s but missed spoof increased18.29 s ->23.09 s. Different optimizer
budget, repeated dev inspection, whole-recording context and unknown encoder
pretraining overlap prevent equal-budget or unseen-domain claims. Eval closed.

Evidence: docs/validation/localization_ssl_convergence_train_v1.json and
localization_ssl_convergence_dev_v1.json; docs/LOCALIZATION_SSL_RESULTS.md.
Report updated with separate protocol/failure/results/timelines. Next work
requires a separately fixed independent evaluation protocol; no adaptive solver,
seed or threshold search. Older entries below are historical.

# Frozen encoder smoke: current state

## Latest: fixed head stopped at convergence gate — 2026-10-11

The user requested continuation and authorized Colab if useful. Fresh local
RAM was 5,269,901,312 bytes, with 60,061,753,344 free disk bytes. A new
`outputs/localization/ssl-head-v2` invocation started fitting from the verified
caches. Fixed lbfgs max_iter 300 emitted ConvergenceWarning, which stopped the
worker before any model/dev score/result was saved. No SSL metric exists.
See [fitting audit](validation/localization_ssl_head_convergence_v2.json) and
[SSL results status](LOCALIZATION_SSL_RESULTS.md).

The outer monitor accidentally measured the Windows venv redirector; its
4,648,960-byte peak is invalid as numerical-worker RAM. Repair is limited to
the separate SSL launchers using the existing direct-base-Python bootstrap.
Original logs/config/sources are preserved and frozen baseline helpers remain
unchanged. A separate train-only convergence protocol is pending a scientific
scope decision; no automatic iteration increase or dev evaluation.

The extraction below remains independently verified. Older head/pre-launch
descriptions are historical snapshots.

## Latest: 192-record extraction verified; head stopped before launch — 2026-10-11

After explicit user approval, the final reviewed `ssl_extraction_inputs_v2.zip`
(14,698,353 bytes) was uploaded to the existing T4 session. V1 input preparation
remains historical and was never executed. Supervisor extraction completed
128 train / 64 dev recordings with exit zero in 18.2717 seconds; external
peak RSS was 1,496,715,264 bytes (1.394 GiB). No eval or native frame-label arrays
were opened; selection metadata includes utterance classes/label summaries.

The 93,337,242-byte result archive was recovered without re-extraction.
Normal download and HTML output encountered transport/display failures;
12 visible-output parts were combined and the whole SHA-256, ZIP CRC/safe paths,
executed sources, identity, resource receipts and all 192 finite float32
arrays independently verified. There are 32,710 aligned 768-dimensional bins.
See [extraction audit](validation/localization_ssl_extraction_v1.json).

The separate local head invocation stopped at its **pre-launch** RAM guard:
2,004,860,928 bytes were available immediately before invocation, below the
2 GiB minimum. No worker, fitting, model, dev score or metric run started.
The stopped directory is preserved; no automatic retry or relaxed limit.
See [head pre-launch audit](validation/localization_ssl_head_prelaunch_v1.json).

Next: inspect fresh local RAM with adequate margin, then explicitly launch a
fresh head output directory using the already verified cache. Do not rerun
encoder extraction. The same-frame sampling preflight passed, and 35 SSL tests
passed. The separate metric adapter was independently reviewed but not executed.
The baseline and frozen metrics remain unchanged. The older entries below
describe their state at the time; they are not current status.

## Latest: fresh supervised GPU smoke passed — 2026-10-11

Colab access succeeded in the current chat without changing permission files.
A new free T4 VM had no old payload or environment, so the approved small
payload, frozen supervisor v2 and setup helpers were uploaded again. The
isolated dependency check and CUDA/encoder import probe passed; no encoder
packages were installed globally. The same pinned checkpoint was fetched
once in 7.292 seconds and verified by exact byte count and SHA-256.

One fresh attempt on `LA_T_1963829` completed under supervisor v2, with
parent/worker exit code zero and supervisor `status: complete`. The original
worker, manifest, feature geometry and resource limits were unchanged.
The exit-time zombie process lacked VmRSS/VmHWM; v2 recorded that event,
observed owned-child exit and completed its final validation.

- Encoder extraction: 1.9117 seconds; entire supervised attempt: 12.6383 seconds.
- Observed worker peak RSS: 1,356,173,312 bytes (about 1.263 GiB).
- Immediate pre-load available RAM: 11,316,109,312 bytes; disk stayed above 20 GiB.
- GPU peak allocated/reserved memory: 453/498 MiB.
- Encoder states: 368; aligned output: 369 x 768 finite float32.
- Independently recomputed CPU/CUDA maximum absolute difference: 0.0001273155;
  RMS difference: 0.000001425069. Numerical diagnostic only, with no speedup claim.

The recovered archive and local feature/source checks passed. Evidence:
[fresh v2 smoke audit](validation/localization_ssl_colab_smoke_v2.json).
Artifacts remain ignored under `outputs/colab/session-v2/`. The browser
download event timed out; the same archive was recovered through visible
notebook output and its size/hash verified before extraction.

This is a single-recording resource/numerical gate, not classifier training
or a benchmark result. No labels, dev audio or eval were opened. No full
192-recording extraction started. V1 remains stopped; the preliminary CPU
reference remains a numerical reference, not a past strict gate pass.
Next: implement/review resumable extraction with source/environment provenance
and per-recording caches before the separately bounded 192-recording run.
Keep the baseline, metrics and report fixed. Historical entries follow.

The optional Windows CPU environment is installed and locked separately from
the baseline. PyTorch 2.10.0+cpu, Transformers 4.57.6, NumPy 2.2.6 and
SoundFile 0.13.1 passed imports, a restricted-loading compatibility probe and
`pip check`. Direct wheel URLs and SHA-256 hashes are in
`requirements/ssl-windows.lock`. This lock is Windows/Python 3.12 specific.
Do not use its wheels in Colab or install it into the baseline environment.

The official 380,267,417-byte checkpoint is present under ignored
`data/models/wav2vec2-base-v1/`. Its size and SHA-256 match the frozen plan.
Download took 45.44 seconds. The downloader does not silently replace an
existing attempt or automatically restart a failed transfer.

## Two retained local attempts

The preliminary CPU smoke (`outputs/jobs/ssl-smoke-v1`) completed on the
7.3787-second longest selected train recording, `LA_T_1963829`:

- Feature extraction: 1.0589 seconds; cold encoder load: 0.2021 seconds.
- Entire supervised worker: 9.703 seconds, including imports and validation.
- Worker lifetime peak: 853,299,200 bytes (813.77 MiB).
- Actual encoder frames: 368; aligned output: 369 x 768 finite float32.

An independent review caught a missing repeat of the 4 GiB start check
immediately before model loading. The first measurement therefore does not
qualify as the strict resource-gate pass. Its features are a preliminary
numerical reference only. The corrected worker and helper hashes were frozen
before a fresh attempt, `outputs/jobs/ssl-smoke-v2`. That attempt stopped at
the immediate pre-load RAM check. It was not automatically restarted.
Both attempts, logs, plans and exact source snapshots are retained.

No classifier was fitted. Neither smoke opened labels, dev audio or eval.
The 192-recording extraction has not started. Deprecation warnings about
the legacy checkpoint configuration and `torch_dtype` remain in the logs.

## Colab progress: payload verified, bounded GPU smoke started

The notebook `notebooks/ssl_colab_smoke.ipynb` and ignored bundle
`outputs/colab/ssl_colab_smoke_v1.zip` are ready locally. The bundle is
381,663,715 bytes (about 382 MB). It contains only:

- The already verified model weights, model card and processor/config files.
- One public PartialSpoof train recording, with its source attribution.
- The preliminary CPU feature reference for that recording.
- The geometry helper, portable bounded CUDA runner and hash manifest.

There are no labels, dev/eval recordings, private recordings, credentials,
company data or full project checkout in the bundle.

The initial upload was rejected by automatic approval review because general
Colab permission did not specifically authorize transferring the local payload.
Ekin then explicitly approved the named notebook and bundle. That approval
resolved the upload gate. The notebook was uploaded to the
[live Colab session](https://colab.research.google.com/drive/1TP5Pyib95cAlYdLHTamyK5ZS7-uIhZ1r).
The older bundle inventory preserves the state before that approval.

A free Tesla T4 runtime connected. The hardware probe reported Python
3.13.15, PyTorch 2.11.0+cu130, 11,790,440 kB available RAM and about 75.47 GB
free disk. These are a recorded snapshot; the worker must repeat its checks.
No paid compute was requested.

The v1 notebook resolved and hash-locked the encoder dependencies, then
installed them globally. Its global `pip check` failed with six conflicts:
missing IPython/jedi, and version conflicts involving google-colab/requests,
gcsfs/fsspec, diffusers/huggingface-hub, gradio/huggingface-hub and
datasets/fsspec. The failure and environment were preserved. A separate venv
now reuses the runtime GPU Torch dependency directories and installs the
same hash-locked encoder packages. Its dependency check and CUDA/encoder
import probe passed. The first probe incorrectly expected a matrix sum of
16 instead of 8; its failure was retained before the corrected probe passed.
This was an environment check, not encoder extraction.

The new `notebooks/ssl_colab_smoke_v2.ipynb` is a clean fresh-runtime recipe
that avoids global encoder installation. Its six code cells and embedded
helper pass AST checks, but it has not been executed end-to-end. The original
v1 notebook has five code cells and remains unchanged. Neither recipe should
be described as a replay of the actual session's failed attempts.

Three embedded upload-widget chooser attempts failed. A Files-panel upload
started but transferred slowly; the observed partial file was 18,874,368
bytes before a reload stopped the client transfer. That is not a complete
or verified bundle. The transport fallback is now prepared: upload a
1,214,355-byte ZIP containing the small approved payload, then fetch only the
unchanged checkpoint directly from the official pinned Hugging Face revision
inside Colab. The small ZIP SHA-256 is
`44bf452f97f4ce8bc0ccc2d9741dd9ca50704e2df1bca136d0ae8ee480dbcd2b`.
`scripts/prepare_colab_ssl_payload.py` checks it, limits the checkpoint to
380,267,417 bytes, applies a 180-second transfer budget and validates the
same frozen checkpoint hash. It has no automatic retry and does not run the
model. This changes transport, not weights, audio or feature rules.

The small payload upload completed after reconnection. The official
checkpoint fetch took 4.6465 seconds and passed exact size and SHA-256 checks:
380,267,417 bytes and
`3249fe98bfc62fcbc26067f724716a6ec49d12c4728a2af1df659013905dff21`.
The unchanged GPU worker completed, but its parent supervisor stopped
with an empty error during process exit. This is not a strict gate pass.
Worker source SHA-256 remains
`c3c38b21e0d1b3a672831bff57cd8fec33d9dc7f2d91e2227176b11528e21337`.
The extraction took 2.1058 seconds; 368 states aligned to 369 x 768 finite
float32 features. GPU peak allocated/reserved memory was 453/498 MiB.
The supervisor sampled 1,356,595,200 bytes peak process memory.
Independent local validation reproduced maximum CPU/CUDA absolute difference
0.0001273155 and RMS difference 0.0000014251. This is a numerical diagnostic,
not a speedup or accuracy claim. The cold GPU and local CPU timings use
different hardware and Torch builds.

The entire result archive was recovered from visible notebook output after
two browser download attempts failed. Its SHA-256, feature checksum, shape,
dtype and finite values were checked locally. A disappearing /proc memory
field is a plausible exit-race cause; the parent traceback was not captured.
Preserve v1 as stopped and review/test a separate supervisor before retrying. No classifier has been fitted and the
192-recording extraction has not begun.
Keep the 4 GiB pre-load / 2 GiB runtime available-RAM limits, 3 GiB worker
peak, 20 GiB disk floor and 180-second worker timeout. Report CPU/CUDA
numerical differences and GPU memory before considering full extraction.

## Evidence

- [Experiment plan](LOCALIZATION_SSL_PLAN.md)
- [Environment audit](validation/localization_ssl_environment_v1.json)
- [Checkpoint receipt](validation/localization_ssl_checkpoint_v1.json)
- [Both local smoke attempts](validation/localization_ssl_smoke_v1.json)
- [Colab bundle inventory](validation/localization_ssl_colab_bundle_v1.json)
- [First GPU smoke audit](validation/localization_ssl_colab_smoke_v1.json)
- [Current Colab session audit](validation/localization_ssl_colab_session_v1.json)
- [Clean v2 notebook audit](validation/localization_ssl_colab_notebook_v2.json)
- [PyTorch advisory and patched versions](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p)

The separate supervisor v2 is now reviewed and source-frozen; nine stdlib
regression tests passed without installing packages. It pins the original
worker and manifest and keeps all resource limits. Transfer to the existing
Colab notebook was rejected by browser security policy as permission denied.
No v2 cloud execution started and no workaround was attempted. Restore
browser access before the fresh bounded attempt; preserve v1 as stopped.
Evidence: [supervisor review](validation/localization_ssl_colab_supervisor_v2.json).


## Completed follow-up artifact locations

- Train-only accepted stage: `outputs/localization/ssl-convergence-train-v1`
- Fixed-dev no-fit stage: `outputs/localization/ssl-convergence-dev-v1`
- Frozen metric/error adapter: `outputs/localization/ssl-convergence-diagnostics-v1`
- Compact verification: `docs/validation/localization_ssl_convergence_train_v1.json`
  and `localization_ssl_convergence_dev_v1.json`
- Ten public derived timeline SVGs: `docs/figures/ssl_convergence_v1`

Default invocations of `run_ssl_convergence_train.py`,
`score_ssl_convergence_dev.py`, and both `verify_ssl_convergence_*.py`
print a plan only. Train execution requires a fresh ignored output directory.
Scoring requires the separate independent train verification JSON; it has no fit
call. Both retain source, cache, model, process and resource evidence. The metric
adapter keeps the original upstream evaluator and LFCC sources unchanged.
Do not rerun the encoder or overwrite any completed/stopped directory.

### Test environments

Final checks passed 347 main-environment tests plus 3 RangeEER tests in the
existing isolated metric environment, 350 total. The 36 SSL checks include a real
Windows 64 MiB allocation/PID probe. Running every test inside the detection
venv alone fails the isolated RangeEER import because pandas is intentionally
absent there. No dependency installation or frozen metric source change was
needed. See `docs/validation/localization_ssl_final_checks_v1.json`.
