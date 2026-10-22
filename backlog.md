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

# Current step - fixed SSL head failed convergence; launcher repair

2026-10-11 continuation: Fresh RAM 5.27 GB allowed a new head-v2 fit from the
verified cache. lbfgs stopped at the planned 300-iteration convergence gate.
No model, dev score or metric was produced. The Windows venv redirector was
also incorrectly monitored; its reported peak is not valid worker RAM.
Preserve the failure; repair only the new SSL launchers with direct interpreter
bootstrap and a real allocation probe. A separately recorded train-only
convergence follow-up is pending a scope decision. Eval remains closed.
See docs/LOCALIZATION_SSL_RESULTS.md and
docs/validation/localization_ssl_head_convergence_v2.json.

# Historical step - 192 SSL caches verified; local head pre-launch RAM stop

2026-10-11: Explicitly approved public input transfer completed using final
reviewed v2 bundle (same 128 train / 64 dev scope). T4 extraction passed with
parent exit zero, 18.2717 s cumulative duration and 1.394 GiB peak RSS.
The complete archive and all 192 caches were independently verified locally.
See docs/validation/localization_ssl_extraction_v1.json.

Local head v1 stopped **before worker launch** at the unchanged 2 GiB RAM gate.
No classifier fit, dev scores or SSL metrics exist. Preserve its stopped
directory and docs/validation/localization_ssl_head_prelaunch_v1.json.
Next: fresh RAM inspection and explicit new bounded head attempt from the
verified cache, then the separately reviewed metric adapter. No encoder rerun,
automatic retry, limit relaxation or eval access. 35 SSL tests passed.
See docs/LOCALIZATION_SSL_RUN.md and docs/LOCALIZATION_SSL_EXTRACTION_RUN.md.

# Historical step - fresh supervisor v2 GPU smoke passed

2026-10-11: Colab access succeeded in this chat and a new free T4 runtime
connected. Old VM files were absent. The approved small payload and frozen
supervisor were reuploaded; fresh isolated dependency/CUDA checks and the
same pinned checkpoint size/hash verification passed.

The single-recording attempt completed with supervisor status complete and
exit code zero. Extraction took 1.9117 s; supervised duration 12.6383 s;
peak RSS 1,356,173,312 bytes; GPU allocated/reserved peak 453/498 MiB.
Local archive/source/feature validation passed, including 369 x 768 finite
float32 and independent CPU difference recomputation. See
docs/validation/localization_ssl_colab_smoke_v2.json and docs/LOCALIZATION_SSL_RUN.md.

No SSL classifier training or 192-recording extraction started. V1 remains
stopped and is not a gate pass. Next implement/review bounded resumable
extraction before scaling; baseline, metrics and report remain fixed.
The browser denial entries below are historical and were not bypassed.

# Historical step - corrected supervisor ready; browser permission denied

Read-only diagnosis on 2026-10-11 found general Colab browsing allowed,
but this chat's separate browser session still explicitly denies the same
origin. See docs/BROWSER_ACCESS_DIAGNOSTIC.md. No permission files edited;
the user needs a supported way to revoke the chat-specific denial.

2026-10-11 resume check: user authorized continuing, but the in-app browser
has no remaining tabs. Opening the saved notebook was rejected because a
saved user permission preference blocks colab.research.google.com. Chat
approval did not clear that setting. No alternate browser or access bypass
was attempted; cloud runtime state is currently unknown. User must remove
the saved site block before the corrected GPU check can run.

User explicitly approved the notebook and smoke payload transfer to Google.
Live notebook: https://colab.research.google.com/drive/1TP5Pyib95cAlYdLHTamyK5ZS7-uIhZ1r
Read docs/LOCALIZATION_SSL_RUN.md and the Colab smoke/session audits first.
T4, Python 3.13.15, Torch 2.11.0+cu130. Isolated encoder environment passes
pip check and CUDA/encoder import. Global dependency/probe failures retained.
The full upload was stopped while partial; the small payload and direct
pinned checkpoint fetch passed. GPU worker v1 produced verified features,
but supervisor status is stopped. Do not treat worker COMPLETE as gate PASS.
No classifier training or full 192-recording extraction started.

Completed: separate supervisor v2 implemented, reviewed, source-frozen and
nine stdlib regression tests passed. See
docs/validation/localization_ssl_colab_supervisor_v2.json. Original worker unchanged.
Next: restore browser access, transfer the frozen supervisor to the existing
Colab notebook, then make one fresh bounded attempt with original inputs.
The browser security policy rejected that transfer as permission denied.
No corrected code was uploaded and no second GPU attempt started.
Do not bypass this access block. The new blank Colab cell is
cell-4ExZphakunCk; the live runtime may expire, so inspect before continuing.
Keep previous attempts. New clean notebook v2 is an unexecuted recipe.
Do not modify the frozen LFCC baseline, metric code, or report yet.

# Latest milestone - frozen SSL comparison plan, 2026-10-10

The next experiment is specified in docs/LOCALIZATION_SSL_PLAN.md and
configs/localization_ssl_v1.json: frozen wav2vec2-base final states, 768
features, the same 128 train / 64 dev recordings and a matching logistic
head. Official checkpoint revision, weight size/hash and small metadata
hashes are recorded. No weights, optional packages or audio were downloaded.
No training was started. Five geometry tests passed; all 192 audited
recordings passed the offline geometry preflight.

Resume from docs/validation/localization_ssl_plan_v1.json. This is a plan,
not a model result or an executed encoder shape check. Next implement:
1. An isolated locked encoder environment and restricted weights-only loader.
2. A bounded verified checkpoint fetch and externally supervised smoke on
   LA_T_1963829 (the longest selected train recording, about 7.38 seconds).
3. Resumable feature extraction only after smoke passes, then a separate
   768-dimensional head with the same sampled train frame IDs as LFCC.
4. The frozen metric/timeline comparison, retaining failures and closed eval.

The first probe passed the 4 GiB available-RAM start gate narrowly. Recheck
before loading; do not assume that model/framework peak will fit. Keep the
baseline, metric environment, earlier model source and current report fixed.
The report contains completed results only. Historical entries follow.

# Latest milestone - localization comparison and report complete, 2026-10-10

Completed: native-reference search, explicit native 10 ms-derived range
diagnostic, baseline error analysis, one frozen median/MAD context ablation,
independent context review and extended PDF/LaTeX report. No eval used and
no new audio downloaded. Original baseline remains unchanged.

Custom 20 ms EER: baseline 39.50%, context 42.92%. Derived-reference bounded
patched upstream RangeEER: baseline 39.38%, context 42.97%. Context did not
help. Both range operating points match independent duration arithmetic.
Publisher RTTM is absent from checked archives; exact publisher parity is
unverified, so never label these results official benchmark numbers.

Current evidence:
- docs/LOCALIZATION_RESULTS.md and docs/LOCALIZATION_METRICS.md
- docs/validation/localization_context_v1.json and localization_context_review_v1.json
- docs/validation/localization_references_v1.json and localization_range_results_v1.json
- docs/LOCALIZATION_ERRORS.md and docs/report/audio_deepfake_report.pdf

Do next: plan one stronger acoustic representation before any new model
download or training. Fix the checkpoint/license, feature geometry, train
sampling, resource budget and dev comparison beforehand; keep eval closed.
Avoid repeated tuning on the small inspected dev subset. Publisher RTTM
remains a documented external/reference issue, not an excuse to rename the
derived reference official. Local demo still serves the full-recording
detector; do not promote weak localization predictions as reliable decisions.

First context attempt: outputs/jobs/localization-context-20261010-v1 stopped
after fitting on a Windows status replace sharing error; no dev result.
Same-rule infrastructure retry completed in localization-context-20261010-v2.
Metric outputs: outputs/localization/range-derived-v1. No job remains queued.
Historical entries follow; their old pending tasks have been superseded.

# Latest milestone - localization pilot complete, 2026-10-10

Latest job: `outputs/jobs/localization-pilot-20261010T163921940Z/status.json`
(complete, actual training true, no lock remains). All 192 real audio/native
label alignments passed. Seed45 used 9,048 balanced train frames and scored
64 dev recordings. Custom 20 ms duration EER 39.50%; many false alarms.
Independent review passed 16 checks. This is not official RangeEER/SegmentEER.
Read docs/LOCALIZATION_RESULTS.md and docs/validation/localization_pilot_v1.json.

Next:
- Inspect native binary RTTM in the verified train/dev audio archives;
  upstream reference clue is `con_data/rttm_2cls_0sil` (no extension).
  Use a bounded reader, avoid eval and preserve provenance. No new download.
- Validate the native reference/time-grid adapter before the documented
  bounded patched upstream RangeEER run; name custom diagnostics separately.
- Examine false alarms and boundaries on the ten preselected timelines.
- Freeze the planned recording-context normalization ablation before comparing.
- Extend PDF/LaTeX after metric review and error analysis.

Entries below are historical. Do not restart completed work because an older
entry says pending.

# Latest audio resume - 2026-10-10, 19:06 Istanbul

Update at about 19:15 Istanbul: train archive complete and published MD5
verified (`c4853ddd831e8e96b0e279fc0a512e7e`); selected train audio extracted.
The same job is now downloading dev, confirmed growth to 542,113,792 bytes
(26.7%). Available RAM was 6,414,098,432 bytes. Training has not started.

RAM recovered to 6,542,790,656 available bytes; free disk 62,062,759,936 bytes.
Resumed the unchanged bounded downloader. Latest job:
`outputs/jobs/partial-audio-20261010T160629331Z/status.json`, launch supervisor
13608, initial worker 11120. Running status and train byte growth to
1,530,391,332 confirmed. Preserve all pins, partial bytes and cumulative
charges. Check this job first; earlier entries below are historical.

# Latest audio stop - 2026-10-10

The third job stopped at 18:52:09 Istanbul on the RAM guard. Both processes
have exited; no lock remains. Train retains 1,527,245,604 bytes (75.34%).
The latest RAM probe returned 1,245,564,928 available bytes, below the 2 GiB
floor, so no new job was launched. Preserve partial bytes and the cumulative
1,528,823,808-byte charge. Next: restore RAM headroom, recheck resources,
then resume according to docs/LOCALIZATION_RUN.md. Dev and training remain
pending. Entries below record earlier launches.

# Latest audio resume after RAM stop - 2026-10-10

The second job stopped on the audio RAM guard at 18:45:12 Istanbul.
1,461,185,316 partial bytes were retained. Fresh available RAM was
2,513,739,776 bytes, above the unchanged 2 GiB floor; free disk was
62,040,276,992 bytes. A new job was launched at 18:51, supervisor PID 21944:
`outputs/jobs/partial-audio-20261010T155124521Z/status.json`.
Check this latest job and the transfer journal first. No guard was relaxed,
no budget reset, no training started. Earlier entries below are historical.

# Latest audio resume - 2026-10-10

The first audio job stopped after a connection EOF at 18:37:47 Istanbul.
889,711,396 partial bytes were retained. A new detached job was launched at
18:38 with unchanged source pins, selection and cumulative transfer budget:
`outputs/jobs/partial-audio-20261010T153826745Z/status.json`.
Supervisor launch PID 12592, initial worker PID 14892. Check actual status
and byte growth; process identities are historical. The earlier job remains
preserved. No training has started. See [resume guide](docs/LOCALIZATION_RUN.md).

# Current localization work - 2026-10-10

Metadata verification is complete. Read [LOCALIZATION_RUN.md](docs/LOCALIZATION_RUN.md)
before continuing. The bounded audio preparation job was launched at 18:08
Istanbul time; it does not train a model. Inspect the live
[transfer audit](docs/validation/partial_audio.json) and
`outputs/jobs/partial-audio-20261010T150830073Z/status.json` for actual status.
Supervisor launch PID 16136, initial train worker PID 22720; verify process
identity before acting. Partial bytes and cumulative budget must be preserved.

Completed: four metadata files (92,230,352 bytes), MD5 checks, fixed seed 45
128/64 balanced subset, 8+8 smoke IDs, ten balanced dev timeline IDs; strict
native-label adapters, feature mapping, temporal model API, SVG timeline,
resumable audio downloader, and isolated evaluator compatibility checks.
40 focused project tests and three separate metric tests passed.

Open: finish/verify train and dev archives, inspect real sample counts and
upstream binary RTTM, resolve terminal-bin alignment, freeze model/config,
then fit the bounded pilot and produce real prediction timelines. Nonspeech
VAD labels cannot replace native binary truth. 11 train / four dev metadata
vectors differ from ceil(VAD duration / 10 ms). The candidate strict adapter
must be tested with actual audio, not relaxed automatically. Metric reporting
must name the documented bounded patched upstream variant.

No real-data localization training or accuracy evaluation has started.
Earlier preparation and detection entries below are historical snapshots.

# Localization preparation — 2026-10-10

PartialSpoof preparation has started. Read the [pilot plan](docs/LOCALIZATION_PILOT_PLAN.md)
and [protocol review](docs/PARTIALSPOOF_PROTOCOL.md) first for the new work.
The detection report and its frozen sources remain unchanged.

Current gate: verify small protocol metadata locally, then resolve evaluator
conformance and real audio/label alignment before freezing subset IDs or training.
Planned first fit: 128 train / 64 dev recordings, seed 45, LFCC plus a temporal
logistic baseline; eight recordings per split first for alignment smoke checks.
These are draft sizes, not a selected or trained dataset. Audio acquisition
may require full ~2 GB train/dev archives; establish exact download/storage
budgets before fetching them. Eval stays closed. No localization training or
audio download has started.

# Latest milestone

The revised [project report](docs/report/audio_deepfake_report.pdf),
[LaTeX source](docs/report/audio_deepfake_report.tex), figures and GitHub results
pages use a simpler layout, mean-only grayscale figures, the author's full
name Özgür Ekin Sönmez, sources and an AI-assistance acknowledgment. The
three-seed negative result is unchanged. An
[EkinOS summary](docs/EKINOS_SUMMARY.md) is ready. Eval remains closed.

The local [CLI/browser demo](README.md#local-demo) is complete. It uses the
fixed, hash-verified seed 43 model, accepts bounded mono 16 kHz WAV/FLAC
inputs and reports raw scores. No authenticity probability is claimed.
The [fixed latency plan](docs/validation/demo_latency_plan.json) and
[measured audit](docs/validation/demo_latency.json) record 60 observations
from 20 first official dev rows, three passes, one numerical thread:
12.40 ms mean / 11.64 ms median / 16.92 ms p95. The sample is one bona fide
speaker; cache/process warm, model load and browser overhead excluded.

Next research milestone: a small PartialSpoof localization pilot. Begin by
checking public dataset/version/license, label time alignment, official
splits and current evaluation protocol. Document source overlap with
ASVspoof 2019; do not present it as independent cross-dataset validation.
Choose a bounded subset and model before downloading or training, inspect
current resources, and freeze the pilot plan. No localization run has
started. Preserve the completed detection/robustness protocols and audits.

All launch/stop/recovery entries below are historical. The completed robustness
audit, revised report and local demo supersede their next-action text. No
new training or dataset download was started during either reporting milestone.

# Backlog

Updated: 2026-10-10. Scope A is selected: ASVspoof 2019 LA, CPU LFCC-GMM, then one narrowband robustness experiment. See [the starting plan](docs/START_PLAN.md).

## Current state and next work

### Narrowband RAM stop and recovery launched — 2026-10-10

Current authoritative status:
`outputs/jobs/full-robustness-recovery-20261010T104130247Z/status.json`.
Supervisor PID 10840, launched 13:41 Istanbul time. Read this status first;
the original job status deliberately remains stopped.

Original job stopped at 13:33:13 Istanbul during seed 44's first chunk:
`Resource stop: insufficient available system RAM`. Worker peak was
217,456,640 bytes, about 207 MiB; the running system RAM floor was 1 GiB.
The 26 completed chunks for seeds 42/43 are intact and passed exact input,
score-join and source-pin validation. They will not be recomputed.

The separate recovery source files preserve original execution pins and
protocol. All cached audio is reverified, then only seed 44 is scored. Its
original failed 90.844 seconds and smoke 3.1026651 seconds count against the
same 1,800-second seed budget. Recovery pins include the original status,
failure log, preflight, smoke and completed chunks. Failure history will be
included in the final public audit. Eight focused recovery tests and the
actual interpreter bootstrap passed; parallel review found no blocker.
Fresh available RAM before launch was 5,501,374,464 bytes, free disk about
63.1 billion bytes; Intel Iris Xe remains the GPU. Resource thresholds were
kept unchanged.

Next: inspect recovery status. If complete, review the full result audit and
update README/RESULTS before the demo latency step. If stopped again, preserve
the new logs/chunks and review the exact reason; no automatic lock takeover or
repeat launch. See [the recovery guide](docs/FULL_ROBUSTNESS_RUN.md).

### Full-dev narrowband job launched — 2026-10-10

Authoritative status:
`outputs/jobs/full-robustness-20261010T095134725Z/status.json`.
Supervisor PID 30012; launched 12:51 Istanbul time. An independent native
process check confirmed it survived the launching terminal and entered
`preflight` with worker PID 18740. This is scoring only, with no training or
download. The supervisor verifies all audio, runs the fixed 20-record smoke
per seed, scores seeds 42/43/44 sequentially in 2,000-record chunks, then
validates and publishes results. Read current status before assuming completion.

Twenty focused checks passed, including frozen-protocol, worker, supervisor
and finalizer tests. Parallel review found and resolved the initial smoke and
metric-path interface mismatches. All frozen source pins stayed unchanged.
The execution sources are pinned separately in
`outputs/robustness-full-v1/execution.json`. Guards include one numerical
thread, 3.5 GiB available RAM at stage start, 1 GiB running floor, 2 GiB worker
peak, 20 GiB disk reserve and 30 minutes per seed including chunk startup and
measured smoke time. Resource waits are separate. See
[the run guide](docs/FULL_ROBUSTNESS_RUN.md).

Next session: inspect status and logs first. If complete, read
`docs/validation/full_robustness_results_v1.json`, update RESULTS/README with
the measured comparison and failure cases, then measure local demo latency.
If stopped, preserve chunks/logs and investigate before any recovery. Do not
change execution sources during the job or rerun completed clean training.
The older freeze entry below is historical; its next-action text is superseded.

### Full-dev narrowband protocol frozen — 2026-10-10

Protocol: `docs/FULL_ROBUSTNESS_PLAN.md`; compact immutable contract:
`docs/validation/full_robustness_protocol_v1.json` (SHA-256
`806fd09f488449d6e276e4b36b60ddc016441be2b62e7a223d3bc74d7ea6d913`).
The freeze covers all fixed seeds 42/43/44, exact models/clean scores, source and
package pins, transformation, comparisons, thresholds and reporting rules.

New full-ID partition: 9,609 calibration records / 8 speakers (945 bona fide,
8,664 spoof), and 15,235 analysis records / 12 speakers (1,603 bona fide,
13,632 spoof). All six attacks occur in both groups; groups and train speakers
are disjoint. Full IDs/CSV manifests are under ignored
`outputs/full-robustness-protocol-v1/`; the pilot audit was preserved.

Per-model thresholds from clean calibration bona fide scores are frozen:
seed42 0.7167018055915833; seed43 0.7260986566543579;
seed44 0.7200403809547424. All accept 898/945 calibration bona fide records.
Five tests and a parallel review passed. Rebuilding the contract and checking
all saved partition hashes after writing matched exactly.

Next: implement/test a separate guarded full-dev narrowband runner that reads
and verifies this contract. Reuse the saved models; do not retrain, alter the
pilot runner guards, retune or select one favorable seed. Then launch sequential
bounded transformed scoring after rechecking machine resources. This turn
performed metadata freezing only: no transformed scoring, training or download.
Clean dev/pilot results were already inspected; this is not blind independent
validation. Eval remains closed.

### Three-seed confirmation complete — 2026-10-10

The sequential job completed successfully at 04:44:56 Istanbul time. Seeds
42/43/44 each have 24,844 complete official dev scores. All mixtures converged
without warnings. Mean EER is 6.3441% (population std 0.3316 percentage points);
mean min t-DCF is 0.172367 (population std 0.014112). Eval remains closed.

Post-run checks verified seed 43/44 score/model hashes, official input provenance,
training IDs and exact dev score order. Direct official-reference metric
recomputation and aggregate recomputation matched exactly. Evidence:
`docs/validation/full_multiseed_summary.json`,
`docs/validation/full_multiseed_review.json`, `docs/RESULTS.md`.
Both workers exited successfully; no further run is scheduled by this job.

Next: freeze a separate full-dev narrowband protocol with a new full-dev speaker
partition audit before transformed scoring. Keep the same transformation and
ranking definitions; do not select a favorable seed or retune from these results.
Then measure the local file-scoring demo latency. Historical launch notes below
are preserved and superseded by this completed state.

### Remaining seeds launched sequentially — 2026-10-10

Current authoritative job:
`outputs/jobs/full-multiseed-20261010T012814223Z/status.json`.
Started 04:28 Istanbul time, supervisor PID 19852. Independent launch check:
`running`, stage `seed43`, worker PID 10992. The RAM gate passed with
5,078,228,992 bytes available and 63,663,435,776 bytes free disk.
The worker first verifies inputs, then fits and scores seed 43. Seed 44 starts
only after seed 43 passes completion checks; no concurrent fits or seed 42 rerun.

Next session: read this job status first. If complete, inspect
`docs/validation/full_multiseed_summary.json` and the last report entry, then
independently review seed 43/44 artifacts and update `docs/RESULTS.md` with the
three fixed seeds. If stopped, inspect its named stage and preserve outputs.
The job will not overwrite existing run directories or silently retry a seed.
All 11 multiseed tests passed; native process query confirmed independent launch.
The public audit excludes local absolute commands/paths; full commands remain
in ignored job logs. See `docs/MULTISEED_RUN.md`.

After that review: freeze the full-dev narrowband protocol and new full-dev
speaker partition audit before transformed scoring. Then continue the local
file-scoring demo/latency milestone. Eval and localization remain separate gates.

### Seed 42 complete and independently verified — 2026-10-10

The recovery finished successfully at 03:44 Istanbul time. All 24,844 official
dev scores are present: EER 5.8902%, minimum normalized t-DCF 0.152678.
Both mixtures converged without warnings. The independent review verified
all 50,224 audio hashes, complete protocol metadata/order, unchanged model and
score prefix, package/source pins and six exact waveform score reproductions.
Direct official-reference recomputation matched every reported metric.

Evidence: `docs/validation/full_independent_review.json`,
`docs/validation/full_baseline_seed42.json`, `docs/RESULTS.md`.
The same-dev EER-threshold error audit is `docs/validation/full_seed42_errors.json`:
150 bona fide rejections, 366 A04 accepts and 949 A06 accepts; one threshold tie.

Next: run the preplanned seeds 43 then 44 sequentially with unchanged settings.
The prior-result gate passed against the real independent review. Their launcher
and guarded aggregation are implemented; see `docs/MULTISEED_RUN.md`.
After completion, inspect the multiseed job status and
`docs/validation/full_multiseed_summary.json` before planning the full narrowband
confirmation protocol. Do not reuse the pilot partition audit for full dev.
Eval remains closed. Historical launch/stop observations below are preserved.

### Scoring-only recovery launched — 2026-10-10

Current job: `outputs/jobs/scoring-recovery-seed42-20261010T004116241Z/status.json`.
Started at 03:41 Istanbul time, supervisor PID 28900. The first independent
check showed `running`, stage `preflight`, worker PID 19432. Available RAM at
the ready gate was 5,094,617,088 bytes; free disk 63,709,179,904 bytes.

The Windows probe memory leak is fixed and tested. The saved model, all 25,380
training IDs, original source/config pins and 22,815-score prefix passed the
real checkpoint smoke check. Three existing scores reproduced exactly.
No original artifacts were changed. No retraining or eval scoring is scheduled.

The detached job automatically performs preflight, three remaining-score chunks,
final input checks, complete dev metrics, independent score validation, a compact
public audit and a report entry. See `docs/SCORING_RECOVERY.md`.

Next session: read this job's current status first. If `complete`, inspect
`outputs/scoring-recovery-seed42/summary.json`,
`docs/validation/full_baseline_seed42.json` and the final entry in `reports.md`.
Full dev scores then support planning the frozen robustness confirmation; eval
stays closed until its separate gate is satisfied. If `stopped`, preserve all
outputs and read the named stage's stderr before recovery. Do not relaunch into
the existing recovery directory or repeat training automatically.

### Latest check: model fitted, dev scoring stopped — 2026-10-10

Job `outputs/jobs/full-seed42-20261009T235915974Z/` stopped at 03:26 Istanbul.
Preparation completed successfully in 708.906 seconds: all 50,224 records verified.
The model was fitted from all 25,380 train recordings and saved at 03:21:48.
Dev scoring stopped at 22,815 / 24,844 records when process peak crossed the
configured 2 GiB budget (observed 2,147,807,232 bytes). This was an explicit
resource stop, not the timeout or evidence of Codex closing the process.
Native queries found supervisor and both recorded workers absent.

- Saved model: `outputs/baseline-full-seed42/model.joblib`; SHA-256
  `701fc3d8107cc8b2696b5552dd9e1b4751e44d2d197cfd5d1dfe7e4814add101`
  matches `run_state.json`.
- Partial `scores.csv`: 22,815 unique dev IDs, all finite; no full metrics or
  successful summary exist. Do not report this as a completed experiment.
- Next: implement and test scoring-only recovery from this verified saved model,
  validating the existing score prefix against the official dev protocol and
  preserving the failed run. Investigate the scoring memory growth before retry.
  Retraining and audio preparation should not be necessary.
- No recovery job has been launched by this status check. Eval remains closed.

### Restart launched, waiting for RAM — 2026-10-10

The new detached supervisor started at 02:59 Istanbul time (PID 19124).
Its first independent check showed `waiting_resources`, with no worker yet:
3,318,919,168 bytes available RAM, below the 3.5 GiB headroom gate.
The startup wait expires at 03:14 Istanbul time if resources do not become ready.

- Current authoritative status: `outputs/jobs/full-seed42-20261009T235915974Z/status.json`.
- Preparation now has 60 minutes; the fixed seed 42 runner still has 30 minutes.
- Each stage can wait up to 15 minutes for RAM before its worker starts.
- Old preparation PID 16532 was absent in a native process query; only its stale
  `outputs/full-data/prepare.lock` was removed. Audio and journals were preserved.
- Next: read this new status and logs first. If stopped for resource wait expiry,
  free RAM and review before relaunching. If complete, independently audit model,
  all dev scores and metrics. Do not infer training success from the launch.
- Seeds 43/44 and eval remain unscheduled. No Colab job was started.

### Stopped full train/dev preparation — 2026-10-10

The detached seed 42 pipeline started at 02:27 Istanbul time and stopped at
02:47 because preparation exceeded its 20-minute engineering budget. It verified
47,750 / 50,224 records (95.1%) before the timeout. Training never started.
No audio/format error was logged. Saved files and the preparation journal remain.
The journal's last `running` state is stale; supervisor `stopped` is authoritative.

- Authoritative status: `outputs/jobs/full-seed42-20261009T232740327Z/status.json`
- Historical supervisor PID: `11404`; preparation worker PID: `16532` (stopped).
- Logs: the same timestamped job folder; see `docs/FULL_RUN.md`.
- Intended result: `outputs/baseline-full-seed42/summary.json` with `status: ok`.
- Next: review remaining files and the stale preparation lock/PID before recovery.
  Use a revised preparation time budget based on the measured throughput; current
  resume revalidates existing files, so it does not process only the remainder.
  No restart or Colab job has been launched. Seeds 43/44 and eval are not scheduled.

Earlier startup trials are preserved. A Windows venv redirector monitoring bug
was caught and the initial preparation worker stopped before training. The fixed
launcher monitors the real interpreter; PID/package-version regression tests passed.
The historical job above used the fixed implementation.

### Completed pilot

The fixed 3,000-record pilot and narrowband experiment are complete and independently
checked. Clean EER: 8.8%; min t-DCF: 0.2006. Absolute score change lost to the simple
margin baseline for ranking errors. Detailed evidence is in `reports.md` and
`docs/validation/pilot_independent_review.json`. Eval stays closed.

Local completed runs: `outputs/baseline-pilot-seed42/` and
`outputs/robustness-pilot-seed42/`. These paths already exist; use new output
directories for repeat runs. Data/model artifacts remain ignored.

Next, in order:

1. Recover the timed-out preparation and then measure seed 42 before scaling.
   Preparation and runner are implemented and tested. Keep model settings
   fixed, preserve natural class proportions and exclude 142 unkeyed dev members
   through exact protocol joins; do not retune on eval.
2. Check MATLAB LFCC parity or continue explicitly labeling this model an adaptation.
3. Lock the final protocol before extracting/scoring eval; keep custom stress results separate.
4. Add a small result page or plot and measure local demo latency with a clear timing scope.
5. Start the PartialSpoof label/RTTM pilot after the detection milestone is documented.

The verified archive and recovery chunks are cached locally. The abandoned
Zenodo chunk cache is only an earlier speed trial. Optional cache cleanup can
wait; keep the completed archive and verified pilot audio for reproducibility.

## Done

- [x] Read the project brief and repository instructions.
- [x] Check CPU, GPU, RAM and free disk.
- [x] Research datasets, licenses and official evaluation with two parallel agents; record sources in `reports.md`.
- [x] Create notes, research report and starting plan.
- [x] Select scope A and keep localization as the next milestone.
- [x] Add the Python package, CLI checks, pilot config and simple English README.
- [x] Add ignore rules for datasets, recordings, environments, caches and model weights.
- [x] Remove personal biography and private vault paths from the public brief.
- [x] Initialize local Git. No remote publication was requested.

## Step 1 - complete

- [x] Install and verify the project in the local isolated Python environment.
- [x] Run dependency, FLAC round-trip and audio input checks; record exact package versions in `requirements/windows-py312.txt` and `reports.md`.
- [x] Check Git ignore rules for environment, cache, dataset, model and secret-file paths.

## Step 2 - fixed pilot audio complete

- [x] Choose ignored local storage: protocols in `data/metadata/asvspoof2019/`, later audio under `data/asvspoof2019/LA/`. Do not add C: and G: free space together until their relationship is known.
- [x] Recheck free RAM/disk before the bounded download: about 2.94 GiB available RAM and 78.94 GiB free disk; use sequential processing and a small download cap.
- [x] Save ASVspoof 2019 LA version, license, published archive checksum, protocol URLs and observed SHA256 metadata. Full archive size/MD5 are now independently verified.
- [x] Parse all 121,461 official CM rows; validate labels/attacks, duplicate IDs and cross-split speaker separation. Only pilot audio availability is verified.
- [x] Keep eval out of development. Export fixed class-balanced 2,000 train / 1,000 dev pilot lists with seed 42 and class/attack distributions.
- [x] Add a small metadata-only fetch script with hash/size validation and no audio-archive fallback.
- [x] Retrieve only the frozen 100-record train/dev sample through official archive-member URLs, with no full-archive fallback.
- [x] Verify selected paths, SHA256/size, 16 kHz mono PCM16 format and finite/non-empty audio. Full-split audio is still absent; `--audio-root` continues to mean full-split existence checks.
- [x] Choose access for the remaining 2,900 pilot recordings: separate resumable archive command with published size/MD5 and pinned manifests. Existing smoke downloader caps are unchanged.
- [x] Download and verify the official archive; extract and independently validate only the fixed 3,000 train/dev files. See `docs/validation/pilot_data_seed42.json` and `docs/PILOT_DATA.md`.
- [ ] Review the code license choice before public release. Dataset and third-party licenses remain separate.

## Steps 3-6 - baseline and first experiment

- [x] Measure runtime, peak RAM and feature size on 100 train/dev recordings. Resource summary: `docs/validation/audio_pilot_100.json`.
- [x] Implement LFCC extraction and document differences from the official MATLAB reference in `docs/LFCC.md`.
- [x] Verify official 2019 EER/min t-DCF reference, score-ID joining, dev ASV scores, costs and golden examples. External-reference adapter and tests are ready; see `docs/METRICS.md`.
- [ ] Compare Python LFCC outputs with a shared MATLAB fixture before claiming exact baseline reproduction.
- [x] Add a hash-pinned external evaluator/ASV setup command, preserving its separate license. Current integration tests skip if reference files are absent; the real cache passed `--check-only`.
- [x] Pin the matching official dev ASV file in the training/scoring runner; eval is excluded.
- [x] Implement a bounded-frame streaming LFCC-GMM and guarded experiment runner; train and independently verify the 50-train / 50-dev smoke run. See `docs/validation/baseline_smoke100.json`.
- [x] Export the combined frozen 3,000-record selection (SHA-256 `fe2811f468705c571a5e2e617d0623d3d18fdd61782129bf07e5dc2d11b472df`) and wire pinned selection/journal paths in `configs/pilot.toml`.
- [x] Run and independently verify the 32-component, 100k-frame baseline on the fixed pilot. Both class models converged without warnings.
- [x] Freeze the narrowband transform, speaker-disjoint calibration/analysis protocol and partition before full-pilot results. Signal processing and reliability helpers pass unit tests.
- [x] Compare original and 16-to-8-to-16 kHz scores; use the frozen BF-only calibration threshold on separate dev speakers.
- [x] Keep clean pilot and custom stress results in separate tables; include bona fide errors and attack-level failures.
- [ ] If the pilot fits the budget, run full train/dev and locked eval with seeds 42/43/44.
- [x] Add the trusted local file-scoring CLI and check that a real smoke-model score matches its saved dev score. It returns a diagnostic score, not an authenticity claim.
- [x] Deliver the full-pilot report and first robustness result, including the negative reliability finding.
- [ ] Measure demo latency before reporting it; single-file score consistency is checked, but not a latency benchmark.

## Colab - when useful

- [ ] Keep training logic in the Python package; notebook only handles setup and commands.
- [ ] Check actual GPU/VRAM, RAM, disk and package versions at session start.
- [ ] Save the commit, config, seed and manifest with each run.
- [ ] Keep data on the VM disk while running and save checkpoints/logs to persistent storage.
- [ ] Verify checkpoint resume before a long run. No subscription or Colab session has been created.

## Later - localization

- [ ] Use PartialSpoof v1.2; inspect archive layout before assuming a small pilot means a small download.
- [ ] Verify 10 ms labels, RTTM and 20 ms output alignment, including silence, padding and spoof=0 mapping.
- [ ] Build LFCC + frame classifier, then a small 1D TCN if needed.
- [ ] Add the official RangeEER adapter, SegmentEER and utterance EER; select demo thresholds on dev.
- [ ] Ablate local features vs recording median/MAD context; analyze failures by manipulated duration and spoof fraction.
- [ ] Test smoothing separately and check whether it removes short fake sections.
- [ ] Consider a small official SSL inference pilot only after measuring time/RAM.
- [ ] Defer LlamaPartialSpoof external evaluation; verify version, eval-only protocol, license and source overlap first.

## Open limits and deferred work

- [ ] ASVspoof 5 full distribution is 142.3 GB and exceeds current free disk. Revisit selected files and code licenses later.
- [ ] ASVspoof 2019 and PartialSpoof share source material; testing between them is not independent cross-dataset evidence.
- [ ] Smoke and 3,000-record pilot training/scoring time and process RAM are measured. Full official benchmark accuracy and demo latency remain unmeasured.
- [ ] Codec/noise expansion, abstention, ONNX and API deployment follow the first baseline.
- [ ] Review rough notes and local paths before public release. Do not commit private audio, company data or credentials.
- [ ] Source-linked milestone summaries are in `reports.md`; no personal vault was updated.
