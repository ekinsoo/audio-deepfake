# Localization preparation and resume

The pilot, one context ablation and derived-reference metric checks are complete.
The PDF/LaTeX report now covers detection and localization;
see [localization results](LOCALIZATION_RESULTS.md). Read the [pilot plan](LOCALIZATION_PILOT_PLAN.md),
[metadata audit](validation/partial_metadata.json), and
[metric compatibility review](RANGE_EER_CONFORMANCE.md).

## Current audio job

Latest training job: `outputs/jobs/localization-pilot-20261010T163921940Z/status.json`.
Completed at 19:39:25 Istanbul, with actual training confirmed, no fit warnings
and unchanged source pins. Model/scores are in its ignored `pilot/` directory.
Do not repeat completed preparation or fitting. Context completed under
`outputs/jobs/localization-context-20261010-v2/`; its first failed status-I/O
attempt is retained separately. Range diagnostics completed under
`outputs/localization/range-derived-v1/`. Publisher RTTM was not found in
the checked distribution, so its parity remains unverified. False-alarm
analysis is in LOCALIZATION_ERRORS.md. Next: plan a stronger representation,
with one frozen comparison and compute budget. Preparation history follows.

Completed at 19:35:35 Istanbul on 10 October. Both train and dev archives
passed their published MD5 checks; all 128 train and 64 dev selected WAVs
have completion receipts and individual hashes. Audio preparation is finished.
The following resume description is historical. Continue with
[the training sequence](LOCALIZATION_TRAINING.md), not another audio download.

Resumed at 19:34 Istanbul on 10 October. Train is complete and its publisher
MD5 is verified. The earlier job left a stale running status and lock at
1,959,788,544 dev bytes. Both recorded processes (13608/20840) were absent;
the reviewed lock contained 20840. The review is preserved in that job's
`recovery_review.json`; the shutdown cause is unknown. Only this stale lock
was removed. Existing limits, pins and cumulative transfer history remain.
Available RAM before resuming was 5,463,306,240 bytes. Byte growth in the
new job confirmed a resume. Launch identities are historical; inspect status
and processes before acting.

- Status: `outputs/jobs/partial-audio-20261010T163421519Z/status.json`
- Logs: `stdout.log` and `stderr.log` in the same job folder.
- Transfer progress: [partial_audio.json](validation/partial_audio.json).
- Ignored files and receipts: `data/partialspoof/`.

The process runs separately from the launcher shell. It downloads only the
official train/dev archives, checks their full MD5 before inspection, and
extracts only the fixed 128/64 WAV files. It never starts training. Total
archive size is 4,055,073,906 bytes. Cumulative charged payload budget is
4.5 GB across runs, including failed reads; network/TLS overhead is excluded.

Each split has a 30-minute cooperative budget and a 1,865-second external
worker timeout. A slow connection may stop before an archive finishes.
Partial files and charged-byte history are kept. A stopped download is not
a completed preparation or a failed model experiment. Do not erase journals
to reset budgets. System sleep, shutdown or loss of connectivity can stop
network progress.

From PowerShell at the project root:

```powershell
Get-Content docs/validation/partial_audio.json
Get-Content outputs/jobs/partial-audio-20261010T163421519Z/status.json
Get-Content outputs/jobs/partial-audio-20261010T163421519Z/stderr.log -Tail 20
```

The earlier job stopped at 18:37:47 after a connection EOF. Its 889,711,396
partial bytes were retained. This job resumes that exact offset with unchanged
source/metadata/selection pins and cumulative transfer budget. The transfer
journal may retain the earlier `last_error` while the new attempt is running;
use current job logs and byte growth to distinguish a new failure.

The second job stopped at 18:45:12 on the RAM guard, retaining 1,461,185,316
bytes. The latest launch followed a fresh resource check above the unchanged
2 GiB available-RAM floor. No safety limit or cumulative budget was relaxed.

## Resume rules

1. Read the current status and error, then verify supervisor/worker processes
   are no longer running before starting another job. The downloader also
   uses an exclusive lock.
2. Preserve `.part` files, `audio_*_events.jsonl`, the public transfer journal
   and immutable completed receipts. HTTP resumes must return exact matching
   `206 Content-Range`. A server returning `200` cannot be appended to a
   partial archive.
3. After a forced process kill, a stale `audio_prepare.lock` requires an
   explicit review of its PID before removal. Never remove an active lock.
4. Recheck RAM/disk and run the new job launcher. Completed splits with
   matching receipts are skipped. This does not change the transfer budget.

```powershell
# Default is read-only: print the transfer plan.
.venv/Scripts/python.exe scripts/prepare_partial_audio.py

# Only after status/process/resource checks: start a new hidden bounded job.
powershell -ExecutionPolicy Bypass -File scripts/start_partial_audio.ps1
```

The supervisor pins its source, downloader, metadata and selection hashes.
Do not edit pinned inputs during an active job. If the resource or time limit
needs revision, preserve the earlier plan and evidence before designing a
separate recovery path. The acquisition script refuses changed active pins.

## After both archives are verified

Audit the eight train and eight dev smoke files against their actual sample
counts, native 10 ms arrays, VAD timestamps and the upstream binary RTTM.
The audio preparer currently extracts selected WAV files only; add a separate
reviewed metadata-reader step for the binary RTTM rather than assuming VAD
label 0 is equivalent.

The strict candidate `labels_for_audio` helper must pass actual files before
use. Some metadata lengths differ from VAD duration by one bin. Do not trim
or pad labels silently, or discard selected IDs after observing predictions.
Keep native binary labels unchanged: VAD nonspeech can be either class.

Next freeze the LFCC mapping, terminal-bin policy, normalization, seed 45,
frame reservoir, convergence requirement, timing/memory guards and source
hashes. Only then run the bounded LFCC/logistic pilot. Train normalization
uses retained train frames only. Treat convergence warnings as incomplete
training. The provided model API has synthetic tests, not a fitted dataset
model. The timeline renderer has no real predictions yet.

Real-data metric results must name the bounded patched upstream evaluator
and its hashes, retain an external timeout, and state any reference adapter
differences. Synthetic conformance does not establish paper parity. Eval
audio and metadata payloads remain unused.

## Focused verification

Use the existing detector environment for timeline, features, model and
data-preparation tests. Use the separate metric environment for evaluator
tests; its dependency versions differ deliberately.

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -p 'test_localization*.py' -v
.venv/Scripts/python.exe -m unittest discover -s tests -p 'test_partial*.py' -v
.cache/partial-metric-env/venv/Scripts/python.exe -m unittest discover -s tests -p test_range_eer.py -v
```

On Windows, if temporary-file access is restricted, set `TEMP` and `TMP` to
the project's `.cache/tmp` folder first. The RangeEER checks intentionally
reproduce an eight-second timeout in the earlier compatibility variant.
