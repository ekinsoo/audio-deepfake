# Recover full dev scoring

The seed 42 model finished training. The original run stopped during dev scoring
when its process peak crossed 2 GiB. Keep that run intact under
`outputs/baseline-full-seed42/`; it is still a failed run, not a completed result.

The cause was the Windows memory probe: each call created new ctypes structures
and pointer types retained by Python's pointer cache. About 119,000 checks
accumulated roughly 2 GiB. Cached structures fix this without changing LFCC,
GMM fitting or score calculation. Both the worker and supervisor probes use
cached Windows types. The original source hashes remain in the old job record;
recovery records the instrumentation change separately.

## Start once

From the repository root:

```powershell
& .\scripts\start_scoring_recovery.ps1
```

The launcher starts a hidden independent process and prints the exact job status
path. Keep the computer awake and on power. Do not launch a second recovery.
The script refuses existing recovery output and job locks.

## Automatic steps

1. Verify the complete official train/dev selection and preparation journal,
   audio hashes, original model hash, training IDs, configuration, source and
   package provenance. Validate the existing 22,815 score rows against the exact
   official dev prefix and check a few scores again with the saved model.
2. Score only the remaining 2,029 recordings in fresh worker processes of at
   most 1,000 recordings. Write complete chunks atomically in a separate output
   directory. No model fitting, dataset download or eval scoring is involved.
3. Recheck provenance, combine the original prefix and complete chunks in
   official order, then compute EER and minimum normalized t-DCF with the pinned
   official 2019 v1 evaluator and dev ASV scores.

Output: `outputs/scoring-recovery-seed42/`. Only a successful `summary.json`
and a complete 24,844-row `scores.csv` represent finished dev scoring. The summary
records recovery provenance and timing; it must not pretend this was one
uninterrupted full run. These are development diagnostics for a Python baseline
adaptation. Eval remains closed.

The supervisor keeps the 2 GiB process peak limit, 1 GiB running RAM floor and
20 GiB free disk reserve. Before each step it waits up to 15 minutes for two
consecutive 3.5 GiB available RAM readings. Each worker has a separate 30-minute
wall-time budget, including checks. `waiting_resources` means no worker is
active; `running` names the current step; `stopped` records a reason.

Read the launcher's timestamped `outputs/jobs/scoring-recovery-seed42-*/status.json`
and per-step stdout/stderr files when checking later. No automatic agent wakeup
or scheduled notification is attached. A stopped job requires review before a
retry; saved chunks and original artifacts are never silently overwritten.

After successful completion, the supervisor independently verifies the final
score hashes and every row against the pinned official dev protocol. It then
writes the small public audit `docs/validation/full_baseline_seed42.json` and
appends the measured dev metrics to `reports.md`. The large model and score
files stay in ignored output directories.
