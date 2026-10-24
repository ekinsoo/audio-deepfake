# Run the full narrowband experiment

This run reuses the three saved models and cached public ASVspoof 2019 LA
audio. It does not train or download data. The fixed choices are in
[the frozen plan](../FULL_ROBUSTNESS_PLAN.md).

On the original Windows environment, start from the project root:

```powershell
powershell -NoProfile -File scripts/start_full_robustness.ps1
```

The launcher prints a supervisor PID and a `status.json` path. It starts the
actual Python interpreter in a hidden window, so closing the terminal or
Codex does not stop the job. Keep Windows awake; shutdown and restart stop it.
Read the printed status file to see the current stage, child PID, elapsed time,
peak memory and any stop reason. Each stage also has stdout/stderr logs next
to that file.

The supervisor checks resources, verifies all 50,224 train/dev audio files,
then runs a fixed 20-record smoke check for each model. The smoke reproduces
saved clean scores within an absolute tolerance of 1e-10 and measures
transformed inference without choosing new parameters. Full scoring starts
only after these checks pass.

Models 42, 43 and 44 run sequentially in chunks of at most 2,000 dev recordings.
Each chunk starts a fresh process. One numerical thread is used. Available
RAM must be at least 3.5 GiB before each stage and 1 GiB while it runs. Worker
peak memory is limited to 2 GiB; disk reserve is 20 GiB. All scoring chunks
for one seed share a 30-minute wall-time budget including worker startup.
Resource waits are separate and stop after 15 minutes per stage.

Recoverable chunk files and execution pins are under the ignored
`outputs/robustness-full-v1/`. The finalizer requires complete ordered score
joins for all three seeds, rechecks clean metrics, and verifies every audio
hash again before publishing results. The compact public audit is
`docs/validation/full_robustness_results_v1.json`; it appears only after all
checks pass. Narrowband and average results are custom dev diagnostics;
eval remains closed.

If a stage stops, preserve its status, logs, lock information and partial
chunks. The launcher refuses existing output. Review the stop reason and
process ownership before building an explicit recovery step; do not delete
the output or automatically take over a lock. There is no automatic retry or
new training. The frozen contract is historical and stays unchanged when
execution finishes.

## Recovery after the first RAM stop

The first run stopped during seed 44's first chunk when available system RAM
fell below 1 GiB. Seeds 42 and 43 have 26 complete, verified chunk files.
The saved models and the fixed scientific protocol are unchanged.

For this exact stopped state, a separate recovery launcher is available:

```powershell
powershell -NoProfile -File scripts/start_full_robustness_recovery.ps1
```

It refuses an active lock, a present original supervisor PID or previous
recovery artifacts. It pins the original status, failure log, smoke, preflight
and completed chunks, checks all audio again, and runs only seed 44. The
failed attempt's 90.844 seconds and seed 44's 3.1026651-second smoke remain
inside the original 30-minute seed budget. The independent recovery audit
checks the original finalizer's successful-stage rules and additionally
requires the failure history and total time including failed work. New
recovery code has separate source pins; original execution sources stay fixed.

The launcher prints a new status path. Use that status for the active job;
the first run's status intentionally remains `stopped`. If recovery stops,
keep its artifacts too; this launcher is for one reviewed recovery, not
automatic repeated retries.
