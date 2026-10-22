# Run full train/dev locally

This job prepares the complete official train/dev audio from the cached archive,
then runs **seed 42 only**. It has no OpenAI API calls. Eval audio stays closed.
See [the fixed plan](FULL_CONFIRMATION_PLAN.md) for settings and limits.

## Start on Windows

Use the project environment and run from the repository root:

```powershell
& .\scripts\start_full_run.ps1
```

The launcher starts a hidden Python supervisor and returns its PID and status
file. The process runs independently of the chat. Keep the computer awake and
connected to power. Do not run another launcher while the job lock exists.

On Windows, the launcher uses the venv's underlying Python executable directly
and explicitly loads the project venv packages. This avoids supervising a venv
redirector instead of the actual worker. Tests check that the monitored child
PID equals the worker's own PID and that imported package versions match.

The job first validates the cached archive, extracts only the 50,224 official
train/dev members and verifies their audio. The 142 unkeyed dev files are excluded.
It writes a separate journal under ignored `outputs/full-data/`; the previous
pilot journal and results are preserved. No network download is performed.

After preparation succeeds, training and clean dev scoring run with the fixed
configuration in `configs/full_seed42.toml`. Only the current class-balanced
100k-frame reservoir is fitted, although every official train recording
contributes candidate frames. Dev retains its original class proportions.

## Check later

The launcher prints the exact timestamped folder under `outputs/jobs/`:

- `status.json`: supervisor status, current stage, PIDs, elapsed time, resource
  observations, exit codes and any stop reason.
- `prepare.stdout.log` / `prepare.stderr.log`: preparation progress and errors.
- `baseline.stdout.log` / `baseline.stderr.log`: runner output and errors.
- `supervisor.stdout.log` / `supervisor.stderr.log`: supervisor diagnostics.
- `launch.json`: the launch PID and command.

Read the printed status path directly, for example:

```powershell
Get-Content -LiteralPath outputs/jobs/EXACT-JOB-FOLDER/status.json
```

`running` means the pipeline is active; `stage: prepare` means training has not
started yet. `waiting_resources` means no worker is running: the supervisor is
waiting for RAM before the named stage. `stopped` includes the reason and must be reviewed. `complete`
requires successful preparation and complete model/dev score artifacts.
Do not infer success from an existing directory or a partial score file.

The actual runner output is `outputs/baseline-full-seed42/`. During the run,
`run_state.json` records progress. The fitted `model.joblib` is saved before dev
scoring starts. `summary.json` with `status: ok` is written only after all 24,844
dev scores and metrics complete. If the supervisor terminates a child, the
runner's last progress can still say `running`; supervisor status is authoritative
for whether that process is active.

## Limits and interruption

Before each stage, the supervisor waits up to 15 minutes for two consecutive
readings, five seconds apart, with at least 3.5 GiB available RAM. This adds
headroom above the workers' unchanged 3 GiB startup requirement. The job lock
remains held and no worker starts while waiting. Each wait has its own recorded
deadline; expiry stops the job. The two waits can total at most 30 minutes and
are separate from worker runtime budgets. Free disk must be at least 20 GiB;
insufficient disk stops immediately. Preparation also checks the missing audio
requirement plus the disk reserve.
The supervisor checks resources about once per second. It stops its own child
if process peak memory exceeds 2 GiB, available system RAM falls below 1 GiB,
preparation exceeds 60 minutes or the full runner exceeds 30 minutes. The runner
also checks resources between recordings. These are engineering budgets, not
guaranteed runtimes. The supervisor's runner budget includes preflight checks.

A stopped run preserves logs and existing artifacts. Do not delete output
directories or stale locks automatically. Verify their PIDs are no longer
running and inspect the reason first. Preparation can resume after that review;
an existing model/run directory requires explicit recovery work or a new run
path. This first launcher intentionally refuses to overwrite a previous seed 42
run. It does not automatically retry, resume model fitting, or launch seeds 43/44.

Ask for a result check later. There is no scheduled notification or automatic
agent wakeup attached to this process.
