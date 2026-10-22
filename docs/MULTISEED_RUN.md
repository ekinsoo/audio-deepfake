# Finish the fixed three-seed confirmation

Seed 42 has complete official dev scores and an independent audit. Both
mixtures converged without warnings. The remaining fixed seeds are 43 and 44;
they run sequentially with the same data, LFCC definition, frame budget and GMM
settings. Only the experiment seed changes. No new dataset download or audio
preparation is needed, and eval remains closed.

## Start once

```powershell
& .\scripts\start_full_multiseed.ps1
```

The launcher returns a hidden supervisor PID and the exact timestamped status
path under `outputs/jobs/full-multiseed-*/`. It uses the actual Python interpreter
with project venv packages. Keep the computer awake and on power.

Before fitting, the job requires a matching successful independent seed 42
review, verifies recovered scores/model/provenance and checks that the new
configs differ only in seed. Existing seed outputs and conflicting job locks
are refused. Source/config hashes are frozen for this launch.

Each worker rechecks official protocols and all train/dev audio hashes, fits
one seed and scores every official dev recording. It saves the model before
scoring, then writes a successful summary only after complete scoring/metrics.
Outputs are `outputs/baseline-full-seed43/` and `outputs/baseline-full-seed44/`.

## Limits and results

Before each seed, wait up to 15 minutes for two 3.5 GiB available RAM readings,
five seconds apart. Each runner has a 30-minute wall-time limit, including
preflight. Keep at least 20 GiB free disk, stop below 1 GiB available RAM or
above 2 GiB process peak. The Windows probe leak is fixed; budgets are unchanged.

`status.json` names the current seed, owned worker PID, resource observations
and stop reason. Per-seed stdout/stderr logs are in the same job directory.
Do not mistake a fitted model or partial score file for a complete result.

When both seeds finish, the job validates their artifacts and writes
`docs/validation/full_multiseed_summary.json`, then appends measured results to
`reports.md`. Report each seed plus mean and population standard deviation
(`ddof=0`) for EER and minimum normalized t-DCF. Nonconverged seeds stay in the
report; no seed is silently replaced or selected for its score. Three seeds
describe variation, not statistical significance. Seed 42 recovery timing is
separate from the new uninterrupted run timings.

If a worker fails, logs and outputs remain. The job does not automatically
retry, change hyperparameters, overwrite a model or advance to eval. Review the
last status before restarting. There is no scheduled agent wakeup or notification.
