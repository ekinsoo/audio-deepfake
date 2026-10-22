# Full train/dev confirmation

Written: 2026-10-10, after the first pilot. Status: fixed protocol; execution
status is tracked in [backlog.md](../backlog.md) and local job artifacts.

## Purpose

Check whether the small CPU detector remains useful on the complete official
ASVspoof 2019 LA train/dev splits. Keep the model settings fixed rather than
tuning against the disappointing score-change result. This is a confirmation
run of an adaptation, not an exact reproduction of the MATLAB baseline.

The pilot used 2,000 balanced train and 1,000 balanced dev recordings. The
complete official splits have different class proportions:

| Split | Total | Bona fide | Spoof |
| --- | ---: | ---: | ---: |
| Train | 25,380 | 2,580 | 22,800 |
| Dev | 24,844 | 2,548 | 22,296 |

Preserve every official train/dev ID and its original split. Do not balance,
subsample or move records for full-set evaluation. Keep eval audio closed.
The archive's dev directory contains 142 additional unkeyed `LA_D_A*.flac`
members beyond the 24,844 official CM dev records. They were identified from
ZIP member names only, without decoding. Exclude them by joining exact official
protocol paths; never create the full-dev manifest with a directory glob.
The source and protocol hashes are recorded in [DATA_PROTOCOL.md](DATA_PROTOCOL.md)
and `configs/datasets/asvspoof2019_la.json`.

## Fixed model and run order

Use the existing 60-dimensional LFCC definition, no normalization, 32 diagonal
GMM components per class, maximum 50 EM iterations, tolerance 0.001 and
regularization 0.000001. Retain at most 50,000 frames per class with the current
train-only random-priority sampler. All official train recordings contribute
candidate frames; the cap means the model does not fit every training frame.
Longer recordings contribute proportionally more candidates. State this limit
with any full-split result.

Run seeds **42, 43 and 44**, sequentially. First measure seed 42 and review
resources and convergence before starting the other two. Save failed or
nonconverged runs; do not silently replace a seed or increase the iteration
limit after seeing metrics. Report each seed and mean/standard deviation,
without treating three seeds as a statistical significance test.

For the first confirmation, score clean dev only. Reuse the hash-pinned official
2019 v1 evaluator and matching dev ASV scores. Report minimum normalized t-DCF
and EER, score direction, class/attack failures and threshold-boundary handling.
Any later full-dev bandwidth experiment must have its own protocol and report;
the pilot's balanced speaker-partition audit cannot be reused for different IDs.

## Required implementation

The current `run_baseline.py` deliberately accepts only balanced smoke/pilot
manifests. Changing counts in `configs/pilot.toml` cannot run this experiment.
Keep that pilot configuration and its historical outputs unchanged.

The separate implementation is now `scripts/prepare_full_data.py` and
`scripts/run_full_baseline.py`, configured by `configs/full_seed42.toml`.
The [Windows run guide](FULL_RUN.md) describes the detached supervisor, logs,
completion checks and interruption handling. The implementation steps below
remain the acceptance criteria.

1. Add a separate full train/dev preparation path. Verify the cached complete
   archive size and MD5; pin official key hashes and exact split IDs. Extract
   only train/dev members, with bounded per-member size, path checks, atomic
   writes and a recoverable hash journal. Match existing pilot files against
   the verified archive before adopting them. Avoid rewriting the pilot journal.
2. Add a full-split runner or an explicitly guarded full stage. Require exact
   official train/dev ID, label, attack and speaker metadata; reject missing,
   extra, duplicate or eval rows. Preserve natural class proportions. Verify
   selected audio and reference hashes before fitting.
3. Export a source-linked public resource/result audit. Keep audio, model and
   reference source outside Git. Retain configuration, seeds, package versions,
   source hashes, frame counts, convergence, complete finite scores and failures.
4. Review the first actual run independently, then run the remaining seeds only
   if the resource checks pass.

## Resource gates

Use the cached official archive; no additional dataset download is needed.
Inspect the ZIP central directory for train/dev byte totals before extraction,
without reading eval audio. Keep at least 20 GiB free disk after the estimated
additional extraction requirement. Recheck available RAM and disk at run start.

The [local central-directory inventory](validation/full_train_dev_inventory.json)
found all 50,224 keyed train/dev members, with no missing or duplicate keyed
paths. Their original FLAC bytes total **3,159,565,801** (about 2.94 GiB).
The observed free disk was 64,960,409,600 bytes (about 60.5 GiB), with about
3.64 GiB available system RAM. Even a separate full train/dev copy leaves about
57.6 GiB free. This inventory read member metadata only: it does not establish
decoded duration, waveform validity or peak decoded-file memory.

Use one numerical thread and no parallel training jobs. Initial engineering
budgets are **2 GiB process peak memory** and **30 minutes processing per seed**;
these are stop-and-review budgets, not measured full-set requirements. The new
external supervisor enforces limits even during native GMM fitting; standalone
runner checks are cooperative between recordings. Start only with at least
3 GiB available system RAM, and stop safely if memory pressure becomes material.

The pilot measured 39.223 seconds training, including extraction/sampling, and
8.766 seconds scoring 1,000 dev records. Record-count extrapolation is only a
planning estimate: frame reservoir maintenance, different durations, imbalance,
disk reads and system load can change full-set cost. Measure the first seed
instead of promising a duration from those pilot numbers.
The simple record-count projection is about 8.3 minutes for training and 3.6
minutes for clean dev scoring per seed. It is neither a measured result nor an
upper bound; the 30-minute review budget includes margin for that uncertainty.

The CPU GMM does not require Colab. Reassess a GPU session when a later neural
or localization model has a measured GPU requirement.

## Eval gate

Do not extract or score eval in this step. Before any eval run, save an immutable
final protocol identifying models/seeds, metrics, operating thresholds and any
custom robustness conditions. Use dev only for choices. Report clean official
split evaluation separately from custom transformed tests, and keep the Python
adaptation label until MATLAB feature parity is demonstrated.
