# Getting started

Date: 2026-10-09. The fixed 2,000 train / 1,000 dev pilot and first narrowband experiment are complete. Clean pilot EER is 8.8%; absolute score change did not beat the simple margin baseline for ranking errors. All selected audio and the complete archive checksum are verified. Results and limits are in [reports.md](../reports.md).

## Direction

Start with full-recording detection on ASVspoof 2019 LA. Build a CPU LFCC-GMM baseline, then run one narrowband robustness experiment. Localization is the next milestone. The first goal is a reliable data, scoring, and evaluation pipeline rather than a long model comparison.

## Steps and completion checks

| Step | Work | Done when |
|---|---|---|
| 1 — complete | Local environment and project structure | Python and an isolated environment work; README, dependency definition, `.gitignore`, `src/`, `configs/`, and `tests/` exist; imports and a short audio read/write check pass |
| 2 — pilot complete | Data access and split manifests | Published archive size/MD5 and all 3,000 selected audio checked; source, format, hashes and split separation recorded. Eval stays closed for development |
| 3 — complete | Small data and metric checks | 100 train/dev recordings profiled; LFCC configuration and limits saved; official 2019 external evaluator parity and input guards tested. No detector score or training measured |
| 4 — pilot complete | First working baseline | 32-component LFCC-GMM with seed 42 trained on 2,000 train and scored 1,000 dev recordings; configuration, convergence, scores, runtime/RAM and errors saved |
| 5 — pilot complete | One research question | Frozen narrowband/average comparison and speaker-separated reliability analysis completed; absolute score change lost to the margin baseline; clean and custom results reported separately |
| 6 | First complete milestone | If the pilot fits the budget, train on full official train; use dev for selection; open eval after the config is fixed; run 3 seeds, analyze errors by attack, and deliver a short report and local file-scoring demo |

Review the output of each step before starting the next. Pilot training is not an official benchmark result. If step 3 exceeds the resource budget, reduce the frame/feature cache or move suitable CPU work to Colab. Set performance expectations after the measurement pipeline works.

The next detection step is specified in [FULL_CONFIRMATION_PLAN.md](FULL_CONFIRMATION_PLAN.md).
It preserves official class proportions, excludes unkeyed archive members, and
measures seed 42 before the other seeds. The existing pilot runner needs a separate
guarded full-split path before this plan can run.

## Colab

Keep code, small checks, reports, and the demo local. Use Colab when GPU training becomes useful. A normal CPU LFCC-GMM implementation does not automatically use a GPU. Colab may help more with a small CNN/TCN or a later AASIST/SSL experiment; none is needed for the first baseline.

The initial research found that GPU type, resource access, and limits vary. Free sessions can run for up to 12 hours depending on availability and may stop when idle. VM files are temporary, and a paid plan is not a guarantee of a fixed GPU. Google also recommends copying archives to the VM disk and extracting there instead of reading many small files directly from Drive. Check the [official Colab FAQ](https://research.google.com/colaboratory/faq.html) again before a planned run.

When using Colab:

- Use the notebook only for setup and repository commands. Keep training code in `src/`.
- Measure GPU/VRAM, RAM, free disk, and package versions at the start. Do not assume a particular GPU.
- Use the same commit, config, seed, and manifest. Change paths through configuration.
- Process data on VM disk. Check persistent storage quota separately; data and weights stay out of Git.
- Save checkpoints, optimizer/scheduler state, RNG state, config, and logs to persistent storage. Verify resume behavior with a short run.

A free Colab pilot is a reasonable first option. Do not choose a subscription before measuring the need. This project detects and evaluates existing public recordings; generating speech is outside the scope.

## Next milestone

After A: PartialSpoof label/RTTM reader → 20 ms frame baseline → RangeEER → recording median/MAD context ablation. A and B share source material, so tests between them must not be presented as independent external generalization. Sources are in [reports.md](../reports.md); open work is in [backlog.md](../backlog.md).
