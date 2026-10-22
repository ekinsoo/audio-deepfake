# Development

## Working on the project

Keep model code in `src/audio_deepfake/`, runnable experiments in `scripts/`,
and their settings in `configs/`. Add or adjust tests with a behavior change.

```bash
git switch -c experiment/short-description
# Make changes, then:
python scripts/run_tests.py
git add path/to/changed-files
git commit -m "Describe the change"
git push -u origin experiment/short-description
```

Open a pull request to review an experiment before merging to `main`. Small
README or documentation edits can also use a normal commit on `main`.

For a new experiment, write down the question, dataset split, settings and
compute budget before running. Use a new output directory. Keep earlier runs
and their failures; append a new result rather than replacing an old metric.
Record the seed, software versions, source/input hashes and limitations.

The existing experiment sources are hash-pinned. Editing them is allowed,
but old artifacts will then fail their identity checks. Preserve the original
version in Git and record a new configuration/audit for a changed experiment;
do not rewrite old hashes to make unrelated results pass.

## Testing

```bash
python scripts/run_tests.py
```

This runs the portable unit suite, excluding two integration modules. It
requires no audio download, pretrained model or external metric checkout.
Some official-evaluator parity tests skip when their external files are absent.
The same command runs on pushes and pull requests in GitHub Actions.

Two additional suites need prepared environments:

- `python scripts/run_tests.py --suite range`: use the isolated metric
  environment from [RANGE_EER_CONFORMANCE.md](RANGE_EER_CONFORMANCE.md). It needs
  the pinned upstream files and its own dependencies.
- `python scripts/run_tests.py --suite process`: Windows only, using the audited
  `.venv` layout and numerical versions. It checks the real worker PID and RAM
  monitor with a bounded allocation; it does not fit a model.

The saved experiment validation covered 350 tests across those environments.
A clean checkout has fewer runnable integration checks; this is separate from
reproducing the reported model metrics.

## Data and artifacts

Do not commit audio, dataset archives, checkpoints, model files or feature
caches. They belong under ignored `data/`, `outputs/` or `.cache/`. Do not
commit credentials or company recordings. Check `git diff --cached --stat`
and `git status` before a push.

Compact public result audits and derived figures belong in `docs/`. Historical
logs contain original local paths and runtime details; those identify the
recorded run and are not portable installation settings.

See [DATA_PROTOCOL.md](DATA_PROTOCOL.md) and
[PARTIALSPOOF_PROTOCOL.md](PARTIALSPOOF_PROTOCOL.md) before fetching datasets.
Their licenses and the external evaluator/checkpoint licenses are separate.
