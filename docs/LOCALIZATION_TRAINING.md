# First localization training run

The queue is started explicitly with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_localization_pilot.ps1 -AudioJobDirectory outputs/jobs/partial-audio-20261010T163421519Z
```

The launcher prints its status file under `outputs/jobs/localization-pilot-*`.
It runs independently of the launch shell. The status distinguishes waiting,
feature extraction, actual fitting, scoring, completion and stopping. It does
not resume a failed fit automatically. A new run uses a new job directory.
Source edits invalidate the frozen run. The parent enforces the worker
timeout and RAM limits even while the numerical solver is running.

This is a small PartialSpoof v1.2 pilot: 128 official train recordings and
64 official dev recordings, selected before fitting with seed 45. It is a
new LFCC/logistic baseline, not a reproduction of the authors' neural model.
Eval remains closed. The ASVspoof detector report is a separate experiment.

## Execution order

1. Complete both archive checks and selected-audio receipts.
2. Check selected WAV hashes, sample rates, channels and actual sample counts
   against the unchanged native binary vectors. Preserve VAD as a separate
   diagnostic; nonspeech is not a binary class.
3. Freeze the alignment audit, source hashes, IDs and model configuration.
4. Fit normalization and the classifier on train frames only.
5. Save dev scores and the ten preselected prediction timelines.

The intended first fit uses 60 LFCC dimensions, nearest-frame mapping onto
20 ms output bins, minimum pooling of native 10 ms binary labels, and a
clipped final interval. The candidate length contract is ceil(N / 160) native
labels for N actual samples at 16 kHz. It must pass the real audio audit;
arrays are never silently shortened to match. VAD endpoints can differ from
WAV duration, so VAD coverage does not establish this contract.

The classifier uses StandardScaler and logistic regression (C=1, lbfgs,
max_iter=300), seed 45, and at most 50,000 retained frames per native class.
Sampling is restricted to train and balanced to the smaller retained class.
One numerical thread is used. Convergence warnings fail the run.

## Limits and interpretation

The initial worker budget is five minutes, with a 1 GiB process peak limit,
2 GiB available system RAM floor and 20 GiB disk reserve. The parent process
must enforce limits during fitting, including when library code is running.
Failed or interrupted runs retain their logs and are not reported as complete.
The queue stops on a failed prerequisite; it does not retry indefinitely.

Higher raw scores support bona fide audio. These scores are not calibrated
probabilities. A fixed zero threshold can produce initial timelines and
missed-spoof/false-alarm duration diagnostics. Any threshold chosen on dev
must be labelled a dev diagnostic, not independent evaluation.

Pooled 20 ms classification diagnostics are custom pilot results. They must
not be called official 10 ms SegmentEER or RangeEER. Native binary RTTM and
the documented evaluator adapter are separate metric prerequisites. Do not
substitute VAD spoof ranges for native binary ground truth.

The LFCC windows and delta features use adjacent and future samples. This
implementation is offline; 20 ms output bins do not imply 20 ms latency.
These small, balanced subsets do not establish full-benchmark performance
or generalization to new datasets. PartialSpoof and ASVspoof share source
material. See [pilot plan](LOCALIZATION_PILOT_PLAN.md) and
[run and recovery guide](LOCALIZATION_RUN.md).
