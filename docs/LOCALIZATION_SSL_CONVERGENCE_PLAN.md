# Separate train-only convergence follow-up â€” 2026-10-11

The original frozen SSL comparison used `max_iter=300` and stopped with a
`ConvergenceWarning`. No accepted model or development scores were produced.
That failure remains the original experiment's outcome; this follow-up does
not retroactively turn it into a completed comparison.

Before any development scoring, preregister one separate train-only fit with
`max_iter=3000`. This is the only optimization change. Keep C=1, lbfgs,
default tolerance 0.0001, seed 45, one numerical thread, the same frozen
float32 768-dimensional features, and StandardScaler fitted only on the
retained training frames. Keep all 128 train IDs and exactly 4,524 retained
frames per native class. Their ordered ID/bin digest must remain
`0fb814aa3cf8be436f199a2f971e1ccc31eb616a617b097767e93730aeabc82a`.

The extraction identity must remain
`93a5ee49123949554a27c4acced1f75f4742425bde9c0ca866103018241568fa`.
Verify all 192 feature caches and receipts before loading training labels or
using training features. Checking dev cache integrity is permitted; dev
features do not enter the fit, scaler, optimizer, sampling, or model choice.
The compact labels JSON also contains dev labels; parsing the shared file
must be recorded honestly, while only its train arrays enter computations.
No dev decision function, scores, metrics, or timelines are generated here.

Use an external supervisor observing the actual base Python worker through
`-S` and the project package bootstrap. Keep the existing 300-second total
worker deadline, 3 GiB worker peak, 2 GiB minimum available system RAM and
20 GiB disk floor. Preserve logs and stop on any warning, limit, source/input
drift, or failure. No automatic retry, hyperparameter tuning, alternate
classifier, or further iteration increase is allowed under this protocol.

Only a converged fit may write a model and train-only result. Independently
check convergence, source/input/cache identities and real worker resource
supervision before a separately executed evaluation of the original fixed
64 dev IDs and the same ten preselected timelines. Report the follow-up
separately from the failed 300-iteration experiment and the unchanged LFCC
baseline. The follow-up uses a different optimizer budget from the LFCC300
baseline; it is not an equal-training-budget comparison. Repeated dev inspection and unknown pretraining overlap remain
limitations; no official benchmark or unseen-corpus claim follows.

Sources: [new configuration](../configs/localization_ssl_convergence_v1.json),
[original frozen plan](LOCALIZATION_SSL_PLAN.md),
[original convergence failure](validation/localization_ssl_head_convergence_v2.json),
[verified extraction](validation/localization_ssl_extraction_v1.json),
[matching sampling preflight](validation/localization_ssl_sampling_preflight_v1.json).
