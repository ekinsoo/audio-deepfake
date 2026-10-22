# Single offline context ablation

This is the one exploratory ablation specified in the original pilot plan.
It compares a recording with its own context, using the same fixed 128 train
and 64 dev PartialSpoof v1.2 recordings as the baseline. There is no parameter
search, new data, eval access or claim of a new method.

First extract the unchanged 60-dimensional LFCC features and map them to the
same 20 ms bins as the baseline. For each recording and each feature dimension,
calculate the median over all aligned bins. Calculate MAD as the median
absolute deviation from that median. Transform each value by:

`(value - median) / max(1.4826 * MAD, 0.001)`

Apply this separately to every static, delta and double-delta feature. Include
the final partial bin with equal weight. Use float64 and do not clip. Constant
dimensions become zero; the fixed 0.001 floor prevents division by zero.
No labels enter this transform. Each dev recording supplies only its own
unlabelled audio statistics. Statistics are never pooled across dev recordings.
The full recording includes future audio, so this is explicitly offline.

Everything after this transform remains unchanged: train-only class-balanced
reservoir, at most 50,000 frames per class, seed 45, train-only StandardScaler,
logistic regression C=1, lbfgs, max_iter=300 and one numerical thread. Source
hashes and the exact rule are frozen in `validation/localization_context_frozen.json`
before fitting. No source or old baseline result may be overwritten.

Compare the same custom duration-weighted 20 ms closest-grid EER and fixed-zero
missed-spoof/false-alarm seconds. EER is a sweep on the same dev subset and is
not a held-out operating threshold. Ten timelines use the same preselected IDs.
Native RangeEER uses a separately audited evaluator and remains distinct.

Before execution require at least 2 GiB available RAM and 20 GiB free disk.
The parent independently enforces a 300 second timeout and 1 GiB worker peak
RAM. Stop on any convergence warning. A failed attempt is retained and is not
automatically restarted. This small CPU pilot requires no GPU.
