# Frozen speech features: first comparison

Question: can a pretrained speech representation reduce the fragmented
false alarms of the LFCC pilot when the labelled data and classifier stay
fixed? Improvement is a hypothesis. This is an exploratory development
comparison, not a new method or an official benchmark.

## One candidate

Use `facebook/wav2vec2-base`, revision
`0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8`. The official model card declares
Apache-2.0 and 16 kHz input. The checkpoint is audio-pretrained, without an
ASR fine-tuning head. The API reports one PyTorch weight file of 380,267,417
bytes; its SHA-256 is recorded in the configuration. Preserve attribution
and license notices in any redistribution. Keep all weights outside Git.

Select the final transformer output, 768 dimensions, before seeing results.
No layer search, layer averaging, LFCC concatenation, smoothing, context
normalization or encoder fine-tuning. Load the base encoder only; run
`eval()` and inference mode in float32 with batch size one and two CPU threads.
Disable augmentation and dropout through inference mode. Use the pinned
processor's per-recording zero-mean/unit-variance waveform normalization.
No batch padding or attention mask is needed for one complete recording.

WavLM Base+ was considered. Its model card links a separate UniSpeech license;
that must not be inferred from a different source-code license. Wav2vec 2.0
has a clearer checkpoint license declaration for this first comparison.
This selection does not claim it will outperform WavLM.

## Preserve the comparison

Use the same 128 train and 64 development recordings, seed 45, native binary
labels and actual-audio alignment audit. Do not read eval. Pretraining data
provenance does not establish recording-level disjointness with PartialSpoof;
no unseen-corpus generalization claim is supported. ASVspoof/PartialSpoof
source overlap and repeated inspection of this dev subset remain limitations.

The checkpoint convolutions have a 400-sample receptive field and 320-sample
stride. For an unpadded N-sample recording, frame count is
`1 + floor((N - 400) / 320)`, and frame j has center `200 + 320*j`.
Use nearest centers to map to the existing 320-sample bins, with earlier-frame
ties and the actual midpoint of the clipped final bin. Repeat edge states
where needed; do not pad audio or alter labels. Reject N < 400. Check real
encoder frame counts before fitting. Transformer output uses future audio
from the whole recording: these are offline features, not 25 ms context.

Cache aligned float32 features with audio, encoder, processor, environment
and source hashes. The existing selection contains 32,710 bins, about 96 MiB
of raw 768-dimensional features. Measure actual disk and RSS; this is not an
upper bound on model or framework memory. No new dataset download is needed.

The old classifier module requires exactly 60 features and is hash-frozen.
Implement a separate dimension-aware head; do not modify the saved baseline.
Match its sorted-ID, class-specific random-priority sampler and publish the
selected `(recording ID, bin index)` digest to demonstrate equal sampling.
Retain 4,524 frames per class on this selection. Fit StandardScaler on those
train frames only, then logistic regression C=1, lbfgs, max_iter=300, seed 45,
one numerical thread. Stop on convergence warnings; do not increase iterations
after looking at dev results. Classifier feature dimension necessarily changes.

Compare custom duration-weighted 20 ms EER, fixed-zero missed-spoof and false
alarm seconds, predicted-run fragmentation, and the same ten timelines.
Run the existing derived-reference range diagnostic separately, with its
publisher-reference caveat. Report resource cost alongside accuracy. Do not
choose a deployable threshold from the inspected dev EER operating point.

## Resource and execution gates

The current machine has 15.71 GiB total RAM. A fresh Windows probe found
about 4.0 GiB available and 57.5 GiB free disk. GPU inventory could not be
refreshed through CIM; earlier evidence identifies Intel Iris Xe and no
NVIDIA device. Plan CPU execution, not CUDA. Probe again at execution time.

Before installing a separate optional encoder environment, fix package
versions and record its lock. Leave the baseline and metric environments
unchanged. The published checkpoint is a PyTorch binary: require an updated
PyTorch with restricted `weights_only` loading, no remote code execution,
exact revision, size and hash validation, and no arbitrary pickle loader.
Compatibility must be checked before downloading the weights.

Allow at most 500 MB checkpoint transfer, 2 GiB optional dependency transfer
and 1 GiB feature-cache storage, with at least 20 GiB free disk throughout.
Stop rather than switch checkpoint, layer or feature rules automatically.

Require at least 4 GiB available system RAM before model loading. An external
supervisor must enforce a 3 GiB worker peak, 2 GiB minimum available system
RAM and a 180-second timeout for the first recording. The smoke recording is
the longest selected train recording (ID recorded in preflight). No labels,
dev scores or model fit enter that measurement. Check shape, finite values,
actual memory and extraction speed. On failure retain logs and stop.

If the measured smoke passes, allow at most 30 minutes for all 192 feature
extractions and 300 seconds for head fitting, retaining per-recording caches
for resume. A successful smoke is required, not assumed from these budgets.
If local extraction is unsuitable, use a GPU Colab session for the exact
same frozen encoder extraction only. Record its hardware and dependency
lock first; preserve feature rules, weights and IDs. Never silently switch
between CPU and GPU caches without recording numerical differences.

## Sources and current state

- [Official model card](https://huggingface.co/facebook/wav2vec2-base)
- [Pinned configuration](https://huggingface.co/facebook/wav2vec2-base/blob/0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8/config.json)
- [Pinned processor](https://huggingface.co/facebook/wav2vec2-base/blob/0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8/preprocessor_config.json)
- [Original paper](https://arxiv.org/abs/2006.11477)
- [WavLM card and its license link](https://huggingface.co/microsoft/wavlm-base-plus)

The machine-readable configuration and offline geometry preflight are ready.
Only small official metadata files were fetched. No encoder weights,
dependency packages, new audio or training were started at this planning
milestone. Environment locking, secure loading, supervised smoke and the
dimension-aware classifier remain implementation work before a result exists.
