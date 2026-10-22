# Report evidence

This note checks the numbers used in the project report. The independent [machine-readable review](validation/report_evidence_review.json) was produced from the saved score tables. It verified the frozen source and artifact hashes, exact official dev joins, clean-score equality, score arithmetic, speaker partition, calibration thresholds, nine detection metric calculations, error rankings and selected failure examples. It did not train models or rescore audio.

## Detection

Each full model was trained using all 25,380 official train recordings. Each condition contains all 24,844 official dev recordings: 2,548 bona fide and 22,296 spoof. EER is a percentage below; minimum normalized t-DCF is unitless. Lower is better for both.

| Seed | Clean EER (%) | Narrowband EER (%) | Average EER (%) | Clean min t-DCF | Narrowband min t-DCF | Average min t-DCF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 5.8902 | 18.5637 | 8.5853 | 0.15268 | 0.47002 | 0.19226 |
| 43 | 6.6729 | 24.9602 | 9.3018 | 0.17938 | 0.56799 | 0.21117 |
| 44 | 6.4694 | 22.9946 | 9.1891 | 0.18504 | 0.55146 | 0.20027 |
| Mean | 6.3441 | 22.1728 | 9.0254 | 0.17237 | 0.52982 | 0.20123 |
| Population SD | 0.3316 | 2.6752 | 0.3146 | 0.01411 | 0.04282 | 0.00775 |

The clean results use the official dev population and the pinned ASVspoof 2019 Python evaluator. This is a small sklearn LFCC-GMM adaptation, not an exact reproduction of the official MATLAB baseline. Narrowband and fixed averaging are custom stress experiments. Their t-DCF uses the unchanged official ASV scores, so it does not evaluate an entire telephone-channel speaker verification system. No held-out eval result is available.

The transform is a 16 kHz to 8 kHz to 16 kHz resampling round trip, with the original sample count restored. It approximates bandwidth loss. It does not simulate a codec. The average is fixed at `clean / 2 + narrowband / 2`.

## Error ranking

The ranking question is whether a diagnostic score puts clean-model mistakes first. The calibration subset has 9,609 recordings from eight dev speakers, including 945 bona fide recordings. The analysis subset has 15,235 recordings from twelve other dev speakers: 1,603 bona fide and 13,632 spoof. All three models use this same partition.

The decision threshold is the 48th ascending clean calibration bona fide score (zero-based index 47). A score at or above the threshold is accepted as bona fide. This accepts 898/945 calibration bona fide recordings, or 95.026%. The ranking target is a clean prediction error at this threshold; it is not an error at the EER threshold.

| Seed | Threshold | Errors / 15,235 | Error prevalence (%) | Delta AUROC | Delta AP | Margin AUROC | Margin AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.716702 | 774 | 5.0804 | 0.50363 | 0.04643 | 0.93786 | 0.36954 |
| 43 | 0.726099 | 844 | 5.5399 | 0.33822 | 0.03820 | 0.93694 | 0.39114 |
| 44 | 0.720040 | 853 | 5.5989 | 0.31338 | 0.03736 | 0.93389 | 0.38943 |
| Mean | — | — | 5.4064 | 0.38508 | 0.04066 | 0.93623 | 0.38337 |
| Population SD | — | — | 0.2318 | 0.08444 | 0.00409 | 0.00169 | 0.00980 |

Delta means `abs(narrowband - clean)`. Margin means `-abs(clean - threshold)`, so larger values mean closer to the decision boundary. Higher AUROC and non-interpolated average precision (AP) are better. Error prevalence gives an AP reference level. Delta AP falls below prevalence for each seed. Seed 42 delta AUROC is close to chance; the other two seeds are below chance. The protocol did not allow reversing the score after seeing this result.

Margin ranks errors much better in this experiment. It is a diagnostic score, not a calibrated probability or a validated rejection policy. These results do not establish how it behaves on new attacks, speakers from another dataset, or deployment audio.

## Failure cases

At the calibration threshold, analysis errors are concentrated in bona fide recordings and attacks A04/A06:

| Seed | Bona fide errors / 1,603 | A04 errors / 2,272 | A06 errors / 2,272 | A01/A02/A03/A05 errors |
| --- | ---: | ---: | ---: | ---: |
| 42 | 144 | 181 | 449 | 0 |
| 43 | 174 | 174 | 496 | 0 |
| 44 | 147 | 237 | 469 | 0 |

Zero errors for these four attack groups applies only to this dev subset and fixed threshold. It is not evidence of perfect generalization.

Ten examples per seed were selected by the frozen rule: take the first ID in each delta-sign and top-quarter-disagreement stratum, deduplicate, then fill by descending absolute delta and ID. Top quarters are computed over all analysis recordings. The selections are illustrative and are not a random sample. IDs and exact scores are preserved in the [completed experiment audit](validation/full_robustness_results_v1.json).

## Pilot comparison

The pilot used 2,000 train and 1,000 dev recordings with balanced labels. Its seed-42 clean EER was 8.8%; fixed averaging gave 8.4%. The full experiment used the natural dev label distribution and three models. Averaging worsened EER for every full model: the mean rose from 6.34% to 9.03%.

This comparison does not isolate why the pilot and full results differ. Training recordings, sampled frame contents, evaluated population and seed coverage changed. The pilot dev subset is included in full dev. The training frame cap did **not** change: both stages retain at most 50,000 frames per class, or 100,000 total. The small pilot gain did not carry over to the larger dev experiment; no causal explanation is established.

## Resource evidence

| Seed | Successful chunk processing (s) | Successful chunk supervisor wall time (s) | Largest successful chunk process peak RSS (MiB) |
| --- | ---: | ---: | ---: |
| 42 | 391.10 | 420.51 | 210.93 |
| 43 | 582.23 | 619.83 | 211.21 |
| 44 | 292.35 | 319.98 | 210.75 |

These are CPU measurements with one numerical thread. Processing time is accumulated successful chunk work. These totals exclude interrupted work, startup, preflight, recovery, resource waiting and finalization; they are not end-to-end elapsed times. Per-process peak RSS does not include all machine memory or filesystem cache. The run was interrupted by its available-RAM guard and later resumed from verified completed chunks. Timing differences should not be interpreted as a seed-specific speed advantage.

## Claims to keep narrow

- This is a reproducible full-recording detection baseline and a negative bandwidth-stress experiment. Partial manipulation localization has not been implemented or evaluated.
- The protocol was frozen before full transformed scoring, after inspecting clean full-dev results and pilot stress results. It is not blind preregistration or independent validation.
- Calibration and analysis speakers are disjoint inside dev. They are not new independent datasets, and both use the same known attack taxonomy as train.
- Means and population standard deviations summarize the three fixed model seeds. They are not confidence intervals or independent dataset replications.
- The current evidence supports keeping clean scores as the detection baseline and investigating margin-based error triage separately. It does not support a general robustness claim for the proposed averaging or delta method.

The immutable [protocol](FULL_ROBUSTNESS_PLAN.md), [completed result audit](validation/full_robustness_results_v1.json), and [independent evidence review](validation/report_evidence_review.json) contain the experiment details and hashes.
