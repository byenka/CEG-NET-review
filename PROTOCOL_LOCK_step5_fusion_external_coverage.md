# PROTOCOL LOCK - step 5: operating point of score averaging on the external cohorts

Locked 2026-09-25 23:29 (CST), before the run. No result of this step exists when this file is written, and nothing in
this file is changed after the results exist. Script: `_LGOVNET_NEW_SEP/scripts/step5_fusion_external_coverage.py`.
Output: `_LGOVNET_NEW_SEP/results/step5_fusion_external_coverage.json`.

## Purpose

Section 5.1 reports the split-conformal coverage gap and the AURC of the full gate against raw confidence on four
cohorts, but the three-pass score mean (F), the stronger ranker of Section 5.5, has a coverage gap only in domain
(+0.046, `outputs/0506/selective_conformal/fusion_headtohead.json`). `asal_data.md` records its external AURC and gap as
"tidak tersedia". This step measures, on all four cohorts, where a source-calibrated threshold lands for F and for the
gate that reads F (F+E), next to the detector score (M0) and the full gate (B6).

## Fixed design

- Populations: the evaluation populations of the paper, identical to `baselines.json`.
  OIA-DDR validation `outputs/0506/evsds_v2sec/oiaddr/valid`, OIA-DDR test `outputs/0506/evsds_v2sec/oiaddr/test`,
  IDRiD test and TJDR test `outputs/0506/evsds_pss/{idrid,tjdr}/test`, FGADR train `outputs/0506/evsds_pss/fgadr/train`
  (64,214 / 8,042 / 23,181 / 538,986 detections, including the 114 lesion-free images).
- Detector caches: the merged caches that produced the published T15, `E:/tmp/zerobox/merged/{primary,flip,v2root}`.
- Target: `governance_retain` (RETAIN versus SUPPRESS), fitting and evaluation.
- F: mean of the primary score, the best-IoU flip-pass score and the best-IoU secondary-detector score, each non-primary
  pass matched to the primary box at IoU >= 0.15 and contributing 0 when unmatched (the recipe of
  `_septem/scripts_optionB/fusion_external.py`). FGADR has no secondary cache, so F there is the two-pass mean
  (primary + flip), and its threshold is calibrated on the two-pass mean of OIA-DDR validation.
- F+E: the paper's gate recipe (`tools.eval_fp_reduction_benchmark.fit_mlp`, 16 tanh units, 1,500 iterations, lr 0.1)
  fitted on OIA-DDR validation with inputs `[ev_sas, ev_sdca, ev_aps, ev_pss, ev_sds, ev_sds_nsrc, F]`, seeds 0-4.
  FGADR uses the same inputs with the two-pass F on both validation and FGADR (the protocol of Supplementary S4).
- M0 and B6: `tools.eval_fp_reduction_benchmark.build_methods`, unchanged, seeds 0-4 for B6.
- Threshold: split-conformal quantile rule at alpha = 0.20 on OIA-DDR validation positives
  (`tools.eval_selective_conformal_baselines.conformal_threshold`), one threshold per score and per seed, applied
  unchanged to every cohort. Coverage gap = delivered minus 0.80 retention of RETAIN targets.
- Learned methods: coverage, retained fraction and AURC are means over the five seeds.
- Bootstrap: 2,000 image-clustered resamples of each evaluation cohort, generator seed 0, thresholds and fitted models
  held fixed, 95% percentile intervals. Paired quantities: |g_B6| - |g_F| and |g_F+E| - |g_F|, where each learned gap is
  the mean over seeds within the resample. AURC is reported as a point estimate only.

## Reproduction controls (K0), all must pass before anything is interpreted

Tolerance 1e-4 unless stated.
1. M0 and B6 coverage gap and AURC equal `outputs/0506/selective_conformal/baselines.json` on all four cohorts.
2. F AUPRC equals the published fusion baseline in `E:/tmp/zerobox/results/fusion_merged.json`:
   0.722372, 0.792876, 0.557342, 0.534429 (OIA-DDR, IDRiD, TJDR, FGADR two-pass).
3. F on OIA-DDR test reproduces `fusion_headtohead.json`: coverage gap 0.04565 and AURC 0.38166.
4. F+E five-seed mean AUPRC equals `fusion_merged.json`: 0.733227, 0.775001, 0.540133; FGADR two-pass per-seed AP equals
   `E:/tmp/zerobox/results/fgadr_two_pass_merged.json`: 0.52298, 0.507146, 0.49943, 0.525303, 0.517201 (tolerance 1e-5).

If any control fails, the run stops, the failure is recorded, no quantity is interpreted and the manuscript is not
changed.

## Criteria

- **C1 (primary, comparative):** the full gate has a smaller absolute coverage gap than F on at least 3 of the 4 cohorts
  (FGADR with the two-pass F).
- **C2 (paired):** the number of cohorts whose 95% interval of |g_B6| - |g_F| lies entirely below zero, and the number
  entirely above zero.
- **C3 (descriptive):** the number of cohorts on which F improves on M0 in both the absolute coverage gap and AURC.
- **C4 (descriptive):** the number of cohorts on which F+E has a smaller absolute coverage gap than F.

## Reporting rule and consequence for the manuscript

Every quantity is reported as it comes out, whatever its sign. The consequence for the text is fixed now:
- If C1 is met, Section 5.1 may state that the full gate lands closer to nominal than the three-pass mean on the number
  of cohorts observed, together with the number of paired intervals below zero, and no stronger wording.
- If C1 is not met, Section 5.1 reports the external gaps of the three-pass mean as measured, the claim of Section 5.1
  stays restricted to methods that do not average detector scores, and no operating-point advantage over score
  averaging is claimed.
- In both cases the FGADR entry is labeled as two-pass, and the ranking comparison of Section 5.5 is unchanged.
