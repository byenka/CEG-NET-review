# PROTOCOL LOCK - step 6: FGADR with the three-pass score mean

Locked 2026-09-26 03:30 (CST), before the run. No result of this step exists when this file is written, and nothing in
this file is changed after the results exist. Script: `_LGOVNET_NEW_SEP/scripts/step6_fgadr_three_pass.py`.
Output: `_LGOVNET_NEW_SEP/results/step6_fgadr_three_pass.json`.

## Purpose

The FGADR comparisons against score fusion were computed with a two-pass mean (primary + flip) because the fusion
scripts looked for the secondary-detector cache in a folder that does not hold FGADR. The FGADR secondary cache is
complete (`E:/tmp/zerobox/merged/v2/fgadr/train`, 1,842 images, the detector that produced the published FGADR evidence).
This step recomputes every FGADR comparison with the three-pass mean used for the other cohorts, so that no FGADR entry
of the paper depends on a two-pass mean.

## Fixed design

- Population: FGADR train, `outputs/0506/evsds_pss/fgadr/train/governance_features_train.csv` (538,986 detections,
  1,842 images). Source split: OIA-DDR validation, `outputs/0506/evsds_v2sec/oiaddr/valid`.
- Caches: primary and flip passes `E:/tmp/zerobox/merged/{primary,flip}`; secondary pass for OIA-DDR validation
  `E:/tmp/zerobox/merged/v2root/oiaddr/valid`, for FGADR `E:/tmp/zerobox/merged/v2/fgadr/train`.
- F (three-pass mean): mean of the primary score and of the best-IoU flip and secondary scores, each matched to the
  primary box at IoU >= 0.15 and contributing 0 when unmatched (the recipe of `fusion_external.py`).
- F+E: the paper's gate recipe (`fit_mlp`, 16 tanh units, 1,500 iterations, lr 0.1) fitted on OIA-DDR validation with
  inputs `[ev_sas, ev_sdca, ev_aps, ev_pss, ev_sds, ev_sds_nsrc, F]`, seeds 0-4. These are the same fitted gates as the
  fusion-input gate of the other cohorts.
- M0 and B6: `tools.eval_fp_reduction_benchmark.build_methods`, unchanged, seeds 0-4.
- Quantities, each for FGADR:
  A. AUPRC (average precision) of F.
  B. AUPRC of F+E per seed, mean and sample SD, the mean difference F+E minus F, and per seed a paired image-clustered
     bootstrap interval of AP(F+E) - AP(F) (2,000 resamples, generator seed 0, one image draw of size n_images per
     resample, the scheme of `paired_bootstrap_ap`), counted above and below zero.
  C. AUROC of F for grade-shifting (positive) against within-grade confusions, with the image-clustered bootstrap of
     `clinical_cost_external.py` (2,000 resamples, seed 0).
  D. Split-conformal coverage gap (alpha 0.20, thresholds on OIA-DDR validation), AURC and retained fraction of F and
     F+E, and the paired intervals of |g_B6| - |g_F| and |g_F+E| - |g_F| with the bootstrap of step 5.

## Reproduction controls (K0), all must pass before anything is interpreted

The same code run with the two-pass mean must reproduce the published two-pass numbers, and the unchanged methods must
reproduce theirs. Tolerance 1e-4 unless stated.
1. Two-pass F AUPRC on FGADR = 0.534429 (`fusion_merged.json`).
2. Two-pass F+E per-seed AP = 0.52298, 0.507146, 0.49943, 0.525303, 0.517201, and its per-seed paired intervals equal
   those of `fgadr_two_pass_merged.json` (tolerance 1e-5).
3. Two-pass F grade-shifting AUROC = 0.459355, interval [0.423161, 0.493416] (`clinical_cost_external.json`).
4. Two-pass F coverage gap and the interval of |g_B6| - |g_F| equal step 5 on FGADR (+0.1391, [-0.0503, -0.0436]).
5. M0 and B6 on FGADR: gap and AURC equal `baselines.json`; B6 AUPRC equals 0.482254.
6. The three-pass F+E gates reproduce the published OIA-DDR fusion-input AUPRC 0.733227 when applied to OIA-DDR test.
7. The FGADR secondary cache holds a file for every FGADR image of the evidence CSV.

If any control fails, the run stops, the failure is recorded, nothing is interpreted and the manuscript is not changed.

## Criteria and consequence for the manuscript

Descriptive: every quantity is reported as it comes out, whatever its sign. The criteria of step 5 are recounted over the
four cohorts with FGADR now three-pass and OIA-DDR, IDRiD and TJDR taken unchanged from step 5:
C1 full gate closer to nominal than F on at least 3 of 4 cohorts; C2 paired intervals of |g_B6| - |g_F| below and above
zero; C3 F improving on M0 in both coverage gap and AURC; C4 F+E closer to nominal than F.

Consequence fixed now: every FGADR entry that used the two-pass mean (Section 5.5, T12, T15, Supplementary S1 and S4,
Section 5.1, contribution 1, Conclusion) is replaced by the three-pass value; every statement that FGADR lacks a
secondary cache or uses two passes is removed; the two-pass FGADR follow-up table of Supplementary S4 is removed and S4
reports the four cohorts together; the operating-point wording follows the step-5 rule with the recounted C1 and C2
(number of cohorts and number of intervals below zero, no stronger wording), and the abstract keeps "on all four" only
if the recounted C1 is 4 of 4.
