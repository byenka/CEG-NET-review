# PROTOCOL LOCK - step 7: sensitivity of the operating-point result to the miscoverage level alpha

Locked 2026-10-31, before the run. Nothing in this file is changed after the results exist. The result file records
the SHA-256 of this file, so any later edit is detectable.

## Purpose

Every operating-point result of the paper is measured at one miscoverage level, alpha = 0.20 (intended true-positive
retention 0.80). Section 5.1 reports that the full gate (CEG-Net) lands closer to the intended retention than the raw
detector score on all four datasets, and three earlier checks (image-disjoint calibration, step 2; exact-TP target,
step 3; paired intervals, step 3) all stay at alpha = 0.20. This step measures whether the ordering holds at two other
operating points, alpha = 0.10 and alpha = 0.30. No earlier run in the project has used any alpha other than 0.20.

## Fixed design

- Features and evaluation populations: identical to part 2 of step 3. OIA-DDR validation for fitting and calibration
  (`outputs/0506/evsds_v2sec/oiaddr/valid`), and the four evaluation datasets as listed in
  `tools/eval_selective_conformal_baselines.COHORTS` (OIA-DDR test, IDRiD test, TJDR test, FGADR train).
- Positive class: the paper's `governance_retain` target. The exact-TP target is not part of this step.
- Methods: detector score (M0), knowledge-graph gate (B4) and full gate (B6, CEG-Net), built by
  `tools/eval_fp_reduction_benchmark.build_methods` with its unchanged recipe, seeds 0-4, fitted on OIA-DDR validation.
  Scores are obtained through `scores_for` of `_LGOVNET_NEW_SEP/scripts/step3_exact_tp_and_gap_ci.py`, unchanged.
- Threshold: split-conformal quantile rule on the OIA-DDR validation positives
  (`tools/eval_selective_conformal_baselines.conformal_threshold`), applied unchanged to every dataset. This is the
  published protocol of Section 5.1 and Table S15, in which the gates are calibrated on the rows they were fitted on.
- Miscoverage levels: alpha in {0.10, 0.20, 0.30}, fixed here. No other level is added afterwards.
- Coverage gap: delivered minus intended retention, g = coverage - (1 - alpha). For the gates the coverage is the mean
  over the five seeds, then the nominal is subtracted, exactly as in step 3.
- Bootstrap: 2,000 image-clustered resamples of each evaluation dataset, generator seed 0, thresholds and fitted gates
  held fixed, 95% percentile intervals of |g_gate| - |g_M0|. The same resamples serve all three alpha levels, and the
  resampling sequence is the one of step 3.

## Control (checked before any alpha = 0.10 or 0.30 result is read)

At alpha = 0.20 the run must reproduce the published step 3 values of `governance_retain`: the point gaps of M0, B4
and B6 on all four datasets, and the 95% intervals of |g_gate| - |g_M0| for B4 and B6, each within 1e-6. If the point
control fails the script stops before the bootstrap and writes only the control diagnostics. If the interval control
fails the result is marked invalid, no finding is drawn from it, and the discrepancy is investigated and disclosed.

## Primary criterion (P1, CEG-Net)

P1 is met if |g_B6| < |g_M0| in all eight cells formed by alpha in {0.10, 0.30} and the four datasets (point
estimates, as in the published tables). The count k of the eight cells is reported whether or not P1 is met.

## Secondary quantities (descriptive, no pass/fail)

- The same eight-cell comparison for B4.
- The number of datasets whose paired interval of |g_gate| - |g_M0| lies wholly below zero, per alpha and gate.
- The signed gaps of every method at every alpha, including alpha = 0.20 for reference.

## Expectation stated before the run

TJDR is the thinnest cell at alpha = 0.20: B6 is ahead of M0 by 0.036 (-0.020 against -0.056) with an interval that
contains zero, and B4 by 0.006 (-0.050 against -0.056). In the image-disjoint run of step 2 the B4 margin on TJDR was
0.0034 and 0.0036 in the two split directions. A reversal, if any, is most likely there, and most likely for B4.

## Reporting rule

Every quantity is reported as it comes out, whatever its sign. The result goes to one supplementary table with all
three alpha levels and one sentence in Section 5.1. The abstract, the highlights, the contributions and every
statement made at alpha = 0.20 stay unchanged. If P1 is not met, the text states the count and names the dataset and
alpha of each reversal. No further alpha level, seed set, bootstrap setting, target or method is added to rescue or
strengthen the result.

## Out of scope

Three-pass score mean (F), fusion-input gate (F+E), margin, evidence-linear model, ensembles, the exact-TP target and
the image-disjoint calibration protocol. They are not run in this step.

## Outputs

- Script: `_LGOVNET_NEW_SEP/scripts/step7_alpha_sensitivity.py`
- Result: `_LGOVNET_NEW_SEP/results/step7_alpha_sensitivity.json` (records the SHA-256 of this file)
