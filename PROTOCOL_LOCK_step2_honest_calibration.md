# Protocol lock — Step 2: honest calibration re-run of the operating-point result

Written 2026-09-18, **before the experiment was run**. The criteria below must not be changed
once the numbers are visible. The outcome is reported whatever it is.

## Background

The audit of 2026-09-18 established that in the published protocol the gate is fitted on the
OIA-DDR validation split and its operating-point threshold is calibrated on the **same rows**
(`tools/eval_fp_reduction_benchmark.py` fits on `valid`; `tools/eval_selective_conformal_baselines.py`
then calls `conformal_threshold(sv, valid["label"], ALPHA)` with `sv` the in-sample score). The
calibration scores are therefore in-sample for every method with fitted parameters, and honest
for every method without them. The methods with fitted parameters are the ones the claim favours.
This step removes the overlap and re-measures.

## Question

Does the operating-point claim of Section 5.1 survive when the threshold is calibrated on data
the gate never saw, with every method calibrated on that same held-out data?

## Population and splits

OIA-DDR validation has 149 images and 39,710 candidate detections. It is split **by image**,
never by detection, into two folds by a seed-free deterministic rule
(`sha256(image_id)` parity). Fold A holds 74 images, fold B 75.

Three configurations are run, and the contrast between the first two isolates the effect of
calibration honesty at a fixed fitting budget:

| Config | Gate fitted on | Threshold calibrated on | Purpose |
|---|---|---|---|
| H1 | fold A | **fold B** | honest, half fitting budget |
| H2 | fold A | fold A | in-sample control, same half budget |
| H1' | fold B | **fold A** | honest, folds swapped |

The published protocol (fit on all 149 images, calibrate on the same rows) is the fourth point
of comparison and is quoted from the manuscript, not recomputed.

The primary reported number is the mean of H1 and H1'. Both directions are also reported
separately, because a single split of 74 images carries split variance that a mean over two
directions does not remove.

## Methods, all calibrated on the same held-out fold

| Code | Score | Fitted parameters |
|---|---|---|
| S | primary detector score `pred_score` | none |
| M | top-1 minus top-2 margin `unc_margin` | none |
| EL | logistic regression on the four evidence signals | yes |
| KG | MLP on the four evidence signals and the score | yes |
| FG | full gate: KG inputs plus the two multi-source signals | yes |

Learned methods are averaged over the five fixed seeds (0-4). Methods without parameters are
unaffected by the fitting budget, which biases this experiment **against** the gates; that
asymmetry is stated with the result rather than corrected.

## Threshold rule

The same rule as the manuscript: calibrate on the positives of the calibration fold at
miscoverage `alpha = 0.20` and take the `ceil((n+1)(1-alpha))/n` quantile of the positive
nonconformity scores. The threshold is then frozen and applied unchanged to OIA-DDR test,
IDRiD, TJDR and FGADR. No target-domain data enters the threshold, the fit, or any decision.

This is a split by image with a finite-sample quantile correction. It is **not** an exact
conformal guarantee and is not described as one: detections within an image are dependent, so
the exchangeable unit is closer to the image than to the detection, and the correction assumes
exchangeable detections.

## Reported quantities

Per method and per cohort: empirical coverage, signed coverage gap, absolute coverage gap,
AURC, retained fraction, and an image-clustered bootstrap confidence interval on the coverage
gap with 2,000 resamples. AURC intervals are reported where the cohort is small enough for the
existing convention (n <= 120,000) and omitted on FGADR, as in the published tables.

## Criteria, locked in advance

The Section 5.1 claim is: among the single-detector methods measured, the two gates are the
only ones that improve on raw confidence in **both** the coverage gap and the risk-coverage
area, on **all four** cohorts. It decomposes into three testable parts.

- **D1.** Both gates have a smaller absolute coverage gap than S on all four cohorts.
- **D2.** Both gates have a lower AURC than S on all four cohorts.
- **D3.** Neither M nor EL satisfies D1 and D2 simultaneously on all four cohorts.

Verdict:

- **A, CLAIM SURVIVES.** D1, D2 and D3 all hold.
- **B, CLAIM PARTIALLY SURVIVES.** D2 holds on all four cohorts for at least one gate, and D1
  holds on at least two of four cohorts for that same gate, but the full conjunction fails.
- **C, CLAIM FAILS.** D1 holds on at most one cohort for both gates, or some method without
  fitted parameters satisfies D1 and D2 on all four cohorts while a gate does not.

## Secondary, not part of the verdict

The difference H2 minus H1 on the in-domain coverage gap measures the magnitude of the
in-sample calibration effect at a fixed fitting budget. It is reported for its own sake and
does not enter the criteria.

## Isolation

The script reads the existing CSVs in place and writes only under
`_LGOVNET_NEW_SEP/results/`. No detector is run. The manuscript is not touched by this step.
