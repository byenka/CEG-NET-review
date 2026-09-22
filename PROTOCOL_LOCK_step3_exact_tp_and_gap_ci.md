# PROTOCOL LOCK - step 3: exact-TP sensitivity and paired coverage-gap intervals

Locked 2026-09-22, before the run. Nothing in this file is changed after the results exist.

## Purpose

1. **Exact-TP sensitivity.** The RETAIN target of the paper is composite (exact true positives, clinically close
   confusions and localization-weak detections; 72.7% of OIA-DDR test positives are localization-weak). Measure whether
   the operating-point result and the ranking comparison hold when the positive class is the exact true positive only.
2. **Paired interval for the coverage-gap advantage.** Under the composite target, report an image-clustered paired
   bootstrap interval for |g_gate| - |g_raw| on each cohort, where g is the delivered minus the intended 0.80 retention.
3. **Pointwise risk-coverage dominance.** Check whether "at every coverage the full gate carries less risk than the
   detector score" holds on a grid of coverages.

## Fixed design

- Features: `outputs/0506/evsds_v2sec/oiaddr/{valid,test}`, `outputs/0506/evsds_pss/{idrid,tjdr}/test`,
  `outputs/0506/evsds_pss/fgadr/train` (the evaluation populations of the paper, 538,986 FGADR and 23,181 TJDR detections).
- Methods: detector score (M0), knowledge-graph gate (B4), full gate (B6), built by
  `tools/eval_fp_reduction_benchmark.build_methods` with its unchanged recipe, seeds 0-4, fitted on OIA-DDR validation.
- Labels: part 1 uses `positive="tp_exact"` for fitting AND evaluation (category == TP_EXACT); parts 2 and 3 use the
  paper's `governance_retain`.
- Threshold: split-conformal quantile rule at alpha = 0.20 on OIA-DDR validation positives
  (`tools/eval_selective_conformal_baselines.conformal_threshold`), applied unchanged to every cohort.
- Gate values are means over the five seeds (coverage, AURC); AUPRC gains are also reported per seed.
- Bootstrap: 2,000 image-clustered resamples of the evaluation cohort, generator seed 0, thresholds and fitted gates held
  fixed; 95% percentile intervals. Paired: every method is evaluated on the same resample.
- Pointwise dominance: risk (share of SUPPRESS targets among retained detections) at coverages 0.01, 0.02, ..., 1.00;
  gate risk is the mean of the five seed curves. Dominance holds only if the gate risk is lower at every grid point.

## Reporting rule

Descriptive, no pass/fail criterion. Every quantity is reported as it comes out, whatever its sign. The exact-TP result is
a sensitivity analysis: it is shown whether or not the gate gains, and no ranking gain is claimed from it. If pointwise
dominance fails at any grid point, the text says "lower AURC" instead of "at every coverage".
