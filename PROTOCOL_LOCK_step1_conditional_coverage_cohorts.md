# Protocol lock — Step 1: conditional coverage across all four cohorts

**before the experiment was run**. The criteria below must not be changed
once the numbers are visible. The outcome is reported whatever it is.

## Background

Step 0 (`PROTOCOL_LOCK_step0_conditional_coverage.md`) showed on OIA-DDR test that a
marginally calibrated split-conformal threshold can be near-perfect overall and still collapse
inside a subgroup: the five-detector ensemble reached a marginal gap of +0.000 while covering
almost none of the 1,874 true lesions in the stratum no other detector corroborates
(gap −0.800). That experiment used the five `detcmp` detectors, whose caches exist only for
OIA-DDR, so it says nothing about the external cohorts.

The deployed evidence vector carries two annotation-free stratification variables that **do**
exist on all four cohorts, and a third that records evidence availability. This step asks the
Step 0 question of all four cohorts, using those.

## Question

Does a split-conformal threshold calibrated **marginally** on OIA-DDR validation deliver
near-nominal coverage **inside each stratum** on OIA-DDR test, IDRiD, TJDR and FGADR?

## Population

Every primary-detector candidate in each cohort, target `governance_retain`, identical to
`tools/eval_selective_conformal_baselines.py`. Same CSVs, same task, same alpha.

## Stratifications, all annotation-free

| Code | Variable | Strata |
|---|---|---|
| A | `ev_sds_nsrc` | 0 (primary only) and 2 (secondary detector corroborates) |
| B | `ev_sdca` against delta = 0.30 | low conflict, high conflict — the paper's own tier axis |
| C | `ev_pss_status` | perturbation stability measured, or placeholder |

## Methods

| Code | Score |
|---|---|
| S | primary detector score `pred_score` |
| M | top-1 minus top-2 margin `unc_margin` |
| KG | knowledge-graph gate, four evidence signals plus score, five seeds |
| S+E | full gate, adds the two multi-source signals, five seeds |

## Procedure

For each method: compute the **marginal** split-conformal threshold on OIA-DDR validation at
`alpha = 0.20` with the `ceil((n+1)(1-alpha))/n` quantile of the positive nonconformity scores,
exactly as `conformal_threshold` in `tools/eval_selective_conformal_baselines.py`. Apply it
unchanged to each cohort and measure the empirical coverage inside each stratum. Learned methods
are averaged over the five fixed seeds. Per-stratum gap = empirical coverage minus 0.80.

A stratum is **eligible** when it holds at least 200 positive detections. Smaller strata are
reported but never trigger a criterion.

## Criteria, locked in advance

- **C1, diagnostic.** MET if at least one method has `|gap| > 0.15` in at least one eligible
  stratum of at least one cohort. This establishes that the marginal guarantee is insufficient
  beyond the in-domain cohort as well.
- **C2, comparative.** MET if, on stratification A, the full gate's worst eligible-stratum
  `|gap|` is smaller than the detector score's worst eligible-stratum `|gap|` in at least
  three of the four cohorts.
- **C3, margin check.** The main article reports that the margin has a smaller *marginal*
  coverage gap than the gate on TJDR and FGADR. C3 is MET if, on those two cohorts, the margin
  also has the smaller worst eligible-stratum `|gap|` on stratification A. If C3 fails, the
  margin's marginal advantage does not survive conditioning, and the main article must say so.

C2 and C3 are independent. Each is reported separately, met or not.

## Secondary, explicitly NOT part of any criterion

Per-stratum (Mondrian) calibration on OIA-DDR validation, applied to each cohort. Reported with
a clear exploratory label. It cannot be used to judge the criteria above, which concern the
behaviour of marginally calibrated methods only.

## Isolation

The script reads the existing CSVs in place and writes only under `_LGOVNET_NEW_SEP/results/`.
No detector is run and nothing is refitted except the gate, on the source validation split, with
the five seeds already fixed.
