# Step 4 results: calibration of the gate outputs

Run 2026-09-23 under `PROTOCOL_LOCK_step4_gate_calibration.md`. Nothing here has been written to the manuscript.

Reproduction check: the three published variants were recomputed and matched
`outputs/0506/selective_conformal/baselines.json` on all four cohorts, twelve comparisons, to within 1e-9. The gate
numbers below were read only after that check passed.

## Expected calibration error, 15 equal-width bins

Learned methods are the mean over the five gate seeds.

| Cohort | Prevalence | Raw score | Temperature | Isotonic | Score-MLP | KG gate (B4) | Full gate (B6) | B6 minus isotonic |
|---|---|---|---|---|---|---|---|---|
| OIA-DDR test | 0.3820 | 0.1182 | 0.0836 | 0.0367 | 0.0413 | 0.0266 | **0.0223** | -0.0144 |
| IDRiD | 0.5194 | 0.1108 | 0.1035 | **0.0397** | 0.0397 | 0.0455 | 0.0422 | +0.0026 |
| TJDR | 0.1383 | **0.0537** | 0.0622 | 0.1507 | 0.1617 | 0.1472 | 0.1499 | -0.0008 |
| FGADR | 0.1659 | **0.1017** | 0.1273 | 0.2508 | 0.2493 | 0.2136 | 0.2037 | -0.0471 |

Seed spread of the full gate: 0.0057 in domain, 0.0051 on IDRiD, 0.0008 on TJDR, 0.0014 on FGADR.
In-sample reference on the validation split, where the gate is fitted: 0.0161.

## Brier score

| Cohort | Raw score | Temperature | Isotonic | KG gate (B4) | Full gate (B6) |
|---|---|---|---|---|---|
| OIA-DDR test | 0.2107 | 0.2039 | 0.1974 | 0.1842 | **0.1798** |
| IDRiD | 0.2207 | 0.2205 | 0.2090 | 0.2037 | **0.1993** |
| TJDR | **0.1036** | 0.1048 | 0.1287 | 0.1262 | 0.1258 |
| FGADR | **0.1290** | 0.1313 | 0.1821 | 0.1669 | 0.1599 |

## Direction of the error

| Cohort | Mean predicted, full gate | Observed positive rate | Difference |
|---|---|---|---|
| OIA-DDR test | 0.3871 | 0.3820 | +0.0051 |
| IDRiD | 0.4903 | 0.5194 | -0.0291 |
| TJDR | 0.2882 | 0.1383 | +0.1499 |
| FGADR | 0.3696 | 0.1659 | +0.2037 |

The validation split on which the gate is fitted has a positive rate of 0.3560. The mean prediction of the gate stays
between 0.29 and 0.49 on every cohort while the observed rate ranges from 0.14 to 0.52. The gate moves toward the
target rate but far too little, so the error under shift is a prevalence shift carried from the source, not noise. The
sign follows the same rule on every cohort: the gate over-predicts where the target rate is below the source rate and
under-predicts where it is above.

## Reading against the rule fixed before the run

The rule was: gate ECE at or below the isotonic variant on the external cohorts supports the probability reading, above
it makes the result a limitation.

- In domain the full gate has the lowest ECE and the lowest Brier score of every method measured, including isotonic
  regression fitted on the same validation split.
- On the external cohorts the full gate is below isotonic on TJDR (-0.0008) and FGADR (-0.0471) and above it on IDRiD
  (+0.0026, smaller than the 0.0051 spread across seeds).
- That comparison is a low bar on two cohorts, because isotonic regression itself degrades under shift and ends worse
  than the raw score on TJDR (0.1507 against 0.0537) and FGADR (0.2508 against 0.1017).
- In absolute terms an ECE near 0.15 on TJDR and 0.20 on FGADR is large. Outside the source domain the gate output
  should not be read as a calibrated probability by anyone, and neither should the temperature or isotonic variants.

## What this means for the manuscript, if the author decides to use it

1. The in-domain result supports the wording already in the paper. The certificate field `retain_probability` is a
   calibrated probability on the cohort where the gate is fitted and evaluated, better calibrated than the isotonic
   baseline of Supplementary S16.
2. The external result is a limitation and reads naturally beside the coverage-gap result: the operating point transfers
   better with the gate than with raw confidence, while the probability scale itself does not transfer for any method.
3. The decision-curve analysis is not undermined, because it is run on OIA-DDR test only, where the full gate is
   the best-calibrated method measured here. The published net-benefit table (`make_s8_net_benefit.py`, reading
   `net_benefit_isotonic_check.json`) calibrates every system with isotonic regression fitted on the OIA-DDR
   validation split. The bootstrap contrast for the deployed gate (`net_benefit_bootstrap.json`, contrast
   `C_gate_terpasang`) reads the raw gate output, which is the case this step shows to be calibrated in domain.
   Corrected 2026-09-24: an earlier version of this note wrongly said temperature scaling was applied and that the
   raw gate output was never read.

## Files

- `PROTOCOL_LOCK_step4_gate_calibration.md`, `step4_gate_calibration.py`, `step4_gate_calibration.json`,
  `step4_gate_calibration.log`
- `verify_agg_equivalence.py` and `.log`: proof that reducing the script's memory use changed no number. IDRiD was
  recomputed with the original logic and every per-seed and mean value matched exactly, largest difference 0.000e+00.
