# Protocol lock — Step 0: conditional coverage by corroboration stratum

 before the experiment is run. The criteria below must not be changed after the results are seen. Results are reported regardless of the outcome.

## Background

All conformal guarantees measured so far are marginal. The five-detector ensemble achieves an overall coverage gap of +0.000 on the OIA-DDR test. Marginal guarantees are known to hide subgroup failures: coverage can be high in boxes corroborated by many detectors and low in boxes without corroboration, while the average remains nominal.

The positive rate according to the number of detectors that agree ranges from 0.086 (0 members) to 0.844 (5 members), an almost tenfold range. A single global threshold cannot serve both extremes equally well. Step 0 tests this hypothesis.

## Question

Does the conformal threshold calibrated marginally on OIA-DDR validation provide coverage close to nominal within each corroboration stratum on the OIA-DDR test?

## Population

All candidate detections from the primary detector on the OIA-DDR test, target `governance_retain`, identical to protocol `tools/eval_fp_reduction_benchmark.py`.

## Stratum taxonomy

`k` = number of members from the five detcmp detectors (`fasterrcnn`, `fasterrcnn_v2`, `fcos`, `retinanet`, `retinanet_v2`) that have a box matched to the primary box at a minimum IoU of 0.15. `k` ranges from 0 to 5. This taxonomy uses no annotations.

## Methods under test

| Code | Score |
|---|---|
| S | primary detector score |
| S+E | full gate, five seeds, fitted on validation |
| E5 | mean of the five detcmp detectors |
| E6 | mean of the primary plus five detcmp detectors |

## Procedure

For each method: compute the split-conformal marginal threshold on OIA-DDR validation with miscoverage `alpha = 0.20`, using the quantile `ceil((n+1)(1-alpha))/n` of positive nonconformities, exactly as in `tools/eval_selective_conformal_baselines.py`. Apply that threshold unchanged to the test set, then measure empirical coverage within each stratum.

Per-stratum difference = empirical coverage minus 0.80.

## Criteria, locked in advance

- **DEAD.** If every method has `|difference| <= 0.05` in **every** stratum, the marginal guarantee is sufficient, the idea of conditional coverage has no room, and the work is stopped.
- **CONTINUE.** If at least one method has `|difference| > 0.15` in at least one stratum containing at least 200 positive detections.
- **GRAY.** If the maximum difference lies between 0.05 and 0.15, the signal is weak; results are reported and the decision is left to the authors without automatic continuation.

Strata with fewer than 200 positive detections are reported but **not** used to trigger the CONTINUE criterion, because coverage in small samples is unstable.

## Secondary outputs, NOT part of the criteria

As a preview of Step 1, the script also computes the conformal threshold **per stratum** (Mondrian) calibrated on validation and applied to the test set. These values are reported with clear labels and **must not** be used to assess the criteria above, because the criteria concern only the behavior of the marginal methods.

## Isolation

The script only reads the cache and CSV files in their original locations. All outputs are written under `_LGOVNET_NEW_SEP/`. The `_FINAL_Latex` manuscript is not touched.
