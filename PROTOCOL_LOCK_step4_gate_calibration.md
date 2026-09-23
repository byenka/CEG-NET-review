# PROTOCOL LOCK - step 4: calibration of the gate outputs

Locked 2026-09-23, before the run. Nothing in this file is changed after the results exist.

This step is exploratory. Its files stay in `_cache/` and nothing from it enters the manuscript until the author decides.

## Purpose

The manuscript calls the gate output a retain probability (Sections 1 and 3) and the decision-curve analysis thresholds
it on a probability axis (Section 6). The calibration table of Supplementary S16 reports the expected calibration error
(ECE) and the Brier score of the detector score only, in three variants: raw, temperature-scaled and isotonic. No ECE or
Brier is reported for any gate. Measure them, so that the probability reading of the gate output is either supported by
evidence or dropped.

Descriptive only. No pass/fail criterion is defined, and no claim of superiority is prepared in advance.

## Fixed design

- Features: `outputs/0506/evsds_v2sec/oiaddr/{valid,test}`, `outputs/0506/evsds_pss/{idrid,tjdr}/test`,
  `outputs/0506/evsds_pss/fgadr/train`. These are the evaluation populations of the paper, including the 114 lesion-free
  images (538,986 FGADR and 23,181 TJDR detections).
- Label: the paper's composite `governance_retain` target (RETAIN versus SUPPRESS).
- Methods, built by `tools.eval_fp_reduction_benchmark.build_methods` with its unchanged recipe, fitted on the OIA-DDR
  validation split only and applied unchanged to every cohort, seeds 0-4:
  - `M0_score` as the three published variants: raw (min-max to the unit interval), temperature scaling with the
    temperature fitted on validation, and isotonic regression fitted on validation.
  - `Evidence-LogReg`, `Score-MLP`, `KG-gate-MLP (ours)` (B4), `KG-gate+SDS (B5)`, `KG-gate+SDS+corrob (B6)`.
- Metrics: ECE with the 15 equal-width bins of `tools.eval_selective_conformal_baselines.ece`, the Brier score of the
  same module, the mean predicted probability, the observed positive rate and their signed difference.
- Learned methods are reported as the mean and standard deviation over the five seeds. Deterministic methods are
  reported once.
- Evaluation is on the OIA-DDR test split and the three external cohorts. The OIA-DDR validation split is reported
  separately and is marked in-sample for every fitted method, so it is not part of any comparison.

## Reproduction check, run before the new numbers are read

The script recomputes the three published variants and compares them with `outputs/0506/selective_conformal/baselines.json`,
the frozen source of Supplementary S16. A difference above 1e-9 aborts the run.

## Two asymmetries recorded in advance

1. The gates, the two calibrated score variants, `Score-MLP` and `Evidence-LogReg` are all fitted to the RETAIN target on
   the validation split. The raw detector score is not fitted to that target at all, so part of its ECE is a definition
   mismatch rather than miscalibration. The fair comparators of the gate are the temperature and isotonic variants and
   the two other learned methods, not the raw score.
2. The published raw and temperature variants are rescaled to the unit interval by a min-max transform
   (`to_unit`), which is not a probability transform. The gate output is a sigmoid and is read directly, without
   rescaling. The comparison is therefore between a probability and two rescaled scores, and it is reported as such.

## Reading rule fixed in advance

- If the gate ECE is at or below the isotonic variant on the external cohorts, the probability reading of the gate output
  is supported and a table may be added.
- If the gate ECE is above the isotonic variant, the result is reported as a limitation, or the manuscript stops calling
  the output a probability. Renaming has a known cost: 634,423 emitted certificates carry the field `retain_probability`
  and their SHA-256 values are recorded in `results/certificates/manifest.json`.
- In both cases the numbers are reported as measured.

## Outputs

- `_cache/step4_gate_calibration.py` (the script)
- `_cache/step4_gate_calibration.json` (the results)
- `_cache/step4_gate_calibration.log` (the run log)

## Promotion note, added 2026-09-23 after the run

The locked text above is unchanged. The files moved out of the working `_cache/` folder to the locations used by the
earlier protocol locks:

- protocol: `_LGOVNET_NEW_SEP/PROTOCOL_LOCK_step4_gate_calibration.md`, repository root `PROTOCOL_LOCK_step4_gate_calibration.md`
- script: `_LGOVNET_NEW_SEP/scripts/step4_gate_calibration.py`, repository `scripts/conditional_coverage/step4_gate_calibration.py`
- results and log: `_LGOVNET_NEW_SEP/results/`, repository `results/conditional_coverage/`

The only edit to the result file is the `protocol` field, which now names this path instead of the `_cache` one. Every
number in it is byte-identical to the run.
