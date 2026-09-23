# Protocol lock — prior validation against the expert-annotation endpoint

**before** the experiment was run. The criteria, controls, and interpretation rules
below must not be changed after the results are seen. Results will be reported regardless of the outcome.

## Question

Does the conflict score from the author-defined lesion-pair prior predict class errors determined
independently by expert annotations, and does it add information beyond ordinary detector ambiguity?

This differs from Table 13 of the manuscript, which, according to the manuscript itself, is **partly definitional**:
the categories there are formed using an SDM threshold and then evaluated with SDCA, which also derives from SDM.
Here, the outcome comes from annotations without using SDM.

## Population

Detections with `best_iou >= 0.5` and non-empty `best_gt_class`, i.e., detections localized
on annotated lesions. Counts verified before locking:

| Cohort | Localized | Misclassified | Rate | Images |
|---|---:|---:|---:|---:|
| OIA-DDR validation (fitting only) | 3,530 | 615 | 17.4% | 146 |
| OIA-DDR test | 6,800 | 1,309 | 19.2% | 224 |
| IDRiD | 945 | 118 | 12.5% | 27 |
| TJDR | 621 | 126 | 20.3% | 82 |
| FGADR | 15,046 | 1,941 | 12.9% | 1,214 |

## Outcome (from expert annotations only)

Primary: `mismatch = 1` when `pred_class != best_gt_class`, meaning the label emitted by the detector
differs from the expert class. Sensitivity analysis: the same outcome using `majority_class` (ROI argmax),
because the two differ for 4.41% of detections.

SDM is not used at all to define the outcome.

## Predictors (from model outputs and the prior only; no GT)

| Code | Signal | Role |
|---|---|---|
| SDCA | `ev_sdca`, author-defined prior conflict | primary method |
| MARGIN | `unc_margin` | **key control**: ambiguity without pair structure |
| ENTROPY | `unc_entconf` | alternative ambiguity control |
| MSP | `unc_msp` | confidence control |
| PERM | SDM with the 6 off-diagonal values permuted, 5 draws, seeds 101–105 | structure control |
| EMP | empirical annotation-based relation: P(GT=b \| pred=a) with add-1 smoothing | data-based relation control |

EMP is estimated **only on OIA-DDR validation**. There is no OIA-DDR training-split CSV; validation is
the fitting split used throughout the manuscript and is separate from the test set and external cohorts.

MARGIN, ENTROPY, and MSP are oriented in the direction that **maximizes AUROC on validation**, then
their signs are locked and used as-is on the test set and external cohorts. This orientation deliberately
favors the controls, not the primary method.

## Metrics

AUROC and AUPRC for predicting mismatch; odds ratio for high- versus low-conflict detections at the
manuscript's own threshold (SDCA >= 0.30); image-clustered bootstrap with 2,000 resamples, seed 0.

Incremental value beyond ambiguity is tested with logistic regression **fit on OIA-DDR validation**
and then applied to the test set and external cohorts: MARGIN-only model versus MARGIN + SDCA model.

## Device-positive control

The stored `ev_sdca` must be reconstructible from `SDM_norm[argmax ROI, runner-up]` for
all rows. If reconstruction fails, the results must not be interpreted.

## Passing criteria (all required)

- **A1** SDCA AUROC > 0.5 with a bootstrap CI that does not include 0.5, on the OIA-DDR test set **and** on
  at least 2 of the 3 external cohorts.
- **A2** SDCA outperforms the permutation control: AUROC difference > 0 with a paired CI > 0 on the
  OIA-DDR test set.
- **A3** Incremental value beyond ambiguity: AUROC(MARGIN + SDCA) > AUROC(MARGIN) with a paired CI
  > 0 on the OIA-DDR test set **and** on at least 2 of the 3 external cohorts.

## Interpretation rules (locked in advance)

1. If A1 through A3 are met **but** AUROC EMP >= AUROC SDCA in at least 2 cohorts, the valid claim is
   that the annotation-derived relation is at least as informative as the author's prior. This must be
   reported, and replacing the prior with the empirical relation becomes the more defensible option.
2. If A3 fails, the only result established is that "more uncertain models are wrong more often." This is
   **not** a claim about the prior and must not be written as prior validation.
3. Maximum claim if everything passes: *the fixed lesion-relation signal has retrospective
   criterion validity against expert-derived lesion annotations across cohorts*. **Not** a claim
   about clinical harm or patient safety.

## Isolation

The script reads only the evidence CSVs and graph at their original locations. All outputs are written under
`_NATIVE/`. The manuscript is not touched until the results are available and the author decides.
