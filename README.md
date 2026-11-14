# CEG-Net: Complementary Evidence for Auditable Diabetic Retinopathy Lesion Triage

Anonymous code repository accompanying the submitted manuscript. It contains the
implementation of the evidence computation and of the post-hoc gate, the fixed
clinical knowledge graph as the pipeline reads it at run time, the scripts that
generate manuscript tables, and their seed-level result files. Historical
generators retain research-workspace paths; see the reproduction boundaries below.

The gate is **post-hoc**: it scores the outputs of a frozen detector and never
changes detection. The tables and figures in the paper are regenerated from the
JSON files in `results/` without a GPU and without re-running the detector. The
JSON files themselves are produced from the per-detection evidence vectors and the
cached detections in `results/evidence/` and `results/detector_outputs/`.

## Layout

| Path | Contents |
|---|---|
| `src/` | Evidence signals, knowledge graph, and the modules of an earlier design (see the next two rows) |
| `src/kg/` | Graph schema and the fixed lesion-pair prior (`schema.py`), builder, dataset statistics. `encoder.py` (a graph-attention encoder) is exploratory and does not enter the reported gate |
| `src/governance/` | `evidence.py` computes the evidence signals (SAS, SDCA, APS, PSS and the multi-source signals). `gate.py`, `routing.py`, `refine.py` and `certificate.py` are **legacy modules of an earlier four-way design** and produce no reported number; they are kept because older training scripts import them. Where the reported gate, tiers and certificates live is listed below |
| `configs/` | `kg.yaml` holds the node set, the five edge types and the fixed author-defined severity costs |
| `tools/` | Table generators, benchmark and evaluation entry points |
| `journal/figures/scripts/` | Figure generators, one per figure, with the shared style module |
| `scripts/` | Standalone analyses reported in the paper and its appendices |
| `results/tables_source/` | Seed-level JSON behind the main and supplementary tables |
| `results/published_tables/` | The tables as typeset in the paper (source file names; LaTeX assigns the numbers), against which the generators are checked |
| `results/experiments/` | JSON behind the added baselines and the robustness checks |
| `results/certificates/` | Certificate manifest, the five verification checks, one sample record per cohort, and the full certificates for all four cohorts, compressed |
| `results/evidence/` | Per-detection evidence vectors for OIA-DDR validation and test, IDRiD, TJDR and FGADR (gzipped CSV, local path columns removed), and the scores of the five comparison detectors on the OIA-DDR candidates |
| `results/detector_outputs/` | Cached detections per cohort: the primary detector, its horizontally flipped pass, and the secondary detector (zip). The TJDR and FGADR secondary exports include the lesion-free images (`merge_note.json` in each zip) |
| `results/conditional_coverage/` | Results of the conditional-coverage, image-disjoint calibration, miscoverage-level sensitivity, ensemble, gate-calibration and score-averaging operating-point analyses, with their run logs |
| `scripts/conditional_coverage/` | Scripts of those analyses and of their tables |
| `PROTOCOL_LOCK_*.md` | Protocols written before each locked analysis was run |

## Where the paper lives in the code

| Paper element | Code |
|---|---|
| Evidence signals SAS, SDCA, APS, PSS and the multi-source signals | `src/governance/evidence.py` (`compute_sas`, `compute_sdca`, `compute_aps`, `compute_pss`, `compute_source_disagreement`), assembled per detection by `tools/audit_stage3c_governance_targets.py`, which also writes the RETAIN target |
| Lesion-pair prior $\mathrm{SDM}_{\mathrm{norm}}$ | costs in `configs/kg.yaml`, table built by `build_clinical_prior_sdm` in `src/kg/schema.py` and written by `tools/audit_stage3a_kg.py`. The costs are an author-defined severity ordering; the identifiers that mention ICDR are historical |
| Gate $g(x)$ | `fit_mlp` and `apply_mlp` in `tools/eval_fp_reduction_benchmark.py` |
| Operating point $\tau^\star$ (true-positive retention 0.80) | `threshold_for_tp_ret`, same file |
| Split-conformal threshold, coverage gap, risk-coverage area | `conformal_threshold`, `coverage_at`, `aurc` in `tools/eval_selective_conformal_baselines.py`; external cohorts and score averaging in `scripts/conditional_coverage/` |
| Four tiers (Algorithm 1) | `assign` in `tools/make_tier_routing.py`, $\delta = 0.30$ |
| Audit certificates | `tools/emit_certificates.py`, checked by `tools/verify_certificates.py` |
| Score fusion and the fusion-input gate | `scripts/gate_on_fusion.py` (in domain), `scripts/fusion_external.py` (external), `scripts/conditional_coverage/step6_fgadr_three_pass.py` (FGADR) |

Internal identifiers keep the project's earlier working name, LGov-Net (`lgovnet`,
`LGG`, `governance_*`), and some paths name research-workspace folders
(`_septem/`, `_NATIVE/`, `_LGOVNET_NEW_SEP/`, `outputs/0506/`). Both refer to the
CEG-Net pipeline described in the paper.

## Reproducing the tables

Every table of the paper that reports a result is regenerated from the bundled
files, without a GPU, without fitting a detector and without the image datasets,
and compared with the typeset version in `results/published_tables/`, ignoring
LaTeX comment-only lines while checking all table content exactly:

```bash
python tools/verify_tables_against_generators.py
```

The script runs each generator with its file writes redirected into a temporary
directory, so nothing in the repository is overwritten, and exits non-zero if any
table differs. One generator currently requires a NumPy version with
`numpy.trapezoid`; without it, 30 tables match and that generator fails. Three
files are not generated: `T_impl_hyperparams.tex` and `T_impl_computational.tex`
state settings rather than results, and `T14_category_breakdown.tex` is not
typeset. To write the tables instead, run a generator directly; each writes under
`_FINAL_Latex/tables/`. For the dataset table, the ground-truth columns are the
published counts, recomputed from the annotation XML when the datasets are placed
under `data/`; the evaluated-image counts and detections per image are read from
`results/detector_outputs/`.

The finalization tables alone can also be rendered with

```bash
python scripts/finalization/render_final_tables.py
```

which writes `T15_fusion_evidence.tex` and `S10_native_gate_control.tex`, with the
FGADR row of T15 read from `results/conditional_coverage/step6_fgadr_three_pass.json`.

Two checks run from the bundled artifacts without a model. The certificate
verification regenerates the supplementary table from `results/certificates/verification.json`:

```bash
python tools/make_certificate_verification_table.py
```

`tools/emit_certificates.py` and `tools/verify_certificates.py` produce that JSON from
the per-detection evidence vectors in `results/evidence/`, read at the workspace paths
in their source. The emitted
certificates for all four cohorts (OIA-DDR, IDRiD, TJDR, FGADR; 634,423 records in total)
are bundled compressed (gzip) so the records themselves can be inspected without
regenerating them, and `manifest.json` records the SHA-256 of every source file.
The annotation-grounded endpoint is reproduced the same way, from
`results/experiments/annotation_endpoint.json` with `scripts/make_annotation_endpoint_table.py`;
its criteria were fixed before the run in `PROTOCOL_LOCK_annotation_endpoint.md`.

The table generators above resolve their inputs in either layout. The experiment
runners that produced the JSON files (for example `tools/eval_fp_reduction_benchmark.py`
and the scripts under `scripts/`) read the uncompressed evidence CSVs and detector
caches at research-workspace paths such as `outputs/0506/`; unpack
`results/evidence/` and `results/detector_outputs/` and map those paths before
re-running an experiment. Figure scripts likewise have different input requirements.
The architecture schematic can be rendered with
`python scripts/finalization/render_architecture.py`.

The primary MLP protocol (the exploratory graph control uses its separately documented optimizer): the gate is fitted on the
OIA-DDR validation split only (one hidden layer of 16 tanh units, full-batch
gradient descent, 1500 iterations, learning rate 0.1), over five initialization
seeds; AUPRC is average precision; confidence intervals are image-clustered
paired bootstrap with 2000 resamples; the operating point is the threshold that
retains 0.80 of the RETAIN targets (exact true positives, clinically close
confusions and localization-weak detections) on validation.

The eight-detector panel is the one exception to that protocol. It uses a
lighter per-detector configuration at a single seed: single-source agreement,
the geometric anatomy approximation and a placeholder perturbation stability.
Its eight benchmark files are included as
`results/tables_source/*_fp_benchmark.json`, and the four-tier routing table is
produced with the knowledge-graph gate at seed 0 rather than the full gate.

## Standalone analyses in `scripts/`

| Script | Question it answers |
|---|---|
| `baselines_nokg.py` | Do test-time augmentation and detector ensembling, without any knowledge graph, explain the gain? |
| `gate_on_fusion.py` | Does the evidence still add value when the fused detector score is itself an input? |
| `fusion_external.py` | Does that hold outside OIA-DDR, under the zero-shot protocol? |
| `clinical_cost_fusion.py` | Does score ranking distinguish the author-defined prior-cost categories? |
| `clinical_cost_external.py` | Does that answer hold on the three external cohorts as well? |
| `make_s6_clinical_cost_external.py` | Builds the cross-cohort clinical-cost table. |
| `export_fused_scores.py` | Writes the fused scores that the tier-routing table reads. |
| `circularity_check.py` | Does the gain survive when the label no longer uses the semantic-distance matrix? |
| `linear_gate_transfer.py` | Does a linear gate transfer as well as the MLP gate? |
| `t12_tie_audit.py` | Is any reported AUPRC sensitive to the ordering of tied scores? |
| `make_s5_circularity.py` | Builds the label-circularity supplementary table. |
| `target_recalibration.py` | Does a small labeled sample from the target domain restore the operating point that does not transfer? |
| `source_domain_sensitivity.py` | Does the gating result depend on OIA-DDR being the cohort the gate is fitted on? |
| `net_benefit_v2.py` | What is the net benefit of each system across cost ratios, thresholding calibrated probabilities? Also writes the score cache the bootstrap reads. |
| `net_benefit_isotonic_check.py` | Does the gate still lead once score fusion is given its strongest reasonable calibrator? |
| `net_benefit_bootstrap.py` | Do the net-benefit margins survive an image-clustered bootstrap? Run `net_benefit_v2.py` first, which writes the score cache. |
| `net_benefit.py` | Superseded and kept so the correction stays visible: it thresholded score quantiles, which caps net benefit at the threshold probability. Use `net_benefit_v2.py`; do not cite `net_benefit.json`. |
| `make_s7_target_recalibration.py` | Builds the target-recalibration supplementary table. |
| `make_s8_net_benefit.py` | Builds the net-benefit supplementary table. |
| `make_s9_source_domain_sensitivity.py` | Builds the source-domain-sensitivity supplementary table. |

Analysis scripts read their inputs and write their own result files. Table generators write LaTeX; do not rerun historical experiments over archived results.

## Finalization evidence and limits

Score averaging is the main ranking comparator. On OIA-DDR, three-pass averaging
reaches AP 0.7224 and a gate whose raw-score input is replaced by fusion reaches
0.7332. The additional gain does not transfer to IDRiD, TJDR or FGADR. On FGADR,
three-pass averaging reaches AP 0.5348 against 0.5085 for the fusion-input gate,
with all five paired intervals below zero (locked in
`PROTOCOL_LOCK_step6_fgadr_three_pass.md`). No detector was retrained.

On the operating point the order reverses. Under a protocol locked before the run
(`PROTOCOL_LOCK_step5_fusion_external_coverage.md`), the split-conformal threshold
calibrated on OIA-DDR validation at a nominal retention of 0.80 gives score averaging
coverage gaps of +0.046, +0.116, +0.039 and +0.112 on OIA-DDR test, IDRiD, TJDR and
FGADR, against +0.019, +0.088, -0.020 and +0.092 for the full gate. The gate is closer to nominal on all four cohorts, and the paired
image-clustered interval of the difference in absolute gap lies below zero on three
and includes zero on TJDR, while averaging keeps the lower risk-coverage area on all
four. Script `scripts/conditional_coverage/step5_fusion_external_coverage.py`, result
`results/conditional_coverage/step5_fusion_external_coverage.json`. The FGADR entry
comes from step 6 (`scripts/conditional_coverage/step6_fgadr_three_pass.py`, result
`results/conditional_coverage/step6_fgadr_three_pass.json`), which repeats every FGADR
comparison with the three-pass mean; the step-5 file keeps its superseded two-pass FGADR
row. Both runs read the merged detector caches that include the lesion-free images,
which correspond to the zips in `results/detector_outputs/`; the script headers give
the unpacking layout.

`results/finalization/` contains the archived JSON used by the new tables.
`scripts/finalization/fgadr_two_pass.py` preserves the superseded two-pass FGADR
follow-up for provenance; it reads the CSV evidence and detector caches at the workspace paths in its
source, writes beside itself, and refuses to overwrite a completed result. This
is an experiment runner, not needed for table regeneration. The original graph
control source is retained at `_NATIVE/scripts/native_gate_experiment.py`; it also
reads the evidence and graph artifacts at workspace paths. Its archived snapshot does
not establish a topology-specific gain over rewiring. The primary system uses a
direct fixed prior lookup; anatomy remains exploratory.

The prior costs are author-defined, not validated clinical harm. Audit records
can accompany fusion without changing its ranking. Native audit output does not
mean only the gate can provide certificates.

## What is not in this repository

Model checkpoints of the primary and the secondary detector are not included because of their size.

## Data

No new data was collected. The four fundus datasets are public and must be
obtained from their original providers: OIA-DDR, IDRiD, TJDR and FGADR. The
manuscript cites each one.

## License

Released for peer review. A permissive open-source license will be attached on
publication.
