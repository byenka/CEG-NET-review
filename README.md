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
| `src/` | Evidence pillars, knowledge graph, gate, routing, certificate, losses, models |
| `src/kg/` | Graph schema, builder, encoder, dataset statistics |
| `src/governance/` | `evidence.py` (SAS, SDCA, APS, PSS), `gate.py`, `routing.py`. `certificate.py` is a legacy schema kept as documentation; the reported certificates come from `tools/emit_certificates.py` |
| `configs/` | `kg.yaml` holds the node set, the five edge types and the fixed author-defined severity costs |
| `tools/` | Table generators, benchmark and evaluation entry points |
| `journal/figures/scripts/` | Figure generators, one per figure, with the shared style module |
| `scripts/` | Standalone analyses reported in the paper and its appendices |
| `results/tables_source/` | Seed-level JSON behind the main and supplementary tables |
| `results/experiments/` | JSON behind the added baselines and the robustness checks |
| `results/certificates/` | Certificate manifest, the five verification checks, one sample record per cohort, and the full certificates for all four cohorts, compressed |
| `results/evidence/` | Per-detection evidence vectors for OIA-DDR validation and test, IDRiD, TJDR and FGADR (gzipped CSV, local path columns removed), and the scores of the five comparison detectors on the OIA-DDR candidates |
| `results/detector_outputs/` | Cached detections per cohort: the primary detector, its horizontally flipped pass, and the secondary detector (zip) |
| `results/conditional_coverage/` | Results of the conditional-coverage, image-disjoint calibration and ensemble analyses |
| `scripts/conditional_coverage/` | Scripts of those analyses and of their tables |
| `PROTOCOL_LOCK_*.md` | Protocols written before each locked analysis was run |

## Reproducing the tables

The finalization tables are reproducible directly from the bundled JSON, without
fitting a model or accessing detector caches:

```bash
python scripts/finalization/render_final_tables.py
```

This writes `T15_fusion_evidence.tex`, `S10_native_gate_control.tex`, and
`S11_fgadr_fusion_followup.tex` under `_FINAL_Latex/tables/`. These are source
filenames; manuscript numbering is assigned by LaTeX. The generator also applies
editorial caption corrections to T8/S6 if those files are present.

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

Older generators and data are retained. They are not all turnkey: many expect
`outputs/0506/` or `_septem/scripts_optionB/` paths from the research workspace,
and some require per-detection evidence. Map their documented input paths before
running them; bundled JSON alone does not reproduce the entire pipeline.
Figure scripts likewise have different input requirements. The architecture
schematic can be rendered with `python scripts/finalization/render_architecture.py`.

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
0.7332. The additional gain does not transfer to IDRiD or TJDR. A separate FGADR
follow-up uses primary-plus-flip averaging at both source fitting and target
evaluation: AP 0.5344 for averaging versus 0.5144 for the gate, with all five paired
intervals below zero. Historical secondary evidence is retained, so this is not a
two-pass compute-budget comparison. No detector was retrained.

`results/finalization/` contains the archived JSON used by the new tables.
`scripts/finalization/fgadr_two_pass.py` preserves the exact executed follow-up;
it reads the CSV evidence and detector caches at the workspace paths in its
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
