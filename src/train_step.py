"""
train_step.py
LGov-Net Integration Step — Phase 14 (orchestration only).

This module implements one minimal end-to-end training step that connects
every LGov-Net component in the correct order.  It does NOT contain any
training loop, Trainer class, callback system, registry, or dataloader
boilerplate — it only orchestrates the existing modules.

Closed-loop data flow
─────────────────────
detector outputs
    │  class_probs + head_predictions  (per bbox, per image)
    ▼
KG encoder   (LGovNetEncoder)
    │  node_embeddings [N=9, d=64]
    │  SDM_learned (pairwise L2 distances)
    │  Compat (lesion×zone anatomical scores)
    ▼
Evidence computation  (evidence.build_evidence_batch)
    │  g(b_j) = [SAS, SDCA_norm, APS, PSS]  ∈ R^4 per bbox
    ▼
Gate input assembly   (gate.build_pred_embedding + gate.concat_gate_input)
    │  gate_input = [SAS, SDCA_norm, APS, PSS, e_pred]  ∈ R^{4+d} per bbox
    ▼
Governance Gate       (LGG.gumbel_forward — training; LGG.forward — inference)
    │  tier_logits [B, 4], tier_probs [B, 4]
    ▼
Routing               (routing.route_bbox_batch)
    │  target_tier records — supervision targets for L_gate
    ▼
R3 Refinement         (refine.refine_r3_batch)
    │  upgraded / downgraded / kept_as records
    ▼
SDGL Loss             (SDGLLoss)
    │  L_SDGL = λ1·L_R1 + λ2·L_R2 + λ3·Σ_j w_j·L_R3
    │  L_KG   = Σ_{fixed edges} (||e_ci − e_cj||² − w_target)²
    ▼
Gate Loss             (GateLoss)
    │  L_gate = CE(tier_logits, target_tier_indices)
    ▼
Total Loss            (compute_total_loss)
    └► L_total = L_SDGL + λ_gate·L_gate + λ_KG·L_KG
                 + λ_eff·L_eff (hook, 0.0 until implemented)
                 + λ_aux·L_aux (hook, 0.0 until implemented)

Public API
──────────
  run_lgovnet_step(...)                → StepOutput dict
  build_step_debug_info(...)           → dict
  summarize_tier_stats(routing_batch)  → dict
  summarize_fallback_usage(evidence_batch) → dict

STUBS (non-blocking, clearly marked)
──────────────────────────────────────
  S1 — detector_stub(image_ids, cfg)  : real backbone not yet integrated.
       Returns synthetic DetectorBox dicts with random class_probs.
       Phase 15 replaces this with a real forward pass.
  S2 — L_R1 / L_R3 are classification-only NLL (no smooth-L1 / GIoU).
       SDGL stubs are already documented in sdgl.py.
  S3 — PSS is always placeholder (FALLBACK_PSS = 0.5).
       Real perturbation loop belongs in Phase 15.
  S4 — KG graph artifacts are built fresh from scratch each step (no
       caching).  Phase 15 should cache and only re-run when embeddings
       are updated.

Dependencies
────────────
  src/kg/encoder.py            (LGovNetEncoder, build_gat_encoder,
                                build_encoder_inputs, compute_sdm,
                                compute_compat, normalize_sdca)
  src/kg/builder.py            (build_graph_from_artifacts)
  src/governance/evidence.py   (build_evidence_batch)
  src/governance/gate.py       (LGG, build_pred_embedding, concat_gate_input,
                                get_gumbel_tau)
  src/governance/routing.py    (route_bbox_batch, TIER_NAMES)
  src/governance/refine.py     (refine_r3_batch)
  src/governance/contracts.py  (make_placeholder_pss, LESION_CLASSES)
  src/losses/sdgl.py           (SDGLLoss, compute_total_loss)
  src/losses/gate_loss.py      (GateLoss)
  src/utils/pipeline_metrics.py (compute_pipeline_metrics)

  NO dependency on any external training framework.
"""

from __future__ import annotations

import sys
import os
import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Project root insertion (enables `python src/train_step.py` from root) ──
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.join(_here, "..")
if _root not in sys.path:
    sys.path.insert(0, _root)

from lgovnet.src.kg.encoder import (
    LGovNetEncoder,
    build_gat_encoder,
    build_encoder_inputs,
    compute_sdm,
    compute_compat,
    normalize_sdca,
    LESION_NAMES,
    ZONE_NAMES,
)
from lgovnet.src.kg.builder import build_graph_from_artifacts
from lgovnet.src.governance.evidence import build_evidence_batch
from lgovnet.src.governance.gate import (
    LGG,
    build_pred_embedding,
    concat_gate_input,
    get_gumbel_tau,
)
from lgovnet.src.governance.routing import route_bbox_batch, TIER_NAMES
from lgovnet.src.governance.refine import refine_r3_batch
from lgovnet.src.governance.contracts import make_placeholder_pss, LESION_CLASSES
from lgovnet.src.losses.sdgl import SDGLLoss, compute_total_loss
from lgovnet.src.losses.gate_loss import GateLoss
from lgovnet.src.utils.pipeline_metrics import compute_pipeline_metrics

# ── Constants ──────────────────────────────────────────────────────────────────

#: Tier name → integer index (0=R1, 1=R2, 2=R3, 3=R4).
_TIER_INDEX: Dict[str, int] = {t: i for i, t in enumerate(TIER_NAMES)}

#: Canonical lesion class order (must match LESION_CLASSES and LESION_NAMES).
_LESION_CLASSES: List[str] = list(LESION_CLASSES)

#: Zone list (must match encoder.ZONE_NAMES).
_ZONES: List[str] = list(ZONE_NAMES)


# ══════════════════════════════════════════════════════════════════════════════
# STUB: Detector interface
# ══════════════════════════════════════════════════════════════════════════════


def detector_stub(
    image_ids: List[str],
    bboxes_per_image: int = 4,
    seed: int = 0,
) -> Dict[str, List[dict]]:
    """STUB — synthetic detector output for testing/integration.

    Returns DetectorBox dicts grouped by image_id.  All class_probs are
    drawn from a Dirichlet distribution so they sum to 1.  Head predictions
    mimic 4 independent detection heads.

    Replace this function with the real backbone forward pass in Phase 15.

    Args:
        image_ids:          List of image identifiers.
        bboxes_per_image:   Number of synthetic bboxes to generate per image.
        seed:               Random seed for reproducibility.

    Returns:
        {image_id: [DetectorBox, ...]}
        Each DetectorBox has:
          bbox_id, assigned_zone, ensemble_probs, head_predictions
    """
    torch.manual_seed(seed)
    random.seed(seed)

    results: Dict[str, List[dict]] = {}
    for img_id in image_ids:
        boxes: List[dict] = []
        for b_idx in range(bboxes_per_image):
            bbox_id = f"{img_id}_b{b_idx:02d}"
            zone = random.choice(_ZONES)

            # Ensemble probs: Dirichlet(1,1,1,1) → roughly uniform
            ens_raw = torch.distributions.Dirichlet(torch.ones(4)).sample()
            ensemble_probs: Dict[str, float] = {
                c: float(p) for c, p in zip(_LESION_CLASSES, ens_raw)
            }

            # 4 head predictions (one per lesion class head)
            head_predictions: List[dict] = []
            for h_idx, head_class in enumerate(_LESION_CLASSES):
                h_raw = torch.distributions.Dirichlet(torch.ones(4)).sample()
                head_predictions.append(
                    {
                        "head": head_class,
                        "bbox": [
                            float(10 + b_idx * 50 + h_idx),
                            float(10 + b_idx * 50),
                            float(60 + b_idx * 50 + h_idx),
                            float(60 + b_idx * 50),
                        ],
                        "class_probs": {
                            c: float(p) for c, p in zip(_LESION_CLASSES, h_raw)
                        },
                        "score": float(ensemble_probs[head_class]),
                    }
                )

            boxes.append(
                {
                    "bbox_id": bbox_id,
                    "assigned_zone": zone,
                    "ensemble_probs": ensemble_probs,
                    "head_predictions": head_predictions,
                }
            )
        results[img_id] = boxes

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Summary helpers — called BEFORE the step returns
# ══════════════════════════════════════════════════════════════════════════════


def summarize_tier_stats(routing_batch: List[dict]) -> dict:
    """Count how many bboxes landed in each governance tier.

    Args:
        routing_batch: Output of ``route_bbox_batch()``, one dict per bbox.

    Returns:
        {
            "num_r1": int,
            "num_r2": int,
            "num_r3": int,
            "num_r4": int,
            "tier_distribution": {0: int, 1: int, 2: int, 3: int},
            "total_bboxes": int,
        }
    """
    counts = {t: 0 for t in TIER_NAMES}
    for rec in routing_batch:
        tier_name = rec.get("target_tier", "R4")
        counts[tier_name] = counts.get(tier_name, 0) + 1

    return {
        "num_r1": counts["R1"],
        "num_r2": counts["R2"],
        "num_r3": counts["R3"],
        "num_r4": counts["R4"],
        "tier_distribution": {_TIER_INDEX[t]: counts[t] for t in TIER_NAMES},
        "total_bboxes": len(routing_batch),
    }


def summarize_fallback_usage(evidence_batch: List[dict]) -> dict:
    """Count evidence quality issues (SAS fallbacks and missing zones).

    Args:
        evidence_batch: Output of ``build_evidence_batch()``.

    Returns:
        {
            "fallback_sas_count": int,    — bboxes using degraded-proxy SAS
            "missing_zone_count": int,    — bboxes with no assigned_zone
            "total_evidence": int,
        }
    """
    fallback_sas = 0
    missing_zone = 0
    for ev in evidence_batch:
        warnings = ev.get("warnings", [])
        if any("DEGRADED PROXY" in w for w in warnings):
            fallback_sas += 1
        if ev.get("zone") is None:
            missing_zone += 1

    return {
        "fallback_sas_count": fallback_sas,
        "missing_zone_count": missing_zone,
        "total_evidence": len(evidence_batch),
    }


def build_step_debug_info(
    evidence_batch: List[dict],
    routing_batch: List[dict],
    refinement_records: List[dict],
    gate_output: dict,
    loss_dict: dict,
    pipeline_cfg: dict,
) -> dict:
    """Assemble all debug diagnostics for one integration step.

    Combines all required debug metrics (specified in the FASE 14 brief) plus
    additional tensor-shape summaries useful for debugging.

    Required debug metrics:
      - fallback_sas_count
      - tier_distribution
      - r3_refined_to_r2_count
      - r3_downgraded_to_r4_count
      - num_r1 / num_r2 / num_r3 / num_r4
      - mean_sdca
      - mean_aps

    Args:
        evidence_batch:     Output of ``build_evidence_batch()``.
        routing_batch:      Output of ``route_bbox_batch()``.
        refinement_records: Output of ``refine_r3_batch()`` (may be empty).
        gate_output:        Dict from ``LGG.gumbel_forward()`` or ``LGG.forward()``.
        loss_dict:          Dict from ``compute_total_loss()`` wrapper.
        pipeline_cfg:       Config dict passed to the step (for reference).

    Returns:
        Flat dict with all debug metrics.
    """
    tier_stats = summarize_tier_stats(routing_batch)
    fallback_stats = summarize_fallback_usage(evidence_batch)
    pipeline_metrics = compute_pipeline_metrics(
        evidence_batch, routing_batch, refinement_records
    )

    # Mean SDCA and mean APS across the evidence batch
    sdca_vals = [ev["sdca_norm"] for ev in evidence_batch if "sdca_norm" in ev]
    aps_vals = [ev["aps"] for ev in evidence_batch if "aps" in ev]
    mean_sdca = float(sum(sdca_vals) / len(sdca_vals)) if sdca_vals else 0.0
    mean_aps = float(sum(aps_vals) / len(aps_vals)) if aps_vals else 0.0

    # Gate confidence stats
    tier_probs = gate_output.get("tier_probs")
    gate_confidence = None
    if tier_probs is not None:
        gate_confidence = float(tier_probs.max(dim=-1).values.mean().item())

    return {
        # ── Tier counts ────────────────────────────────────────────────────
        "num_r1": tier_stats["num_r1"],
        "num_r2": tier_stats["num_r2"],
        "num_r3": tier_stats["num_r3"],
        "num_r4": tier_stats["num_r4"],
        "total_bboxes": tier_stats["total_bboxes"],
        "tier_distribution": tier_stats["tier_distribution"],
        # ── SAS / evidence quality ─────────────────────────────────────────
        "fallback_sas_count": fallback_stats["fallback_sas_count"],
        "missing_zone_count": fallback_stats["missing_zone_count"],
        # ── R3 refinement outcomes ─────────────────────────────────────────
        "r3_refined_to_r2_count": pipeline_metrics["r3_refined_to_r2_count"],
        "r3_downgraded_to_r4_count": pipeline_metrics["r3_downgraded_to_r4_count"],
        # ── Clinical signal means ──────────────────────────────────────────
        "mean_sdca": mean_sdca,
        "mean_aps": mean_aps,
        # ── Gate diagnostics ───────────────────────────────────────────────
        "gate_mean_confidence": gate_confidence,
        # ── Loss scalars ──────────────────────────────────────────────────
        "loss_total": float(loss_dict["loss_total"].item()),
        "loss_sdgl": float(loss_dict["loss_sdgl"].item()),
        "loss_gate": float(loss_dict["loss_gate"].item()),
        "loss_kg": float(loss_dict["loss_kg"].item()),
        # ── Config trace ──────────────────────────────────────────────────
        "gumbel_tau": pipeline_cfg.get("gumbel_tau", None),
        "training_step": pipeline_cfg.get("step", None),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main integration step
# ══════════════════════════════════════════════════════════════════════════════


def run_lgovnet_step(
    image_ids: List[str],
    detector_outputs: Dict[str, List[dict]],
    gt_boxes: Dict[str, List[dict]],
    kg_artifacts: dict,
    encoder: LGovNetEncoder,
    gate: LGG,
    sdgl_fn: SDGLLoss,
    gate_loss_fn: GateLoss,
    step: int = 0,
    cfg: Optional[dict] = None,
) -> dict:
    """Execute one LGov-Net integration step (orchestrator only).

    Connects detector outputs → KG encoder → evidence → gate → routing →
    refinement → SDGL + gate + KG loss → total loss.

    NOTE: this function does NOT call .backward() or optimizer.step().
    Calling code (training loop) is responsible for those operations.

    Args:
        image_ids:
            List of image identifier strings in the batch.
        detector_outputs:
            {image_id: [DetectorBox, ...]}
            Each DetectorBox must have: bbox_id, assigned_zone,
            ensemble_probs, head_predictions.
            May be synthetic (detector_stub) while backbone is pending.
        gt_boxes:
            {image_id: [{"bbox": [x1,y1,x2,y2], "class_label": str}, ...]}
            Ground-truth bounding boxes used by routing for tier derivation.
            May be empty dicts ({}) for R4-only batches or stub runs.
        kg_artifacts:
            Output dict from ``builder.build_graph_from_artifacts()``.
            Must contain: nodes, edges, node_features, stats.
        encoder:
            LGovNetEncoder module (training mode).
            Rows 0–3 of its output correspond to LESION_NAMES.
        gate:
            LGG (Learned Governance Gate) module.
            Must be instantiated with embedding_dim matching
            encoder.embedding_dim (default: both 64).
        sdgl_fn:
            SDGLLoss module instance.
        gate_loss_fn:
            GateLoss module instance.
        step:
            Global training step index (0-indexed).
            Used for Gumbel temperature annealing.
        cfg:
            Optional config dict overriding defaults.  Supported keys:
              gumbel_tau_start   (default 1.0)
              gumbel_tau_end     (default 0.1)
              gumbel_tau_anneal_steps (default 10_000)
              lambda_gate        (default 1.0)
              lambda_kg          (default 0.1)
              use_kg_embedding   (default True)
              routing_thresholds (default None → routing defaults)
              refine_thresholds  (default None → refine defaults)
              gate_hard          (default False)

    Returns:
        StepOutput dict:
        {
          "loss_total":           Tensor — ready for .backward()
          "loss_sdgl":            Tensor
          "loss_gate":            Tensor
          "loss_kg":              Tensor
          "tier_stats":           dict   — num_r1/r2/r3/r4
          "routing_records":      list[dict] — one per bbox
          "refinement_records":   list[dict] — one per R3 bbox
          "debug_info":           dict   — all required debug metrics
          "gate_output":          dict   — tier_logits/tier_probs/tier_hard
                                           (PATCH-15.5 P1: audit/eval path)
        }
    """
    # ── Unpack / apply config defaults ────────────────────────────────────────
    cfg = cfg or {}
    tau_start: float = float(cfg.get("gumbel_tau_start", 1.0))
    tau_end: float = float(cfg.get("gumbel_tau_end", 0.1))
    tau_anneal_steps: int = int(cfg.get("gumbel_tau_anneal_steps", 10_000))
    lambda_gate: float = float(cfg.get("lambda_gate", 1.0))
    lambda_kg: float = float(cfg.get("lambda_kg", 0.1))
    use_kg_emb: bool = bool(cfg.get("use_kg_embedding", True))
    routing_thr: Optional[dict] = cfg.get("routing_thresholds", None)
    refine_thr: Optional[dict] = cfg.get("refine_thresholds", None)
    gate_hard: bool = bool(cfg.get("gate_hard", False))

    device = next(encoder.parameters()).device

    # ── Gumbel temperature for this step ─────────────────────────────────────
    tau = get_gumbel_tau(
        step=step,
        tau_start=tau_start,
        tau_end=tau_end,
        tau_anneal_steps=tau_anneal_steps,
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Step 1: KG encoder forward pass
    # Produces node embeddings [N=9, d], SDM (L2 distances), Compat scores.
    # ──────────────────────────────────────────────────────────────────────────
    enc_inputs = build_encoder_inputs(kg_artifacts, device=device)
    # Shape checks
    assert enc_inputs["x"].shape[1] == encoder.node_feat_dim, (
        f"encoder input dim mismatch: "
        f"x.shape[1]={enc_inputs['x'].shape[1]} vs "
        f"encoder.node_feat_dim={encoder.node_feat_dim}"
    )

    node_embeddings: torch.Tensor = encoder(
        enc_inputs["x"],
        enc_inputs["edge_index"],
        enc_inputs["edge_type_ids"],
        enc_inputs["edge_weights"],
    )  # [N, d]

    assert node_embeddings.shape == (
        len(enc_inputs["node_names"]),
        encoder.embedding_dim,
    ), f"node_embeddings shape unexpected: {node_embeddings.shape}"

    # SDM (raw pairwise distances) from current embeddings
    sdm_raw: dict = compute_sdm(
        node_embeddings,
        enc_inputs["node_names"],
    )
    # SDM_norm: normalized for [0,1] input to evidence / gate
    sdm_norm: dict = normalize_sdca(sdm_raw, method="log_scaled")

    # Compat scores: {(lesion, zone): sigmoid float}
    compat_scores: dict = compute_compat(
        node_embeddings,
        encoder.W_compat,
        enc_inputs["node_names"],
        compat_prior=encoder.compat_prior,
    )

    # Lesion embeddings [C=4, d] — rows 0–3 in LESION_NAMES order
    lesion_node_idxs = [enc_inputs["node_names"].index(c) for c in LESION_NAMES]
    lesion_embeddings: torch.Tensor = node_embeddings[lesion_node_idxs]  # [4, d]
    assert lesion_embeddings.shape == (
        4,
        encoder.embedding_dim,
    ), f"lesion_embeddings shape unexpected: {lesion_embeddings.shape}"

    # ──────────────────────────────────────────────────────────────────────────
    # Step 2: Flatten detector outputs to a single batch list
    # Also build placeholder PSS results (STUB S3)
    # ──────────────────────────────────────────────────────────────────────────
    all_boxes: List[dict] = []
    all_gt: List[dict] = []

    for img_id in image_ids:
        boxes = detector_outputs.get(img_id, [])
        gts = gt_boxes.get(img_id, [])
        all_boxes.extend(boxes)
        all_gt.extend(gts)  # Shared GT pool across images for routing stub

    if len(all_boxes) == 0:
        raise ValueError("run_lgovnet_step received an empty bbox batch.")

    B = len(all_boxes)

    # Placeholder PSS for every bbox (STUB S3)
    pss_results: dict = {
        box["bbox_id"]: make_placeholder_pss(box["bbox_id"]) for box in all_boxes
    }

    # ──────────────────────────────────────────────────────────────────────────
    # Step 3: Evidence computation
    # g(b_j) = [SAS, SDCA_norm, APS, PSS] per bbox
    # ──────────────────────────────────────────────────────────────────────────
    evidence_batch: List[dict] = build_evidence_batch(
        boxes=all_boxes,
        pss_results=pss_results,
        sdm_norm=sdm_norm,
        compat_scores=compat_scores,
    )

    assert (
        len(evidence_batch) == B
    ), f"evidence_batch length mismatch: {len(evidence_batch)} vs {B}"

    # Build evidence tensor [B, 4]
    ev_vectors: torch.Tensor = torch.tensor(
        [ev["evidence_vector"] for ev in evidence_batch],
        dtype=torch.float32,
        device=device,
    )  # [B, 4]
    assert ev_vectors.shape == (
        B,
        4,
    ), f"ev_vectors shape unexpected: {ev_vectors.shape}"

    # ──────────────────────────────────────────────────────────────────────────
    # Step 4: Gate input assembly
    # e_pred = Σ_c P(c|b_j) · e_c  →  concat [evidence_vec | e_pred]
    # ──────────────────────────────────────────────────────────────────────────
    # Ensemble probs tensor [B, C=4] in LESION_NAMES order
    ens_probs_t: torch.Tensor = torch.tensor(
        [
            [float(box["ensemble_probs"].get(c, 0.0)) for c in LESION_NAMES]
            for box in all_boxes
        ],
        dtype=torch.float32,
        device=device,
    )  # [B, 4]

    e_pred: torch.Tensor = build_pred_embedding(
        ens_probs_t, lesion_embeddings
    )  # [B, d]
    assert e_pred.shape == (
        B,
        encoder.embedding_dim,
    ), f"e_pred shape unexpected: {e_pred.shape}"

    gate_input: torch.Tensor = concat_gate_input(
        ev_vectors, e_pred, use_kg_embedding=use_kg_emb
    )  # [B, 4+d] or [B, 4]

    # ──────────────────────────────────────────────────────────────────────────
    # Step 5: Governance Gate forward (Gumbel during training)
    # ──────────────────────────────────────────────────────────────────────────
    gate_output: dict = gate.gumbel_forward(
        gate_input,
        temperature=tau,
        hard=gate_hard,
    )
    # gate_output["tier_logits"] : [B, 4]
    # gate_output["tier_probs"]  : [B, 4]
    # gate_output["tier_hard"]   : [B] int64

    tier_logits: torch.Tensor = gate_output["tier_logits"]
    assert tier_logits.shape == (
        B,
        4,
    ), f"tier_logits shape unexpected: {tier_logits.shape}"

    # Detector log-probs for SDGL loss — use ensemble_probs as class logits
    # (STUB: real backbone should supply log-probabilities from its classifier)
    log_probs_batch: torch.Tensor = F.log_softmax(ens_probs_t, dim=-1)  # [B, 4]

    # ──────────────────────────────────────────────────────────────────────────
    # Step 6: Routing — derive tier pseudo-labels from GT + evidence
    # These are supervision targets (NOT from gate output).
    # ──────────────────────────────────────────────────────────────────────────
    routing_batch: List[dict] = route_bbox_batch(
        boxes=all_boxes,
        gt_boxes=all_gt,
        sdm_norm=sdm_norm,
        compat_scores=compat_scores,
        thresholds=routing_thr,
    )

    assert (
        len(routing_batch) == B
    ), f"routing_batch length mismatch: {len(routing_batch)} vs {B}"

    # ──────────────────────────────────────────────────────────────────────────
    # Step 7: R3 refinement — only on R3-routed bboxes
    # ──────────────────────────────────────────────────────────────────────────
    r3_indices: List[int] = [
        i for i, r in enumerate(routing_batch) if r["target_tier"] == "R3"
    ]
    r3_boxes: List[dict] = [all_boxes[i] for i in r3_indices]

    refinement_records: List[dict] = (
        refine_r3_batch(
            boxes=r3_boxes,
            sdm_norm=sdm_norm,
            compat_scores=compat_scores,
            thresholds=refine_thr,
        )
        if r3_boxes
        else []
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Step 8: Assemble SDGL loss inputs from routing records
    # ──────────────────────────────────────────────────────────────────────────
    tier_targets: List[int] = [_TIER_INDEX[r["target_tier"]] for r in routing_batch]
    # Target class index: use majority_class from routing (ensemble argmax)
    target_class_indices: List[int] = [
        (
            _LESION_CLASSES.index(r["majority_class"])
            if r["majority_class"] in _LESION_CLASSES
            else 0
        )
        for r in routing_batch
    ]
    ensemble_probs_list: List[dict] = [box["ensemble_probs"] for box in all_boxes]
    # consensus = num_heads_correct / 4; aps from evidence
    consensus_batch: List[float] = [
        float(r["num_heads_correct"]) / 4.0 for r in routing_batch
    ]
    aps_batch: List[float] = [ev["aps"] for ev in evidence_batch]

    # ──────────────────────────────────────────────────────────────────────────
    # Step 9: SDGL + L_KG loss
    # ──────────────────────────────────────────────────────────────────────────
    sdgl_out: dict = sdgl_fn(
        log_probs_batch=log_probs_batch,
        tier_targets=tier_targets,
        target_class_indices=target_class_indices,
        ensemble_probs_batch=ensemble_probs_list,
        sdm_norm=sdm_norm,
        consensus_batch=consensus_batch,
        aps_batch=aps_batch,
        lesion_embeddings=lesion_embeddings,
    )
    l_sdgl: torch.Tensor = sdgl_out["l_sdgl"]
    l_kg: torch.Tensor = sdgl_out["l_kg"]

    # ──────────────────────────────────────────────────────────────────────────
    # Step 10: Gate loss — CE(tier_logits, pseudo-labels from routing)
    # ──────────────────────────────────────────────────────────────────────────
    target_tier_indices: torch.Tensor = torch.tensor(
        tier_targets, dtype=torch.long, device=device
    )  # [B]

    gate_loss_out: dict = gate_loss_fn(tier_logits, target_tier_indices)
    l_gate: torch.Tensor = gate_loss_out["l_gate"]

    # ──────────────────────────────────────────────────────────────────────────
    # Step 11: Total loss
    # L_total = L_SDGL + λ_gate·L_gate + λ_KG·L_KG
    # ──────────────────────────────────────────────────────────────────────────
    loss_total: torch.Tensor = compute_total_loss(
        l_sdgl=l_sdgl,
        l_gate=l_gate,
        l_kg=l_kg,
        lambda_gate=lambda_gate,
        lambda_kg=lambda_kg,
        l_eff=None,  # hook — not yet active
        lambda_eff=0.0,
        l_aux=None,  # hook — not yet active
        lambda_aux=0.0,
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Step 12: Package loss dict + debug info
    # ──────────────────────────────────────────────────────────────────────────
    loss_dict = {
        "loss_total": loss_total,
        "loss_sdgl": l_sdgl,
        "loss_gate": l_gate,
        "loss_kg": l_kg,
    }

    tier_stats = summarize_tier_stats(routing_batch)

    debug_info = build_step_debug_info(
        evidence_batch=evidence_batch,
        routing_batch=routing_batch,
        refinement_records=refinement_records,
        gate_output=gate_output,
        loss_dict=loss_dict,
        pipeline_cfg={
            "gumbel_tau": tau,
            "step": step,
            "lambda_gate": lambda_gate,
            "lambda_kg": lambda_kg,
            "use_kg_embedding": use_kg_emb,
        },
    )

    return {
        "loss_total": loss_total,
        "loss_sdgl": l_sdgl,
        "loss_gate": l_gate,
        "loss_kg": l_kg,
        "tier_stats": tier_stats,
        "routing_records": routing_batch,
        "refinement_records": refinement_records,
        "debug_info": debug_info,
        # P1 (PATCH-15.5): expose gate output for downstream auditability.
        # Contains: tier_logits [B,4], tier_probs [B,4], tier_hard [B].
        # Consumers (eval_step, certificate builder) call .tolist() on tensors.
        "gate_output": gate_output,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Smoke test / __main__ demo
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """Minimal smoke test — 1 synthetic batch, no real images or GT required.

    Usage:
        python src/train_step.py

    Expected output:
        [PHASE14 SMOKE] loss_total: <float>
        [PHASE14 SMOKE] tier_distribution: {...}
        [PHASE14 SMOKE] fallback_sas_count: <int>
        [PHASE14 SMOKE] backward OK — gradients flow
        [PHASE14 SMOKE] PASS
    """
    import traceback

    print("=" * 60)
    print("PHASE 14 — LGov-Net integration step smoke test")
    print("=" * 60)

    try:
        # ── Build KG graph artifacts (minimal: use builder) ────────────────
        from lgovnet.src.kg.builder import build_graph_from_artifacts
        from src.data.ontology_utils import load_ontology  # may not exist → fallback
    except ImportError:
        # builder.build_graph_from_artifacts may require ontology artifacts;
        # we supply a minimal synthetic graph structure directly.
        pass

    # ── Build a tiny synthetic KG artifact ────────────────────────────────
    # N=9 nodes: 4 lesion + 5 zone
    node_names = [
        "MA",
        "HE",
        "Hard_EX",
        "Soft_EX",
        "Foveal",
        "Parafoveal",
        "Peripapillary",
        "Vascular_Arcades",
        "Peripheral",
    ]
    node_features: dict = {}
    for i, nm in enumerate(node_names):
        feat_dim = 17 if i < 4 else 9  # lesion: 17, zone: 9
        feat = [0.0] * feat_dim
        feat[i % feat_dim] = 1.0  # crude one-hot-like
        node_features[nm] = feat

    nodes = [{"name": nm, "idx": i} for i, nm in enumerate(node_names)]

    # Minimal edge list: a few morphological_confusability edges (undirected)
    _edge_pairs = [(0, 1), (0, 2), (1, 2), (2, 3), (0, 3), (1, 3)]
    edges = {
        "morphological_confusability": [
            {"src": s, "dst": d, "weight": 0.5, "directed": False}
            for s, d in _edge_pairs
        ],
        "severity_transition": [
            {"src": 0, "dst": 1, "weight": 0.1, "directed": False},
            {"src": 0, "dst": 3, "weight": 2.0, "directed": False},
        ],
        "spatial_distribution": [
            {"src": 0, "dst": 4, "weight": 0.3, "directed": False},
            {"src": 2, "dst": 4, "weight": 0.8, "directed": False},
        ],
        "pathophysiological_mechanism": [
            {"src": 0, "dst": 1, "weight": 0.4, "directed": True},
        ],
        "co_occurrence_pattern": [
            {"src": 1, "dst": 2, "weight": 0.2, "directed": False},
        ],
    }
    kg_artifacts = {
        "nodes": nodes,
        "edges": edges,
        "node_features": node_features,
        "stats": {},
    }

    # ── Instantiate models ─────────────────────────────────────────────────
    torch.manual_seed(42)
    embedding_dim = 64
    encoder = build_gat_encoder(
        node_feat_dim=17,
        hidden_dim=64,
        embedding_dim=embedding_dim,
    )
    gate = LGG(evidence_dim=4, embedding_dim=embedding_dim, hidden_dim=128)
    sdgl_fn = SDGLLoss(lambda1=1.0, lambda2=1.0, lambda3=0.5, lambda_kg=0.1)
    gate_loss_fn = GateLoss(epsilon=0.05)

    encoder.train()
    gate.train()

    # ── Synthetic batch ────────────────────────────────────────────────────
    image_ids = ["img_0001", "img_0002"]
    detector_outputs = detector_stub(image_ids, bboxes_per_image=4, seed=7)

    # Minimal GT (one per image)
    gt_boxes = {
        "img_0001": [
            {"bbox": [10.0, 10.0, 60.0, 60.0], "class_label": "MA"},
        ],
        "img_0002": [
            {"bbox": [15.0, 15.0, 65.0, 65.0], "class_label": "Hard_EX"},
        ],
    }

    # ── Run the integration step ───────────────────────────────────────────
    result = run_lgovnet_step(
        image_ids=image_ids,
        detector_outputs=detector_outputs,
        gt_boxes=gt_boxes,
        kg_artifacts=kg_artifacts,
        encoder=encoder,
        gate=gate,
        sdgl_fn=sdgl_fn,
        gate_loss_fn=gate_loss_fn,
        step=0,
        cfg={
            "gumbel_tau_start": 1.0,
            "gumbel_tau_end": 0.1,
            "gumbel_tau_anneal_steps": 10_000,
            "lambda_gate": 1.0,
            "lambda_kg": 0.1,
            "use_kg_embedding": True,
        },
    )

    # ── Validate return contract ───────────────────────────────────────────
    required_keys = {
        "loss_total",
        "loss_sdgl",
        "loss_gate",
        "loss_kg",
        "tier_stats",
        "routing_records",
        "refinement_records",
        "debug_info",
    }
    missing = required_keys - set(result.keys())
    assert not missing, f"StepOutput missing keys: {missing}"

    loss_total = result["loss_total"]
    assert isinstance(loss_total, torch.Tensor), "loss_total must be a Tensor"
    assert loss_total.shape == (), f"loss_total must be scalar, got {loss_total.shape}"
    assert not torch.isnan(loss_total), "loss_total is NaN!"
    assert not torch.isinf(loss_total), "loss_total is Inf!"

    dbg = result["debug_info"]
    required_debug = {
        "fallback_sas_count",
        "tier_distribution",
        "r3_refined_to_r2_count",
        "r3_downgraded_to_r4_count",
        "num_r1",
        "num_r2",
        "num_r3",
        "num_r4",
        "mean_sdca",
        "mean_aps",
    }
    missing_debug = required_debug - set(dbg.keys())
    assert not missing_debug, f"debug_info missing keys: {missing_debug}"

    print(f"[PHASE14 SMOKE] loss_total        : {loss_total.item():.6f}")
    print(f"[PHASE14 SMOKE] loss_sdgl         : {result['loss_sdgl'].item():.6f}")
    print(f"[PHASE14 SMOKE] loss_gate         : {result['loss_gate'].item():.6f}")
    print(f"[PHASE14 SMOKE] loss_kg           : {result['loss_kg'].item():.6f}")
    print(f"[PHASE14 SMOKE] tier_distribution : {dbg['tier_distribution']}")
    print(
        f"[PHASE14 SMOKE] num R1/R2/R3/R4   : "
        f"{dbg['num_r1']}/{dbg['num_r2']}/{dbg['num_r3']}/{dbg['num_r4']}"
    )
    print(f"[PHASE14 SMOKE] fallback_sas_count: {dbg['fallback_sas_count']}")
    print(f"[PHASE14 SMOKE] r3→R2             : {dbg['r3_refined_to_r2_count']}")
    print(f"[PHASE14 SMOKE] r3→R4             : {dbg['r3_downgraded_to_r4_count']}")
    print(f"[PHASE14 SMOKE] mean_sdca         : {dbg['mean_sdca']:.4f}")
    print(f"[PHASE14 SMOKE] mean_aps          : {dbg['mean_aps']:.4f}")
    print(f"[PHASE14 SMOKE] gumbel_tau        : {dbg['gumbel_tau']:.4f}")

    # ── Backward pass check ───────────────────────────────────────────────
    # loss_total must be a finite scalar with a grad_fn; at least one param
    # in encoder or gate must receive a gradient.  Not every parameter is
    # guaranteed to receive a gradient: if all bboxes are routed to R4 the
    # SDGL classification loss is zero (no contribution), so some encoder
    # params may have no gradient on this specific dummy batch.
    assert loss_total.requires_grad, "loss_total has no grad_fn"
    loss_total.backward()
    params_with_grad = [
        p
        for p in list(encoder.parameters()) + list(gate.parameters())
        if p.requires_grad and p.grad is not None
    ]
    assert len(params_with_grad) > 0, (
        "No parameters received any gradient after backward(). "
        "Check that loss_total has a non-trivial computation graph."
    )
    print(
        f"[PHASE14 SMOKE] backward OK — {len(params_with_grad)} params have gradients"
    )

    print("[PHASE14 SMOKE] PASS")
    print("=" * 60)
