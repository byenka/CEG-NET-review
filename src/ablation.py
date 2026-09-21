"""
ablation.py
Ablation pack for LGov-Net — Phase 16.

Provides a minimal, runnable, and audit-friendly ablation infrastructure for
testing Contributions C1, C2, and C3 in isolation.

Design principles:
  - Re-use existing modules; do NOT copy or re-implement any module logic.
  - Each ablation is a named configuration/switch, not a new trainer.
  - Structural overrides (changes that cannot be expressed as cfg keys) are
    implemented as injection points inside _run_lgovnet_step_structural().
    That function calls the SAME module functions as run_lgovnet_step, but
    passes different intermediate values at clearly labelled override points.
  - Consolidated and unsupported ablations are returned with status="SKIPPED".
  - Honest caveats are propagated from ablation.yaml to results.

Public API:
  load_ablation_cases(yaml_path)              -> List[dict]
  apply_ablation_overrides(base_cfg, case)    -> dict
  run_single_ablation(case, ...)              -> dict
  run_ablation_suite(cases, ...)              -> List[dict]
  summarize_ablation_results(results)         -> dict

Structural override codes (see ablation.yaml):
  "none"                   pure cfg override, calls run_lgovnet_step directly
  "without_aps"            compat_scores={} before evidence/routing
  "without_sdca"           sdm_norm={} before evidence/routing/SDGL
  "without_gumbel"         gate.forward() instead of gate.gumbel_forward()
  "without_refinement"     skip refine_r3_batch()
  "baseline_no_governance" force all routing tier = R1
  "without_r3_weighting"   uniform consensus=1.0, aps=1.0 for R3 SDGL weights

Dependencies:
  src/train_step.py          (run_lgovnet_step — used for cfg-based ablations)
  src/kg/encoder.py          (build_encoder_inputs, compute_sdm, etc.)
  src/governance/evidence.py (build_evidence_batch)
  src/governance/gate.py     (LGG, build_pred_embedding, concat_gate_input,
                              get_gumbel_tau)
  src/governance/routing.py  (route_bbox_batch)
  src/governance/refine.py   (refine_r3_batch)
  src/governance/contracts.py(make_placeholder_pss, LESION_CLASSES)
  src/losses/sdgl.py         (SDGLLoss, compute_total_loss)
  src/losses/gate_loss.py    (GateLoss)
  src/utils/pipeline_metrics.py (compute_pipeline_metrics)
"""

from __future__ import annotations

import os
import sys
import traceback
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# ── Project root insertion ─────────────────────────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.join(_here, "..")
if _root not in sys.path:
    sys.path.insert(0, _root)

# yaml with fallback to a minimal parser
try:
    import yaml as _yaml

    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

# ── Module imports (same as train_step.py, never duplicated) ──────────────────
from lgovnet.src.kg.encoder import (
    LGovNetEncoder,
    build_encoder_inputs,
    compute_sdm,
    compute_compat,
    normalize_sdca,
    LESION_NAMES,
    ZONE_NAMES,
)
from lgovnet.src.governance.evidence import build_evidence_batch
from lgovnet.src.governance.gate import (
    LGG,
    build_pred_embedding,
    concat_gate_input,
    get_gumbel_tau,
    TIER_NAMES,
)
from lgovnet.src.governance.routing import route_bbox_batch
from lgovnet.src.governance.refine import refine_r3_batch
from lgovnet.src.governance.contracts import make_placeholder_pss, LESION_CLASSES
from lgovnet.src.losses.sdgl import SDGLLoss, compute_total_loss
from lgovnet.src.losses.gate_loss import GateLoss
from lgovnet.src.utils.pipeline_metrics import compute_pipeline_metrics
from lgovnet.src.train_step import (
    run_lgovnet_step,
    summarize_tier_stats,
    summarize_fallback_usage,
    build_step_debug_info,
)

# ── Constants ─────────────────────────────────────────────────────────────────
_TIER_INDEX: Dict[str, int] = {t: i for i, t in enumerate(TIER_NAMES)}
_LESION_CLASSES: List[str] = list(LESION_CLASSES)
_ZONES: List[str] = list(ZONE_NAMES)

#: Neutral APS value (0.5 = no anatomical penalty or bonus).
_NEUTRAL_APS: float = 0.5
#: Neutral SDCA fallback (0.0 = no KG class-separation signal).
_NEUTRAL_SDCA: float = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# YAML loader
# ══════════════════════════════════════════════════════════════════════════════


def load_ablation_cases(yaml_path: str) -> List[dict]:
    """Load ablation case definitions from configs/ablation.yaml.

    Args:
        yaml_path: Absolute or relative path to ablation.yaml.

    Returns:
        List of ablation case dicts.  Each dict has at minimum:
          name, contribution, description, cfg_overrides,
          structural_override, supported.

    Raises:
        FileNotFoundError: if yaml_path does not exist.
        ImportError: if PyYAML is not installed.
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"ablation.yaml not found: {yaml_path}")
    if not _YAML_AVAILABLE:
        raise ImportError(
            "PyYAML is required for load_ablation_cases(). "
            "Install via: pip install pyyaml"
        )
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = _yaml.safe_load(f)
    cases: List[dict] = data.get("ablation_cases", [])
    return cases


# ══════════════════════════════════════════════════════════════════════════════
# Config override helper
# ══════════════════════════════════════════════════════════════════════════════


def apply_ablation_overrides(base_cfg: dict, case: dict) -> dict:
    """Return a new cfg dict with ablation overrides applied.

    Does not modify base_cfg.  Merges cfg_overrides on top of base_cfg.

    Args:
        base_cfg:  Base integration step config (e.g. from train.yaml).
        case:      Ablation case dict from load_ablation_cases().

    Returns:
        New dict: copy of base_cfg with case["cfg_overrides"] merged in.
    """
    merged = deepcopy(base_cfg)
    overrides: dict = case.get("cfg_overrides", {}) or {}
    merged.update(overrides)
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# Structural ablation orchestrator
# ══════════════════════════════════════════════════════════════════════════════


def _run_lgovnet_step_structural(
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
    structural_override: str = "none",
) -> dict:
    """Like run_lgovnet_step but with structural ablation injection points.

    Calls the same module functions as run_lgovnet_step in the same 12-step
    order.  Each injection point is labelled:
       # ABLATION OVERRIDE: <code> — <rationale>

    Supported structural_override codes:
      "without_aps"            — compat_scores neutralised before evidence/routing
      "without_sdca"           — sdm_norm neutralised before evidence/routing
      "without_gumbel"         — gate.forward() instead of gate.gumbel_forward()
      "without_refinement"     — refinement_records forced to []
      "baseline_no_governance" — all routing tiers forced to R1
      "without_r3_weighting"   — uniform consensus=1.0, aps=1.0 in SDGL

    Args:
        (same as run_lgovnet_step)
        structural_override: Ablation override code string.

    Returns:
        Same StepOutput dict schema as run_lgovnet_step()
        (including "gate_output" per PATCH-15.5).
    """
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

    tau = get_gumbel_tau(
        step=step,
        tau_start=tau_start,
        tau_end=tau_end,
        tau_anneal_steps=tau_anneal_steps,
    )

    # ── Step 1: KG encoder ────────────────────────────────────────────────────
    enc_inputs = build_encoder_inputs(kg_artifacts, device=device)
    node_embeddings: torch.Tensor = encoder(
        enc_inputs["x"],
        enc_inputs["edge_index"],
        enc_inputs["edge_type_ids"],
        enc_inputs["edge_weights"],
    )
    sdm_raw: dict = compute_sdm(node_embeddings, enc_inputs["node_names"])
    sdm_norm: dict = normalize_sdca(sdm_raw, method="log_scaled")
    compat_scores: dict = compute_compat(
        node_embeddings,
        encoder.W_compat,
        enc_inputs["node_names"],
        compat_prior=encoder.compat_prior,
    )
    lesion_node_idxs = [enc_inputs["node_names"].index(c) for c in LESION_NAMES]
    lesion_embeddings: torch.Tensor = node_embeddings[lesion_node_idxs]

    # ABLATION OVERRIDE: without_aps — neutralise compat_scores.
    # All APS lookups via compute_aps(compat_scores={}) return default 0.5.
    if structural_override == "without_aps":
        compat_scores = {}

    # ABLATION OVERRIDE: without_sdca — neutralise sdm_norm.
    # All SDCA lookups return _SDCA_FALLBACK=0.0.
    if structural_override == "without_sdca":
        sdm_norm = {}

    # ── Step 2: Flatten detector outputs; build placeholder PSS ───────────────
    all_boxes: List[dict] = []
    all_gt: List[dict] = []
    for img_id in image_ids:
        all_boxes.extend(detector_outputs.get(img_id, []))
        all_gt.extend(gt_boxes.get(img_id, []))

    if not all_boxes:
        raise ValueError("_run_lgovnet_step_structural: empty bbox batch.")

    B = len(all_boxes)
    pss_results: dict = {
        box["bbox_id"]: make_placeholder_pss(box["bbox_id"]) for box in all_boxes
    }

    # ── Step 3: Evidence computation ──────────────────────────────────────────
    evidence_batch: List[dict] = build_evidence_batch(
        boxes=all_boxes,
        pss_results=pss_results,
        sdm_norm=sdm_norm,
        compat_scores=compat_scores,
    )
    ev_vectors: torch.Tensor = torch.tensor(
        [ev["evidence_vector"] for ev in evidence_batch],
        dtype=torch.float32,
        device=device,
    )

    # ── Step 4: Gate input ────────────────────────────────────────────────────
    ens_probs_t: torch.Tensor = torch.tensor(
        [
            [float(box["ensemble_probs"].get(c, 0.0)) for c in LESION_NAMES]
            for box in all_boxes
        ],
        dtype=torch.float32,
        device=device,
    )
    e_pred: torch.Tensor = build_pred_embedding(ens_probs_t, lesion_embeddings)
    gate_input: torch.Tensor = concat_gate_input(
        ev_vectors, e_pred, use_kg_embedding=use_kg_emb
    )

    # ── Step 5: Gate forward ──────────────────────────────────────────────────
    # ABLATION OVERRIDE: without_gumbel — use plain softmax forward.
    # Loss computation is identical; only gradient path through tier assignment differs.
    if structural_override == "without_gumbel":
        gate_output: dict = gate.forward(gate_input)
    else:
        gate_output = gate.gumbel_forward(gate_input, temperature=tau, hard=gate_hard)

    tier_logits: torch.Tensor = gate_output["tier_logits"]
    log_probs_batch: torch.Tensor = F.log_softmax(ens_probs_t, dim=-1)

    # ── Step 6: Routing ───────────────────────────────────────────────────────
    routing_batch: List[dict] = route_bbox_batch(
        boxes=all_boxes,
        gt_boxes=all_gt,
        sdm_norm=sdm_norm,
        compat_scores=compat_scores,
        thresholds=routing_thr,
    )

    # ABLATION OVERRIDE: baseline_no_governance — force all tiers to R1.
    # Equivalent to a vanilla detector with no governance mechanism.
    if structural_override == "baseline_no_governance":
        routing_batch = [
            {
                **r,
                "target_tier": "R1",
                "target_tier_index": 0,
                "reason": "ABLATION:baseline_no_governance (all bboxes forced to R1)",
            }
            for r in routing_batch
        ]

    # ── Step 7: R3 refinement ─────────────────────────────────────────────────
    # ABLATION OVERRIDE: without_refinement — skip refine_r3_batch entirely.
    # NOTE: refinement currently only affects certificates/debug_info, not
    # SDGL loss computation. Loss values will be identical to baseline_full.
    if structural_override == "without_refinement":
        refinement_records: List[dict] = []
    else:
        r3_indices = [
            i for i, r in enumerate(routing_batch) if r["target_tier"] == "R3"
        ]
        r3_boxes = [all_boxes[i] for i in r3_indices]
        refinement_records = (
            refine_r3_batch(
                boxes=r3_boxes,
                sdm_norm=sdm_norm,
                compat_scores=compat_scores,
                thresholds=refine_thr,
            )
            if r3_boxes
            else []
        )

    # ── Step 8: Assemble SDGL inputs ─────────────────────────────────────────
    tier_targets: List[int] = [_TIER_INDEX[r["target_tier"]] for r in routing_batch]
    target_class_indices: List[int] = [
        (
            _LESION_CLASSES.index(r["majority_class"])
            if r.get("majority_class") in _LESION_CLASSES
            else 0
        )
        for r in routing_batch
    ]
    ensemble_probs_list: List[dict] = [box["ensemble_probs"] for box in all_boxes]
    consensus_batch: List[float] = [
        float(r["num_heads_correct"]) / 4.0 for r in routing_batch
    ]
    aps_batch: List[float] = [ev["aps"] for ev in evidence_batch]

    # ABLATION OVERRIDE: without_r3_weighting — uniform w_j = 1.0 for all R3 boxes.
    # consensus × APS weighting is disabled: every R3 box contributes equally.
    if structural_override == "without_r3_weighting":
        consensus_batch = [1.0] * len(consensus_batch)
        aps_batch = [1.0] * len(aps_batch)

    # ── Step 9: SDGL + L_KG loss ─────────────────────────────────────────────
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

    # ── Step 10: Gate loss ────────────────────────────────────────────────────
    target_tier_indices: torch.Tensor = torch.tensor(
        tier_targets, dtype=torch.long, device=device
    )
    gate_loss_out: dict = gate_loss_fn(tier_logits, target_tier_indices)
    l_gate: torch.Tensor = gate_loss_out["l_gate"]

    # ── Step 11: Total loss ───────────────────────────────────────────────────
    loss_total: torch.Tensor = compute_total_loss(
        l_sdgl=l_sdgl,
        l_gate=l_gate,
        l_kg=l_kg,
        lambda_gate=lambda_gate,
        lambda_kg=lambda_kg,
        l_eff=None,
        lambda_eff=0.0,
        l_aux=None,
        lambda_aux=0.0,
    )

    # ── Step 12: Debug info + return ─────────────────────────────────────────
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
        "gate_output": gate_output,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Single ablation runner
# ══════════════════════════════════════════════════════════════════════════════


def run_single_ablation(
    case: dict,
    image_ids: List[str],
    detector_outputs: Dict[str, List[dict]],
    gt_boxes: Dict[str, List[dict]],
    kg_artifacts: dict,
    encoder: LGovNetEncoder,
    gate: LGG,
    sdgl_fn: SDGLLoss,
    gate_loss_fn: GateLoss,
    base_cfg: Optional[dict] = None,
    step: int = 0,
) -> dict:
    """Run one ablation case and return its standardised result dict.

    Handles both cfg-based and structural ablations.  For cases not supported
    (supported=false), returns immediately with status="SKIPPED".

    Special per-ablation model variants:
      without_kg_embedding:     new LGG(embedding_dim=0) instantiated here.
      without_sdm_soft_targets: new SDGLLoss(eta=0.0) instantiated here.

    Args:
        case:             One dict from load_ablation_cases().
        image_ids:        List of image IDs in the demo/smoke batch.
        detector_outputs: {image_id: [DetectorBox, ...]} (may be synthetic).
        gt_boxes:         {image_id: [{bbox, class_label}, ...]}
        kg_artifacts:     Output of build_graph_from_artifacts() or synthetic.
        encoder:          LGovNetEncoder (eval mode for smoke tests).
        gate:             LGG built for the default configuration.
        sdgl_fn:          SDGLLoss instance.
        gate_loss_fn:     GateLoss instance.
        base_cfg:         Base cfg dict (keys from train.yaml).
        step:             Global training step index.

    Returns:
        Ablation result dict with keys:
          ablation_name, contribution, status, config_overrides,
          structural_override, caveat (if any),
          loss_total, loss_sdgl, loss_gate, loss_kg,
          tier_distribution, fallback_sas_count,
          mean_sdca, mean_aps, num_certificates,
          error (if status=="ERROR")
    """
    abl_name: str = case.get("name", "unnamed")
    contribution: Optional[str] = case.get("contribution")
    supported: bool = bool(case.get("supported", True))
    structural_override: str = case.get("structural_override", "none")
    cfg_overrides: dict = case.get("cfg_overrides", {}) or {}
    caveat: Optional[str] = case.get("caveat")

    # ── Skipped cases ─────────────────────────────────────────────────────────
    if not supported:
        return {
            "ablation_name": abl_name,
            "contribution": contribution,
            "status": "SKIPPED",
            "reason": case.get("note", "marked supported=false in ablation.yaml"),
            "config_overrides": cfg_overrides,
            "structural_override": structural_override,
            "loss_total": None,
            "loss_sdgl": None,
            "loss_gate": None,
            "loss_kg": None,
            "tier_distribution": None,
            "fallback_sas_count": None,
            "mean_sdca": None,
            "mean_aps": None,
            "num_certificates": None,
        }

    # ── Merge config ──────────────────────────────────────────────────────────
    cfg: dict = apply_ablation_overrides(base_cfg or {}, case)

    # ── Per-ablation model variants ───────────────────────────────────────────
    # without_kg_embedding: gate must have embedding_dim=0 (no e_pred input).
    # We instantiate a fresh gate with same hidden_dim but embedding_dim=0.
    # This represents a re-initialised model — expected for ablation research.
    run_gate = gate
    if abl_name == "without_kg_embedding":
        run_gate = LGG(
            evidence_dim=4,
            embedding_dim=0,
            hidden_dim=gate.fc1.out_features,  # preserve hidden_dim
        )
        run_gate.to(next(gate.parameters()).device)
        run_gate.eval()
        cfg["use_kg_embedding"] = False  # ensure concat_gate_input drops e_pred

    # without_sdm_soft_targets: SDGLLoss with eta=0.0.
    # cfg["sdgl_eta"]=0.0 is set in ablation.yaml but SDGLLoss.eta is a
    # constructor param, not a cfg key in run_lgovnet_step.
    run_sdgl = sdgl_fn
    if abl_name == "without_sdm_soft_targets":
        run_sdgl = SDGLLoss(
            lambda1=sdgl_fn.lambda1,
            lambda2=sdgl_fn.lambda2,
            lambda3=sdgl_fn.lambda3,
            lambda_kg=sdgl_fn.lambda_kg,
            eta=0.0,
        )

    # ── Execute step ──────────────────────────────────────────────────────────
    try:
        if structural_override == "none":
            # Config-based ablation — delegate directly to run_lgovnet_step.
            step_output = run_lgovnet_step(
                image_ids=image_ids,
                detector_outputs=detector_outputs,
                gt_boxes=gt_boxes,
                kg_artifacts=kg_artifacts,
                encoder=encoder,
                gate=run_gate,
                sdgl_fn=run_sdgl,
                gate_loss_fn=gate_loss_fn,
                step=step,
                cfg=cfg,
            )
        else:
            # Structural ablation — use injection-point orchestrator.
            step_output = _run_lgovnet_step_structural(
                image_ids=image_ids,
                detector_outputs=detector_outputs,
                gt_boxes=gt_boxes,
                kg_artifacts=kg_artifacts,
                encoder=encoder,
                gate=run_gate,
                sdgl_fn=run_sdgl,
                gate_loss_fn=gate_loss_fn,
                step=step,
                cfg=cfg,
                structural_override=structural_override,
            )
    except Exception as exc:
        return {
            "ablation_name": abl_name,
            "contribution": contribution,
            "status": "ERROR",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "config_overrides": cfg_overrides,
            "structural_override": structural_override,
            "loss_total": None,
            "loss_sdgl": None,
            "loss_gate": None,
            "loss_kg": None,
            "tier_distribution": None,
            "fallback_sas_count": None,
            "mean_sdca": None,
            "mean_aps": None,
            "num_certificates": None,
        }

    # ── Extract required output fields ────────────────────────────────────────
    dbg: dict = step_output.get("debug_info", {}) or {}
    tier_dist_raw: dict = step_output["tier_stats"].get("tier_distribution", {})
    # Convert integer keys to string tier names for readable output
    tier_dist = {
        TIER_NAMES[k]: v for k, v in tier_dist_raw.items() if isinstance(k, int)
    }

    result = {
        "ablation_name": abl_name,
        "contribution": contribution,
        "status": "OK",
        "config_overrides": cfg_overrides,
        "structural_override": structural_override,
        # Loss scalars
        "loss_total": round(float(step_output["loss_total"].item()), 6),
        "loss_sdgl": round(float(step_output["loss_sdgl"].item()), 6),
        "loss_gate": round(float(step_output["loss_gate"].item()), 6),
        "loss_kg": round(float(step_output["loss_kg"].item()), 6),
        # Governance metrics
        "tier_distribution": tier_dist,
        "fallback_sas_count": dbg.get("fallback_sas_count", 0),
        "mean_sdca": round(float(dbg.get("mean_sdca", 0.0)), 6),
        "mean_aps": round(float(dbg.get("mean_aps", 0.0)), 6),
        # Certificates count (same as total bboxes in this integration)
        "num_certificates": step_output["tier_stats"].get("total_bboxes", 0),
    }

    if caveat:
        result["caveat"] = caveat.strip()

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Suite runner
# ══════════════════════════════════════════════════════════════════════════════


def run_ablation_suite(
    cases: List[dict],
    image_ids: List[str],
    detector_outputs: Dict[str, List[dict]],
    gt_boxes: Dict[str, List[dict]],
    kg_artifacts: dict,
    encoder: LGovNetEncoder,
    gate: LGG,
    sdgl_fn: SDGLLoss,
    gate_loss_fn: GateLoss,
    base_cfg: Optional[dict] = None,
    step: int = 0,
    verbose: bool = True,
) -> List[dict]:
    """Run all ablation cases in sequence and return aggregated results.

    Args:
        cases:         List from load_ablation_cases().
        (other args):  Same batch + model args as run_single_ablation().
        verbose:       Print progress to stdout.

    Returns:
        List of result dicts, one per case (including SKIPPED cases).
    """
    results: List[dict] = []
    total = len(cases)
    for idx, case in enumerate(cases, 1):
        name = case.get("name", "unnamed")
        contrib = case.get("contribution") or "—"
        if verbose:
            supported_flag = "SKIP" if not case.get("supported", True) else "RUN "
            print(f"  [{idx:02d}/{total}] {supported_flag} [{contrib:>4}] {name}")

        result = run_single_ablation(
            case=case,
            image_ids=image_ids,
            detector_outputs=detector_outputs,
            gt_boxes=gt_boxes,
            kg_artifacts=kg_artifacts,
            encoder=encoder,
            gate=gate,
            sdgl_fn=sdgl_fn,
            gate_loss_fn=gate_loss_fn,
            base_cfg=base_cfg,
            step=step,
        )

        status = result.get("status", "?")
        if verbose:
            if status == "OK":
                lt = result.get("loss_total")
                print(f"            → status=OK  loss_total={lt:.6f}")
            elif status == "SKIPPED":
                print(f"            → SKIPPED: {result.get('reason','')[:60]}")
            else:
                print(f"            → ERROR: {result.get('error','')[:80]}")

        results.append(result)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════


def summarize_ablation_results(results: List[dict]) -> dict:
    """Compute a simple summary over all ablation results.

    Returns:
        {
          "total_cases":   int,
          "ok_count":      int,
          "skipped_count": int,
          "error_count":   int,
          "by_contribution": {
              "C1": [loss_total, ...],
              "C2": [...],
              "C3": [...],
              "baseline": [...],
          },
          "baseline_full_loss": float | None,
          "cases_with_caveats": [name, ...],
        }
    """
    total = len(results)
    ok_count = sum(1 for r in results if r.get("status") == "OK")
    skipped_count = sum(1 for r in results if r.get("status") == "SKIPPED")
    error_count = sum(1 for r in results if r.get("status") == "ERROR")

    by_contrib: Dict[str, list] = {"C1": [], "C2": [], "C3": [], "baseline": []}
    baseline_full_loss: Optional[float] = None
    caveats: List[str] = []

    for r in results:
        if r.get("status") != "OK":
            continue
        c = r.get("contribution") or "baseline"
        if c not in by_contrib:
            by_contrib[c] = []
        lt = r.get("loss_total")
        if lt is not None:
            by_contrib[c].append({"name": r["ablation_name"], "loss_total": lt})
        if r["ablation_name"] == "baseline_full":
            baseline_full_loss = lt
        if r.get("caveat"):
            caveats.append(r["ablation_name"])

    return {
        "total_cases": total,
        "ok_count": ok_count,
        "skipped_count": skipped_count,
        "error_count": error_count,
        "by_contribution": by_contrib,
        "baseline_full_loss": baseline_full_loss,
        "cases_with_caveats": caveats,
    }
