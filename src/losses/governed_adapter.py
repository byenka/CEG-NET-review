"""
governed_adapter.py
FASE 19D.2 — Minimal adapter: MultiHeadSequentialDetector outputs → SDGL/L_gate/L_KG inputs.

This module converts the raw outputs of the multi-head sequential detector,
combined with governance records from evidence.py and routing.py, into the
tensor contract expected by:
  - SDGLLoss  (src/losses/sdgl.py)         — L_SDGL + L_KG
  - GateLoss  (src/losses/gate_loss.py)    — L_gate
  - compute_total_loss (src/losses/sdgl.py)

No training logic, no warmstart, no model changes.
All functions are pure tensor/Python — safe to call inside or outside autocast.

Output contract (GovernedBatch):
  {
    "class_logits"    : Tensor[B, C]        — mean cls logits across heads (fp32)
    "log_probs"       : Tensor[B, C]        — F.log_softmax(class_logits)
    "bbox_preds"      : Tensor[B, 4]        — mean bbox predictions across heads
    "target_classes"  : Tensor[B]  int64    — GT class index per bbox
    "target_bboxes"   : Tensor[B, 4]        — GT bbox (normalised [0,1])
    "tier_indices"    : Tensor[B]  int64    — 0=R1 1=R2 2=R3 3=R4
    "sdca"            : Tensor[B]  float32  — SDCA_norm per bbox
    "aps"             : Tensor[B]  float32  — APS per bbox
    "sas"             : Tensor[B]  float32  — SAS per bbox
    "pss"             : Tensor[B]  float32  — PSS per bbox
    "consensus"       : Tensor[B]  float32  — num_heads_correct/4
    "ensemble_probs"  : list[dict]           — original {class: float} per bbox
    "majority_classes": list[str]            — argmax of ensemble_probs per bbox
    "n_r1" : int    — number of R1 boxes in batch
    "n_r2" : int    — number of R2 boxes in batch
    "n_r3" : int    — number of R3 boxes in batch
    "n_r4" : int    — number of R4 boxes in batch
  }

Lesion embedding sub-contract for L_KG:
  extract_lesion_embeddings(node_embeddings, node_names) -> Tensor[4, d]

Gate input sub-contract for L_gate:
  build_gate_input_tensors(governed_batch, lesion_embeddings, node_names)
    -> Tensor[B, evidence_dim] and optional Tensor[B, d]

Public API:
  build_governed_batch(
      head_outputs, target_classes, target_bboxes,
      det_boxes, evidence_batch, routing_batch, device
  ) -> dict   (GovernedBatch)

  extract_lesion_embeddings(node_embeddings, node_names) -> Tensor[4, d]

  build_gate_tensors(governed_batch, lesion_embeddings, node_names)
    -> {"evidence_vectors": Tensor[B,4], "e_pred": Tensor[B,d],
        "tier_indices": Tensor[B]}

  LESION_NAMES : List[str]  — canonical order ["MA","HE","Hard_EX","Soft_EX"]
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.join(_here, "..", "..")
if _root not in sys.path:
    sys.path.insert(0, _root)

# Canonical lesion class order — must match ontology.LESION_NODES
LESION_NAMES: List[str] = ["MA", "HE", "Hard_EX", "Soft_EX"]
_LESION_IDX: Dict[str, int] = {c: i for i, c in enumerate(LESION_NAMES)}
NUM_CLASSES: int = len(LESION_NAMES)

# Tier name → index
_TIER_IDX: Dict[str, int] = {"R1": 0, "R2": 1, "R3": 2, "R4": 3}


# ══════════════════════════════════════════════════════════════════════════════
# Main adapter
# ══════════════════════════════════════════════════════════════════════════════


def build_governed_batch(
    head_outputs: List[Tuple[torch.Tensor, torch.Tensor]],
    target_classes: torch.Tensor,
    target_bboxes: torch.Tensor,
    det_boxes: List[dict],
    evidence_batch: List[dict],
    routing_batch: List[dict],
    device: Optional[torch.device] = None,
) -> dict:
    """Build the GovernedBatch tensor dict from multi-head detector outputs.

    Converts per-head outputs and governance records into the unified tensor
    contract required by SDGLLoss, GateLoss, and compute_total_loss.

    Args:
        head_outputs:   List of (cls_logits [B,C], bbox_pred [B,4]) — one tuple
                        per head. Comes from MultiHeadSequentialDetector.forward().
                        Tensors must be detached (no grad needed for adapter logic).
        target_classes: Tensor[B] int64 — GT class index per item in the batch.
        target_bboxes:  Tensor[B, 4] float — GT bbox (normalised [0,1]).
        det_boxes:      List[dict] length B — DetectorBox dicts from
                        build_detector_output_for_image() (one per batch item).
                        Must be in the same order as target_classes.
        evidence_batch: List[dict] length B — output of build_evidence_batch().
                        Each record must have "evidence_vector" [SAS,SDCA,APS,PSS].
                        If a record is None/missing, fallback 0.5 is used.
        routing_batch:  List[dict] length B — output of route_bbox_batch().
                        Each record must have "target_tier_index", "aps",
                        "sdca_norm", "num_heads_correct", "majority_class".
        device:         Target device for output tensors. Inferred from
                        head_outputs[0][0] if None.

    Returns:
        GovernedBatch dict (see module docstring).

    Notes:
        - When all bboxes are R4, class_logits and log_probs are still valid
          tensors; SDGLLoss will produce l_sdgl=0 (no gradient from SDGL).
        - Mean logits are cast to fp32 explicitly (safe under AMP).
        - B is inferred from head_outputs[0][0].shape[0].
    """
    if device is None:
        device = head_outputs[0][0].device

    B = head_outputs[0][0].shape[0]
    n_heads = len(head_outputs)

    # ── 1. Mean logits + log-probs across heads ────────────────────────────
    # Stack: [n_heads, B, C] → mean over heads → [B, C]
    cls_stack = torch.stack([ho[0].float() for ho in head_outputs], dim=0)  # [H, B, C]
    mean_cls_logits = cls_stack.mean(dim=0)  # [B, C]
    log_probs = F.log_softmax(mean_cls_logits, dim=-1)  # [B, C]

    # ── 2. Mean bbox predictions across heads ─────────────────────────────
    bbox_stack = torch.stack([ho[1].float() for ho in head_outputs], dim=0)  # [H, B, 4]
    mean_bbox_preds = bbox_stack.mean(dim=0)  # [B, 4]

    # ── 3. Per-box governance scalars from routing records ─────────────────
    tier_indices_list: List[int] = []
    sdca_list: List[float] = []
    aps_list: List[float] = []
    consensus_list: List[float] = []
    majority_classes: List[str] = []

    for i in range(B):
        rec = routing_batch[i] if i < len(routing_batch) else {}
        tier_indices_list.append(int(rec.get("target_tier_index", 3)))  # default R4
        sdca_list.append(float(rec.get("sdca_norm", 0.5)))
        aps_list.append(float(rec.get("aps", 0.5)))
        n_correct = int(rec.get("num_heads_correct", 0))
        consensus_list.append(n_correct / max(n_heads, 1))
        maj = rec.get("majority_class")
        # Fallback: use argmax of mean logits if routing didn't provide one
        if maj is None or maj not in _LESION_IDX:
            maj = LESION_NAMES[int(mean_cls_logits[i].argmax().item())]
        majority_classes.append(maj)

    # ── 4. Per-box evidence scalars [SAS, SDCA, APS, PSS] ─────────────────
    sas_list: List[float] = []
    pss_list: List[float] = []
    ensemble_probs_list: List[dict] = []

    for i in range(B):
        ev = evidence_batch[i] if i < len(evidence_batch) else {}
        ev_vec = ev.get("evidence_vector") if ev else None
        if ev_vec and len(ev_vec) >= 4:
            sas_list.append(float(ev_vec[0]))
            pss_list.append(float(ev_vec[3]))
        else:
            sas_list.append(0.5)
            pss_list.append(0.5)

        # ensemble_probs from DetectorBox
        det_box = det_boxes[i] if i < len(det_boxes) else {}
        ep = det_box.get("ensemble_probs", {})
        if not ep:
            # Build from mean logits as fallback
            probs_t = F.softmax(mean_cls_logits[i].detach(), dim=-1)
            ep = {c: float(probs_t[j]) for j, c in enumerate(LESION_NAMES)}
        ensemble_probs_list.append(ep)

    # ── 5. Assemble output tensors ─────────────────────────────────────────
    tier_t = torch.tensor(tier_indices_list, dtype=torch.int64, device=device)
    sdca_t = torch.tensor(sdca_list, dtype=torch.float32, device=device)
    aps_t = torch.tensor(aps_list, dtype=torch.float32, device=device)
    sas_t = torch.tensor(sas_list, dtype=torch.float32, device=device)
    pss_t = torch.tensor(pss_list, dtype=torch.float32, device=device)
    consensus_t = torch.tensor(consensus_list, dtype=torch.float32, device=device)

    # Tier counts
    n_r1 = int((tier_t == 0).sum().item())
    n_r2 = int((tier_t == 1).sum().item())
    n_r3 = int((tier_t == 2).sum().item())
    n_r4 = int((tier_t == 3).sum().item())

    return {
        "class_logits": mean_cls_logits.to(device),  # [B, C]   fp32
        "log_probs": log_probs.to(device),  # [B, C]   fp32
        "bbox_preds": mean_bbox_preds.to(device),  # [B, 4]   fp32
        "target_classes": target_classes.long().to(device),  # [B]  int64
        "target_bboxes": target_bboxes.float().to(device),  # [B, 4] fp32
        "tier_indices": tier_t,  # [B]      int64
        "sdca": sdca_t,  # [B]      fp32
        "aps": aps_t,  # [B]      fp32
        "sas": sas_t,  # [B]      fp32
        "pss": pss_t,  # [B]      fp32
        "consensus": consensus_t,  # [B]      fp32
        "ensemble_probs": ensemble_probs_list,  # list[dict] length B
        "majority_classes": majority_classes,  # list[str] length B
        "n_r1": n_r1,
        "n_r2": n_r2,
        "n_r3": n_r3,
        "n_r4": n_r4,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Lesion embedding extractor (for L_KG)
# ══════════════════════════════════════════════════════════════════════════════


def extract_lesion_embeddings(
    node_embeddings: torch.Tensor,
    node_names: List[str],
) -> torch.Tensor:
    """Slice lesion node rows from the full KG node embedding matrix.

    Args:
        node_embeddings: Tensor[N, d] — all 9 KG node embeddings from encoder.
        node_names:      Ordered list of node names matching row indices
                         (from build_encoder_inputs()["node_names"]).

    Returns:
        Tensor[4, d] — rows for ["MA","HE","Hard_EX","Soft_EX"] in that order.

    Raises:
        ValueError: if any lesion class is missing from node_names.
    """
    indices: List[int] = []
    for lesion in LESION_NAMES:
        if lesion not in node_names:
            raise ValueError(
                f"Lesion '{lesion}' not found in node_names={node_names}. "
                "Check KG artifact build."
            )
        indices.append(node_names.index(lesion))

    return node_embeddings[indices]  # [4, d]


# ══════════════════════════════════════════════════════════════════════════════
# Gate input builder (for L_gate)
# ══════════════════════════════════════════════════════════════════════════════


def build_gate_tensors(
    governed_batch: dict,
    node_embeddings: torch.Tensor,
    node_names: List[str],
    use_kg_embedding: bool = True,
) -> dict:
    """Build gate input tensors from a GovernedBatch and KG embeddings.

    Produces all tensors needed to call LGG.forward() + GateLoss.forward().

    Args:
        governed_batch:   Output of build_governed_batch().
        node_embeddings:  Tensor[N, d] — KG node embeddings from encoder.
        node_names:       Node name ordering from build_encoder_inputs()["node_names"].
        use_kg_embedding: If True (default), compute e_pred per bbox by
                          indexing the majority class lesion embedding.
                          If False, e_pred is zeros[B, 0] (ablation mode;
                          matches LGG(embedding_dim=0)).

    Returns:
        {
          "evidence_vectors": Tensor[B, 4]      — [SAS, SDCA, APS, PSS]
          "e_pred"           : Tensor[B, d]      — majority-class KG embedding
                               or Tensor[B, 0]   when use_kg_embedding=False
          "gate_input"       : Tensor[B, 4+d]    — concat for LGG.forward()
                               or Tensor[B, 4]   when use_kg_embedding=False
          "tier_indices"     : Tensor[B] int64   — supervision target for GateLoss
        }
    """
    device = governed_batch["sas"].device
    B = governed_batch["sas"].shape[0]

    evidence_vectors = torch.stack(
        [
            governed_batch["sas"],
            governed_batch["sdca"],
            governed_batch["aps"],
            governed_batch["pss"],
        ],
        dim=-1,
    ).to(
        device
    )  # [B, 4]

    if use_kg_embedding:
        lesion_emb = extract_lesion_embeddings(node_embeddings, node_names)
        # e_pred[i] = embedding of the majority predicted class for bbox i
        majority_idxs = torch.tensor(
            [_LESION_IDX.get(c, 0) for c in governed_batch["majority_classes"]],
            dtype=torch.long,
            device=device,
        )
        e_pred = lesion_emb[majority_idxs]  # [B, d]
    else:
        d = node_embeddings.shape[-1]
        e_pred = torch.zeros(B, 0, dtype=torch.float32, device=device)

    gate_input = torch.cat([evidence_vectors, e_pred], dim=-1)  # [B, 4+d] or [B,4]

    return {
        "evidence_vectors": evidence_vectors,  # [B, 4]
        "e_pred": e_pred,  # [B, d] or [B, 0]
        "gate_input": gate_input,  # [B, 4+d] or [B, 4]
        "tier_indices": governed_batch["tier_indices"],  # [B] int64
    }


# ══════════════════════════════════════════════════════════════════════════════
# SDGLLoss input builder (convenience)
# ══════════════════════════════════════════════════════════════════════════════


def build_sdgl_inputs(governed_batch: dict) -> dict:
    """Extract the exact keyword arguments needed for SDGLLoss.forward().

    Bridges GovernedBatch → SDGLLoss argument names, handling types.
    Call as: sdgl_loss(**build_sdgl_inputs(gb), sdm_norm=sdm_norm,
                       lesion_embeddings=lesion_emb)

    Args:
        governed_batch: Output of build_governed_batch().

    Returns:
        {
          "log_probs_batch"      : Tensor[B, C]
          "tier_targets"         : List[int]       — 0..3 per bbox
          "target_class_indices" : List[int]       — GT class index per bbox
          "ensemble_probs_batch" : List[dict]      — {class: float} per bbox
          "consensus_batch"      : List[float]
          "aps_batch"            : List[float]
          # caller must add: sdm_norm, lesion_embeddings, (fixed_edges)
        }
    """
    B = governed_batch["log_probs"].shape[0]
    return {
        "log_probs_batch": governed_batch["log_probs"],  # [B, C]
        "tier_targets": governed_batch["tier_indices"].tolist(),  # List[int]
        "target_class_indices": governed_batch["target_classes"].tolist(),  # List[int]
        "ensemble_probs_batch": governed_batch["ensemble_probs"],  # List[dict]
        "consensus_batch": governed_batch["consensus"].tolist(),  # List[float]
        "aps_batch": governed_batch["aps"].tolist(),  # List[float]
    }


# ══════════════════════════════════════════════════════════════════════════════
# Warmstart diagnostic helpers
# [DIAGNOSTIC ONLY] — override routing/evidence for SDGL tier pathway testing.
# NOT for production training or final experiment comparisons.
# ══════════════════════════════════════════════════════════════════════════════

# Canonical tier values used for synthetic injection.
# Keys: tier_index (0=R1,1=R2,2=R3,3=R4)
# Values: (num_heads_correct, sdca_norm, aps, pss, sas)
_WARMSTART_TIER_VALUES: Dict[int, tuple] = {
    0: (4, 0.88, 0.92, 0.85, 0.90),  # R1 — strong consensus
    1: (3, 0.72, 0.78, 0.70, 0.75),  # R2 — moderate consensus
    2: (2, 0.58, 0.62, 0.55, 0.60),  # R3 — partial consensus
    3: (0, 0.28, 0.30, 0.25, 0.30),  # R4 — low consensus (keeps low values)
}
_TIER_NAMES: Dict[int, str] = {0: "R1", 1: "R2", 2: "R3", 3: "R4"}


def inject_warmstart_tiers(
    routing_batch: List[dict],
    B: int,
    majority_classes: Optional[List[str]] = None,
    tier_schedule: Optional[List[int]] = None,
) -> List[dict]:
    """[DIAGNOSTIC] Override routing_batch to inject controlled tier assignments.

    Replaces tier assignments so that non-R4 tiers (R1/R2/R3) appear in the
    batch. Uses a round-robin schedule by default: [R1, R2, R3, R4, R1, ...].
    The original routing result is preserved under 'natural_tier_index'.

    Args:
        routing_batch:   List[dict] from route_bbox_batch() — length B.
        B:               Batch size.
        majority_classes: Optional class names to use as majority_class.
                         Falls back to routing_batch values if not provided.
        tier_schedule:   Explicit tier indices [0..3] per batch item, length B.
                         Default: round-robin [0, 1, 2, 3, 0, 1, ...].

    Returns:
        New List[dict] with warmstart tier assignments.
        Each record has 'warmstart_injected': True and 'natural_tier_index'.
    """
    if tier_schedule is None:
        tier_schedule = [i % 4 for i in range(B)]
    if len(tier_schedule) != B:
        raise ValueError(f"tier_schedule length {len(tier_schedule)} != batch size {B}")

    result: List[dict] = []
    for i in range(B):
        orig = routing_batch[i] if i < len(routing_batch) else {}
        natural_tier = int(orig.get("target_tier_index", 3))
        ws_tier = int(tier_schedule[i])
        n_correct, sdca, aps, pss, sas = _WARMSTART_TIER_VALUES[ws_tier]
        maj = (
            majority_classes[i]
            if (majority_classes and i < len(majority_classes))
            else orig.get("majority_class", LESION_NAMES[0])
        )
        result.append(
            {
                "target_tier_index": ws_tier,
                "target_tier": _TIER_NAMES[ws_tier],
                "num_heads_correct": n_correct,
                "sdca_norm": sdca,
                "aps": aps,
                "majority_class": maj,
                # Diagnostic traceability fields
                "natural_tier_index": natural_tier,
                "natural_tier": _TIER_NAMES.get(natural_tier, "R4"),
                "warmstart_injected": True,
            }
        )
    return result


def inject_warmstart_evidence(
    evidence_batch: List[dict],
    tier_indices: List[int],
) -> List[dict]:
    """[DIAGNOSTIC] Override evidence vectors to match warmstart tier values.

    Replaces SAS/SDCA/APS/PSS in evidence_batch[i]["evidence_vector"] with
    values consistent with tier_indices[i] from _WARMSTART_TIER_VALUES.

    Args:
        evidence_batch:  List[dict] from build_evidence_batch() — length B.
        tier_indices:    Warmstart tier index per batch item (0..3).

    Returns:
        New evidence_batch with injected evidence vectors.
        Each record has 'warmstart_injected': True.
    """
    result: List[dict] = []
    for i, ev in enumerate(evidence_batch):
        tier = int(tier_indices[i]) if i < len(tier_indices) else 3
        _, sdca, aps, pss, sas = _WARMSTART_TIER_VALUES[tier]
        new_ev = dict(ev) if isinstance(ev, dict) else {}
        new_ev["evidence_vector"] = [sas, sdca, aps, pss]
        new_ev["warmstart_injected"] = True
        result.append(new_ev)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Smoke test  (run: python -m src.losses.governed_adapter)
# ══════════════════════════════════════════════════════════════════════════════


def _smoke_test(device_str: str = "cpu") -> None:
    """Dummy smoke test — creates synthetic inputs and validates output shapes."""
    import torch.nn.functional as F  # noqa: F811

    device = torch.device(device_str)
    B, C, H, n_heads, d = 4, 4, 4, 4, 64

    print("=" * 60)
    print("governed_adapter.py — smoke test")
    print(f"  B={B}  C={C}  n_heads={n_heads}  embed_dim={d}  device={device}")
    print("=" * 60)

    # ── Synthetic head outputs ─────────────────────────────────────────────
    head_outputs = [
        (
            torch.randn(B, C, device=device),
            torch.sigmoid(torch.randn(B, 4, device=device)),
        )
        for _ in range(n_heads)
    ]

    # ── Synthetic targets ──────────────────────────────────────────────────
    target_classes = torch.randint(0, C, (B,), device=device)
    target_bboxes = torch.rand(B, 4, device=device)

    # ── Synthetic DetectorBox dicts ────────────────────────────────────────
    det_boxes = []
    for b in range(B):
        probs = F.softmax(torch.rand(C), dim=-1)
        ep = {c: float(probs[i]) for i, c in enumerate(LESION_NAMES)}
        det_boxes.append({"ensemble_probs": ep, "bbox_id": f"img{b:02d}_b00"})

    # ── Synthetic evidence_batch ───────────────────────────────────────────
    evidence_batch = [{"evidence_vector": [0.7, 0.4, 0.6, 0.5]} for _ in range(B)]

    # ── Synthetic routing_batch ────────────────────────────────────────────
    tiers = [0, 1, 2, 3]  # R1, R2, R3, R4 — one of each for B=4
    routing_batch = [
        {
            "target_tier_index": tiers[b],
            "sdca_norm": 0.3 + b * 0.1,
            "aps": 0.6,
            "num_heads_correct": n_heads - b,
            "majority_class": LESION_NAMES[b % C],
        }
        for b in range(B)
    ]

    # ── Build governed batch ───────────────────────────────────────────────
    gb = build_governed_batch(
        head_outputs=head_outputs,
        target_classes=target_classes,
        target_bboxes=target_bboxes,
        det_boxes=det_boxes,
        evidence_batch=evidence_batch,
        routing_batch=routing_batch,
        device=device,
    )

    # ── Report shapes ──────────────────────────────────────────────────────
    print("\n[1] GovernedBatch tensor shapes:")
    tensor_keys = [
        "class_logits",
        "log_probs",
        "bbox_preds",
        "target_classes",
        "target_bboxes",
        "tier_indices",
        "sdca",
        "aps",
        "sas",
        "pss",
        "consensus",
    ]
    for k in tensor_keys:
        v = gb[k]
        print(f"  {k:<20} {tuple(v.shape)}  dtype={v.dtype}")
    print(f"  {'ensemble_probs':<20} list[dict]  len={len(gb['ensemble_probs'])}")
    print(f"  {'majority_classes':<20} list[str]   val={gb['majority_classes']}")
    print(
        f"  tier dist  n_r1={gb['n_r1']}  n_r2={gb['n_r2']}  n_r3={gb['n_r3']}  n_r4={gb['n_r4']}"
    )

    # ── Lesion embeddings ──────────────────────────────────────────────────
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
    node_embeddings = torch.randn(9, d, device=device)
    lesion_emb = extract_lesion_embeddings(node_embeddings, node_names)
    print(f"\n[2] extract_lesion_embeddings output:")
    print(f"  lesion_emb  {tuple(lesion_emb.shape)}  dtype={lesion_emb.dtype}")

    # ── Gate tensors ───────────────────────────────────────────────────────
    gt = build_gate_tensors(gb, node_embeddings, node_names, use_kg_embedding=True)
    print(f"\n[3] build_gate_tensors output (use_kg_embedding=True):")
    for k, v in gt.items():
        print(f"  {k:<20} {tuple(v.shape)}  dtype={v.dtype}")

    gt_nokg = build_gate_tensors(
        gb, node_embeddings, node_names, use_kg_embedding=False
    )
    print(f"\n[4] build_gate_tensors output (use_kg_embedding=False / ablation):")
    for k, v in gt_nokg.items():
        print(f"  {k:<20} {tuple(v.shape)}  dtype={v.dtype}")

    # ── SDGL inputs ────────────────────────────────────────────────────────
    sdgl_in = build_sdgl_inputs(gb)
    print(f"\n[5] build_sdgl_inputs output:")
    for k, v in sdgl_in.items():
        if hasattr(v, "shape"):
            print(f"  {k:<26} Tensor{tuple(v.shape)}  dtype={v.dtype}")
        else:
            print(f"  {k:<26} {type(v).__name__}  len={len(v)}")

    # ── Quick integration with SDGLLoss ────────────────────────────────────
    try:
        from lgovnet.src.losses.sdgl import SDGLLoss

        sdgl = SDGLLoss()
        sdm_norm = {
            (LESION_NAMES[i], LESION_NAMES[j]): {"raw": 1.0, "norm": 0.5}
            for i in range(C)
            for j in range(i + 1, C)
        }
        res = sdgl(
            **sdgl_in,
            sdm_norm=sdm_norm,
            lesion_embeddings=lesion_emb,
        )
        print(f"\n[6] SDGLLoss output:")
        for k, v in res.items():
            print(f"  {k:<12} {float(v):.6f}")
    except Exception as exc:
        print(f"\n[6] SDGLLoss call: SKIPPED ({exc})")

    # ── Quick integration with GateLoss ───────────────────────────────────
    try:
        from lgovnet.src.losses.gate_loss import GateLoss

        gate_loss_fn = GateLoss()
        # Synthetic tier_logits from LGG (not yet wired — simulate)
        tier_logits = torch.randn(B, 4, device=device)
        gl_res = gate_loss_fn(tier_logits, gt["tier_indices"])
        print(f"\n[7] GateLoss output:")
        for k, v in gl_res.items():
            print(f"  {k:<12} {float(v):.6f}")
    except Exception as exc:
        print(f"\n[7] GateLoss call: SKIPPED ({exc})")

    print("\n" + "=" * 60)
    print("governed_adapter smoke test: PASS")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="governed_adapter smoke test")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()
    _smoke_test(device_str=args.device)
