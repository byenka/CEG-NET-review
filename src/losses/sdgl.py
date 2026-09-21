"""
sdgl.py
Semantic Distance Governed Loss (SDGL) — LGov-Net Contribution 3.

Implements the tiered classification loss and KG-alignment auxiliary loss
that form the core of the closed optimization loop described in the
architecture spec (architecture-lgov-net-v2_rev.txt, Contribution 3).

Loss formula (architecture spec, exact):
  L_SDGL  = λ1·L_det^R1  +  λ2·L_soft^R2  +  λ3·Σ_j w_j·L_det^R3
  L_KG    = Σ_{(i,j) ∈ E_fixed} (SDM(c_i, c_j) − w_target)²
  L_total = L_SDGL  +  λ_gate·L_gate  +  λ_KG·L_KG
            + λ_eff·L_eff  (hook, 0 until implemented)
            + λ_aux·L_aux  (hook, 0 until implemented)

Tier treatment:
  R1 — Certified:  standard cross-entropy on hard (argmax) class target.
  R2 — Provisional: cross-entropy with SDM-shaped soft target q_SDM.
       q_SDM(c|b_j) = softmax(log P_ens(c|b_j) / τ_SDM(c))
       τ_SDM(c)     = 1 + η · SDM_norm(c, c_majority)
       Effect: classes close to majority in KG (low τ) → sharper target;
               classes far from majority (high τ) → damped gradients.
  R3 — Refined: same cross-entropy as R1 but scaled by per-box weight
       w_j = consensus · APS
       consensus = num_heads_correct / 4  (from routing target dict)
  R4 — Excluded: no gradient contribution.

KG-alignment (L_KG):
  Anchors learned lesion embeddings to fixed clinical distances.
  For each fixed edge (c_i, c_j) with clinical target weight w_target:
    loss += (||e_ci − e_cj||₂ − w_target)²
  The fixed edges are severity_transition pairs from ontology.SEVERITY_TRANSITION.
  L_KG controls drift of the KG embedding space away from clinical reality.

Public API:
  compute_tau_sdm(ensemble_probs, sdm_norm, eta) -> Tensor[C]
  build_q_sdm(ensemble_probs, sdm_norm, eta)     -> Tensor[C]
  compute_r1_loss(log_probs, target_class_idx)   -> Tensor[]
  compute_r2_loss(log_probs, q_sdm)              -> Tensor[]
  compute_r3_loss(log_probs, target_class_idx, consensus, aps) -> Tensor[]
  compute_l_kg(lesion_embeddings, fixed_edges)   -> Tensor[]
  class SDGLLoss(nn.Module)

L_total contract lives in this module as ``compute_total_loss()``.
Gate loss (L_gate) is imported from gate_loss.py.

Dependencies:
  - PyTorch
  - src/data/ontology.py (LESION_CLASSES, SEVERITY_TRANSITION)
  NO dependency on routing.py, refine.py, or training loop.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from lgovnet.src.data.ontology import LESION_NODES as _LESION_NODES
from lgovnet.src.data.ontology import SEVERITY_TRANSITION

# ── Canonical lesion class ordering (matches contracts.py, encoder.py) ────────
LESION_CLASSES: List[str] = sorted(_LESION_NODES.keys(), key=lambda c: _LESION_NODES[c])

_LESION_IDX: Dict[str, int] = {c: i for i, c in enumerate(LESION_CLASSES)}
NUM_CLASSES: int = len(LESION_CLASSES)


# ═══════════════════════════════════════════════════════════════════════════════
# τ_SDM and q_SDM helpers
# ═══════════════════════════════════════════════════════════════════════════════


def compute_tau_sdm(
    majority_class: str,
    sdm_norm: Dict[Tuple[str, str], Dict[str, float]],
    eta: float = 1.0,
) -> torch.Tensor:
    """Compute per-class temperature τ_SDM for one bbox.

    Formula (architecture spec):
        τ_SDM(c) = 1 + η · SDM_norm(c, c_majority)

    For the majority class itself: SDM_norm = 0 → τ = 1 (standard temperature).
    For classes distant from the majority: τ > 1 → denominator grows → softer
    distribution → dampened gradient signal.

    Args:
        majority_class: Predicted majority class name (argmax of ensemble_probs).
        sdm_norm:       Normalized SDM dict — keys (ci, cj) with lower index
                        first, values {"raw": float, "norm": float}.
        eta:            Scaling factor η ≥ 0. η=0 → all τ=1 (no modulation).

    Returns:
        Tensor[C] — per-class temperature, C = len(LESION_CLASSES).
        Values ≥ 1.0.
    """
    taus = []
    for c in LESION_CLASSES:
        if c == majority_class:
            taus.append(1.0)
            continue

        # Canonical key: lower lesion index first
        if _LESION_IDX.get(majority_class, 99) <= _LESION_IDX.get(c, 99):
            key = (majority_class, c)
        else:
            key = (c, majority_class)

        entry = sdm_norm.get(key)
        sdm_val = float(entry["norm"]) if entry is not None else 0.0
        taus.append(1.0 + eta * sdm_val)

    return torch.tensor(taus, dtype=torch.float32)  # [C]


def build_q_sdm(
    ensemble_probs: Dict[str, float],
    sdm_norm: Dict[Tuple[str, str], Dict[str, float]],
    eta: float = 1.0,
) -> torch.Tensor:
    """Build the SDM-shaped soft target distribution q_SDM for one R2 bbox.

    Formula (architecture spec):
        q_SDM(c|b_j) = softmax( log P_ens(c|b_j) / τ_SDM(c) )

    Dividing log-probabilities by τ_SDM:
      - Classes clinically close to majority (small SDM → small τ → τ ≈ 1):
        sharpened log-prob → q_SDM concentrates on those classes.
      - Classes clinically far (large SDM → large τ):
        suppressed log-prob → q_SDM de-emphasises them.

    Args:
        ensemble_probs: {class_name: probability} from DetectorBox.
                        Must contain all 4 LESION_CLASSES.
        sdm_norm:       Normalized SDM dict from encoder.normalize_sdca().
        eta:            Temperature scaling factor.

    Returns:
        Tensor[C] — soft target distribution summing to 1.  C = 4.
    """
    # Build P_ens vector in canonical LESION_CLASSES order
    p_ens = torch.tensor(
        [float(ensemble_probs.get(c, 1e-9)) for c in LESION_CLASSES],
        dtype=torch.float32,
    ).clamp(min=1e-9)

    majority_class = LESION_CLASSES[int(torch.argmax(p_ens).item())]
    tau = compute_tau_sdm(majority_class, sdm_norm, eta)  # [C]

    log_p = torch.log(p_ens)  # [C]
    scaled_log_p = log_p / tau  # element-wise division by τ_SDM

    return F.softmax(scaled_log_p, dim=0)  # [C], sums to 1


# ═══════════════════════════════════════════════════════════════════════════════
# Per-tier loss functions
# ═══════════════════════════════════════════════════════════════════════════════


def compute_r1_loss(
    log_probs: torch.Tensor,
    target_class_idx: int,
) -> torch.Tensor:
    """L_det^R1 — standard cross-entropy on hard class target.

    R1 detections are certified correct (≥3/4 heads right, GT matched).
    No KG modulation — treat exactly like standard supervised detection.

    STUB — classification term only.
    Architecture spec defines the full R1 detection loss as:
        L_det^R1 = L_focal + L_smooth-L1 + L_GIoU
    The bbox regression terms (smooth-L1 for box coordinates, GIoU for
    localization quality) are NOT included here.  They require access to
    the detector's box-regression head outputs, which are not available
    until the backbone is integrated in Phase 14.
    This function will be extended in Phase 14 to accept predicted_box and
    target_box arguments and compute the full three-term detection loss.

    Args:
        log_probs:        Tensor[C] — log-softmax probabilities from the
                          detector's class head for this bbox.
        target_class_idx: int in [0, C-1] — GT class index.

    Returns:
        Scalar tensor — NLL = −log_prob[target_class_idx].
        (classification term only; regression terms pending Phase 14)
    """
    return -log_probs[target_class_idx]


def compute_r2_loss(
    log_probs: torch.Tensor,
    q_sdm: torch.Tensor,
) -> torch.Tensor:
    """L_soft^R2 — KL divergence from predicted log-probs to q_SDM target.

    L_soft^R2(b_j) = −Σ_c  q_SDM(c) · log_probs(c)
                   = KL(q_SDM || p_pred) − H(q_SDM)    [H is const w.r.t. θ]

    We implement as cross-entropy H(q_SDM, p_pred), which equals KL up to a
    constant and is standard practice for soft-label supervision.

    Args:
        log_probs: Tensor[C] — log-softmax output of detector head.
        q_sdm:     Tensor[C] — SDM-shaped soft target from build_q_sdm().
                   Must sum to 1.

    Returns:
        Scalar tensor — cross-entropy loss.
    """
    return -(q_sdm * log_probs).sum()


def compute_r3_loss(
    log_probs: torch.Tensor,
    target_class_idx: int,
    consensus: float,
    aps: float,
) -> torch.Tensor:
    """Weighted L_det^R3 — reduced-confidence cross-entropy for R3 boxes.

    w_j = consensus × APS
    L_det^R3(b_j) = w_j · (−log_prob[target_class_idx])

    consensus = num_heads_correct / 4  (from routing target record).
    APS = Compat(majority_class, assigned_zone) from evidence computation.

    Both are in [0, 1]. Their product down-weights boxes where the detector
    is uncertain (low consensus) or anatomically implausible (low APS).

    Args:
        log_probs:        Tensor[C] — log-softmax output.
        target_class_idx: int — GT (or majority) class index.
        consensus:        float in [0, 1] — fraction of heads correct.
        aps:              float in [0, 1] — anatomical plausibility score.

    Returns:
        Scalar tensor — w_j-scaled NLL.
    """
    w_j = float(consensus) * float(aps)
    return w_j * (-log_probs[target_class_idx])


# ═══════════════════════════════════════════════════════════════════════════════
# KG-alignment auxiliary loss
# ═══════════════════════════════════════════════════════════════════════════════


def compute_l_kg(
    lesion_embeddings: torch.Tensor,
    fixed_edges: Optional[Dict[Tuple[str, str], float]] = None,
) -> torch.Tensor:
    """L_KG — KG-alignment loss anchoring embeddings to clinical distances.

    Formula (architecture spec):
        L_KG = Σ_{(i,j) ∈ E_fixed} (||e_ci − e_cj||₂ − w_target)²

    E_fixed = severity_transition pairs from ontology.SEVERITY_TRANSITION.
    w_target = the clinical severity cost (e.g. 0.0 for MA↔HE, 2.0 for MA↔Soft_EX).

    Prevents the learned KG embeddings from drifting away from the clinical
    geometry encoded in the ICDR grading ontology.

    Args:
        lesion_embeddings: Tensor[C, d] — one embedding per lesion class.
                           Row order must match LESION_CLASSES.
        fixed_edges:       Optional override dict {(ci, cj): w_target}.
                           Keys must use canonical class names.
                           If None, uses SEVERITY_TRANSITION from ontology.

    Returns:
        Scalar tensor — sum of squared distance residuals.
    """
    if fixed_edges is None:
        # Build from ontology.SEVERITY_TRANSITION (frozenset keys → ordered tuples)
        fixed_edges = {}
        for fs_key, cost in SEVERITY_TRANSITION.items():
            pair = sorted(fs_key, key=lambda c: _LESION_IDX.get(c, 99))
            if len(pair) == 2:
                fixed_edges[(pair[0], pair[1])] = float(cost)

    loss = torch.zeros(
        1, dtype=lesion_embeddings.dtype, device=lesion_embeddings.device
    )

    for (ci, cj), w_target in fixed_edges.items():
        idx_i = _LESION_IDX.get(ci)
        idx_j = _LESION_IDX.get(cj)
        if idx_i is None or idx_j is None:
            continue
        e_i = lesion_embeddings[idx_i]
        e_j = lesion_embeddings[idx_j]
        sdm_learned = torch.norm(e_i - e_j, p=2)
        loss = loss + (sdm_learned - w_target) ** 2

    return loss.squeeze()


# ═══════════════════════════════════════════════════════════════════════════════
# SDGLLoss module
# ═══════════════════════════════════════════════════════════════════════════════


class SDGLLoss(nn.Module):
    """Semantic Distance Governed Loss — complete L_SDGL + L_KG implementation.

    Accepts a batch of (log_probs, tier, metadata) records and computes the
    tiered loss.  All tier routing and evidence computation must be done
    before calling this module (no routing logic inside).

    STUB NOTE — R1 and R3 use classification-only NLL.
    The full detection loss for R1/R3 per the architecture spec is:
        L_det = L_focal + L_smooth-L1 + L_GIoU
    Bbox regression terms are pending backbone integration (Phase 14).
    This module will be extended in Phase 14 to accept box-regression outputs.

    Args:
        lambda1: Weight for L_det^R1. Default 1.0.
        lambda2: Weight for L_soft^R2. Default 1.0.
        lambda3: Weight for Σ w_j·L_det^R3. Default 0.5.
        lambda_kg: Weight for L_KG. Default 0.1.
        eta:     τ_SDM scaling factor for R2 soft targets. Default 1.0.
    """

    def __init__(
        self,
        lambda1: float = 1.0,
        lambda2: float = 1.0,
        lambda3: float = 0.5,
        lambda_kg: float = 0.1,
        eta: float = 1.0,
    ) -> None:
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda_kg = lambda_kg
        self.eta = eta

    def forward(
        self,
        log_probs_batch: torch.Tensor,
        tier_targets: List[int],
        target_class_indices: List[int],
        ensemble_probs_batch: List[Dict[str, float]],
        sdm_norm: Dict[Tuple[str, str], Dict[str, float]],
        consensus_batch: List[float],
        aps_batch: List[float],
        lesion_embeddings: torch.Tensor,
        fixed_edges: Optional[Dict[Tuple[str, str], float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute L_SDGL and L_KG for a batch.

        Args:
            log_probs_batch:       Tensor[B, C] — log-softmax class scores.
            tier_targets:          List[int] length B — 0=R1,1=R2,2=R3,3=R4.
            target_class_indices:  List[int] length B — GT/majority class index.
            ensemble_probs_batch:  List[dict] length B — per-bbox ensemble probs
                                   (needed for R2 soft-target construction).
            sdm_norm:              Normalized SDM dict from encoder.
            consensus_batch:       List[float] length B — num_heads_correct/4.
            aps_batch:             List[float] length B — APS per bbox.
            lesion_embeddings:     Tensor[C, d] — current KG node embeddings.
            fixed_edges:           Optional fixed-edge override for L_KG.

        Returns:
            {
              "l_r1":   Tensor[] — mean R1 loss (0 if no R1 boxes),
              "l_r2":   Tensor[] — mean R2 loss (0 if no R2 boxes),
              "l_r3":   Tensor[] — mean weighted R3 loss (0 if no R3 boxes),
              "l_sdgl": Tensor[] — λ1·l_r1 + λ2·l_r2 + λ3·l_r3,
              "l_kg":   Tensor[] — KG alignment loss,
            }
        """
        device = log_probs_batch.device
        zero = torch.zeros(1, device=device).squeeze()

        r1_losses: List[torch.Tensor] = []
        r2_losses: List[torch.Tensor] = []
        r3_losses: List[torch.Tensor] = []

        for i in range(len(tier_targets)):
            tier = tier_targets[i]

            if tier == 3:  # R4 — excluded
                continue

            lp = log_probs_batch[i]  # [C]
            tc = target_class_indices[i]

            if tier == 0:  # R1
                r1_losses.append(compute_r1_loss(lp, tc))

            elif tier == 1:  # R2
                q = build_q_sdm(ensemble_probs_batch[i], sdm_norm, self.eta)
                q = q.to(device)
                r2_losses.append(compute_r2_loss(lp, q))

            elif tier == 2:  # R3
                r3_losses.append(
                    compute_r3_loss(lp, tc, consensus_batch[i], aps_batch[i])
                )

        l_r1 = torch.stack(r1_losses).mean() if r1_losses else zero
        l_r2 = torch.stack(r2_losses).mean() if r2_losses else zero
        l_r3 = torch.stack(r3_losses).mean() if r3_losses else zero

        l_sdgl = self.lambda1 * l_r1 + self.lambda2 * l_r2 + self.lambda3 * l_r3
        l_kg = compute_l_kg(lesion_embeddings, fixed_edges)

        return {
            "l_r1": l_r1,
            "l_r2": l_r2,
            "l_r3": l_r3,
            "l_sdgl": l_sdgl,
            "l_kg": l_kg,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# L_total contract
# ═══════════════════════════════════════════════════════════════════════════════


def compute_total_loss(
    l_sdgl: torch.Tensor,
    l_gate: torch.Tensor,
    l_kg: torch.Tensor,
    lambda_gate: float = 1.0,
    lambda_kg: float = 0.1,
    l_eff: Optional[torch.Tensor] = None,
    lambda_eff: float = 0.0,
    l_aux: Optional[torch.Tensor] = None,
    lambda_aux: float = 0.0,
) -> torch.Tensor:
    """L_total contract (architecture spec).

    L_total = L_SDGL + λ_gate·L_gate + λ_KG·L_KG
              + λ_eff·L_eff (hook, 0 until implemented)
              + λ_aux·L_aux (hook, 0 until implemented)

    L_eff and L_aux are hook positions for future efficiency /
    auxiliary losses.  Pass them as None (default) until implemented;
    they contribute 0 to the total.

    Args:
        l_sdgl:      Tiered detection loss from SDGLLoss.
        l_gate:      Gate supervision loss from GateLoss.
        l_kg:        KG alignment loss from compute_l_kg / SDGLLoss.
        lambda_gate: Weight for L_gate. Default 1.0.
        lambda_kg:   Weight for L_KG. Default 0.1.
        l_eff:       Hook for future efficiency loss. None → 0.
        lambda_eff:  Weight for l_eff. Default 0.0.
        l_aux:       Hook for future auxiliary loss. None → 0.
        lambda_aux:  Weight for l_aux. Default 0.0.

    Returns:
        Scalar tensor — total loss ready for .backward().
    """
    total = l_sdgl + lambda_gate * l_gate + lambda_kg * l_kg

    if l_eff is not None and lambda_eff != 0.0:
        total = total + lambda_eff * l_eff

    if l_aux is not None and lambda_aux != 0.0:
        total = total + lambda_aux * l_aux

    return total
