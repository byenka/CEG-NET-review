"""
gate_loss.py
Gate supervision loss (L_gate) for LGov-Net Contribution 3.

Supervises the Learned Governance Gate (LGG) to match the pseudo-label
tier targets produced by routing.py.  L_gate is the only component that
directly trains the LGG parameters.

Formula:
  L_gate = CrossEntropy(tier_logits, target_tier_indices)

The target tier indices come from derive_gate_target() / route_bbox_batch()
in routing.py — they are derived from GT matching and clinical evidence,
never from the gate's own output.

Optionally, label smoothing can be applied to prevent the gate from
becoming over-confident about noisy pseudo-labels:
  L_gate = (1 - ε) · CE(logits, hard_target) + ε · mean(CE(logits, uniform))
  implemented via F.cross_entropy(..., label_smoothing=epsilon).

Public API:
  compute_l_gate(tier_logits, target_tier_indices, epsilon) -> Tensor[]
  class GateLoss(nn.Module)

Dependencies:
  - PyTorch
  NO dependency on sdgl.py, routing.py, refine.py, or training loop.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Canonical tier names — must match gate.py
TIER_NAMES: List[str] = ["R1", "R2", "R3", "R4"]
NUM_TIERS: int = len(TIER_NAMES)


# ═══════════════════════════════════════════════════════════════════════════════
# Functional interface
# ═══════════════════════════════════════════════════════════════════════════════


def compute_l_gate(
    tier_logits: torch.Tensor,
    target_tier_indices: torch.Tensor,
    epsilon: float = 0.0,
    tier_weights: Optional[torch.Tensor] = None,
    focal_gamma: float = 0.0,
    sample_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """L_gate — cross-entropy supervision of the governance gate.

    L_gate = CrossEntropy(tier_logits, target_tier_indices)

    Targets are pseudo-labels from routing.derive_gate_target(), based on
    GT matching and clinical evidence.  They are NOT derived from the gate's
    own predictions — using gate output as the supervision target would cause
    circular training.

    Args:
        tier_logits:          Tensor[B, 4] — raw (pre-softmax) gate outputs.
                              Comes from LGG.forward()["tier_logits"].
        target_tier_indices:  Tensor[B] int64 — tier index per bbox
                              (0=R1, 1=R2, 2=R3, 3=R4).
                              Comes from routing "target_tier_index" field.
        epsilon:              Label smoothing in [0, 1).  0 = hard CE (default).
                              Recommended: 0.05–0.1 to handle noisy pseudo-labels.

    Returns:
        Scalar tensor — weighted mean gate loss over the batch.
        Returns zero (no gradient) if the batch is empty.
    """
    if tier_logits.shape[0] == 0:
        return torch.zeros(1, device=tier_logits.device).squeeze()

    ce = F.cross_entropy(
        tier_logits,
        target_tier_indices,
        weight=tier_weights,
        label_smoothing=float(epsilon),
        reduction="none",
    )

    gamma = float(focal_gamma)
    if gamma > 0.0:
        pt = (
            F.softmax(tier_logits, dim=-1)
            .gather(1, target_tier_indices.view(-1, 1))
            .squeeze(1)
        )
        pt = pt.clamp(min=1e-6, max=1.0 - 1e-6)
        ce = ((1.0 - pt) ** gamma) * ce

    if sample_weights is not None:
        sw = sample_weights.to(device=tier_logits.device, dtype=ce.dtype)
        sw = sw.clamp(min=0.0)
        denom = sw.sum().clamp(min=1e-6)
        return (ce * sw).sum() / denom

    return ce.mean()


# ═══════════════════════════════════════════════════════════════════════════════
# GateLoss module
# ═══════════════════════════════════════════════════════════════════════════════


class GateLoss(nn.Module):
    """Gate supervision loss module for LGG training.

    Wraps compute_l_gate with configurable label smoothing and optional
    per-tier class weighting (to handle tier imbalance in training data).

    Args:
        epsilon:      Label smoothing coefficient. Default 0.0 (hard CE).
        tier_weights: Optional Tensor[4] or list[float] — per-tier class
                      weights for imbalanced tier distributions.
                      E.g. [1.0, 1.0, 2.0, 1.0] up-weights R3 supervision.
                      None → uniform weights.
        focal_gamma:  Optional focal gamma for hard-example emphasis.
                  0.0 disables focal scaling (default).
    """

    def __init__(
        self,
        epsilon: float = 0.0,
        tier_weights: Optional[List[float]] = None,
        focal_gamma: float = 0.0,
    ) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.focal_gamma = float(focal_gamma)

        if tier_weights is not None:
            w = torch.tensor(tier_weights, dtype=torch.float32)
            self.register_buffer("tier_weights", w)
        else:
            self.register_buffer("tier_weights", None)

    def forward(
        self,
        tier_logits: torch.Tensor,
        target_tier_indices: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute L_gate for a batch.

        Args:
            tier_logits:          Tensor[B, 4] — gate logits from LGG.forward().
            target_tier_indices:  Tensor[B] int64 — from routing target records.

        Returns:
            {
              "l_gate": Tensor[] — scalar CE loss,
              "tier_acc": Tensor[] — fraction of correct tier predictions (detached),
            }
        """
        if tier_logits.shape[0] == 0:
            zero = torch.zeros(1, device=tier_logits.device).squeeze()
            return {"l_gate": zero, "tier_acc": zero}

        # Optionally apply per-tier class weights
        weight = getattr(self, "tier_weights", None)
        if weight is not None:
            weight = weight.to(tier_logits.device)

        l_gate = compute_l_gate(
            tier_logits,
            target_tier_indices,
            epsilon=float(self.epsilon),
            tier_weights=weight,
            focal_gamma=float(self.focal_gamma),
            sample_weights=sample_weights,
        )

        # Accuracy metric (no gradient needed) for logging
        with torch.no_grad():
            pred_hard = torch.argmax(tier_logits, dim=-1)
            tier_acc = (pred_hard == target_tier_indices).float().mean()
            tier_counts = torch.bincount(
                target_tier_indices,
                minlength=NUM_TIERS,
            ).to(dtype=torch.float32)
            tier_correct = torch.bincount(
                target_tier_indices[pred_hard == target_tier_indices],
                minlength=NUM_TIERS,
            ).to(dtype=torch.float32)
            tier_acc_per_tier = torch.where(
                tier_counts > 0,
                tier_correct / tier_counts.clamp(min=1.0),
                torch.zeros_like(tier_counts),
            )

        out: Dict[str, torch.Tensor] = {
            "l_gate": l_gate,
            "tier_acc": tier_acc,
        }

        for idx, tier_name in enumerate(TIER_NAMES):
            key = tier_name.lower()
            out[f"tier_count_{key}"] = tier_counts[idx]
            out[f"tier_acc_{key}"] = tier_acc_per_tier[idx]

        return out
