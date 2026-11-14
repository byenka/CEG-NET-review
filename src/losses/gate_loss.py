

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
