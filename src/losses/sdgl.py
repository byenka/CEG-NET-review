

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

    return -log_probs[target_class_idx]


def compute_r2_loss(
    log_probs: torch.Tensor,
    q_sdm: torch.Tensor,
) -> torch.Tensor:

    return -(q_sdm * log_probs).sum()


def compute_r3_loss(
    log_probs: torch.Tensor,
    target_class_idx: int,
    consensus: float,
    aps: float,
) -> torch.Tensor:

    w_j = float(consensus) * float(aps)
    return w_j * (-log_probs[target_class_idx])


# ═══════════════════════════════════════════════════════════════════════════════
# KG-alignment auxiliary loss
# ═══════════════════════════════════════════════════════════════════════════════


def compute_l_kg(
    lesion_embeddings: torch.Tensor,
    fixed_edges: Optional[Dict[Tuple[str, str], float]] = None,
) -> torch.Tensor:

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

    total = l_sdgl + lambda_gate * l_gate + lambda_kg * l_kg

    if l_eff is not None and lambda_eff != 0.0:
        total = total + lambda_eff * l_eff

    if l_aux is not None and lambda_aux != 0.0:
        total = total + lambda_aux * l_aux

    return total
