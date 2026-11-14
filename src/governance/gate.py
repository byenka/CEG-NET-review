

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Constants ─────────────────────────────────────────────────────────────────

# Canonical tier names — index 0..3 maps to R1..R4
TIER_NAMES: List[str] = ["R1", "R2", "R3", "R4"]
NUM_TIERS: int = len(TIER_NAMES)

# Canonical lesion ordering for e_pred computation
_LESION_CLASSES: List[str] = ["MA", "HE", "Hard_EX", "Soft_EX"]


# ═══════════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════════


def build_pred_embedding(
    ensemble_probs_batch: torch.Tensor,
    lesion_embeddings: torch.Tensor,
) -> torch.Tensor:

    # [B, C] @ [C, d] → [B, d]
    return ensemble_probs_batch @ lesion_embeddings


def concat_gate_input(
    evidence_vectors: torch.Tensor,
    e_pred: torch.Tensor,
    use_kg_embedding: bool = True,
) -> torch.Tensor:

    if use_kg_embedding:
        return torch.cat([evidence_vectors, e_pred], dim=-1)
    return evidence_vectors


def gumbel_soft_routing(
    logits: torch.Tensor,
    temperature: float = 1.0,
    hard: bool = False,
) -> torch.Tensor:

    return F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=-1)


def get_gumbel_tau(
    step: int,
    tau_start: float = 1.0,
    tau_end: float = 0.1,
    tau_anneal_steps: int = 10_000,
) -> float:

    if tau_anneal_steps <= 0:
        raise ValueError(f"tau_anneal_steps must be > 0, got {tau_anneal_steps}")
    progress = min(step / tau_anneal_steps, 1.0)
    return tau_start + (tau_end - tau_start) * progress


def predict_tier(
    tier_probs: torch.Tensor,
    tier_names: Optional[List[str]] = None,
) -> dict:

    if tier_names is None:
        tier_names = TIER_NAMES

    tier_hard = torch.argmax(tier_probs, dim=-1)  # [B]
    confidence = tier_probs.max(dim=-1).values  # [B]
    tier_labels = [tier_names[i.item()] for i in tier_hard]

    return {
        "tier_hard": tier_hard,
        "tier_labels": tier_labels,
        "confidence": confidence,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# LGG — Learned Governance Gate
# ═══════════════════════════════════════════════════════════════════════════════


class LGG(nn.Module):


    def __init__(
        self,
        evidence_dim: int = 4,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        gate_dim = evidence_dim + embedding_dim
        self.fc1 = nn.Linear(gate_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, NUM_TIERS)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, gate_input: torch.Tensor) -> Dict[str, object]:

        h = F.relu(self.norm1(self.fc1(gate_input)))  # [B, hidden_dim]
        h = self.drop(h)
        logits = self.fc2(h)  # [B, 4]
        probs = F.softmax(logits, dim=-1)  # [B, 4]
        tier_hard = torch.argmax(logits, dim=-1)  # [B]

        return {
            "tier_logits": logits,
            "tier_probs": probs,
            "tier_hard": tier_hard,
            "tier_names": TIER_NAMES,
        }

    def gumbel_forward(
        self,
        gate_input: torch.Tensor,
        temperature: float = 1.0,
        hard: bool = False,
    ) -> Dict[str, object]:

        h = F.relu(self.norm1(self.fc1(gate_input)))
        h = self.drop(h)
        logits = self.fc2(h)  # [B, 4]
        gumbel = gumbel_soft_routing(
            logits, temperature=temperature, hard=hard
        )  # [B, 4]
        tier_hard = torch.argmax(logits, dim=-1)  # [B]

        return {
            "tier_logits": logits,
            "tier_probs": gumbel,
            "tier_hard": tier_hard,
            "tier_names": TIER_NAMES,
            "gumbel_soft": gumbel,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════════════


def build_lgg(
    evidence_dim: int = 4,
    embedding_dim: int = 64,
    hidden_dim: int = 128,
    dropout: float = 0.1,
) -> LGG:

    return LGG(
        evidence_dim=evidence_dim,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Demo (run as: python -m src.governance.gate)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    torch.manual_seed(42)
    B = 3  # batch of 3 bboxes
    D = 64  # embedding_dim
    C = 4  # lesion classes

    # ── Fake evidence vectors [B, 4] ─────────────────────────────────────────
    # Each row: [SAS, SDCA_norm, APS, PSS]
    ev = torch.tensor(
        [
            [0.75, 0.31, 0.72, 0.50],  # bbox 0: Hard_EX in Foveal, PSS placeholder
            [0.45, 0.85, 0.50, 0.50],  # bbox 1: ambiguous, high KG separation
            [0.28, 0.46, 0.38, 0.50],  # bbox 2: low agree, suspicious zone
        ],
        dtype=torch.float32,
    )

    # ── Fake lesion embeddings [C, D] & ensemble probs [B, C] ─────────────
    lesion_emb = torch.randn(C, D) * 0.3  # [4, 64] — from LGovNetEncoder
    ensemble_p = torch.softmax(torch.randn(B, C), dim=-1)  # [3, 4] — sums to 1

    # ── Build gate input ─────────────────────────────────────────────────────
    e_pred = build_pred_embedding(ensemble_p, lesion_emb)  # [B, 64]
    gate_input = concat_gate_input(ev, e_pred)  # [B, 68]

    # ── Standard forward ─────────────────────────────────────────────────────
    gate = build_lgg(evidence_dim=4, embedding_dim=D)
    gate.eval()
    with torch.no_grad():
        out = gate(gate_input)

    print("=== LGG (Learned Governance Gate) demo ===")
    print(f"gate_input shape : {gate_input.shape}   [B={B}, 4+d={4+D}]")
    print(f"tier_logits shape: {out['tier_logits'].shape}")
    print(f"tier_probs  shape: {out['tier_probs'].shape}")
    print()
    print(f"{'bbox':>5}  {'R1':>7}  {'R2':>7}  {'R3':>7}  {'R4':>7}  tier")
    print("  " + "─" * 48)
    for i in range(B):
        probs = out["tier_probs"][i]
        hard = out["tier_hard"][i].item()
        row = f"  {i:>3}  " + "  ".join(f"{p:.4f}" for p in probs.tolist())
        row += f"  → {TIER_NAMES[hard]}"
        print(row)

    print()

    # ── Gumbel-Softmax forward (training mode) ───────────────────────────────
    gate.train()
    out_g = gate.gumbel_forward(gate_input, temperature=0.5, hard=False)
    print(f"Gumbel τ=0.5 tier_probs (soft):")
    for i in range(B):
        probs = out_g["tier_probs"][i]
        hard = out_g["tier_hard"][i].item()
        row = f"  bbox{i}  " + "  ".join(f"{p:.4f}" for p in probs.tolist())
        row += f"  → {TIER_NAMES[hard]}"
        print(row)

    print()

    # ── predict_tier ─────────────────────────────────────────────────────────
    gate.eval()
    with torch.no_grad():
        out2 = gate(gate_input)
    pred = predict_tier(out2["tier_probs"])
    print(f"predict_tier labels : {pred['tier_labels']}")
    print(f"confidence          : {[round(c.item(),4) for c in pred['confidence']]}")

    # ── Parameter count ───────────────────────────────────────────────────────
    n_params = sum(p.numel() for p in gate.parameters())
    print(f"\nLGG parameters: {n_params:,}")
    print()
    print("Tensor contract:")
    print(f"  Input  : [B, evidence_dim + embedding_dim] = [B, {4+D}]")
    print(f"  Logits : [B, {NUM_TIERS}]")
    print(f"  Probs  : [B, {NUM_TIERS}]  (softmax or Gumbel-soft)")
    print(f"  Hard   : [B]  (int64, tier index 0–{NUM_TIERS-1})")
