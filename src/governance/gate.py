"""
gate.py
Learned Governance Gate (LGG) for LGov-Net (Contribution 2).

Takes per-bbox evidence and routes each bbox to one of 4 governance tiers
(R1–R4) using a 2-layer MLP with optional Gumbel-Softmax for differentiable
routing during training.

Architecture:
  input:   g(b_j) = [SAS, SDCA_norm, APS, PSS, e_pred]  ∈ R^{4 + d}
  hidden:  Linear(4+d → hidden_dim) → LayerNorm → ReLU → Dropout
  output:  Linear(hidden_dim → 4)  →  tier logits  ∈ R^4
  routing: softmax(logits)                           →  tier probs  ∈ Δ³
           Gumbel-Softmax(logits) during training    →  soft one-hot ∈ Δ³
           argmax(logits) during inference            →  hard tier ∈ {0,1,2,3}

Tier semantics (interpretation only — not enforced here):
  R1 — Accept: high SAS + high APS + stable PSS → accepted as-is
  R2 — Soft-correct: moderate evidence → apply KG-guided correction
  R3 — Escalate: conflicted evidence → defer to radiologist / second head
  R4 — Reject: very low SAS + low APS → suppress / discard bbox

  NOTE: which tier does what is defined in the paper / downstream postprocessing.
  This module only produces the routing distribution. It does NOT implement
  any correction, filtering, or escalation logic.

e_pred construction:
  e_pred(b_j) = Σ_c  ensemble_probs[c] · e_c
  where e_c is the KG embedding of lesion class c (from LGovNetEncoder).
  This soft-weighted embedding encodes the distributional class evidence in
  the KG embedding space.  Helper: build_pred_embedding().

Usage:
  from src.governance.gate import LGG, build_pred_embedding, concat_gate_input

  # Full model (default — Contribution 2 as designed):
  gate = LGG(evidence_dim=4, embedding_dim=64)
  e_pred = build_pred_embedding(ensemble_probs_batch, lesion_embeddings)  # [B, 64]
  gate_input = concat_gate_input(evidence_vectors, e_pred)                # [B, 68]
  out = gate(gate_input)

  # Ablation: gate without KG embedding (C2 ablation "gate without KG"):
  gate_no_kg = LGG(evidence_dim=4, embedding_dim=0)
  gate_input_no_kg = concat_gate_input(evidence_vectors, e_pred, use_kg_embedding=False)
  out_ablation = gate_no_kg(gate_input_no_kg)                             # [B, 4]

  # out["tier_probs"]  → Tensor[B, 4]
  # out["tier_hard"]   → Tensor[B]   (inference)

Dependencies:
  - PyTorch   (no PyG)
  - evidence.py supplies evidence_vector=(SAS, SDCA_norm, APS, PSS)
  - encoder.py supplies lesion embeddings and e_pred
  - contracts.py is NOT imported here (gate is contract-agnostic)
"""

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
    """Compute the soft-weighted predicted class embedding e_pred for a batch.

    e_pred(b_j) = Σ_c  P(c | b_j) · e_c

    where e_c is the KG embedding for lesion class c and P(c | b_j) comes
    from the detector's ensemble_probs (averaged over all heads).

    This produces a single embedding in the KG space that is a convex
    combination of the 4 lesion embeddings, weighted by how confident the
    detector is in each class.  It encodes distributional evidence rather
    than committing to a hard class before the gate decision.

    Args:
        ensemble_probs_batch: Tensor[B, C] — per-class probabilities, C=4.
                              Must sum to 1 per row.
                              Row ordering must match _LESION_CLASSES.
        lesion_embeddings:    Tensor[C, d] — KG embeddings for the 4 lesion
                              nodes, in _LESION_CLASSES order.
                              Typically rows 0–3 of LGovNetEncoder output.

    Returns:
        Tensor[B, d] — one predicted embedding per bbox in the batch.
    """
    # [B, C] @ [C, d] → [B, d]
    return ensemble_probs_batch @ lesion_embeddings


def concat_gate_input(
    evidence_vectors: torch.Tensor,
    e_pred: torch.Tensor,
    use_kg_embedding: bool = True,
) -> torch.Tensor:
    """Concatenate scalar evidence signals and (optionally) predicted embedding.

    Produces the full gate input vector:
        use_kg_embedding=True  (default):
            gate_input = [SAS, SDCA_norm, APS, PSS, e_pred]  ∈ R^{4 + d}
        use_kg_embedding=False  (ablation — C2 "gate without KG"):
            gate_input = [SAS, SDCA_norm, APS, PSS]          ∈ R^4

    The ablation mode is used to test whether the KG embedding component
    (e_pred) contributes meaningfully to routing beyond the scalar signals.
    It corresponds to the "gate without KG" condition in the C2 ablation table.

    IMPORTANT: when use_kg_embedding=False, the LGG must be instantiated with
    embedding_dim=0  (i.e. LGG(embedding_dim=0)) so that gate_dim = 4 matches
    the input width.  Using a model with embedding_dim > 0 against this output
    will raise a shape mismatch at the first linear layer.

    Args:
        evidence_vectors:  Tensor[B, 4] — [SAS, SDCA_norm, APS, PSS] per bbox.
        e_pred:            Tensor[B, d] — predicted embedding per bbox.
                           Ignored when use_kg_embedding=False.
        use_kg_embedding:  If True (default), append e_pred.
                           If False, return evidence_vectors only (ablation).

    Returns:
        Tensor[B, 4+d] when use_kg_embedding=True,
        Tensor[B, 4]   when use_kg_embedding=False.
    """
    if use_kg_embedding:
        return torch.cat([evidence_vectors, e_pred], dim=-1)
    return evidence_vectors


def gumbel_soft_routing(
    logits: torch.Tensor,
    temperature: float = 1.0,
    hard: bool = False,
) -> torch.Tensor:
    """Differentiable soft routing via Gumbel-Softmax.

    During training, Gumbel noise injected into logits allows gradients to
    flow through the discrete tier selection.  Lower temperature sharpens
    the distribution toward one-hot.

    Args:
        logits:      Tensor[B, 4] — raw gate output scores.
        temperature: Gumbel-Softmax temperature τ > 0.  Typically annealed
                     from 1.0 → 0.1 during training.  1.0 = soft; 0.1 ≈ hard.
        hard:        If True, use straight-through estimator: forward pass
                     returns one-hot, backward pass uses soft gradients.

    Returns:
        Tensor[B, 4] — soft (or straight-through hard) routing probabilities.
    """
    return F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=-1)


def get_gumbel_tau(
    step: int,
    tau_start: float = 1.0,
    tau_end: float = 0.1,
    tau_anneal_steps: int = 10_000,
) -> float:
    """Return the Gumbel-Softmax temperature τ at a given training step.

    Uses linear annealing from tau_start down to tau_end over tau_anneal_steps.
    After tau_anneal_steps the temperature is clamped at tau_end.

    Formula:
        τ(t) = tau_start + (tau_end - tau_start) * min(t / tau_anneal_steps, 1)

    Args:
        step:             Current training step (0-indexed).
        tau_start:        Initial temperature (default 1.0 — soft).
        tau_end:          Final temperature after annealing (default 0.1 — near-hard).
        tau_anneal_steps: Number of steps over which to anneal (default 10 000).
                          Must be > 0.

    Returns:
        float in [tau_end, tau_start].
    """
    if tau_anneal_steps <= 0:
        raise ValueError(f"tau_anneal_steps must be > 0, got {tau_anneal_steps}")
    progress = min(step / tau_anneal_steps, 1.0)
    return tau_start + (tau_end - tau_start) * progress


def predict_tier(
    tier_probs: torch.Tensor,
    tier_names: Optional[List[str]] = None,
) -> dict:
    """Convert tier probabilities to hard tier assignments for inference.

    Args:
        tier_probs:  Tensor[B, 4] — softmax probabilities per tier.
        tier_names:  List of tier label strings. Defaults to TIER_NAMES.

    Returns:
        {
            "tier_hard":    Tensor[B] int64 — argmax tier index per bbox.
            "tier_labels":  list[str] length B — tier name per bbox.
            "confidence":   Tensor[B] — max prob per bbox (gate confidence).
        }
    """
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
    """Learned Governance Gate — 2-layer MLP routing bbox to tier R1–R4.

    Takes the concatenated gate input [evidence_vector || e_pred] and
    produces a routing distribution over 4 governance tiers.

    Architecture:
        fc1:   Linear(gate_dim → hidden_dim)
        norm1: LayerNorm(hidden_dim)
        act:   ReLU
        drop:  Dropout(dropout)
        fc2:   Linear(hidden_dim → 4)

    gate_dim = evidence_dim + embedding_dim = 4 + d (default 4 + 64 = 68).

    Args:
        evidence_dim:   Dimension of the scalar evidence vector (default 4).
                        Must match len(evidence_vector) from evidence.py.
        embedding_dim:  Dimension of e_pred (= LGovNetEncoder.embedding_dim,
                        default 64).
        hidden_dim:     Width of the hidden layer (default 128).
        dropout:        Dropout rate applied after activation (default 0.1).
    """

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
        """Standard forward pass — produces logits and softmax probabilities.

        Use this during training (with Gumbel routing) or for computing loss.

        Args:
            gate_input: Tensor[B, evidence_dim + embedding_dim]
                        Built via concat_gate_input(evidence_vectors, e_pred).

        Returns:
            {
                "tier_logits": Tensor[B, 4]  — raw pre-softmax scores.
                "tier_probs":  Tensor[B, 4]  — softmax probabilities.
                "tier_hard":   Tensor[B]     — argmax tier index (int64).
                "tier_names":  list[str]     — ["R1","R2","R3","R4"].
            }
        """
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
        """Forward pass with Gumbel-Softmax for differentiable routing.

        Use during training when gradients must flow through the discrete
        tier assignment.  Set hard=True for straight-through estimator.

        Args:
            gate_input:  Tensor[B, gate_dim].
            temperature: Gumbel-Softmax temperature τ.  Anneal toward 0 to
                         approximate hard argmax while keeping gradients.
            hard:        Straight-through mode: forward = one-hot,
                         backward = soft gradient.

        Returns:
            Same keys as forward(), with "tier_probs" replaced by Gumbel
            soft/hard one-hot.  Additionally includes:
                "gumbel_soft": Tensor[B, 4] — raw Gumbel-Softmax output.
        """
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
    """Return a freshly initialised LGG module.

    Args:
        evidence_dim:   Length of scalar evidence vector (default 4).
        embedding_dim:  Dimension of e_pred from LGovNetEncoder (default 64).
        hidden_dim:     Hidden layer width (default 128).
        dropout:        Dropout rate (default 0.1).

    Returns:
        LGG in training mode.
    """
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
