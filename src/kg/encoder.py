"""
encoder.py
2-layer heterogeneous GAT encoder for LGov-Net (Contribution 1).

Architecture (architecture-lgov-net-v2_rev.txt):

  Input per node:
    x_v = [one-hot(9) || clinical_attr(8)] ∈ R^17  (lesion nodes)
    x_v = [one-hot(9)]                     ∈ R^9   (zone nodes)
    All padded/projected to unified hidden_dim before message passing.

  Edge-type embedding:
    e_r ∈ R^edge_type_dim  per relation type r  (5 types for LGov-Net)

  Attention — heterogeneous, single attention head per layer:
    α_vu = softmax_v( LeakyReLU( a^T [W·x_v ‖ W·x_u ‖ e_r] ) )
             v = target node, u = source node, r = edge type

  Layer 1:  h_v = ELU( Σ_u α_vu · W · x_u )   ∈ R^hidden_dim
  Layer 2:  e_v = GAT²(h_v, G)                  ∈ R^embedding_dim (default 64)

  Directed edges (pathophysiological_mechanism): MA→HE, MA→Hard_EX, MA→Soft_EX
    message flows source→target only (no reverse added).
  Undirected edges: both directions included in edge_index.

Outputs:
  SDM_learned(c_i, c_j) = ‖e_ci - e_cj‖₂      lesion-pair semantic distance
  Compat(c, a)           = σ(e_c^T W_compat e_a) lesion-zone plausibility

No PyTorch Geometric dependency — implemented with pure PyTorch.
Governance Gate, SDGL, and detector are NOT in scope for this module.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Edge type registry ────────────────────────────────────────────────────────

EDGE_TYPE_TO_IDX: dict = {
    "morphological_confusability": 0,
    "pathophysiological_mechanism": 1,
    "spatial_distribution": 2,
    "severity_transition": 3,
    "co_occurrence_pattern": 4,
}
NUM_EDGE_TYPES: int = len(EDGE_TYPE_TO_IDX)

# Node ordering — must match ontology.ALL_NODES indices
NODE_ORDER: list = [
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
LESION_NAMES: list = ["MA", "HE", "Hard_EX", "Soft_EX"]
ZONE_NAMES: list = [
    "Foveal",
    "Parafoveal",
    "Peripapillary",
    "Vascular_Arcades",
    "Peripheral",
]

# ── Compat prior calibration ──────────────────────────────────────────────────
# Additive logit bias for Compat(c, a) = σ(e_c^T W_compat e_a + prior[c, a]).
# Shape: [len(LESION_NAMES), len(ZONE_NAMES)] = [4, 5].
# Values set to +1.0 for clinically HIGH_AFFINITY pairs (from schema.py),
# 0.0 elsewhere.  +1.0 biases sigmoid to σ(1.0) ≈ 0.731 at init even with
# near-zero logit from random W_compat — a minimal, traceable prior nudge.
# Matches HIGH_AFFINITY set in schema.py:
#   Hard_EX ↔ Foveal/Parafoveal, Soft_EX ↔ Peripapillary
#          Foveal Parafo Peripa Vasc Periph
#   MA       0.0    0.0   0.0   0.0   0.0
#   HE       0.0    0.0   0.0   0.0   0.0
#   Hard_EX  1.0    1.0   0.0   0.0   0.0
#   Soft_EX  0.0    0.0   1.0   0.0   0.0
_COMPAT_PRIOR_HIGH_AFFINITY: list = [
    [0.0, 0.0, 0.0, 0.0, 0.0],  # MA
    [0.0, 0.0, 0.0, 0.0, 0.0],  # HE
    [1.0, 1.0, 0.0, 0.0, 0.0],  # Hard_EX → Foveal, Parafoveal
    [0.0, 0.0, 1.0, 0.0, 0.0],  # Soft_EX → Peripapillary
]


# ── Graph preparation ─────────────────────────────────────────────────────────


def normalize_sdca(
    sdm_dict: dict,
    method: str = "log_scaled",
    eps: float = 1e-6,
) -> dict:
    """Normalize raw SDM distances into a bounded range for governance gate input.

    Produces both the raw and normalized SDCA value for each lesion pair.

    Methods:
        "log_scaled":     norm = log(1 + raw) / log(1 + max_raw)
                          Bounded [0, 1].  Compresses large values gently while
                          preserving ordering.  Robust when raw range is unknown
                          before training.  Recommended default for early phases.
        "divide_by_max":  norm = raw / max_raw
                          Simple linear rescaling to [0, 1].
                          Sensitive to outliers if one pair dominates.

    Args:
        sdm_dict: {(ci, cj): float} — output of compute_sdm().
        method:   Normalization method. One of "log_scaled", "divide_by_max".
        eps:      Minimum denominator to prevent division by zero when all
                  distances are near zero (e.g. at random init).

    Returns:
        {(ci, cj): {"raw": float, "norm": float}}
        All "norm" values are in [0.0, 1.0].
    """
    import math

    values = list(sdm_dict.values())
    max_raw = max(values) if values else 0.0

    result: dict = {}

    if method == "log_scaled":
        denom = math.log(1.0 + max_raw) if max_raw > eps else eps
        for pair, raw in sdm_dict.items():
            norm = math.log(1.0 + raw) / denom
            result[pair] = {"raw": raw, "norm": float(norm)}

    elif method == "divide_by_max":
        denom = max_raw if max_raw > eps else eps
        for pair, raw in sdm_dict.items():
            result[pair] = {"raw": raw, "norm": float(raw / denom)}

    else:
        raise ValueError(
            f"Unknown SDCA normalization method: '{method}'. "
            "Choose 'log_scaled' or 'divide_by_max'."
        )

    return result


def build_encoder_inputs(
    kg: dict,
    node_feat_dim: int = 17,
    device: "str | torch.device" = "cpu",
) -> dict:
    """Convert a BuiltGraph dict (from builder.py) to encoder-ready tensors.

    Handles feature padding (zone nodes dim=9 padded to node_feat_dim),
    undirected→bidirectional edge expansion, and directed edge preservation.

    Args:
        kg:            Output dict from builder.build_graph_from_artifacts().
        node_feat_dim: Unified input dim after padding. Must be >= max node
                       feature dim (17 for LGov-Net lesion nodes).
        device:        Target PyTorch device string or object.

    Returns:
        Dict with keys:
          "x"             : Tensor[N, node_feat_dim]  padded feature matrix
          "edge_index"    : Tensor[2, E]              expanded edge indices
          "edge_type_ids" : Tensor[E]  int64          edge type per edge
          "edge_weights"  : Tensor[E]  float32        KG init weights per edge
          "node_names"    : list[str]                 name at each row index
    """
    device = torch.device(device)
    node_features: dict = kg["node_features"]  # {name: list[float]}
    nodes: list = kg["nodes"]  # sorted by idx

    # ── Feature matrix (zero-pad short vectors) ───────────────────────────────
    x_rows: list = []
    ordered_names: list = []
    for node in sorted(nodes, key=lambda n: n["idx"]):
        feat = list(node_features[node["name"]])
        if len(feat) < node_feat_dim:
            feat = feat + [0.0] * (node_feat_dim - len(feat))
        x_rows.append(feat[:node_feat_dim])
        ordered_names.append(node["name"])

    x = torch.tensor(x_rows, dtype=torch.float32, device=device)  # [N, node_feat_dim]

    # ── Edge lists ────────────────────────────────────────────────────────────
    src_list: list = []
    dst_list: list = []
    type_list: list = []
    weight_list: list = []

    for etype, edge_list in kg["edges"].items():
        type_id = EDGE_TYPE_TO_IDX[etype]
        for e in edge_list:
            s, d, w = e["src"], e["dst"], e["weight"]
            directed = e["directed"]
            src_list.append(s)
            dst_list.append(d)
            type_list.append(type_id)
            weight_list.append(w)
            if not directed:
                # undirected → add reverse direction for message passing
                src_list.append(d)
                dst_list.append(s)
                type_list.append(type_id)
                weight_list.append(w)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long, device=device)
    edge_type_ids = torch.tensor(type_list, dtype=torch.long, device=device)
    edge_weights = torch.tensor(weight_list, dtype=torch.float32, device=device)

    return {
        "x": x,
        "edge_index": edge_index,
        "edge_type_ids": edge_type_ids,
        "edge_weights": edge_weights,
        "node_names": ordered_names,
    }


# ── Attention utility ─────────────────────────────────────────────────────────


def _scatter_softmax(
    logits: torch.Tensor,
    target_indices: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Numerically-stable per-target-node softmax over edge scores.

    Args:
        logits:         [E] raw attention scores.
        target_indices: [E] destination node index per edge.
        num_nodes:      Total node count N.

    Returns:
        [E] attention weights that sum to 1 per target node.
    """
    # Subtract global max for stability (safe: N=9, typical E~80 edges)
    logits = logits - logits.detach().max()
    exp_vals = torch.exp(logits)
    sum_exp = torch.zeros(num_nodes, dtype=exp_vals.dtype, device=exp_vals.device)
    sum_exp.scatter_add_(0, target_indices, exp_vals)
    return exp_vals / (sum_exp[target_indices] + 1e-9)


# ── HeteroGATLayer ────────────────────────────────────────────────────────────


class HeteroGATLayer(nn.Module):
    """Single layer of heterogeneous graph attention.

    Implements the attention equation from the architecture spec:
        α_vu = softmax_v( LeakyReLU( a^T [W·x_v ‖ W·x_u ‖ e_r] ) )
        h_v  = Σ_u α_vu · (W · x_u)

    where v is the target node, u is the source node, and r is the edge type.
    Attention is modulated by KG edge weights (clinical priors) when supplied.

    Args:
        in_dim:           Input feature dimension.
        out_dim:          Output feature dimension.
        edge_type_dim:    Dimension of edge-type embedding e_r.
        num_edge_types:   Number of distinct edge types (5 for LGov-Net).
        leaky_relu_slope: Negative slope for LeakyReLU in attention scoring.
        dropout:          Dropout applied to normalized attention weights.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        edge_type_dim: int,
        num_edge_types: int = NUM_EDGE_TYPES,
        leaky_relu_slope: float = 0.2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.slope = leaky_relu_slope
        self.dropout = dropout

        # Node feature projection W
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        # Edge type embedding table e_r
        self.edge_emb = nn.Embedding(num_edge_types, edge_type_dim)
        # Attention scoring vector a: maps [W·x_v ‖ W·x_u ‖ e_r] → scalar
        self.attn_vec = nn.Linear(2 * out_dim + edge_type_dim, 1, bias=False)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.attn_vec.weight)
        nn.init.normal_(self.edge_emb.weight, std=0.02)

    def forward(
        self,
        x: torch.Tensor,  # [N, in_dim]
        edge_index: torch.Tensor,  # [2, E]
        edge_type_ids: torch.Tensor,  # [E] int64
        edge_weights: Optional[torch.Tensor],  # [E] float32 or None
    ) -> torch.Tensor:
        """Run one heterogeneous GAT layer.

        Args:
            x:             Node feature matrix [N, in_dim].
            edge_index:    Edge connectivity [2, E]: row 0 = src, row 1 = dst.
            edge_type_ids: Integer edge type index per edge [E].
            edge_weights:  Optional KG prior weights [E]. When supplied, raw
                           attention scores are multiplied by these weights
                           before softmax so clinical priors influence routing.

        Returns:
            Updated node representations [N, out_dim].
        """
        N = x.size(0)
        src_idx, dst_idx = edge_index[0], edge_index[1]

        Wx = self.W(x)  # [N, out_dim]
        e_r = self.edge_emb(edge_type_ids)  # [E, edge_type_dim]

        # Concatenate projected target, projected source, edge embedding
        attn_in = torch.cat([Wx[dst_idx], Wx[src_idx], e_r], dim=-1)  # [E, ...]
        attn_raw = self.attn_vec(attn_in).squeeze(-1)  # [E]
        attn_raw = F.leaky_relu(attn_raw, negative_slope=self.slope)

        # Modulate by KG edge weights (clinical priors bias the attention)
        if edge_weights is not None:
            attn_raw = attn_raw * edge_weights

        attn = _scatter_softmax(attn_raw, dst_idx, N)  # [E]
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        # Aggregate messages: h_v = Σ_u α_vu · W · x_u
        msgs = Wx[src_idx] * attn.unsqueeze(-1)  # [E, out_dim]
        out = torch.zeros(N, self.out_dim, dtype=x.dtype, device=x.device)
        out.scatter_add_(0, dst_idx.unsqueeze(-1).expand_as(msgs), msgs)
        return out


# ── LGovNetEncoder ────────────────────────────────────────────────────────────


class LGovNetEncoder(nn.Module):
    """2-layer heterogeneous GAT encoder for LGov-Net KG (Contribution 1).

    Produces embedding_dim-dimensional representations for all 9 KG nodes
    given the padded feature matrix and edge structure from build_encoder_inputs().

    Encoder pipeline:
        input_proj  (Linear, no bias)  → ELU
        HeteroGATLayer-1 (hidden_dim → hidden_dim) → ELU → dropout
        HeteroGATLayer-2 (hidden_dim → embedding_dim)

    Also holds W_compat (learnable [embed_dim × embed_dim] matrix) used by
    compute_compat() to score lesion-zone plausibility.

    Args:
        node_feat_dim:    Input dim after padding (17 for LGov-Net).
        hidden_dim:       Width of Layer 1 output and input_proj target (default 64).
        embedding_dim:    Final node embedding dimension e_v (default 64).
        edge_type_dim:    Edge-type embedding dimension e_r (default 16).
        num_edge_types:   Number of KG edge types (5).
        dropout:          Dropout on attention weights and between GAT layers.
        leaky_relu_slope: Negative slope for LeakyReLU inside attention.
        compat_init_bias_mode: Initial logit bias for Compat scores.
            "high_affinity_prior": add +1.0 bias to HIGH_AFFINITY lesion-zone
                pairs (Hard_EX↔Foveal/Parafoveal, Soft_EX↔Peripapillary).
                Gives σ(1.0) ≈ 0.731 for high-affinity pairs at init vs 0.5
                for all others.  Traceable clinical prior without hardcoding.
            "none": no bias, pure bilinear form (original behavior).
    """

    def __init__(
        self,
        node_feat_dim: int = 17,
        hidden_dim: int = 64,
        embedding_dim: int = 64,
        edge_type_dim: int = 16,
        num_edge_types: int = NUM_EDGE_TYPES,
        dropout: float = 0.1,
        leaky_relu_slope: float = 0.2,
        compat_init_bias_mode: str = "high_affinity_prior",
    ) -> None:
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout

        # Project heterogeneous input features to a common hidden_dim
        self.input_proj = nn.Linear(node_feat_dim, hidden_dim, bias=False)

        # Layer 1
        self.gat1 = HeteroGATLayer(
            in_dim=hidden_dim,
            out_dim=hidden_dim,
            edge_type_dim=edge_type_dim,
            num_edge_types=num_edge_types,
            leaky_relu_slope=leaky_relu_slope,
            dropout=dropout,
        )
        # Layer 2
        self.gat2 = HeteroGATLayer(
            in_dim=hidden_dim,
            out_dim=embedding_dim,
            edge_type_dim=edge_type_dim,
            num_edge_types=num_edge_types,
            leaky_relu_slope=leaky_relu_slope,
            dropout=dropout,
        )

        # Compatibility weight matrix for Compat(c, a) = σ(e_c^T W_compat e_a + prior)
        self.W_compat = nn.Parameter(torch.empty(embedding_dim, embedding_dim))

        # Compat prior bias: non-trainable, shape [4, 5] (lesion × zone).
        # Registered as a buffer so it moves with the model and is saved in
        # state_dict, but is NOT a learnable parameter.
        if compat_init_bias_mode == "high_affinity_prior":
            prior_data = torch.tensor(
                _COMPAT_PRIOR_HIGH_AFFINITY, dtype=torch.float32
            )  # [4, 5]
        else:
            prior_data = torch.zeros(len(LESION_NAMES), len(ZONE_NAMES))
        self.register_buffer("compat_prior", prior_data)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.xavier_uniform_(self.W_compat)

    def forward_node_embeddings(
        self,
        x: torch.Tensor,  # [N, node_feat_dim]
        edge_index: torch.Tensor,  # [2, E]
        edge_type_ids: torch.Tensor,  # [E] int64
        edge_weights: Optional[torch.Tensor] = None,  # [E] float32
    ) -> torch.Tensor:
        """Compute all N=9 node embeddings from padded features and KG structure.

        Args:
            x:             Node feature matrix [N, node_feat_dim].
            edge_index:    [2, E] — row 0 src, row 1 dst (undirected already
                           expanded to both directions by build_encoder_inputs).
            edge_type_ids: [E] integer edge type codes.
            edge_weights:  Optional KG prior weights [E]. Supplied via
                           build_encoder_inputs → "edge_weights".

        Returns:
            [N, embedding_dim] — e_v for all 9 KG nodes.
            Indices match build_encoder_inputs()["node_names"].
        """
        h = F.elu(self.input_proj(x))  # [N, hidden_dim]
        h = F.elu(
            self.gat1(h, edge_index, edge_type_ids, edge_weights)
        )  # [N, hidden_dim]
        h = F.dropout(h, p=self.dropout_rate, training=self.training)
        e = self.gat2(h, edge_index, edge_type_ids, edge_weights)  # [N, embed_dim]
        return e

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type_ids: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """nn.Module standard interface — delegates to forward_node_embeddings."""
        return self.forward_node_embeddings(x, edge_index, edge_type_ids, edge_weights)


# ── SDM and Compat ────────────────────────────────────────────────────────────


def compute_sdm(
    embeddings: torch.Tensor,
    node_names: list,
    lesion_names: Optional[list] = None,
) -> dict:
    """Compute pairwise SDM_learned between all lesion node pairs.

    SDM_learned(c_i, c_j) = ‖e_ci - e_cj‖₂

    Clinical interpretation: SDM(MA, HE) < SDM(MA, Hard_EX) because
    confusing MA↔HE does not change ICDR severity grade while confusing
    MA↔Hard_EX does (severity_transition edge weights 0.0 vs 1.0).

    Args:
        embeddings:   [N, embedding_dim] tensor from forward_node_embeddings().
        node_names:   Ordered list of node names matching row indices of
                      embeddings (from build_encoder_inputs()["node_names"]).
        lesion_names: Subset of lesion class names. Defaults to all 4.

    Returns:
        {(c_i, c_j): float} — all unique unordered pairs (i < j by lesion index).
        Floats allow JSON serialization and downstream numpy/scalar use.
    """
    if lesion_names is None:
        lesion_names = LESION_NAMES

    name_to_idx = {name: i for i, name in enumerate(node_names)}
    sdm: dict = {}

    for i in range(len(lesion_names)):
        for j in range(i + 1, len(lesion_names)):
            ci, cj = lesion_names[i], lesion_names[j]
            ei = embeddings[name_to_idx[ci]]
            ej = embeddings[name_to_idx[cj]]
            sdm[(ci, cj)] = torch.norm(ei - ej, p=2).item()

    return sdm


def compute_compat(
    embeddings: torch.Tensor,
    W_compat: torch.Tensor,
    node_names: list,
    lesion_names: Optional[list] = None,
    zone_names: Optional[list] = None,
    compat_prior: Optional[torch.Tensor] = None,
) -> dict:
    """Compute anatomical plausibility Compat(c, a) for all lesion-zone pairs.

    Compat(c, a) = σ( e_c^T · W_compat · e_a + prior[c_idx, a_idx] )

    The optional `compat_prior` (shape [4, 5]) adds a fixed logit bias before
    sigmoid.  Set to encoder.compat_prior to activate HIGH_AFFINITY calibration.
    When None, behaves exactly as the original bilinear form (prior=0 everywhere).

    Used downstream as APS (Anatomical Plausibility Score) per detected bbox:
    a bbox in zone a claiming class c gets APS = Compat(c, a).
    APS < 0.5 triggers a downgrade in the governance gate.

    Args:
        embeddings:    [N, embedding_dim] from forward_node_embeddings().
        W_compat:      [embedding_dim, embedding_dim] — encoder.W_compat parameter.
        node_names:    Ordered list matching embedding row indices.
        lesion_names:  Lesion classes to score. Defaults to all 4.
        zone_names:    Zone names to score. Defaults to all 5.
        compat_prior:  Optional [len(lesion_names), len(zone_names)] additive bias.
                       Pass encoder.compat_prior to apply HIGH_AFFINITY calibration.

    Returns:
        {(lesion_name, zone_name): float} — sigmoid scores in [0, 1].
    """
    if lesion_names is None:
        lesion_names = LESION_NAMES
    if zone_names is None:
        zone_names = ZONE_NAMES

    name_to_idx = {name: i for i, name in enumerate(node_names)}
    compat: dict = {}

    for c_idx, lesion in enumerate(lesion_names):
        e_c = embeddings[name_to_idx[lesion]]  # [embed_dim]
        for a_idx, zone in enumerate(zone_names):
            e_a = embeddings[name_to_idx[zone]]  # [embed_dim]
            logit = e_c @ W_compat @ e_a
            if compat_prior is not None:
                logit = logit + compat_prior[c_idx, a_idx]
            score = torch.sigmoid(logit)
            compat[(lesion, zone)] = score.item()

    return compat


# ── Factory function ──────────────────────────────────────────────────────────


def build_gat_encoder(
    node_feat_dim: int = 17,
    hidden_dim: int = 64,
    embedding_dim: int = 64,
    edge_type_dim: int = 16,
    dropout: float = 0.1,
    compat_init_bias_mode: str = "high_affinity_prior",
) -> LGovNetEncoder:
    """Return a freshly initialized LGovNetEncoder with Xavier-initialized weights.

    Args:
        node_feat_dim:         Input dimension after padding (17 for LGov-Net).
        hidden_dim:            Width of intermediate representations.
        embedding_dim:         Output embedding dimension (must match SDGL config).
        edge_type_dim:         Dimension of edge-type embedding vectors.
        dropout:               Dropout on attention weights and between layers.
        compat_init_bias_mode: "high_affinity_prior" | "none".
                               Controls additive logit bias in Compat scoring.
                               See LGovNetEncoder docstring for details.

    Returns:
        LGovNetEncoder module in training mode.
    """
    return LGovNetEncoder(
        node_feat_dim=node_feat_dim,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        edge_type_dim=edge_type_dim,
        dropout=dropout,
        compat_init_bias_mode=compat_init_bias_mode,
    )
