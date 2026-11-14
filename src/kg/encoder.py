

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

    # Subtract global max for stability (safe: N=9, typical E~80 edges)
    logits = logits - logits.detach().max()
    exp_vals = torch.exp(logits)
    sum_exp = torch.zeros(num_nodes, dtype=exp_vals.dtype, device=exp_vals.device)
    sum_exp.scatter_add_(0, target_indices, exp_vals)
    return exp_vals / (sum_exp[target_indices] + 1e-9)


# ── HeteroGATLayer ────────────────────────────────────────────────────────────


class HeteroGATLayer(nn.Module):


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

        return self.forward_node_embeddings(x, edge_index, edge_type_ids, edge_weights)


# ── SDM and Compat ────────────────────────────────────────────────────────────


def compute_sdm(
    embeddings: torch.Tensor,
    node_names: list,
    lesion_names: Optional[list] = None,
) -> dict:

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

    return LGovNetEncoder(
        node_feat_dim=node_feat_dim,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        edge_type_dim=edge_type_dim,
        dropout=dropout,
        compat_init_bias_mode=compat_init_bias_mode,
    )
