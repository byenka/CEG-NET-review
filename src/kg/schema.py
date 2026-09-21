"""
schema.py
Knowledge Graph schema for LGov-Net — static structural definition.

Defines G = (V, E) with 9 nodes and 5 edge types.

Fixed vs Trainable edge summary:
  Edge type                    Fixed/Trainable  Directed
  ---------------------------  ---------------  --------
  morphological_confusability  FIXED            No
  pathophysiological_mechanism TRAINABLE weight No topo.; DIRECTED
  spatial_distribution         TRAINABLE        No (bipartite)
  severity_transition          FIXED            No
  co_occurrence_pattern        TRAINABLE        No

Single source of truth for numeric priors: src/data/ontology.py.
This file constructs the graph objects; it does not duplicate raw values.
"""

from dataclasses import dataclass, field

# Minimum operational weight for message-passing edges.
# Clinical cost of 0.0 (e.g. severity_transition MA↔HE) is a valid semantic
# fact but would create a dead edge in GAT attention modulation.
# init_weight = max(clinical_cost, MIN_EDGE_WEIGHT_EPSILON).
# Sourced from configs/kg.yaml: min_edge_weight_epsilon.
MIN_EDGE_WEIGHT_EPSILON: float = 0.05


# ── Dataclasses ───────────────────────────────────────────────────────────────


@dataclass
class NodeSpec:
    """A single node in the LGov-Net KG.

    Attributes:
        idx:        Global node index (0–8). Must match ontology.ALL_NODES.
        name:       Canonical node name, e.g. 'MA', 'Foveal'.
        node_type:  'lesion' for C=4 lesion nodes, 'zone' for ETDRS zones.
        feat_dim:   Dimension of x_v = [one-hot(9) || clinical_attr(8)].
                    9 for zone nodes (no clinical attrs beyond one-hot);
                    17 for lesion nodes.
    """

    idx: int
    name: str
    node_type: str  # "lesion" | "zone"
    feat_dim: int = 17


@dataclass
class EdgeSpec:
    """A single edge (or undirected edge pair) in the LGov-Net KG.

    For undirected edges, src < dst by convention. Symmetry (adding the
    reverse direction) is the caller's responsibility during GNN message
    passing — only ONE EdgeSpec is stored here.

    For directed edges (pathophysiological_mechanism) src is the upstream
    node (MA) and message flows src → dst only.

    Attributes:
        src:          Source node index.
        dst:          Destination node index.
        src_name:     Source node name (for readability / JSON output).
        dst_name:     Destination node name.
        edge_type:    One of the 5 LGov-Net edge type strings.
        directed:     True only for pathophysiological_mechanism.
        trainable:    False = weight frozen at init_weight throughout training.
                      True  = weight is a learnable parameter, initialised to
                              init_weight.
        init_weight:  Starting weight value.
                      Fixed edges: clinically derived (ICDR / kappa studies).
                      Trainable edges: placeholder 1.0, or dataset-derived
                      value set later via build_dataset_stats.py.
    """

    src: int
    dst: int
    src_name: str
    dst_name: str
    edge_type: str
    directed: bool
    trainable: bool
    init_weight: float
    clinical_cost: float = 0.0
    # clinical_cost: the original ICDR-derived semantic value (e.g. 0.0 for
    #   MA↔HE severity_transition). Preserved for SDM/SDGL computation.
    # init_weight:   operational weight used for message passing (>= epsilon).
    #   For severity_transition: max(clinical_cost, MIN_EDGE_WEIGHT_EPSILON).
    #   For all other edge types: equals clinical_cost or dataset-derived value.


# ── Node construction ─────────────────────────────────────────────────────────


def build_default_nodes() -> list:
    """Return ordered list of 9 NodeSpec objects.

    Order and indices are identical to ontology.ALL_NODES:
      0 MA | 1 HE | 2 Hard_EX | 3 Soft_EX |
      4 Foveal | 5 Parafoveal | 6 Peripapillary |
      7 Vascular_Arcades | 8 Peripheral

    Zone nodes use feat_dim=9 (one-hot only; no clinical attr vector).
    Lesion nodes use feat_dim=17 (one-hot(9) + clinical_attr(8)).

    Returns:
        List of NodeSpec, length 9.
    """
    from lgovnet.src.data.ontology import LESION_NODES, ZONE_NODES

    nodes: list = []

    for name, idx in sorted(LESION_NODES.items(), key=lambda kv: kv[1]):
        nodes.append(NodeSpec(idx=idx, name=name, node_type="lesion", feat_dim=17))

    for name, idx in sorted(ZONE_NODES.items(), key=lambda kv: kv[1]):
        nodes.append(NodeSpec(idx=idx, name=name, node_type="zone", feat_dim=9))

    nodes.sort(key=lambda n: n.idx)
    assert len(nodes) == 9, f"Expected 9 nodes, got {len(nodes)}"
    return nodes


# ── Edge construction ─────────────────────────────────────────────────────────


def build_default_edges() -> list:
    """Return complete list of EdgeSpec objects across all 5 edge types.

    Returns:
        List of EdgeSpec. Undirected edges have ONE entry (src < dst).
        Directed edges have ONE entry (src → dst).

    Trainable edges with init_weight=1.0 are uniform placeholders until
    tools/build_dataset_stats.py computes dataset-specific values.
    """
    from lgovnet.src.data.ontology import (
        ALL_NODES,
        LESION_NODES,
        ZONE_NODES,
        MORPHOLOGICAL_CONFUSABILITY,
        PATHOPHYSIOLOGICAL_MECHANISM,
        SEVERITY_TRANSITION,
    )

    edges: list = []

    def _idx(name: str) -> int:
        return ALL_NODES[name]

    # ── 1. morphological_confusability: UNDIRECTED, FIXED ─────────────────
    # Source: inter-observer kappa studies.
    for pair, weight in MORPHOLOGICAL_CONFUSABILITY.items():
        a, b = sorted(pair, key=lambda n: ALL_NODES[n])
        edges.append(
            EdgeSpec(
                src=_idx(a),
                dst=_idx(b),
                src_name=a,
                dst_name=b,
                edge_type="morphological_confusability",
                directed=False,
                trainable=False,
                init_weight=weight,
            )
        )

    # ── 2. pathophysiological_mechanism: DIRECTED, TRAINABLE weight ────────
    # Topology fixed (MA is upstream for all three).
    # Weights are trainable — initial values from architecture spec.
    for (src_name, dst_name), weight in PATHOPHYSIOLOGICAL_MECHANISM.items():
        edges.append(
            EdgeSpec(
                src=_idx(src_name),
                dst=_idx(dst_name),
                src_name=src_name,
                dst_name=dst_name,
                edge_type="pathophysiological_mechanism",
                directed=True,
                trainable=True,
                init_weight=weight,
            )
        )

    # ── 3. spatial_distribution: BIPARTITE lesion↔zone, TRAINABLE ─────────
    # All 4×5=20 bipartite pairs included. init_weight=1.0 (uniform).
    # High-affinity pairs (HardEX↔Parafoveal/Foveal, SoftEX↔Peripapillary)
    # are given init_weight=2.0 to bias initialisation toward clinical priors.
    HIGH_AFFINITY: set = {
        frozenset({"Hard_EX", "Parafoveal"}),
        frozenset({"Hard_EX", "Foveal"}),
        frozenset({"Soft_EX", "Peripapillary"}),
    }
    for l_name in sorted(LESION_NODES.keys(), key=lambda n: LESION_NODES[n]):
        for z_name in sorted(ZONE_NODES.keys(), key=lambda n: ZONE_NODES[n]):
            w = 2.0 if frozenset({l_name, z_name}) in HIGH_AFFINITY else 1.0
            edges.append(
                EdgeSpec(
                    src=_idx(l_name),
                    dst=_idx(z_name),
                    src_name=l_name,
                    dst_name=z_name,
                    edge_type="spatial_distribution",
                    directed=False,
                    trainable=True,
                    init_weight=w,
                )
            )

    # ── 4. severity_transition: UNDIRECTED, FIXED ─────────────────────────
    # Source: ICDR 5-level scale + 4:2:1 rule.
    # clinical_cost stores the true ICDR semantic cost (may be 0.0 for MA↔HE).
    # init_weight is clamped to MIN_EDGE_WEIGHT_EPSILON so the edge remains
    # active in GAT message passing even when clinical_cost == 0.0.
    for pair, weight in SEVERITY_TRANSITION.items():
        a, b = sorted(pair, key=lambda n: ALL_NODES[n])
        edges.append(
            EdgeSpec(
                src=_idx(a),
                dst=_idx(b),
                src_name=a,
                dst_name=b,
                edge_type="severity_transition",
                directed=False,
                trainable=False,
                clinical_cost=float(weight),
                init_weight=max(float(weight), MIN_EDGE_WEIGHT_EPSILON),
            )
        )

    # ── 5. co_occurrence_pattern: UNDIRECTED, TRAINABLE ───────────────────
    # All unique lesion pairs. init_weight=1.0; overwritten by
    # tools/build_dataset_stats.py with OIA-DDR co-occurrence values.
    lesion_names = sorted(LESION_NODES.keys(), key=lambda n: LESION_NODES[n])
    for i, a in enumerate(lesion_names):
        for b in lesion_names[i + 1 :]:
            edges.append(
                EdgeSpec(
                    src=_idx(a),
                    dst=_idx(b),
                    src_name=a,
                    dst_name=b,
                    edge_type="co_occurrence_pattern",
                    directed=False,
                    trainable=True,
                    init_weight=1.0,
                )
            )

    return edges


# ── Export ────────────────────────────────────────────────────────────────────


def export_nodes_edges(
    nodes_path: "str | None" = None,
    edges_path: "str | None" = None,
) -> "tuple[list, list]":
    """Build and optionally save nodes and edges to JSON files.

    Args:
        nodes_path: Output path for nodes.json. Skipped if None.
        edges_path: Output path for edges.json. Skipped if None.

    Returns:
        (nodes, edges) — lists of dicts (JSON-serialisable).
    """
    import dataclasses
    import json
    from pathlib import Path

    nodes = build_default_nodes()
    edges = build_default_edges()

    nodes_dicts = [dataclasses.asdict(n) for n in nodes]
    edges_dicts = [dataclasses.asdict(e) for e in edges]

    if nodes_path is not None:
        p = Path(nodes_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(nodes_dicts, f, indent=2)

    if edges_path is not None:
        p = Path(edges_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(edges_dicts, f, indent=2)

    return nodes_dicts, edges_dicts


def build_adjacency_dict() -> dict:
    """Return {edge_type: [(src_idx, dst_idx, init_weight), ...]} for all 5 types.

    Convenience wrapper over build_default_edges() for downstream use
    (e.g. GNN message passing setup).

    Returns:
        Dict keyed by edge type string, values are sorted lists of
        (src, dst, init_weight) tuples.
    """
    adj: dict = {}
    for e in build_default_edges():
        adj.setdefault(e.edge_type, []).append((e.src, e.dst, e.init_weight))
    return adj


def build_clinical_prior_sdm() -> dict:
    """Build a grading-aware clinical-prior SDM from ICDR severity_transition weights.

    Contribution 1 claim: 'grading-aware SDM' means semantic distance between
    lesion classes is proportional to the ICDR grade cost of confusing them.
    This function implements that claim deterministically — no encoder training
    required, fully traceable to the ICDR 5-level scale.

    Formula:
        SDM_clinical(ci, cj) = ICDR_cost(ci, cj) / max(ICDR_costs)

    Where ICDR_cost comes from SEVERITY_TRANSITION in ontology.py:
        MA ↔ HE       = 0.0  (no grade change — most similar)
        MA ↔ Hard_EX  = 1.0  (one grade change)
        MA ↔ Soft_EX  = 2.0  (two grade change — most different)
        HE ↔ Hard_EX  = 1.0
        HE ↔ Soft_EX  = 2.0
        Hard_EX↔Soft_EX = 1.5

    Clinical ordering guaranteed by construction:
        SDM(MA,HE)=0.00 < SDM(MA,Hard_EX)=0.50 < SDM(MA,Soft_EX)=1.00  ✓
        SDM(HE,Hard_EX)=0.50 < SDM(HE,Soft_EX)=1.00                     ✓

    Note on MA↔HE (SDM=0.0): this is a clinically correct zero-cost pair,
    not a missing entry. Governance gate behavior: SDCA=0.0 for MA↔HE
    ambiguous boxes signals 'clinically safe confusion' to the LGG —
    correct behavior per architecture spec.

    Returns:
        {(ci, cj): {"raw": float, "norm": float, "source": str}}
        Keys are ordered (lesion_i, lesion_j) with i < j by LESION_NODES index.
        All "norm" values are in [0.0, 1.0].
    """
    from lgovnet.src.data.ontology import SEVERITY_TRANSITION, LESION_NODES

    lesion_names = sorted(LESION_NODES.keys(), key=lambda n: LESION_NODES[n])

    raw: dict = {}
    for i in range(len(lesion_names)):
        for j in range(i + 1, len(lesion_names)):
            ci, cj = lesion_names[i], lesion_names[j]
            cost = float(SEVERITY_TRANSITION.get(frozenset({ci, cj}), 0.0))
            raw[(ci, cj)] = cost

    max_cost = max(raw.values()) if raw else 1.0
    if max_cost < 1e-9:
        max_cost = 1.0

    result: dict = {}
    for pair, cost in raw.items():
        result[pair] = {
            "raw": cost,
            "norm": round(cost / max_cost, 6),
            "source": "severity_transition_icdr",
        }
    return result
