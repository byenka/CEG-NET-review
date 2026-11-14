

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


    idx: int
    name: str
    node_type: str  # "lesion" | "zone"
    feat_dim: int = 17


@dataclass
class EdgeSpec:


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

    adj: dict = {}
    for e in build_default_edges():
        adj.setdefault(e.edge_type, []).append((e.src, e.dst, e.init_weight))
    return adj


def build_clinical_prior_sdm() -> dict:

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
