"""
builder.py
Assemble the LGov-Net Knowledge Graph for the encoder.

Combines:
  - Static structural schema (schema.py) — node/edge topology + clinical priors
  - Dataset statistics (outputs of build_dataset_stats.py) — co-occurrence
    and zone-affinity data to initialize trainable edge weights

Two modes:
  static_only  — clinical priors only, no dataset required, trainable edges
                 keep uniform init_weight=1.0 placeholders.
  dataset_init — trainable co_occurrence_pattern and spatial_distribution
                 edges are re-initialized from dataset statistics.

Output contract  (all functions return the same BuiltGraph named-tuple-like dict):
    {
        "nodes":    list[dict]        — 9 entries, ordered by idx
        "edges":    dict[str, list]   — keyed by edge_type, each list sorted by (src, dst)
        "node_features": dict[str, list[float]]  — {node_name: feature_vector}
        "meta":     dict              — build provenance and counts
    }

    Edge dicts per type:
        {"src": int, "dst": int, "weight": float,
         "trainable": bool, "directed": bool}

Node feature vector contract (matches NodeSpec.feat_dim):
    lesion node (feat_dim=17):  one_hot(9) + clinical_attr(8)   — all floats
    zone node   (feat_dim=9):   one_hot(9)                       — all floats

GAT, SDM, Compat, and training are NOT in scope for this module.
"""

import csv
import json
from pathlib import Path

# ── Internal helpers ──────────────────────────────────────────────────────────


def _load_json(path: "str | Path") -> "dict | list":
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_cooccurrence_csv(csv_path: "str | Path") -> "dict[str, dict[str, int]]":
    """Read cooccurrence_matrix.csv → {cls_a: {cls_b: int}}.

    The CSV has a header row like: ,MA,HE,Hard_EX,Soft_EX
    and data rows like:            MA,314,230,199,99
    """
    result: dict = {}
    csv_path = Path(csv_path)
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        col_names = header[1:]  # skip leading empty cell
        for row in reader:
            row_name = row[0]
            result[row_name] = {
                col_names[i]: int(row[i + 1]) for i in range(len(col_names))
            }
    return result


def _load_zone_affinity_csv(
    csv_path: "str | Path",
) -> "dict[str, dict[str, float]]":
    """Read lesion_zone_affinity.csv → {lesion: {zone: float}} for observed only.

    Rows with observed==False (conceptual-only nodes) are skipped — their
    zone values are blank by design (no fake statistics).
    """
    result: dict = {}
    csv_path = Path(csv_path)
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("observed", "False") != "True":
                continue
            lesion = row["lesion"]
            result[lesion] = {
                zone: float(row[zone])
                for zone in row
                if zone not in ("lesion", "observed") and row[zone] != ""
            }
    return result


def _build_one_hot(idx: int, n: int = 9) -> "list[float]":
    """Return a length-n one-hot vector with 1.0 at position idx."""
    vec = [0.0] * n
    vec[idx] = 1.0
    return vec


# ── Node features ─────────────────────────────────────────────────────────────


def get_node_features(nodes: "list[dict]") -> "dict[str, list[float]]":
    """Build the feature vector for each KG node.

    Lesion nodes (feat_dim=17):  one_hot(9) + CLINICAL_ATTRS(8)
    Zone nodes   (feat_dim=9):   one_hot(9)

    Args:
        nodes: List of node dicts as stored in nodes.json (keys: idx, name,
               node_type, feat_dim).

    Returns:
        OrderedDict-compatible plain dict {node_name: [float, ...]} with
        deterministic ordering (sorted by node idx).  All values are floats
        so the dict can be fed directly into numpy/torch without conversion.
    """
    from lgovnet.src.data.ontology import CLINICAL_ATTRS_NORMALIZED

    features: dict = {}
    for node in sorted(nodes, key=lambda n: n["idx"]):
        one_hot = _build_one_hot(node["idx"], n=9)
        if node["node_type"] == "lesion":
            attrs = CLINICAL_ATTRS_NORMALIZED[node["name"]]
            features[node["name"]] = one_hot + attrs  # length 17
        else:
            features[node["name"]] = one_hot  # length 9
    return features


# ── Edge accessors ────────────────────────────────────────────────────────────


def get_edge_index_by_type(
    edges: "dict[str, list[dict]]",
    edge_type: str,
    include_reverse: bool = False,
) -> "list[list[int]]":
    """Return [[src_indices], [dst_indices]] for a given edge type.

    Args:
        edges:           Output of build_graph_from_artifacts()["edges"].
        edge_type:       One of the 5 LGov-Net edge type strings.
        include_reverse: If True, also append the reverse direction for
                         undirected edges (gives symmetric adjacency).
                         Directed edges (pathophysiological_mechanism) are
                         never reversed regardless of this flag.

    Returns:
        [src_list, dst_list] — two parallel integer lists, suitable for
        use as edge_index in PyTorch Geometric.
    """
    edge_list = edges.get(edge_type, [])
    src_list: list = []
    dst_list: list = []
    for e in edge_list:
        src_list.append(e["src"])
        dst_list.append(e["dst"])
        if include_reverse and not e["directed"]:
            src_list.append(e["dst"])
            dst_list.append(e["src"])
    return [src_list, dst_list]


def get_edge_weight_by_type(
    edges: "dict[str, list[dict]]",
    edge_type: str,
    include_reverse: bool = False,
) -> "list[float]":
    """Return weight list aligned with get_edge_index_by_type().

    Args:
        edges:           Output of build_graph_from_artifacts()["edges"].
        edge_type:       Target edge type string.
        include_reverse: Mirror of the same flag in get_edge_index_by_type().

    Returns:
        List of floats, same length as each src/dst list returned by
        get_edge_index_by_type() with the same arguments.
    """
    edge_list = edges.get(edge_type, [])
    weights: list = []
    for e in edge_list:
        weights.append(e["weight"])
        if include_reverse and not e["directed"]:
            weights.append(e["weight"])
    return weights


# ── Main builder ──────────────────────────────────────────────────────────────


def build_graph_from_artifacts(
    nodes_json: "str | Path" = "outputs/graphs/nodes.json",
    edges_json: "str | Path" = "outputs/graphs/edges.json",
    cooccurrence_csv: "str | Path | None" = None,
    zone_affinity_csv: "str | Path | None" = None,
) -> dict:
    """Assemble the KG dict from pre-built artifact files.

    Two operational modes depending on which CSVs are supplied:

    static_only (both CSV paths are None):
        Loads nodes + edges from nodes.json / edges.json.
        Trainable edges keep the init_weight values from schema.py.
        No dataset access required.

    dataset_init (one or both CSV paths supplied):
        co_occurrence_pattern weights are re-initialized from the
        co-occurrence matrix (normalized by max pair count).
        spatial_distribution weights are re-initialized from the
        normalized zone-affinity matrix.
        Fixed edges (morphological_confusability, severity_transition)
        are NEVER modified by dataset statistics.

    Args:
        nodes_json:       Path to outputs/graphs/nodes.json (from build_static_kg.py).
        edges_json:       Path to outputs/graphs/edges.json (from build_static_kg.py).
        cooccurrence_csv: Path to outputs/stats/cooccurrence_matrix.csv.
                          If None, co_occurrence_pattern weights are unchanged.
        zone_affinity_csv: Path to outputs/stats/lesion_zone_affinity.csv.
                           If None, spatial_distribution weights are unchanged.

    Returns:
        {
          "nodes": list[dict],
          "edges": dict[str, list[dict]],   # keyed by edge_type
          "node_features": dict[str, list[float]],
          "meta": dict,
        }
        Output is fully deterministic — same inputs produce identical dicts.
    """
    nodes_json = Path(nodes_json)
    edges_json = Path(edges_json)

    if not nodes_json.exists():
        raise FileNotFoundError(
            f"[builder] nodes.json not found: {nodes_json}. "
            f"Run 'python tools/build_static_kg.py' first."
        )
    if not edges_json.exists():
        raise FileNotFoundError(
            f"[builder] edges.json not found: {edges_json}. "
            f"Run 'python tools/build_static_kg.py' first."
        )

    nodes: list = _load_json(nodes_json)  # list[dict]
    raw_edges: list = _load_json(edges_json)  # list[dict]

    # ── Load optional dataset stats ──────────────────────────────────────────
    cooc_matrix: "dict | None" = None
    zone_aff: "dict | None" = None
    mode = "static_only"

    if cooccurrence_csv is not None:
        cooc_matrix = _load_cooccurrence_csv(cooccurrence_csv)
        mode = "dataset_init"
    if zone_affinity_csv is not None:
        zone_aff = _load_zone_affinity_csv(zone_affinity_csv)
        mode = "dataset_init"

    # ── Pre-compute normalization denominators ────────────────────────────────
    # co_occurrence: normalize by max off-diagonal co-occurrence count so that
    # the most frequently co-occurring pair gets weight=1.0 and all others are
    # proportionally scaled.
    cooc_max: float = 1.0
    if cooc_matrix is not None:
        off_diag = [
            cooc_matrix[a][b] for a in cooc_matrix for b in cooc_matrix if a != b
        ]
        cooc_max = float(max(off_diag)) if off_diag else 1.0
        if cooc_max == 0.0:
            cooc_max = 1.0

    # ── Build edge lookup helpers ─────────────────────────────────────────────
    def _cooc_weight(src_name: str, dst_name: str) -> float:
        if cooc_matrix is None:
            return None
        count = cooc_matrix.get(src_name, {}).get(dst_name, 0)
        return count / cooc_max

    def _zone_weight(lesion_name: str, zone_name: str) -> float:
        if zone_aff is None:
            return None
        if lesion_name not in zone_aff:
            # conceptual-only node — no data, return None to keep prior weight
            return None
        return zone_aff[lesion_name].get(zone_name, 0.0)

    # ── Assemble edges grouped by type ────────────────────────────────────────
    FIXED_TYPES = {"morphological_confusability", "severity_transition"}

    edges_by_type: dict = {}
    for e in raw_edges:
        etype = e["edge_type"]
        is_fixed = not e["trainable"]

        if is_fixed:
            # Fixed edges: weight is immutable regardless of data.
            weight = e["init_weight"]
        elif etype == "co_occurrence_pattern":
            new_w = _cooc_weight(e["src_name"], e["dst_name"])
            weight = new_w if new_w is not None else e["init_weight"]
        elif etype == "spatial_distribution":
            new_w = _zone_weight(e["src_name"], e["dst_name"])
            weight = new_w if new_w is not None else e["init_weight"]
        else:
            # pathophysiological_mechanism: trainable but not dataset-initialized.
            weight = e["init_weight"]

        edge_dict = {
            "src": e["src"],
            "dst": e["dst"],
            "src_name": e["src_name"],
            "dst_name": e["dst_name"],
            "weight": weight,
            "clinical_cost": e.get("clinical_cost", 0.0),
            "trainable": e["trainable"],
            "directed": e["directed"],
        }
        edges_by_type.setdefault(etype, []).append(edge_dict)

    # Deterministic order within each edge type: sort by (src, dst)
    for etype in edges_by_type:
        edges_by_type[etype].sort(key=lambda x: (x["src"], x["dst"]))

    # ── Node features ─────────────────────────────────────────────────────────
    node_features = get_node_features(nodes)

    # ── Meta ──────────────────────────────────────────────────────────────────
    edge_counts = {etype: len(edge_list) for etype, edge_list in edges_by_type.items()}
    trainable_count = sum(
        1 for etype, elist in edges_by_type.items() for e in elist if e["trainable"]
    )
    fixed_count = sum(
        1 for etype, elist in edges_by_type.items() for e in elist if not e["trainable"]
    )

    meta = {
        "build_mode": mode,
        "n_nodes": len(nodes),
        "n_edge_types": len(edges_by_type),
        "n_edges_total": sum(edge_counts.values()),
        "n_edges_fixed": fixed_count,
        "n_edges_trainable": trainable_count,
        "edge_counts_by_type": edge_counts,
        "cooccurrence_csv": str(cooccurrence_csv) if cooccurrence_csv else None,
        "zone_affinity_csv": str(zone_affinity_csv) if zone_affinity_csv else None,
        "nodes_json": str(nodes_json),
        "edges_json": str(edges_json),
        "lesion_order": [
            n["name"]
            for n in sorted(nodes, key=lambda x: x["idx"])
            if n["node_type"] == "lesion"
        ],
        "zone_order": [
            n["name"]
            for n in sorted(nodes, key=lambda x: x["idx"])
            if n["node_type"] == "zone"
        ],
    }

    return {
        "nodes": sorted(nodes, key=lambda n: n["idx"]),
        "edges": edges_by_type,
        "node_features": node_features,
        "meta": meta,
    }


# ── Persistence ───────────────────────────────────────────────────────────────


def save_kg(kg: dict, output_path: "str | Path") -> None:
    """Save the built KG dict to JSON for audit and reproducibility.

    Creates parent directory if it does not exist.

    Args:
        kg:          Output of build_graph_from_artifacts().
        output_path: Destination .json path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(kg, f, indent=2)
    print(f"  Saved KG: {output_path}")


def load_kg(kg_path: "str | Path") -> dict:
    """Load a previously saved KG dict from JSON.

    Args:
        kg_path: Path to a .json file written by save_kg().

    Returns:
        KG dict matching the contract in build_graph_from_artifacts().
    """
    with open(kg_path, "r", encoding="utf-8") as f:
        return json.load(f)
