

import csv
import json
from pathlib import Path

# ── Internal helpers ──────────────────────────────────────────────────────────


def _load_json(path: "str | Path") -> "dict | list":
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_cooccurrence_csv(csv_path: "str | Path") -> "dict[str, dict[str, int]]":

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

    vec = [0.0] * n
    vec[idx] = 1.0
    return vec


# ── Node features ─────────────────────────────────────────────────────────────


def get_node_features(nodes: "list[dict]") -> "dict[str, list[float]]":

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

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(kg, f, indent=2)
    print(f"  Saved KG: {output_path}")


def load_kg(kg_path: "str | Path") -> dict:

    with open(kg_path, "r", encoding="utf-8") as f:
        return json.load(f)
