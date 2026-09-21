"""
build_static_kg.py
Build and save the static LGov-Net Knowledge Graph from clinical priors only.

Does NOT require dataset access. Uses ontology.py + schema.py exclusively.
Trainable edges are included with their placeholder init_weight values
(1.0 uniform, or 2.0 for high-affinity spatial_distribution pairs).

The co_occurrence_pattern and spatial_distribution edges will be updated
with real dataset statistics by tools/build_dataset_stats.py.

Outputs:
  outputs/graphs/nodes.json   — 9 NodeSpec dicts
  outputs/graphs/edges.json   — all EdgeSpec dicts (37 total)

Usage:
    python tools/build_static_kg.py
    python tools/build_static_kg.py \\
        --nodes_out outputs/graphs/nodes.json \\
        --edges_out outputs/graphs/edges.json
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lgovnet.src.kg.schema import build_adjacency_dict, export_nodes_edges

# Expected edge count per type for sanity-check:
#   morphological_confusability  : 3  (MA↔HE, HardEX↔SoftEX, MA↔HardEX)
#   pathophysiological_mechanism : 3  (MA→HE, MA→HardEX, MA→SoftEX)
#   spatial_distribution         : 20 (4 lesions × 5 zones)
#   severity_transition          : 6  (all unique lesion pairs)
#   co_occurrence_pattern        : 6  (all unique lesion pairs, C(4,2))
EXPECTED_EDGE_COUNTS: dict = {
    "morphological_confusability": 3,
    "pathophysiological_mechanism": 3,
    "spatial_distribution": 20,
    "severity_transition": 6,
    "co_occurrence_pattern": 6,
}
EXPECTED_TOTAL_EDGES: int = sum(EXPECTED_EDGE_COUNTS.values())  # 38


def run(nodes_out: str, edges_out: str) -> None:
    """Build the static KG and save to nodes_out / edges_out."""

    nodes_dicts, edges_dicts = export_nodes_edges(
        nodes_path=nodes_out,
        edges_path=edges_out,
    )

    # ── Sanity checks ─────────────────────────────────────────────────────
    assert len(nodes_dicts) == 9, f"Expected 9 nodes, got {len(nodes_dicts)}"

    by_type: dict = {}
    for e in edges_dicts:
        by_type.setdefault(e["edge_type"], []).append(e)

    ok = True
    for etype, expected in EXPECTED_EDGE_COUNTS.items():
        got = len(by_type.get(etype, []))
        status = "OK" if got == expected else f"FAIL (expected {expected}, got {got})"
        if got != expected:
            ok = False
        print(f"  {etype:<35} {got:>3} edges  [{status}]")

    assert ok, "Edge count mismatch — check schema.py"

    # ── Terminal summary ──────────────────────────────────────────────────
    sep = "=" * 58
    n_fixed = sum(1 for e in edges_dicts if not e["trainable"])
    n_trainable = sum(1 for e in edges_dicts if e["trainable"])
    n_directed = sum(1 for e in edges_dicts if e["directed"])

    print()
    print(sep)
    print("  LGov-Net Static KG — Build Summary")
    print(sep)
    print(
        f"  Nodes          : {len(nodes_dicts):>3}  "
        f"(lesion={sum(1 for n in nodes_dicts if n['node_type']=='lesion')}, "
        f"zone={sum(1 for n in nodes_dicts if n['node_type']=='zone')})"
    )
    print(f"  Edges total    : {len(edges_dicts):>3}")
    print(f"    Fixed        : {n_fixed:>3}")
    print(f"    Trainable    : {n_trainable:>3}")
    print(f"    Directed     : {n_directed:>3}  (all pathophysiological_mechanism)")
    print()
    print(f"  Saved nodes -> {nodes_out}")
    print(f"  Saved edges -> {edges_out}")
    print(sep)

    # ── Per-edge-type detail ──────────────────────────────────────────────
    EDGE_META: dict = {
        "morphological_confusability": ("FIXED", "undirected"),
        "pathophysiological_mechanism": ("TRAINABLE", "directed  "),
        "spatial_distribution": ("TRAINABLE", "undirected"),
        "severity_transition": ("FIXED", "undirected"),
        "co_occurrence_pattern": ("TRAINABLE", "undirected"),
    }
    print()
    print(f"  {'Edge type':<35} {'Training':<12} {'Direction':<11} {'Count':>5}")
    print("  " + "-" * 65)
    for etype, (trainability, direction) in EDGE_META.items():
        count = len(by_type.get(etype, []))
        print(f"  {etype:<35} {trainability:<12} {direction:<11} {count:>5}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build static LGov-Net KG from clinical priors (no dataset required)",
    )
    parser.add_argument(
        "--nodes_out",
        default="outputs/graphs/nodes.json",
        help="Output path for nodes JSON (default: outputs/graphs/nodes.json)",
    )
    parser.add_argument(
        "--edges_out",
        default="outputs/graphs/edges.json",
        help="Output path for edges JSON (default: outputs/graphs/edges.json)",
    )
    args = parser.parse_args()
    run(args.nodes_out, args.edges_out)


if __name__ == "__main__":
    main()
