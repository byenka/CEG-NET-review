#!/usr/bin/env python
"""
eval_stage3d_c1_topology_ablation.py
Phase 19N — Stage 3D C1 Topology Ablation: KG graph structure contribution.

Completes the C1 mandatory ablations from architecture v3:
  • Directed vs undirected edges.
  • Clinical KG vs random graph baseline.
  • (Optional) Fixed-only vs all-trainable edges.

Tests whether the KG TOPOLOGY (directionality and clinical structure)
produces materially different LGG routing performance, by running the
frozen GAT encoder over modified KG graphs and using the resulting
4×64 lesion embeddings as e_pred in [SAS, e_pred] LGG input.

Variants:
  B0                        : score threshold (no model, baseline)
  B_class_only              : SAS + one-hot(pred_class)   [non-KG reference]
  B_kg_clinical_directed    : SAS + GAT(clinical KG, original edges)
  B_kg_clinical_undirected  : SAS + GAT(clinical KG, all directed → bidirectional)
  B_kg_random_topology_s*   : SAS + GAT(random topology, 5 seeds, same edge counts)

Decision rule:
  C1 KG topology contributes if:
    • B_kg_clinical_directed consistently > B_kg_random_topology
    • B_kg_clinical_directed differs from B_kg_clinical_undirected
       (either direction; the question is whether direction matters)

Operating point protocol: same as Stage 3D.1/3D.2/3D.3.

Inputs:
  outputs/stage3c_audit/governance_features_{train,valid,test}.csv
  outputs/stage3_audit/kg_encoder_init.pt
  outputs/graphs/nodes.json
  outputs/graphs/edges.json

Outputs:
  outputs/stage3d_ablation/c1_topology_results.csv
  outputs/stage3d_ablation/bootstrap_c1_topology_ci.csv
  outputs/stage3d_ablation/c1_topology_decision.md

Usage:
    python tools/eval_stage3d_c1_topology_ablation.py \\
      --stage3c_dir outputs/stage3c_audit \\
      --stage3a_dir outputs/stage3_audit \\
      --graphs_dir  outputs/graphs \\
      --output_dir  outputs/stage3d_ablation
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from sklearn.metrics import average_precision_score
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    _SK = True
except ImportError:
    _SK = False
    print("[Stage3D.C1Topo] WARNING: scikit-learn missing.")

try:
    import torch

    _TORCH = True
except ImportError:
    _TORCH = False
    print("[Stage3D.C1Topo] ERROR: torch required. Cannot run without GAT encoder.")
    sys.exit(1)


# ── Constants ─────────────────────────────────────────────────────────────────

_CLASS_ORDER = ["MA", "HE", "Hard_EX", "Soft_EX"]
_CLASS_IDX = {c: i for i, c in enumerate(_CLASS_ORDER)}

_RETAIN_CATS = frozenset({"TP_EXACT", "CLINICALLY_CLOSE_MISMATCH", "LOCALIZATION_WEAK"})

_TP_TARGETS = [0.85, 0.80, 0.75]
_THRESHOLD_GRID = np.round(np.linspace(0.001, 0.999, 499), 4)
_N_BOOTSTRAP = 1000
_BOOTSTRAP_SEED = 42
_N_RANDOM_SEEDS = 5


# ── KG topology variant builders ──────────────────────────────────────────────


def _make_undirected(kg: dict) -> dict:
    """Return a copy of kg with all directed edges duplicated as reverse-direction.

    For each edge with directed=True, add the reverse edge (dst → src) with the
    same weight and edge_type. Mark all edges as directed=False in the copy.
    """
    new_kg = copy.deepcopy(kg)
    for etype, edges in list(new_kg["edges"].items()):
        seen_pairs: set = set()
        new_list: List[dict] = []
        for e in edges:
            new_list.append({**e, "directed": False})
            seen_pairs.add((e["src"], e["dst"]))
        for e in edges:
            if e.get("directed", False):
                reverse_key = (e["dst"], e["src"])
                if reverse_key not in seen_pairs:
                    new_list.append(
                        {
                            **e,
                            "src": e["dst"],
                            "dst": e["src"],
                            "src_name": e["dst_name"],
                            "dst_name": e["src_name"],
                            "directed": False,
                        }
                    )
                    seen_pairs.add(reverse_key)
        new_kg["edges"][etype] = new_list
    return new_kg


def _make_random_topology(kg: dict, rng: np.random.Generator) -> dict:
    """Return a copy of kg with edge connectivity randomized per edge type.

    Preserves: number of edges per edge type, weight distribution per edge type,
              directed flag, and edge type membership.
    Randomizes: source/destination node pairs (sampled uniformly without self-loops).
    """
    new_kg = copy.deepcopy(kg)
    nodes = new_kg["nodes"]
    n_nodes = len(nodes)
    name_by_idx = {n["idx"]: n["name"] for n in nodes}

    for etype, edges in list(new_kg["edges"].items()):
        n_edges = len(edges)
        weights = [e["weight"] for e in edges]
        # Random pairs without self-loops, may have duplicates
        new_edges: List[dict] = []
        for k in range(n_edges):
            src = int(rng.integers(0, n_nodes))
            dst = int(rng.integers(0, n_nodes))
            while dst == src:
                dst = int(rng.integers(0, n_nodes))
            new_edges.append(
                {
                    "src": src,
                    "dst": dst,
                    "src_name": name_by_idx[src],
                    "dst_name": name_by_idx[dst],
                    "weight": weights[k],
                    "clinical_cost": 0.0,
                    "trainable": edges[k].get("trainable", True),
                    "directed": edges[k].get("directed", False),
                }
            )
        new_kg["edges"][etype] = new_edges
    return new_kg


# ── GAT embedding extraction ──────────────────────────────────────────────────


def _load_encoder(ckpt_path: Path, device: str = "cpu"):
    from lgovnet.src.kg.encoder import LGovNetEncoder, build_encoder_inputs

    state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    encoder = LGovNetEncoder(
        node_feat_dim=17,
        hidden_dim=64,
        embedding_dim=64,
        edge_type_dim=16,
        dropout=0.0,
    )
    encoder.load_state_dict(state, strict=False)
    encoder.eval()
    encoder = encoder.to(device)
    return encoder, build_encoder_inputs


def _kg_to_lesion_embeddings(
    kg: dict,
    encoder,
    build_encoder_inputs,
    device: str = "cpu",
) -> np.ndarray:
    """Run forward pass and extract 4×64 lesion embeddings."""
    enc_inputs = build_encoder_inputs(kg, device=device)
    with torch.no_grad():
        embeddings = encoder.forward_node_embeddings(
            enc_inputs["x"],
            enc_inputs["edge_index"],
            enc_inputs["edge_type_ids"],
            enc_inputs["edge_weights"],
        ).cpu()
    node_names: List[str] = enc_inputs["node_names"]
    lesion = np.zeros((4, 64), dtype=np.float32)
    for i, cls in enumerate(_CLASS_ORDER):
        if cls in node_names:
            lesion[i] = embeddings[node_names.index(cls)].numpy()
    return lesion


def _l2_pairwise(embeddings: np.ndarray) -> List[Tuple[str, str, float]]:
    """Pairwise L2 distances between lesion embeddings (for diagnostic print)."""
    pairs: List[Tuple[str, str, float]] = []
    for i in range(4):
        for j in range(i + 1, 4):
            d = float(np.linalg.norm(embeddings[i] - embeddings[j]))
            pairs.append((_CLASS_ORDER[i], _CLASS_ORDER[j], round(d, 4)))
    return pairs


# ── Data parse / build X ──────────────────────────────────────────────────────


def _load_csv(path: Path) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _parse(
    records: List[dict],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    sas_list, cls_list, y_list, un_list, tp_list, img_list = [], [], [], [], [], []
    for r in records:
        try:
            sas = float(r["ev_sas"])
            if not math.isfinite(sas):
                continue
        except (KeyError, ValueError):
            continue
        cat = r.get("category", "")
        cls = r.get("pred_class", "")
        if cls not in _CLASS_IDX or cat not in (
            "TP_EXACT",
            "CLINICALLY_CLOSE_MISMATCH",
            "LOCALIZATION_WEAK",
            "UNSAFE_MISMATCH",
            "LOW_CONF_CANDIDATE",
            "BACKGROUND_FP",
        ):
            continue
        sas_list.append(sas)
        cls_list.append(_CLASS_IDX[cls])
        y_list.append(1 if cat in _RETAIN_CATS else 0)
        un_list.append(cat == "UNSAFE_MISMATCH")
        tp_list.append(cat == "TP_EXACT")
        img_list.append(str(r.get("image_id", "")))
    return (
        np.array(sas_list, dtype=np.float64).reshape(-1, 1),
        np.array(cls_list, dtype=np.int32),
        np.array(y_list, dtype=np.int32),
        np.array(un_list, dtype=bool),
        np.array(tp_list, dtype=bool),
        img_list,
    )


def _build_X(
    sas: np.ndarray, cls_idx: np.ndarray, emb: Optional[np.ndarray], oh: bool
) -> np.ndarray:
    cols = [sas]
    if emb is not None:
        cols.append(emb[cls_idx])
    if oh:
        cols.append(np.eye(4, dtype=np.float32)[cls_idx])
    return np.concatenate(cols, axis=1).astype(np.float64)


# ── MLP / operating point ─────────────────────────────────────────────────────


def _train_mlp(X: np.ndarray, y: np.ndarray, seed: int = 42) -> Tuple[object, object]:
    sc = StandardScaler()
    m = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=400,
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        learning_rate_init=0.001,
        verbose=False,
    )
    m.fit(sc.fit_transform(X), y)
    return m, sc


def _proba(m: object, sc: object, X: np.ndarray) -> np.ndarray:
    return m.predict_proba(sc.transform(X))[:, 1].astype(np.float64)


def _find_thresh(y: np.ndarray, score: np.ndarray, target: float) -> Optional[float]:
    n_ret = int((y == 1).sum())
    if n_ret == 0:
        return None
    best: Optional[float] = None
    for t in sorted(_THRESHOLD_GRID):
        if int(((y == 1) & (score >= float(t))).sum()) / n_ret >= target:
            best = float(t)
    return best


def _eval_at(
    y: np.ndarray, score: np.ndarray, is_un: np.ndarray, thresh: float
) -> dict:
    kept = score >= thresh
    n_ret = int((y == 1).sum())
    n_sup = int((y == 0).sum())
    tp_r = int(((y == 1) & kept).sum()) / max(n_ret, 1)
    fp_s = int(((y == 0) & ~kept).sum()) / max(n_sup, 1)
    prec = int(y[kept].sum()) / max(int(kept.sum()), 1)
    un_s = int((is_un & ~kept).sum()) / int(is_un.sum()) if is_un.any() else None
    auprc = (
        float(average_precision_score(y, score))
        if (_SK and len(np.unique(y)) > 1)
        else None
    )
    return {
        "tp_ret": round(tp_r, 4),
        "fp_supp": round(fp_s, 4),
        "precision": round(prec, 4),
        "unsafe_supp": round(un_s, 4) if un_s is not None else None,
        "auprc": round(auprc, 4) if auprc is not None else None,
    }


# ── Bootstrap ─────────────────────────────────────────────────────────────────


def _boot_gain(
    img_ids: List[str],
    y: np.ndarray,
    sc_m: np.ndarray,
    sc_b0: np.ndarray,
    th_m: float,
    th_b0: float,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    if not _SK:
        return {"mean": None, "ci_lo": None, "ci_hi": None, "frac_pos": None, "n": 0}
    rng = np.random.default_rng(seed)
    uniq = list(dict.fromkeys(img_ids))
    img2idx: Dict[str, List[int]] = {}
    for i, img in enumerate(img_ids):
        img2idx.setdefault(img, []).append(i)
    gains: List[float] = []
    for _ in range(n_boot):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        idx = []
        [idx.extend(img2idx[img]) for img in samp]
        ia = np.array(idx, dtype=np.int64)
        yb = y[ia]
        if yb.sum() == 0 or (1 - yb).sum() == 0:
            continue
        sup = yb == 0
        n_s = int(sup.sum())
        if n_s == 0:
            continue
        gains.append(
            int((sup & (sc_m[ia] < th_m)).sum()) / n_s
            - int((sup & (sc_b0[ia] < th_b0)).sum()) / n_s
        )
    if len(gains) < 10:
        return {
            "mean": None,
            "ci_lo": None,
            "ci_hi": None,
            "frac_pos": None,
            "n": len(gains),
        }
    arr = np.array(gains)
    return {
        "mean": round(float(arr.mean()), 4),
        "ci_lo": round(float(np.percentile(arr, 2.5)), 4),
        "ci_hi": round(float(np.percentile(arr, 97.5)), 4),
        "frac_pos": round(float((arr > 0).mean()), 4),
        "n": len(gains),
    }


# ── Writers ───────────────────────────────────────────────────────────────────

_RES_FIELDS = [
    "variant",
    "topology",
    "tp_ret_target",
    "thresh_valid",
    "tp_ret_valid",
    "fp_supp_valid",
    "auprc_valid",
    "tp_ret_test",
    "fp_supp_test",
    "auprc_test",
    "fp_supp_gain_over_b0",
    "fp_supp_gain_over_class_only",
    "fp_supp_gain_over_clinical_directed",
    "achieves_target_test",
]
_BOOT_FIELDS = [
    "variant",
    "tp_ret_target",
    "mean_gain",
    "ci_lo",
    "ci_hi",
    "frac_positive",
    "n",
    "significant",
]


def _write_csv(rows: List[dict], fields: List[str], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _fmt(v: Optional[float], fmt: str = ".4f") -> str:
    return format(v, fmt) if v is not None else "N/A"


def _write_md(
    res_rows: List[dict],
    boot_rows: List[dict],
    variant_order: List[str],
    clinical_vs_random_wins: int,
    clinical_vs_random_total: int,
    directed_vs_undirected_delta: Dict[float, float],
    path: Path,
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Stage 3D C1 Topology Ablation — KG graph structure contribution\n\n")
        f.write(
            "> **Key questions**:\n"
            "> 1. Does clinical KG topology beat random topology?\n"
            "> 2. Does directed-vs-undirected matter for KG governance?\n\n"
        )

        f.write("## Variant FP_supp on test split (per TP target)\n\n")
        for tgt in _TP_TARGETS:
            f.write(f"### TP_ret ≥ {tgt:.2f}\n\n")
            f.write(
                "| Variant | Topology | Thresh V | FP_supp V | FP_supp T "
                "| Gain/B0 | Gain/class_only | Gain/clinical_directed |\n"
            )
            f.write("|---|---|---:|---:|---:|---:|---:|---:|\n")
            for name in variant_order:
                row = next(
                    (
                        r
                        for r in res_rows
                        if r["variant"] == name
                        and abs(r.get("tp_ret_target", -1) - tgt) < 1e-6
                    ),
                    None,
                )
                if row is None:
                    continue
                f.write(
                    f"| {name} | {row['topology']} "
                    f"| {_fmt(row.get('thresh_valid'))} "
                    f"| {_fmt(row.get('fp_supp_valid'))} "
                    f"| {_fmt(row.get('fp_supp_test'))} "
                    f"| {_fmt(row.get('fp_supp_gain_over_b0'), '+.4f') if row.get('fp_supp_gain_over_b0') is not None else '—'} "
                    f"| {_fmt(row.get('fp_supp_gain_over_class_only'), '+.4f') if row.get('fp_supp_gain_over_class_only') is not None else '—'} "
                    f"| {_fmt(row.get('fp_supp_gain_over_clinical_directed'), '+.4f') if row.get('fp_supp_gain_over_clinical_directed') is not None else '—'} |\n"
                )
            f.write("\n")

        f.write("## Bootstrap CI vs B0 (1000 resamples)\n\n")
        for tgt in _TP_TARGETS:
            f.write(f"### TP_ret ≥ {tgt:.2f}\n\n")
            f.write("| Variant | Mean gain | 95% CI | P(>0) | Sig |\n")
            f.write("|---|---:|---|---:|:---:|\n")
            for name in variant_order:
                row = next(
                    (
                        r
                        for r in boot_rows
                        if r.get("variant") == name
                        and abs(r.get("tp_ret_target", -1) - tgt) < 1e-6
                    ),
                    None,
                )
                if row is None:
                    continue
                ci = (
                    f"[{_fmt(row.get('ci_lo'))}, {_fmt(row.get('ci_hi'))}]"
                    if row.get("ci_lo") is not None
                    else "N/A"
                )
                f.write(
                    f"| {name} "
                    f"| {_fmt(row.get('mean_gain'), '+.4f') if row.get('mean_gain') is not None else 'N/A'} "
                    f"| {ci} "
                    f"| {_fmt(row.get('frac_positive'))} "
                    f"| {row.get('significant', '—')} |\n"
                )
            f.write("\n")

        f.write("## C1 Topology Decision\n\n```\n")
        thr = (
            int(0.75 * clinical_vs_random_total) if clinical_vs_random_total > 0 else 1
        )
        verdict_random = "PASS" if clinical_vs_random_wins >= thr else "FAIL"
        f.write(
            f"Clinical directed vs random topology: "
            f"clinical wins {clinical_vs_random_wins}/{clinical_vs_random_total} "
            f"(need ≥{thr} = 75%) → {verdict_random}\n\n"
        )
        f.write("Directed vs undirected ΔFP_supp (clinical KG):\n")
        for tgt, d in directed_vs_undirected_delta.items():
            sign = "+" if d >= 0 else ""
            f.write(f"  TP_ret ≥ {tgt:.2f}: Δ = {sign}{d:.4f}\n")
        f.write("\n")
        if verdict_random == "PASS":
            f.write(
                "→ Clinical KG topology contributes governance signal beyond random graph.\n"
                "→ Paper claim: 'directed clinical-prior graph structure is necessary'.\n"
            )
        else:
            f.write(
                "→ Clinical KG topology does NOT consistently beat random topology at random-init GAT.\n"
                "→ This is the EXPECTED result for random-initialized encoder: the GAT cannot exploit\n"
                "  clinical structure without end-to-end training.\n"
                "→ Paper framing: clinical KG topology contribution requires Mode 2 trained encoder.\n"
                "  In Mode 1, the deterministic SDM-row realization of e_pred (from Stage 3D.3)\n"
                "  is the recommended e_pred channel.\n"
            )
        f.write("```\n\n")

        f.write(
            "## Reading guide for paper Section 5.5\n\n"
            "This ablation completes the C1 mandatory ablation contract from architecture v3:\n"
            "  • Directed vs undirected edges → reported above.\n"
            "  • Clinical KG vs random graph baseline → reported above.\n\n"
            "Combined with Stage 3D.C1 (SDM-design ablation) the C1 ablation evidence is complete.\n"
        )

        f.write("\n## Limitations\n\n")
        f.write(
            "- GAT encoder is random-initialized (kg_encoder_init.pt). The expected effect of\n"
            "  topology variation at random init is small; this ablation is a sanity check, not\n"
            "  a positive test for the encoder design. The positive test belongs to Mode 2.\n"
            "- The 'random topology' baseline preserves edge counts per type and weight values,\n"
            "  varying only source/destination pairs.\n"
            "- Five random seeds are used; results aggregated as best/worst/median.\n"
        )


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3D C1 Topology Ablation.")
    parser.add_argument("--stage3c_dir", default="outputs/stage3c_audit")
    parser.add_argument("--stage3a_dir", default="outputs/stage3_audit")
    parser.add_argument("--graphs_dir", default="outputs/graphs")
    parser.add_argument("--output_dir", default="outputs/stage3d_ablation")
    parser.add_argument("--n_bootstrap", type=int, default=_N_BOOTSTRAP)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    s3c = Path(args.stage3c_dir)
    s3a = Path(args.stage3a_dir)
    graphs_dir = Path(args.graphs_dir)

    print("[Stage3D.C1Topo] === Phase 19N C1 Topology Ablation ===")

    # ── Load encoder + KG ──────────────────────────────────────────────────────
    _root = str(Path(__file__).parent.parent)
    if _root not in sys.path:
        sys.path.insert(0, _root)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Stage3D.C1Topo] device={device}")

    ckpt = s3a / "kg_encoder_init.pt"
    if not ckpt.exists():
        print(f"[Stage3D.C1Topo] ERROR: encoder checkpoint not found: {ckpt}")
        sys.exit(1)
    encoder, build_encoder_inputs = _load_encoder(ckpt, device=device)

    # Load original KG
    from lgovnet.src.kg.builder import build_graph_from_artifacts, load_kg, save_kg

    kg_cache_path = s3a / "kg_graph.json"
    if kg_cache_path.exists():
        kg_clinical_directed = load_kg(kg_cache_path)
        print(f"[Stage3D.C1Topo] Loaded cached KG: {kg_cache_path}")
    else:
        nodes_p = graphs_dir / "nodes.json"
        edges_p = graphs_dir / "edges.json"
        if not nodes_p.exists() or not edges_p.exists():
            print(f"[Stage3D.C1Topo] ERROR: missing nodes/edges in {graphs_dir}")
            sys.exit(1)
        kg_clinical_directed = build_graph_from_artifacts(nodes_p, edges_p)
        save_kg(kg_clinical_directed, kg_cache_path)
        print(f"[Stage3D.C1Topo] Built and cached KG → {kg_cache_path}")

    # Build variants
    print("\n[Stage3D.C1Topo] === Building KG topology variants ===")
    kg_clinical_undirected = _make_undirected(kg_clinical_directed)
    random_kgs: List[Tuple[str, dict]] = []
    for s in range(_N_RANDOM_SEEDS):
        rng = np.random.default_rng(seed=1000 + s)
        random_kgs.append(
            (
                f"B_kg_random_topology_s{s}",
                _make_random_topology(kg_clinical_directed, rng),
            )
        )

    # ── Compute lesion embeddings for each variant ─────────────────────────────
    print("\n[Stage3D.C1Topo] === Computing GAT embeddings per variant ===")
    embeddings: Dict[str, np.ndarray] = {}

    embeddings["B_kg_clinical_directed"] = _kg_to_lesion_embeddings(
        kg_clinical_directed, encoder, build_encoder_inputs, device=device
    )
    print(f"[Stage3D.C1Topo]   B_kg_clinical_directed: ", end="")
    for a, b, d in _l2_pairwise(embeddings["B_kg_clinical_directed"]):
        print(f"||{a}-{b}||={d:.3f}  ", end="")
    print()

    embeddings["B_kg_clinical_undirected"] = _kg_to_lesion_embeddings(
        kg_clinical_undirected, encoder, build_encoder_inputs, device=device
    )
    print(f"[Stage3D.C1Topo]   B_kg_clinical_undirected: ", end="")
    for a, b, d in _l2_pairwise(embeddings["B_kg_clinical_undirected"]):
        print(f"||{a}-{b}||={d:.3f}  ", end="")
    print()

    for name, kg_rnd in random_kgs:
        embeddings[name] = _kg_to_lesion_embeddings(
            kg_rnd, encoder, build_encoder_inputs, device=device
        )

    # ── Load data ──────────────────────────────────────────────────────────────
    print("\n[Stage3D.C1Topo] === Loading governance features ===")
    recs: Dict[str, List[dict]] = {}
    for sp in ["train", "valid", "test"]:
        fp = s3c / f"governance_features_{sp}.csv"
        if not fp.exists():
            print(f"[Stage3D.C1Topo] ERROR: {fp} not found.")
            sys.exit(1)
        recs[sp] = _load_csv(fp)

    sas_tr, cls_tr, y_tr, un_tr, _, _ = _parse(recs["train"])
    sas_va, cls_va, y_va, un_va, _, img_va = _parse(recs["valid"])
    sas_te, cls_te, y_te, un_te, _, img_te = _parse(recs["test"])

    for sp, y in [("train", y_tr), ("valid", y_va), ("test", y_te)]:
        print(
            f"[Stage3D.C1Topo]   {sp}: {len(y)} boxes  "
            f"RETAIN={y.sum()} SUPPRESS={(1-y).sum()}"
        )

    # ── Variant definitions ────────────────────────────────────────────────────
    all_variants: List[Tuple[str, str, Optional[np.ndarray], bool]] = [
        ("B0", "score", None, False),
        ("B_class_only", "onehot", None, True),
        (
            "B_kg_clinical_directed",
            "clinical_directed",
            embeddings["B_kg_clinical_directed"],
            False,
        ),
        (
            "B_kg_clinical_undirected",
            "clinical_undirected",
            embeddings["B_kg_clinical_undirected"],
            False,
        ),
    ]
    for name, _ in random_kgs:
        all_variants.append((name, "random", embeddings[name], False))

    variant_order = [v[0] for v in all_variants]
    print(f"\n[Stage3D.C1Topo] Variants: {variant_order}")

    # ── Train + score ──────────────────────────────────────────────────────────
    scores_va: Dict[str, np.ndarray] = {}
    scores_te: Dict[str, np.ndarray] = {}

    for name, topo, emb, use_oh in all_variants:
        X_tr = _build_X(sas_tr, cls_tr, emb, use_oh)
        X_va = _build_X(sas_va, cls_va, emb, use_oh)
        X_te = _build_X(sas_te, cls_te, emb, use_oh)

        if name == "B0":
            scores_va[name] = X_va[:, 0]
            scores_te[name] = X_te[:, 0]
            print(f"[Stage3D.C1Topo] {name}: passthrough  shape={X_va.shape}")
        elif _SK:
            print(
                f"[Stage3D.C1Topo] Training {name:<30} [{topo}]  shape={X_tr.shape} ...",
                end=" ",
                flush=True,
            )
            m, sc = _train_mlp(X_tr, y_tr)
            print(f"({getattr(m,'n_iter_','?')} iter)")
            scores_va[name] = _proba(m, sc, X_va)
            scores_te[name] = _proba(m, sc, X_te)

    # ── Calibrate operating points ─────────────────────────────────────────────
    print(f"\n[Stage3D.C1Topo] === Operating point calibration ===")
    b0_thresh: Dict[float, Optional[float]] = {}
    b0_fp_te: Dict[float, Optional[float]] = {}
    for tgt in _TP_TARGETS:
        th = _find_thresh(y_va, scores_va["B0"], tgt)
        b0_thresh[tgt] = th
        b0_fp_te[tgt] = (
            _eval_at(y_te, scores_te["B0"], un_te, th)["fp_supp"]
            if th is not None
            else None
        )

    res_rows: List[dict] = []
    all_thresh: Dict[Tuple[str, float], Optional[float]] = {}
    clin_fp: Dict[float, Optional[float]] = {}
    co_fp: Dict[float, Optional[float]] = {}

    for name, topo, emb, use_oh in all_variants:
        if name not in scores_va:
            continue
        sv, st = scores_va[name], scores_te[name]
        for tgt in _TP_TARGETS:
            th = _find_thresh(y_va, sv, tgt)
            all_thresh[(name, tgt)] = th
            if th is None:
                continue
            m_te = _eval_at(y_te, st, un_te, th)
            if name == "B_kg_clinical_directed":
                clin_fp[tgt] = m_te["fp_supp"]
            if name == "B_class_only":
                co_fp[tgt] = m_te["fp_supp"]

    for name, topo, emb, use_oh in all_variants:
        if name not in scores_va:
            continue
        sv, st = scores_va[name], scores_te[name]
        for tgt in _TP_TARGETS:
            th = all_thresh.get((name, tgt))
            if th is None:
                print(
                    f"[Stage3D.C1Topo]   {name:<30} TP>={tgt:.2f}: cannot reach target"
                )
                continue
            m_va = _eval_at(y_va, sv, un_va, th)
            m_te = _eval_at(y_te, st, un_te, th)

            b0_ref = b0_fp_te.get(tgt)
            co_ref = co_fp.get(tgt)
            cl_ref = clin_fp.get(tgt)
            gain_b0 = (
                round(m_te["fp_supp"] - b0_ref, 4)
                if b0_ref is not None and name != "B0"
                else None
            )
            gain_co = (
                round(m_te["fp_supp"] - co_ref, 4)
                if co_ref is not None and name not in ("B0", "B_class_only")
                else None
            )
            gain_cl = (
                round(m_te["fp_supp"] - cl_ref, 4)
                if cl_ref is not None and name not in ("B0", "B_kg_clinical_directed")
                else None
            )

            gstr = f"{gain_b0:+.4f}" if gain_b0 is not None else "(B0)"
            print(
                f"[Stage3D.C1Topo]   {name:<30} TP>={tgt:.2f}: "
                f"thresh={th:.4f}  "
                f"FP_supp(V={m_va['fp_supp']:.3f} T={m_te['fp_supp']:.3f})  "
                f"gain/B0={gstr}"
            )

            res_rows.append(
                {
                    "variant": name,
                    "topology": topo,
                    "tp_ret_target": tgt,
                    "thresh_valid": th,
                    "tp_ret_valid": m_va["tp_ret"],
                    "fp_supp_valid": m_va["fp_supp"],
                    "auprc_valid": m_va["auprc"],
                    "tp_ret_test": m_te["tp_ret"],
                    "fp_supp_test": m_te["fp_supp"],
                    "auprc_test": m_te["auprc"],
                    "fp_supp_gain_over_b0": gain_b0,
                    "fp_supp_gain_over_class_only": gain_co,
                    "fp_supp_gain_over_clinical_directed": gain_cl,
                    "achieves_target_test": (m_te["tp_ret"] or 0) >= tgt,
                }
            )

    # ── Bootstrap ──────────────────────────────────────────────────────────────
    print(f"\n[Stage3D.C1Topo] === Bootstrap CI ({args.n_bootstrap} resamples) ===")
    boot_rows: List[dict] = []
    for name, topo, _, _ in all_variants:
        if name == "B0" or name not in scores_te:
            continue
        for tgt in _TP_TARGETS:
            th_m = all_thresh.get((name, tgt))
            th_b0 = b0_thresh.get(tgt)
            if th_m is None or th_b0 is None:
                continue
            b = _boot_gain(
                img_te,
                y_te,
                scores_te[name],
                scores_te["B0"],
                th_m,
                th_b0,
                args.n_bootstrap,
                _BOOTSTRAP_SEED,
            )
            sig = "YES" if b["ci_lo"] is not None and b["ci_lo"] > 0 else "no"
            print(
                f"[Stage3D.C1Topo]   {name:<30} TP>={tgt:.2f}: "
                f"gain={_fmt(b['mean'], '+.4f')}  "
                f"CI=[{_fmt(b['ci_lo'])},{_fmt(b['ci_hi'])}]  "
                f"P(>0)={_fmt(b['frac_pos'])}  sig={sig}"
            )
            boot_rows.append(
                {
                    "variant": name,
                    "tp_ret_target": tgt,
                    "mean_gain": b["mean"],
                    "ci_lo": b["ci_lo"],
                    "ci_hi": b["ci_hi"],
                    "frac_positive": b["frac_pos"],
                    "n": b["n"],
                    "significant": sig,
                }
            )

    # ── Clinical vs random comparison ──────────────────────────────────────────
    print(f"\n[Stage3D.C1Topo] === Clinical-directed vs random topology ===")
    clin_vs_rand_wins = 0
    clin_vs_rand_total = 0
    for tgt in _TP_TARGETS:
        clin = clin_fp.get(tgt)
        if clin is None:
            continue
        for s in range(_N_RANDOM_SEEDS):
            rname = f"B_kg_random_topology_s{s}"
            rrow = next(
                (
                    r
                    for r in res_rows
                    if r["variant"] == rname
                    and abs(r.get("tp_ret_target", -1) - tgt) < 1e-6
                ),
                None,
            )
            if rrow is None or rrow.get("fp_supp_test") is None:
                continue
            clin_vs_rand_total += 1
            win = clin > rrow["fp_supp_test"]
            if win:
                clin_vs_rand_wins += 1
            print(
                f"[Stage3D.C1Topo]   clinical({clin:.4f}) {'>' if win else '<='} "
                f"rand_s{s}({rrow['fp_supp_test']:.4f}) TP>={tgt:.2f} → "
                f"{'WIN' if win else 'lose'}"
            )

    # ── Directed vs undirected delta ───────────────────────────────────────────
    directed_vs_undirected_delta: Dict[float, float] = {}
    for tgt in _TP_TARGETS:
        d = next(
            (
                r
                for r in res_rows
                if r["variant"] == "B_kg_clinical_directed"
                and abs(r.get("tp_ret_target", -1) - tgt) < 1e-6
            ),
            None,
        )
        u = next(
            (
                r
                for r in res_rows
                if r["variant"] == "B_kg_clinical_undirected"
                and abs(r.get("tp_ret_target", -1) - tgt) < 1e-6
            ),
            None,
        )
        if (
            d is None
            or u is None
            or d.get("fp_supp_test") is None
            or u.get("fp_supp_test") is None
        ):
            directed_vs_undirected_delta[tgt] = 0.0
            continue
        directed_vs_undirected_delta[tgt] = round(
            d["fp_supp_test"] - u["fp_supp_test"], 4
        )

    print(f"\n[Stage3D.C1Topo] === C1 Topology Verdict ===")
    threshold_required = int(0.75 * clin_vs_rand_total) if clin_vs_rand_total > 0 else 1
    print(
        f"[Stage3D.C1Topo]   clinical vs random wins: {clin_vs_rand_wins}/{clin_vs_rand_total} "
        f"(need ≥ {threshold_required})"
    )
    print(f"[Stage3D.C1Topo]   directed vs undirected Δ per TP target:")
    for tgt, d in directed_vs_undirected_delta.items():
        print(f"[Stage3D.C1Topo]     TP_ret ≥ {tgt:.2f}: Δ = {d:+.4f}")

    # ── Write outputs ──────────────────────────────────────────────────────────
    rp = out_dir / "c1_topology_results.csv"
    _write_csv(res_rows, _RES_FIELDS, rp)
    print(f"\n[Stage3D.C1Topo] {rp}")

    bp = out_dir / "bootstrap_c1_topology_ci.csv"
    _write_csv(boot_rows, _BOOT_FIELDS, bp)
    print(f"[Stage3D.C1Topo] {bp}")

    md = out_dir / "c1_topology_decision.md"
    _write_md(
        res_rows,
        boot_rows,
        variant_order,
        clin_vs_rand_wins,
        clin_vs_rand_total,
        directed_vs_undirected_delta,
        md,
    )
    print(f"[Stage3D.C1Topo] {md}")

    verdict = "PASS" if clin_vs_rand_wins >= threshold_required else "FAIL"
    print(f"\n[Stage3D.C1Topo] Done.  C1 topology verdict: {verdict}")


if __name__ == "__main__":
    main()
