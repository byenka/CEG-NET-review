#!/usr/bin/env python
"""
eval_stage3d3_epred_ablation.py
Phase 19N — Stage 3D.3: e_pred ablation.

Tests whether KG-derived class embeddings (e_pred) contribute beyond
simple class one-hot encoding in the LGG.

Architecture v3 specifies LGG input as [SAS, SDCA, APS, PSS, e_pred].
Currently e_pred is absent from all Stage 3D experiments.
This ablation fills that gap.

Two e_pred implementations are tested:
  1. SDM-row embedding (4-dim, deterministic, clinical):
       e_pred[class] = row of the clinical SDM matrix
       e.g. e_pred[MA] = [0.0, 0.0, 0.5, 1.0]  (distances to HE, HardEX, SoftEX)
       Hypothesis: SDM-row encodes CLINICAL ORDERING directly from the KG.
       Advantage: no PyTorch dependency, reproducible, traceable.

  2. GAT embedding (64-dim, from frozen kg_encoder_init.pt):
       e_pred[class] = node embedding from 2-layer HeteroGAT on the KG.
       Hypothesis: GAT embeddings encode KG graph structure (edges + weights).
       Requires: torch, kg_encoder_init.pt checkpoint, KG builder.

Decision rule:
  If e_pred_sdmrow > B2_class_only  → SDM-row (clinical ordering) adds value
  If e_pred_gat    > B2_class_only  → GAT embedding adds value beyond class ID
  If both ≈ B2_class_only           → KG embedding is not superior to class identity

All experiments use the same operating point protocol as Stage 3D.1/3D.2.

Inputs:
  outputs/stage3c_audit/governance_features_{train,valid,test}.csv
  outputs/stage3_audit/sdm_norm_clinical_prior.json   (SDM-row e_pred)
  outputs/stage3_audit/kg_encoder_init.pt             (GAT e_pred, optional)

Outputs:
  outputs/stage3d_ablation/epred_ablation_results.csv
  outputs/stage3d_ablation/bootstrap_epred_fp_supp_gain_ci.csv
  outputs/stage3d_ablation/epred_ablation_decision.md

Usage:
    python tools/eval_stage3d3_epred_ablation.py \\
      --stage3c_dir  outputs/stage3c_audit \\
      --stage3a_dir  outputs/stage3_audit \\
      --output_dir   outputs/stage3d_ablation

    python tools/eval_stage3d3_epred_ablation.py --no_gat  # skip GAT embedding
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── sklearn guard ─────────────────────────────────────────────────────────────
try:
    from sklearn.metrics import average_precision_score
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    _SK = True
except ImportError:
    _SK = False
    print("[Stage3D.3] WARNING: scikit-learn not installed. Only B0 will run.")

# ── torch guard (for GAT embedding) ───────────────────────────────────────────
try:
    import torch

    _TORCH = True
except ImportError:
    _TORCH = False
    print("[Stage3D.3] NOTE: torch not imported. GAT e_pred will be skipped.")

# ── Constants ─────────────────────────────────────────────────────────────────

_RETAIN_CATS = frozenset({"TP_EXACT", "CLINICALLY_CLOSE_MISMATCH", "LOCALIZATION_WEAK"})
_UNSAFE_CAT = "UNSAFE_MISMATCH"

_CLASS_ORDER = ["MA", "HE", "Hard_EX", "Soft_EX"]
_CLASS_IDX = {c: i for i, c in enumerate(_CLASS_ORDER)}

_TP_TARGETS = [0.85, 0.80, 0.75]
_THRESHOLD_GRID = np.round(np.linspace(0.001, 0.999, 499), 4)
_N_BOOTSTRAP = 1000
_BOOTSTRAP_SEED = 42


# ── e_pred: SDM-row embedding ─────────────────────────────────────────────────


def _load_sdm_row_embeddings(sdm_path: Path) -> Optional[np.ndarray]:
    """Load clinical SDM and create 4-dim row embeddings for each class.

    SDM-row embedding for class c = [SDM(c, MA), SDM(c, HE), SDM(c, HardEX), SDM(c, SoftEX)]
    with SDM(c, c) = 0.

    Returns ndarray shape (4, 4): one 4-dim row per class in _CLASS_ORDER.
    This is a DETERMINISTIC clinical embedding derived from the grading-aware SDM.
    """
    if not sdm_path.exists():
        print(f"[Stage3D.3]   SDM file not found: {sdm_path}")
        return None
    try:
        with open(sdm_path, encoding="utf-8") as f:
            sdm_data = json.load(f)
    except Exception as e:
        print(f"[Stage3D.3]   Failed to load SDM: {e}")
        return None

    # Build 4x4 symmetric SDM matrix
    # JSON keys are serialized as "ci__cj" by _key_to_str() in audit_stage3a_kg.py
    sdm_mat = np.zeros((4, 4), dtype=np.float32)
    for key_str, val in sdm_data.items():
        try:
            if isinstance(key_str, str) and "__" in key_str:
                parts = key_str.split("__", 1)
                ci, cj = parts[0].strip(), parts[1].strip()
            else:
                continue
        except Exception:
            continue

        if ci not in _CLASS_IDX or cj not in _CLASS_IDX:
            continue
        # val may be {"raw": x, "norm": y} or just float
        if isinstance(val, dict):
            norm_val = float(val.get("norm", val.get("raw", 0.0)))
        else:
            norm_val = float(val)

        i, j = _CLASS_IDX[ci], _CLASS_IDX[cj]
        sdm_mat[i, j] = norm_val
        sdm_mat[j, i] = norm_val  # symmetric

    print(f"[Stage3D.3]   SDM-row embeddings loaded (4×4 matrix)")
    print(
        f"[Stage3D.3]   Clinical ordering check: "
        f"SDM(MA,HE)={sdm_mat[0,1]:.3f} < "
        f"SDM(MA,HardEX)={sdm_mat[0,2]:.3f} < "
        f"SDM(MA,SoftEX)={sdm_mat[0,3]:.3f}"
    )
    return sdm_mat  # [4, 4] — row i = embedding for class i


# ── e_pred: GAT embedding ──────────────────────────────────────────────────────


def _load_gat_embeddings(
    checkpoint_path: Path,
    stage3a_dir: Path,
) -> Optional[np.ndarray]:
    """Load kg_encoder_init.pt and extract lesion node embeddings (64-dim).

    Returns ndarray shape (4, 64): one 64-dim embedding per lesion class.
    Returns None if checkpoint or KG artifacts are unavailable.
    """
    if not _TORCH:
        print("[Stage3D.3]   GAT: torch not available, skipping.")
        return None
    if not checkpoint_path.exists():
        print(f"[Stage3D.3]   GAT: checkpoint not found: {checkpoint_path}")
        return None

    # Try to import the KG encoder and builder from src
    _src_root = str(Path(__file__).parent.parent)
    if _src_root not in sys.path:
        sys.path.insert(0, _src_root)
    try:
        from lgovnet.src.kg.encoder import (
            LGovNetEncoder,
            build_encoder_inputs,
            LESION_NAMES,
        )
        from lgovnet.src.kg.builder import build_graph_from_artifacts  # type: ignore
    except ImportError as e:
        print(f"[Stage3D.3]   GAT: cannot import encoder/builder: {e}")
        return None

    # Look for KG artifacts (kg_graph.json is a combined save from save_kg())
    kg_artifact_candidates = [
        stage3a_dir / "kg_graph.json",
        stage3a_dir / "kg_artifacts" / "kg_graph.json",
        Path("outputs/graphs/kg_graph.json"),  # from build_static_kg.py
        Path("outputs/stage3_audit/kg_graph.json"),
    ]
    kg_path = next((p for p in kg_artifact_candidates if p.exists()), None)
    if kg_path is None:
        print(
            "[Stage3D.3]   GAT: kg_graph.json not found — "
            "will build from nodes.json/edges.json."
        )
        kg_data = None
    else:
        from lgovnet.src.kg.builder import load_kg  # type: ignore

        kg_data = load_kg(kg_path)
        print(f"[Stage3D.3]   GAT: loaded kg_graph.json from {kg_path}")

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Stage3D.3]   GAT: using device={device}")
        state = torch.load(
            str(checkpoint_path), map_location=device, weights_only=False
        )
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

        if kg_data is not None:
            enc_inputs = build_encoder_inputs(kg_data, device=device)
        else:
            # Fallback: build from artifact files
            from lgovnet.src.kg.builder import build_graph_from_artifacts  # type: ignore

            nodes_candidates = [
                stage3a_dir / "nodes.json",
                stage3a_dir.parent / "graphs" / "nodes.json",
                Path("outputs/graphs/nodes.json"),
            ]
            edges_candidates = [
                stage3a_dir / "edges.json",
                stage3a_dir.parent / "graphs" / "edges.json",
                Path("outputs/graphs/edges.json"),
            ]
            nodes_p = next((p for p in nodes_candidates if p.exists()), None)
            edges_p = next((p for p in edges_candidates if p.exists()), None)
            if nodes_p is None or edges_p is None:
                print(
                    "[Stage3D.3]   GAT: nodes.json/edges.json not found. "
                    "Skipping GAT embedding."
                )
                return None
            kg_built = build_graph_from_artifacts(nodes_p, edges_p)
            # Cache for future runs
            try:
                from lgovnet.src.kg.builder import save_kg  # type: ignore

                save_path = stage3a_dir / "kg_graph.json"
                save_kg(kg_built, save_path)
                print(f"[Stage3D.3]   GAT: saved kg_graph.json → {save_path}")
            except Exception:
                pass
            enc_inputs = build_encoder_inputs(kg_built, device=device)

        with torch.no_grad():
            embeddings = encoder.forward_node_embeddings(
                enc_inputs["x"],
                enc_inputs["edge_index"],
                enc_inputs["edge_type_ids"],
                enc_inputs["edge_weights"],
            )  # [9, 64]
            embeddings = embeddings.cpu()

        # Extract lesion nodes (first 4)
        node_names = enc_inputs["node_names"]
        lesion_embs = np.zeros((4, 64), dtype=np.float32)
        for i, cls in enumerate(_CLASS_ORDER):
            if cls in node_names:
                idx = node_names.index(cls)
                lesion_embs[i] = embeddings[idx].numpy()
            else:
                print(f"[Stage3D.3]   GAT: class {cls} not in node_names")

        print(f"[Stage3D.3]   GAT embeddings loaded (4×64)")
        # L2 distances between lesion embeddings (should reflect clinical ordering)
        for i in range(4):
            for j in range(i + 1, 4):
                d = float(np.linalg.norm(lesion_embs[i] - lesion_embs[j]))
                print(
                    f"[Stage3D.3]     ‖e_{_CLASS_ORDER[i]} - e_{_CLASS_ORDER[j]}‖ = {d:.4f}"
                )
        return lesion_embs  # [4, 64]

    except Exception as e:
        print(f"[Stage3D.3]   GAT: failed to load encoder: {e}")
        return None


# ── Data loading ──────────────────────────────────────────────────────────────


def _load_csv(path: Path) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _parse(
    records: List[dict],
) -> Tuple[
    Dict[str, np.ndarray],  # feature arrays
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[str],
]:
    keys_float = ["pred_score", "ev_sas", "ev_sdca", "ev_aps", "ev_pss"]
    feat: Dict[str, list] = {k: [] for k in keys_float}
    feat["class_idx"] = []
    y_list, un_list, tp_list, img_list = [], [], [], []

    for r in records:
        try:
            vals = {}
            for k in keys_float:
                raw = r.get(k, "")
                if raw == "" or raw is None:
                    raise ValueError(f"empty {k}")
                vals[k] = float(raw)
        except (ValueError, KeyError):
            continue
        if any(not math.isfinite(v) for v in vals.values()):
            continue
        cls = r.get("pred_class", "")
        if cls not in _CLASS_IDX:
            continue
        for k in keys_float:
            feat[k].append(vals[k])
        feat["class_idx"].append(_CLASS_IDX[cls])
        y_list.append(int(r.get("label", 0)))
        un_list.append(r.get("category") == _UNSAFE_CAT)
        tp_list.append(r.get("category") == "TP_EXACT")
        img_list.append(str(r.get("image_id", "")))

    arrays = {
        k: np.array(v, dtype=np.float32) for k, v in feat.items() if k != "class_idx"
    }
    arrays["class_idx"] = np.array(feat["class_idx"], dtype=np.int32)
    arrays["ev_sdca_clinical"] = arrays.pop("ev_sdca")
    # One-hot encoding
    for i, cls in enumerate(_CLASS_ORDER):
        arrays[f"oh_{cls}"] = (arrays["class_idx"] == i).astype(np.float32)

    return (
        arrays,
        np.array(y_list, dtype=np.int32),
        np.array(un_list, dtype=bool),
        np.array(tp_list, dtype=bool),
        img_list,
    )


# ── Build feature matrix for each variant ────────────────────────────────────


def _build_X(
    feat: Dict[str, np.ndarray],
    variant_features: List[str],
    class_embeddings: Optional[np.ndarray] = None,
    embed_type: str = "",
) -> np.ndarray:
    """Build feature matrix. Special tokens:
    'e_pred' → expand class embeddings (class_embeddings must be provided)
    """
    cols = []
    for f in variant_features:
        if f == "e_pred":
            if class_embeddings is None:
                raise ValueError("e_pred requested but class_embeddings is None")
            # [N, embed_dim] using class_idx to look up embeddings
            emb = class_embeddings[feat["class_idx"]]  # [N, D]
            cols.append(emb)
        elif f in feat:
            cols.append(feat[f].reshape(-1, 1))
        else:
            raise KeyError(f"Feature '{f}' not found in feat dict")
    return np.concatenate(cols, axis=1).astype(np.float64)


# ── Model ─────────────────────────────────────────────────────────────────────


def _train_mlp(X: np.ndarray, y: np.ndarray, seed: int = 42) -> Tuple[object, object]:
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    m = MLPClassifier(
        hidden_layer_sizes=(64, 32),  # Larger network for high-dim e_pred
        max_iter=400,
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        learning_rate_init=0.001,
        verbose=False,
    )
    m.fit(Xs, y)
    return m, sc


def _proba(m: object, sc: object, X: np.ndarray) -> np.ndarray:
    Xs = sc.transform(X)
    if hasattr(m, "predict_proba"):
        return m.predict_proba(Xs)[:, 1].astype(np.float64)
    return m.decision_function(Xs).astype(np.float64)


# ── Threshold operations ───────────────────────────────────────────────────────


def _find_thresh(
    y_true: np.ndarray, y_score: np.ndarray, target: float
) -> Optional[float]:
    ret = y_true == 1
    n_ret = int(ret.sum())
    if n_ret == 0:
        return None
    best: Optional[float] = None
    for t in sorted(_THRESHOLD_GRID):
        if int((ret & (y_score >= float(t))).sum()) / n_ret >= target:
            best = float(t)
    return best


def _at(
    y_true: np.ndarray, y_score: np.ndarray, is_un: np.ndarray, thresh: float
) -> dict:
    yp = (y_score >= thresh).astype(np.int32)
    ret = y_true == 1
    sup = y_true == 0
    n_ret = int(ret.sum())
    n_sup = int(sup.sum())
    n_un = int(is_un.sum())
    tp_ret = int((ret & (yp == 1)).sum()) / max(n_ret, 1)
    fp_sup = int((sup & (yp == 0)).sum()) / max(n_sup, 1)
    kept = int((yp == 1).sum())
    prec = int((y_true[yp == 1]).sum()) / max(kept, 1)
    un_sup = int((is_un & (yp == 0)).sum()) / n_un if n_un > 0 else None
    auprc = (
        float(average_precision_score(y_true, y_score))
        if (_SK and len(np.unique(y_true)) > 1)
        else None
    )
    return {
        "tp_ret": round(tp_ret, 4),
        "fp_supp": round(fp_sup, 4),
        "precision": round(prec, 4),
        "recall_loss": round(1 - tp_ret, 4),
        "unsafe_supp": round(float(un_sup), 4) if un_sup is not None else None,
        "auprc": round(auprc, 4) if auprc is not None else None,
    }


# ── Bootstrap: FP_supp_gain CI ─────────────────────────────────────────────────


def _boot_gain(
    img_ids: List[str],
    y_true: np.ndarray,
    sc_model: np.ndarray,
    sc_b0: np.ndarray,
    th_model: float,
    th_b0: float,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    if not _SK:
        return {
            "mean_gain": None,
            "ci_lo": None,
            "ci_hi": None,
            "frac_positive": None,
            "n": 0,
        }
    rng = np.random.default_rng(seed)
    unique = list(dict.fromkeys(img_ids))
    n_u = len(unique)
    img2idx: Dict[str, List[int]] = {}
    for i, img in enumerate(img_ids):
        img2idx.setdefault(img, []).append(i)

    gains: List[float] = []
    for _ in range(n_boot):
        samp = rng.choice(unique, size=n_u, replace=True)
        idx: List[int] = []
        for img in samp:
            idx.extend(img2idx[img])
        ia = np.array(idx, dtype=np.int64)
        y_b = y_true[ia]
        if y_b.sum() == 0 or (1 - y_b).sum() == 0:
            continue
        sup = y_b == 0
        n_s = int(sup.sum())
        if n_s == 0:
            continue
        fp_m = int((sup & (sc_model[ia] < th_model)).sum()) / n_s
        fp_b = int((sup & (sc_b0[ia] < th_b0)).sum()) / n_s
        gains.append(fp_m - fp_b)

    if len(gains) < 10:
        return {
            "mean_gain": None,
            "ci_lo": None,
            "ci_hi": None,
            "frac_positive": None,
            "n": len(gains),
        }
    arr = np.array(gains)
    return {
        "mean_gain": round(float(arr.mean()), 4),
        "ci_lo": round(float(np.percentile(arr, 2.5)), 4),
        "ci_hi": round(float(np.percentile(arr, 97.5)), 4),
        "frac_positive": round(float((arr > 0).mean()), 4),
        "n": len(gains),
    }


# ── Output writers ────────────────────────────────────────────────────────────

_RES_FIELDS = [
    "variant",
    "embed_type",
    "n_features",
    "tp_ret_target",
    "thresh_valid",
    "tp_ret_valid",
    "fp_supp_valid",
    "precision_valid",
    "auprc_valid",
    "tp_ret_test",
    "fp_supp_test",
    "precision_test",
    "auprc_test",
    "fp_supp_gain_over_b0",
    "fp_supp_gain_over_class_only",
    "achieves_target_test",
]
_GAIN_FIELDS = [
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


def _fmt(v: Optional[float], f: str = ".4f") -> str:
    return format(v, f) if v is not None else "N/A"


def _write_md(
    res_rows: List[dict],
    gain_rows: List[dict],
    variant_order: List[str],
    path: Path,
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Stage 3D.3 — e_pred Ablation Decision\n\n")
        f.write(
            "> **Key question**: Does KG-derived class embedding (e_pred) "
            "outperform simple class one-hot encoding?\n\n"
        )

        for target in _TP_TARGETS:
            f.write(f"## TP_ret ≥ {target:.2f}\n\n")
            f.write(
                "| Variant | Embed | #Feat | Thresh V "
                "| FP_supp V | FP_supp T | Gain/B0 | Gain/class_only |\n"
            )
            f.write("|---|---|---:|---:|---:|---:|---:|---:|\n")
            for name in variant_order:
                row = next(
                    (
                        r
                        for r in res_rows
                        if r["variant"] == name
                        and abs(r.get("tp_ret_target", -1) - target) < 1e-6
                    ),
                    None,
                )
                if row is None:
                    continue
                g0 = row.get("fp_supp_gain_over_b0")
                gc = row.get("fp_supp_gain_over_class_only")
                f.write(
                    f"| {name} | {row.get('embed_type','?')} "
                    f"| {row.get('n_features','?')} "
                    f"| {_fmt(row.get('thresh_valid'))} "
                    f"| {_fmt(row.get('fp_supp_valid'))} "
                    f"| {_fmt(row.get('fp_supp_test'))} "
                    f"| {_fmt(g0, '+.4f') if g0 is not None else '—'} "
                    f"| {_fmt(gc, '+.4f') if gc is not None else '—'} |\n"
                )
            f.write("\n")

        f.write("## Bootstrap FP_supp_gain CI (vs B0, 1000 resamples)\n\n")
        for target in _TP_TARGETS:
            f.write(f"### TP_ret ≥ {target:.2f}\n\n")
            f.write("| Variant | Mean gain | 95% CI | P(>0) | Significant |\n")
            f.write("|---|---:|---|---:|:---:|\n")
            for name in variant_order:
                row = next(
                    (
                        r
                        for r in gain_rows
                        if r.get("variant") == name
                        and abs(r.get("tp_ret_target", -1) - target) < 1e-6
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
                    f"| {row.get('significant','—')} |\n"
                )
            f.write("\n")

        # Verdict
        f.write("## Verdict\n\n")
        f.write("```\n")
        class_only_fp = {}
        b0_fp = {}
        for row in res_rows:
            if row["variant"] == "B2_class_only":
                class_only_fp[(row.get("tp_ret_target"), "test")] = row.get(
                    "fp_supp_test"
                )
            if row["variant"] == "B0":
                b0_fp[(row.get("tp_ret_target"), "test")] = row.get("fp_supp_test")

        any_epred_wins_class_only = False
        for name in variant_order:
            if "epred" not in name.lower() and "e_pred" not in name.lower():
                continue
            for target in _TP_TARGETS:
                row = next(
                    (
                        r
                        for r in res_rows
                        if r["variant"] == name
                        and abs(r.get("tp_ret_target", -1) - target) < 1e-6
                    ),
                    None,
                )
                if row is None:
                    continue
                gc = row.get("fp_supp_gain_over_class_only")
                if gc is not None and gc > 0:
                    any_epred_wins_class_only = True
                    f.write(
                        f"e_pred ({name}) beats class_only at TP>={target:.2f}: "
                        f"+{gc:.4f} → KG embedding adds value\n"
                    )

        if not any_epred_wins_class_only:
            f.write(
                "No e_pred variant consistently outperforms class one-hot.\n"
                "→ KG encoder initialization does not provide richer class signal than identity.\n"
                "→ For CAS Tier 1: this motivates training the KG encoder end-to-end.\n"
            )
        else:
            f.write(
                "→ KG-derived class embeddings add signal beyond class identity.\n"
                "→ C2 evidence claim is stronger with e_pred included.\n"
            )
        f.write("```\n\n")

        f.write("## Implications for CAS Tier 1\n\n")
        f.write(
            "| Finding | Paper action |\n"
            "|---|---|\n"
            "| e_pred > class_only | Claim: KG encoder encodes clinical structure usable for routing |\n"
            "| e_pred ≈ class_only | Claim: governance framework + auditability; encoder needs training |\n"
            "| SDM-row > class_only | Claim: clinical ordering (ICDR) is directly useful for routing |\n"
            "| SDM-row ≈ class_only | SDM distances and class identity carry equivalent governance info |\n"
        )
        f.write("\n## Next step for stronger paper\n\n")
        f.write(
            "Regardless of e_pred result, the following will strengthen the paper:\n"
            "1. 4-tier routing (R1/R2/R3/R4) — more aligned with architecture v3\n"
            "2. Comparison table with published DR detection methods on IDRiD\n"
            "3. TTA-based SDCA diversity (if time allows) — real uncertainty signal\n"
            "4. Clinical certificate examples in paper (per-bbox audit trail)\n"
        )


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 3D.3: e_pred ablation — GAT vs SDM-row vs class one-hot."
    )
    parser.add_argument("--stage3c_dir", default="outputs/stage3c_audit")
    parser.add_argument("--stage3a_dir", default="outputs/stage3_audit")
    parser.add_argument("--output_dir", default="outputs/stage3d_ablation")
    parser.add_argument(
        "--no_gat", action="store_true", help="Skip GAT embedding (use SDM-row only)."
    )
    parser.add_argument("--n_bootstrap", type=int, default=_N_BOOTSTRAP)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    s3c = Path(args.stage3c_dir)
    s3a = Path(args.stage3a_dir)

    print("[Stage3D.3] === Phase 19N Stage 3D.3 e_pred Ablation ===")

    # ── Load e_pred embeddings ─────────────────────────────────────────────────
    sdm_path = s3a / "sdm_norm_clinical_prior.json"
    sdm_row_emb = _load_sdm_row_embeddings(sdm_path)

    gat_emb: Optional[np.ndarray] = None
    if not args.no_gat and _TORCH:
        ckpt_path = s3a / "kg_encoder_init.pt"
        gat_emb = _load_gat_embeddings(ckpt_path, s3a)

    # ── Load CSV ───────────────────────────────────────────────────────────────
    recs: Dict[str, List[dict]] = {}
    for sp in ["train", "valid", "test"]:
        fp = s3c / f"governance_features_{sp}.csv"
        if not fp.exists():
            print(f"[Stage3D.3] ERROR: {fp} not found.")
            sys.exit(1)
        recs[sp] = _load_csv(fp)

    feat_tr, y_tr, un_tr, tp_tr, _ = _parse(recs["train"])
    feat_va, y_va, un_va, tp_va, img_va = _parse(recs["valid"])
    feat_te, y_te, un_te, tp_te, img_te = _parse(recs["test"])

    for sp, y in [("train", y_tr), ("valid", y_va), ("test", y_te)]:
        print(
            f"[Stage3D.3]   {sp}: {len(y)} boxes  "
            f"RETAIN={y.sum()} ({100*y.mean():.1f}%)  "
            f"SUPPRESS={(1-y).sum()} ({100*(1-y.mean()):.1f}%)"
        )

    # ── Define variants ────────────────────────────────────────────────────────
    # (name, embed_type, feature_list, class_embeddings or None)
    variants_spec: List[Tuple[str, str, List[str], Optional[np.ndarray]]] = []

    # Baselines (repeated from 3D.2 for direct comparison)
    variants_spec += [
        ("B0", "score", ["pred_score"], None),
        ("B2_clinical", "sdca", ["ev_sas", "ev_sdca_clinical"], None),
        (
            "B2_class_only",
            "one_hot",
            ["ev_sas"] + [f"oh_{c}" for c in _CLASS_ORDER],
            None,
        ),
    ]

    # SDM-row variants
    if sdm_row_emb is not None:
        variants_spec += [
            ("B_sdmrow", "sdm_row_4d", ["ev_sas", "e_pred"], sdm_row_emb),
            (
                "B_sdmrow_sdca",
                "sdm_row_4d+sdca",
                ["ev_sas", "ev_sdca_clinical", "e_pred"],
                sdm_row_emb,
            ),
        ]

    # GAT variants
    if gat_emb is not None:
        variants_spec += [
            ("B_gat", "gat_64d", ["ev_sas", "e_pred"], gat_emb),
            (
                "B_gat_sdca",
                "gat_64d+sdca",
                ["ev_sas", "ev_sdca_clinical", "e_pred"],
                gat_emb,
            ),
        ]

    variant_order = [v[0] for v in variants_spec]
    print(f"[Stage3D.3] Variants: {variant_order}")

    # ── Build X arrays ─────────────────────────────────────────────────────────
    def _X(
        feat: Dict[str, np.ndarray], flist: List[str], cemb: Optional[np.ndarray]
    ) -> np.ndarray:
        return _build_X(feat, flist, class_embeddings=cemb)

    # ── Train models and collect scores ───────────────────────────────────────
    scores_va: Dict[str, np.ndarray] = {}
    scores_te: Dict[str, np.ndarray] = {}

    for name, etype, flist, cemb in variants_spec:
        X_tr_v = _X(feat_tr, flist, cemb)
        X_va_v = _X(feat_va, flist, cemb)
        X_te_v = _X(feat_te, flist, cemb)

        if name == "B0":
            scores_va[name] = X_va_v[:, 0].astype(np.float64)
            scores_te[name] = X_te_v[:, 0].astype(np.float64)
            print(f"[Stage3D.3] {name}: passthrough (shape={X_va_v.shape})")
        elif _SK:
            print(
                f"[Stage3D.3] Training {name} MLP  features={flist}  "
                f"shape={X_tr_v.shape}..."
            )
            m, sc = _train_mlp(X_tr_v, y_tr)
            n_it = getattr(m, "n_iter_", "?")
            print(f"[Stage3D.3]   converged in {n_it} iter")
            scores_va[name] = _proba(m, sc, X_va_v)
            scores_te[name] = _proba(m, sc, X_te_v)
        else:
            print(f"[Stage3D.3] SKIP {name}: sklearn not available")

    # ── Operating point calibration ────────────────────────────────────────────
    b0_th: Dict[float, Optional[float]] = {}
    b0_fp_te: Dict[float, Optional[float]] = {}
    class_only_fp_te: Dict[float, Optional[float]] = {}

    for tgt in _TP_TARGETS:
        th = _find_thresh(y_va, scores_va["B0"], tgt)
        b0_th[tgt] = th
        b0_fp_te[tgt] = _at(y_te, scores_te["B0"], un_te, th)["fp_supp"] if th else None

    print(f"\n[Stage3D.3] Operating point calibration...")
    res_rows: List[dict] = []
    all_thresholds: Dict[Tuple[str, float], Optional[float]] = {}

    for name, etype, flist, cemb in variants_spec:
        if name not in scores_va:
            continue
        sc_va = scores_va[name]
        sc_te = scores_te[name]
        n_feat = _X(feat_va, flist, cemb).shape[1] if name != "B0" else 1

        for tgt in _TP_TARGETS:
            th = _find_thresh(y_va, sc_va, tgt)
            all_thresholds[(name, tgt)] = th
            if th is None:
                print(f"[Stage3D.3]   {name:<22} TP>={tgt:.2f}: cannot reach target")
                res_rows.append(
                    {
                        "variant": name,
                        "embed_type": etype,
                        "n_features": n_feat,
                        "tp_ret_target": tgt,
                        **{
                            k: None
                            for k in _RES_FIELDS
                            if k
                            not in (
                                "variant",
                                "embed_type",
                                "n_features",
                                "tp_ret_target",
                            )
                        },
                    }
                )
                continue

            m_va = _at(y_va, sc_va, un_va, th)
            m_te = _at(y_te, sc_te, un_te, th)
            b0_ref = b0_fp_te.get(tgt)
            gain_b0 = (
                round(m_te["fp_supp"] - b0_ref, 4)
                if (b0_ref is not None and name != "B0")
                else None
            )

            # Store class_only fp for gain comparison
            if name == "B2_class_only":
                class_only_fp_te[tgt] = m_te["fp_supp"]

            gstr = f"{gain_b0:+.4f}" if gain_b0 is not None else "(B0)"
            print(
                f"[Stage3D.3]   {name:<22} TP>={tgt:.2f}: "
                f"thresh={th:.4f}  "
                f"FP_supp(V={m_va['fp_supp']:.3f} T={m_te['fp_supp']:.3f})  "
                f"gain/B0={gstr}"
            )
            res_rows.append(
                {
                    "variant": name,
                    "embed_type": etype,
                    "n_features": n_feat,
                    "tp_ret_target": tgt,
                    "thresh_valid": th,
                    "tp_ret_valid": m_va["tp_ret"],
                    "fp_supp_valid": m_va["fp_supp"],
                    "precision_valid": m_va["precision"],
                    "auprc_valid": m_va["auprc"],
                    "tp_ret_test": m_te["tp_ret"],
                    "fp_supp_test": m_te["fp_supp"],
                    "precision_test": m_te["precision"],
                    "auprc_test": m_te["auprc"],
                    "fp_supp_gain_over_b0": gain_b0,
                    "fp_supp_gain_over_class_only": None,  # filled below
                    "achieves_target_test": (m_te["tp_ret"] or 0) >= tgt,
                }
            )

    # Fill gain_over_class_only
    for row in res_rows:
        tgt = row.get("tp_ret_target")
        fp_t = row.get("fp_supp_test")
        co_ref = class_only_fp_te.get(tgt)
        if (
            fp_t is not None
            and co_ref is not None
            and row["variant"] not in ("B0", "B2_class_only")
        ):
            row["fp_supp_gain_over_class_only"] = round(fp_t - co_ref, 4)

    # ── Bootstrap FP_supp_gain CI ──────────────────────────────────────────────
    print(f"\n[Stage3D.3] Bootstrap FP_supp_gain CI ({args.n_bootstrap} resamples)...")
    gain_rows: List[dict] = []

    for name, etype, _, _ in variants_spec:
        if name in ("B0",) or name not in scores_te:
            continue
        for tgt in _TP_TARGETS:
            th_m = all_thresholds.get((name, tgt))
            th_b0 = b0_th.get(tgt)
            if th_m is None or th_b0 is None:
                continue
            boot = _boot_gain(
                img_te,
                y_te,
                scores_te[name],
                scores_te["B0"],
                th_m,
                th_b0,
                args.n_bootstrap,
                _BOOTSTRAP_SEED,
            )
            sig = "YES" if (boot["ci_lo"] is not None and boot["ci_lo"] > 0) else "no"
            print(
                f"[Stage3D.3]   {name:<22} TP>={tgt:.2f}: "
                f"gain={_fmt(boot['mean_gain'], '+.4f')}  "
                f"CI=[{_fmt(boot['ci_lo'])},{_fmt(boot['ci_hi'])}]  "
                f"P(>0)={_fmt(boot['frac_positive'])}  sig={sig}"
            )
            gain_rows.append(
                {
                    "variant": name,
                    "tp_ret_target": tgt,
                    **boot,
                    "significant": sig,
                }
            )

    # ── Write outputs ──────────────────────────────────────────────────────────
    rp = out_dir / "epred_ablation_results.csv"
    _write_csv(res_rows, _RES_FIELDS, rp)
    print(f"\n[Stage3D.3] {rp}")

    gp = out_dir / "bootstrap_epred_fp_supp_gain_ci.csv"
    _write_csv(gain_rows, _GAIN_FIELDS, gp)
    print(f"[Stage3D.3] {gp}")

    md = out_dir / "epred_ablation_decision.md"
    _write_md(res_rows, gain_rows, variant_order, md)
    print(f"[Stage3D.3] {md}")

    print(f"\n[Stage3D.3] Done.")
    print(
        "[Stage3D.3] Key question: Does B_sdmrow or B_gat beat B2_class_only?\n"
        "  → YES: KG-derived class embedding adds governance signal.\n"
        "  → NO:  KG encoder needs training; reframe as framework contribution."
    )


if __name__ == "__main__":
    main()
