#!/usr/bin/env python
"""
eval_stage3d_operating_points.py
Phase 19N — Stage 3D.1: Operating Point Calibration for LGG baselines.

Sweeps decision thresholds on the validation split and selects operating
points at TP_retention >= 0.85, 0.80, 0.75 for each model.
Selected thresholds are evaluated on the held-out test split.

Decision rule:
  LGG is operationally useful only if it achieves greater FP_suppression
  than B0 at the same TP_retention constraint on the test split.

Inputs:
  outputs/stage3c_audit/governance_features_train.csv
  outputs/stage3c_audit/governance_features_valid.csv
  outputs/stage3c_audit/governance_features_test.csv

Outputs:
  outputs/stage3d_lgg/operating_point_results.csv
  outputs/stage3d_lgg/pareto_valid.csv
  outputs/stage3d_lgg/pareto_test.csv
  outputs/stage3d_lgg/bootstrap_ci.csv
  outputs/stage3d_lgg/stage3d_operating_point_decision.md

Usage:
    python tools/eval_stage3d_operating_points.py \\
      --stage3c_dir outputs/stage3c_audit \\
      --output_dir  outputs/stage3d_lgg

    python tools/eval_stage3d_operating_points.py --no_mlp  # LR + B0 only
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── sklearn import guard ───────────────────────────────────────────────────────

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False
    print("[Stage3D.1] WARNING: scikit-learn not installed. Only B0 will run.")

# ── Constants ──────────────────────────────────────────────────────────────────

_RETAIN_CATS = frozenset({"TP_EXACT", "CLINICALLY_CLOSE_MISMATCH", "LOCALIZATION_WEAK"})
_UNSAFE_CAT = "UNSAFE_MISMATCH"

_FEATURE_SETS: Dict[str, List[str]] = {
    "B0": ["pred_score"],
    "B1": ["ev_sas"],
    "B2": ["ev_sas", "ev_sdca"],
    "B3": ["ev_sas", "ev_sdca", "ev_aps", "ev_pss"],
}

_TP_TARGETS = [0.85, 0.80, 0.75]

# 499-point grid: 0.002, 0.004, ..., 0.998 (fine enough for clinical operating point)
_THRESHOLD_GRID = np.round(np.linspace(0.001, 0.999, 499), 4)

_N_BOOTSTRAP = 1000
_BOOTSTRAP_SEED = 42


# ── Data loading ───────────────────────────────────────────────────────────────


def _load_csv(path: Path) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _extract_all(
    records: List[dict],
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Extract all feature columns + labels from CSV records.

    Drops rows with missing or non-finite values in ANY column so that
    all models (B0-B3) operate on the exact same set of rows.

    Returns:
        features : dict col->float32 array
        y_true   : int32 array (1=RETAIN, 0=SUPPRESS)
        is_unsafe: bool array
        is_tp    : bool array (TP_EXACT category only)
        image_ids: list[str] aligned with above arrays
    """
    all_cols = ["pred_score", "ev_sas", "ev_sdca", "ev_aps", "ev_pss"]
    feat_lists: Dict[str, list] = {c: [] for c in all_cols}
    y_list: list = []
    unsafe_list: list = []
    tp_list: list = []
    img_list: list = []

    for r in records:
        try:
            vals = {}
            for c in all_cols:
                raw = r.get(c, "")
                if raw == "" or raw is None:
                    raise ValueError(f"empty {c}")
                vals[c] = float(raw)
        except (ValueError, KeyError):
            continue
        if any(not math.isfinite(v) for v in vals.values()):
            continue
        for c in all_cols:
            feat_lists[c].append(vals[c])
        y_list.append(int(r.get("label", 0)))
        unsafe_list.append(r.get("category") == _UNSAFE_CAT)
        tp_list.append(r.get("category") == "TP_EXACT")
        img_list.append(str(r.get("image_id", "")))

    features = {c: np.array(v, dtype=np.float32) for c, v in feat_lists.items()}
    return (
        features,
        np.array(y_list, dtype=np.int32),
        np.array(unsafe_list, dtype=bool),
        np.array(tp_list, dtype=bool),
        img_list,
    )


# ── Model training ─────────────────────────────────────────────────────────────


def _get_X(features: Dict[str, np.ndarray], cols: List[str]) -> np.ndarray:
    if len(cols) == 1:
        return features[cols[0]].reshape(-1, 1).astype(np.float64)
    return np.column_stack([features[c] for c in cols]).astype(np.float64)


def _train_lr(X: np.ndarray, y: np.ndarray) -> Tuple[object, object]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = LogisticRegression(
        max_iter=500, class_weight="balanced", random_state=42, solver="lbfgs"
    )
    model.fit(Xs, y)
    return model, scaler


def _train_mlp(X: np.ndarray, y: np.ndarray) -> Tuple[object, object]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        max_iter=300,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        learning_rate_init=0.001,
        verbose=False,
    )
    model.fit(Xs, y)
    return model, scaler


def _get_proba(model: object, scaler: object, X: np.ndarray) -> np.ndarray:
    Xs = scaler.transform(X)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(Xs)[:, 1].astype(np.float64)
    return model.decision_function(Xs).astype(np.float64)


# ── Metrics at a single threshold ─────────────────────────────────────────────


def _at_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    is_unsafe: np.ndarray,
    is_tp: np.ndarray,
    threshold: float,
) -> dict:
    y_pred = (y_score >= threshold).astype(np.int32)
    retain_mask = y_true == 1
    suppress_mask = y_true == 0
    n_retain = int(retain_mask.sum())
    n_suppress = int(suppress_mask.sum())
    n_unsafe = int(is_unsafe.sum())
    n_tp = int(is_tp.sum())

    tp_ret = int((retain_mask & (y_pred == 1)).sum()) / max(n_retain, 1)
    fp_supp = int((suppress_mask & (y_pred == 0)).sum()) / max(n_suppress, 1)
    tp_exact_ret = int((is_tp & (y_pred == 1)).sum()) / max(n_tp, 1)
    retained = int((y_pred == 1).sum())
    precision = int((y_true[y_pred == 1]).sum()) / max(retained, 1)
    recall_loss = 1.0 - tp_ret
    unsafe_supp: Optional[float] = (
        int((is_unsafe & (y_pred == 0)).sum()) / n_unsafe if n_unsafe > 0 else None
    )

    if _SKLEARN_OK and len(np.unique(y_true)) > 1:
        auprc: Optional[float] = float(average_precision_score(y_true, y_score))
    else:
        auprc = None

    def _r(v: Optional[float], d: int = 4) -> Optional[float]:
        return round(v, d) if v is not None else None

    return {
        "threshold": round(float(threshold), 4),
        "tp_ret": _r(tp_ret),
        "tp_exact_ret": _r(tp_exact_ret),
        "fp_supp": _r(fp_supp),
        "precision": _r(precision),
        "recall_loss": _r(recall_loss),
        "unsafe_supp": _r(unsafe_supp),
        "retained_total": retained,
        "n_boxes": int(len(y_true)),
        "auprc": _r(auprc),
    }


# ── Operating point selection ─────────────────────────────────────────────────


def _find_operating_point(
    y_true: np.ndarray,
    y_score: np.ndarray,
    tp_ret_target: float,
    thresholds: np.ndarray,
) -> Optional[float]:
    """Return the highest threshold where TP_ret >= target on this split.

    Higher threshold = more suppressive.  We sweep ascending and keep
    updating best_thresh whenever the target is still met, so the final
    value is the most suppressive operating point that satisfies the
    TP retention constraint.
    """
    retain_mask = y_true == 1
    n_retain = int(retain_mask.sum())
    if n_retain == 0:
        return None

    best_thresh: Optional[float] = None
    for t in sorted(thresholds):
        y_pred = (y_score >= float(t)).astype(np.int32)
        tp_ret = int((retain_mask & (y_pred == 1)).sum()) / n_retain
        if tp_ret >= tp_ret_target:
            best_thresh = float(t)
    return best_thresh


# ── Pareto curves ─────────────────────────────────────────────────────────────


def _pareto_rows(
    model_name: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    is_unsafe: np.ndarray,
    thresholds: np.ndarray,
) -> List[dict]:
    retain_mask = y_true == 1
    suppress_mask = y_true == 0
    n_retain = int(retain_mask.sum())
    n_suppress = int(suppress_mask.sum())
    n_unsafe = int(is_unsafe.sum())

    rows = []
    for t in thresholds:
        y_pred = (y_score >= float(t)).astype(np.int32)
        tp_ret = int((retain_mask & (y_pred == 1)).sum()) / max(n_retain, 1)
        fp_supp = int((suppress_mask & (y_pred == 0)).sum()) / max(n_suppress, 1)
        retained = int((y_pred == 1).sum())
        precision = int((y_true[y_pred == 1]).sum()) / max(retained, 1)
        unsafe_supp = (
            int((is_unsafe & (y_pred == 0)).sum()) / n_unsafe if n_unsafe > 0 else ""
        )
        rows.append({
            "model": model_name,
            "threshold": round(float(t), 4),
            "tp_ret": round(tp_ret, 4),
            "fp_supp": round(fp_supp, 4),
            "precision": round(precision, 4),
            "unsafe_supp": round(float(unsafe_supp), 4) if unsafe_supp != "" else "",
        })
    return rows


# ── Bootstrap AUPRC CI ────────────────────────────────────────────────────────


def _bootstrap_ci(
    image_ids: List[str],
    y_true: np.ndarray,
    model_scores: Dict[str, np.ndarray],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, dict]:
    """Bootstrap 95% CI for AUPRC by resampling images (not boxes)."""
    if not _SKLEARN_OK or len(y_true) == 0:
        return {}

    rng = np.random.default_rng(seed)
    # Deduplicate image list preserving insertion order
    unique_imgs: List[str] = list(dict.fromkeys(image_ids))
    n_images = len(unique_imgs)

    img_to_idx: Dict[str, List[int]] = {}
    for i, img in enumerate(image_ids):
        img_to_idx.setdefault(img, []).append(i)

    boot_auprcs: Dict[str, List[float]] = {nm: [] for nm in model_scores}

    for _ in range(n_bootstrap):
        sampled = rng.choice(unique_imgs, size=n_images, replace=True)
        indices: List[int] = []
        for img in sampled:
            indices.extend(img_to_idx[img])
        idx_arr = np.array(indices, dtype=np.int64)

        y_b = y_true[idx_arr]
        if y_b.sum() == 0 or (1 - y_b).sum() == 0:
            continue

        for nm, scores in model_scores.items():
            boot_auprcs[nm].append(
                float(average_precision_score(y_b, scores[idx_arr]))
            )

    result: Dict[str, dict] = {}
    for nm, auprcs in boot_auprcs.items():
        if len(auprcs) < 10:
            result[nm] = {"mean": None, "ci_lo": None, "ci_hi": None, "n": len(auprcs)}
        else:
            arr = np.array(auprcs)
            result[nm] = {
                "mean": round(float(arr.mean()), 4),
                "ci_lo": round(float(np.percentile(arr, 2.5)), 4),
                "ci_hi": round(float(np.percentile(arr, 97.5)), 4),
                "n": len(auprcs),
            }
    return result


# ── Formatting helpers ────────────────────────────────────────────────────────


def _f(v: Optional[float], fmt: str = ".4f") -> str:
    return format(v, fmt) if v is not None else "N/A"


# ── Decision document ─────────────────────────────────────────────────────────


def _write_md(
    op_rows: List[dict],
    boot_va: Dict[str, dict],
    boot_te: Dict[str, dict],
    model_order: List[str],
    path: Path,
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Stage 3D.1 — Operating Point Calibration Decision\n\n")
        f.write(
            "> **Decision rule**: LGG is operationally useful only if it achieves "
            "higher FP_suppression than B0 at the same TP_retention constraint "
            "on the **test split**.\n\n"
        )
        f.write(
            "Thresholds are selected on the **validation split** and applied to "
            "the **test split** without further tuning.\n\n"
        )

        # Per-target operating point tables
        for target in _TP_TARGETS:
            f.write(f"## Operating point: TP_ret ≥ {target:.2f}\n\n")
            f.write(
                "| Model | Thresh (V) | TP_ret V | FP_supp V "
                "| TP_ret T | FP_supp T | Prec T | AUPRC T | FP gain vs B0 |\n"
            )
            f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")

            b0_fp_t = None
            for row in op_rows:
                if abs(row.get("tp_ret_target", -1) - target) > 1e-6 and row["model"] == "B0":
                    continue
                if abs(row.get("tp_ret_target", -1) - target) < 1e-6 and row["model"] == "B0":
                    b0_fp_t = row.get("fp_supp_test")
                    break

            for name in model_order:
                row = next(
                    (r for r in op_rows
                     if r["model"] == name and abs(r.get("tp_ret_target", -1) - target) < 1e-6),
                    None,
                )
                if row is None:
                    continue
                th = row.get("thresh_selected_valid")
                gain = row.get("fp_supp_gain_over_b0")
                f.write(
                    f"| {name} "
                    f"| {_f(th) if th is not None else 'N/A'} "
                    f"| {_f(row.get('tp_ret_valid'))} "
                    f"| {_f(row.get('fp_supp_valid'))} "
                    f"| {_f(row.get('tp_ret_test'))} "
                    f"| {_f(row.get('fp_supp_test'))} "
                    f"| {_f(row.get('precision_test'))} "
                    f"| {_f(row.get('auprc_test'))} "
                    f"| {_f(gain, '+.4f') if gain is not None else '—'} |\n"
                )
            f.write("\n")

        # Bootstrap CI
        f.write("## Bootstrap AUPRC confidence intervals (1000 resamples, by image)\n\n")
        b0_va = boot_va.get("B0", {})
        b0_te = boot_te.get("B0", {})

        for split_label, ci_dict, b0_ref in [
            ("valid", boot_va, b0_va),
            ("test",  boot_te, b0_te),
        ]:
            f.write(f"### {split_label}\n\n")
            f.write(
                "| Model | Mean AUPRC | 95% CI | Gain vs B0 | Gain CI | Significant |\n"
            )
            f.write("|---|---:|---|---:|---|:---:|\n")
            b0_mn = b0_ref.get("mean")
            b0_lo = b0_ref.get("ci_lo")
            b0_hi = b0_ref.get("ci_hi")
            for name in model_order:
                c = ci_dict.get(name, {})
                mn = c.get("mean")
                lo = c.get("ci_lo")
                hi = c.get("ci_hi")
                ci_str = f"[{_f(lo)}, {_f(hi)}]" if lo is not None else "N/A"
                if mn is not None and b0_mn is not None:
                    gain = mn - b0_mn
                    g_lo = (lo - b0_hi) if (lo is not None and b0_hi is not None) else None
                    g_hi = (hi - b0_lo) if (hi is not None and b0_lo is not None) else None
                    gain_str = f"{gain:+.4f}"
                    gci_str = (
                        f"[{g_lo:+.4f}, {g_hi:+.4f}]"
                        if (g_lo is not None and g_hi is not None) else "N/A"
                    )
                    sig = "**YES**" if (g_lo is not None and g_lo > 0) else "no"
                else:
                    gain_str, gci_str, sig = "—", "N/A", "—"
                f.write(
                    f"| {name} | {_f(mn)} | {ci_str} "
                    f"| {gain_str} | {gci_str} | {sig} |\n"
                )
            f.write("\n")

        # Verdict
        f.write("## Verdict\n\n")
        any_useful = False
        lines = []
        for target in _TP_TARGETS:
            best_gain: Optional[float] = None
            best_model: Optional[str] = None
            for row in op_rows:
                if abs(row.get("tp_ret_target", -1) - target) > 1e-6:
                    continue
                if row["model"] == "B0":
                    continue
                g = row.get("fp_supp_gain_over_b0")
                if g is not None and (best_gain is None or g > best_gain):
                    best_gain = g
                    best_model = row["model"]
            if best_gain is not None and best_gain > 0:
                any_useful = True
                lines.append(
                    f"TP_ret ≥ {target:.2f}: {best_model} improves FP_supp "
                    f"by +{best_gain:.4f} over B0  → LGG USEFUL"
                )
            else:
                lines.append(
                    f"TP_ret ≥ {target:.2f}: No model improves FP_supp over B0  "
                    f"→ LGG not useful at this target"
                )

        f.write("```\n" + "\n".join(lines) + "\n```\n\n")

        if any_useful:
            f.write(
                "**LGG provides measurable FP suppression improvement at one or more "
                "TP retention targets. Contribution C2 has operational evidence.**\n"
            )
        else:
            f.write(
                "**LGG does not improve FP suppression over B0 at matched TP retention "
                "targets. Current contribution is limited to auditability and structured "
                "evidence (C1) rather than automatic performance gain.**\n"
            )

        f.write("\n## Limitations\n\n")
        f.write(
            "1. SDCA is a deterministic proxy — encodes (pred_class, score > threshold), "
            "not real detector uncertainty.\n"
            "2. APS = PSS = 0.5 (constants) → B3 = B2 by construction.\n"
            "3. LOCALIZATION_WEAK vs BACKGROUND_FP avg_SAS difference ≈ 0.05 → "
            "poor separability from evidence features alone.\n"
            "4. Gain CI uses conservative bound: [lo_model − hi_B0, hi_model − lo_B0].\n"
        )


# ── CSV writers ───────────────────────────────────────────────────────────────

_OP_FIELDS = [
    "model", "tp_ret_target", "thresh_selected_valid",
    "tp_ret_valid", "tp_exact_ret_valid", "fp_supp_valid",
    "precision_valid", "recall_loss_valid", "unsafe_supp_valid", "auprc_valid",
    "tp_ret_test", "tp_exact_ret_test", "fp_supp_test",
    "precision_test", "recall_loss_test", "unsafe_supp_test", "auprc_test",
    "fp_supp_gain_over_b0", "achieves_target_valid", "achieves_target_test",
]

_PARETO_FIELDS = ["model", "threshold", "tp_ret", "fp_supp", "precision", "unsafe_supp"]

_BOOT_FIELDS = [
    "model", "split", "mean_auprc", "ci_lo", "ci_hi", "n",
    "gain_vs_b0", "gain_ci_lo", "gain_ci_hi", "significant",
]


def _write_csv(rows: List[dict], fields: List[str], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 3D.1: Operating Point Calibration."
    )
    parser.add_argument(
        "--stage3c_dir", default="outputs/stage3c_audit",
        help="Directory with governance_features_*.csv from Stage 3C.",
    )
    parser.add_argument(
        "--output_dir", default="outputs/stage3d_lgg",
        help="Output directory (default: outputs/stage3d_lgg).",
    )
    parser.add_argument("--no_mlp", action="store_true", help="Skip MLP baselines.")
    parser.add_argument(
        "--n_bootstrap", type=int, default=_N_BOOTSTRAP,
        help=f"Bootstrap resamples (default: {_N_BOOTSTRAP}).",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    s3c_dir = Path(args.stage3c_dir)

    print("[Stage3D.1] === Phase 19N Stage 3D.1 Operating Point Calibration ===")
    print(f"[Stage3D.1] stage3c_dir={s3c_dir}  output_dir={out_dir}")

    # ── Load CSVs ──────────────────────────────────────────────────────────────
    recs: Dict[str, List[dict]] = {}
    for split in ["train", "valid", "test"]:
        fpath = s3c_dir / f"governance_features_{split}.csv"
        if not fpath.exists():
            print(f"[Stage3D.1] ERROR: {fpath} not found.")
            print(
                f"[Stage3D.1]   Run: python tools/audit_stage3c_governance_targets.py "
                f"--split {split} ..."
            )
            sys.exit(1)
        recs[split] = _load_csv(fpath)

    # ── Extract aligned arrays ─────────────────────────────────────────────────
    feat_tr, y_tr, unsafe_tr, tp_tr, _ = _extract_all(recs["train"])
    feat_va, y_va, unsafe_va, tp_va, img_va = _extract_all(recs["valid"])
    feat_te, y_te, unsafe_te, tp_te, img_te = _extract_all(recs["test"])

    for split, y in [("train", y_tr), ("valid", y_va), ("test", y_te)]:
        print(
            f"[Stage3D.1]   {split}: {len(y)} boxes  "
            f"RETAIN={y.sum()} ({100*y.mean():.1f}%)  "
            f"SUPPRESS={(1-y).sum()} ({100*(1-y.mean()):.1f}%)"
        )

    # ── Train models, collect probability scores ───────────────────────────────
    scores_va: Dict[str, np.ndarray] = {}
    scores_te: Dict[str, np.ndarray] = {}

    # B0: raw detection score (no model)
    scores_va["B0"] = feat_va["pred_score"].astype(np.float64)
    scores_te["B0"] = feat_te["pred_score"].astype(np.float64)
    print("[Stage3D.1] B0: pred_score used directly.")

    if _SKLEARN_OK:
        for bline in ["B1", "B2", "B3"]:
            cols = _FEATURE_SETS[bline]
            X_tr = _get_X(feat_tr, cols)
            X_va = _get_X(feat_va, cols)
            X_te = _get_X(feat_te, cols)

            print(f"[Stage3D.1] Training {bline} LR  features={cols}...")
            lr_m, lr_sc = _train_lr(X_tr, y_tr)
            scores_va[f"{bline}_LR"] = _get_proba(lr_m, lr_sc, X_va)
            scores_te[f"{bline}_LR"] = _get_proba(lr_m, lr_sc, X_te)

            if not args.no_mlp:
                print(f"[Stage3D.1] Training {bline} MLP...")
                mlp_m, mlp_sc = _train_mlp(X_tr, y_tr)
                n_iters = getattr(mlp_m, "n_iter_", "?")
                print(f"[Stage3D.1]   converged in {n_iters} iterations")
                scores_va[f"{bline}_MLP"] = _get_proba(mlp_m, mlp_sc, X_va)
                scores_te[f"{bline}_MLP"] = _get_proba(mlp_m, mlp_sc, X_te)

    # Ordered list of models present
    _candidates = ["B0"]
    for b in ["B1", "B2", "B3"]:
        _candidates += [f"{b}_LR"]
        if not args.no_mlp:
            _candidates += [f"{b}_MLP"]
    model_order = [m for m in _candidates if m in scores_va]
    print(f"[Stage3D.1] Models ready: {model_order}")

    # ── Pareto curves ──────────────────────────────────────────────────────────
    print(f"\n[Stage3D.1] Building Pareto curves ({len(_THRESHOLD_GRID)} thresholds)...")
    pareto_va_rows: List[dict] = []
    pareto_te_rows: List[dict] = []

    for name in model_order:
        pareto_va_rows.extend(
            _pareto_rows(name, y_va, scores_va[name], unsafe_va, _THRESHOLD_GRID)
        )
        pareto_te_rows.extend(
            _pareto_rows(name, y_te, scores_te[name], unsafe_te, _THRESHOLD_GRID)
        )

    # ── Operating point calibration ────────────────────────────────────────────
    print(f"\n[Stage3D.1] Selecting operating points (TP targets={_TP_TARGETS})...")

    # B0 reference FP_supp on test at each target (threshold selected on valid)
    b0_fp_supp_test: Dict[float, Optional[float]] = {}
    for target in _TP_TARGETS:
        th = _find_operating_point(y_va, scores_va["B0"], target, _THRESHOLD_GRID)
        if th is not None:
            m = _at_threshold(y_te, scores_te["B0"], unsafe_te, tp_te, th)
            b0_fp_supp_test[target] = m["fp_supp"]
        else:
            b0_fp_supp_test[target] = None

    op_rows: List[dict] = []

    for name in model_order:
        sc_va = scores_va[name]
        sc_te = scores_te[name]

        for target in _TP_TARGETS:
            best_thresh = _find_operating_point(y_va, sc_va, target, _THRESHOLD_GRID)

            if best_thresh is None:
                print(
                    f"[Stage3D.1]   {name:<12} TP>={target:.2f}: "
                    f"cannot reach target on valid — all boxes would be retained"
                )
                op_rows.append({
                    "model": name, "tp_ret_target": target,
                    "thresh_selected_valid": None,
                    **{k: None for k in _OP_FIELDS if k not in ("model", "tp_ret_target", "thresh_selected_valid")},
                    "achieves_target_valid": False, "achieves_target_test": None,
                })
                continue

            m_va = _at_threshold(y_va, sc_va, unsafe_va, tp_va, best_thresh)
            m_te = _at_threshold(y_te, sc_te, unsafe_te, tp_te, best_thresh)

            b0_ref = b0_fp_supp_test.get(target)
            fp_gain = (
                round((m_te["fp_supp"] - b0_ref), 4)
                if (b0_ref is not None and name != "B0")
                else None
            )

            row = {
                "model": name,
                "tp_ret_target": target,
                "thresh_selected_valid": best_thresh,
                "tp_ret_valid": m_va["tp_ret"],
                "tp_exact_ret_valid": m_va["tp_exact_ret"],
                "fp_supp_valid": m_va["fp_supp"],
                "precision_valid": m_va["precision"],
                "recall_loss_valid": m_va["recall_loss"],
                "unsafe_supp_valid": m_va["unsafe_supp"],
                "auprc_valid": m_va["auprc"],
                "tp_ret_test": m_te["tp_ret"],
                "tp_exact_ret_test": m_te["tp_exact_ret"],
                "fp_supp_test": m_te["fp_supp"],
                "precision_test": m_te["precision"],
                "recall_loss_test": m_te["recall_loss"],
                "unsafe_supp_test": m_te["unsafe_supp"],
                "auprc_test": m_te["auprc"],
                "fp_supp_gain_over_b0": fp_gain,
                "achieves_target_valid": (m_va["tp_ret"] or 0) >= target,
                "achieves_target_test": (m_te["tp_ret"] or 0) >= target,
            }
            op_rows.append(row)

            flag_t = "✓" if (m_te["tp_ret"] or 0) >= target else "✗"
            gain_str = f"{fp_gain:+.4f}" if fp_gain is not None else "(B0 ref)"
            print(
                f"[Stage3D.1]   {name:<12} TP>={target:.2f}: "
                f"thresh={best_thresh:.4f}  "
                f"TP_ret(V={_f(m_va['tp_ret'])}, T={_f(m_te['tp_ret'])}{flag_t})  "
                f"FP_supp(V={_f(m_va['fp_supp'])}, T={_f(m_te['fp_supp'])})  "
                f"gain={gain_str}"
            )

    # ── Bootstrap AUPRC CI ─────────────────────────────────────────────────────
    print(
        f"\n[Stage3D.1] Bootstrap AUPRC CI "
        f"({args.n_bootstrap} resamples by image_id)..."
    )
    boot_va = _bootstrap_ci(img_va, y_va, scores_va, args.n_bootstrap, _BOOTSTRAP_SEED)
    boot_te = _bootstrap_ci(img_te, y_te, scores_te, args.n_bootstrap, _BOOTSTRAP_SEED)

    # Print and collect bootstrap CSV rows
    boot_rows: List[dict] = []
    b0_va = boot_va.get("B0", {}); b0_te = boot_te.get("B0", {})
    b0_va_mn = b0_va.get("mean"); b0_va_lo = b0_va.get("ci_lo"); b0_va_hi = b0_va.get("ci_hi")
    b0_te_mn = b0_te.get("mean"); b0_te_lo = b0_te.get("ci_lo"); b0_te_hi = b0_te.get("ci_hi")

    print(f"  {'Model':<12} {'Valid AUPRC 95%CI':>30}  {'Test AUPRC 95%CI':>30}")
    for name in model_order:
        va_c = boot_va.get(name, {}); te_c = boot_te.get(name, {})
        va_mn = va_c.get("mean"); va_lo = va_c.get("ci_lo"); va_hi = va_c.get("ci_hi")
        te_mn = te_c.get("mean"); te_lo = te_c.get("ci_lo"); te_hi = te_c.get("ci_hi")

        va_str = (f"{va_mn:.4f} [{va_lo:.4f},{va_hi:.4f}]"
                  if va_mn is not None else "N/A")
        te_str = (f"{te_mn:.4f} [{te_lo:.4f},{te_hi:.4f}]"
                  if te_mn is not None else "N/A")
        print(f"  {name:<12} {va_str:>30}  {te_str:>30}")

        for split_lbl, c_mn, c_lo, c_hi, c_n, ref_mn, ref_lo, ref_hi in [
            ("valid", va_mn, va_lo, va_hi, va_c.get("n"),
             b0_va_mn, b0_va_lo, b0_va_hi),
            ("test",  te_mn, te_lo, te_hi, te_c.get("n"),
             b0_te_mn, b0_te_lo, b0_te_hi),
        ]:
            if c_mn is not None and ref_mn is not None:
                gain = c_mn - ref_mn
                g_lo = (c_lo - ref_hi) if (c_lo is not None and ref_hi is not None) else None
                g_hi = (c_hi - ref_lo) if (c_hi is not None and ref_lo is not None) else None
                sig = "YES" if (g_lo is not None and g_lo > 0) else "no"
            else:
                gain = g_lo = g_hi = None
                sig = "-"

            boot_rows.append({
                "model": name, "split": split_lbl,
                "mean_auprc": c_mn, "ci_lo": c_lo, "ci_hi": c_hi, "n": c_n,
                "gain_vs_b0": round(gain, 4) if gain is not None else None,
                "gain_ci_lo": round(g_lo, 4) if g_lo is not None else None,
                "gain_ci_hi": round(g_hi, 4) if g_hi is not None else None,
                "significant": sig,
            })

    # ── Write outputs ──────────────────────────────────────────────────────────
    op_path = out_dir / "operating_point_results.csv"
    _write_csv(op_rows, _OP_FIELDS, op_path)
    print(f"\n[Stage3D.1] {op_path}")

    pva_path = out_dir / "pareto_valid.csv"
    pte_path = out_dir / "pareto_test.csv"
    _write_csv(pareto_va_rows, _PARETO_FIELDS, pva_path)
    _write_csv(pareto_te_rows, _PARETO_FIELDS, pte_path)
    print(f"[Stage3D.1] {pva_path}")
    print(f"[Stage3D.1] {pte_path}")

    boot_path = out_dir / "bootstrap_ci.csv"
    _write_csv(boot_rows, _BOOT_FIELDS, boot_path)
    print(f"[Stage3D.1] {boot_path}")

    md_path = out_dir / "stage3d_operating_point_decision.md"
    _write_md(op_rows, boot_va, boot_te, model_order, md_path)
    print(f"[Stage3D.1] {md_path}")

    print(f"\n[Stage3D.1] Done. See stage3d_operating_point_decision.md for verdict.")
    print(
        "[Stage3D.1] Key question answered:\n"
        "  Does any LGG model suppress more FP than B0 at the same TP_retention? "
        "→ check fp_supp_gain_over_b0 column in operating_point_results.csv"
    )


if __name__ == "__main__":
    main()
