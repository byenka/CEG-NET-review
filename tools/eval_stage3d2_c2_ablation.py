#!/usr/bin/env python
"""
eval_stage3d2_c2_ablation.py
Phase 19N — Stage 3D.2: C2 Ablation Audit.

Determines whether B2_MLP gain over B0 (confirmed in Stage 3D.1) comes from
clinical KG-derived SDCA, or merely from nonlinear score/class transformation.

Ablation variants tested (all use MLP + SAS, varying only the second feature):
  B0               : score threshold (baseline, no model)
  B1               : [SAS]
  B2_clinical      : [SAS, SDCA_clinical]  ← model under evaluation
  B2_uniform       : [SAS, 0.5]            ← SDM collapses to uniform distance
  B2_shuffled      : [SAS, SDCA_shuffled]  ← SDCA shuffled in train, orig in val/test
  B2_score_pct     : [SAS, score_percentile] ← replace SDCA with score rank
  B2_class_only    : [SAS, one_hot(pred_class)] ← 5 features: SAS + 4 class bits
  B2_random_sdm_s* : [SAS, SDCA_random]  ← random class-conditioned values, 5 seeds

Decision rule:
    KG-derived SDCA contribution is only confirmed if B2_clinical consistently
    and significantly beats B0 plus the main artifact controls
    (B2_uniform/B2_shuffled/B2_score_pct/B2_class_only/B2_random_sdm)
    across TP_ret targets on the test split.

Inputs:
  outputs/stage3c_audit/governance_features_{train,valid,test}.csv

Outputs:
  outputs/stage3d_ablation/c2_ablation_results.csv
  outputs/stage3d_ablation/bootstrap_fp_supp_gain_ci.csv
  outputs/stage3d_ablation/c2_ablation_decision.md

Usage:
    python tools/eval_stage3d2_c2_ablation.py \\
      --stage3c_dir outputs/stage3c_audit \\
      --output_dir  outputs/stage3d_ablation

    python tools/eval_stage3d2_c2_ablation.py --lr_only  # skip MLP (faster debug)
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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── sklearn guard ─────────────────────────────────────────────────────────────
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    _SK = True
except ImportError:
    _SK = False
    print("[Stage3D.2] WARNING: scikit-learn not installed. Only B0 will run.")

# ── Constants ─────────────────────────────────────────────────────────────────

_RETAIN_CATS = frozenset({"TP_EXACT", "CLINICALLY_CLOSE_MISMATCH", "LOCALIZATION_WEAK"})
_UNSAFE_CAT = "UNSAFE_MISMATCH"

# Class ordering used for one-hot encoding and random SDM runner-up computation
_CLASS_ORDER = ["MA", "HE", "Hard_EX", "Soft_EX"]
_CLASS_IDX = {c: i for i, c in enumerate(_CLASS_ORDER)}

# Runner-up for each class: first non-predicted class by index order
# MA(0) → HE(1), HE(1) → MA(0), Hard_EX(2) → MA(0), Soft_EX(3) → MA(0)
_RUNNER_UP_IDX = {0: 1, 1: 0, 2: 0, 3: 0}

_N_RANDOM_SEEDS = 5
_TP_TARGETS = [0.85, 0.80, 0.75]
_THRESHOLD_GRID = np.round(np.linspace(0.001, 0.999, 499), 4)
_N_BOOTSTRAP = 1000
_BOOTSTRAP_SEED = 42


# ── Data loading ──────────────────────────────────────────────────────────────


def _load_csv(path: Path) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _parse_records(records: List[dict]) -> Tuple[
    Dict[str, np.ndarray],  # base features + ev_sdca_clinical, one-hot, score_pct
    np.ndarray,  # y_true
    np.ndarray,  # is_unsafe
    np.ndarray,  # is_tp
    List[str],  # image_ids
]:
    """Parse records into base feature arrays.

    Returns features dict with:
      pred_score, ev_sas, ev_sdca_clinical,
      class_MA, class_HE, class_Hard_EX, class_Soft_EX,
      class_idx  (int array, for random SDM lookup)
    """
    keys_float = ["pred_score", "ev_sas", "ev_sdca", "ev_aps", "ev_pss"]
    feat_lists: Dict[str, list] = {k: [] for k in keys_float}
    feat_lists["class_idx"] = []
    y_list: list = []
    unsafe_list: list = []
    tp_list: list = []
    img_list: list = []

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

        pred_class_str = r.get("pred_class", "")
        if pred_class_str not in _CLASS_IDX:
            continue

        for k in keys_float:
            feat_lists[k].append(vals[k])
        feat_lists["class_idx"].append(_CLASS_IDX[pred_class_str])
        y_list.append(int(r.get("label", 0)))
        unsafe_list.append(r.get("category") == _UNSAFE_CAT)
        tp_list.append(r.get("category") == "TP_EXACT")
        img_list.append(str(r.get("image_id", "")))

    n = len(y_list)
    feat = {
        k: np.array(v, dtype=np.float32)
        for k, v in feat_lists.items()
        if k != "class_idx"
    }
    feat["class_idx"] = np.array(feat_lists["class_idx"], dtype=np.int32)
    # Rename ev_sdca → ev_sdca_clinical
    feat["ev_sdca_clinical"] = feat.pop("ev_sdca")
    # Compute percentile rank of pred_score (normalized 0-1)
    if n > 0:
        from scipy.stats import rankdata  # type: ignore[import]

        feat["score_pct"] = (rankdata(feat["pred_score"]) / n).astype(np.float32)
    else:
        feat["score_pct"] = np.zeros(n, dtype=np.float32)
    # One-hot encodings
    for i, cls in enumerate(_CLASS_ORDER):
        feat[f"class_{cls}"] = (feat["class_idx"] == i).astype(np.float32)

    return (
        feat,
        np.array(y_list, dtype=np.int32),
        np.array(unsafe_list, dtype=bool),
        np.array(tp_list, dtype=bool),
        img_list,
    )


# ── SDCA variant generators ───────────────────────────────────────────────────


def _make_sdca_uniform(n: int) -> np.ndarray:
    """All class pairs assigned distance 0.5 (uniform SDM)."""
    return np.full(n, 0.5, dtype=np.float32)


def _make_sdca_shuffled(sdca_clinical: np.ndarray, seed: int = 42) -> np.ndarray:
    """Shuffle clinical SDCA values within this split (breaks semantic meaning)."""
    rng = np.random.default_rng(seed)
    shuffled = sdca_clinical.copy()
    rng.shuffle(shuffled)
    return shuffled


def _make_sdca_random(class_idx: np.ndarray, seed: int) -> np.ndarray:
    """Random class-conditioned SDCA using a random SDM (one per seed).

    Generates a random 4x4 symmetric distance matrix (diag=0, values in [0,1]).
    Assigns SDCA[box] = random_SDM[pred_class_idx, runner_up_idx].
    Runner-up = first non-predicted class by index order.
    """
    rng = np.random.default_rng(seed + 1000)  # offset to avoid overlap with other seeds
    # Random symmetric 4x4 SDM
    raw = rng.uniform(0.0, 1.0, size=(4, 4)).astype(np.float32)
    sdm = (raw + raw.T) / 2.0
    np.fill_diagonal(sdm, 0.0)

    sdca = np.zeros(len(class_idx), dtype=np.float32)
    for box_i, pred_idx in enumerate(class_idx):
        runner_idx = _RUNNER_UP_IDX[int(pred_idx)]
        sdca[box_i] = sdm[int(pred_idx), runner_idx]
    return sdca


def _build_feature_matrix(
    feat: Dict[str, np.ndarray],
    variant: str,
    shuffle_seed: int = 99,
    random_seed: int = 0,
    is_train: bool = False,
) -> np.ndarray:
    """Build feature matrix X for a given ablation variant.

    variant must be one of:
      B0, B1, B2_clinical, B2_uniform, B2_shuffled,
      B2_score_pct, B2_class_only, B2_random_sdm
    """
    sas = feat["ev_sas"].reshape(-1, 1)

    if variant == "B0":
        return feat["pred_score"].reshape(-1, 1)
    elif variant == "B1":
        return sas
    elif variant == "B2_clinical":
        return np.column_stack([sas, feat["ev_sdca_clinical"]])
    elif variant == "B2_uniform":
        return np.column_stack([sas, _make_sdca_uniform(len(sas))])
    elif variant == "B2_shuffled":
        # Shuffle train to break semantic meaning; use original for val/test
        sdca = (
            _make_sdca_shuffled(feat["ev_sdca_clinical"], seed=shuffle_seed)
            if is_train
            else feat["ev_sdca_clinical"]
        )
        return np.column_stack([sas, sdca])
    elif variant == "B2_score_pct":
        return np.column_stack([sas, feat["score_pct"]])
    elif variant == "B2_class_only":
        cols = [sas] + [feat[f"class_{c}"].reshape(-1, 1) for c in _CLASS_ORDER]
        return np.column_stack(cols)
    elif variant == "B2_random_sdm":
        sdca_r = _make_sdca_random(feat["class_idx"], seed=random_seed)
        return np.column_stack([sas, sdca_r])
    else:
        raise ValueError(f"Unknown variant: {variant}")


# ── Model training ─────────────────────────────────────────────────────────────


def _train_lr(X: np.ndarray, y: np.ndarray) -> Tuple[object, object]:
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    m = LogisticRegression(
        max_iter=500, class_weight="balanced", random_state=42, solver="lbfgs"
    )
    m.fit(Xs, y)
    return m, sc


def _train_mlp(X: np.ndarray, y: np.ndarray, seed: int = 42) -> Tuple[object, object]:
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    m = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        max_iter=300,
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        learning_rate_init=0.001,
        verbose=False,
    )
    m.fit(Xs, y)
    return m, sc


def _proba(model: object, sc: object, X: np.ndarray) -> np.ndarray:
    Xs = sc.transform(X)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(Xs)[:, 1].astype(np.float64)
    return model.decision_function(Xs).astype(np.float64)


# ── Threshold operations ───────────────────────────────────────────────────────


def _find_op_thresh(
    y_true: np.ndarray, y_score: np.ndarray, target: float
) -> Optional[float]:
    """Highest threshold on this split where TP_ret >= target."""
    retain = y_true == 1
    n_ret = int(retain.sum())
    if n_ret == 0:
        return None
    best: Optional[float] = None
    for t in sorted(_THRESHOLD_GRID):
        tp_ret = int((retain & (y_score >= float(t))).sum()) / n_ret
        if tp_ret >= target:
            best = float(t)
    return best


def _at_thresh(
    y_true: np.ndarray, y_score: np.ndarray, is_unsafe: np.ndarray, thresh: float
) -> dict:
    y_pred = (y_score >= thresh).astype(np.int32)
    ret = y_true == 1
    sup = y_true == 0
    n_ret = int(ret.sum())
    n_sup = int(sup.sum())
    n_un = int(is_unsafe.sum())
    tp_ret = int((ret & (y_pred == 1)).sum()) / max(n_ret, 1)
    fp_sup = int((sup & (y_pred == 0)).sum()) / max(n_sup, 1)
    n_kept = int((y_pred == 1).sum())
    prec = int((y_true[y_pred == 1]).sum()) / max(n_kept, 1)
    un_sup = int((is_unsafe & (y_pred == 0)).sum()) / n_un if n_un > 0 else None
    auprc = (
        float(average_precision_score(y_true, y_score))
        if (_SK and len(np.unique(y_true)) > 1)
        else None
    )
    return {
        "tp_ret": round(tp_ret, 4),
        "fp_supp": round(fp_sup, 4),
        "precision": round(prec, 4),
        "recall_loss": round(1.0 - tp_ret, 4),
        "unsafe_supp": round(float(un_sup), 4) if un_sup is not None else None,
        "auprc": round(auprc, 4) if auprc is not None else None,
        "retained": n_kept,
    }


# ── Bootstrap: FP_supp_gain CI ────────────────────────────────────────────────


def _bootstrap_fp_supp_gain(
    image_ids: List[str],
    y_true: np.ndarray,
    score_model: np.ndarray,
    score_b0: np.ndarray,
    thresh_model: float,
    thresh_b0: float,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """Bootstrap 95% CI for FP_supp_gain = FP_supp(model) - FP_supp(B0).

    Resamples images (with replacement) from the TEST split.
    Thresholds are fixed (selected from validation); only test set is bootstrapped.
    """
    if not _SK:
        return {
            "mean_gain": None,
            "ci_lo": None,
            "ci_hi": None,
            "frac_positive": None,
            "n": 0,
        }

    rng = np.random.default_rng(seed)
    unique_imgs = list(dict.fromkeys(image_ids))
    n_imgs = len(unique_imgs)
    img_to_idx: Dict[str, List[int]] = {}
    for i, img in enumerate(image_ids):
        img_to_idx.setdefault(img, []).append(i)

    gains: List[float] = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(unique_imgs, size=n_imgs, replace=True)
        idx: List[int] = []
        for img in sampled:
            idx.extend(img_to_idx[img])
        idx_arr = np.array(idx, dtype=np.int64)

        y_b = y_true[idx_arr]
        if y_b.sum() == 0 or (1 - y_b).sum() == 0:
            continue
        s_m = score_model[idx_arr]
        s_b = score_b0[idx_arr]

        ret = y_b == 1
        sup = y_b == 0
        n_s = int(sup.sum())
        if n_s == 0:
            continue

        fp_m = int((sup & (s_m >= thresh_model)).sum()) / n_s
        fp_b = int((sup & (s_b >= thresh_b0)).sum()) / n_s
        # fp_supp = fraction of suppressed
        fp_supp_m = int((sup & (s_m < thresh_model)).sum()) / n_s
        fp_supp_b = int((sup & (s_b < thresh_b0)).sum()) / n_s
        gains.append(fp_supp_m - fp_supp_b)

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


# ── Bootstrap: AUPRC CI ───────────────────────────────────────────────────────


def _bootstrap_auprc(
    image_ids: List[str],
    y_true: np.ndarray,
    scores_dict: Dict[str, np.ndarray],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, dict]:
    if not _SK:
        return {}
    rng = np.random.default_rng(seed)
    unique_imgs = list(dict.fromkeys(image_ids))
    n_imgs = len(unique_imgs)
    img_to_idx: Dict[str, List[int]] = {}
    for i, img in enumerate(image_ids):
        img_to_idx.setdefault(img, []).append(i)

    boot: Dict[str, List[float]] = {nm: [] for nm in scores_dict}
    for _ in range(n_bootstrap):
        sampled = rng.choice(unique_imgs, size=n_imgs, replace=True)
        idx: List[int] = []
        for img in sampled:
            idx.extend(img_to_idx[img])
        idx_arr = np.array(idx, dtype=np.int64)
        y_b = y_true[idx_arr]
        if y_b.sum() == 0 or (1 - y_b).sum() == 0:
            continue
        for nm, sc in scores_dict.items():
            boot[nm].append(float(average_precision_score(y_b, sc[idx_arr])))

    result: Dict[str, dict] = {}
    for nm, vals in boot.items():
        if len(vals) < 10:
            result[nm] = {"mean": None, "ci_lo": None, "ci_hi": None, "n": len(vals)}
        else:
            arr = np.array(vals)
            result[nm] = {
                "mean": round(float(arr.mean()), 4),
                "ci_lo": round(float(np.percentile(arr, 2.5)), 4),
                "ci_hi": round(float(np.percentile(arr, 97.5)), 4),
                "n": len(vals),
            }
    return result


# ── Output writers ────────────────────────────────────────────────────────────

_ABL_FIELDS = [
    "variant",
    "tp_ret_target",
    "model_type",
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
    "achieves_target_valid",
    "achieves_target_test",
]

_BOOT_GAIN_FIELDS = [
    "variant",
    "tp_ret_target",
    "model_type",
    "mean_gain",
    "ci_lo",
    "ci_hi",
    "frac_positive",
    "n",
    "significant",
]


def _f(v: Optional[float], fmt: str = ".4f") -> str:
    return format(v, fmt) if v is not None else "N/A"


def _write_csv(rows: List[dict], fields: List[str], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _write_decision_md(
    abl_rows: List[dict],
    gain_ci_rows: List[dict],
    auprc_ci_te: Dict[str, dict],
    variant_order: List[str],
    path: Path,
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Stage 3D.2 — C2 Ablation Decision\n\n")
        f.write(
            "> **Decision rule**: KG-derived SDCA contributes meaningfully to C2 if "
            "B2_clinical consistently and significantly outperforms B0 plus the main artifact controls "
            "(B2_uniform, B2_shuffled, B2_score_pct, B2_class_only, B2_random_sdm).\n\n"
        )

        # Per-target FP_supp_gain table
        for target in _TP_TARGETS:
            f.write(f"## TP_ret target ≥ {target:.2f}\n\n")
            f.write(
                "| Variant | Type | Thresh V | TP_ret V | FP_supp V "
                "| TP_ret T | FP_supp T | Gain vs B0 |\n"
            )
            f.write("|---|---|---:|---:|---:|---:|---:|---:|\n")
            for name in variant_order:
                row = next(
                    (
                        r
                        for r in abl_rows
                        if r["variant"] == name
                        and abs(r.get("tp_ret_target", -1) - target) < 1e-6
                    ),
                    None,
                )
                if row is None:
                    continue
                gain = row.get("fp_supp_gain_over_b0")
                f.write(
                    f"| {name} | {row.get('model_type','?')} "
                    f"| {_f(row.get('thresh_valid'))} "
                    f"| {_f(row.get('tp_ret_valid'))} "
                    f"| {_f(row.get('fp_supp_valid'))} "
                    f"| {_f(row.get('tp_ret_test'))} "
                    f"| {_f(row.get('fp_supp_test'))} "
                    f"| {_f(gain, '+.4f') if gain is not None else '—'} |\n"
                )
            f.write("\n")

        # Bootstrap FP_supp_gain CI
        f.write("## Bootstrap FP_supp_gain CI (1000 resamples, by image)\n\n")
        for target in _TP_TARGETS:
            f.write(f"### TP_ret ≥ {target:.2f}\n\n")
            f.write("| Variant | Mean gain | 95% CI | Frac>0 | Significant |\n")
            f.write("|---|---:|---|---:|:---:|\n")
            for name in variant_order:
                row = next(
                    (
                        r
                        for r in gain_ci_rows
                        if r["variant"] == name
                        and abs(r.get("tp_ret_target", -1) - target) < 1e-6
                        and r.get("model_type") == "MLP"
                    ),
                    None,
                )
                if row is None:
                    continue
                ci_str = (
                    f"[{_f(row.get('ci_lo'))}, {_f(row.get('ci_hi'))}]"
                    if row.get("ci_lo") is not None
                    else "N/A"
                )
                f.write(
                    f"| {name} "
                    f"| {_f(row.get('mean_gain'), '+.4f') if row.get('mean_gain') is not None else 'N/A'} "
                    f"| {ci_str} "
                    f"| {_f(row.get('frac_positive')) if row.get('frac_positive') is not None else 'N/A'} "
                    f"| {row.get('significant', '-')} |\n"
                )
            f.write("\n")

        # AUPRC CI
        f.write("## AUPRC bootstrap CI (test split)\n\n")
        f.write("| Variant | Mean AUPRC | 95% CI | Gain vs B0 | Significant |\n")
        f.write("|---|---:|---|---:|:---:|\n")
        b0_c = auprc_ci_te.get("B0", {})
        b0_mn = b0_c.get("mean")
        b0_lo = b0_c.get("ci_lo")
        b0_hi = b0_c.get("ci_hi")
        for name in variant_order:
            c = auprc_ci_te.get(name, {})
            mn = c.get("mean")
            lo = c.get("ci_lo")
            hi = c.get("ci_hi")
            ci_str = f"[{_f(lo)}, {_f(hi)}]" if lo is not None else "N/A"
            if mn is not None and b0_mn is not None:
                gain = mn - b0_mn
                g_lo = (lo - b0_hi) if (lo and b0_hi) else None
                sig = "YES" if (g_lo is not None and g_lo > 0) else "no"
                gain_str = f"{gain:+.4f}"
            else:
                gain_str, sig = "—", "—"
            f.write(f"| {name} | {_f(mn)} | {ci_str} | {gain_str} | {sig} |\n")
        f.write("\n")

        # Verdict
        f.write("## C2 Ablation Verdict\n\n")
        f.write("```\n")

        # Check if B2_clinical beats all controls consistently
        controls = [
            "B2_uniform_MLP",
            "B2_shuffled_MLP",
            "B2_score_pct_MLP",
            "B2_class_only_MLP",
        ]
        random_names = [f"B2_random_sdm_s{s}_MLP" for s in range(_N_RANDOM_SEEDS)]

        clinical_wins = 0
        total_comparisons = 0
        positive_vs_b0_targets = 0
        sig_positive_vs_b0_targets = 0
        score_pct_better_targets = 0
        for target in _TP_TARGETS:
            clin_row = next(
                (
                    r
                    for r in abl_rows
                    if r["variant"] == "B2_clinical"
                    and r.get("model_type") == "MLP"
                    and abs(r.get("tp_ret_target", -1) - target) < 1e-6
                ),
                None,
            )
            clin_gain = clin_row.get("fp_supp_gain_over_b0") if clin_row else None
            if clin_gain is not None and clin_gain > 0:
                positive_vs_b0_targets += 1

            clin_ci = next(
                (
                    r
                    for r in gain_ci_rows
                    if r["variant"] == "B2_clinical"
                    and r.get("model_type") == "MLP"
                    and abs(r.get("tp_ret_target", -1) - target) < 1e-6
                ),
                None,
            )
            if (
                clin_ci is not None
                and clin_ci.get("ci_lo") is not None
                and clin_ci.get("ci_lo") > 0
            ):
                sig_positive_vs_b0_targets += 1

            score_pct_row = next(
                (
                    r
                    for r in abl_rows
                    if r["variant"] == "B2_score_pct"
                    and r.get("model_type") == "MLP"
                    and abs(r.get("tp_ret_target", -1) - target) < 1e-6
                ),
                None,
            )
            score_pct_gain = (
                score_pct_row.get("fp_supp_gain_over_b0") if score_pct_row else None
            )
            if (
                clin_gain is not None
                and score_pct_gain is not None
                and score_pct_gain > clin_gain
            ):
                score_pct_better_targets += 1

            for ctrl in controls + random_names:
                ctrl_vname = ctrl.replace("_MLP", "")
                ctrl_row = next(
                    (
                        r
                        for r in abl_rows
                        if r["variant"] == ctrl_vname
                        and r.get("model_type") == "MLP"
                        and abs(r.get("tp_ret_target", -1) - target) < 1e-6
                    ),
                    None,
                )
                ctrl_gain = ctrl_row.get("fp_supp_gain_over_b0") if ctrl_row else None
                total_comparisons += 1
                if (
                    clin_gain is not None
                    and ctrl_gain is not None
                    and clin_gain > ctrl_gain
                ):
                    clinical_wins += 1

        win_rate = clinical_wins / max(total_comparisons, 1)

        f.write(
            f"B2_clinical win rate over controls: "
            f"{clinical_wins}/{total_comparisons} ({100*win_rate:.0f}%)\n\n"
        )
        f.write(
            f"Positive FP_supp gain over B0 at {positive_vs_b0_targets}/{len(_TP_TARGETS)} TP targets\n"
        )
        f.write(
            f"Significant positive gain over B0 at {sig_positive_vs_b0_targets}/{len(_TP_TARGETS)} TP targets\n"
        )
        f.write(
            f"B2_score_pct beats B2_clinical at {score_pct_better_targets}/{len(_TP_TARGETS)} TP targets\n\n"
        )

        if (
            win_rate >= 0.75
            and sig_positive_vs_b0_targets == len(_TP_TARGETS)
            and score_pct_better_targets == 0
        ):
            f.write(
                "VERDICT: B2_clinical consistently outperforms controls.\n"
                "→ KG-derived SDCA provides genuine signal beyond score/class artifacts.\n"
                "→ C2 claim is supported by ablation evidence.\n"
            )
        elif (
            win_rate >= 0.50
            and positive_vs_b0_targets == len(_TP_TARGETS)
            and score_pct_better_targets == 0
        ):
            f.write(
                "VERDICT: B2_clinical shows mixed evidence, but not a clean win.\n"
                "→ Any gain is fragile because it is not significant against B0 across all TP targets.\n"
                "→ C2 language must stay qualified and cannot be the main performance headline.\n"
            )
        else:
            f.write(
                "VERDICT: B2_clinical does NOT consistently outperform controls.\n"
                "→ The current evidence does not show a robust gain over B0 and key score/class controls.\n"
                "→ Any apparent gain is more plausibly explained by nonlinear score/class transformation than by KG alone.\n"
                "→ C2 claim must be limited to structured auditability, not performance gain.\n"
                "→ Reframe contribution as: governance framework + certificate + C1 clinical KG.\n"
            )
        f.write("```\n\n")

        f.write("## Interpretation guide\n\n")
        f.write(
            "| Result | Meaning |\n"
            "|---|---|\n"
            "| B2_clinical > B2_uniform | Clinical distances matter (not just any constant) |\n"
            "| B2_clinical > B2_shuffled | SDCA carries semantic content, not just distributional properties |\n"
            "| B2_clinical > B2_score_pct | SDCA adds signal beyond score rank |\n"
            "| B2_clinical > B2_class_only | Clinical distance structure matters, not just class identity |\n"
            "| B2_clinical > B2_random_sdm | Clinical SDM values specifically matter, not just any class mapping |\n"
        )

        f.write("\n## Limitations\n\n")
        f.write(
            "1. SDCA is still a deterministic proxy from the current single-head detector bridge.\n"
            "2. This ablation intentionally isolates the SAS+SDCA slice only; APS/PSS are not part of the tested C2 claim here.\n"
            "3. B2_shuffled trains on shuffled train SDCA but evaluates on original val/test.\n"
            "   Interpretation: tests whether SDCA structure in train matters to the learned model.\n"
            "4. B2_random_sdm uses runner_up = first non-predicted class by index order.\n"
            "   This is an approximation; true runner_up requires re-running evidence pipeline.\n"
        )


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 3D.2: C2 Ablation — does gain come from clinical SDCA or artifact?"
    )
    parser.add_argument("--stage3c_dir", default="outputs/stage3c_audit")
    parser.add_argument("--output_dir", default="outputs/stage3d_ablation")
    parser.add_argument(
        "--lr_only",
        action="store_true",
        help="Train LR only instead of MLP (faster, for debugging).",
    )
    parser.add_argument("--n_bootstrap", type=int, default=_N_BOOTSTRAP)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    s3c = Path(args.stage3c_dir)

    print("[Stage3D.2] === Phase 19N Stage 3D.2 C2 Ablation ===")
    print(f"[Stage3D.2] stage3c_dir={s3c}  output_dir={out_dir}")

    # ── Load CSVs ──────────────────────────────────────────────────────────────
    recs: Dict[str, List[dict]] = {}
    for split in ["train", "valid", "test"]:
        fpath = s3c / f"governance_features_{split}.csv"
        if not fpath.exists():
            print(f"[Stage3D.2] ERROR: {fpath} not found.")
            sys.exit(1)
        recs[split] = _load_csv(fpath)

    # ── Parse to feature arrays ───────────────────────────────────────────────
    print("[Stage3D.2] Parsing feature arrays...")
    feat_tr, y_tr, un_tr, tp_tr, _ = _parse_records(recs["train"])
    feat_va, y_va, un_va, tp_va, img_va = _parse_records(recs["valid"])
    feat_te, y_te, un_te, tp_te, img_te = _parse_records(recs["test"])

    for split, y in [("train", y_tr), ("valid", y_va), ("test", y_te)]:
        print(
            f"[Stage3D.2]   {split}: {len(y)} boxes  "
            f"RETAIN={y.sum()} ({100*y.mean():.1f}%)  "
            f"SUPPRESS={(1-y).sum()} ({100*(1-y.mean()):.1f}%)"
        )

    # ── Define ablation variants ───────────────────────────────────────────────
    # Each entry: (variant_name, model_type, X_tr, X_va, X_te)
    # model_type: "threshold" | "LR" | "MLP"
    Variant = Tuple[str, str, np.ndarray, np.ndarray, np.ndarray]
    variants: List[Variant] = []
    model_type = "LR" if args.lr_only else "MLP"

    def _add(
        name: str, mtype: str, X_tr: np.ndarray, X_va: np.ndarray, X_te: np.ndarray
    ) -> None:
        variants.append((name, mtype, X_tr, X_va, X_te))

    # B0: score threshold
    _add(
        "B0",
        "threshold",
        feat_tr["pred_score"].reshape(-1, 1),
        feat_va["pred_score"].reshape(-1, 1),
        feat_te["pred_score"].reshape(-1, 1),
    )

    # B1: SAS only
    _add(
        "B1",
        model_type,
        _build_feature_matrix(feat_tr, "B1"),
        _build_feature_matrix(feat_va, "B1"),
        _build_feature_matrix(feat_te, "B1"),
    )

    # B2_clinical: SAS + clinical SDCA
    _add(
        "B2_clinical",
        model_type,
        _build_feature_matrix(feat_tr, "B2_clinical"),
        _build_feature_matrix(feat_va, "B2_clinical"),
        _build_feature_matrix(feat_te, "B2_clinical"),
    )

    # B2_uniform: SAS + 0.5 constant
    _add(
        "B2_uniform",
        model_type,
        _build_feature_matrix(feat_tr, "B2_uniform"),
        _build_feature_matrix(feat_va, "B2_uniform"),
        _build_feature_matrix(feat_te, "B2_uniform"),
    )

    # B2_shuffled: SAS + shuffled SDCA in train, original in val/test
    _add(
        "B2_shuffled",
        model_type,
        _build_feature_matrix(feat_tr, "B2_shuffled", is_train=True),
        _build_feature_matrix(feat_va, "B2_shuffled", is_train=False),
        _build_feature_matrix(feat_te, "B2_shuffled", is_train=False),
    )

    # B2_score_pct: SAS + score percentile
    _add(
        "B2_score_pct",
        model_type,
        _build_feature_matrix(feat_tr, "B2_score_pct"),
        _build_feature_matrix(feat_va, "B2_score_pct"),
        _build_feature_matrix(feat_te, "B2_score_pct"),
    )

    # B2_class_only: SAS + one-hot pred_class
    _add(
        "B2_class_only",
        model_type,
        _build_feature_matrix(feat_tr, "B2_class_only"),
        _build_feature_matrix(feat_va, "B2_class_only"),
        _build_feature_matrix(feat_te, "B2_class_only"),
    )

    # B2_random_sdm x5 seeds
    for seed in range(_N_RANDOM_SEEDS):
        _add(
            f"B2_random_sdm_s{seed}",
            model_type,
            _build_feature_matrix(feat_tr, "B2_random_sdm", random_seed=seed),
            _build_feature_matrix(feat_va, "B2_random_sdm", random_seed=seed),
            _build_feature_matrix(feat_te, "B2_random_sdm", random_seed=seed),
        )

    variant_order = [v[0] for v in variants]
    print(f"[Stage3D.2] Variants: {variant_order}")

    # ── Train models, collect scores ───────────────────────────────────────────
    scores_va: Dict[str, np.ndarray] = {}
    scores_te: Dict[str, np.ndarray] = {}

    for name, mtype, X_tr_v, X_va_v, X_te_v in variants:
        if mtype == "threshold":
            scores_va[name] = X_va_v[:, 0].astype(np.float64)
            scores_te[name] = X_te_v[:, 0].astype(np.float64)
            print(f"[Stage3D.2] {name}: score passthrough  shape={X_va_v.shape}")
        elif _SK:
            train_fn = _train_lr if mtype == "LR" else _train_mlp
            print(f"[Stage3D.2] Training {name} ({mtype})  shape={X_tr_v.shape}...")
            m, sc = train_fn(X_tr_v, y_tr)
            n_it = getattr(m, "n_iter_", "?")
            if mtype == "MLP":
                print(f"[Stage3D.2]   converged in {n_it} iter")
            scores_va[name] = _proba(m, sc, X_va_v)
            scores_te[name] = _proba(m, sc, X_te_v)
        else:
            print(f"[Stage3D.2] SKIP {name}: sklearn not available")

    # ── Operating point evaluation ─────────────────────────────────────────────
    print(f"\n[Stage3D.2] Operating point calibration...")

    # B0 reference thresholds and test FP_supp (pre-compute for gain calculation)
    b0_thresh: Dict[float, Optional[float]] = {}
    b0_fp_supp_te: Dict[float, Optional[float]] = {}
    for tgt in _TP_TARGETS:
        th = _find_op_thresh(y_va, scores_va["B0"], tgt)
        b0_thresh[tgt] = th
        if th is not None:
            m = _at_thresh(y_te, scores_te["B0"], un_te, th)
            b0_fp_supp_te[tgt] = m["fp_supp"]
        else:
            b0_fp_supp_te[tgt] = None

    abl_rows: List[dict] = []

    for name, mtype, _, _, _ in variants:
        if name not in scores_va:
            continue
        sc_va = scores_va[name]
        sc_te = scores_te[name]

        for tgt in _TP_TARGETS:
            th = _find_op_thresh(y_va, sc_va, tgt)
            if th is None:
                abl_rows.append(
                    {
                        "variant": name,
                        "tp_ret_target": tgt,
                        "model_type": mtype,
                        **{
                            k: None
                            for k in _ABL_FIELDS
                            if k not in ("variant", "tp_ret_target", "model_type")
                        },
                        "achieves_target_valid": False,
                        "achieves_target_test": None,
                    }
                )
                print(f"[Stage3D.2]   {name:<22} TP>={tgt:.2f}: cannot reach target")
                continue

            m_va = _at_thresh(y_va, sc_va, un_va, th)
            m_te = _at_thresh(y_te, sc_te, un_te, th)
            b0_ref = b0_fp_supp_te.get(tgt)
            gain = (
                round((m_te["fp_supp"] - b0_ref), 4)
                if (b0_ref is not None and name != "B0")
                else None
            )
            flag = "✓" if (m_te["tp_ret"] or 0) >= tgt else "✗"
            gstr = f"{gain:+.4f}" if gain is not None else "(B0)"
            print(
                f"[Stage3D.2]   {name:<22} TP>={tgt:.2f}: "
                f"thresh={th:.4f}  "
                f"FP_supp(V={m_va['fp_supp']:.3f} T={m_te['fp_supp']:.3f})  "
                f"gain={gstr}"
            )
            abl_rows.append(
                {
                    "variant": name,
                    "tp_ret_target": tgt,
                    "model_type": mtype,
                    "thresh_valid": th,
                    "tp_ret_valid": m_va["tp_ret"],
                    "fp_supp_valid": m_va["fp_supp"],
                    "precision_valid": m_va["precision"],
                    "auprc_valid": m_va["auprc"],
                    "tp_ret_test": m_te["tp_ret"],
                    "fp_supp_test": m_te["fp_supp"],
                    "precision_test": m_te["precision"],
                    "auprc_test": m_te["auprc"],
                    "fp_supp_gain_over_b0": gain,
                    "achieves_target_valid": (m_va["tp_ret"] or 0) >= tgt,
                    "achieves_target_test": (m_te["tp_ret"] or 0) >= tgt,
                }
            )

    # ── Bootstrap: FP_supp_gain CI ─────────────────────────────────────────────
    print(f"\n[Stage3D.2] Bootstrap FP_supp_gain CI ({args.n_bootstrap} resamples)...")
    gain_ci_rows: List[dict] = []

    for name, mtype, _, _, _ in variants:
        if name == "B0" or name not in scores_te:
            continue
        for tgt in _TP_TARGETS:
            th_model = next(
                (
                    r["thresh_valid"]
                    for r in abl_rows
                    if r["variant"] == name
                    and abs(r.get("tp_ret_target", -1) - tgt) < 1e-6
                ),
                None,
            )
            th_b0 = b0_thresh.get(tgt)
            if th_model is None or th_b0 is None:
                continue
            boot = _bootstrap_fp_supp_gain(
                img_te,
                y_te,
                scores_te[name],
                scores_te["B0"],
                th_model,
                th_b0,
                n_bootstrap=args.n_bootstrap,
                seed=_BOOTSTRAP_SEED,
            )
            sig = "YES" if (boot["ci_lo"] is not None and boot["ci_lo"] > 0) else "no"
            print(
                f"[Stage3D.2]   {name:<22} TP>={tgt:.2f}: "
                f"gain={_f(boot['mean_gain'], '+.4f') if boot['mean_gain'] is not None else 'N/A'}  "
                f"CI=[{_f(boot['ci_lo'])},{_f(boot['ci_hi'])}]  "
                f"P(>0)={_f(boot['frac_positive'])}  sig={sig}"
            )
            gain_ci_rows.append(
                {
                    "variant": name,
                    "tp_ret_target": tgt,
                    "model_type": mtype,
                    **boot,
                    "significant": sig,
                }
            )

    # ── Bootstrap: AUPRC CI ────────────────────────────────────────────────────
    print(f"\n[Stage3D.2] Bootstrap AUPRC CI...")
    auprc_ci_te = _bootstrap_auprc(
        img_te, y_te, scores_te, n_bootstrap=args.n_bootstrap, seed=_BOOTSTRAP_SEED
    )

    # ── Write outputs ──────────────────────────────────────────────────────────
    abl_path = out_dir / "c2_ablation_results.csv"
    _write_csv(abl_rows, _ABL_FIELDS, abl_path)
    print(f"\n[Stage3D.2] {abl_path}")

    gain_ci_path = out_dir / "bootstrap_fp_supp_gain_ci.csv"
    _write_csv(gain_ci_rows, _BOOT_GAIN_FIELDS, gain_ci_path)
    print(f"[Stage3D.2] {gain_ci_path}")

    md_path = out_dir / "c2_ablation_decision.md"
    _write_decision_md(abl_rows, gain_ci_rows, auprc_ci_te, variant_order, md_path)
    print(f"[Stage3D.2] {md_path}")

    print(f"\n[Stage3D.2] Done. Read c2_ablation_decision.md for C2 verdict.")
    print(
        "[Stage3D.2] Key question: Is B2_clinical consistently better than ALL controls?\n"
        "  → Yes: KG-derived SDCA contributes. C2 claim is supported.\n"
        "  → No:  Gain is score/class artifact. Reframe C2 as auditability contribution."
    )


if __name__ == "__main__":
    main()
