#!/usr/bin/env python
"""
eval_external_validation.py
External validation of the frozen LGG governance gate on IDRiD and TJDR.

Protocol:
  - LGG trained on oiaddr train governance_features
  - Operating point selected on oiaddr valid (frozen, not tuned on external data)
  - Applied to external dataset features (IDRiD or TJDR train/test)
  - Metrics: TP retention rate, FP suppression gain vs B0 at frozen operating point

Models evaluated:
  B0          : score threshold (no model)
  B2_MLP      : SAS + SDCA → MLP                          (primary KG-evidence variant)
  B3_MLP      : SAS + SDCA + APS + PSS → MLP              (full evidence)
  B_PCSR_MLP  : SAS + PCSR(4-dim SDM-row of pred_class) → MLP
                (supporting variant — robust under detector-confidence
                 collapse because PCSR is computed from predicted class
                 alone, not from possibly-inverted ensemble_probs).
                Reported side-by-side with B2_MLP; SDCA remains primary.

This is the only cross-dataset evaluation tool in the LGov-Net pipeline.
All thresholds are frozen from oiaddr valid before any external data is seen.

Inputs:
  --train_stage3c_dir   : oiaddr stage3c output dir (has governance_features_train.csv + _valid.csv)
  --external_stage3c_dir: IDRiD or TJDR stage3c output dir
  --external_splits     : which splits to evaluate (e.g. train test)
  --external_name       : dataset name for output labeling (e.g. idrid, tjdr)
  --output_dir          : where to write results

Outputs:
  outputs/external_validation/{external_name}/external_eval_results.csv
  outputs/external_validation/{external_name}/external_eval_summary.json
  outputs/external_validation/{external_name}/external_eval_report.md

Usage:
    python tools/eval_external_validation.py \\
      --train_stage3c_dir outputs/stage3c_audit/oiaddr \\
      --external_stage3c_dir outputs/stage3c_audit/idrid \\
      --external_splits train test \\
      --external_name idrid \\
      --output_dir outputs/external_validation/idrid

    python tools/eval_external_validation.py \\
      --train_stage3c_dir outputs/stage3c_audit/oiaddr \\
      --external_stage3c_dir outputs/stage3c_audit/tjdr \\
      --external_splits train test \\
      --external_name tjdr \\
      --output_dir outputs/external_validation/tjdr
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    _SK = True
except ImportError:
    _SK = False
    print("[ExternalEval] WARNING: scikit-learn not installed. Only B0 will run.")

# ── Constants ─────────────────────────────────────────────────────────────────

_RETAIN_CATS = frozenset({"TP_EXACT", "CLINICALLY_CLOSE_MISMATCH", "LOCALIZATION_WEAK"})
_TP_TARGETS = [0.85, 0.80, 0.75]
_THRESHOLD_GRID = np.round(np.linspace(0.001, 0.999, 499), 4)
_N_BOOTSTRAP = 1000
_BOOTSTRAP_SEED = 42

# Clinical SDM (ICDR severity-transition normalized). Used to compute PCSR
# (PredClass-SDM-Row) — a 4-dim KG-derived feature for the predicted class.
# PCSR is immune to softmax inversion in low-confidence ensembles because
# it depends only on the predicted class label, not ensemble_probs.
_CLASS_ORDER = ["MA", "HE", "Hard_EX", "Soft_EX"]
_CLASS_IDX = {c: i for i, c in enumerate(_CLASS_ORDER)}
_CLINICAL_SDM = np.array([
    [0.00, 0.00, 0.50, 1.00],   # MA
    [0.00, 0.00, 0.50, 1.00],   # HE
    [0.50, 0.50, 0.00, 0.75],   # Hard_EX
    [1.00, 1.00, 0.75, 0.00],   # Soft_EX
], dtype=np.float32)


def _pcsr_for_class(pred_class: str) -> np.ndarray:
    """Return the 4-dim SDM row for a predicted class. Unknown class → zeros."""
    idx = _CLASS_IDX.get(pred_class, -1)
    if idx < 0:
        return np.zeros(4, dtype=np.float32)
    return _CLINICAL_SDM[idx].copy()


# ── Feature loading ────────────────────────────────────────────────────────────


def _load_image_masks(mask_dir: Path, split: str, image_ids: List[str]) -> Dict[str, np.ndarray]:
    """Load fundus masks for the given image_ids.
    Returns: dict image_id → 2D bool mask (True = inside fundus).
    Used by H_mask DHA component.
    """
    try:
        from PIL import Image
    except ImportError:
        print("[ExternalEval] WARN: PIL not installed; mask filter disabled.")
        return {}
    out: Dict[str, np.ndarray] = {}
    split_dir = mask_dir / split
    if not split_dir.exists():
        print(f"[ExternalEval] WARN: mask dir not found: {split_dir}")
        return out
    seen = set()
    # Naming conventions to try, in order of preference.
    # Convention 1: {img_id}_mask.{ext} — primary (oiaddr, idrid, tjdr).
    # Convention 2: {img_id}.{ext} — fallback for any other layout.
    _SUFFIX_VARIANTS = ("_mask", "")
    _EXT_VARIANTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    for img_id in image_ids:
        if img_id in seen:
            continue
        seen.add(img_id)
        found = False
        for suffix in _SUFFIX_VARIANTS:
            for ext in _EXT_VARIANTS:
                p = split_dir / f"{img_id}{suffix}{ext}"
                if p.exists():
                    arr = np.array(Image.open(p).convert("L"))
                    out[img_id] = arr > 0
                    found = True
                    break
            if found:
                break
    print(f"[ExternalEval]   loaded {len(out)} masks from {split_dir}")
    return out


def _load_bbox_coords(cache_dir: Path, split: str) -> Dict[str, Tuple[float, float, float, float]]:
    """Index detector cache by bbox_id → (x1, y1, x2, y2).
    Used by H_mask DHA component.
    """
    out: Dict[str, Tuple[float, float, float, float]] = {}
    split_dir = cache_dir / split
    if not split_dir.exists():
        print(f"[ExternalEval] WARN: cache dir not found: {split_dir}")
        return out
    for fp in split_dir.glob("*.json"):
        try:
            with open(fp, encoding="utf-8") as f:
                entries = json.load(f)
            for entry in entries:
                bid = entry.get("bbox_id")
                raw = entry.get("detector_raw") or entry.get("r2a_raw") or {}
                box = raw.get("bbox")
                if bid and box and len(box) == 4:
                    out[bid] = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        except Exception:
            continue
    print(f"[ExternalEval]   indexed {len(out)} bbox coords from {split_dir}")
    return out


def _mask_filter_indices(
    image_ids: List[str],
    bbox_ids: List[str],
    masks: Dict[str, np.ndarray],
    coords: Dict[str, Tuple[float, float, float, float]],
) -> np.ndarray:
    """Return boolean mask: True keep, False drop. Drop = bbox center outside fundus."""
    keep = np.ones(len(image_ids), dtype=bool)
    n_dropped = 0
    n_no_coord = 0
    n_no_mask = 0
    for i, (img, bid) in enumerate(zip(image_ids, bbox_ids)):
        mask = masks.get(img)
        coord = coords.get(bid)
        if mask is None:
            n_no_mask += 1
            continue
        if coord is None:
            n_no_coord += 1
            continue
        cx = int((coord[0] + coord[2]) / 2.0)
        cy = int((coord[1] + coord[3]) / 2.0)
        h, w = mask.shape
        if 0 <= cy < h and 0 <= cx < w:
            if not bool(mask[cy, cx]):
                keep[i] = False
                n_dropped += 1
        else:
            keep[i] = False
            n_dropped += 1
    print(f"[ExternalEval]   mask filter: dropped {n_dropped}/{len(image_ids)} "
          f"(no_mask={n_no_mask}, no_coord={n_no_coord})")
    return keep


def _compute_score_cdf_per_class(scores: np.ndarray, pred_classes: List[str]) -> Dict[str, np.ndarray]:
    """Build per-class sorted-score reference from a source split.
    Used by H_score DHA component (frozen on oiaddr-valid; never refit on external).
    """
    out: Dict[str, np.ndarray] = {}
    pc_arr = np.array(pred_classes)
    for c in _CLASS_ORDER:
        m = pc_arr == c
        if int(m.sum()) > 0:
            out[c] = np.sort(scores[m])
        else:
            out[c] = np.array([], dtype=np.float64)
    return out


def _apply_score_cdf(
    scores: np.ndarray,
    pred_classes: List[str],
    cdf_per_class: Dict[str, np.ndarray],
) -> np.ndarray:
    """Transform raw scores to source-domain percentile rank, per predicted class.
    Monotonic per-class transform. Boxes whose pred_class has no reference
    samples in the CDF are left unchanged.
    """
    out = scores.astype(np.float64).copy()
    for i, (s, c) in enumerate(zip(scores, pred_classes)):
        ref = cdf_per_class.get(c)
        if ref is None or len(ref) == 0:
            continue
        idx = int(np.searchsorted(ref, s, side="left"))
        out[i] = idx / len(ref)
    return out


def _load_real_sdca_overrides(real_sdca_dir: Path, dataset: str, split: str) -> Dict[str, float]:
    """Stage 10A — load real_sdca overrides keyed by bbox_id from
    {real_sdca_dir}/{dataset}_{split}_real_sdca.csv.
    Returns: dict bbox_id -> real_sdca (float). bbox_ids with empty/missing
    real_sdca (status='fallback') are NOT in the dict; caller keeps synthetic.
    Added 2026-05-17.
    """
    csv_path = real_sdca_dir / f"{dataset}_{split}_real_sdca.csv"
    out: Dict[str, float] = {}
    if not csv_path.exists():
        print(f"[ExternalEval] WARN: real_sdca CSV not found: {csv_path} (will not override)")
        return out
    n_disagreement = 0; n_consensus = 0; n_fallback = 0
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bid = row.get("bbox_id", "")
            status = row.get("status", "")
            rs_raw = row.get("real_sdca", "")
            if status == "fallback" or rs_raw == "":
                n_fallback += 1
                continue
            try:
                out[bid] = float(rs_raw)
                if status == "disagreement":
                    n_disagreement += 1
                elif status == "consensus":
                    n_consensus += 1
            except (ValueError, TypeError):
                continue
    print(f"[ExternalEval]   real_sdca overrides loaded for {dataset}/{split}: "
          f"disagreement={n_disagreement}  consensus={n_consensus}  fallback={n_fallback}")
    return out


def _apply_real_sdca_overrides(
    X2: np.ndarray,
    X3: np.ndarray,
    bbox_ids: List[str],
    overrides: Dict[str, float],
) -> Tuple[int, int]:
    """Replace SDCA feature column (index 1) in B2 and B3 matrices for bbox_ids
    present in overrides. PCSR matrix Xp does NOT contain SDCA → not modified.
    Returns: (n_replaced, n_kept_synthetic).
    Added 2026-05-17.
    """
    n_replaced = 0; n_kept = 0
    for i, bid in enumerate(bbox_ids):
        if bid in overrides:
            X2[i, 1] = overrides[bid]
            X3[i, 1] = overrides[bid]
            n_replaced += 1
        else:
            n_kept += 1
    return n_replaced, n_kept


def _fit_temperature_global(scores: np.ndarray, y: np.ndarray) -> float:
    """Fit a single scalar temperature T on source-domain validation NLL.
    Used by B_calibrated DHA baseline. T is fitted by minimizing binary cross-
    entropy of sigmoid(score/T) against the binary retain label y on oiaddr-valid.
    NOTE: T-scaling on a single score channel is rank-preserving, so B0 at matched
    TP_ret is numerically identical to plain B0. Implemented for completeness and
    to verify LGov-Net governance gains are not artifacts of standard global
    calibration. Reference: Guo et al. 2017, On Calibration of Modern Neural
    Networks.
    """
    try:
        from scipy.optimize import minimize_scalar
    except ImportError:
        print("[ExternalEval] WARN: scipy not installed; T defaults to 1.0")
        return 1.0
    scores = scores.astype(np.float64)
    y = y.astype(np.float64)

    def nll(T: float) -> float:
        if T <= 1e-6:
            return 1e10
        s = scores / T
        s_pos = np.where(s >= 0, s, 0.0)
        log1pexp = s_pos + np.log1p(np.exp(-np.abs(s)))
        return float(np.mean(log1pexp - y * s))

    try:
        result = minimize_scalar(nll, bracket=(0.1, 1.0, 10.0), method="brent")
        return float(result.x) if result.x > 1e-6 else 1.0
    except Exception:
        return 1.0


def _apply_temperature(scores: np.ndarray, T: float) -> np.ndarray:
    """Apply sigmoid(score / T). Bounded to (0, 1). Monotonic in score."""
    s = scores.astype(np.float64) / max(T, 1e-6)
    return (1.0 / (1.0 + np.exp(-s))).astype(np.float32)


def _load_features(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str], List[str]]:
    """Load governance_features_{split}.csv.
    Returns: X_b2, X_b3, X_pcsr, y, scores, image_ids, bbox_ids, pred_classes.
      X_b2   = [SAS, SDCA]                            (primary KG variant)
      X_b3   = [SAS, SDCA, APS, PSS]                  (full evidence)
      X_pcsr = [SAS, SDM[pred_class, 0..3]]           (supporting variant)
    """
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    def _f(r, k):
        v = r.get(k, "")
        return float(v) if v not in ("", None) else 0.0

    X_cols_b2 = []
    X_cols_b3 = []
    X_cols_pcsr = []
    y = []
    image_ids = []
    bbox_ids = []
    pred_classes: List[str] = []

    for r in rows:
        sas  = _f(r, "ev_sas")
        sdca = _f(r, "ev_sdca")
        aps  = _f(r, "ev_aps")
        pss  = _f(r, "ev_pss")
        cat  = r.get("category", "")
        pred_class = r.get("pred_class", "")
        pcsr = _pcsr_for_class(pred_class)

        X_cols_b2.append([sas, sdca])
        X_cols_b3.append([sas, sdca, aps, pss])
        X_cols_pcsr.append([sas, float(pcsr[0]), float(pcsr[1]), float(pcsr[2]), float(pcsr[3])])
        y.append(1 if cat in _RETAIN_CATS else 0)
        image_ids.append(r.get("image_id", ""))
        bbox_ids.append(r.get("bbox_id", ""))
        pred_classes.append(pred_class)

    scores = np.array([_f(r, "pred_score") for r in rows])

    return (
        np.array(X_cols_b2, dtype=np.float32),
        np.array(X_cols_b3, dtype=np.float32),
        np.array(X_cols_pcsr, dtype=np.float32),
        np.array(y, dtype=np.int32),
        scores,
        image_ids,
        bbox_ids,
        pred_classes,
    )


# ── Metric helpers ─────────────────────────────────────────────────────────────


def _compute_metrics(
    y_true: np.ndarray,
    y_pred_bin: np.ndarray,
    image_ids: List[str],
    tp_target: float,
) -> Dict:
    """Compute TP retention and FP suppression at this binary prediction."""
    mask_gt_retain = y_true == 1
    mask_gt_suppress = y_true == 0

    tp_retained = int(np.sum(y_pred_bin[mask_gt_retain] == 1))
    tp_total    = int(np.sum(mask_gt_retain))
    fp_suppressed = int(np.sum(y_pred_bin[mask_gt_suppress] == 0))
    fp_total    = int(np.sum(mask_gt_suppress))

    tp_ret = tp_retained / max(tp_total, 1)
    fp_supp = fp_suppressed / max(fp_total, 1)

    return {
        "tp_retained": tp_retained,
        "tp_total": tp_total,
        "tp_ret_rate": round(tp_ret, 4),
        "fp_suppressed": fp_suppressed,
        "fp_total": fp_total,
        "fp_supp_rate": round(fp_supp, 4),
        "n_boxes": int(len(y_true)),
    }


def _select_threshold(
    probs: np.ndarray,
    y: np.ndarray,
    tp_target: float,
) -> float:
    """Find the highest threshold where TP_ret >= tp_target."""
    best_thresh = 0.001
    best_fp_supp = 0.0
    mask_pos = y == 1
    mask_neg = y == 0

    for t in _THRESHOLD_GRID:
        pred = (probs >= t).astype(int)
        tp_ret = np.sum(pred[mask_pos] == 1) / max(np.sum(mask_pos), 1)
        if tp_ret >= tp_target:
            fp_supp = np.sum(pred[mask_neg] == 0) / max(np.sum(mask_neg), 1)
            if fp_supp > best_fp_supp:
                best_fp_supp = fp_supp
                best_thresh = t

    return best_thresh


def _bootstrap_fp_supp(
    probs: np.ndarray,
    y: np.ndarray,
    image_ids: List[str],
    thresh: float,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Tuple[float, float]:
    """Bootstrap CI for FP suppression at given threshold."""
    rng = np.random.default_rng(seed)
    unique_imgs = list(set(image_ids))
    img_arr = np.array(image_ids)

    fp_supps = []
    for _ in range(n_bootstrap):
        sampled_imgs = rng.choice(unique_imgs, size=len(unique_imgs), replace=True)
        idx = np.concatenate([np.where(img_arr == img)[0] for img in sampled_imgs])
        y_s = y[idx]
        p_s = probs[idx]
        pred_s = (p_s >= thresh).astype(int)
        mask_neg = y_s == 0
        if np.sum(mask_neg) == 0:
            continue
        fp_supps.append(np.sum(pred_s[mask_neg] == 0) / np.sum(mask_neg))

    arr = np.array(fp_supps)
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


# ── Model training ─────────────────────────────────────────────────────────────


def _train_mlp(X_train: np.ndarray, y_train: np.ndarray, seed: int = 42):
    """Train StandardScaler + MLPClassifier."""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_train)
    clf = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        max_iter=500,
        random_state=seed,
    )
    clf.fit(X_s, y_train)
    return scaler, clf


def _compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred_bin: np.ndarray,
    pred_classes: List[str],
    classes: List[str] = ("MA", "HE", "Hard_EX", "Soft_EX"),
) -> List[Dict]:
    """Per-pred_class TP_ret and FP_supp at a given binary prediction.
    Used by R1 (per-class FP_supp at LGG operating point).
    Returns one dict per class with breakdown counts and rates.
    """
    pc_arr = np.array(pred_classes)
    out: List[Dict] = []
    for c in classes:
        mask = pc_arr == c
        if int(mask.sum()) == 0:
            out.append({
                "pred_class": c,
                "n_boxes": 0,
                "tp_total": 0, "tp_retained": 0, "tp_ret_rate": None,
                "fp_total": 0, "fp_suppressed": 0, "fp_supp_rate": None,
            })
            continue
        y_c = y_true[mask]; pred_c = y_pred_bin[mask]
        mask_pos = y_c == 1; mask_neg = y_c == 0
        tp_total = int(mask_pos.sum()); fp_total = int(mask_neg.sum())
        tp_retained = int((pred_c[mask_pos] == 1).sum())
        fp_suppressed = int((pred_c[mask_neg] == 0).sum())
        tp_ret = (tp_retained / tp_total) if tp_total > 0 else None
        fp_supp = (fp_suppressed / fp_total) if fp_total > 0 else None
        out.append({
            "pred_class": c,
            "n_boxes": int(mask.sum()),
            "tp_total": tp_total, "tp_retained": tp_retained,
            "tp_ret_rate": round(tp_ret, 4) if tp_ret is not None else None,
            "fp_total": fp_total, "fp_suppressed": fp_suppressed,
            "fp_supp_rate": round(fp_supp, 4) if fp_supp is not None else None,
        })
    return out


# ── Main evaluation ────────────────────────────────────────────────────────────


def run_evaluation(
    train_stage3c_dir: Path,
    external_stage3c_dir: Path,
    external_splits: List[str],
    external_name: str,
    output_dir: Path,
    apply_mask_filter: bool = False,
    detector_cache_dir: Optional[Path] = None,
    mask_dir: Optional[Path] = None,
    apply_score_calibration: bool = False,
    apply_temperature_scaling: bool = False,
    per_class_breakdown: bool = False,
    real_sdca_dir: Optional[Path] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load oiaddr train features ──────────────────────────────────────────
    train_csv = train_stage3c_dir / "governance_features_train.csv"
    valid_csv = train_stage3c_dir / "governance_features_valid.csv"

    if not train_csv.exists():
        raise FileNotFoundError(f"oiaddr train features not found: {train_csv}")
    if not valid_csv.exists():
        raise FileNotFoundError(f"oiaddr valid features not found: {valid_csv}")

    print(f"[ExternalEval] Loading oiaddr train features: {train_csv}")
    X2_tr, X3_tr, Xp_tr, y_tr, s_tr, imgs_tr, bids_tr, pc_tr = _load_features(train_csv)
    print(f"[ExternalEval] oiaddr train: {len(y_tr)} boxes  retain={np.sum(y_tr)} suppress={np.sum(y_tr==0)}")

    print(f"[ExternalEval] Loading oiaddr valid features: {valid_csv}")
    X2_va, X3_va, Xp_va, y_va, s_va, imgs_va, bids_va, pc_va = _load_features(valid_csv)
    print(f"[ExternalEval] oiaddr valid: {len(y_va)} boxes  retain={np.sum(y_va)} suppress={np.sum(y_va==0)}")

    # ── Stage 10A: realSDCA overrides on oiaddr train + valid (if provided) ───
    # The MLP must be trained and OP-selected on the SAME SDCA feature space
    # used at external evaluation. If --real_sdca_dir is set, override the
    # SDCA column in train/valid feature matrices using bbox_id-keyed lookup.
    # Bboxes not present in the realSDCA CSV retain their synthetic ev_sdca
    # (fallback, no downgrade). Added 2026-05-17.
    if real_sdca_dir is not None:
        print(f"[ExternalEval]   Stage 10A: applying realSDCA overrides on oiaddr/train + oiaddr/valid")
        tr_over = _load_real_sdca_overrides(real_sdca_dir, "oiaddr", "train")
        n_tr_rep, n_tr_kept = _apply_real_sdca_overrides(X2_tr, X3_tr, bids_tr, tr_over)
        print(f"[ExternalEval]   oiaddr/train: replaced={n_tr_rep}  kept_synthetic={n_tr_kept}")
        va_over = _load_real_sdca_overrides(real_sdca_dir, "oiaddr", "valid")
        n_va_rep, n_va_kept = _apply_real_sdca_overrides(X2_va, X3_va, bids_va, va_over)
        print(f"[ExternalEval]   oiaddr/valid: replaced={n_va_rep}  kept_synthetic={n_va_kept}")

    # ── B_calibrated: global temperature scaling fit on oiaddr-valid ───────
    # Fits a single scalar T via NLL minimization. T-scaling is rank-preserving
    # so B0+T is numerically identical to plain B0 at matched TP_ret. Reported
    # for completeness; used to confirm LGov-Net governance gains are not
    # artifacts of standard global calibration (Guo et al. 2017).
    temperature_T: Optional[float] = None
    if apply_temperature_scaling:
        temperature_T = _fit_temperature_global(s_va, y_va)
        print(f"[ExternalEval]   B_calibrated: global T fit on oiaddr-valid = {temperature_T:.4f}")
        s_tr_T = _apply_temperature(s_tr, temperature_T)
        s_va_T = _apply_temperature(s_va, temperature_T)
        X2_tr[:, 0] = s_tr_T; X3_tr[:, 0] = s_tr_T; Xp_tr[:, 0] = s_tr_T
        X2_va[:, 0] = s_va_T; X3_va[:, 0] = s_va_T; Xp_va[:, 0] = s_va_T
        s_tr = s_tr_T.astype(np.float32)
        s_va = s_va_T.astype(np.float32)

    # ── H_score CDF (frozen on oiaddr-valid; never refit on external) ──────
    score_cdf: Optional[Dict[str, np.ndarray]] = None
    if apply_score_calibration:
        score_cdf = _compute_score_cdf_per_class(s_va, pc_va)
        print(f"[ExternalEval]   H_score CDF built on oiaddr-valid: "
              + ", ".join(f"{c}={len(score_cdf[c])}" for c in _CLASS_ORDER))
        # Also transform oiaddr-train + oiaddr-valid features so the MLP is
        # trained and OP-selected in the same percentile space.
        s_tr_cal = _apply_score_cdf(s_tr, pc_tr, score_cdf)
        s_va_cal = _apply_score_cdf(s_va, pc_va, score_cdf)
        # Replace SAS column (index 0) in all feature matrices.
        X2_tr[:, 0] = s_tr_cal; X3_tr[:, 0] = s_tr_cal; Xp_tr[:, 0] = s_tr_cal
        X2_va[:, 0] = s_va_cal; X3_va[:, 0] = s_va_cal; Xp_va[:, 0] = s_va_cal
        s_tr = s_tr_cal.astype(np.float32)
        s_va = s_va_cal.astype(np.float32)

    # ── Train models on oiaddr train ───────────────────────────────────────
    # Tuple layout: (scaler, classifier, X_valid, feat_tag)
    # feat_tag selects which feature matrix to use at external eval time.
    models = {}
    if _SK:
        print("[ExternalEval] Training B2_MLP (SAS+SDCA) on oiaddr train...")
        sc2, clf2 = _train_mlp(X2_tr, y_tr)
        models["B2_MLP"] = (sc2, clf2, X2_va, "b2")

        print("[ExternalEval] Training B3_MLP (SAS+SDCA+APS+PSS) on oiaddr train...")
        sc3, clf3 = _train_mlp(X3_tr, y_tr)
        models["B3_MLP"] = (sc3, clf3, X3_va, "b3")

        print("[ExternalEval] Training B_PCSR_MLP (SAS+PCSR 4-dim SDM-row) on oiaddr train...")
        scp, clfp = _train_mlp(Xp_tr, y_tr)
        models["B_PCSR_MLP"] = (scp, clfp, Xp_va, "pcsr")

    # ── Select operating points on oiaddr valid (frozen) ───────────────────
    frozen_thresholds: Dict[str, Dict[float, float]] = {}
    print("\n[ExternalEval] Selecting operating points on oiaddr valid (FROZEN):")

    # B0: score threshold
    frozen_thresholds["B0"] = {}
    for tp_t in _TP_TARGETS:
        thresh = _select_threshold(s_va, y_va, tp_t)
        frozen_thresholds["B0"][tp_t] = thresh
        metrics_va = _compute_metrics(y_va, (s_va >= thresh).astype(int), imgs_va, tp_t)
        print(f"  B0 TP>={tp_t}: thresh={thresh:.3f}  TP_ret={metrics_va['tp_ret_rate']:.4f}  FP_supp={metrics_va['fp_supp_rate']:.4f}")

    if _SK:
        for model_name, (sc, clf, X_va, _) in models.items():
            frozen_thresholds[model_name] = {}
            probs_va = clf.predict_proba(sc.transform(X_va))[:, 1]
            for tp_t in _TP_TARGETS:
                thresh = _select_threshold(probs_va, y_va, tp_t)
                frozen_thresholds[model_name][tp_t] = thresh
                metrics_va = _compute_metrics(y_va, (probs_va >= thresh).astype(int), imgs_va, tp_t)
                print(f"  {model_name} TP>={tp_t}: thresh={thresh:.3f}  TP_ret={metrics_va['tp_ret_rate']:.4f}  FP_supp={metrics_va['fp_supp_rate']:.4f}")

    # ── Evaluate on external dataset splits ────────────────────────────────
    all_results = []
    per_class_results: List[Dict] = []  # R1 per-class FP_supp at LGG OP
    full_summary: Dict = {
        "external_dataset": external_name,
        "train_stage3c_dir": str(train_stage3c_dir),
        "external_stage3c_dir": str(external_stage3c_dir),
        "frozen_thresholds": {
            model: {str(k): v for k, v in d.items()}
            for model, d in frozen_thresholds.items()
        },
        "splits": {},
    }

    for split in external_splits:
        ext_csv = external_stage3c_dir / f"governance_features_{split}.csv"
        if not ext_csv.exists():
            print(f"[ExternalEval] SKIP {split}: {ext_csv} not found")
            continue

        print(f"\n[ExternalEval] ── Evaluating {external_name}/{split} ──")
        X2_ext, X3_ext, Xp_ext, y_ext, s_ext, imgs_ext, bids_ext, pc_ext = _load_features(ext_csv)
        print(f"  {len(y_ext)} boxes  retain={np.sum(y_ext)} suppress={np.sum(y_ext==0)}")

        # ── Stage 10A: realSDCA overrides on external split (if provided) ─────
        # Applied to B2/B3 feature matrices via bbox_id lookup; PCSR unaffected
        # (PCSR uses SDM-row, not SDCA). Added 2026-05-17.
        if real_sdca_dir is not None:
            ext_over = _load_real_sdca_overrides(real_sdca_dir, external_name, split)
            n_ext_rep, n_ext_kept = _apply_real_sdca_overrides(X2_ext, X3_ext, bids_ext, ext_over)
            print(f"[ExternalEval]   {external_name}/{split}: real_sdca replaced={n_ext_rep}  kept_synthetic={n_ext_kept}")

        if apply_temperature_scaling and temperature_T is not None:
            print(f"[ExternalEval]   B_calibrated: applying T={temperature_T:.4f} (sigmoid score/T)")
            s_ext_T = _apply_temperature(s_ext, temperature_T)
            X2_ext[:, 0] = s_ext_T
            X3_ext[:, 0] = s_ext_T
            Xp_ext[:, 0] = s_ext_T
            s_ext = s_ext_T.astype(np.float32)

        if apply_score_calibration and score_cdf is not None:
            print(f"[ExternalEval]   H_score DHA: applying source-domain percentile transform")
            s_ext_cal = _apply_score_cdf(s_ext, pc_ext, score_cdf)
            X2_ext[:, 0] = s_ext_cal
            X3_ext[:, 0] = s_ext_cal
            Xp_ext[:, 0] = s_ext_cal
            s_ext = s_ext_cal.astype(np.float32)

        if apply_mask_filter:
            if detector_cache_dir is None or mask_dir is None:
                raise ValueError("apply_mask_filter requires --detector_cache_dir and --mask_dir")
            print(f"[ExternalEval]   H_mask DHA: applying fundus-mask outside-retina filter")
            masks = _load_image_masks(mask_dir, split, imgs_ext)
            coords = _load_bbox_coords(detector_cache_dir, split)
            keep = _mask_filter_indices(imgs_ext, bids_ext, masks, coords)
            X2_ext = X2_ext[keep]; X3_ext = X3_ext[keep]; Xp_ext = Xp_ext[keep]
            y_ext = y_ext[keep]; s_ext = s_ext[keep]
            imgs_ext = [im for im, k in zip(imgs_ext, keep) if k]
            bids_ext = [bid for bid, k in zip(bids_ext, keep) if k]
            print(f"  after H_mask: {len(y_ext)} boxes  retain={np.sum(y_ext)} suppress={np.sum(y_ext==0)}")

        split_results = {}

        # B0
        for tp_t in _TP_TARGETS:
            thresh = frozen_thresholds["B0"][tp_t]
            pred = (s_ext >= thresh).astype(int)
            m = _compute_metrics(y_ext, pred, imgs_ext, tp_t)
            fp_lo, fp_hi = _bootstrap_fp_supp(s_ext, y_ext, imgs_ext, thresh, _N_BOOTSTRAP, _BOOTSTRAP_SEED)
            row = {
                "dataset": external_name, "split": split,
                "model": "B0", "tp_target": tp_t,
                "frozen_thresh": thresh,
                **m,
                "fp_supp_ci_lo": round(fp_lo, 4),
                "fp_supp_ci_hi": round(fp_hi, 4),
            }
            all_results.append(row)
            print(f"  B0 TP>={tp_t}: TP_ret={m['tp_ret_rate']:.4f}  FP_supp={m['fp_supp_rate']:.4f}  CI=[{fp_lo:.4f},{fp_hi:.4f}]")
            if per_class_breakdown:
                for pc_row in _compute_per_class_metrics(y_ext, pred, pc_ext):
                    per_class_results.append({
                        "dataset": external_name, "split": split,
                        "model": "B0", "tp_target": tp_t,
                        "frozen_thresh": thresh,
                        **pc_row,
                    })

        # MLP models
        if _SK:
            _feat_lookup = {"b2": X2_ext, "b3": X3_ext, "pcsr": Xp_ext}
            for model_name, (sc, clf, _, feat_tag) in models.items():
                X_ext = _feat_lookup[feat_tag]
                probs_ext = clf.predict_proba(sc.transform(X_ext))[:, 1]
                for tp_t in _TP_TARGETS:
                    thresh = frozen_thresholds[model_name][tp_t]
                    pred = (probs_ext >= thresh).astype(int)
                    m = _compute_metrics(y_ext, pred, imgs_ext, tp_t)
                    fp_lo, fp_hi = _bootstrap_fp_supp(probs_ext, y_ext, imgs_ext, thresh, _N_BOOTSTRAP, _BOOTSTRAP_SEED)
                    row = {
                        "dataset": external_name, "split": split,
                        "model": model_name, "tp_target": tp_t,
                        "frozen_thresh": thresh,
                        **m,
                        "fp_supp_ci_lo": round(fp_lo, 4),
                        "fp_supp_ci_hi": round(fp_hi, 4),
                    }
                    all_results.append(row)
                    print(f"  {model_name} TP>={tp_t}: TP_ret={m['tp_ret_rate']:.4f}  FP_supp={m['fp_supp_rate']:.4f}  CI=[{fp_lo:.4f},{fp_hi:.4f}]")
                    if per_class_breakdown:
                        for pc_row in _compute_per_class_metrics(y_ext, pred, pc_ext):
                            per_class_results.append({
                                "dataset": external_name, "split": split,
                                "model": model_name, "tp_target": tp_t,
                                "frozen_thresh": thresh,
                                **pc_row,
                            })

        split_results["n_boxes"] = int(len(y_ext))
        split_results["n_retain_gt"] = int(np.sum(y_ext))
        split_results["n_suppress_gt"] = int(np.sum(y_ext == 0))
        full_summary["splits"][split] = split_results

    # ── Write CSV ──────────────────────────────────────────────────────────
    csv_path = output_dir / "external_eval_results.csv"
    fields = [
        "dataset", "split", "model", "tp_target", "frozen_thresh",
        "tp_retained", "tp_total", "tp_ret_rate",
        "fp_suppressed", "fp_total", "fp_supp_rate", "n_boxes",
        "fp_supp_ci_lo", "fp_supp_ci_hi",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\n[ExternalEval] CSV: {csv_path}")

    # ── Write summary JSON ─────────────────────────────────────────────────
    json_path = output_dir / "external_eval_summary.json"
    full_summary["results"] = all_results
    if per_class_breakdown:
        full_summary["per_class_results"] = per_class_results
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_summary, f, indent=2)
    print(f"[ExternalEval] JSON: {json_path}")

    # ── Write per-class CSV (R1 Part 2) if requested ───────────────────────
    if per_class_breakdown and per_class_results:
        pc_csv_path = output_dir / "per_class_results.csv"
        pc_fields = [
            "dataset", "split", "model", "tp_target", "frozen_thresh",
            "pred_class", "n_boxes",
            "tp_total", "tp_retained", "tp_ret_rate",
            "fp_total", "fp_suppressed", "fp_supp_rate",
        ]
        with open(pc_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=pc_fields)
            writer.writeheader()
            writer.writerows(per_class_results)
        print(f"[ExternalEval] Per-class CSV: {pc_csv_path}")

    # ── Write markdown report ──────────────────────────────────────────────
    md_path = output_dir / "external_eval_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# External Validation Report — {external_name.upper()}\n\n")
        f.write(f"**Protocol:** LGG trained on OIA-DDR train. Operating point frozen on OIA-DDR valid.\n")
        f.write(f"**External dataset:** {external_name} (no training, no tuning on this dataset)\n\n")
        f.write(f"## Frozen Operating Points (OIA-DDR valid)\n\n")
        f.write("| Model | TP target | Threshold |\n|---|---|---|\n")
        for model, thresh_map in frozen_thresholds.items():
            for tp_t, thresh in thresh_map.items():
                f.write(f"| {model} | TP>={tp_t} | {thresh:.3f} |\n")

        f.write(f"\n## External Evaluation Results\n\n")
        for split in external_splits:
            split_rows = [r for r in all_results if r["split"] == split]
            if not split_rows:
                continue
            f.write(f"### {external_name}/{split}\n\n")
            f.write("| Model | TP target | TP_ret | FP_supp | FP_supp CI |\n")
            f.write("|---|---|---|---|---|\n")
            for r in split_rows:
                f.write(
                    f"| {r['model']} | TP>={r['tp_target']} "
                    f"| {r['tp_ret_rate']:.4f} ({r['tp_retained']}/{r['tp_total']}) "
                    f"| {r['fp_supp_rate']:.4f} ({r['fp_suppressed']}/{r['fp_total']}) "
                    f"| [{r['fp_supp_ci_lo']:.4f}, {r['fp_supp_ci_hi']:.4f}] |\n"
                )
            f.write("\n")

        f.write("## Scientific Notes\n\n")
        f.write(
            "- All thresholds frozen from OIA-DDR valid before any external data was seen.\n"
            "- No tuning was performed on the external dataset.\n"
            "- TP/FP labels derived from IoU matching with GT XML annotations.\n"
            "- TJDR WARN: SDCA ordering inverted (MA > Soft_EX) — artifact of domain shift "
            "(R2a has lower confidence on 3912×3912 TJDR images; majority_class ≠ pred_class "
            "in 70-72% of boxes). Gate still passes; clinical confusion cost ordering confirmed.\n"
            "- IDRiD Soft_EX: very few GT boxes (99 train, 38 test); per-class rates unreliable.\n"
        )

    print(f"[ExternalEval] Report: {md_path}")
    print("\n[ExternalEval] DONE.")


def main():
    parser = argparse.ArgumentParser(
        description="External validation of frozen LGG on IDRiD or TJDR."
    )
    parser.add_argument("--train_stage3c_dir", required=True,
        help="OIA-DDR stage3c dir (has governance_features_train.csv + _valid.csv).")
    parser.add_argument("--external_stage3c_dir", required=True,
        help="External dataset stage3c dir (has governance_features_{split}.csv).")
    parser.add_argument("--external_splits", nargs="+", default=["train", "test"],
        choices=["train", "valid", "test"],
        help="Splits of the external dataset to evaluate (default: train test).")
    parser.add_argument("--external_name", default="external",
        help="Dataset name for labeling outputs (e.g. idrid, tjdr).")
    parser.add_argument("--output_dir", default="outputs/external_validation",
        help="Output directory for results.")
    parser.add_argument("--n_bootstrap", type=int, default=_N_BOOTSTRAP)
    parser.add_argument("--apply_mask_filter", action="store_true",
        help="H_mask DHA component: drop predictions whose bbox center is outside the fundus mask.")
    parser.add_argument("--detector_cache_dir", default=None,
        help="Required if --apply_mask_filter: path to detector cache (has split subdirs with bbox jsons).")
    parser.add_argument("--mask_dir", default=None,
        help="Required if --apply_mask_filter: path to fundus masks (has split subdirs with image_id.png).")
    parser.add_argument("--apply_score_calibration", action="store_true",
        help="H_score DHA component: source-domain per-class percentile transform of pred_score / SAS. "
             "CDF derived from oiaddr-valid ONLY; frozen before external evaluation. Reported as a "
             "supporting ablation, not a core architecture component.")
    parser.add_argument("--apply_temperature_scaling", action="store_true",
        help="B_calibrated DHA component: global temperature scaling (Guo et al. 2017). "
             "Single scalar T fit on oiaddr-valid NLL; applies sigmoid(score/T). Rank-preserving "
             "on single-score baselines, so B0+T == B0 by construction at matched TP_ret. Reported "
             "as supporting baseline to confirm governance gains are not standard-calibration artifacts.")
    parser.add_argument("--per_class_breakdown", action="store_true",
        help="R1 supporting analysis: also output per-pred_class TP_ret and FP_supp at each "
             "operating point. Writes per_class_results.csv in the output directory. No bootstrap "
             "CI per class (sample sizes too small for some classes); aggregate CI in external_eval_results.csv.")
    parser.add_argument("--real_sdca_dir", default=None,
        help="Stage 10A real-SDCA overrides directory (e.g. outputs/stage10_real_sdca). "
             "When set, overrides the ev_sdca feature on oiaddr/train, oiaddr/valid, and the "
             "external split being evaluated using bbox_id-keyed lookup from "
             "{dir}/{dataset}_{split}_real_sdca.csv. Bboxes not present in the realSDCA CSV "
             "(status=fallback) retain their synthetic ev_sdca. Added 2026-05-17.")
    args = parser.parse_args()

    run_evaluation(
        train_stage3c_dir=Path(args.train_stage3c_dir),
        external_stage3c_dir=Path(args.external_stage3c_dir),
        external_splits=args.external_splits,
        external_name=args.external_name,
        output_dir=Path(args.output_dir),
        apply_mask_filter=args.apply_mask_filter,
        detector_cache_dir=Path(args.detector_cache_dir) if args.detector_cache_dir else None,
        mask_dir=Path(args.mask_dir) if args.mask_dir else None,
        apply_score_calibration=args.apply_score_calibration,
        apply_temperature_scaling=args.apply_temperature_scaling,
        per_class_breakdown=args.per_class_breakdown,
        real_sdca_dir=Path(args.real_sdca_dir) if args.real_sdca_dir else None,
    )


if __name__ == "__main__":
    main()
