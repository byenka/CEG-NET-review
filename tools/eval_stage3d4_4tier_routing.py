#!/usr/bin/env python
"""
eval_stage3d4_4tier_routing.py
Phase 19N — Stage 3D.4: 2-tier vs 4-tier Routing Ablation.

Tests whether 4-tier governance routing (R1/R2/R3/R4) provides better
clinical governance quality than binary gate (RETAIN/SUPPRESS).

Architecture v3 specifies 4-tier routing as the target governance contract.
Current experiments use binary gate. This ablation validates the upgrade.

Tier mapping from governance categories:
  R1 (Certified)    : TP_EXACT             → highest confidence retain
  R2 (Provisional)  : CLINICALLY_CLOSE_MISMATCH + LOCALIZATION_WEAK → retain with flag
  R3 (Contested)    : UNSAFE_MISMATCH + LOW_CONF_CANDIDATE → needs clinical review
  R4 (Rejected)     : BACKGROUND_FP         → safe to suppress

Key governance metrics:
  R1_precision  : P(true R1 | model says R1) — certified detection reliability
  R4_purity     : P(true R4 | model suppresses) — safe-to-reject reliability
  tier_breakdown: among retained, fraction at each tier
  confusion_cost: weighted misrouting cost (R1→R4 is worst)

Models compared:
  M0_score    : score threshold (baseline, binary)
    M_binary    : MLP binary gate on selected evidence features
    M_4tier     : MLP 4-class on selected evidence features — predicts tier directly
  M_cascade   : 3 cascaded binary classifiers (ordinal thresholding)

Operating point protocol (consistent with Stage 3D.1/3D.2/3D.3):
  Thresholds / hyperparams selected on valid at TP_ret >= 0.85 / 0.80 / 0.75.
  All metrics reported on held-out test split.
  4-tier collapsed to 2-tier for FP_supp/TP_ret comparison.
  Additionally: R1_precision and R4_purity reported per operating point.

Inputs:
  outputs/stage3c_audit/governance_features_{train,valid,test}.csv

Outputs:
  outputs/stage3d_ablation/tier_routing_results.csv
  outputs/stage3d_ablation/tier_breakdown.csv
  outputs/stage3d_ablation/tier_routing_decision.md

Usage:
    python tools/eval_stage3d4_4tier_routing.py \\
      --stage3c_dir outputs/stage3c_audit \\
      --output_dir  outputs/stage3d_ablation
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from sklearn.metrics import average_precision_score, confusion_matrix
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    _SK = True
except ImportError:
    _SK = False
    print("[Stage3D.4] WARNING: scikit-learn not installed. Only M0 will run.")


def _assert_not_outputs_mirror_script() -> None:
    script_norm = str(Path(__file__).resolve()).replace("\\", "/").lower()
    if "/outputs/tools/" in script_norm:
        raise RuntimeError(
            "[Stage3D.4] Refusing to run from outputs/tools mirror. "
            "Run the canonical root script under tools/."
        )


_assert_not_outputs_mirror_script()

# ── Constants ─────────────────────────────────────────────────────────────────

_CLASS_ORDER = ["MA", "HE", "Hard_EX", "Soft_EX"]
_CLASS_IDX = {c: i for i, c in enumerate(_CLASS_ORDER)}

# 4-tier mapping
_TIER_MAP = {
    "TP_EXACT": 3,  # R1 Certified
    "CLINICALLY_CLOSE_MISMATCH": 2,  # R2 Provisional
    "LOCALIZATION_WEAK": 2,  # R2 Provisional
    "UNSAFE_MISMATCH": 1,  # R3 Contested
    "LOW_CONF_CANDIDATE": 1,  # R3 Contested
    "BACKGROUND_FP": 0,  # R4 Rejected
}
_TIER_LABELS = {
    0: "R4_Rejected",
    1: "R3_Contested",
    2: "R2_Provisional",
    3: "R1_Certified",
}

# Binary mapping (same as all prior Stage 3D)
_BINARY_RETAIN = frozenset(
    {"TP_EXACT", "CLINICALLY_CLOSE_MISMATCH", "LOCALIZATION_WEAK"}
)

# Weighted confusion cost matrix: cost[pred_tier, true_tier]
# Cost of routing true R1 to R4 (4 tiers wrong) = catastrophic
# Cost of routing true R4 to R1 (false certification) = also bad
# Diagonal = 0
_CONFUSION_COST = np.array(
    [
        # pred\true  R4   R3   R2   R1
        [0, 1, 2, 4],  # pred=R4
        [1, 0, 1, 3],  # pred=R3
        [2, 1, 0, 2],  # pred=R2
        [4, 3, 2, 0],  # pred=R1
    ],
    dtype=np.float32,
)

_TP_TARGETS = [0.85, 0.80, 0.75]
_THRESHOLD_GRID = np.round(np.linspace(0.001, 0.999, 499), 4)
_N_BOOTSTRAP = 1000
_BOOTSTRAP_SEED = 42

_FEATURE_PRESETS: Dict[str, List[str]] = {
    "legacy_sas_sdca": ["ev_sas", "ev_sdca"],
    "sas_sdca_pss": ["ev_sas", "ev_sdca", "ev_pss"],
    "sas_sdca_pss_aps": ["ev_sas", "ev_sdca", "ev_pss", "ev_aps"],
}

_CANONICAL_STAGE3C_DIR = (
    Path(__file__).resolve().parents[1] / "outputs" / "stage3c_audit"
)
_SUSPICIOUS_STAGE3C_MARKERS = (
    "/outputs/artefak/",
    "/outputs/src/",
    "/outputs/tools/",
    "/legacy/",
    "/historical/",
    "/stale/",
    "/stage3c_audit_pre",
)
_LEGACY_STAGE3C_METADATA_COLUMNS = {
    "split",
    "source_output_dir",
    "source_cache_dir",
    "source_annotations_dir",
    "compare_baseline_source",
}


def _norm_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").lower()


def _detect_suspicious_path(path: Path) -> Optional[str]:
    norm = _norm_path(path)
    for marker in _SUSPICIOUS_STAGE3C_MARKERS:
        if marker in norm:
            return marker
    return None


def _source_output_root_key(source_output_dir: str, split: str) -> str:
    """Normalize per-split Stage3C output dirs to one shared source root."""

    path = Path(source_output_dir)
    if path.name.lower() == split:
        path = path.parent
    return str(path.resolve())


def _validate_split_isolation(recs_by_split: Dict[str, List[dict]]) -> List[str]:
    errors: List[str] = []
    image_sets: Dict[str, set] = {}
    bbox_sets: Dict[str, set] = {}
    output_roots: Dict[str, str] = {}

    for split, records in recs_by_split.items():
        split_values = {
            str(r.get("split", "")).strip().lower()
            for r in records
            if str(r.get("split", "")).strip()
        }
        if split_values and split_values != {split}:
            errors.append(
                f"Split column mismatch for {split}: values={sorted(split_values)}"
            )

        image_sets[split] = {
            str(r.get("image_id", "")).strip()
            for r in records
            if str(r.get("image_id", "")).strip()
        }
        bbox_sets[split] = {
            str(r.get("bbox_id", "")).strip()
            for r in records
            if str(r.get("bbox_id", "")).strip()
        }

        source_outputs = {
            str(r.get("source_output_dir", "")).strip()
            for r in records
            if str(r.get("source_output_dir", "")).strip()
        }
        if len(source_outputs) != 1:
            errors.append(
                f"Split {split} has inconsistent source_output_dir values: {sorted(source_outputs)}"
            )
        elif source_outputs:
            output_roots[split] = _source_output_root_key(
                next(iter(source_outputs)), split
            )

    split_pairs = [("train", "valid"), ("train", "test"), ("valid", "test")]
    for left, right in split_pairs:
        ov_images = image_sets.get(left, set()) & image_sets.get(right, set())
        if ov_images:
            errors.append(
                f"Split isolation violated: image_id overlap between {left} and {right} (n={len(ov_images)})"
            )
        ov_bbox = bbox_sets.get(left, set()) & bbox_sets.get(right, set())
        if ov_bbox:
            errors.append(
                f"Split isolation violated: bbox_id overlap between {left} and {right} (n={len(ov_bbox)})"
            )

    if len(set(output_roots.values())) > 1:
        errors.append(
            "Cross-root Stage3C source detected across splits: " + str(output_roots)
        )

    return errors


# ── Data loading ──────────────────────────────────────────────────────────────


def _load_csv(path: Path) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_stage3c_audit(stage3c_dir: Path, split: str) -> dict:
    audit_path = stage3c_dir / f"governance_target_audit_{split}.json"
    if not audit_path.exists():
        raise RuntimeError(
            f"Missing provenance audit for split '{split}': {audit_path}"
        )
    with open(audit_path, encoding="utf-8") as f:
        audit = json.load(f)
    if str(audit.get("stage")) != "3C":
        raise RuntimeError(
            f"{audit_path} is not a Stage 3C audit (stage={audit.get('stage')})"
        )
    if str(audit.get("split")) != split:
        raise RuntimeError(
            f"Split mismatch: CSV split '{split}' but audit split '{audit.get('split')}'"
        )
    return audit


def _validate_and_load_stage3c_split(
    stage3c_dir: Path,
    split: str,
    features_path: Path,
    required_columns: List[str],
    required_finite_columns: List[str],
    allow_legacy_schema: bool = False,
) -> Tuple[List[dict], dict]:
    with open(features_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        records = list(reader)
        header = list(reader.fieldnames or [])

    missing = [col for col in required_columns if col not in header]
    if missing:
        missing_set = set(missing)
        if allow_legacy_schema and missing_set <= _LEGACY_STAGE3C_METADATA_COLUMNS:
            stage3c_root = str(stage3c_dir.resolve())
            print(
                "[Stage3D.4] WARN: legacy Stage3C schema detected for historical "
                f"baseline {features_path}; hydrating missing metadata columns: "
                f"{', '.join(missing)}"
            )
            for row in records:
                row.setdefault("split", split)
                row.setdefault("source_output_dir", stage3c_root)
                row.setdefault("source_cache_dir", "")
                row.setdefault("source_annotations_dir", "")
                row.setdefault("compare_baseline_source", stage3c_root)
        else:
            raise RuntimeError(
                f"{features_path} missing required columns: {', '.join(missing)}"
            )
    if not records:
        raise RuntimeError(f"{features_path} is empty")

    for col in required_finite_columns:
        finite_count = 0
        for row in records:
            raw = str(row.get(col, "")).strip()
            if raw == "":
                continue
            try:
                value = float(raw)
            except ValueError:
                continue
            if math.isfinite(value):
                finite_count += 1
        if finite_count == 0:
            raise RuntimeError(
                f"{features_path} has no finite values for required column '{col}'"
            )

    audit = _load_stage3c_audit(stage3c_dir, split)
    agg_total_raw = (audit.get("aggregation") or {}).get("total_boxes")
    if agg_total_raw is not None:
        try:
            agg_total = int(agg_total_raw)
        except (TypeError, ValueError):
            raise RuntimeError(
                f"Invalid aggregation.total_boxes in split '{split}': {agg_total_raw}"
            )
        if agg_total != len(records):
            raise RuntimeError(
                f"Row-count provenance mismatch for split '{split}': "
                f"CSV={len(records)} vs audit aggregation.total_boxes={agg_total}"
            )

    return records, audit


def _has_finite_feature(records: List[dict], col: str) -> bool:
    for row in records:
        raw = str(row.get(col, "")).strip()
        if raw == "":
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        if math.isfinite(value):
            return True
    return False


def _all_splits_have_feature(recs_by_split: Dict[str, List[dict]], col: str) -> bool:
    return all(_has_finite_feature(recs, col) for recs in recs_by_split.values())


def _is_quality_anatomy_available(audits_by_split: Dict[str, dict]) -> bool:
    for audit in audits_by_split.values():
        anatomy = audit.get("anatomy") or {}
        if str(anatomy.get("quality_mode")) != "strict":
            return False
        if str(anatomy.get("provider_mode")) != "external_pretrained":
            return False
        if not bool(anatomy.get("strict_anatomy_gate", False)):
            return False
    return True


def _resolve_feature_columns(
    feature_mode: str,
    recs_by_split: Dict[str, List[dict]],
    audits_by_split: Dict[str, dict],
) -> List[str]:
    if feature_mode in _FEATURE_PRESETS:
        cols = list(_FEATURE_PRESETS[feature_mode])
    elif feature_mode == "auto":
        if not _all_splits_have_feature(recs_by_split, "ev_pss"):
            raise RuntimeError(
                "feature_mode=auto requires finite ev_pss values on train/valid/test."
            )
        quality_anatomy = _is_quality_anatomy_available(audits_by_split)
        if quality_anatomy:
            if not _all_splits_have_feature(recs_by_split, "ev_aps"):
                raise RuntimeError(
                    "Quality anatomy detected from Stage3C audits but ev_aps is "
                    "missing/non-finite on one or more splits."
                )
            cols = list(_FEATURE_PRESETS["sas_sdca_pss_aps"])
        else:
            cols = list(_FEATURE_PRESETS["sas_sdca_pss"])
    else:
        raise RuntimeError(f"Unknown feature_mode: {feature_mode}")

    if cols[0] != "ev_sas":
        raise RuntimeError(
            "First feature must be ev_sas so M0_score remains SAS-threshold comparable."
        )

    if "ev_pss" in cols and not _all_splits_have_feature(recs_by_split, "ev_pss"):
        raise RuntimeError(
            "Selected feature set requires ev_pss but finite values are missing "
            "on one or more splits."
        )
    if "ev_aps" in cols and not _all_splits_have_feature(recs_by_split, "ev_aps"):
        raise RuntimeError(
            "Selected feature set requires ev_aps but finite values are missing "
            "on one or more splits."
        )
    return cols


def _parse(records: List[dict], feature_cols: List[str]) -> Tuple[
    np.ndarray,  # X: [N, D] — selected governance features
    np.ndarray,  # y_binary: [N] — 0/1 SUPPRESS/RETAIN
    np.ndarray,  # y_tier: [N] — 0..3 R4..R1
    np.ndarray,  # is_unsafe: [N] bool
    List[str],  # image_ids
    List[str],  # categories (raw)
]:
    X_list, yb_list, yt_list, un_list, img_list, cat_list = [], [], [], [], [], []

    for r in records:
        x_row: List[float] = []
        try:
            for col in feature_cols:
                value = float(r[col])
                if not math.isfinite(value):
                    raise ValueError(f"non-finite value for {col}")
                x_row.append(value)
        except (KeyError, ValueError):
            continue
        cat = r.get("category", "")
        if cat not in _TIER_MAP:
            continue
        X_list.append(x_row)
        yb_list.append(1 if cat in _BINARY_RETAIN else 0)
        yt_list.append(_TIER_MAP[cat])
        un_list.append(cat == "UNSAFE_MISMATCH")
        img_list.append(str(r.get("image_id", "")))
        cat_list.append(cat)

    X = np.array(X_list, dtype=np.float64)
    yb = np.array(yb_list, dtype=np.int32)
    yt = np.array(yt_list, dtype=np.int32)
    un = np.array(un_list, dtype=bool)
    return X, yb, yt, un, img_list, cat_list


# ── Threshold / operating point helpers ───────────────────────────────────────


def _binary_retain_score(
    X: np.ndarray,
    m: object,
    sc: object,
    is_4tier: bool = False,
) -> np.ndarray:
    """Return P(retain) for binary model, or P(R1)+P(R2) for 4-tier model."""
    Xs = sc.transform(X)
    probs = m.predict_proba(Xs)
    if is_4tier:
        # Classes: model.classes_ should be [0,1,2,3]
        classes = list(m.classes_)
        retain_p = np.zeros(len(X), dtype=np.float64)
        for tier in [2, 3]:  # R2 + R1
            if tier in classes:
                retain_p += probs[:, classes.index(tier)]
        return retain_p
    else:
        classes = list(m.classes_)
        return probs[:, classes.index(1)].astype(np.float64)


def _find_thresh(
    y_binary: np.ndarray,
    score: np.ndarray,
    target: float,
) -> Optional[float]:
    n_ret = int((y_binary == 1).sum())
    if n_ret == 0:
        return None
    best: Optional[float] = None
    for t in sorted(_THRESHOLD_GRID):
        retained = (score >= float(t)).astype(np.int32)
        tp_r = int(((y_binary == 1) & (retained == 1)).sum()) / n_ret
        if tp_r >= target:
            best = float(t)
    return best


def _metrics_binary(
    y_binary: np.ndarray,
    y_tier: np.ndarray,
    score: np.ndarray,
    is_unsafe: np.ndarray,
    thresh: float,
) -> dict:
    retained = score >= thresh
    suppressed = ~retained
    n_ret = int((y_binary == 1).sum())
    n_sup = int((y_binary == 0).sum())

    tp_ret = int(((y_binary == 1) & retained).sum()) / max(n_ret, 1)
    fp_sup = int(((y_binary == 0) & suppressed).sum()) / max(n_sup, 1)
    prec = int((y_binary[retained] == 1).sum()) / max(int(retained.sum()), 1)

    # Tier breakdown WITHIN retained predictions
    ret_tiers = y_tier[retained]
    tier_breakdown_ret: Dict[int, float] = {}
    n_r = len(ret_tiers)
    for t in range(4):
        tier_breakdown_ret[t] = int((ret_tiers == t).sum()) / max(n_r, 1)

    # Tier breakdown WITHIN suppressed predictions
    sup_tiers = y_tier[suppressed]
    tier_breakdown_sup: Dict[int, float] = {}
    n_s = len(sup_tiers)
    for t in range(4):
        tier_breakdown_sup[t] = int((sup_tiers == t).sum()) / max(n_s, 1)

    # R1_precision: of things called R1 by routing, fraction truly R1
    # Approximated here as: of things RETAINED, fraction truly R1
    r1_prec = tier_breakdown_ret.get(3, 0.0)
    r4_purity = tier_breakdown_sup.get(0, 0.0)

    un_sup = (
        int((is_unsafe & suppressed).sum()) / int(is_unsafe.sum())
        if is_unsafe.any()
        else None
    )

    return {
        "tp_ret": round(tp_ret, 4),
        "fp_supp": round(fp_sup, 4),
        "precision": round(prec, 4),
        "r1_prec": round(r1_prec, 4),
        "r4_purity": round(r4_purity, 4),
        "unsafe_supp": round(un_sup, 4) if un_sup is not None else None,
        "ret_R1": round(tier_breakdown_ret.get(3, 0.0), 4),
        "ret_R2": round(tier_breakdown_ret.get(2, 0.0), 4),
        "ret_R3": round(tier_breakdown_ret.get(1, 0.0), 4),
        "ret_R4": round(tier_breakdown_ret.get(0, 0.0), 4),
        "sup_R1": round(tier_breakdown_sup.get(3, 0.0), 4),
        "sup_R2": round(tier_breakdown_sup.get(2, 0.0), 4),
        "sup_R3": round(tier_breakdown_sup.get(1, 0.0), 4),
        "sup_R4": round(tier_breakdown_sup.get(0, 0.0), 4),
    }


def _weighted_confusion_cost(
    y_tier: np.ndarray,
    pred_tier: np.ndarray,
) -> float:
    """Mean per-box confusion cost using clinical cost matrix."""
    costs = []
    for true_t, pred_t in zip(y_tier, pred_tier):
        costs.append(float(_CONFUSION_COST[int(pred_t), int(true_t)]))
    return round(float(np.mean(costs)), 4) if costs else 0.0


def _4tier_argmax(
    y_tier: np.ndarray,
    X: np.ndarray,
    m: object,
    sc: object,
) -> Tuple[np.ndarray, float]:
    """Predict tier by argmax. Returns (predicted_tier, weighted_confusion_cost)."""
    Xs = sc.transform(X)
    pred = m.predict(Xs).astype(np.int32)
    cost = _weighted_confusion_cost(y_tier, pred)
    return pred, cost


# ── Bootstrap ─────────────────────────────────────────────────────────────────


def _bootstrap_gain(
    img_ids: List[str],
    y_binary: np.ndarray,
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
        idx = []
        for img in samp:
            idx.extend(img2idx[img])
        ia = np.array(idx, dtype=np.int64)
        yb = y_binary[ia]
        if yb.sum() == 0 or (1 - yb).sum() == 0:
            continue
        sup = yb == 0
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


def _bootstrap_r1_prec(
    img_ids: List[str],
    y_tier: np.ndarray,
    score: np.ndarray,
    thresh: float,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """Bootstrap CI for R1_precision (fraction of retained that are true R1)."""
    rng = np.random.default_rng(seed + 1)
    unique = list(dict.fromkeys(img_ids))
    n_u = len(unique)
    img2idx: Dict[str, List[int]] = {}
    for i, img in enumerate(img_ids):
        img2idx.setdefault(img, []).append(i)
    vals: List[float] = []
    for _ in range(n_boot):
        samp = rng.choice(unique, size=n_u, replace=True)
        idx = []
        for img in samp:
            idx.extend(img2idx[img])
        ia = np.array(idx, dtype=np.int64)
        retained = score[ia] >= thresh
        n_r = int(retained.sum())
        if n_r == 0:
            continue
        r1_p = int((y_tier[ia][retained] == 3).sum()) / n_r
        vals.append(r1_p)
    if len(vals) < 10:
        return {"mean": None, "ci_lo": None, "ci_hi": None}
    arr = np.array(vals)
    return {
        "mean": round(float(arr.mean()), 4),
        "ci_lo": round(float(np.percentile(arr, 2.5)), 4),
        "ci_hi": round(float(np.percentile(arr, 97.5)), 4),
    }


# ── Output writers ────────────────────────────────────────────────────────────

_RES_FIELDS = [
    "model",
    "routing_type",
    "feature_columns",
    "compare_baseline_source",
    "source_stage3c_dir",
    "tp_ret_target",
    "thresh_valid",
    "tp_ret_valid",
    "fp_supp_valid",
    "r1_prec_valid",
    "r4_purity_valid",
    "tp_ret_test",
    "fp_supp_test",
    "r1_prec_test",
    "r4_purity_test",
    "fp_supp_gain_over_m0",
    "r1_prec_gain_over_m0",
    "r4_purity_gain_over_m0",
    "boot_mean_gain",
    "boot_ci_lo",
    "boot_ci_hi",
    "boot_frac_pos",
    "r1_prec_boot_mean",
    "r1_prec_ci_lo",
    "r1_prec_ci_hi",
    "achieves_target_test",
]

_TIER_BREAKDOWN_FIELDS = [
    "model",
    "routing_type",
    "tp_ret_target",
    "split",
    "ret_R1",
    "ret_R2",
    "ret_R3",
    "ret_R4",
    "sup_R1",
    "sup_R2",
    "sup_R3",
    "sup_R4",
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
    breakdown_rows: List[dict],
    path: Path,
    feature_columns: List[str],
    run_provenance: dict,
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Stage 3D.4 — 2-tier vs 4-tier Routing Decision\n\n")
        f.write("Selected feature columns: " + ", ".join(feature_columns) + "\n\n")
        f.write(
            "Source stage3c_dir: " + str(run_provenance.get("stage3c_dir", "")) + "\n\n"
        )
        f.write(
            "Compare baseline source: "
            + str(run_provenance.get("compare_baseline_source", ""))
            + "\n\n"
        )
        f.write(
            "> **Key question**: Does 4-tier routing (R1/R2/R3/R4) provide "
            "better clinical governance quality than binary RETAIN/SUPPRESS?\n\n"
        )
        f.write(
            "Tier mapping:\n"
            "- R1 Certified = TP_EXACT (high-confidence true positive)\n"
            "- R2 Provisional = CLOSE_MISMATCH + LOCALIZATION_WEAK\n"
            "- R3 Contested = UNSAFE_MISMATCH + LOW_CONF\n"
            "- R4 Rejected = BACKGROUND_FP\n\n"
        )

        for target in _TP_TARGETS:
            f.write(f"## TP_ret ≥ {target:.2f}\n\n")
            f.write(
                "| Model | Routing | Thresh V | "
                "FP_supp V | FP_supp T | Gain/M0 | "
                "R1_prec T | R4_purity T |\n"
            )
            f.write("|---|---|---:|---:|---:|---:|---:|---:|\n")
            for r in res_rows:
                if abs(r.get("tp_ret_target", -1) - target) > 1e-6:
                    continue
                g0 = r.get("fp_supp_gain_over_m0")
                f.write(
                    f"| {r['model']} | {r['routing_type']} "
                    f"| {_fmt(r.get('thresh_valid'))} "
                    f"| {_fmt(r.get('fp_supp_valid'))} "
                    f"| {_fmt(r.get('fp_supp_test'))} "
                    f"| {_fmt(g0, '+.4f') if g0 is not None else '—'} "
                    f"| {_fmt(r.get('r1_prec_test'))} "
                    f"| {_fmt(r.get('r4_purity_test'))} |\n"
                )
            f.write("\n")

        f.write("## Tier Breakdown Within Retained Predictions (Test)\n\n")
        f.write(
            "Fraction of RETAINED predictions at each governance tier.\n"
            "Higher R1% = more certified detections, higher governance quality.\n\n"
        )
        for target in _TP_TARGETS:
            f.write(f"### TP_ret ≥ {target:.2f}\n\n")
            f.write("| Model | R1_Cert% | R2_Prov% | R3_Cont% | R4_Rej% |\n")
            f.write("|---|---:|---:|---:|---:|\n")
            for r in breakdown_rows:
                if (
                    abs(r.get("tp_ret_target", -1) - target) > 1e-6
                    or r.get("split") != "test"
                ):
                    continue
                f.write(
                    f"| {r['model']} "
                    f"| {100*r.get('ret_R1', 0):.1f}% "
                    f"| {100*r.get('ret_R2', 0):.1f}% "
                    f"| {100*r.get('ret_R3', 0):.1f}% "
                    f"| {100*r.get('ret_R4', 0):.1f}% |\n"
                )
            f.write("\n")

        f.write("## Bootstrap FP_supp_gain CI (vs M0, 1000 resamples)\n\n")
        for target in _TP_TARGETS:
            f.write(f"### TP_ret ≥ {target:.2f}\n\n")
            f.write("| Model | Mean gain | 95% CI | P(>0) |\n")
            f.write("|---|---:|---|---:|\n")
            for r in res_rows:
                if abs(r.get("tp_ret_target", -1) - target) > 1e-6:
                    continue
                ci = (
                    f"[{_fmt(r.get('boot_ci_lo'))}, {_fmt(r.get('boot_ci_hi'))}]"
                    if r.get("boot_ci_lo") is not None
                    else "N/A"
                )
                f.write(
                    f"| {r['model']} "
                    f"| {_fmt(r.get('boot_mean_gain'), '+.4f') if r.get('boot_mean_gain') is not None else 'N/A'} "
                    f"| {ci} "
                    f"| {_fmt(r.get('boot_frac_pos'))} |\n"
                )
            f.write("\n")

        f.write("## Verdict\n\n")
        f.write("```\n")

        m0_r1: Dict[float, float] = {}
        m0_fp: Dict[float, float] = {}
        m4_r1: Dict[float, float] = {}
        m4_fp: Dict[float, float] = {}
        mb_r1: Dict[float, float] = {}

        for r in res_rows:
            tgt = r.get("tp_ret_target")
            if r["model"] == "M0_score":
                m0_fp[tgt] = r.get("fp_supp_test") or 0
                m0_r1[tgt] = r.get("r1_prec_test") or 0
            if r["model"] == "M_4tier":
                m4_fp[tgt] = r.get("fp_supp_test") or 0
                m4_r1[tgt] = r.get("r1_prec_test") or 0
            if r["model"] == "M_binary":
                mb_r1[tgt] = r.get("r1_prec_test") or 0

        wins_fp = sum(1 for t in _TP_TARGETS if m4_fp.get(t, 0) > m0_fp.get(t, 0))
        wins_r1 = sum(1 for t in _TP_TARGETS if m4_r1.get(t, 0) > mb_r1.get(t, 0))
        f.write(
            f"M_4tier beats M0 FP_supp at {wins_fp}/{len(_TP_TARGETS)} TP targets.\n"
            f"M_4tier R1_precision beats M_binary at {wins_r1}/{len(_TP_TARGETS)} targets.\n\n"
        )

        if wins_r1 >= 2:
            f.write(
                "→ 4-tier routing produces higher fraction of certified (R1) "
                "detections within retained set.\n"
                "→ Architecture v3 4-tier contract validated empirically.\n"
            )
        else:
            f.write(
                "→ 4-tier routing does not consistently improve R1_precision "
                "over binary gate.\n"
                "→ Binary gate with tier-labelled output is sufficient for Mode 1.\n"
            )
        f.write("```\n\n")

        f.write("## Clinical Interpretation\n\n")
        f.write(
            "The key governance value of 4-tier over binary:\n\n"
            "- **Binary** says: retain or suppress (no confidence level)\n"
            "- **4-tier** says: certify (R1), flag provisional (R2), "
            "contest (R3), or reject (R4)\n\n"
            "For clinical deployment:\n"
            "- R1 can be auto-reported\n"
            "- R2 goes to clinician review queue\n"
            "- R3 triggers grading committee review\n"
            "- R4 is silently suppressed\n\n"
            "This audit trail is the core governance contribution "
            "(C2 per architecture v3), independent of whether 4-tier "
            "improves FP_supp over binary gate.\n"
        )

        f.write("\n## Next Step\n\n")
        f.write(
            "1. If 4-tier routing passes: include in paper as governance refinement.\n"
            "2. Run C1 ablation (directed vs undirected KG edges, "
            "clinical SDM vs uniform SDM).\n"
            "3. Build comparison table with published DR detection methods.\n"
            "4. Begin paper draft (Stage 4).\n"
        )


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 3D.4: 2-tier vs 4-tier routing ablation."
    )
    parser.add_argument("--stage3c_dir", default="outputs/stage3c_audit")
    parser.add_argument("--output_dir", default="outputs/stage3d_ablation")
    parser.add_argument(
        "--feature_mode",
        default="auto",
        choices=[
            "auto",
            "sas_sdca_pss",
            "sas_sdca_pss_aps",
            "legacy_sas_sdca",
        ],
        help=(
            "Feature set for learned gates. auto requires ev_pss and includes "
            "ev_aps only when Stage3C audits indicate strict external anatomy quality mode."
        ),
    )
    parser.add_argument(
        "--allow_historical_inputs",
        action="store_true",
        help="Allow suspicious historical/artefact stage3c_dir paths (default: fail-fast).",
    )
    parser.add_argument("--n_bootstrap", type=int, default=_N_BOOTSTRAP)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    s3c = Path(args.stage3c_dir)

    path_warnings: List[str] = []
    marker = _detect_suspicious_path(s3c)
    if marker is not None:
        msg = f"stage3c_dir looks historical/suspicious ({marker}): {s3c}"
        if args.allow_historical_inputs:
            print(f"[Stage3D.4] WARN: {msg} (allowed by --allow_historical_inputs)")
            path_warnings.append(msg)
        else:
            print(
                f"[Stage3D.4] ERROR: {msg} "
                "(set --allow_historical_inputs to override)"
            )
            sys.exit(2)
    stage3c_norm = _norm_path(s3c)
    if (
        s3c.resolve() != _CANONICAL_STAGE3C_DIR.resolve()
        and "/outputs/point123_" not in stage3c_norm
    ):
        msg = (
            f"Non-canonical stage3c_dir detected: {s3c}. "
            f"Canonical default is {_CANONICAL_STAGE3C_DIR}."
        )
        print(f"[Stage3D.4] WARN: {msg}")
        path_warnings.append(msg)
    elif s3c.resolve() != _CANONICAL_STAGE3C_DIR.resolve():
        print(f"[Stage3D.4] INFO: stage3c_dir is a Point123 run artifact root: {s3c}")

    print("[Stage3D.4] === Phase 19N Stage 3D.4 Tier Routing Ablation ===")
    print(
        f"[Stage3D.4] Tier mapping: R1=TP_EXACT | R2=CLOSE+LOC_WEAK | "
        f"R3=UNSAFE+LOW_CONF | R4=BG_FP"
    )

    # ── Load data ──────────────────────────────────────────────────────────────
    required_columns = [
        "bbox_id",
        "image_id",
        "split",
        "source_output_dir",
        "source_cache_dir",
        "source_annotations_dir",
        "compare_baseline_source",
        "category",
        "ev_sas",
        "ev_sdca",
        "ev_pss",
        "ev_aps",
    ]
    required_finite_columns = ["ev_sas", "ev_sdca", "ev_pss"]
    recs: Dict[str, List[dict]] = {}
    audits: Dict[str, dict] = {}
    for sp in ["train", "valid", "test"]:
        fp = s3c / f"governance_features_{sp}.csv"
        if not fp.exists():
            print(f"[Stage3D.4] ERROR: {fp} not found.")
            import sys

            sys.exit(1)
        try:
            recs[sp], audits[sp] = _validate_and_load_stage3c_split(
                stage3c_dir=s3c,
                split=sp,
                features_path=fp,
                required_columns=required_columns,
                required_finite_columns=required_finite_columns,
                allow_legacy_schema=bool(args.allow_historical_inputs),
            )
        except RuntimeError as exc:
            print(f"[Stage3D.4] ERROR: {exc}")
            sys.exit(2)

    split_isolation_errors = _validate_split_isolation(recs)
    if split_isolation_errors:
        print("[Stage3D.4] ERROR: split-isolation validation failed:")
        for err in split_isolation_errors:
            print(f"[Stage3D.4]   - {err}")
        sys.exit(2)
    print("[Stage3D.4] Split isolation check: PASS")

    try:
        feature_columns = _resolve_feature_columns(
            feature_mode=args.feature_mode,
            recs_by_split=recs,
            audits_by_split=audits,
        )
    except RuntimeError as exc:
        print(f"[Stage3D.4] ERROR: {exc}")
        sys.exit(2)

    print(f"[Stage3D.4] Feature mode: {args.feature_mode} -> " f"{feature_columns}")

    run_provenance = {
        "repo_root": str(Path(__file__).resolve().parents[1]),
        "script_path": str(Path(__file__).resolve()),
        "stage3c_dir": str(s3c.resolve()),
        "output_dir": str(out_dir.resolve()),
        "compare_baseline_source": str(s3c.resolve()),
        "feature_mode": args.feature_mode,
        "selected_feature_columns": feature_columns,
        "allow_historical_inputs": bool(args.allow_historical_inputs),
        "path_warnings": path_warnings,
        "split_isolation": "PASS",
    }

    X_tr, yb_tr, yt_tr, un_tr, img_tr, _ = _parse(recs["train"], feature_columns)
    X_va, yb_va, yt_va, un_va, img_va, _ = _parse(recs["valid"], feature_columns)
    X_te, yb_te, yt_te, un_te, img_te, _ = _parse(recs["test"], feature_columns)

    for sp, yb, yt in [
        ("train", yb_tr, yt_tr),
        ("valid", yb_va, yt_va),
        ("test", yb_te, yt_te),
    ]:
        tier_dist = {t: int((yt == t).sum()) for t in range(4)}
        print(
            f"[Stage3D.4]   {sp}: {len(yb)} boxes  "
            f"R1={tier_dist[3]}({100*tier_dist[3]/max(len(yb),1):.1f}%)  "
            f"R2={tier_dist[2]}({100*tier_dist[2]/max(len(yb),1):.1f}%)  "
            f"R3={tier_dist[1]}({100*tier_dist[1]/max(len(yb),1):.1f}%)  "
            f"R4={tier_dist[0]}({100*tier_dist[0]/max(len(yb),1):.1f}%)"
        )

    if not _SK:
        print("[Stage3D.4] sklearn unavailable — only M0 baseline will run.")

    # ── M0: score threshold (binary) ───────────────────────────────────────────
    # Score = X[:, 0] (SAS, which equals detection confidence for single-head R2a)
    sc_m0_va = X_va[:, 0].astype(np.float64)
    sc_m0_te = X_te[:, 0].astype(np.float64)

    # ── Train models ──────────────────────────────────────────────────────────
    models_scores_va: Dict[str, np.ndarray] = {"M0_score": sc_m0_va}
    models_scores_te: Dict[str, np.ndarray] = {"M0_score": sc_m0_te}
    models_is4tier: Dict[str, bool] = {"M0_score": False}
    models_4tier_pred_te: Dict[str, Optional[np.ndarray]] = {"M0_score": None}
    models_4tier_cost_te: Dict[str, Optional[float]] = {"M0_score": None}

    if _SK:
        sc_std = StandardScaler()
        X_tr_s = sc_std.fit_transform(X_tr)
        print(f"[Stage3D.4] Training learned gates with features={feature_columns}")

        # M_binary: MLP 2-class
        m_bin = MLPClassifier(
            hidden_layer_sizes=(32, 16),
            max_iter=300,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            verbose=False,
        )
        m_bin.fit(X_tr_s, yb_tr)
        print(f"[Stage3D.4] M_binary trained ({getattr(m_bin,'n_iter_','?')} iter)")
        models_scores_va["M_binary"] = _binary_retain_score(
            X_va, m_bin, sc_std, is_4tier=False
        )
        models_scores_te["M_binary"] = _binary_retain_score(
            X_te, m_bin, sc_std, is_4tier=False
        )
        models_is4tier["M_binary"] = False
        models_4tier_pred_te["M_binary"] = None
        models_4tier_cost_te["M_binary"] = None

        # M_4tier: MLP 4-class
        m_4t = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=400,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            verbose=False,
        )
        m_4t.fit(X_tr_s, yt_tr)
        n_classes = len(m_4t.classes_)
        print(
            f"[Stage3D.4] M_4tier trained ({getattr(m_4t,'n_iter_','?')} iter) "
            f"classes={n_classes}"
        )
        models_scores_va["M_4tier"] = _binary_retain_score(
            X_va, m_4t, sc_std, is_4tier=True
        )
        models_scores_te["M_4tier"] = _binary_retain_score(
            X_te, m_4t, sc_std, is_4tier=True
        )
        models_is4tier["M_4tier"] = True
        pred_4t, cost_4t = _4tier_argmax(yt_te, X_te, m_4t, sc_std)
        models_4tier_pred_te["M_4tier"] = pred_4t
        models_4tier_cost_te["M_4tier"] = cost_4t
        print(f"[Stage3D.4]   M_4tier weighted confusion cost (test): {cost_4t:.4f}")

        # For M_binary confusion cost: collapse to binary → assign R1=3,R2=2 for retain; R3=1,R4=0 for suppress
        # Use binary predictions mapped to closest tier (coarse comparison)
        bin_pred = (models_scores_te["M_binary"] >= 0.5).astype(np.int32)
        # Map binary to tier: retain=2 (R2 provisional), suppress=0 (R4 rejected)
        bin_tier_approx = np.where(bin_pred == 1, 2, 0).astype(np.int32)
        cost_bin = _weighted_confusion_cost(yt_te, bin_tier_approx)
        models_4tier_pred_te["M_binary"] = bin_tier_approx
        models_4tier_cost_te["M_binary"] = cost_bin
        print(
            f"[Stage3D.4]   M_binary weighted confusion cost (test, approx): {cost_bin:.4f}"
        )

    # ── Operating point calibration ────────────────────────────────────────────
    model_order = ["M0_score", "M_binary", "M_4tier"]
    b0_thresh: Dict[float, Optional[float]] = {}
    b0_fp_te: Dict[float, Optional[float]] = {}
    b0_r1_te: Dict[float, Optional[float]] = {}
    b0_r4_te: Dict[float, Optional[float]] = {}

    for tgt in _TP_TARGETS:
        th = _find_thresh(yb_va, sc_m0_va, tgt)
        b0_thresh[tgt] = th
        if th is not None:
            m0_te = _metrics_binary(yb_te, yt_te, sc_m0_te, un_te, th)
            b0_fp_te[tgt] = m0_te["fp_supp"]
            b0_r1_te[tgt] = m0_te["r1_prec"]
            b0_r4_te[tgt] = m0_te["r4_purity"]

    print(f"\n[Stage3D.4] Operating point calibration...")
    res_rows: List[dict] = []
    breakdown_rows: List[dict] = []
    all_thresh: Dict[Tuple[str, float], Optional[float]] = {}

    for name in model_order:
        if name not in models_scores_va:
            continue
        sv = models_scores_va[name]
        st = models_scores_te[name]

        for tgt in _TP_TARGETS:
            th = _find_thresh(yb_va, sv, tgt)
            all_thresh[(name, tgt)] = th

            if th is None:
                print(f"[Stage3D.4]   {name:<14} TP>={tgt:.2f}: cannot reach target")
                continue

            m_va = _metrics_binary(yb_va, yt_va, sv, un_va, th)
            m_te = _metrics_binary(yb_te, yt_te, st, un_te, th)
            b0_f = b0_fp_te.get(tgt)
            b0_r = b0_r1_te.get(tgt)
            gain_fp = (
                round(m_te["fp_supp"] - b0_f, 4)
                if b0_f is not None and name != "M0_score"
                else None
            )
            gain_r1 = (
                round(m_te["r1_prec"] - b0_r, 4)
                if b0_r is not None and name != "M0_score"
                else None
            )
            gain_r4 = (
                round(m_te["r4_purity"] - b0_r4_te.get(tgt, 0), 4)
                if name != "M0_score"
                else None
            )

            gstr = f"{gain_fp:+.4f}" if gain_fp is not None else "(M0)"
            print(
                f"[Stage3D.4]   {name:<14} TP>={tgt:.2f}: "
                f"thresh={th:.4f}  "
                f"FP_supp(V={m_va['fp_supp']:.3f} T={m_te['fp_supp']:.3f})  "
                f"R1_prec(T={m_te['r1_prec']:.3f})  "
                f"R4_purity(T={m_te['r4_purity']:.3f})  "
                f"gain/M0={gstr}"
            )

            # Bootstrap FP gain
            boot = {
                "mean_gain": None,
                "ci_lo": None,
                "ci_hi": None,
                "frac_positive": None,
                "n": 0,
            }
            r1_boot = {"mean": None, "ci_lo": None, "ci_hi": None}
            if name != "M0_score" and b0_thresh.get(tgt) is not None:
                boot = _bootstrap_gain(
                    img_te,
                    yb_te,
                    st,
                    sc_m0_te,
                    th,
                    b0_thresh[tgt],
                    args.n_bootstrap,
                    _BOOTSTRAP_SEED,
                )
                r1_boot = _bootstrap_r1_prec(
                    img_te,
                    yt_te,
                    st,
                    th,
                    args.n_bootstrap,
                    _BOOTSTRAP_SEED,
                )

            res_rows.append(
                {
                    "model": name,
                    "routing_type": "4tier" if models_is4tier.get(name) else "binary",
                    "feature_columns": "|".join(feature_columns),
                    "compare_baseline_source": run_provenance[
                        "compare_baseline_source"
                    ],
                    "source_stage3c_dir": run_provenance["stage3c_dir"],
                    "tp_ret_target": tgt,
                    "thresh_valid": th,
                    "tp_ret_valid": m_va["tp_ret"],
                    "fp_supp_valid": m_va["fp_supp"],
                    "r1_prec_valid": m_va["r1_prec"],
                    "r4_purity_valid": m_va["r4_purity"],
                    "tp_ret_test": m_te["tp_ret"],
                    "fp_supp_test": m_te["fp_supp"],
                    "r1_prec_test": m_te["r1_prec"],
                    "r4_purity_test": m_te["r4_purity"],
                    "fp_supp_gain_over_m0": gain_fp,
                    "r1_prec_gain_over_m0": gain_r1,
                    "r4_purity_gain_over_m0": gain_r4,
                    "boot_mean_gain": boot["mean_gain"],
                    "boot_ci_lo": boot["ci_lo"],
                    "boot_ci_hi": boot["ci_hi"],
                    "boot_frac_pos": boot.get("frac_positive"),
                    "r1_prec_boot_mean": r1_boot["mean"],
                    "r1_prec_ci_lo": r1_boot["ci_lo"],
                    "r1_prec_ci_hi": r1_boot["ci_hi"],
                    "achieves_target_test": (m_te["tp_ret"] or 0) >= tgt,
                }
            )

            for split_name, m_split in [("valid", m_va), ("test", m_te)]:
                breakdown_rows.append(
                    {
                        "model": name,
                        "routing_type": (
                            "4tier" if models_is4tier.get(name) else "binary"
                        ),
                        "tp_ret_target": tgt,
                        "split": split_name,
                        "ret_R1": m_split["ret_R1"],
                        "ret_R2": m_split["ret_R2"],
                        "ret_R3": m_split["ret_R3"],
                        "ret_R4": m_split["ret_R4"],
                        "sup_R1": m_split["sup_R1"],
                        "sup_R2": m_split["sup_R2"],
                        "sup_R3": m_split["sup_R3"],
                        "sup_R4": m_split["sup_R4"],
                    }
                )

    # ── Report 4-tier confusion costs ─────────────────────────────────────────
    print(f"\n[Stage3D.4] Weighted confusion cost (clinical SDM, argmax on test):")
    for name in model_order:
        cost = models_4tier_cost_te.get(name)
        if cost is not None:
            print(f"[Stage3D.4]   {name:<14}: {cost:.4f}")

    # ── True tier distribution ─────────────────────────────────────────────────
    print(f"\n[Stage3D.4] True tier distribution (test):")
    for t, label in sorted(_TIER_LABELS.items(), reverse=True):
        n = int((yt_te == t).sum())
        print(f"[Stage3D.4]   {label}: {n} ({100*n/max(len(yt_te),1):.1f}%)")

    # ── Write outputs ──────────────────────────────────────────────────────────
    rp = out_dir / "tier_routing_results.csv"
    _write_csv(res_rows, _RES_FIELDS, rp)
    print(f"\n[Stage3D.4] {rp}")

    bp = out_dir / "tier_breakdown.csv"
    _write_csv(breakdown_rows, _TIER_BREAKDOWN_FIELDS, bp)
    print(f"[Stage3D.4] {bp}")

    md = out_dir / "tier_routing_decision.md"
    _write_md(res_rows, breakdown_rows, md, feature_columns, run_provenance)
    print(f"[Stage3D.4] {md}")

    prov_path = out_dir / "stage3d4_run_provenance.json"
    with open(prov_path, "w", encoding="utf-8") as f:
        json.dump(run_provenance, f, indent=2)
    print(f"[Stage3D.4] {prov_path}")

    print(f"\n[Stage3D.4] Done.")
    print(
        "[Stage3D.4] Key questions answered:\n"
        "  1. Does 4-tier improve FP_supp over binary gate?\n"
        "  2. Does 4-tier routing certify more R1 detections within retained set?\n"
        "  3. What fraction of suppressed detections are truly R4 (safe to reject)?\n"
        "  → Check tier_routing_decision.md for clinical governance verdict."
    )


if __name__ == "__main__":
    main()
