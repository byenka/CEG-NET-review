#!/usr/bin/env python
"""
train_stage3d_lgg.py
Phase 19N — Stage 3D: Learned Governance Gate (LGG) training and evaluation.

Trains and evaluates LGG baselines B0–B3 using governance target labels
(from Stage 3C) and evidence features from the current primary-detector pipeline.

Binary action gate:
  RETAIN  (label=1): TP_EXACT + CLINICALLY_CLOSE_MISMATCH + LOCALIZATION_WEAK
  SUPPRESS(label=0): BACKGROUND_FP + LOW_CONF_CANDIDATE + UNSAFE_MISMATCH

Baselines:
  B0:        score-threshold sweep (best-F1 threshold on train)
  B_msp:     max-softmax-probability threshold   — same-family selective-prediction baseline
  B_margin:  class-margin (top1-top2) threshold  — same-family selective-prediction baseline
  B_entropy: predictive-entropy threshold        — same-family selective-prediction baseline
  B_score:   Logistic Regression + MLP on [pred_score]   — score-as-model control
  B1:        Logistic Regression + MLP on [ev_sas]
  B2:        Logistic Regression + MLP on [ev_sas, ev_sdca]
  B3:        Logistic Regression + MLP on [ev_sas, ev_sdca, ev_aps, ev_pss]
  B4:        Logistic Regression + MLP on [pred_score, ev_sas, ev_sdca, ev_aps, ev_pss]

Same-family baselines (B_msp/B_margin/B_entropy) require unc_msp/unc_margin/
unc_entconf columns from Stage 3C. An image-grouped bootstrap reports B4's
AUPRC gain over each with a 95% CI — the comparison reviewers ask for.

Headline test: B4 (fused score+evidence gate) vs B0 (score threshold) and vs
B_score (score-only, same model class). B4 > B_score ⇒ KG evidence adds signal
above the detector score; B4 ≈ B_score ⇒ evidence is redundant given score.

Note on the old placeholder-heavy regime (not universal):
    In earlier runs with constant APS/PSS and synthetic tie-saturated probabilities,
    B3 often collapsed to B2, B2 stayed close to B1, and B1 stayed close to B0.
    This script now audits that regime explicitly instead of hard-coding it into
    the decision document.

Inputs:
  outputs/stage3c_audit/governance_features_train.csv
  outputs/stage3c_audit/governance_features_valid.csv
  outputs/stage3c_audit/governance_features_test.csv

  (Run audit_stage3c_governance_targets.py --split {train,valid,test} first.)

Outputs:
  outputs/stage3d_lgg/baseline_results.csv
  outputs/stage3d_lgg/lgg_train_log.csv
  outputs/stage3d_lgg/lgg_valid_metrics.json
  outputs/stage3d_lgg/lgg_test_metrics.json
    outputs/stage3d_lgg/lgg_models.pkl
  outputs/stage3d_lgg/stage3d_decision.md

Usage:
    python tools/train_stage3d_lgg.py \\
      --stage3c_dir outputs/stage3c_audit \\
      --output_dir outputs/stage3d_lgg

    python tools/train_stage3d_lgg.py \\
      --stage3c_dir outputs/stage3c_audit \\
      --output_dir outputs/stage3d_lgg \\
      --no_mlp           # skip MLP, LR only (faster)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure repo root is importable before optional torch/LGG imports.
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)


def _assert_not_outputs_mirror_script() -> None:
    script_norm = str(Path(__file__).resolve()).replace("\\", "/").lower()
    if "/outputs/tools/" in script_norm:
        raise RuntimeError(
            "[Stage3D] Refusing to run from outputs/tools mirror. "
            "Run the canonical root script under tools/."
        )


def _assert_root_module(module, module_label: str) -> None:
    module_name = str(getattr(module, "__name__", ""))
    module_file = str(Path(getattr(module, "__file__", "")).resolve())
    module_norm = module_file.replace("\\", "/").lower()
    if module_name.startswith("outputs.src.") or "/outputs/src/" in module_norm:
        raise RuntimeError(
            f"[Stage3D] Refusing mirrored module import for {module_label}: "
            f"name={module_name} file={module_file}. Use root src/."
        )


_assert_not_outputs_mirror_script()

import numpy as np

_TORCH_OK = False
_TORCH_IMPORT_ERR: Optional[str] = None
_LGG_MODULE = None
try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    try:
        from src.governance.gate import LGG, get_gumbel_tau

        _LGG_MODULE = sys.modules.get("src.governance.gate")
    except ModuleNotFoundError:
        from lgovnet.src.governance.gate import LGG, get_gumbel_tau

        _LGG_MODULE = sys.modules.get("lgovnet.src.governance.gate")

    if _LGG_MODULE is not None:
        _assert_root_module(_LGG_MODULE, "governance.gate")

    _TORCH_OK = True
except Exception as _torch_exc:  # pragma: no cover - defensive import guard
    _TORCH_IMPORT_ERR = str(_torch_exc)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── sklearn import guard ───────────────────────────────────────────────────────

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, f1_score
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False
    print(
        "[Stage3D] WARNING: scikit-learn not found. Install with: pip install scikit-learn"
    )
    print("[Stage3D] Only B0 (threshold sweep) will run without sklearn.")

# ── Constants ──────────────────────────────────────────────────────────────────

_RETAIN_CATS = frozenset({"TP_EXACT", "CLINICALLY_CLOSE_MISMATCH", "LOCALIZATION_WEAK"})
_SUPPRESS_CATS = frozenset({"BACKGROUND_FP", "LOW_CONF_CANDIDATE", "UNSAFE_MISMATCH"})
_UNSAFE_CAT = "UNSAFE_MISMATCH"

# Feature columns per baseline (from governance_features CSV)
_FEATURE_SETS: Dict[str, List[str]] = {
    "B0": ["pred_score"],
    "B_msp": ["unc_msp"],
    "B_margin": ["unc_margin"],
    "B_entropy": ["unc_entconf"],
    "B_score": ["pred_score"],
    "B1": ["ev_sas"],
    "B2": ["ev_sas", "ev_sdca"],
    "B3": ["ev_sas", "ev_sdca", "ev_aps", "ev_pss"],
    "B4": ["pred_score", "ev_sas", "ev_sdca", "ev_aps", "ev_pss"],
}

# Same-family selective-prediction baselines: each is a single confidence
# score swept for the best-F1 threshold on train (no model trained).
_THRESHOLD_BASELINES = ["B0", "B_msp", "B_margin", "B_entropy"]

_BASELINE_ORDER = [
    "B0",
    "B_msp",
    "B_margin",
    "B_entropy",
    "B_score",
    "B1",
    "B2",
    "B3",
    "B4",
]

_CANONICAL_STAGE3C_DIR = Path(_root) / "outputs" / "stage3c_audit"
_SUSPICIOUS_STAGE3C_MARKERS = (
    "/outputs/artefak/",
    "/outputs/src/",
    "/outputs/tools/",
    "/legacy/",
    "/historical/",
    "/stale/",
    "/stage3c_audit_pre",
)

_STAGE3D_REQUIRED_COLUMNS = sorted(
    set(
        [
            "bbox_id",
            "image_id",
            "split",
            "category",
            "label",
            "source_output_dir",
            "source_cache_dir",
            "source_annotations_dir",
            "anatomy_provider_mode",
            "anatomy_quality_mode",
            "pss_source_json",
            "compare_baseline_source",
        ]
        + [col for cols in _FEATURE_SETS.values() for col in cols]
    )
)
_STAGE3D_LEGACY_METADATA_COLUMNS = {
    "split",
    "source_output_dir",
    "source_cache_dir",
    "source_annotations_dir",
    "anatomy_provider_mode",
    "anatomy_quality_mode",
    "pss_source_json",
    "compare_baseline_source",
}
_STAGE3D_NUMERIC_REQUIRED_COLUMNS = sorted(
    set(["label"] + [col for cols in _FEATURE_SETS.values() for col in cols])
)


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


def _validate_split_isolation(records_by_split: Dict[str, List[dict]]) -> List[str]:
    errors: List[str] = []
    image_sets: Dict[str, set] = {}
    bbox_sets: Dict[str, set] = {}
    output_roots: Dict[str, str] = {}

    for split, records in records_by_split.items():
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

        cache_dirs = {
            str(r.get("source_cache_dir", "")).strip()
            for r in records
            if str(r.get("source_cache_dir", "")).strip()
        }
        for cache_dir in cache_dirs:
            tail = Path(cache_dir).name.lower()
            if tail and tail != split:
                errors.append(
                    f"Split {split} has source_cache_dir with mismatched tail '{Path(cache_dir).name}'"
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


# ── Data loading ───────────────────────────────────────────────────────────────


def _load_features_csv(path: Path) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _validate_and_load_stage3c_split(
    stage3c_dir: Path,
    split: str,
    features_path: Path,
    required_columns: List[str],
    allow_legacy_schema: bool = False,
) -> List[dict]:
    with open(features_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        records = list(reader)
        header = list(reader.fieldnames or [])

    missing = [col for col in required_columns if col not in header]
    if missing:
        missing_set = set(missing)
        if allow_legacy_schema and missing_set <= _STAGE3D_LEGACY_METADATA_COLUMNS:
            stage3c_root = str(stage3c_dir.resolve())
            print(
                "[Stage3D] WARN: legacy Stage3C schema detected for historical "
                f"baseline {features_path}; hydrating missing metadata columns: "
                f"{', '.join(missing)}"
            )
            for row in records:
                row.setdefault("split", split)
                row.setdefault("source_output_dir", stage3c_root)
                row.setdefault("source_cache_dir", "")
                row.setdefault("source_annotations_dir", "")
                row.setdefault("anatomy_provider_mode", "legacy_unknown")
                row.setdefault("anatomy_quality_mode", "legacy_unknown")
                row.setdefault("pss_source_json", "")
                row.setdefault("compare_baseline_source", stage3c_root)
        else:
            raise RuntimeError(
                f"{features_path} missing required columns: {', '.join(missing)}"
            )
    if not records:
        raise RuntimeError(f"{features_path} is empty")

    for col in _STAGE3D_NUMERIC_REQUIRED_COLUMNS:
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

    agg_total_raw = (audit.get("aggregation") or {}).get("total_boxes")
    if agg_total_raw is not None:
        try:
            agg_total = int(agg_total_raw)
        except (TypeError, ValueError):
            raise RuntimeError(
                f"Invalid aggregation.total_boxes in {audit_path}: {agg_total_raw}"
            )
        if agg_total != len(records):
            raise RuntimeError(
                f"Row-count provenance mismatch for split '{split}': "
                f"CSV={len(records)} vs audit aggregation.total_boxes={agg_total}"
            )

    return records


def _extract_arrays(
    records: List[dict],
    feature_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract (X, y, is_unsafe, is_tp_exact) from raw CSV records.

    Skips rows with missing or non-finite feature values.
    """
    X_rows, y_rows, unsafe_rows, tp_rows = [], [], [], []
    for r in records:
        try:
            x_row = [float(r[col]) for col in feature_cols]
        except (ValueError, KeyError):
            continue
        if any(not math.isfinite(v) for v in x_row):
            continue
        X_rows.append(x_row)
        y_rows.append(int(r.get("label", 0)))
        unsafe_rows.append(r.get("category") == _UNSAFE_CAT)
        tp_rows.append(r.get("category") == "TP_EXACT")

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.int32)
    is_unsafe = np.array(unsafe_rows, dtype=bool)
    is_tp = np.array(tp_rows, dtype=bool)
    return X, y, is_unsafe, is_tp


# ── Feature correlation audit ─────────────────────────────────────────────────


def _feature_correlations(records: List[dict]) -> dict:
    """Compute Pearson correlations between evidence features.

    Key check: r(SAS, SDCA). If ~1.0, SDCA is redundant given SAS.
    Expected from the current single-head detector bridge: SDCA = deterministic f(SAS, class) → high correlation.
    """
    cols = ["ev_sas", "ev_sdca", "ev_aps", "ev_pss", "pred_score"]
    data: Dict[str, list] = {c: [] for c in cols}
    for r in records:
        try:
            vals = {c: float(r[c]) for c in cols}
            if all(math.isfinite(v) for v in vals.values()):
                for c in cols:
                    data[c].append(vals[c])
        except (ValueError, KeyError):
            continue

    n = len(data["ev_sas"])
    if n < 2:
        return {}

    arrs = {c: np.array(v, dtype=np.float64) for c, v in data.items()}
    pairs = [
        ("ev_sas", "ev_sdca"),
        ("ev_sas", "pred_score"),
        ("ev_sdca", "pred_score"),
        ("ev_aps", "ev_sas"),
        ("ev_pss", "ev_sas"),
    ]
    result: dict = {"n_samples": n}
    for a, b in pairs:
        xa, xb = arrs[a], arrs[b]
        std_a, std_b = xa.std(), xb.std()
        if std_a < 1e-9 or std_b < 1e-9:
            r = None  # constant feature
        else:
            r = float(np.corrcoef(xa, xb)[0, 1])
        result[f"r({a},{b})"] = round(r, 4) if r is not None else None
    return result


def _feature_variation(records: List[dict]) -> dict:
    cols = ["pred_score", "ev_sas", "ev_sdca", "ev_aps", "ev_pss"]
    values: Dict[str, list] = {c: [] for c in cols}
    for r in records:
        for c in cols:
            try:
                v = float(r[c])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(v):
                values[c].append(v)

    summary: dict = {}
    for c, arr_values in values.items():
        if not arr_values:
            summary[c] = {
                "n": 0,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "is_constant": True,
            }
            continue
        arr = np.array(arr_values, dtype=np.float64)
        std = float(arr.std())
        summary[c] = {
            "n": int(arr.size),
            "mean": round(float(arr.mean()), 4),
            "std": round(std, 4),
            "min": round(float(arr.min()), 4),
            "max": round(float(arr.max()), 4),
            "is_constant": bool(std < 1e-9),
        }
    return summary


def _load_stage3c_audit(stage3c_dir: Path, split: str) -> Optional[dict]:
    path = stage3c_dir / f"governance_target_audit_{split}.json"
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _summarize_stage3c_signal(stage3c_dir: Path) -> dict:
    audit = _load_stage3c_audit(stage3c_dir, "train")
    if not audit:
        return {}

    tie_rates: List[float] = []
    for row in (audit.get("sdca_proxy_audit") or {}).values():
        try:
            tie_rate = float(row.get("tie_rate"))
        except (AttributeError, TypeError, ValueError):
            continue
        if math.isfinite(tie_rate):
            tie_rates.append(tie_rate)

    notes = [n for n in audit.get("gate_notes", []) if isinstance(n, str)]
    return {
        "max_tie_rate": round(max(tie_rates), 4) if tie_rates else None,
        "mean_tie_rate": (
            round(sum(tie_rates) / len(tie_rates), 4) if tie_rates else None
        ),
        "close_vs_unsafe_ordering_confirmed": any(
            "avg_confusion_cost(CLOSE)" in note and "avg_confusion_cost(UNSAFE)" in note
            for note in notes
        ),
    }


# ── Metrics ───────────────────────────────────────────────────────────────────


def _governance_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    is_unsafe: np.ndarray,
    is_tp: np.ndarray,
) -> dict:
    """Compute governance-specific metrics for one (baseline, split) result."""
    retain_mask = y_true == 1
    suppress_mask = y_true == 0

    n_retain = int(retain_mask.sum())
    n_suppress = int(suppress_mask.sum())
    n_unsafe = int(is_unsafe.sum())
    n_tp = int(is_tp.sum())

    tp_retained = int((retain_mask & (y_pred == 1)).sum())
    fp_retained = int((suppress_mask & (y_pred == 1)).sum())
    tp_suppressed = int((retain_mask & (y_pred == 0)).sum())
    fp_suppressed = int((suppress_mask & (y_pred == 0)).sum())
    unsafe_suppressed = int((is_unsafe & (y_pred == 0)).sum())
    tp_exact_retained = int((is_tp & (y_pred == 1)).sum())

    retained_total = int((y_pred == 1).sum())
    suppressed_total = int((y_pred == 0).sum())

    tp_retention = tp_retained / n_retain if n_retain > 0 else 0.0
    fp_suppression = fp_suppressed / n_suppress if n_suppress > 0 else 0.0
    precision_after = tp_retained / retained_total if retained_total > 0 else 0.0
    recall_loss = 1.0 - tp_retention
    tp_exact_retention = tp_exact_retained / n_tp if n_tp > 0 else 0.0
    unsafe_supp_rate = unsafe_suppressed / n_unsafe if n_unsafe > 0 else None

    if _SKLEARN_OK:
        auprc = float(average_precision_score(y_true, y_score))
    else:
        # Fallback: naive AUC via sorted thresholds
        order = np.argsort(y_score)[::-1]
        ys = y_true[order]
        tp_cumsum = np.cumsum(ys)
        fp_cumsum = np.cumsum(1 - ys)
        precision_curve = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-9)
        recall_curve = tp_cumsum / max(n_retain, 1)
        auprc = float(np.trapz(precision_curve[::-1], recall_curve[::-1]))

    return {
        "n_boxes": int(len(y_true)),
        "n_retain_label": n_retain,
        "n_suppress_label": n_suppress,
        "tp_retention": round(float(tp_retention), 4),
        "tp_exact_retention": round(float(tp_exact_retention), 4),
        "fp_suppression": round(float(fp_suppression), 4),
        "precision_after": round(float(precision_after), 4),
        "recall_loss": round(float(recall_loss), 4),
        "auprc": round(auprc, 4),
        "unsafe_total": n_unsafe,
        "unsafe_suppression_rate": (
            round(float(unsafe_supp_rate), 4) if unsafe_supp_rate is not None else None
        ),
        "retained_total": retained_total,
        "suppressed_total": suppressed_total,
    }


# ── B0: score threshold sweep ─────────────────────────────────────────────────


def _b0_threshold_sweep(X_train: np.ndarray, y_train: np.ndarray) -> float:
    """Find the score threshold maximizing F1 on training data."""
    scores = X_train[:, 0]
    best_thresh, best_f1 = 0.15, 0.0
    for t in np.linspace(0.05, 0.90, 171):
        y_pred = (scores >= float(t)).astype(np.int32)
        if _SKLEARN_OK:
            f1 = f1_score(y_train, y_pred, zero_division=0)
        else:
            tp = int(((y_pred == 1) & (y_train == 1)).sum())
            fp = int(((y_pred == 1) & (y_train == 0)).sum())
            fn = int(((y_pred == 0) & (y_train == 1)).sum())
            denom = 2 * tp + fp + fn
            f1 = (2 * tp / denom) if denom > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_thresh = f1, float(t)
    return round(best_thresh, 3)


# ── LR training ───────────────────────────────────────────────────────────────


def _train_lr(X_train: np.ndarray, y_train: np.ndarray) -> Tuple[object, object]:
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_train)
    model = LogisticRegression(
        max_iter=500, class_weight="balanced", random_state=42, solver="lbfgs"
    )
    model.fit(X_sc, y_train)
    return model, scaler


# ── MLP training ──────────────────────────────────────────────────────────────


def _train_mlp(
    X_train: np.ndarray, y_train: np.ndarray
) -> Tuple[object, object, List[dict]]:
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_train)
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
    model.fit(X_sc, y_train)
    train_log = [
        {"epoch": i + 1, "train_loss": round(float(v), 6)}
        for i, v in enumerate(getattr(model, "loss_curve_", []))
    ]
    return model, scaler, train_log


# ── Predict helpers ──────────────────────────────────────────────────────────


def _predict_lr_mlp(
    model: object, scaler: object, X: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    X_sc = scaler.transform(X)
    y_pred = model.predict(X_sc).astype(np.int32)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_sc)[:, 1].astype(np.float32)
    else:
        y_score = model.decision_function(X_sc).astype(np.float32)
    return y_pred, y_score


def _standardize_fit(X_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=0, keepdims=True).astype(np.float32)
    std = X_train.std(axis=0, keepdims=True).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def _standardize_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32)


def _lgg_targets_from_binary(y: np.ndarray) -> np.ndarray:
    """Map binary retain/suppress labels to 4-tier targets for LGG training.

    retain(1)   -> R1 index 0
    suppress(0) -> R4 index 3

    This keeps Stage 3D binary supervision compatible with the 4-tier LGG head.
    """
    return np.where(y.astype(np.int64) == 1, 0, 3).astype(np.int64)


def _augment_gate_input(
    X_evidence: np.ndarray,
    embedding_dim: int,
) -> np.ndarray:
    if embedding_dim <= 0:
        return X_evidence.astype(np.float32)
    zeros = np.zeros((X_evidence.shape[0], embedding_dim), dtype=np.float32)
    return np.concatenate([X_evidence.astype(np.float32), zeros], axis=1)


def _train_lgg_torch(
    X_train_evidence: np.ndarray,
    y_train: np.ndarray,
    *,
    evidence_dim: int,
    embedding_dim: int,
    hidden_dim: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
) -> Tuple["LGG", np.ndarray, np.ndarray, List[dict]]:
    if not _TORCH_OK:
        raise RuntimeError(
            f"PyTorch LGG backend unavailable: {_TORCH_IMPORT_ERR or 'unknown import error'}"
        )

    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train = _augment_gate_input(X_train_evidence, embedding_dim)
    mean, std = _standardize_fit(X_train)
    X_sc = _standardize_apply(X_train, mean, std)
    y_tier = _lgg_targets_from_binary(y_train)

    x_t = torch.from_numpy(X_sc)
    y_t = torch.from_numpy(y_tier)

    ds = TensorDataset(x_t, y_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    model = LGG(
        evidence_dim=evidence_dim,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )

    n_pos = int((y_tier == 0).sum())
    n_neg = int((y_tier == 3).sum())
    class_weights = torch.ones(4, dtype=torch.float32)
    if n_pos > 0 and n_neg > 0:
        tot = n_pos + n_neg
        class_weights[0] = float(tot) / float(2 * n_pos)
        class_weights[3] = float(tot) / float(2 * n_neg)

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_log: List[dict] = []
    anneal_steps = max(epochs * max(len(loader), 1), 1)
    global_step = 0

    model.train()
    for epoch in range(1, epochs + 1):
        losses: List[float] = []
        for xb, yb in loader:
            tau = get_gumbel_tau(
                step=global_step,
                tau_start=1.0,
                tau_end=0.2,
                tau_anneal_steps=anneal_steps,
            )
            out = model.gumbel_forward(xb, temperature=tau, hard=False)
            logits = out["tier_logits"]
            loss = F.cross_entropy(logits, yb, weight=class_weights)

            optim.zero_grad()
            loss.backward()
            optim.step()

            losses.append(float(loss.detach().cpu().item()))
            global_step += 1

        mean_loss = float(np.mean(losses)) if losses else 0.0
        train_log.append(
            {
                "epoch": epoch,
                "train_loss": round(mean_loss, 6),
            }
        )

    return model.eval(), mean, std, train_log


def _predict_lgg_torch(
    model: "LGG",
    X_evidence: np.ndarray,
    *,
    mean: np.ndarray,
    std: np.ndarray,
    embedding_dim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if X_evidence.size == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32)

    X_aug = _augment_gate_input(X_evidence, embedding_dim)
    X_sc = _standardize_apply(X_aug, mean, std)
    x_t = torch.from_numpy(X_sc)

    with torch.no_grad():
        out = model(x_t)
        probs = out["tier_probs"].detach().cpu().numpy()

    # Binary retain confidence: non-R4 mass.
    y_score = (1.0 - probs[:, 3]).astype(np.float32)
    y_pred = (y_score >= 0.5).astype(np.int32)
    return y_pred, y_score


# ── Bootstrap (image-grouped) ─────────────────────────────────────────────────


def _fmt_or_na(v, fmt: str = "+.4f") -> str:
    return format(v, fmt) if v is not None else "N/A"


def _bootstrap_auprc_diff(
    y: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    image_ids: List[str],
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """Image-grouped bootstrap of AUPRC(score_a) - AUPRC(score_b).

    Resamples whole images with replacement (boxes within an image are
    correlated), recomputes the AUPRC difference, returns mean, 95% CI, P(>0).
    """
    if not _SKLEARN_OK or len(y) != len(image_ids):
        return {"mean": None, "ci_lo": None, "ci_hi": None, "frac_pos": None, "n": 0}
    rng = np.random.default_rng(seed)
    img2idx: Dict[str, List[int]] = {}
    for i, im in enumerate(image_ids):
        img2idx.setdefault(im, []).append(i)
    uniq = list(img2idx.keys())
    diffs: List[float] = []
    for _ in range(n_boot):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        idx: List[int] = []
        for im in samp:
            idx.extend(img2idx[im])
        ia = np.array(idx, dtype=np.int64)
        yb = y[ia]
        if yb.sum() == 0 or (len(yb) - yb.sum()) == 0:
            continue
        d = float(
            average_precision_score(yb, score_a[ia])
            - average_precision_score(yb, score_b[ia])
        )
        diffs.append(d)
    if len(diffs) < 10:
        return {
            "mean": None,
            "ci_lo": None,
            "ci_hi": None,
            "frac_pos": None,
            "n": len(diffs),
        }
    arr = np.array(diffs)
    return {
        "mean": round(float(arr.mean()), 4),
        "ci_lo": round(float(np.percentile(arr, 2.5)), 4),
        "ci_hi": round(float(np.percentile(arr, 97.5)), 4),
        "frac_pos": round(float((arr > 0).mean()), 4),
        "n": len(diffs),
    }


# ── CSV / JSON writers ────────────────────────────────────────────────────────


_RESULTS_FIELDS = [
    "baseline",
    "model",
    "split",
    "tp_retention",
    "tp_exact_retention",
    "fp_suppression",
    "precision_after",
    "recall_loss",
    "auprc",
    "unsafe_suppression_rate",
    "retained_total",
    "suppressed_total",
    "n_boxes",
]


def _append_results_csv(rows: List[dict], path: Path) -> None:
    # Always overwrite — never append. An earlier append-mode left stale
    # rows from prior runs in baseline_results.csv (e.g. contradictory
    # negative-AUPRC threshold rows), which is unacceptable for a
    # journal-grade reproducibility package.
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_RESULTS_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _build_lgg_models_bundle(
    all_metrics: Dict[Tuple, dict],
    torch_lgg_artifact: Optional[dict] = None,
) -> Dict[str, dict]:
    """Build a portable Stage 5 warm-start bundle.

    Stage 3D trains sklearn baselines, not the full PyTorch LGG used by Stage 5,
    so there is no honest direct state_dict mapping. The bundle therefore stores
    the Stage 5 architecture config plus provenance/metrics under the keys
    expected by train_stage5_mode2.load_lgg_from_pkl().
    """
    config = {
        "evidence_dim": 4,
        "embedding_dim": 64,
        "hidden_dim": 64,
    }
    bundle: Dict[str, dict] = {}

    for bline in _BASELINE_ORDER:
        for mtype in ["threshold", "LR", "MLP"]:
            valid = all_metrics.get((bline, mtype, "valid"))
            test = all_metrics.get((bline, mtype, "test"))
            if not (valid or test):
                continue

            bundle[f"{bline}_{mtype}"] = {
                "weights": None,
                "config": dict(config),
                "source_baseline": bline,
                "source_model": mtype,
                "feature_columns": list(_FEATURE_SETS.get(bline, [])),
                "state_dict_available": False,
                "notes": (
                    "Stage 3D trains sklearn gates, so no direct PyTorch LGG "
                    "state_dict is available. Stage 5 should treat this as a "
                    "config/provenance warm-start only."
                ),
                "valid_metrics": valid,
                "test_metrics": test,
            }

    for fallback_key, baseline in [
        ("B2_MLP", "B2"),
        ("B3_MLP", "B3"),
        ("B1_MLP", "B1"),
    ]:
        bundle.setdefault(
            fallback_key,
            {
                "weights": None,
                "config": dict(config),
                "source_baseline": baseline,
                "source_model": "MLP",
                "feature_columns": list(_FEATURE_SETS.get(baseline, [])),
                "state_dict_available": False,
                "notes": (
                    "Placeholder entry created because no direct sklearn->LGG "
                    "weight mapping exists in Stage 3D."
                ),
                "valid_metrics": all_metrics.get((baseline, "MLP", "valid")),
                "test_metrics": all_metrics.get((baseline, "MLP", "test")),
            },
        )

    if torch_lgg_artifact is not None:
        bundle["B3_LGG_TORCH"] = {
            "state_dict": torch_lgg_artifact["state_dict"],
            "weights": None,
            "config": {
                "evidence_dim": int(torch_lgg_artifact["evidence_dim"]),
                "embedding_dim": int(torch_lgg_artifact["embedding_dim"]),
                "hidden_dim": int(torch_lgg_artifact["hidden_dim"]),
            },
            "source_baseline": "B3",
            "source_model": "LGG_TORCH",
            "feature_columns": list(_FEATURE_SETS.get("B3", [])),
            "state_dict_available": True,
            "notes": (
                "PyTorch LGG trained in Stage 3D with binary supervision mapped "
                "to R1/R4. Input uses evidence features with zero-padded e_pred "
                "dimensions for Stage 5 shape compatibility."
            ),
            "valid_metrics": all_metrics.get(("B3", "LGG_TORCH", "valid")),
            "test_metrics": all_metrics.get(("B3", "LGG_TORCH", "test")),
            "decision_threshold": float(torch_lgg_artifact["decision_threshold"]),
            "normalizer": {
                "mean": torch_lgg_artifact["normalizer_mean"],
                "std": torch_lgg_artifact["normalizer_std"],
            },
        }

    return bundle


def _write_lgg_models_pkl(
    all_metrics: Dict[Tuple, dict],
    path: Path,
    torch_lgg_artifact: Optional[dict] = None,
) -> None:
    with open(path, "wb") as f:
        pickle.dump(_build_lgg_models_bundle(all_metrics, torch_lgg_artifact), f)


def _write_decision_md(
    all_metrics: dict,
    correlations: dict,
    feature_variation: dict,
    stage3c_signal: dict,
    b0_thresh: float,
    model_types: List[str],
    path: Path,
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Stage 3D LGG Gate — Decision Document\n\n")

        f.write("## Headline (B4-First)\n\n")
        b0_test = all_metrics.get(("B0", "threshold", "test"))
        headline_rows: List[Tuple[str, dict, Optional[dict]]] = []
        for mtype in ["MLP", "LR"]:
            b4_m = all_metrics.get(("B4", mtype, "test"))
            bscore_m = all_metrics.get(("B_score", mtype, "test"))
            if b4_m is not None:
                headline_rows.append((mtype, b4_m, bscore_m))

        if b0_test and headline_rows:
            f.write(
                "| Model | B4 AUPRC | B0 AUPRC | Δ(B4-B0) | B_score AUPRC | Δ(B4-B_score) |\n"
            )
            f.write("|---|---:|---:|---:|---:|---:|\n")
            for mtype, b4_m, bscore_m in headline_rows:
                d_b0 = b4_m["auprc"] - b0_test["auprc"]
                if bscore_m is not None:
                    d_bs = b4_m["auprc"] - bscore_m["auprc"]
                    bscore_auprc = f"{bscore_m['auprc']:.4f}"
                    d_bs_txt = f"{d_bs:+.4f}"
                else:
                    bscore_auprc = "N/A"
                    d_bs_txt = "N/A"
                f.write(
                    f"| {mtype} | {b4_m['auprc']:.4f} | {b0_test['auprc']:.4f} "
                    f"| {d_b0:+.4f} | {bscore_auprc} | {d_bs_txt} |\n"
                )
            f.write(
                "\nInterpretasi utama: fokus klaim praktis pada B4 vs B_score (sinyal evidence di atas score-only), bukan pada B3.\n\n"
            )
        else:
            f.write(
                "Headline B4 tidak dapat dihitung lengkap (B0/B4 test metric tidak tersedia).\n\n"
            )

        aps_stats = feature_variation.get("ev_aps", {})
        pss_stats = feature_variation.get("ev_pss", {})
        aps_constant = bool(aps_stats.get("is_constant", True))
        pss_constant = bool(pss_stats.get("is_constant", True))
        max_tie_rate = stage3c_signal.get("max_tie_rate")
        close_vs_unsafe_ok = stage3c_signal.get("close_vs_unsafe_ordering_confirmed")

        f.write("## Feature correlation audit (train split)\n\n")
        f.write("| Pair | Pearson r | Interpretation |\n")
        f.write("|---|---:|---|\n")
        interp = {
            "r(ev_sas,ev_sdca)": (
                "HIGH → SDCA still looks largely score/SAS-driven in this run"
                if max_tie_rate is None or max_tie_rate < 0.9
                else "HIGH → SDCA is redundant given SAS under the old tie-saturated proxy regime"
            ),
            "r(ev_sas,pred_score)": "HIGH → SAS ≈ detection score (expected)",
            "r(ev_sdca,pred_score)": "HIGH → SDCA ≈ f(score) (confirms deterministic proxy)",
            "r(ev_aps,ev_sas)": (
                "~0 expected only if APS is constant; inspect APS run-regime stats below"
                if aps_constant
                else "Non-zero variation observed: APS is active in this run, but still heuristic"
            ),
            "r(ev_pss,ev_sas)": (
                "~0 expected only if PSS is constant; inspect PSS run-regime stats below"
                if pss_constant
                else "Non-zero variation observed: PSS is active in this run"
            ),
        }
        for k, v in correlations.items():
            if k == "n_samples":
                continue
            note = interp.get(k, "")
            f.write(
                f"| `{k}` | {v if v is not None else 'N/A (constant)'} | {note} |\n"
            )
        f.write(f"\n_n_samples={correlations.get('n_samples', '?')}_\n\n")

        f.write("## Run regime audit\n\n")
        for feature_name in ["ev_aps", "ev_pss"]:
            stats = feature_variation.get(feature_name, {})
            f.write(
                f"- `{feature_name}`: mean={stats.get('mean', 'N/A')}  std={stats.get('std', 'N/A')}  "
                f"range=[{stats.get('min', 'N/A')}, {stats.get('max', 'N/A')}]\n"
            )
        if max_tie_rate is not None:
            f.write(
                f"- train `sdca_proxy_audit` max tie_rate={max_tie_rate}"
                f" (mean={stage3c_signal.get('mean_tie_rate', 'N/A')})\n"
            )
        if not aps_constant or not pss_constant:
            f.write(
                "- This run is not in the old constant-APS/PSS placeholder regime."
                " Do not force `B3 == B2` in interpretation.\n"
            )
        else:
            f.write(
                "- This run still behaves like the old constant-APS/PSS regime."
                " `B3` is expected to track `B2`.\n"
            )
        if max_tie_rate is not None and max_tie_rate < 0.9:
            f.write(
                "- Class probabilities are not fully tie-saturated in the Stage 3C train audit,"
                " so the stale 'synthetic ensemble_probs everywhere' story does not apply verbatim.\n"
            )
        elif max_tie_rate is not None:
            f.write(
                "- Class probabilities still look tie-saturated in the Stage 3C train audit;"
                " SDCA may remain a mostly synthetic proxy.\n"
            )
        if close_vs_unsafe_ok:
            f.write(
                "- Stage 3C confirms clinical ordering between CLOSE and UNSAFE mismatches in train.\n"
            )
        f.write("\n")

        f.write(f"## B0 threshold\n\n")
        f.write(f"Best score threshold (max F1 on train): `{b0_thresh}`\n\n")

        f.write("## Results summary\n\n")
        for split in ["valid", "test"]:
            f.write(f"### {split}\n\n")
            f.write(
                "| Baseline | Model | TP ret. | TP_EXACT ret. | FP supp. "
                "| Prec. | Recall loss | AUPRC | Unsafe supp. |\n"
            )
            f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
            for bline in _BASELINE_ORDER:
                for mtype in model_types:
                    key = (bline, mtype, split)
                    m = all_metrics.get(key)
                    if m is None:
                        continue
                    f.write(
                        f"| {bline} | {mtype} "
                        f"| {m['tp_retention']:.3f} | {m['tp_exact_retention']:.3f} "
                        f"| {m['fp_suppression']:.3f} | {m['precision_after']:.3f} "
                        f"| {m['recall_loss']:.3f} | {m['auprc']:.3f} "
                        f"| {m['unsafe_suppression_rate'] or '-'} |\n"
                    )
            f.write("\n")

        f.write("## Secondary Diagnostics (B1-B3)\n\n")
        f.write(
            "Expected before training must now be conditioned on the observed run regime:\n"
        )
        if aps_constant and pss_constant:
            f.write(
                "- `B3 == B2` remains a valid expectation here because APS/PSS are effectively constant.\n"
            )
        else:
            f.write(
                "- `B3 == B2` is no longer a safe expectation because APS/PSS show non-zero variation.\n"
            )
        if max_tie_rate is not None and max_tie_rate >= 0.9:
            f.write(
                "- `B2 ≈ B1` is still plausible because the SDCA proxy remains close to the old tie-saturated regime.\n"
            )
        else:
            f.write(
                "- `B2 ≈ B1` must be judged empirically; Stage 3C no longer supports the old fully synthetic-proxy assumption.\n"
            )
        f.write(
            "- `B1 ≈ B0` is still the right null because SAS remains strongly score-linked.\n"
            "- LOCALIZATION_WEAK/BACKGROUND_FP remain intrinsically overlapping categories, so large gains are unlikely without richer detector diversity.\n\n"
        )

        f.write("## Limitations\n\n")
        limitation_lines = [
            "**SDCA remains detector-conditioned**: even when class probabilities vary, SDCA is still downstream of the current detector cache rather than an independently learned uncertainty source.",
            "**LOCALIZATION_WEAK separability is limited**: avg_SAS(LOCAL_WEAK)≈0.39 vs avg_SAS(BG_FP)≈0.33, so the distributions still overlap materially.",
            "**Annotation density shift persists**: train has ~3.7× more GT boxes/image than valid, so LGG calibration can move across splits.",
        ]
        if aps_constant:
            limitation_lines.append(
                "**APS is effectively constant in this run**: anatomy does not contribute extra separation, so `B3` should stay close to `B2`."
            )
        else:
            limitation_lines.append(
                "**APS is active but still heuristic**: the signal now varies, but it still comes from the current radial anatomy approximation rather than a validated OD/fovea provider."
            )
        if pss_constant:
            limitation_lines.append(
                "**PSS is effectively constant in this run**: perturbation stability is not adding discriminative power yet."
            )
        else:
            limitation_lines.append(
                "**PSS is active but still narrow**: the current signal comes from the implemented TTA loop only, not a richer perturbation family."
            )
        if max_tie_rate is not None and max_tie_rate >= 0.9:
            limitation_lines.append(
                "**Class probabilities still show heavy tie saturation**: the old synthetic-proxy caveat still applies to SDCA interpretation in this run."
            )
        for idx, line in enumerate(limitation_lines, start=1):
            f.write(f"{idx}. {line}\n")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 3D LGG gate: train and evaluate B0–B3 baselines."
    )
    parser.add_argument(
        "--stage3c_dir",
        default="outputs/stage3c_audit",
        help="Directory containing governance_features_*.csv files from Stage 3C.",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/stage3d_lgg",
        help="Output directory (default: outputs/stage3d_lgg).",
    )
    parser.add_argument(
        "--no_mlp",
        action="store_true",
        help="Skip MLP training (faster, LR + B0 only).",
    )
    parser.add_argument(
        "--enable_torch_lgg",
        action="store_true",
        help=(
            "Compatibility flag (no longer required). PyTorch LGG backend is "
            "enabled by default and exports state_dict into lgg_models.pkl "
            "(key: B3_LGG_TORCH)."
        ),
    )
    parser.add_argument(
        "--disable_torch_lgg",
        action="store_true",
        help=(
            "Disable PyTorch LGG backend and run legacy sklearn-only Stage 3D "
            "baselines. Use only for controlled proxy comparisons."
        ),
    )
    parser.add_argument(
        "--torch_lgg_epochs",
        type=int,
        default=40,
        help="Epochs for PyTorch LGG backend (default: 40).",
    )
    parser.add_argument(
        "--torch_lgg_batch_size",
        type=int,
        default=1024,
        help="Batch size for PyTorch LGG backend (default: 1024).",
    )
    parser.add_argument(
        "--torch_lgg_lr",
        type=float,
        default=1e-3,
        help="Learning rate for PyTorch LGG backend (default: 1e-3).",
    )
    parser.add_argument(
        "--torch_lgg_weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay for PyTorch LGG backend (default: 1e-4).",
    )
    parser.add_argument(
        "--torch_lgg_hidden_dim",
        type=int,
        default=64,
        help="Hidden dim for PyTorch LGG backend (default: 64).",
    )
    parser.add_argument(
        "--torch_lgg_dropout",
        type=float,
        default=0.1,
        help="Dropout for PyTorch LGG backend (default: 0.1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for PyTorch LGG backend (default: 42).",
    )
    parser.add_argument(
        "--allow_historical_inputs",
        action="store_true",
        help="Allow suspicious historical/artefact stage3c_dir paths (default: fail-fast).",
    )
    args = parser.parse_args()
    if args.enable_torch_lgg and args.disable_torch_lgg:
        print(
            "[Stage3D] ERROR: --enable_torch_lgg and --disable_torch_lgg are mutually exclusive."
        )
        sys.exit(2)

    torch_lgg_enabled = not args.disable_torch_lgg

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stage3c_dir = Path(args.stage3c_dir)

    path_warnings: List[str] = []
    marker = _detect_suspicious_path(stage3c_dir)
    if marker is not None:
        msg = f"stage3c_dir looks historical/suspicious ({marker}): {stage3c_dir}"
        if args.allow_historical_inputs:
            print(f"[Stage3D] WARN: {msg} (allowed by --allow_historical_inputs)")
            path_warnings.append(msg)
        else:
            print(
                f"[Stage3D] ERROR: {msg} " "(set --allow_historical_inputs to override)"
            )
            sys.exit(2)
    stage3c_norm = _norm_path(stage3c_dir)
    if (
        stage3c_dir.resolve() != _CANONICAL_STAGE3C_DIR.resolve()
        and "/outputs/point123_" not in stage3c_norm
    ):
        msg = (
            f"Non-canonical stage3c_dir detected: {stage3c_dir}. "
            f"Canonical default is {_CANONICAL_STAGE3C_DIR}."
        )
        print(f"[Stage3D] WARN: {msg}")
        path_warnings.append(msg)
    elif stage3c_dir.resolve() != _CANONICAL_STAGE3C_DIR.resolve():
        print(f"[Stage3D] INFO: stage3c_dir is a Point123 run artifact root: {stage3c_dir}")

    print(f"[Stage3D] === Phase 19N Stage 3D LGG Gate ===")
    print(f"[Stage3D] stage3c_dir={stage3c_dir}  output_dir={out_dir}")
    if not _SKLEARN_OK:
        print("[Stage3D] scikit-learn unavailable — only B0 threshold sweep will run.")
    if torch_lgg_enabled:
        if not _TORCH_OK:
            print(
                "[Stage3D] ERROR: PyTorch LGG backend is REQUIRED by default but unavailable: "
                f"{_TORCH_IMPORT_ERR or 'unknown import error'}"
            )
            print(
                "[Stage3D]   Fix environment or run with --disable_torch_lgg for proxy-only fallback."
            )
            sys.exit(2)
        print(
            "[Stage3D] PyTorch LGG backend ENABLED (default mainline) "
            f"(epochs={args.torch_lgg_epochs}, lr={args.torch_lgg_lr})"
        )
    else:
        print(
            "[Stage3D] PyTorch LGG backend DISABLED by --disable_torch_lgg (proxy-only run)."
        )

    # ── Load CSV files ─────────────────────────────────────────────────────────
    features_files = {
        split: stage3c_dir / f"governance_features_{split}.csv"
        for split in ["train", "valid", "test"]
    }
    for split, fpath in features_files.items():
        if not fpath.exists():
            print(f"[Stage3D] ERROR: {fpath} not found.")
            print(
                f"[Stage3D]   Re-run: python tools/audit_stage3c_governance_targets.py "
                f"--split {split} ..."
            )
            sys.exit(1)

    print(f"[Stage3D] Loading feature CSVs + schema/provenance validation...")
    records: Dict[str, List[dict]] = {}
    for split, fpath in features_files.items():
        try:
            records[split] = _validate_and_load_stage3c_split(
                stage3c_dir=stage3c_dir,
                split=split,
                features_path=fpath,
                required_columns=_STAGE3D_REQUIRED_COLUMNS,
                allow_legacy_schema=bool(args.allow_historical_inputs),
            )
        except RuntimeError as exc:
            print(f"[Stage3D] ERROR: {exc}")
            sys.exit(2)

    split_isolation_errors = _validate_split_isolation(records)
    if split_isolation_errors:
        print("[Stage3D] ERROR: split-isolation validation failed:")
        for err in split_isolation_errors:
            print(f"[Stage3D]   - {err}")
        sys.exit(2)
    print("[Stage3D] Split isolation check: PASS")
    for split, recs in records.items():
        n_retain = sum(1 for r in recs if r.get("label") == "1")
        n_suppress = sum(1 for r in recs if r.get("label") == "0")
        print(
            f"[Stage3D]   {split}: {len(recs)} boxes  "
            f"RETAIN={n_retain} ({100*n_retain/max(len(recs),1):.1f}%)  "
            f"SUPPRESS={n_suppress} ({100*n_suppress/max(len(recs),1):.1f}%)"
        )

    # ── Feature correlation audit (train) ──────────────────────────────────────
    print(f"\n[Stage3D] Feature correlation audit (train):")
    correlations = _feature_correlations(records["train"])
    feature_variation = _feature_variation(records["train"])
    stage3c_signal = _summarize_stage3c_signal(stage3c_dir)
    run_provenance = {
        "repo_root": str(Path(_root).resolve()),
        "script_path": str(Path(__file__).resolve()),
        "stage3c_dir": str(stage3c_dir.resolve()),
        "output_dir": str(out_dir.resolve()),
        "feature_csv_by_split": {
            split: str(path.resolve()) for split, path in features_files.items()
        },
        "compare_baseline_source": str(stage3c_dir.resolve()),
        "allow_historical_inputs": bool(args.allow_historical_inputs),
        "path_warnings": path_warnings,
        "split_isolation": "PASS",
    }
    for k, v in correlations.items():
        if k == "n_samples":
            continue
        tag = ""
        if v is not None and abs(v) > 0.8:
            tag = " ← HIGH (redundant feature)"
        if v is None:
            tag = " ← CONSTANT (zero variance)"
        print(f"[Stage3D]   {k} = {v}{tag}")
    for feature_name in ["ev_aps", "ev_pss"]:
        stats = feature_variation.get(feature_name, {})
        print(
            f"[Stage3D]   {feature_name}: std={stats.get('std')}  "
            f"range=[{stats.get('min')}, {stats.get('max')}]"
        )
    if stage3c_signal.get("max_tie_rate") is not None:
        print(
            f"[Stage3D]   train sdca_proxy tie_rate max={stage3c_signal.get('max_tie_rate')}  "
            f"mean={stage3c_signal.get('mean_tie_rate')}"
        )

    # ── Prepare data arrays ────────────────────────────────────────────────────
    # Store arrays per baseline × split
    data: Dict[str, Dict[str, Tuple]] = {}
    for bline, feat_cols in _FEATURE_SETS.items():
        data[bline] = {}
        for split in ["train", "valid", "test"]:
            X, y, is_unsafe, is_tp = _extract_arrays(records[split], feat_cols)
            data[bline][split] = (X, y, is_unsafe, is_tp)
            print(
                f"[Stage3D]   {bline} {split}: X={X.shape}  "
                f"n_finite={len(y)}  n_nan_dropped={len(records[split])-len(y)}"
            )

    # ── Train and evaluate ────────────────────────────────────────────────────
    all_metrics: Dict[Tuple, dict] = {}
    train_logs: List[dict] = []
    results_rows: List[dict] = []
    model_types: List[str] = ["threshold", "LR", "MLP"]
    if torch_lgg_enabled:
        model_types.append("LGG_TORCH")
    torch_lgg_artifact: Optional[dict] = None

    # --- Threshold baselines: B0 (score) + same-family selective-prediction --
    #     baselines (max-softmax, margin, predictive entropy). Each is a single
    #     confidence score swept for the best-F1 threshold on train.
    b0_thresh = 0.15
    test_threshold_scores: Dict[str, np.ndarray] = {}
    for bline in _THRESHOLD_BASELINES:
        X_trb, y_trb, _, _ = data[bline]["train"]
        thr = _b0_threshold_sweep(X_trb, y_trb)
        if bline == "B0":
            b0_thresh = thr
        print(
            f"\n[Stage3D] {bline}: {_FEATURE_SETS[bline][0]} threshold sweep "
            f"→ best threshold = {thr}"
        )
        for split in ["valid", "test"]:
            X_sp, y_sp, unsafe_sp, tp_sp = data[bline][split]
            scores = X_sp[:, 0].astype(np.float32)
            y_pred = (scores >= thr).astype(np.int32)
            m = _governance_metrics(y_sp, y_pred, scores, unsafe_sp, tp_sp)
            m.update({"baseline": bline, "model": "threshold", "split": split})
            all_metrics[(bline, "threshold", split)] = m
            results_rows.append(m)
            if split == "test":
                test_threshold_scores[bline] = scores
            print(
                f"[Stage3D]   {bline} {split}: TP_ret={m['tp_retention']:.3f}  "
                f"FP_supp={m['fp_suppression']:.3f}  AUPRC={m['auprc']:.4f}"
            )

    # --- Learned gates B_score/B1/B2/B3/B4: LR and optionally MLP -----------
    b4_test_score = None  # B4 MLP test scores, kept for the bootstrap below
    if _SKLEARN_OK:
        for bline in ["B_score", "B1", "B2", "B3", "B4"]:
            feat_cols = _FEATURE_SETS[bline]
            print(f"\n[Stage3D] {bline}: features={feat_cols}")
            X_tr, y_tr, _, _ = data[bline]["train"]

            # Logistic Regression
            print(f"[Stage3D]   Training LR...")
            lr_model, lr_scaler = _train_lr(X_tr, y_tr)

            # Report LR coefficients
            if hasattr(lr_model, "coef_"):
                coefs = lr_model.coef_[0]
                for feat, coef in zip(feat_cols, coefs):
                    print(f"[Stage3D]     LR coef [{feat}] = {coef:.4f}")

            for split in ["valid", "test"]:
                X_sp, y_sp, unsafe_sp, tp_sp = data[bline][split]
                y_pred, y_score = _predict_lr_mlp(lr_model, lr_scaler, X_sp)
                m = _governance_metrics(y_sp, y_pred, y_score, unsafe_sp, tp_sp)
                m.update({"baseline": bline, "model": "LR", "split": split})
                all_metrics[(bline, "LR", split)] = m
                results_rows.append(m)
                print(
                    f"[Stage3D]   {bline} LR {split}: TP_ret={m['tp_retention']:.3f}  "
                    f"FP_supp={m['fp_suppression']:.3f}  AUPRC={m['auprc']:.4f}"
                )

            # MLP
            if not args.no_mlp:
                print(f"[Stage3D]   Training MLP (32→16)...")
                mlp_model, mlp_scaler, mlp_log = _train_mlp(X_tr, y_tr)
                n_iters = getattr(mlp_model, "n_iter_", "?")
                print(f"[Stage3D]     MLP converged in {n_iters} epochs")
                for log_row in mlp_log:
                    log_row["baseline"] = bline
                    train_logs.append(log_row)

                for split in ["valid", "test"]:
                    X_sp, y_sp, unsafe_sp, tp_sp = data[bline][split]
                    y_pred, y_score = _predict_lr_mlp(mlp_model, mlp_scaler, X_sp)
                    if bline == "B4" and split == "test":
                        b4_test_score = y_score
                    m = _governance_metrics(y_sp, y_pred, y_score, unsafe_sp, tp_sp)
                    m.update({"baseline": bline, "model": "MLP", "split": split})
                    all_metrics[(bline, "MLP", split)] = m
                    results_rows.append(m)
                    print(
                        f"[Stage3D]   {bline} MLP {split}: TP_ret={m['tp_retention']:.3f}  "
                        f"FP_supp={m['fp_suppression']:.3f}  AUPRC={m['auprc']:.4f}"
                    )

    # --- Real PyTorch LGG backend (Phase B path): B3 evidence-only gate -----
    if torch_lgg_enabled:
        print("\n[Stage3D] B3 LGG_TORCH: training real PyTorch gate backend...")
        X_tr, y_tr, _, _ = data["B3"]["train"]
        if X_tr.size == 0:
            print("[Stage3D] WARNING: B3 train set is empty; skipping LGG_TORCH.")
        else:
            evidence_dim = X_tr.shape[1]
            embedding_dim = (
                64  # keep Stage5 compatibility: gate_input = [ev(4) || e_pred(64)]
            )
            lgg_model, lgg_mean, lgg_std, lgg_log = _train_lgg_torch(
                X_tr,
                y_tr,
                evidence_dim=evidence_dim,
                embedding_dim=embedding_dim,
                hidden_dim=args.torch_lgg_hidden_dim,
                dropout=args.torch_lgg_dropout,
                epochs=args.torch_lgg_epochs,
                batch_size=args.torch_lgg_batch_size,
                lr=args.torch_lgg_lr,
                weight_decay=args.torch_lgg_weight_decay,
                seed=args.seed,
            )
            for log_row in lgg_log:
                log_row["baseline"] = "B3_LGG_TORCH"
                train_logs.append(log_row)

            # Threshold calibration on train retain-confidence score.
            _, score_tr = _predict_lgg_torch(
                lgg_model,
                X_tr,
                mean=lgg_mean,
                std=lgg_std,
                embedding_dim=embedding_dim,
            )
            lgg_thresh = _b0_threshold_sweep(score_tr.reshape(-1, 1), y_tr)
            print(
                f"[Stage3D]   LGG_TORCH calibrated retain threshold={lgg_thresh} "
                "(on train)"
            )

            for split in ["valid", "test"]:
                X_sp, y_sp, unsafe_sp, tp_sp = data["B3"][split]
                _, y_score = _predict_lgg_torch(
                    lgg_model,
                    X_sp,
                    mean=lgg_mean,
                    std=lgg_std,
                    embedding_dim=embedding_dim,
                )
                y_pred = (y_score >= lgg_thresh).astype(np.int32)
                m = _governance_metrics(y_sp, y_pred, y_score, unsafe_sp, tp_sp)
                m.update({"baseline": "B3", "model": "LGG_TORCH", "split": split})
                all_metrics[("B3", "LGG_TORCH", split)] = m
                results_rows.append(m)
                print(
                    f"[Stage3D]   B3 LGG_TORCH {split}: TP_ret={m['tp_retention']:.3f}  "
                    f"FP_supp={m['fp_suppression']:.3f}  AUPRC={m['auprc']:.4f}"
                )

            torch_lgg_artifact = {
                "state_dict": {
                    k: v.detach().cpu() for k, v in lgg_model.state_dict().items()
                },
                "evidence_dim": int(evidence_dim),
                "embedding_dim": int(embedding_dim),
                "hidden_dim": int(args.torch_lgg_hidden_dim),
                "decision_threshold": float(lgg_thresh),
                "normalizer_mean": lgg_mean.squeeze(0).tolist(),
                "normalizer_std": lgg_std.squeeze(0).tolist(),
            }

    if torch_lgg_enabled and torch_lgg_artifact is None:
        print(
            "[Stage3D] ERROR: PyTorch LGG mode is enabled but no B3_LGG_TORCH artifact was produced."
        )
        print(
            "[Stage3D]   Check Stage 3C features and ensure B3 train split is non-empty."
        )
        sys.exit(2)

    # ── Print comparison table ────────────────────────────────────────────────
    print(f"\n[Stage3D] ════════ BASELINE COMPARISON (valid) ════════")
    print(
        f"  {'Baseline':<5} {'Model':<10} {'TP_ret':>8} {'FP_supp':>8} "
        f"{'Prec':>7} {'RecLoss':>8} {'AUPRC':>7}"
    )
    for bline in _BASELINE_ORDER:
        for mtype in model_types:
            m = all_metrics.get((bline, mtype, "valid"))
            if m is None:
                continue
            print(
                f"  {bline:<5} {mtype:<10} "
                f"{m['tp_retention']:>8.3f} {m['fp_suppression']:>8.3f} "
                f"{m['precision_after']:>7.3f} {m['recall_loss']:>8.3f} "
                f"{m['auprc']:>7.4f}"
            )

    print(f"\n[Stage3D] ════════ BASELINE COMPARISON (test) ═════════")
    print(
        f"  {'Baseline':<5} {'Model':<10} {'TP_ret':>8} {'FP_supp':>8} "
        f"{'Prec':>7} {'RecLoss':>8} {'AUPRC':>7}"
    )
    for bline in _BASELINE_ORDER:
        for mtype in model_types:
            m = all_metrics.get((bline, mtype, "test"))
            if m is None:
                continue
            print(
                f"  {bline:<5} {mtype:<10} "
                f"{m['tp_retention']:>8.3f} {m['fp_suppression']:>8.3f} "
                f"{m['precision_after']:>7.3f} {m['recall_loss']:>8.3f} "
                f"{m['auprc']:>7.4f}"
            )

    # ── Headline: does the fused score+evidence gate beat score alone? ───────
    print(f"\n[Stage3D] ════ HEADLINE: B4 (score+evidence) vs score baselines ════")
    for hsplit in ["valid", "test"]:
        m_b0_h = all_metrics.get(("B0", "threshold", hsplit))
        for mtype in ["LR", "MLP"]:
            m_b4_h = all_metrics.get(("B4", mtype, hsplit))
            m_bs_h = all_metrics.get(("B_score", mtype, hsplit))
            if not (m_b0_h and m_b4_h):
                continue
            line = (
                f"  [{hsplit}] B4 {mtype}: AUPRC={m_b4_h['auprc']:.4f}  "
                f"FP_supp={m_b4_h['fp_suppression']:.3f}  "
                f"TP_ret={m_b4_h['tp_retention']:.3f}  "
                f"| B0 AUPRC={m_b0_h['auprc']:.4f} (Δ={m_b4_h['auprc']-m_b0_h['auprc']:+.4f})"
            )
            if m_bs_h:
                line += (
                    f"  B_score AUPRC={m_bs_h['auprc']:.4f} "
                    f"(Δ={m_b4_h['auprc']-m_bs_h['auprc']:+.4f})"
                )
            print(line)
    print(
        "[Stage3D]   Read: B4 vs B_score Δ>0 ⇒ KG evidence adds signal above "
        "detector score; Δ≈0 ⇒ evidence redundant given score."
    )

    # ── Bootstrap: B4 (MLP) vs same-family baselines (image-grouped AUPRC) ────
    print(
        "\n[Stage3D] ════ BOOTSTRAP: B4 (MLP) AUPRC vs same-family baselines (test) ════"
    )
    n_test = len(records["test"])
    aligned = (
        _SKLEARN_OK
        and b4_test_score is not None
        and len(b4_test_score) == n_test
        and all(
            len(test_threshold_scores.get(b, [])) == n_test
            for b in _THRESHOLD_BASELINES
        )
    )
    if aligned:
        y_te = data["B4"]["test"][1]
        img_te = [str(r.get("image_id", "")) for r in records["test"]]
        for bline in _THRESHOLD_BASELINES:
            b = _bootstrap_auprc_diff(
                y_te,
                b4_test_score,
                test_threshold_scores[bline],
                img_te,
                n_boot=1000,
            )
            sig = "YES" if (b["ci_lo"] is not None and b["ci_lo"] > 0) else "no"
            print(
                f"[Stage3D]   B4 - {bline:<10}: ΔAUPRC={_fmt_or_na(b['mean'])}  "
                f"CI=[{_fmt_or_na(b['ci_lo'])}, {_fmt_or_na(b['ci_hi'])}]  "
                f"P(>0)={_fmt_or_na(b['frac_pos'], '.4f')}  sig={sig}"
            )
        print(
            "[Stage3D]   sig=YES ⇒ B4 beats that same-family baseline with the "
            "95% CI excluding 0."
        )
    else:
        print(
            "[Stage3D]   (skipped — sklearn missing, B4 not trained, or row counts "
            "differ across baselines; check Stage 3C unc_* columns)"
        )

    # ── Check pre-registered predictions ─────────────────────────────────────
    print(f"\n[Stage3D] Pre-registered prediction checks (valid):")
    for mtype in ["LR", "MLP"]:
        m_b2 = all_metrics.get(("B2", mtype, "valid"))
        m_b3 = all_metrics.get(("B3", mtype, "valid"))
        if m_b2 and m_b3:
            delta_b3_b2 = abs(m_b3["auprc"] - m_b2["auprc"])
            status = "CONFIRMED" if delta_b3_b2 < 0.005 else "UNEXPECTED"
            print(
                f"  {mtype}: B3 vs B2 AUPRC diff = {delta_b3_b2:.4f} "
                f"[expected ≈0, {status}]"
            )
        m_b1 = all_metrics.get(("B1", mtype, "valid"))
        if m_b1 and m_b2:
            delta_b2_b1 = abs(m_b2["auprc"] - m_b1["auprc"])
            status = "CONFIRMED" if delta_b2_b1 < 0.01 else "UNEXPECTED"
            print(
                f"  {mtype}: B2 vs B1 AUPRC diff = {delta_b2_b1:.4f} "
                f"[expected ≈0 (SDCA redundant), {status}]"
            )
        m_b0 = all_metrics.get(("B0", "threshold", "valid"))
        if m_b0 and m_b1:
            delta_b1_b0 = m_b1["auprc"] - m_b0["auprc"]
            print(
                f"  {mtype}: B1 vs B0 AUPRC diff = {delta_b1_b0:+.4f} "
                f"[gain of LR/MLP over score threshold]"
            )

    # ── Save outputs ──────────────────────────────────────────────────────────
    results_path = out_dir / "baseline_results.csv"
    _append_results_csv(results_rows, results_path)
    print(f"\n[Stage3D] baseline_results.csv: {results_path}")

    if train_logs:
        log_path = out_dir / "lgg_train_log.csv"
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["baseline", "epoch", "train_loss"])
            writer.writeheader()
            writer.writerows(train_logs)
        print(f"[Stage3D] lgg_train_log.csv: {log_path}")

    for split, json_fname in [
        ("valid", "lgg_valid_metrics.json"),
        ("test", "lgg_test_metrics.json"),
    ]:
        split_metrics = {
            f"{bline}_{mtype}": all_metrics[(bline, mtype, split)]
            for bline in _BASELINE_ORDER
            for mtype in model_types
            if (bline, mtype, split) in all_metrics
        }
        jpath = out_dir / json_fname
        with open(jpath, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "split": split,
                    "provenance": run_provenance,
                    "feature_correlations_train": correlations,
                    "feature_variation_train": feature_variation,
                    "stage3c_signal_train": stage3c_signal,
                    "b0_threshold": b0_thresh,
                    "baselines": split_metrics,
                },
                f,
                indent=2,
            )
        print(f"[Stage3D] {json_fname}: {jpath}")

    models_path = out_dir / "lgg_models.pkl"
    _write_lgg_models_pkl(all_metrics, models_path, torch_lgg_artifact)
    print(f"[Stage3D] lgg_models.pkl: {models_path}")

    md_path = out_dir / "stage3d_decision.md"
    _write_decision_md(
        all_metrics,
        correlations,
        feature_variation,
        stage3c_signal,
        b0_thresh,
        model_types,
        md_path,
    )
    print(f"[Stage3D] stage3d_decision.md: {md_path}")

    prov_path = out_dir / "stage3d_run_provenance.json"
    with open(prov_path, "w", encoding="utf-8") as f:
        json.dump(run_provenance, f, indent=2)
    print(f"[Stage3D] stage3d_run_provenance.json: {prov_path}")

    print(f"\n[Stage3D] Stage 3D complete.")
    closing_lines = [
        "[Stage3D] If B0 ≈ B1 ≈ B2 ≈ B3 in this run:",
        "  → governance signal is still dominated by score-linked evidence rather than richer detector diversity.",
        "  → Next: compare B4 against the deep-ensemble bridge rather than over-interpreting single-cache gains.",
    ]
    if feature_variation.get("ev_aps", {}).get("is_constant", True):
        closing_lines.append(
            "  → APS is still effectively constant here; anatomy remains a follow-up bottleneck."
        )
    if feature_variation.get("ev_pss", {}).get("is_constant", True):
        closing_lines.append(
            "  → PSS is still effectively constant here; perturbation stability remains a follow-up bottleneck."
        )
    if (
        stage3c_signal.get("max_tie_rate") is not None
        and stage3c_signal.get("max_tie_rate") >= 0.9
    ):
        closing_lines.append(
            "  → SDCA still sits near the old tie-saturated proxy regime, so B2 should be interpreted conservatively."
        )
    print("\n".join(closing_lines))


if __name__ == "__main__":
    main()
