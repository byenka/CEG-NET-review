#!/usr/bin/env python
"""
audit_stage3c_governance_targets.py
Phase 19N — Stage 3C: Governance Target Audit

Before training the Learned Governance Gate (LGG), this audit classifies
every detector candidate box into one of six governance target categories using
GT XML annotations, IoU matching, detector score, and clinical-prior SDM.

Goal: verify that governance targets are well-distributed and clinically
meaningful, ensuring LGG learns from an interpretable signal, not a
collapsed or ambiguous label set.

Six target categories:
  TP_EXACT:                  IoU >= 0.5, predicted class == GT class
  CLINICALLY_CLOSE_MISMATCH: IoU >= 0.5, class mismatch, SDM(pred,GT) < 0.3
                              (only MA<->HE: zero ICDR grade cost)
  UNSAFE_MISMATCH:           IoU >= 0.5, class mismatch, SDM(pred,GT) >= 0.3
  LOCALIZATION_WEAK:         0.1 <= IoU < 0.5
  BACKGROUND_FP:             IoU < 0.1, detector score >= 0.15
  LOW_CONF_CANDIDATE:        IoU < 0.1, detector score < 0.15

Governance semantics for Stage 3D LGG:
  RETAIN:   TP_EXACT, CLINICALLY_CLOSE_MISMATCH, LOCALIZATION_WEAK
  SUPPRESS: UNSAFE_MISMATCH, BACKGROUND_FP, LOW_CONF_CANDIDATE

Gate criteria (Stage 3C):
  1. All candidates assigned exactly one category (no unassigned)
  2. No NaN in IoU or evidence signals
  3. TP_EXACT and BACKGROUND_FP both exist (non-zero count)
  4. No single category > 90% of all boxes (no distribution collapse)
  5. avg semantic_confusion_cost(CLINICALLY_CLOSE) <
     avg semantic_confusion_cost(UNSAFE) — if both categories exist
  6. SDCA ordering by class: mean SDCA(MA) < mean SDCA(Hard_EX) < mean SDCA(Soft_EX)

Implementation note (2026-05-24):
    Mainline supports canonical multi-source SAS by attaching
    head_predictions from primary+secondary detector caches via IoU match.
    Use --secondary_cache_dir for non-proxy SAS features.

Outputs:
  outputs/stage3c_audit/governance_target_audit_{split}.json
  outputs/stage3c_audit/governance_target_distribution.csv
  outputs/stage3c_audit/stage3c_decision.md

The per-split JSON is the canonical artifact. Aggregate CSV/Markdown summaries are
regenerated from the latest JSON files in the output directory on every run, so
rerunning a split replaces its old section instead of appending mixed PASS/FAIL
history.

Usage:
    python tools/audit_stage3c_governance_targets.py \\
            --annotations_dir data/oiaddr/xml/valid \
            --cache_dir outputs/detector_cache/primary/oiaddr/valid \
      --kg_dir outputs/stage3_audit \\
      --split valid \\
      --output_dir outputs/stage3c_audit

    python tools/audit_stage3c_governance_targets.py \\
            --annotations_dir data/oiaddr/xml/test \
            --cache_dir outputs/detector_cache/primary/oiaddr/test \
      --kg_dir outputs/stage3_audit \\
      --split test \\
      --output_dir outputs/stage3c_audit
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)


def _assert_not_outputs_mirror_script() -> None:
    script_norm = str(Path(__file__).resolve()).replace("\\", "/").lower()
    if "/outputs/tools/" in script_norm:
        raise RuntimeError(
            "[Stage3C] Refusing to run from outputs/tools mirror. "
            "Run the canonical root script under tools/."
        )


def _assert_root_module(module, module_label: str) -> None:
    module_name = str(getattr(module, "__name__", ""))
    module_file = str(Path(getattr(module, "__file__", "")).resolve())
    module_norm = module_file.replace("\\", "/").lower()
    if module_name.startswith("outputs.src.") or "/outputs/src/" in module_norm:
        raise RuntimeError(
            f"[Stage3C] Refusing mirrored module import for {module_label}: "
            f"name={module_name} file={module_file}. Use root src/."
        )


_assert_not_outputs_mirror_script()


def _import_first(module_names: List[str]):
    last_error: Optional[ModuleNotFoundError] = None
    for module_name in module_names:
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            missing = getattr(exc, "name", "") or ""
            if (
                missing == module_name
                or module_name.startswith(f"{missing}.")
                or missing.startswith(f"{module_name}.")
            ):
                last_error = exc
                continue
            raise
    if last_error is not None:
        raise last_error
    raise ModuleNotFoundError("No import candidates provided")


_xml_parser = _import_first(
    [
        "src.data.xml_parser",
        "lgovnet.src.data.xml_parser",
    ]
)
_assert_root_module(_xml_parser, "data.xml_parser")
parse_xml_file = _xml_parser.parse_xml_file

_ontology = _import_first(
    [
        "src.data.ontology",
        "lgovnet.src.data.ontology",
    ]
)
_assert_root_module(_ontology, "data.ontology")
map_raw_to_kg = _ontology.map_raw_to_kg
LESION_NODES = _ontology.LESION_NODES

_detector_backend = _import_first(
    [
        "src.detector_backend",
        "lgovnet.src.detector_backend",
    ]
)
_assert_root_module(_detector_backend, "detector_backend")
load_cached_predictions = _detector_backend.load_cached_predictions

_anatomy_provider = _import_first(
    [
        "src.governance.anatomy_provider",
        "lgovnet.src.governance.anatomy_provider",
    ]
)
_assert_root_module(_anatomy_provider, "governance.anatomy_provider")
ApproximateAnatomyProvider = _anatomy_provider.ApproximateAnatomyProvider
ExternalPretrainedProvider = _anatomy_provider.ExternalPretrainedProvider
RadialZoneAnatomyProvider = _anatomy_provider.RadialZoneAnatomyProvider

_contracts = _import_first(
    [
        "src.governance.contracts",
        "lgovnet.src.governance.contracts",
    ]
)
_assert_root_module(_contracts, "governance.contracts")
make_placeholder_pss = _contracts.make_placeholder_pss

_evidence = _import_first(
    [
        "src.governance.evidence",
        "lgovnet.src.governance.evidence",
    ]
)
_assert_root_module(_evidence, "governance.evidence")
build_evidence_for_bbox = _evidence.build_evidence_for_bbox

_DATA_ROOT = os.path.join(_root, "data")
_RADIAL_ZONES = ("Foveal", "Parafoveal", "Vascular_Arcades", "Peripheral")
_LESION_CLASSES = ("MA", "HE", "Hard_EX", "Soft_EX")
_CANONICAL_STAGE3C_OUTPUT_DIR = Path(_root) / "outputs" / "stage3c_audit"
_SUSPICIOUS_PATH_MARKERS = (
    "/outputs/artefak/",
    "/outputs/src/",
    "/outputs/tools/",
    "/legacy/",
    "/historical/",
    "/stale/",
    "/stage3c_audit_pre",
)


def _is_external_fallback_context(ctx) -> bool:
    provider_name = str(getattr(ctx, "provider_name", "")).lower()
    provider_validation = getattr(ctx, "provider_validation", {}) or {}
    method = str(provider_validation.get("method", "")).lower()
    return (
        "fallback=approximate" in provider_name
        or method == "external_pretrained_fallback"
    )


def _norm_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").lower()


def _detect_suspicious_path(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    norm = _norm_path(path)
    for marker in _SUSPICIOUS_PATH_MARKERS:
        if marker in norm:
            return marker
    return None


def _assert_split_path_isolation(
    split: str,
    annotations_dir: Path,
    cache_dir: Path,
    secondary_cache_dir: Optional[Path] = None,
) -> None:
    expected = split.lower()
    checks = [
        ("annotations_dir", annotations_dir),
        ("cache_dir", cache_dir),
    ]
    if secondary_cache_dir is not None:
        checks.append(("secondary_cache_dir", secondary_cache_dir))

    for label, path in checks:
        tail = path.name.lower()
        if tail != expected:
            raise RuntimeError(
                f"{label} does not match --split {split}: tail='{path.name}'"
            )


# ── Thresholds ─────────────────────────────────────────────────────────────────

# SDM(pred, GT) threshold: below this → CLINICALLY_CLOSE, else UNSAFE.
# Clinical-prior SDM discrete values: {0.0, 0.5, 0.75, 1.0}.
# Threshold 0.3 captures only MA<->HE (SDM=0.0). All others are UNSAFE.
_SDM_CLOSE_THRESHOLD: float = 0.3

_IOU_TP: float = 0.5  # minimum IoU for a GT match
_IOU_WEAK: float = 0.1  # minimum IoU for localization-weak
_SCORE_LOW_CONF: float = 0.15  # below this score → LOW_CONF_CANDIDATE

# Canonical lesion node ordering for SDM key lookup
_LESION_IDX: Dict[str, int] = {
    name: idx for name, idx in sorted(LESION_NODES.items(), key=lambda kv: kv[1])
}

# Human-readable governance interpretation per category
_CATEGORY_ACTION: Dict[str, str] = {
    "TP_EXACT": "RETAIN",
    "CLINICALLY_CLOSE_MISMATCH": "RETAIN (flag as CLOSE)",
    "UNSAFE_MISMATCH": "SUPPRESS (high clinical cost)",
    "LOCALIZATION_WEAK": "RETAIN (low-confidence flag)",
    "BACKGROUND_FP": "SUPPRESS",
    "LOW_CONF_CANDIDATE": "SUPPRESS",
}


# ── KG artifact loaders ────────────────────────────────────────────────────────


def _load_sdm_norm(path: Path) -> Dict[Tuple[str, str], Dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {tuple(k.split("__", 1)): v for k, v in data.items() if "__" in k}


def _load_compat_scores(path: Path) -> Dict[Tuple[str, str], float]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {tuple(k.split("__", 1)): float(v) for k, v in data.items() if "__" in k}


def _missing_radial_pairs(compat_scores: Dict[Tuple[str, str], float]) -> List[str]:
    missing = []
    for lesion in _LESION_CLASSES:
        for zone in _RADIAL_ZONES:
            if (lesion, zone) not in compat_scores:
                missing.append(f"{lesion}__{zone}")
    return missing


# ── GT annotation loader ──────────────────────────────────────────────────────


def _load_gt_annotations(ann_dir: Path) -> Dict[str, List[dict]]:
    """Load all Pascal VOC XML files → {stem: [{class, bbox}, ...]}."""
    gt: dict = {}
    xml_files = sorted(ann_dir.glob("*.xml"))
    if not xml_files:
        print(f"[Stage3C] WARN: No XML files found in {ann_dir}")
    for xml_path in xml_files:
        stem = xml_path.stem
        parsed = parse_xml_file(xml_path)
        if parsed is None:
            continue
        boxes = []
        for obj in parsed["objects"]:
            canonical = map_raw_to_kg(obj["label_raw"])
            if canonical is None:
                continue
            boxes.append({"class": canonical, "bbox": obj["bbox"]})
        gt[stem] = boxes
    return gt


# ── IoU ───────────────────────────────────────────────────────────────────────


def _iou(a: List[float], b: List[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _extract_bbox_score(box: dict) -> Tuple[List[float], float]:
    raw = box.get("detector_raw") or box.get("r2a_raw") or {}
    bb = raw.get("bbox") or [0.0, 0.0, 0.0, 0.0]
    score = raw.get("score")
    if score is None:
        ep = box.get("ensemble_probs") or {}
        score = max((float(v) for v in ep.values()), default=0.0)
    try:
        score_f = float(score)
    except (TypeError, ValueError):
        score_f = 0.0
    return [float(v) for v in bb], score_f


def _normalise_probs_for_head(box: dict) -> Dict[str, float]:
    raw = box.get("detector_class_probs") or box.get("ensemble_probs") or {}
    probs = {cls: max(float(raw.get(cls, 0.0)), 0.0) for cls in _LESION_CLASSES}
    total = sum(probs.values())
    if total <= 0.0:
        raw_det = box.get("detector_raw") or box.get("r2a_raw") or {}
        pred_cls = raw_det.get("label_name")
        if pred_cls in probs:
            probs = {cls: (1.0 if cls == pred_cls else 0.0) for cls in _LESION_CLASSES}
        else:
            probs = {cls: 1.0 / len(_LESION_CLASSES) for cls in _LESION_CLASSES}
        total = sum(probs.values())
    return {cls: probs[cls] / total for cls in _LESION_CLASSES}


def _build_head_prediction(box: dict, source_id: str) -> dict:
    probs = _normalise_probs_for_head(box)
    _, score = _extract_bbox_score(box)
    return {
        "source": source_id,
        "score": float(score),
        "class_probs": probs,
    }


def _attach_multisource_heads(
    primary_preds: Dict[str, List[dict]],
    secondary_preds: Dict[str, List[dict]],
    iou_threshold: float,
    coverage_score_min: float = 0.05,
) -> dict:
    total_boxes = 0
    matched_boxes = 0
    total_high_conf_boxes = 0
    matched_high_conf_boxes = 0

    for stem, pboxes in primary_preds.items():
        sboxes = secondary_preds.get(stem, [])
        used_secondary = set()

        for pbox in pboxes:
            total_boxes += 1
            p_bbox, p_score = _extract_bbox_score(pbox)
            in_high_conf = p_score >= coverage_score_min
            if in_high_conf:
                total_high_conf_boxes += 1

            best_idx = None
            best_iou = 0.0
            for idx, sbox in enumerate(sboxes):
                if idx in used_secondary:
                    continue
                s_bbox, _ = _extract_bbox_score(sbox)
                ov = _iou(p_bbox, s_bbox)
                if ov > best_iou:
                    best_iou = ov
                    best_idx = idx

            head_preds = [_build_head_prediction(pbox, "primary")]
            if best_idx is not None and best_iou >= iou_threshold:
                used_secondary.add(best_idx)
                head_preds.append(_build_head_prediction(sboxes[best_idx], "secondary"))
                matched_boxes += 1
                if in_high_conf:
                    matched_high_conf_boxes += 1

            pbox["head_predictions"] = head_preds
            pbox["source_count"] = len(head_preds)

    return {
        "total_boxes": total_boxes,
        "matched_boxes": matched_boxes,
        "matched_coverage": (matched_boxes / total_boxes) if total_boxes > 0 else 0.0,
        "total_high_conf_boxes": total_high_conf_boxes,
        "matched_high_conf_boxes": matched_high_conf_boxes,
        "matched_high_conf_coverage": (
            (matched_high_conf_boxes / total_high_conf_boxes)
            if total_high_conf_boxes > 0
            else 0.0
        ),
        "coverage_score_min": coverage_score_min,
        "iou_threshold": iou_threshold,
    }


# ── SDM pred–GT confusion cost ────────────────────────────────────────────────


def _sdm_pred_gt(pred_class: str, gt_class: str, sdm_norm: dict) -> Optional[float]:
    """Look up the clinical SDM confusion cost between predicted and GT class.

    This is NOT the same as SDCA (top-1 vs runner-up). It measures how
    dangerous it is to predict pred_class when the GT is gt_class.
    Returns None if either class is not in LESION_NODES.
    """
    if pred_class not in _LESION_IDX or gt_class not in _LESION_IDX:
        return None
    if pred_class == gt_class:
        return 0.0
    a, b = (
        (pred_class, gt_class)
        if _LESION_IDX[pred_class] < _LESION_IDX[gt_class]
        else (gt_class, pred_class)
    )
    entry = sdm_norm.get((a, b))
    return entry["norm"] if entry else None


# ── Governance target assignment ──────────────────────────────────────────────


def _assign_target(
    pred_class: str,
    pred_bbox: List[float],
    pred_score: float,
    gt_boxes: List[dict],
    sdm_norm: dict,
) -> dict:
    """Assign one governance target category to a single predicted box.

    Uses non-exclusive IoU matching: each predicted box is matched to the
    GT box with the highest IoU, regardless of other predictions.

    Returns:
        {category, best_iou, best_gt_class, semantic_confusion_cost, action}
    """
    best_iou = 0.0
    best_gt_class = None

    for gt in gt_boxes:
        iou_val = _iou(pred_bbox, gt["bbox"])
        if iou_val > best_iou:
            best_iou = iou_val
            best_gt_class = gt["class"]

    if best_iou >= _IOU_TP:
        if pred_class == best_gt_class:
            cat = "TP_EXACT"
            confusion_cost = 0.0
        else:
            confusion_cost = _sdm_pred_gt(pred_class, best_gt_class, sdm_norm)
            if confusion_cost is None:
                confusion_cost = 0.5  # unknown pair — treat as moderate
            cat = (
                "CLINICALLY_CLOSE_MISMATCH"
                if confusion_cost < _SDM_CLOSE_THRESHOLD
                else "UNSAFE_MISMATCH"
            )
    elif best_iou >= _IOU_WEAK:
        cat = "LOCALIZATION_WEAK"
        confusion_cost = None
    else:
        cat = "LOW_CONF_CANDIDATE" if pred_score < _SCORE_LOW_CONF else "BACKGROUND_FP"
        confusion_cost = None
        best_gt_class = None

    return {
        "category": cat,
        "best_iou": round(best_iou, 4),
        "best_gt_class": best_gt_class,
        # gt_class is a VALID supervised target only when the box has a real
        # IoU>=0.5 GT match (TP_EXACT / CLINICALLY_CLOSE / UNSAFE_MISMATCH).
        # Weak-overlap (LOCALIZATION_WEAK) and no-match boxes must NOT inherit a
        # label — downstream must treat them as unresolved, never majority_class.
        "gt_supervised": bool(best_iou >= _IOU_TP and best_gt_class is not None),
        "semantic_confusion_cost": (
            round(confusion_cost, 4) if confusion_cost is not None else None
        ),
        "action": _CATEGORY_ACTION.get(cat, "UNKNOWN"),
    }


# ── SDCA proxy quality audit ──────────────────────────────────────────────────


def _sdca_proxy_for_box(box: dict, pred_class: str) -> dict:
    """Compute SDCA proxy quality fields for one box.

    The current single-head detector bridge constructs ensemble_probs as:
      predicted_class = detection_score
      other 3 classes = (1 - score) / 3  each (tied)

    Consequences:
    - If score > 0.25: majority_class == pred_class; runner_up is first
      non-predicted class by dict iteration order (tie-breaking artifact).
    - If score < 0.25: majority_class != pred_class (non-detected class wins
      argmax because (1-s)/3 > s). SDCA then measures confusion between
      two non-detected classes, not between detected class and nearest alternative.
    - In both cases non-predicted probs are exactly tied (synthetic).

    Returns dict with:
      is_synthetic_prob:        True if all non-predicted probs are equal
      majority_matches_pred:    True if argmax(ensemble_probs) == pred_class
      runner_up_class:          Runner-up class used for SDCA computation
      majority_cls_ev:          Actual majority class in evidence computation
    """
    ep = box.get("ensemble_probs", {})
    result = {
        "is_synthetic_prob": False,
        "majority_matches_pred": True,
        "runner_up_class": None,
        "majority_cls_ev": pred_class,
    }
    if not ep or pred_class not in ep:
        return result

    # Majority class as evidence.py sees it (argmax of ensemble_probs)
    ep_sorted = sorted(ep, key=lambda c: ep[c], reverse=True)
    majority_cls_ev = ep_sorted[0] if ep_sorted else pred_class
    runner_up = next((c for c in ep_sorted if c != majority_cls_ev), None)

    non_pred_vals = [ep[c] for c in ep if c != pred_class]
    is_synthetic = (
        len(non_pred_vals) > 0 and len(set(round(v, 10) for v in non_pred_vals)) == 1
    )

    result["is_synthetic_prob"] = is_synthetic
    result["majority_matches_pred"] = majority_cls_ev == pred_class
    result["runner_up_class"] = runner_up
    result["majority_cls_ev"] = majority_cls_ev
    return result


# ── Per-image processing ───────────────────────────────────────────────────────


def _process_image(
    stem: str,
    pred_boxes: List[dict],
    gt_boxes: List[dict],
    provider,
    sdm_norm: dict,
    compat_scores: dict,
    image_path: Optional[str],
    pss_lookup: Optional[Dict[str, dict]] = None,
) -> List[dict]:
    if not pred_boxes:
        return []

    # Anatomy context (ApproximateAnatomyProvider → low-conf; RadialZone → moderate)
    img_dims: Optional[Tuple[int, int]] = None
    if image_path and os.path.exists(image_path):
        try:
            from PIL import Image as PILImage

            img = PILImage.open(image_path)
            img_dims = (int(img.size[0]), int(img.size[1]))
            ctx = provider(img, image_id=stem)
        except Exception:
            ctx = provider(None, image_id=stem)
    else:
        ctx = provider(None, image_id=stem)
    anatomy_ctx = ctx.to_dict()
    anatomy_provider_name = str(getattr(ctx, "provider_name", ""))
    anatomy_provider_method = str(
        (getattr(ctx, "provider_validation", {}) or {}).get("method", "")
    )
    anatomy_fallback_to_approximate = _is_external_fallback_context(ctx)

    # Pilar 3: attach radial_zone per box when using RadialZoneAnatomyProvider
    if isinstance(provider, RadialZoneAnatomyProvider) and img_dims:
        W, H = img_dims
        for b in pred_boxes:
            raw = b.get("detector_raw") or b.get("r2a_raw") or {}
            bb = raw.get("bbox") or [0, 0, 0, 0]
            cx = (float(bb[0]) + float(bb[2])) / 2.0
            cy = (float(bb[1]) + float(bb[3])) / 2.0
            b["radial_zone"] = RadialZoneAnatomyProvider.assign_radial_zone(
                cx, cy, W, H
            )

    records = []
    for box in pred_boxes:
        bbox_id = box.get("bbox_id", "")
        detector_raw = box.get("detector_raw") or box.get("r2a_raw", {})
        pred_bbox = detector_raw.get("bbox", [0.0, 0.0, 0.0, 0.0])
        pred_score = float(detector_raw.get("score", 0.0))
        pred_class = detector_raw.get("label_name")

        if pred_class is None:
            ep = box.get("ensemble_probs", {})
            pred_class = max(ep, key=lambda c: ep[c]) if ep else None
        if pred_class is None:
            continue

        # SDCA proxy quality (R2a synthetic ensemble_probs)
        proxy = _sdca_proxy_for_box(box, pred_class)

        # Governance target
        target = _assign_target(pred_class, pred_bbox, pred_score, gt_boxes, sdm_norm)

        # Evidence computation (re-run; fast since it's dict lookups only)
        ev_sas = ev_sdca = ev_aps = ev_pss = None
        ev_pss_status = None
        ev_anatomy_uncertain = None
        ev_warnings: List[str] = []
        try:
            # Pilar 2: prefer real TTA PSS when available
            if pss_lookup and bbox_id in pss_lookup:
                pss = pss_lookup[bbox_id]
            else:
                pss = make_placeholder_pss(bbox_id)
            ev = build_evidence_for_bbox(
                box=box,
                pss_result=pss,
                sdm_norm=sdm_norm,
                compat_scores=compat_scores,
                anatomy_context=anatomy_ctx,
            )
            vec = ev.get("evidence_vector", [])
            if len(vec) == 4 and all(math.isfinite(v) for v in vec):
                ev_sas, ev_sdca, ev_aps, ev_pss = vec
            ev_pss_status = ev.get("pss_status")
            ev_anatomy_uncertain = ev.get("anatomy_uncertain")
            ev_warnings = ev.get("warnings") or []
        except Exception:
            pass

        ev_sas_used_fallback = any("SAS fell back" in w for w in ev_warnings)

        cls_probs = box.get("detector_class_probs") or box.get("ensemble_probs") or {}
        unc_msp, unc_margin, unc_entconf = _uncertainty_features(cls_probs)
        majority_class = proxy.get("majority_cls_ev") or pred_class
        tier_target = _MODE2_TIER_BY_CATEGORY.get(target["category"], "R4")
        # P0 leakage fix (2026-05-28): gt_class is a supervised label ONLY for a
        # valid IoU>=0.5 GT match. Unresolved boxes get "" (NOT majority_class),
        # so Stage 5 cannot train the re-classifier toward the detector's own
        # argmax (circular). majority_class stays a separate, clearly-labeled col.
        gt_supervised = bool(target.get("gt_supervised"))
        gt_class = target["best_gt_class"] if gt_supervised else ""
        mode2_probs = _normalise_class_probs(box, pred_class)
        consensus = _consensus_for_mode2(box, gt_class, majority_class, tier_target)

        records.append(
            {
                "bbox_id": bbox_id,
                "image_id": stem,
                "pred_class": pred_class,
                "pred_score": round(pred_score, 4),
                "category": target["category"],
                "action": target["action"],
                "tier_target": tier_target,
                "majority_class": majority_class,
                "gt_class": gt_class,
                "gt_supervised": gt_supervised,
                "consensus": consensus,
                "prob_ma": mode2_probs["MA"],
                "prob_he": mode2_probs["HE"],
                "prob_hardex": mode2_probs["Hard_EX"],
                "prob_softex": mode2_probs["Soft_EX"],
                "best_iou": target["best_iou"],
                "best_gt_class": target["best_gt_class"],
                "semantic_confusion_cost": target["semantic_confusion_cost"],
                "ev_sas": ev_sas,
                "ev_sdca": ev_sdca,
                "ev_aps": ev_aps,
                "ev_pss": ev_pss,
                "ev_pss_status": ev_pss_status,
                "ev_anatomy_uncertain": ev_anatomy_uncertain,
                "unc_msp": unc_msp,
                "unc_margin": unc_margin,
                "unc_entconf": unc_entconf,
                "probs_source": box.get("probs_source", "unknown"),
                "radial_zone": box.get("radial_zone"),
                "anatomy_provider_name": anatomy_provider_name,
                "anatomy_provider_method": anatomy_provider_method,
                "anatomy_fallback_to_approximate": anatomy_fallback_to_approximate,
                "is_synthetic_prob": proxy["is_synthetic_prob"],
                "majority_matches_pred": proxy["majority_matches_pred"],
                "sdca_runner_up": proxy["runner_up_class"],
                "source_count": int(box.get("source_count") or 0),
                "sas_used_fallback": ev_sas_used_fallback,
            }
        )

    return records


# ── Aggregation ────────────────────────────────────────────────────────────────


def _aggregate(records: List[dict], coverage_score_min: float = 0.05) -> dict:
    by_cat: Dict[str, List[dict]] = defaultdict(list)
    by_cls: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for r in records:
        by_cat[r["category"]].append(r)
        by_cls[r["pred_class"]][r["category"]] += 1

    total = len(records)

    def _mean(recs: List[dict], key: str) -> Optional[float]:
        vals = [r[key] for r in recs if r.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    cat_stats: dict = {}
    for cat, recs in by_cat.items():
        cat_stats[cat] = {
            "count": len(recs),
            "pct": round(100.0 * len(recs) / total, 2) if total else 0.0,
            "avg_iou": _mean(recs, "best_iou"),
            "avg_pred_score": _mean(recs, "pred_score"),
            "avg_sas": _mean(recs, "ev_sas"),
            "avg_sdca": _mean(recs, "ev_sdca"),
            "avg_aps": _mean(recs, "ev_aps"),
            "avg_pss": _mean(recs, "ev_pss"),
            "avg_semantic_confusion_cost": _mean(recs, "semantic_confusion_cost"),
            "action": _CATEGORY_ACTION.get(cat, "UNKNOWN"),
        }

    # Per-class SDCA stats across all categories
    cls_sdca: dict = {}
    for cls in ["MA", "HE", "Hard_EX", "Soft_EX"]:
        recs = [
            r for r in records if r["pred_class"] == cls and r["ev_sdca"] is not None
        ]
        cls_sdca[cls] = {
            "n": len(recs),
            "mean_sdca": (
                round(sum(r["ev_sdca"] for r in recs) / len(recs), 4) if recs else None
            ),
        }

    # SDCA proxy quality audit (R2a bridge: ensemble_probs is synthetic)
    sdca_proxy: dict = {}
    for cls in ["MA", "HE", "Hard_EX", "Soft_EX"]:
        recs_cls = [r for r in records if r["pred_class"] == cls]
        if not recs_cls:
            continue
        n = len(recs_cls)
        n_synth = sum(1 for r in recs_cls if r.get("is_synthetic_prob"))
        n_maj_match = sum(1 for r in recs_cls if r.get("majority_matches_pred"))
        runner_ups = [r["sdca_runner_up"] for r in recs_cls if r.get("sdca_runner_up")]
        sdca_proxy[cls] = {
            "n": n,
            "tie_rate": round(n_synth / n, 4),
            "majority_matches_pred_rate": round(n_maj_match / n, 4),
            "runner_up_distribution": dict(Counter(runner_ups)),
        }

    probs_source_counts = dict(
        Counter(r.get("probs_source", "unknown") for r in records)
    )
    pss_status_counts = dict(
        Counter(r.get("ev_pss_status", "missing") for r in records)
    )
    radial_zone_counts = dict(
        Counter(
            r.get("radial_zone") for r in records if r.get("radial_zone") is not None
        )
    )
    source_count_hist = dict(Counter(int(r.get("source_count") or 0) for r in records))
    multisource_boxes = sum(1 for r in records if int(r.get("source_count") or 0) >= 2)
    high_conf_boxes = sum(
        1 for r in records if float(r.get("pred_score") or 0.0) >= coverage_score_min
    )
    multisource_high_conf_boxes = sum(
        1
        for r in records
        if int(r.get("source_count") or 0) >= 2
        and float(r.get("pred_score") or 0.0) >= coverage_score_min
    )
    sas_fallback_boxes = sum(1 for r in records if r.get("sas_used_fallback"))
    anatomy_external_fallback_boxes = sum(
        1 for r in records if r.get("anatomy_fallback_to_approximate")
    )
    anatomy_external_fallback_images = len(
        {r.get("image_id") for r in records if r.get("anatomy_fallback_to_approximate")}
    )
    anatomy_provider_name_counts = dict(
        Counter(r.get("anatomy_provider_name", "") for r in records)
    )
    anatomy_provider_method_counts = dict(
        Counter(r.get("anatomy_provider_method", "") for r in records)
    )

    return {
        "total_boxes": total,
        "by_category": cat_stats,
        "by_pred_class_category": {cls: dict(cats) for cls, cats in by_cls.items()},
        "sdca_by_pred_class": cls_sdca,
        "sdca_proxy_audit": sdca_proxy,
        "prob_source_counts": probs_source_counts,
        "pss_status_counts": pss_status_counts,
        "radial_zone_counts": radial_zone_counts,
        "source_count_hist": source_count_hist,
        "multisource_boxes": multisource_boxes,
        "multisource_coverage": round(multisource_boxes / total, 4) if total else 0.0,
        "high_conf_boxes": high_conf_boxes,
        "multisource_high_conf_boxes": multisource_high_conf_boxes,
        "multisource_high_conf_coverage": (
            round(multisource_high_conf_boxes / high_conf_boxes, 4)
            if high_conf_boxes
            else 0.0
        ),
        "coverage_score_min": coverage_score_min,
        "sas_fallback_boxes": sas_fallback_boxes,
        "sas_fallback_coverage": round(sas_fallback_boxes / total, 4) if total else 0.0,
        "anatomy_external_fallback_boxes": anatomy_external_fallback_boxes,
        "anatomy_external_fallback_images": anatomy_external_fallback_images,
        "anatomy_provider_name_counts": anatomy_provider_name_counts,
        "anatomy_provider_method_counts": anatomy_provider_method_counts,
    }


# ── Gate ──────────────────────────────────────────────────────────────────────


_REQUIRED_CATS = {"TP_EXACT", "BACKGROUND_FP"}
_COLLAPSE_THRESHOLD = 0.90


def _gate_check(
    agg: dict,
    require_multisource_sas: bool = True,
    min_multisource_coverage: float = 0.50,
    coverage_score_min: float = 0.05,
) -> Tuple[bool, List[str], List[str]]:
    errors: List[str] = []
    notes: List[str] = []
    total = agg["total_boxes"]
    by_cat = agg["by_category"]

    if total == 0:
        errors.append("No boxes processed — check cache and annotations dirs")
        return False, errors, notes

    # Required categories must be non-empty
    for cat in _REQUIRED_CATS:
        if by_cat.get(cat, {}).get("count", 0) == 0:
            errors.append(f"Required category '{cat}' has 0 boxes")

    # No distribution collapse
    for cat, s in by_cat.items():
        if s["pct"] / 100.0 > _COLLAPSE_THRESHOLD:
            errors.append(f"Distribution collapsed: {cat}={s['pct']:.1f}%")

    if require_multisource_sas:
        ms_cov = agg.get(
            "multisource_high_conf_coverage", agg.get("multisource_coverage", 0.0)
        )
        if ms_cov < min_multisource_coverage:
            errors.append(
                f"Multi-source SAS high-conf coverage too low: {100.0 * ms_cov:.1f}% "
                f"(<{100.0 * min_multisource_coverage:.1f}%) @score>={coverage_score_min:.2f}"
            )
        sf = agg.get("sas_fallback_boxes", 0)
        if sf > 0:
            errors.append(
                f"SAS fallback present in {sf} boxes — head_predictions missing/invalid"
            )
    else:
        notes.append("Single-source SAS allowed by override (legacy/proxy mode).")

    # Clinical cost ordering: CLOSE confusion cost < UNSAFE confusion cost
    close_cost = by_cat.get("CLINICALLY_CLOSE_MISMATCH", {}).get(
        "avg_semantic_confusion_cost"
    )
    unsafe_cost = by_cat.get("UNSAFE_MISMATCH", {}).get("avg_semantic_confusion_cost")
    if close_cost is not None and unsafe_cost is not None:
        if close_cost < unsafe_cost:
            notes.append(
                f"GOOD: avg_confusion_cost(CLOSE)={close_cost:.4f} < "
                f"avg_confusion_cost(UNSAFE)={unsafe_cost:.4f} — clinical ordering confirmed"
            )
        else:
            errors.append(
                f"Clinical ordering violated: CLOSE cost={close_cost:.4f} >= UNSAFE cost={unsafe_cost:.4f}"
            )
    elif "CLINICALLY_CLOSE_MISMATCH" not in by_cat:
        notes.append(
            "NOTE: No CLINICALLY_CLOSE_MISMATCH found — MA<->HE confusion absent (annotation quality may limit this)"
        )
    elif "UNSAFE_MISMATCH" not in by_cat:
        notes.append(
            "NOTE: No UNSAFE_MISMATCH found — all class mismatches are clinically safe"
        )

    # SDCA clinical ordering: MA < Hard_EX < Soft_EX (expected from clinical-prior SDM)
    sdca_cls = agg.get("sdca_by_pred_class", {})
    ma_sdca = sdca_cls.get("MA", {}).get("mean_sdca")
    hex_sdca = sdca_cls.get("Hard_EX", {}).get("mean_sdca")
    sex_sdca = sdca_cls.get("Soft_EX", {}).get("mean_sdca")
    if ma_sdca is not None and sex_sdca is not None and ma_sdca < sex_sdca:
        notes.append(
            f"GOOD: mean_SDCA(MA)={ma_sdca:.4f} < mean_SDCA(Soft_EX)={sex_sdca:.4f} "
            "— ICDR-aware evidence ordering confirmed in governance targets"
        )
    elif ma_sdca is not None and sex_sdca is not None:
        notes.append(
            f"WARN: mean_SDCA(MA)={ma_sdca:.4f} >= mean_SDCA(Soft_EX)={sex_sdca:.4f} "
            "— unexpected, check evidence computation"
        )

    # SDCA proxy quality — informational, never a gate failure
    proxy = agg.get("sdca_proxy_audit", {})
    if proxy:
        high_tie = all(p.get("tie_rate", 0) > 0.95 for p in proxy.values())
        if high_tie:
            notes.append(
                "SDCA proxy quality: tie_rate ~100% across classes — "
                "ensemble_probs are synthetic (single-head detector bridge; non-predicted probs equal by construction). "
                "SDCA is a deterministic function of predicted class + confidence threshold, "
                "NOT real detector classifier uncertainty. "
                "Paper must note: 'SDCA computed from exported class probabilities; "
                "non-predicted class scores are equal by construction in the current single-head detector bridge.'"
            )
        low_maj = [
            cls
            for cls, p in proxy.items()
            if p.get("majority_matches_pred_rate", 1) < 0.9
        ]
        if low_maj:
            notes.append(
                f"SDCA proxy: majority_class != pred_class for >10%% of boxes in {low_maj}. "
                "Low-confidence detections (score<0.25) produce inverted ensemble_probs — "
                "majority_class becomes a non-detected class. "
                "These boxes contribute SDCA from confusion between non-detected alternatives."
            )

    return len(errors) == 0, errors, notes


# ── CSV output ────────────────────────────────────────────────────────────────


_CSV_FIELDS = [
    "split",
    "category",
    "action",
    "count",
    "pct",
    "avg_iou",
    "avg_pred_score",
    "avg_sas",
    "avg_sdca",
    "avg_aps",
    "avg_pss",
    "avg_semantic_confusion_cost",
]


def _write_csv(split: str, agg: dict, csv_path: Path) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(_distribution_rows(split, agg))


def _distribution_rows(split: str, agg: dict) -> List[dict]:
    cat_order = [
        "TP_EXACT",
        "CLINICALLY_CLOSE_MISMATCH",
        "UNSAFE_MISMATCH",
        "LOCALIZATION_WEAK",
        "BACKGROUND_FP",
        "LOW_CONF_CANDIDATE",
    ]
    rows = []
    for cat in cat_order:
        s = agg["by_category"].get(cat, {})
        rows.append(
            {
                "split": split,
                "category": cat,
                "action": _CATEGORY_ACTION.get(cat, ""),
                "count": s.get("count", 0),
                "pct": s.get("pct", 0.0),
                "avg_iou": s.get("avg_iou", ""),
                "avg_pred_score": s.get("avg_pred_score", ""),
                "avg_sas": s.get("avg_sas", ""),
                "avg_sdca": s.get("avg_sdca", ""),
                "avg_aps": s.get("avg_aps", ""),
                "avg_pss": s.get("avg_pss", ""),
                "avg_semantic_confusion_cost": s.get("avg_semantic_confusion_cost", ""),
            }
        )
    return rows


_RETAIN_CATS_BINARY = frozenset(
    {"TP_EXACT", "CLINICALLY_CLOSE_MISMATCH", "LOCALIZATION_WEAK"}
)


def _uncertainty_features(class_probs):
    """Same-family selective-prediction features from a per-class probability
    dict: max-softmax probability (MSP), margin (top1-top2), and entropy-
    derived confidence (1 - normalized predictive entropy). All three are
    confidence-aligned and in [0, 1]: higher = more confident."""
    if not class_probs:
        return None, None, None
    vals = []
    for v in class_probs.values():
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(fv) and fv >= 0.0:
            vals.append(fv)
    s = sum(vals)
    if not vals or s <= 0.0:
        return None, None, None
    p = sorted((v / s for v in vals), reverse=True)
    msp = p[0]
    margin = p[0] - p[1] if len(p) > 1 else p[0]
    entropy = -sum(pi * math.log(pi) for pi in p if pi > 0.0)
    h_max = math.log(len(p)) if len(p) > 1 else 1.0
    entconf = 1.0 - entropy / h_max if h_max > 0.0 else 1.0
    return round(msp, 6), round(margin, 6), round(entconf, 6)


_FEATURES_CSV_FIELDS = [
    "bbox_id",
    "image_id",
    "split",
    "run_repo_root",
    "source_cache_dir",
    "source_annotations_dir",
    "source_kg_dir",
    "source_output_dir",
    "anatomy_provider_mode",
    "anatomy_quality_mode",
    "anatomy_checkpoint",
    "pss_source_json",
    "compare_baseline_source",
    "pred_class",
    "pred_score",
    "category",
    "label",
    "tier_target",
    "majority_class",
    "gt_class",
    "gt_supervised",
    "consensus",
    "prob_ma",
    "prob_he",
    "prob_hardex",
    "prob_softex",
    "probs_source",
    "radial_zone",
    "anatomy_provider_name",
    "anatomy_provider_method",
    "anatomy_fallback_to_approximate",
    "ev_sas",
    "ev_sdca",
    "ev_aps",
    "ev_pss",
    "ev_pss_status",
    "unc_msp",
    "unc_margin",
    "unc_entconf",
    "best_iou",
    "best_gt_class",
    "semantic_confusion_cost",
]

_MODE2_TIER_BY_CATEGORY: Dict[str, str] = {
    "TP_EXACT": "R1",
    "CLINICALLY_CLOSE_MISMATCH": "R2",
    "UNSAFE_MISMATCH": "R3",
    "LOCALIZATION_WEAK": "R3",
    "BACKGROUND_FP": "R4",
    "LOW_CONF_CANDIDATE": "R4",
}


def _normalise_class_probs(box: dict, pred_class: str) -> Dict[str, float]:
    """Return canonical per-class probabilities for the Mode 2 CSV schema."""
    raw = box.get("detector_class_probs") or box.get("ensemble_probs") or {}
    probs = {cls: float(raw.get(cls, 0.0)) for cls in _LESION_CLASSES}
    total = sum(probs.values())

    if total <= 0.0:
        probs = {cls: 1.0 if cls == pred_class else 0.0 for cls in _LESION_CLASSES}
        total = 1.0

    return {cls: probs[cls] / total for cls in _LESION_CLASSES}


def _consensus_for_mode2(
    box: dict,
    gt_class: str,
    majority_class: str,
    tier_target: str,
) -> float:
    """Estimate consensus for Stage 5 when multi-head votes are absent."""
    heads = box.get("head_predictions") or []
    head_preds = []
    for head in heads:
        class_probs = head.get("class_probs") or {}
        if class_probs:
            head_preds.append(max(class_probs, key=lambda cls: class_probs[cls]))

    if head_preds:
        target_class = gt_class if gt_class in _LESION_CLASSES else majority_class
        if target_class in _LESION_CLASSES:
            correct = sum(1 for pred in head_preds if pred == target_class)
            return round(correct / len(head_preds), 4)

    fallback = {
        "R1": 1.0,
        "R2": 0.5,
        "R3": 0.5,
        "R4": 0.0,
    }
    return fallback.get(tier_target, 0.5)


def _write_features_csv(
    records: List[dict],
    path: Path,
    split: str,
    provenance: dict,
) -> None:
    """Save per-box features + binary label for Stage 3D LGG training.

    label=1: RETAIN_OR_FLAG (TP_EXACT, CLINICALLY_CLOSE_MISMATCH, LOCALIZATION_WEAK)
    label=0: SUPPRESS (BACKGROUND_FP, LOW_CONF_CANDIDATE, UNSAFE_MISMATCH)

    The CSV also carries the extra Mode 2 fields consumed by train_stage5_mode2.
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FEATURES_CSV_FIELDS)
        writer.writeheader()
        for r in records:
            writer.writerow(
                {
                    "bbox_id": r["bbox_id"],
                    "image_id": r["image_id"],
                    "split": split,
                    "run_repo_root": provenance.get("repo_root") or "",
                    "source_cache_dir": provenance.get("cache_dir") or "",
                    "source_annotations_dir": provenance.get("annotations_dir") or "",
                    "source_kg_dir": provenance.get("kg_dir") or "",
                    "source_output_dir": provenance.get("output_dir") or "",
                    "anatomy_provider_mode": provenance.get("anatomy_provider_mode")
                    or "",
                    "anatomy_quality_mode": provenance.get("anatomy_quality_mode")
                    or "",
                    "anatomy_checkpoint": provenance.get("anatomy_checkpoint") or "",
                    "pss_source_json": provenance.get("pss_source_json") or "",
                    "compare_baseline_source": provenance.get("compare_baseline_source")
                    or "",
                    "pred_class": r["pred_class"],
                    "pred_score": r["pred_score"],
                    "category": r["category"],
                    "label": 1 if r["category"] in _RETAIN_CATS_BINARY else 0,
                    "tier_target": r.get("tier_target") or "R4",
                    "majority_class": r.get("majority_class") or r["pred_class"],
                    # P0 leakage fix: NO fallback to best_gt_class/majority_class/
                    # pred_class. Unresolved rows stay empty so Stage 5 excludes
                    # them from supervised R1/R3 loss instead of learning the
                    # detector's own label.
                    "gt_class": r.get("gt_class") or "",
                    "gt_supervised": int(bool(r.get("gt_supervised"))),
                    "consensus": r.get("consensus", 0.5),
                    "prob_ma": r.get("prob_ma", ""),
                    "prob_he": r.get("prob_he", ""),
                    "prob_hardex": r.get("prob_hardex", ""),
                    "prob_softex": r.get("prob_softex", ""),
                    "probs_source": r.get("probs_source") or "unknown",
                    "radial_zone": r.get("radial_zone") or "",
                    "anatomy_provider_name": r.get("anatomy_provider_name") or "",
                    "anatomy_provider_method": (r.get("anatomy_provider_method") or ""),
                    "anatomy_fallback_to_approximate": int(
                        bool(r.get("anatomy_fallback_to_approximate"))
                    ),
                    "ev_sas": r["ev_sas"] if r["ev_sas"] is not None else "",
                    "ev_sdca": r["ev_sdca"] if r["ev_sdca"] is not None else "",
                    "ev_aps": r["ev_aps"] if r["ev_aps"] is not None else "",
                    "ev_pss": r["ev_pss"] if r["ev_pss"] is not None else "",
                    "ev_pss_status": r.get("ev_pss_status") or "",
                    "unc_msp": r["unc_msp"] if r.get("unc_msp") is not None else "",
                    "unc_margin": (
                        r["unc_margin"] if r.get("unc_margin") is not None else ""
                    ),
                    "unc_entconf": (
                        r["unc_entconf"] if r.get("unc_entconf") is not None else ""
                    ),
                    "best_iou": r["best_iou"],
                    "best_gt_class": r["best_gt_class"] or "",
                    "semantic_confusion_cost": (
                        r["semantic_confusion_cost"]
                        if r["semantic_confusion_cost"] is not None
                        else ""
                    ),
                }
            )


def _write_decision_md(
    split: str,
    agg: dict,
    gate_pass: bool,
    notes: List[str],
    errors: List[str],
    md_path: Path,
) -> None:
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_render_decision_section(split, agg, gate_pass, notes, errors))


def _split_sort_key(split: str) -> Tuple[int, str]:
    order = {"train": 0, "valid": 1, "test": 2}
    return (order.get(split, len(order)), split)


def _render_decision_section(
    split: str,
    agg: dict,
    gate_pass: bool,
    notes: List[str],
    errors: List[str],
) -> str:
    lines = [
        f"## Stage 3C Governance Target Audit — {split}",
        "",
        f"**Gate: {'PASS' if gate_pass else 'FAIL'}** | Total boxes: {agg['total_boxes']}",
        "",
        "| Category | Action | Count | % | avg SAS | avg SDCA | avg confusion cost |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    cat_order = [
        "TP_EXACT",
        "CLINICALLY_CLOSE_MISMATCH",
        "UNSAFE_MISMATCH",
        "LOCALIZATION_WEAK",
        "BACKGROUND_FP",
        "LOW_CONF_CANDIDATE",
    ]
    for cat in cat_order:
        s = agg["by_category"].get(cat, {"count": 0, "pct": 0.0})
        lines.append(
            f"| {cat} | {_CATEGORY_ACTION.get(cat, '')} "
            f"| {s.get('count', 0)} | {s.get('pct', 0.0):.1f}% "
            f"| {s.get('avg_sas') or '-'} | {s.get('avg_sdca') or '-'} "
            f"| {s.get('avg_semantic_confusion_cost') or '-'} |"
        )

    lines.extend(["", "**SDCA by predicted class:**", ""])
    for cls in ["MA", "HE", "Hard_EX", "Soft_EX"]:
        info = agg["sdca_by_pred_class"].get(cls, {})
        lines.append(
            f"- {cls}: mean_SDCA={info.get('mean_sdca') or '-'}  n={info.get('n', 0)}"
        )

    lines.extend(
        [
            "",
            "**SDCA proxy audit (current single-head detector bridge — paper-honest reporting):**",
            "",
            "| Class | tie_rate | maj==pred rate | runner_up_dist |",
            "|---|---:|---:|---|",
        ]
    )
    for cls in ["MA", "HE", "Hard_EX", "Soft_EX"]:
        px = agg.get("sdca_proxy_audit", {}).get(cls, {})
        lines.append(
            f"| {cls} | {px.get('tie_rate', 0):.1%} "
            f"| {px.get('majority_matches_pred_rate', 0):.1%} "
            f"| {px.get('runner_up_distribution', {})} |"
        )

    proxy = agg.get("sdca_proxy_audit", {})
    high_tie = bool(proxy) and all(p.get("tie_rate", 0) > 0.95 for p in proxy.values())
    if high_tie:
        lines.extend(
            [
                "",
                "> NOTE: tie_rate~100% confirms ensemble_probs are synthetic. "
                "SDCA is a deterministic proxy, not real detector uncertainty. "
                "Paper limitation: 'SDCA derived from current single-head detector exported probabilities; "
                "non-predicted class scores equal by construction.'",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "> NOTE: tie_rate is not saturated; exported probabilities include non-synthetic variation. "
                "SDCA proxy caveat remains, but Pilar 1 real-class-prob activation is visible in this split.",
            ]
        )

    src_counts = agg.get("prob_source_counts", {})
    pss_counts = agg.get("pss_status_counts", {})
    rz_counts = agg.get("radial_zone_counts", {})
    lines.extend(
        [
            "",
            "**Pilar activation telemetry:**",
            f"- prob_source_counts: {src_counts}",
            f"- pss_status_counts: {pss_counts}",
        ]
    )
    if rz_counts:
        lines.append(f"- radial_zone_counts: {rz_counts}")
    if notes:
        lines.extend(["", "**Notes:**"])
        lines.extend(f"- {note}" for note in notes)
    if errors:
        lines.extend(["", "**Errors:**"])
        lines.extend(f"- ERROR: {error}" for error in errors)
    lines.append("")
    return "\n".join(lines)


def _load_current_reports(out_dir: Path) -> List[dict]:
    reports = []
    for path in sorted(out_dir.glob("governance_target_audit_*.json")):
        with open(path, encoding="utf-8") as f:
            report = json.load(f)
        split = str(
            report.get("split") or path.stem.replace("governance_target_audit_", "")
        )
        report["split"] = split
        reports.append(report)
    reports.sort(key=lambda report: _split_sort_key(str(report.get("split", ""))))
    return reports


def _rewrite_summary_outputs(out_dir: Path) -> Tuple[Path, Path]:
    reports = _load_current_reports(out_dir)
    csv_path = out_dir / "governance_target_distribution.csv"
    md_path = out_dir / "stage3c_decision.md"

    rows = []
    for report in reports:
        rows.extend(
            _distribution_rows(str(report.get("split", "")), report["aggregation"])
        )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    sections = []
    for report in reports:
        section_agg = dict(report["aggregation"])
        section_agg["sdca_proxy_audit"] = report.get("sdca_proxy_audit", {})
        sections.append(
            _render_decision_section(
                str(report.get("split", "")),
                section_agg,
                report.get("gate") == "PASS",
                report.get("gate_notes", []),
                report.get("gate_errors", []),
            )
        )
    header = (
        "# Stage 3C Governance Target Audit Summary\n\n"
        "> Regenerated from the latest `governance_target_audit_*.json` files in this directory. "
        "Rerunning a split replaces its JSON and refreshes this summary without appending stale history.\n\n"
    )
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(header)
        if sections:
            f.write("\n".join(sections))

    return csv_path, md_path


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Stage 3C: governance target audit before LGG training."
    )
    parser.add_argument(
        "--annotations_dir",
        required=True,
        help="Directory of GT Pascal VOC XML files for this split.",
    )
    parser.add_argument(
        "--cache_dir",
        required=True,
        help="Primary detector cache dir (DetectorBox JSON, from export_detector_predictions.py). Current implementation audits one cache at a time.",
    )
    parser.add_argument(
        "--secondary_cache_dir",
        default=None,
        help="Optional secondary detector cache dir (same split) for canonical multi-source SAS.",
    )
    parser.add_argument(
        "--sas_match_iou",
        type=float,
        default=0.15,
        help="IoU threshold for primary-secondary matching when attaching multi-source heads (default: 0.15).",
    )
    parser.add_argument(
        "--coverage_score_min",
        type=float,
        default=0.10,
        help=(
            "Minimum primary-box score used to compute score-filtered "
            "multisource coverage (default: 0.10)."
        ),
    )
    parser.add_argument(
        "--min_multisource_coverage",
        type=float,
        default=0.50,
        help=(
            "Minimum score-filtered multisource coverage required by canonical "
            "SAS gate (default: 0.50)."
        ),
    )
    parser.add_argument(
        "--allow_single_source_sas",
        action="store_true",
        help="Allow legacy single-source/proxy SAS. Mainline should keep this OFF.",
    )
    parser.add_argument(
        "--kg_dir",
        default="outputs/stage3_audit",
        help="Stage 3A KG artifacts dir (sdm_norm_clinical_prior.json, compat_scores.json).",
    )
    parser.add_argument(
        "--split",
        default="valid",
        choices=["train", "valid", "test"],
        help="Dataset split (default: valid).",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=0,
        help="Max images to process (0 = all).",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/stage3c_audit",
        help="Output directory (default: outputs/stage3c_audit).",
    )
    parser.add_argument(
        "--require_clean_output_dir",
        action="store_true",
        help="Fail when output_dir already contains files (prevents old-artifact mixing).",
    )
    parser.add_argument(
        "--pss_dir",
        default=None,
        help="(Pilar 2, gap G2) Dir with TTA pss_values.json. When set, real PSS "
        "replaces the make_placeholder_pss fallback.",
    )
    parser.add_argument(
        "--anatomy_provider",
        default="approximate",
        choices=["approximate", "radial_zone", "external_pretrained"],
        help="(Pilar 3) Anatomy backend: 'approximate' (legacy APS neutral), "
        "'radial_zone' (centre-based APS), or 'external_pretrained' "
        "(checkpoint-backed OD/fovea provider).",
    )
    parser.add_argument(
        "--anatomy_quality_mode",
        default="relaxed",
        choices=["relaxed", "strict"],
        help="Anatomy quality-claim mode. strict requires external_pretrained and "
        "fails the run when runtime falls back to approximate.",
    )
    parser.add_argument(
        "--anatomy_checkpoint",
        default=None,
        help="Required when --anatomy_provider external_pretrained. "
        "Path to ONNX/Torch checkpoint for OD segmentation/localization.",
    )
    parser.add_argument(
        "--anatomy_device",
        default="cpu",
        help="Device for external anatomy model (e.g. cpu, cuda).",
    )
    parser.add_argument(
        "--anatomy_input_size",
        type=int,
        default=512,
        help="Input resize for external anatomy model (default: 512).",
    )
    parser.add_argument(
        "--anatomy_mask_threshold",
        type=float,
        default=0.35,
        help="Mask threshold for external anatomy OD extraction (default: 0.35).",
    )
    parser.add_argument(
        "--anatomy_conf_moderate",
        type=float,
        default=0.55,
        help="External anatomy confidence threshold for od_confidence='moderate'.",
    )
    parser.add_argument(
        "--anatomy_conf_high",
        type=float,
        default=0.80,
        help="External anatomy confidence threshold for od_confidence='high'.",
    )
    parser.add_argument(
        "--anatomy_min_mask_area",
        type=int,
        default=64,
        help="Minimum OD mask area (in resized-mask pixels) for non-low confidence.",
    )
    parser.add_argument(
        "--radial_compat_prior",
        default="outputs/stage3_audit/zone_compat_prior_radial.json",
        help="(Pilar 3) Optional override JSON for radial-zone × class compat. "
        "When absent, Stage 3A compat_scores.json is used if it already contains "
        "all required radial pairs.",
    )
    parser.add_argument(
        "--allow_historical_paths",
        action="store_true",
        help="Allow suspicious historical/artefact paths (default: fail-fast).",
    )
    args = parser.parse_args()

    if not (0.0 <= args.coverage_score_min <= 1.0):
        print("[Stage3C] ERROR: --coverage_score_min must be in [0,1]")
        sys.exit(2)
    if not (0.0 <= args.min_multisource_coverage <= 1.0):
        print("[Stage3C] ERROR: --min_multisource_coverage must be in [0,1]")
        sys.exit(2)
    if not (0.0 < args.anatomy_mask_threshold < 1.0):
        print("[Stage3C] ERROR: --anatomy_mask_threshold must be in (0,1)")
        sys.exit(2)
    if not (0.0 < args.anatomy_conf_moderate <= 1.0):
        print("[Stage3C] ERROR: --anatomy_conf_moderate must be in (0,1]")
        sys.exit(2)
    if not (0.0 < args.anatomy_conf_high <= 1.0):
        print("[Stage3C] ERROR: --anatomy_conf_high must be in (0,1]")
        sys.exit(2)
    if args.anatomy_conf_high < args.anatomy_conf_moderate:
        print("[Stage3C] ERROR: --anatomy_conf_high must be >= --anatomy_conf_moderate")
        sys.exit(2)
    if args.anatomy_input_size <= 0:
        print("[Stage3C] ERROR: --anatomy_input_size must be > 0")
        sys.exit(2)
    if args.anatomy_min_mask_area <= 0:
        print("[Stage3C] ERROR: --anatomy_min_mask_area must be > 0")
        sys.exit(2)
    if args.anatomy_provider == "external_pretrained":
        if not args.anatomy_checkpoint:
            print(
                "[Stage3C] ERROR: --anatomy_checkpoint is required when "
                "--anatomy_provider external_pretrained"
            )
            sys.exit(2)
        if not Path(args.anatomy_checkpoint).exists():
            print(
                f"[Stage3C] ERROR: anatomy checkpoint not found: "
                f"{args.anatomy_checkpoint}"
            )
            sys.exit(2)
    if (
        args.anatomy_quality_mode == "strict"
        and args.anatomy_provider != "external_pretrained"
    ):
        print(
            "[Stage3C] ERROR: --anatomy_quality_mode strict requires "
            "--anatomy_provider external_pretrained"
        )
        sys.exit(2)

    ann_dir = Path(args.annotations_dir)
    cache_dir_input = Path(args.cache_dir)
    secondary_cache_input = (
        Path(args.secondary_cache_dir) if args.secondary_cache_dir else None
    )
    try:
        _assert_split_path_isolation(
            split=args.split,
            annotations_dir=ann_dir,
            cache_dir=cache_dir_input,
            secondary_cache_dir=secondary_cache_input,
        )
    except RuntimeError as exc:
        print(f"[Stage3C] ERROR: split-isolation check failed: {exc}")
        sys.exit(2)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.require_clean_output_dir and any(out_dir.iterdir()):
        print(
            f"[Stage3C] ERROR: output_dir is not empty while "
            f"--require_clean_output_dir is set: {out_dir}"
        )
        sys.exit(2)
    kg_dir = Path(args.kg_dir)

    path_warnings: List[str] = []
    for label, path in [
        ("annotations_dir", ann_dir),
        ("cache_dir", cache_dir_input),
        ("secondary_cache_dir", secondary_cache_input),
        ("kg_dir", kg_dir),
        ("output_dir", out_dir),
    ]:
        marker = _detect_suspicious_path(path)
        if marker is None:
            continue
        msg = f"{label} looks historical/suspicious ({marker}): {path}"
        if args.allow_historical_paths:
            print(f"[Stage3C] WARN: {msg} (allowed by --allow_historical_paths)")
            path_warnings.append(msg)
        else:
            print(
                f"[Stage3C] ERROR: {msg} " "(set --allow_historical_paths to override)"
            )
            sys.exit(2)

    out_dir_norm = _norm_path(out_dir)
    if (
        out_dir.resolve() != _CANONICAL_STAGE3C_OUTPUT_DIR.resolve()
        and "/outputs/point123_" not in out_dir_norm
    ):
        msg = (
            f"Non-canonical output_dir detected: {out_dir}. "
            f"Canonical default is {_CANONICAL_STAGE3C_OUTPUT_DIR}."
        )
        print(f"[Stage3C] WARN: {msg}")
        path_warnings.append(msg)
    elif out_dir.resolve() != _CANONICAL_STAGE3C_OUTPUT_DIR.resolve():
        print(
            f"[Stage3C] INFO: output_dir is a Point123 run artifact root: {out_dir}"
        )

    print(f"[Stage3C] === Phase 19N Stage 3C Governance Target Audit ===")
    print(f"[Stage3C] split={args.split}  annotations_dir={ann_dir}")
    print(f"[Stage3C] IoU thresholds: TP>={_IOU_TP}  WEAK>={_IOU_WEAK}")
    print(
        f"[Stage3C] SDM_CLOSE_THRESHOLD={_SDM_CLOSE_THRESHOLD}  SCORE_LOW_CONF={_SCORE_LOW_CONF}"
    )
    require_multisource_sas = not args.allow_single_source_sas

    # ── Load KG ─────────────────────────────────────────────────────────────
    sdm_path = kg_dir / "sdm_norm_clinical_prior.json"
    compat_path = kg_dir / "compat_scores.json"
    for p in (sdm_path, compat_path):
        if not p.exists():
            print(f"[Stage3C] ERROR: {p} not found. Run audit_stage3a_kg.py first.")
            sys.exit(1)

    sdm_norm = _load_sdm_norm(sdm_path)
    compat_scores = _load_compat_scores(compat_path)
    print(
        f"[Stage3C] Loaded: sdm_norm ({len(sdm_norm)} pairs)  compat_scores ({len(compat_scores)} pairs)"
    )

    # ── Load GT annotations ──────────────────────────────────────────────────
    if not ann_dir.exists():
        print(f"[Stage3C] ERROR: annotations_dir not found: {ann_dir}")
        sys.exit(1)
    gt_by_stem = _load_gt_annotations(ann_dir)
    total_gt = sum(len(v) for v in gt_by_stem.values())
    print(f"[Stage3C] Loaded GT: {len(gt_by_stem)} images  {total_gt} GT boxes")

    # ── Load detector cache ──────────────────────────────────────────────────
    cache_path = Path(args.cache_dir)
    if not cache_path.exists():
        print(f"[Stage3C] ERROR: cache_dir not found: {cache_path}")
        sys.exit(1)
    stems = sorted([f.stem for f in cache_path.glob("*.json")])
    if args.max_images and args.max_images > 0:
        stems = stems[: args.max_images]
    print(f"[Stage3C] Detector cache: {len(stems)} images")
    all_preds = load_cached_predictions(stems, cache_dir=str(cache_path))

    multisource_attach_summary = None
    if args.secondary_cache_dir:
        sec_cache_path = Path(args.secondary_cache_dir)
        if not sec_cache_path.exists():
            print(f"[Stage3C] ERROR: secondary_cache_dir not found: {sec_cache_path}")
            sys.exit(1)
        secondary_preds = load_cached_predictions(stems, cache_dir=str(sec_cache_path))
        multisource_attach_summary = _attach_multisource_heads(
            all_preds,
            secondary_preds,
            iou_threshold=args.sas_match_iou,
            coverage_score_min=args.coverage_score_min,
        )
        print(
            "[Stage3C] Multi-source SAS attach: "
            f"matched={multisource_attach_summary['matched_boxes']}/"
            f"{multisource_attach_summary['total_boxes']} "
            f"({100.0*multisource_attach_summary['matched_coverage']:.1f}%) "
            f"@IoU>={args.sas_match_iou:.2f}"
        )
        print(
            "[Stage3C] Multi-source SAS high-conf: "
            f"matched={multisource_attach_summary['matched_high_conf_boxes']}/"
            f"{multisource_attach_summary['total_high_conf_boxes']} "
            f"({100.0*multisource_attach_summary['matched_high_conf_coverage']:.1f}%) "
            f"@score>={args.coverage_score_min:.2f}"
        )
    elif require_multisource_sas:
        print(
            "[Stage3C] ERROR: canonical SAS requires multi-source head_predictions. "
            "Provide --secondary_cache_dir or use --allow_single_source_sas for legacy proxy mode."
        )
        sys.exit(2)

    # ── PSS loader (Pilar 2) ─────────────────────────────────────────────────
    pss_lookup: Optional[Dict[str, dict]] = None
    pss_source_json: Optional[str] = None
    if args.pss_dir:
        pss_path = Path(args.pss_dir) / "pss_values.json"
        pss_source_json = str(pss_path.resolve())
        if pss_path.exists():
            with open(pss_path, encoding="utf-8") as f:
                pss_lookup = json.load(f)
            print(
                f"[Stage3C] Loaded real PSS: {len(pss_lookup)} values from {pss_path}"
            )
        else:
            print(
                f"[Stage3C] WARN: pss_dir set but {pss_path} missing — using placeholder"
            )

    # ── Radial compat prior (Pilar 3) ────────────────────────────────────────
    if args.anatomy_provider == "radial_zone":
        rcp_path = Path(args.radial_compat_prior)
        if rcp_path.exists():
            with open(rcp_path, encoding="utf-8") as f:
                rcp_raw = json.load(f)
            n_merged = 0
            for k, v in rcp_raw.items():
                if k.startswith("_") or "__" not in k:
                    continue
                a, b = k.split("__", 1)
                compat_scores[(a, b)] = float(v)
                n_merged += 1
            print(
                f"[Stage3C] Merged {n_merged} radial compat entries into compat_scores"
            )
        else:
            missing_pairs = _missing_radial_pairs(compat_scores)
            if missing_pairs:
                print(f"[Stage3C] ERROR: radial_compat_prior not found: {rcp_path}")
                print(
                    "[Stage3C]   compat_scores.json is also missing required radial pairs: "
                    + ", ".join(missing_pairs[:6])
                    + (" ..." if len(missing_pairs) > 6 else "")
                )
                sys.exit(1)
            print(
                f"[Stage3C] WARN: radial_compat_prior not found: {rcp_path} — "
                "using radial compat already present in compat_scores.json"
            )

    # ── AnatomyProvider ──────────────────────────────────────────────────────
    if args.anatomy_provider == "radial_zone":
        provider = RadialZoneAnatomyProvider()
    elif args.anatomy_provider == "external_pretrained":
        provider = ExternalPretrainedProvider(
            checkpoint_path=args.anatomy_checkpoint,
            device=args.anatomy_device,
            input_size=args.anatomy_input_size,
            mask_threshold=args.anatomy_mask_threshold,
            moderate_confidence_threshold=args.anatomy_conf_moderate,
            high_confidence_threshold=args.anatomy_conf_high,
            min_mask_area=args.anatomy_min_mask_area,
        )
    else:
        provider = ApproximateAnatomyProvider(auto_scale_dd=True)
    print(f"[Stage3C] Provider: {provider.provider_name}")
    # Derive images dir from annotations_dir: data/{ds}/xml/{split} -> data/{ds}/images/{split}
    img_dir = str(Path(args.annotations_dir).parent.parent / "images" / args.split)

    # ── Process all images ───────────────────────────────────────────────────
    all_records: List[dict] = []
    n_no_gt = 0
    n_no_pred = 0

    for stem in stems:
        pred_boxes = all_preds.get(stem, [])
        gt_boxes = gt_by_stem.get(stem, [])
        if not pred_boxes:
            n_no_pred += 1
            continue
        if not gt_boxes:
            n_no_gt += 1

        img_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            p = os.path.join(img_dir, stem + ext)
            if os.path.exists(p):
                img_path = p
                break

        recs = _process_image(
            stem,
            pred_boxes,
            gt_boxes,
            provider,
            sdm_norm,
            compat_scores,
            img_path,
            pss_lookup=pss_lookup,
        )
        all_records.extend(recs)

    print(
        f"[Stage3C] Records: {len(all_records)} total boxes  "
        f"({n_no_pred} images no-pred, {n_no_gt} images no-GT-annot)"
    )

    # ── Aggregate ────────────────────────────────────────────────────────────
    agg = _aggregate(all_records, coverage_score_min=args.coverage_score_min)

    # ── Print summary ────────────────────────────────────────────────────────
    cat_order = [
        "TP_EXACT",
        "CLINICALLY_CLOSE_MISMATCH",
        "UNSAFE_MISMATCH",
        "LOCALIZATION_WEAK",
        "BACKGROUND_FP",
        "LOW_CONF_CANDIDATE",
    ]

    print(f"\n[Stage3C] ══════ GOVERNANCE TARGET DISTRIBUTION ({args.split}) ══════")
    print(
        f"  {'Category':<30} {'Count':>7} {'%':>6}  {'avgSAS':>7} {'avgSDCA':>8} "
        f"{'avgConfCost':>12} {'Action'}"
    )
    for cat in cat_order:
        s = agg["by_category"].get(cat, {"count": 0, "pct": 0.0})
        print(
            f"  {cat:<30} {s.get('count', 0):>7} {s.get('pct', 0.0):>5.1f}%  "
            f"{s.get('avg_sas') or '   -':>7}  "
            f"{s.get('avg_sdca') or '    -':>7}  "
            f"{s.get('avg_semantic_confusion_cost') or '         -':>11}  "
            f"{_CATEGORY_ACTION.get(cat, '')}"
        )
    print(f"\n[Stage3C]  SDCA by predicted class (evidence signal):")
    for cls in ["MA", "HE", "Hard_EX", "Soft_EX"]:
        info = agg["sdca_by_pred_class"].get(cls, {"n": 0, "mean_sdca": None})
        print(f"    {cls:10s}: mean_SDCA={info['mean_sdca'] or '-':>7}  n={info['n']}")

    print(
        f"\n[Stage3C]  SDCA proxy audit (current single-head detector bridge — paper-honest reporting):"
    )
    print(f"  {'Class':<10} {'tie_rate':>10} {'maj==pred':>10} {'runner_up_dist'}")
    for cls in ["MA", "HE", "Hard_EX", "Soft_EX"]:
        px = agg.get("sdca_proxy_audit", {}).get(cls)
        if px:
            print(
                f"  {cls:<10} {px['tie_rate']:>9.1%} "
                f"{px['majority_matches_pred_rate']:>9.1%}  "
                f"{px['runner_up_distribution']}"
            )
    proxy = agg.get("sdca_proxy_audit", {})
    high_tie = bool(proxy) and all(
        p.get("tie_rate", 0.0) > 0.95 for p in proxy.values()
    )
    if high_tie:
        print(
            f"  NOTE: tie_rate~100% confirms ensemble_probs are synthetic (single-head detector bridge)."
        )
    else:
        print(
            f"  NOTE: tie_rate is not saturated; exported probabilities include non-synthetic variation."
        )
    print(
        f"  NOTE: maj==pred<100% means low-confidence detections shift majority_class."
    )
    print(f"\n[Stage3C]  Pilar activation telemetry:")
    print(f"    prob_source_counts: {agg.get('prob_source_counts', {})}")
    print(f"    pss_status_counts:  {agg.get('pss_status_counts', {})}")
    print(f"    source_count_hist:  {agg.get('source_count_hist', {})}")
    print(
        f"    multisource_coverage: {100.0*agg.get('multisource_coverage', 0.0):.1f}%  "
        f"sas_fallback: {100.0*agg.get('sas_fallback_coverage', 0.0):.1f}%"
    )
    print(
        f"    multisource_high_conf_coverage: {100.0*agg.get('multisource_high_conf_coverage', 0.0):.1f}% "
        f"@score>={args.coverage_score_min:.2f}"
    )
    print(
        f"    external_fallback_images: {agg.get('anatomy_external_fallback_images', 0)} "
        f"({args.anatomy_quality_mode} mode)"
    )
    if path_warnings:
        print(f"    path_warnings: {len(path_warnings)}")
    if agg.get("radial_zone_counts"):
        print(f"    radial_zone_counts: {agg.get('radial_zone_counts', {})}")
    print(f"  These findings should be acknowledged in the paper SDCA description.")
    print(f"[Stage3C] ═══════════════════════════════════════════════════════")

    # ── Gate ─────────────────────────────────────────────────────────────────
    gate_pass_core, errors, notes = _gate_check(
        agg,
        require_multisource_sas=require_multisource_sas,
        min_multisource_coverage=args.min_multisource_coverage,
        coverage_score_min=args.coverage_score_min,
    )
    strict_anatomy_gate = True
    if args.anatomy_quality_mode == "strict":
        strict_anatomy_gate = agg.get("anatomy_external_fallback_images", 0) == 0
        if not strict_anatomy_gate:
            errors.append(
                "Strict anatomy quality gate failed: external_pretrained "
                "fell back to approximate on one or more images"
            )
    gate_pass = gate_pass_core and strict_anatomy_gate and (len(errors) == 0)
    print(f"\n[Stage3C] GATE: {'PASS' if gate_pass else 'FAIL'}")
    for e in errors:
        print(f"[Stage3C]   ERROR: {e}")
    for n in notes:
        print(f"[Stage3C]   {n}")

    # ── Save outputs ─────────────────────────────────────────────────────────
    run_provenance = {
        "repo_root": str(Path(_root).resolve()),
        "script_path": str(Path(__file__).resolve()),
        "split": args.split,
        "annotations_dir": str(ann_dir.resolve()),
        "cache_dir": str(cache_dir_input.resolve()),
        "secondary_cache_dir": (
            str(secondary_cache_input.resolve()) if secondary_cache_input else None
        ),
        "kg_dir": str(kg_dir.resolve()),
        "output_dir": str(out_dir.resolve()),
        "anatomy_provider_mode": args.anatomy_provider,
        "anatomy_quality_mode": args.anatomy_quality_mode,
        "anatomy_checkpoint": (
            str(Path(args.anatomy_checkpoint).resolve())
            if args.anatomy_checkpoint
            else None
        ),
        "pss_dir": str(Path(args.pss_dir).resolve()) if args.pss_dir else None,
        "pss_source_json": pss_source_json,
        "compare_baseline_source": str(
            (out_dir / f"governance_features_{args.split}.csv").resolve()
        ),
        "allow_historical_paths": bool(args.allow_historical_paths),
        "path_warnings": path_warnings,
    }

    report = {
        "generated_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "stage": "3C",
        "split": args.split,
        "n_images_with_preds": len(stems) - n_no_pred,
        "n_images_no_gt": n_no_gt,
        "n_images_no_pred": n_no_pred,
        "sdm_source": "severity_transition_icdr",
        "thresholds": {
            "iou_tp": _IOU_TP,
            "iou_weak": _IOU_WEAK,
            "sdm_close": _SDM_CLOSE_THRESHOLD,
            "score_low_conf": _SCORE_LOW_CONF,
        },
        "aggregation": {k: v for k, v in agg.items() if k != "sdca_proxy_audit"},
        "sdca_proxy_audit": agg.get("sdca_proxy_audit", {}),
        "prob_source_counts": agg.get("prob_source_counts", {}),
        "pss_status_counts": agg.get("pss_status_counts", {}),
        "radial_zone_counts": agg.get("radial_zone_counts", {}),
        "anatomy": {
            "provider_mode": args.anatomy_provider,
            "quality_mode": args.anatomy_quality_mode,
            "external_fallback_boxes": agg.get("anatomy_external_fallback_boxes", 0),
            "external_fallback_images": agg.get("anatomy_external_fallback_images", 0),
            "provider_name_counts": agg.get("anatomy_provider_name_counts", {}),
            "provider_method_counts": agg.get("anatomy_provider_method_counts", {}),
            "strict_anatomy_gate": strict_anatomy_gate,
        },
        "sas_mode": {
            "require_multisource_sas": require_multisource_sas,
            "secondary_cache_dir": args.secondary_cache_dir,
            "attach_summary": multisource_attach_summary,
            "coverage_score_min": args.coverage_score_min,
            "min_multisource_coverage": args.min_multisource_coverage,
        },
        "gate": "PASS" if gate_pass else "FAIL",
        "gate_errors": errors,
        "gate_notes": notes,
        "path_warnings": path_warnings,
        "provenance": run_provenance,
    }
    json_path = out_dir / f"governance_target_audit_{args.split}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[Stage3C] Report JSON: {json_path}")

    csv_path, md_path = _rewrite_summary_outputs(out_dir)
    print(f"[Stage3C] CSV: {csv_path}")

    print(f"[Stage3C] Decision doc: {md_path}")

    features_path = out_dir / f"governance_features_{args.split}.csv"
    _write_features_csv(
        all_records,
        features_path,
        split=args.split,
        provenance=run_provenance,
    )
    print(
        f"[Stage3C] Features CSV (for Stage 3D): {features_path}  ({len(all_records)} rows)"
    )

    if gate_pass:
        print(
            f"\n[Stage3C] Stage 3C PASS ({args.split}). Governance targets well-distributed."
            f"\n  After both valid + test pass, proceed to Stage 3D:"
            f"\n  Design LGG gate with baselines B0..B4 (headline compare on B4)."
            f"\n  B0: score-threshold only"
            f"\n  B1: SAS-only gate"
            f"\n  B2: SAS + SDCA"
            f"\n  B3: SAS + SDCA + APS + PSS (evidence-only full vector)"
            f"\n  B4: pred_score + SAS + SDCA + APS + PSS (practical headline)"
        )

    sys.exit(0 if gate_pass else 1)


if __name__ == "__main__":
    main()
