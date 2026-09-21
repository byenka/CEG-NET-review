"""online_evidence.py — training-time evidence builder for Mode 2 closed-loop.

Builds a per-detection [SAS, SDCA, APS, PSS] evidence vector and tier
supervision target during Mode 2 detector training, so LGG can be called
inside the training loop and GateLoss can backpropagate into the KG encoder
and LGG parameters.

REV-7 (2026-06-02) — Sprint 1+2 evidence-faithful bridge patch:
  • build_proposal_records now reads real Faster R-CNN class probabilities
    from det["class_probs"] when present (Sprint 1 — R1 bridge).
  • records carry "class_probs_vec" (canonical 4-dim) and "prob_source"
    ("real_roi_softmax" | "synthetic_score_residual") for telemetry.
  • records_to_tensors uses SOFT e_pred = P(class|roi) @ lesion_embeddings
    when class_probs_vec is available (architecture §C2 claim-valid path).
  • Optional secondary-detector cache (Sprint 2 — multi-source SAS) is
    consumed via load_secondary_detector_cache + build_proposal_records'
    new ``secondary_cache`` / ``image_ids`` params. When supplied, real
    cross-detector agreement SAS replaces the single-source max-softmax
    proxy and per-record sas_source is labelled "multi_source_*".
  • New helper evidence_fidelity_stats(records) emits gate (0) telemetry:
    real_prob_coverage, soft_epred_coverage, multi_source_sas_coverage,
    runnerup_unique_pairs, sdca_distinct_values, sdca_std, sas_distinct,
    sas_std, aps_std — to be written into mode2_summary.json + training
    CSV by the trainer.

Honest framing (pre REV-7 fallback path remains available):
  - When det["class_probs"] is absent, SDCA reverts to the synthetic
    bridge with deterministic canonical runner-up (4-value SDCA collapse).
    This is now flagged as NEGATIVE-ABLATION-only by the architecture
    contract (§EVIDENCE-FIDELITY CONTRACT, §C3 (0)).
  - When secondary_cache is None, SAS reverts to single-source proxy
    (= max(class_probs)). This is now flagged as PENDING-novelty by the
    architecture (§NOVELTY TABLE Multi-source SAS row).
  - PSS = 0.5 constant remains (real TTA cache integration deferred).

Tier category mapping matches Stage 3C ``_assign_target`` byte-for-byte:
  IoU>=0.5 + pred==gt          -> TP_EXACT                  -> R1 (tier=0)
  IoU>=0.5 + mismatch + close  -> CLINICALLY_CLOSE_MISMATCH -> R2 (tier=1)
  IoU>=0.5 + mismatch + unsafe -> UNSAFE_MISMATCH           -> R3 (tier=2)
  0.1<=IoU<0.5                 -> LOCALIZATION_WEAK         -> R3 (tier=2)
  IoU<0.1 + score>=0.15        -> BACKGROUND_FP             -> R4 (tier=3)
  IoU<0.1 + score<0.15         -> LOW_CONF_CANDIDATE        -> R4 (tier=3)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

# ─── Canonical mappings (must stay in sync with ontology + Stage 3C audit) ────
LESION_CLASSES: Tuple[str, ...] = ("MA", "HE", "Hard_EX", "Soft_EX")
_LESION_IDX: Dict[str, int] = {c: i for i, c in enumerate(LESION_CLASSES)}

# Faster R-CNN label index convention used in this repo:
#   class 0 = background, classes 1..4 = LESION_CLASSES in canonical order.
FRCNN_LABEL_TO_CLASS: Dict[int, str] = {
    (i + 1): name for i, name in enumerate(LESION_CLASSES)
}

# Stage 3C thresholds (kept aligned with audit_stage3c_governance_targets.py).
_IOU_TP: float = 0.5
_IOU_WEAK: float = 0.1
_SCORE_LOW_CONF: float = 0.15
_SDM_CLOSE: float = 0.3

# Multi-source SAS matching threshold (Sprint 2).
_IOU_MATCH_SECONDARY: float = 0.3

_CATEGORY_TO_TIER: Dict[str, int] = {
    "TP_EXACT": 0,
    "CLINICALLY_CLOSE_MISMATCH": 1,
    "UNSAFE_MISMATCH": 2,
    "LOCALIZATION_WEAK": 2,
    "BACKGROUND_FP": 3,
    "LOW_CONF_CANDIDATE": 3,
}


# ─── Geometry / KG helpers ────────────────────────────────────────────────────


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    union = aa + bb - inter
    return inter / union if union > 0.0 else 0.0


def _sdm_norm_lookup(c1: str, c2: str, sdm_norm: Dict) -> Optional[float]:
    if c1 == c2:
        return 0.0
    if c1 not in _LESION_IDX or c2 not in _LESION_IDX:
        return None
    a, b = (c1, c2) if _LESION_IDX[c1] < _LESION_IDX[c2] else (c2, c1)
    entry = sdm_norm.get((a, b))
    if entry is None:
        return None
    try:
        return float(entry["norm"])
    except (KeyError, TypeError, ValueError):
        return None


def load_compat_scores(path: Path) -> Dict[Tuple[str, str], float]:
    """Load ``compat_scores.json`` produced by Stage 3A.

    Mirrors ``tools/audit_stage3c_governance_targets._load_compat_scores``.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {tuple(k.split("__", 1)): float(v) for k, v in data.items() if "__" in k}


def _synthetic_class_probs(pred_class: str, pred_score: float) -> Dict[str, float]:
    """Single-head training-time class-prob proxy (fallback when real probs absent).

    Matches the Stage 3C 'synthetic ensemble_probs' regime. Predicted class gets
    ``pred_score``; the other lesion classes split ``(1 - pred_score)`` evenly.
    Tied non-predicted probs are deterministic for downstream SDCA top-2 lookup
    (canonical runner-up by LESION_CLASSES order). REV-7 architecture flags this
    as NEGATIVE-ABLATION-only when used in a positive C3 claim path.
    """
    others = [c for c in LESION_CLASSES if c != pred_class]
    p_other = (1.0 - float(pred_score)) / max(len(others), 1)
    out: Dict[str, float] = {pred_class: float(pred_score)}
    for c in others:
        out[c] = float(p_other)
    return out


def _real_class_probs_dict(probs_row: Sequence[float]) -> Dict[str, float]:
    """Convert a canonical-order probability vector [P_MA, P_HE, P_HardEX, P_SoftEX]
    to a class-keyed dict. Caller must ensure the row is already in the order of
    LESION_CLASSES (which is what predict_with_class_probs() returns after the
    background column drop).
    """
    n = min(len(probs_row), len(LESION_CLASSES))
    return {LESION_CLASSES[i]: float(probs_row[i]) for i in range(n)}


def _probs_row_from_det(det: Dict, idx: int) -> Optional[List[float]]:
    """Extract row ``idx`` from a detection dict's ``class_probs`` field.

    Accepts torch.Tensor [N, C] or list-of-list. Returns canonical-order list of
    length len(LESION_CLASSES) or None if unavailable. The predict_with_class_probs()
    helper in src/governance/multiclass_inference.py already drops the background
    column, so columns 0..3 = MA, HE, Hard_EX, Soft_EX.
    """
    probs = det.get("class_probs")
    if probs is None:
        return None
    try:
        if hasattr(probs, "detach"):  # torch.Tensor
            if idx >= probs.shape[0]:
                return None
            row = probs[idx].detach().cpu().tolist()
        else:
            if idx >= len(probs):
                return None
            row = list(probs[idx])
        # Truncate/pad to len(LESION_CLASSES) just in case.
        if len(row) < len(LESION_CLASSES):
            return None
        return [float(row[i]) for i in range(len(LESION_CLASSES))]
    except (IndexError, TypeError, ValueError):
        return None


# ─── Secondary detector cache loader (Sprint 2 — multi-source SAS) ────────────


def load_secondary_detector_cache(
    cache_dir: Optional[Path],
) -> Dict[str, List[Dict]]:
    """Load secondary-detector (e.g., RetinaNet r1a) predictions per image_id.

    The expected layout matches tools/export_detector_predictions.py output:
    ``cache_dir / <image_id>.json`` is a JSON list of per-bbox records with
    ``detector_raw.bbox / label / label_name / score`` and optional
    ``detector_class_probs``. Missing dir or unreadable files yield an empty
    cache, falling back to single-source SAS at training time.

    Returns:
        Dict mapping image_id (str) → list of normalized records with keys
        ``bbox`` [x1, y1, x2, y2], ``label_name``, ``score``,
        ``class_probs`` (dict).
    """
    out: Dict[str, List[Dict]] = {}
    if cache_dir is None:
        return out
    p = Path(cache_dir)
    if not p.exists() or not p.is_dir():
        return out
    for fp in sorted(p.glob("*.json")):
        image_id = fp.stem
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, list):
            continue
        entries: List[Dict] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            raw = item.get("detector_raw") or item.get("r2a_raw") or {}
            bbox = raw.get("bbox")
            label_name = raw.get("label_name")
            score = raw.get("score", 0.0)
            if (
                not isinstance(bbox, (list, tuple))
                or len(bbox) != 4
                or not isinstance(label_name, str)
            ):
                continue
            entries.append(
                {
                    "bbox": [float(v) for v in bbox],
                    "label_name": label_name,
                    "score": float(score),
                    "class_probs": item.get("detector_class_probs") or {},
                }
            )
        if entries:
            out[image_id] = entries
    return out


def _multi_source_sas(
    primary_box: Sequence[float],
    primary_score: float,
    primary_label: str,
    secondary_per_image: Optional[List[Dict]],
) -> Tuple[float, str]:
    """Multi-source agreement SAS via IoU matching.

    Returns:
        (sas_value, sas_source) where sas_source ∈ {
            "single_source_proxy",          # no secondary cache available
            "multi_source_unmatched",       # secondary had boxes but no IoU>=_IOU_MATCH_SECONDARY match
            "multi_source_agreed",          # matched and label agreement
            "multi_source_conflict",        # matched but label disagreement
        }
    """
    if not secondary_per_image:
        return float(primary_score), "single_source_proxy"
    best_iou = 0.0
    best_sec: Optional[Dict] = None
    for sec in secondary_per_image:
        v = _iou(primary_box, sec["bbox"])
        if v > best_iou:
            best_iou = v
            best_sec = sec
    if best_sec is None or best_iou < _IOU_MATCH_SECONDARY:
        # Secondary had boxes but none overlap → conservative SAS halving.
        return float(primary_score) * 0.5, "multi_source_unmatched"
    sec_label = str(best_sec.get("label_name", ""))
    sec_score = float(best_sec.get("score", 0.0))
    if sec_label == primary_label:
        # Real cross-detector agreement: blend scores + IoU boost.
        sas = 0.5 * (float(primary_score) + sec_score) + 0.1 * float(best_iou)
        return float(min(1.0, sas)), "multi_source_agreed"
    # Label conflict: penalize. Keeps SAS in [0, 1].
    sas = 0.5 * max(float(primary_score), sec_score) - 0.1 * float(best_iou)
    return float(max(0.0, sas)), "multi_source_conflict"


# ─── Tier + evidence assignment ───────────────────────────────────────────────


def _assign_category(
    pred_class: str,
    pred_score: float,
    pred_box: Sequence[float],
    gt_records: List[dict],
    sdm_norm: Dict,
) -> Tuple[str, Optional[str], float]:
    best_iou = 0.0
    best_gt_class: Optional[str] = None
    for gt in gt_records:
        v = _iou(pred_box, gt["bbox"])
        if v > best_iou:
            best_iou = v
            best_gt_class = gt["class"]
    if best_iou >= _IOU_TP:
        if pred_class == best_gt_class:
            return "TP_EXACT", best_gt_class, best_iou
        sdm_v = _sdm_norm_lookup(pred_class, best_gt_class or "", sdm_norm)
        if sdm_v is None:
            sdm_v = 0.5
        return (
            "CLINICALLY_CLOSE_MISMATCH"
            if sdm_v < _SDM_CLOSE
            else "UNSAFE_MISMATCH"
        ), best_gt_class, best_iou
    if best_iou >= _IOU_WEAK:
        return "LOCALIZATION_WEAK", best_gt_class, best_iou
    if pred_score >= _SCORE_LOW_CONF:
        return "BACKGROUND_FP", None, best_iou
    return "LOW_CONF_CANDIDATE", None, best_iou


def _build_evidence_vec(
    pred_class: str,
    class_probs: Dict[str, float],
    radial_zone: str,
    sdm_norm: Dict,
    compat_scores: Dict[Tuple[str, str], float],
    pred_score: float,
    override_sas: Optional[float] = None,
) -> List[float]:
    # SAS: multi-source value if provided, else single-source proxy = max-softmax.
    if override_sas is not None:
        sas = float(override_sas)
    else:
        sas = float(max(class_probs.values())) if class_probs else float(pred_score)
    # SDCA: sdm_norm(top1, top2). Under REAL class probs this is a continuous
    # signal; under synthetic-tied probs it collapses to <=4 deterministic
    # values per predicted class (§EVIDENCE-FIDELITY CONTRACT 2026-06-01).
    sdca = 0.5
    if class_probs and len(class_probs) >= 2:
        ranked = sorted(class_probs.items(), key=lambda kv: kv[1], reverse=True)
        top1, top2 = ranked[0][0], ranked[1][0]
        sdca_v = _sdm_norm_lookup(top1, top2, sdm_norm)
        if sdca_v is not None:
            sdca = float(sdca_v)
    # APS: real compat_scores lookup.
    aps = 0.5
    if compat_scores and radial_zone:
        v = compat_scores.get((pred_class, radial_zone))
        if v is not None:
            aps = float(v)
    # PSS: placeholder (no TTA at training-time in current revision).
    pss = 0.5
    return [sas, sdca, aps, pss]


def build_proposal_records(
    detections: List[Dict],
    targets: List[Dict],
    image_sizes: List[Tuple[int, int]],
    sdm_norm: Dict,
    compat_scores: Dict[Tuple[str, str], float],
    label_to_class: Optional[Dict[int, str]] = None,
    *,
    image_ids: Optional[List[str]] = None,
    secondary_cache: Optional[Dict[str, List[Dict]]] = None,
) -> List[Dict]:
    """Build per-detection records with evidence + tier index.

    Args:
        detections:    list (length B) of Faster R-CNN eval-mode outputs. Each
                       item is a dict with ``boxes`` [N,4], ``labels`` [N],
                       ``scores`` [N], and (optionally, when produced by
                       ``predict_with_class_probs``) ``class_probs`` [N, 4].
        targets:       list (length B) of training targets:
                       ``{"boxes": Tensor[M,4], "labels": Tensor[M]}``.
        image_sizes:   ``[(H, W), ...]`` per image.
        sdm_norm:      Stage 3A SDM lookup table.
        compat_scores: Stage 3A compat scores (lesion × zone).
        label_to_class: optional FRCNN label-idx → class-name map. Defaults to
                        :data:`FRCNN_LABEL_TO_CLASS`.
        image_ids:     optional list (length B) of image_id strings used to
                       look up the secondary-detector cache. Required when
                       ``secondary_cache`` is provided.
        secondary_cache: optional dict mapping image_id → list of secondary-
                       detector records (see :func:`load_secondary_detector_cache`).
                       Enables multi-source SAS at training time. When None or
                       empty for a given image, SAS falls back to single-source
                       max-softmax proxy and ``sas_source`` is set accordingly.

    Returns:
        Flat list of detection records. Each record has::

            {
              "image_idx":      int,
              "image_id":       Optional[str],
              "pred_class":     str,
              "majority_class": str,
              "pred_score":     float,
              "evidence_vec":   [sas, sdca, aps, pss],
              "class_probs_vec": [P_MA, P_HE, P_Hard_EX, P_Soft_EX],
              "prob_source":    "real_roi_softmax" | "synthetic_score_residual",
              "sas_source":     "multi_source_agreed" | "multi_source_conflict"
                                | "multi_source_unmatched" | "single_source_proxy",
              "tier_index":     int,  # 0..3
              "category":       str,
              "best_iou":       float,
              "radial_zone":    str,
            }
    """
    # Lazy import to keep module import free of heavy package side-effects.
    try:
        from src.governance.anatomy_provider import RadialZoneAnatomyProvider
    except ModuleNotFoundError:
        from lgovnet.src.governance.anatomy_provider import (  # type: ignore
            RadialZoneAnatomyProvider,
        )

    lbl_map = label_to_class or FRCNN_LABEL_TO_CLASS
    records: List[Dict] = []
    for img_idx, (det, tgt, (H, W)) in enumerate(
        zip(detections, targets, image_sizes)
    ):
        boxes = det.get("boxes")
        labels = det.get("labels")
        scores = det.get("scores")
        if boxes is None or labels is None or scores is None:
            continue
        boxes_l = boxes.detach().cpu().tolist()
        labels_l = labels.detach().cpu().tolist()
        scores_l = scores.detach().cpu().tolist()

        gt_boxes = tgt.get("boxes")
        gt_labels = tgt.get("labels")
        gt_records: List[Dict] = []
        if gt_boxes is not None and gt_labels is not None:
            for gb, gl in zip(
                gt_boxes.detach().cpu().tolist(),
                gt_labels.detach().cpu().tolist(),
            ):
                name = lbl_map.get(int(gl))
                if name is not None:
                    gt_records.append({"class": name, "bbox": gb})

        # Secondary detector entries for this image (Sprint 2 multi-source SAS).
        image_id_str: Optional[str] = None
        sec_for_image: Optional[List[Dict]] = None
        if image_ids is not None and img_idx < len(image_ids):
            image_id_str = str(image_ids[img_idx])
        if secondary_cache and image_id_str is not None:
            sec_for_image = secondary_cache.get(image_id_str)

        for i, (box, lab, sc) in enumerate(zip(boxes_l, labels_l, scores_l)):
            pred_class = lbl_map.get(int(lab))
            if pred_class is None:
                continue
            category, _gt_class, best_iou = _assign_category(
                pred_class, float(sc), box, gt_records, sdm_norm
            )
            tier_idx = _CATEGORY_TO_TIER.get(category, 3)
            cx = (float(box[0]) + float(box[2])) / 2.0
            cy = (float(box[1]) + float(box[3])) / 2.0
            zone = RadialZoneAnatomyProvider.assign_radial_zone(cx, cy, W, H)

            # Sprint 1: real class-probability bridge.
            probs_row = _probs_row_from_det(det, i)
            if probs_row is not None:
                cls_probs = _real_class_probs_dict(probs_row)
                prob_source = "real_roi_softmax"
                class_probs_vec = probs_row
            else:
                cls_probs = _synthetic_class_probs(pred_class, float(sc))
                prob_source = "synthetic_score_residual"
                class_probs_vec = [cls_probs.get(c, 0.0) for c in LESION_CLASSES]

            # Sprint 2: multi-source SAS.
            if sec_for_image is not None:
                sas_value, sas_source = _multi_source_sas(
                    box, float(sc), pred_class, sec_for_image
                )
            else:
                sas_value = (
                    float(max(cls_probs.values())) if cls_probs else float(sc)
                )
                sas_source = "single_source_proxy"

            ev = _build_evidence_vec(
                pred_class,
                cls_probs,
                zone,
                sdm_norm,
                compat_scores,
                float(sc),
                override_sas=sas_value,
            )
            records.append(
                {
                    "image_idx": img_idx,
                    "image_id": image_id_str,
                    "pred_class": pred_class,
                    "majority_class": pred_class,
                    "pred_score": float(sc),
                    "evidence_vec": ev,
                    "class_probs_vec": class_probs_vec,
                    "prob_source": prob_source,
                    "sas_source": sas_source,
                    "tier_index": int(tier_idx),
                    "category": category,
                    "best_iou": float(best_iou),
                    "radial_zone": zone,
                }
            )
    return records


def records_to_tensors(
    records: List[Dict],
    lesion_embeddings: torch.Tensor,
    device: torch.device,
) -> Optional[Dict[str, torch.Tensor]]:
    """Convert proposal records to LGG-ready tensors.

    REV-7: when every record carries a full ``class_probs_vec`` of length
    len(LESION_CLASSES), e_pred is computed SOFT as
    ``prob_matrix @ lesion_embeddings`` (architecture §C2 claim-valid path).
    Otherwise it falls back to the hard ``index_select(majority_class)``
    lookup. ``e_pred_source`` is added to the returned dict so the trainer
    can include it in mode2_summary telemetry.

    Args:
        records: output of :func:`build_proposal_records`.
        lesion_embeddings: Tensor[len(LESION_CLASSES), d] — KG lesion
            embeddings (used to build e_pred).
        device: tensor device.

    Returns:
        ``None`` when ``records`` is empty. Otherwise::

            {
              "evidence_vectors": Tensor[N, 4]      (no grad — supervision side)
              "e_pred":           Tensor[N, d]      (grad through lesion_emb)
              "e_pred_source":    str               ("soft" | "hard")
              "gate_input":       Tensor[N, 4 + d]
              "tier_targets":     Tensor[N] int64
              "majority_idx":     Tensor[N] int64
            }
    """
    if not records:
        return None
    ev = torch.tensor(
        [r["evidence_vec"] for r in records],
        dtype=torch.float32,
        device=device,
    )
    tier_targets = torch.tensor(
        [r["tier_index"] for r in records],
        dtype=torch.int64,
        device=device,
    )
    majority_idx = torch.tensor(
        [_LESION_IDX.get(r["majority_class"], 0) for r in records],
        dtype=torch.int64,
        device=device,
    )

    # Sprint 1: soft e_pred when class_probs_vec is uniformly available.
    n_classes = len(LESION_CLASSES)
    has_full_probs = all(
        isinstance(r.get("class_probs_vec"), (list, tuple))
        and len(r["class_probs_vec"]) == n_classes
        for r in records
    )
    if has_full_probs:
        prob_matrix = torch.tensor(
            [r["class_probs_vec"] for r in records],
            dtype=torch.float32,
            device=device,
        )
        # [N, d] = [N, C] @ [C, d]; gradient flows through lesion_embeddings.
        e_pred = prob_matrix @ lesion_embeddings
        e_pred_source = "soft"
    else:
        e_pred = lesion_embeddings.index_select(0, majority_idx)
        e_pred_source = "hard"

    gate_input = torch.cat([ev, e_pred], dim=-1)
    return {
        "evidence_vectors": ev,
        "e_pred": e_pred,
        "e_pred_source": e_pred_source,
        "gate_input": gate_input,
        "tier_targets": tier_targets,
        "majority_idx": majority_idx,
    }


def summarise_records(records: List[Dict]) -> Dict[str, int]:
    """Return tier and category counts for telemetry."""
    out = {"n_total": len(records),
           "n_r1": 0, "n_r2": 0, "n_r3": 0, "n_r4": 0}
    for r in records:
        t = int(r.get("tier_index", 3))
        if t == 0:
            out["n_r1"] += 1
        elif t == 1:
            out["n_r2"] += 1
        elif t == 2:
            out["n_r3"] += 1
        else:
            out["n_r4"] += 1
    return out


def _std_float(xs: List[float]) -> float:
    n = len(xs)
    if n == 0:
        return 0.0
    m = sum(xs) / n
    return (sum((x - m) ** 2 for x in xs) / n) ** 0.5


def evidence_fidelity_stats(records: List[Dict]) -> Dict[str, float]:
    """Gate (0) telemetry for the architecture's evidence-fidelity contract.

    Returns a flat dict suitable for direct merge into ``mode2_summary.json``::

        {
          "ev_n_records":               int,
          "ev_real_prob_coverage":      float in [0, 1],
          "ev_soft_epred_coverage":     float in [0, 1],
          "ev_multi_source_sas_coverage": float in [0, 1],
          "ev_n_multi_source_agreed":   int,
          "ev_n_multi_source_conflict": int,
          "ev_runnerup_unique_pairs":   int,
          "ev_sdca_distinct_values":    int,
          "ev_sdca_std":                float,
          "ev_sas_distinct_values":     int,
          "ev_sas_std":                 float,
          "ev_aps_distinct_values":     int,
          "ev_aps_std":                 float,
          "ev_pss_std":                 float,
        }

    Architecture gate (0) thresholds (CONDITIONAL CLAIM RULE):
      ev_real_prob_coverage       >= 0.95
      ev_soft_epred_coverage      >= 0.95
      ev_multi_source_sas_coverage >= 0.80   (Stage 5B)
      ev_runnerup_unique_pairs    >= 5
      ev_sas_distinct_values      >= 20      (Stage 5B)
    """
    n = len(records)
    base: Dict[str, float] = {
        "ev_n_records": 0,
        "ev_real_prob_coverage": 0.0,
        "ev_soft_epred_coverage": 0.0,
        "ev_multi_source_sas_coverage": 0.0,
        "ev_n_multi_source_agreed": 0,
        "ev_n_multi_source_conflict": 0,
        "ev_runnerup_unique_pairs": 0,
        "ev_sdca_distinct_values": 0,
        "ev_sdca_std": 0.0,
        "ev_sas_distinct_values": 0,
        "ev_sas_std": 0.0,
        "ev_aps_distinct_values": 0,
        "ev_aps_std": 0.0,
        "ev_pss_std": 0.0,
    }
    if n == 0:
        return base
    n_real = 0
    n_soft = 0
    n_multi_sas = 0
    n_multi_agreed = 0
    n_multi_conflict = 0
    runnerup_pairs = set()
    sas_vals: List[float] = []
    sdca_vals: List[float] = []
    aps_vals: List[float] = []
    pss_vals: List[float] = []
    for r in records:
        if r.get("prob_source") == "real_roi_softmax":
            n_real += 1
        cv = r.get("class_probs_vec")
        if isinstance(cv, (list, tuple)) and len(cv) == len(LESION_CLASSES):
            n_soft += 1
            # Runner-up = second-highest class by probability.
            ranked = sorted(range(len(cv)), key=lambda j: -float(cv[j]))
            if len(ranked) >= 2:
                runnerup_pairs.add(
                    (LESION_CLASSES[ranked[0]], LESION_CLASSES[ranked[1]])
                )
        ss = r.get("sas_source", "")
        if ss.startswith("multi_source"):
            n_multi_sas += 1
            if ss == "multi_source_agreed":
                n_multi_agreed += 1
            elif ss == "multi_source_conflict":
                n_multi_conflict += 1
        ev = r.get("evidence_vec")
        if isinstance(ev, (list, tuple)) and len(ev) >= 4:
            sas_vals.append(float(ev[0]))
            sdca_vals.append(float(ev[1]))
            aps_vals.append(float(ev[2]))
            pss_vals.append(float(ev[3]))

    base["ev_n_records"] = int(n)
    base["ev_real_prob_coverage"] = float(n_real / n)
    base["ev_soft_epred_coverage"] = float(n_soft / n)
    base["ev_multi_source_sas_coverage"] = float(n_multi_sas / n)
    base["ev_n_multi_source_agreed"] = int(n_multi_agreed)
    base["ev_n_multi_source_conflict"] = int(n_multi_conflict)
    base["ev_runnerup_unique_pairs"] = int(len(runnerup_pairs))
    base["ev_sdca_distinct_values"] = int(
        len({round(x, 4) for x in sdca_vals})
    )
    base["ev_sdca_std"] = float(_std_float(sdca_vals))
    base["ev_sas_distinct_values"] = int(
        len({round(x, 4) for x in sas_vals})
    )
    base["ev_sas_std"] = float(_std_float(sas_vals))
    base["ev_aps_distinct_values"] = int(
        len({round(x, 4) for x in aps_vals})
    )
    base["ev_aps_std"] = float(_std_float(aps_vals))
    base["ev_pss_std"] = float(_std_float(pss_vals))
    return base
