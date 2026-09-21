"""clinical_metrics.py — clinical decision metrics for MedIA defensibility.

Beyond detection mAP, MedIA reviewers expect clinically-relevant metrics that
characterize the practical decision support quality of a lesion detector.
This module computes, per lesion class and aggregated:

- Sensitivity @ fixed specificity (default 0.95) — operating point clinical use
- AUROC (per-class, from box-level scores) — discrimination quality
- AUPRC (per-class) — useful for highly imbalanced classes (MA, Soft_EX)
- Brier score — probabilistic calibration of box-level scores
- Expected Calibration Error (ECE, 10 bins) — calibration of confidence
- Recall @ FP_per_image budget (e.g. 1, 3, 5 FP/img) — clinical FP budget

Inputs are computed from a list of per-image (boxes, labels, scores) and
matching list of (gt_boxes, gt_labels). A detection is "correct" if it has
IoU >= match_iou with any unmatched GT of the same class.

The module is dependency-light: numpy only (no sklearn) so it can be embedded
in eval scripts without changing the environment contract. Tensors are
converted to numpy before computation.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

LESION_CLASSES = ("MA", "HE", "Hard_EX", "Soft_EX")
FRCNN_LABEL_TO_CLASS: Dict[int, str] = {
    (i + 1): name for i, name in enumerate(LESION_CLASSES)
}


# ─── geometry ─────────────────────────────────────────────────────────────────


def _box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between [Na,4] and [Nb,4] xyxy boxes."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    a_x1 = a[:, None, 0]
    a_y1 = a[:, None, 1]
    a_x2 = a[:, None, 2]
    a_y2 = a[:, None, 3]
    b_x1 = b[None, :, 0]
    b_y1 = b[None, :, 1]
    b_x2 = b[None, :, 2]
    b_y2 = b[None, :, 3]
    inter_x1 = np.maximum(a_x1, b_x1)
    inter_y1 = np.maximum(a_y1, b_y1)
    inter_x2 = np.minimum(a_x2, b_x2)
    inter_y2 = np.minimum(a_y2, b_y2)
    inter_w = np.clip(inter_x2 - inter_x1, a_min=0.0, a_max=None)
    inter_h = np.clip(inter_y2 - inter_y1, a_min=0.0, a_max=None)
    inter = inter_w * inter_h
    area_a = np.clip(a_x2 - a_x1, 0.0, None) * np.clip(a_y2 - a_y1, 0.0, None)
    area_b = np.clip(b_x2 - b_x1, 0.0, None) * np.clip(b_y2 - b_y1, 0.0, None)
    union = area_a + area_b - inter
    return np.where(
        union > 0, inter / np.maximum(union, 1e-9), np.zeros_like(union)
    ).astype(np.float32)


# ─── core: per-detection tp/fp labelling per class ────────────────────────────


def _per_class_tp_fp(
    detections: List[Dict],
    targets: List[Dict],
    cls_label: int,
    match_iou: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Return (scores_sorted_desc, tp_flags_sorted_desc, n_gt_total, n_images)."""
    scores_list: List[float] = []
    tp_list: List[int] = []
    n_gt_total = 0
    for det, tgt in zip(detections, targets):
        # gather predictions of this class
        d_labels = det.get("labels")
        d_scores = det.get("scores")
        d_boxes = det.get("boxes")
        if d_labels is None or d_scores is None or d_boxes is None:
            continue
        d_labels = np.asarray(d_labels)
        d_scores = np.asarray(d_scores, dtype=np.float32)
        d_boxes = np.asarray(d_boxes, dtype=np.float32)
        mask = d_labels == cls_label
        d_boxes = d_boxes[mask]
        d_scores = d_scores[mask]
        # gather ground-truths of this class
        t_labels = np.asarray(tgt.get("labels", []))
        t_boxes = np.asarray(tgt.get("boxes", []), dtype=np.float32).reshape(-1, 4)
        gt_mask = t_labels == cls_label
        t_boxes = t_boxes[gt_mask]
        n_gt_total += int(t_boxes.shape[0])
        if d_boxes.shape[0] == 0:
            continue
        # greedy match by descending score
        order = np.argsort(-d_scores)
        d_boxes = d_boxes[order]
        d_scores = d_scores[order]
        used = np.zeros(t_boxes.shape[0], dtype=bool)
        if t_boxes.shape[0] > 0:
            iou = _box_iou(d_boxes, t_boxes)
        else:
            iou = np.zeros((d_boxes.shape[0], 0), dtype=np.float32)
        for i in range(d_boxes.shape[0]):
            scores_list.append(float(d_scores[i]))
            assigned = False
            if iou.shape[1] > 0:
                # best unused GT
                best_j = -1
                best_v = match_iou
                for j in range(t_boxes.shape[0]):
                    if used[j]:
                        continue
                    if iou[i, j] >= best_v:
                        best_v = float(iou[i, j])
                        best_j = j
                if best_j >= 0:
                    used[best_j] = True
                    assigned = True
            tp_list.append(1 if assigned else 0)
    n_images = len(detections)
    scores_arr = np.asarray(scores_list, dtype=np.float32)
    tp_arr = np.asarray(tp_list, dtype=np.int32)
    order = np.argsort(-scores_arr)
    return scores_arr[order], tp_arr[order], n_gt_total, n_images


# ─── PR / ROC / AUC ───────────────────────────────────────────────────────────


def _auc_from_xy(xs: np.ndarray, ys: np.ndarray) -> float:
    if xs.size < 2:
        return 0.0
    # Trapezoidal integration with sorted x.
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]
    return float(np.trapz(ys, xs))


def per_class_auroc_auprc(
    detections: List[Dict],
    targets: List[Dict],
    cls_label: int,
    match_iou: float = 0.5,
) -> Dict[str, float]:
    scores, tp, n_gt, _ = _per_class_tp_fp(detections, targets, cls_label, match_iou)
    if scores.size == 0 or n_gt == 0:
        return {"AUROC": 0.0, "AUPRC": 0.0, "n_gt": n_gt, "n_pred": int(scores.size)}
    # ROC: x = FPR = cumFP / N_negative_proxy ; here "negatives" = all FP detections
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(1 - tp)
    n_pos = float(n_gt)
    n_neg = float(max(int(cum_fp[-1]), 1))
    tpr = cum_tp / n_pos
    fpr = cum_fp / n_neg
    # prepend (0,0) and append (1,1) on tpr/fpr for ROC
    roc_x = np.concatenate(([0.0], fpr, [1.0]))
    roc_y = np.concatenate(([0.0], tpr, [1.0]))
    auroc = _auc_from_xy(roc_x, roc_y)
    # PR
    precision = cum_tp / np.maximum(np.arange(1, scores.size + 1, dtype=np.float32), 1)
    recall = cum_tp / n_pos
    # standard "all-points" AUPRC integration
    auprc = _auc_from_xy(recall, precision)
    return {
        "AUROC": float(auroc),
        "AUPRC": float(auprc),
        "n_gt": int(n_gt),
        "n_pred": int(scores.size),
    }


def sensitivity_at_specificity(
    detections: List[Dict],
    targets: List[Dict],
    cls_label: int,
    specificity: float = 0.95,
    match_iou: float = 0.5,
) -> Dict[str, float]:
    """Sensitivity (recall) at the operating point with given specificity.

    Specificity = 1 - FPR, computed against the per-detection FP labelling.
    Approximates the clinical operating point where the system rejects 95%
    of false positives.
    """
    scores, tp, n_gt, _ = _per_class_tp_fp(detections, targets, cls_label, match_iou)
    if scores.size == 0 or n_gt == 0:
        return {"sensitivity": 0.0, "threshold": 1.0, "n_gt": n_gt}
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(1 - tp)
    n_pos = float(n_gt)
    n_neg = float(max(int(cum_fp[-1]), 1))
    fpr = cum_fp / n_neg
    sensitivity = cum_tp / n_pos
    # find smallest k where (1 - fpr[k]) >= specificity, i.e. fpr[k] <= 1 - specificity
    target_fpr = 1.0 - float(specificity)
    valid = fpr <= target_fpr
    if not np.any(valid):
        return {"sensitivity": 0.0, "threshold": float(scores[0]), "n_gt": n_gt}
    k = int(
        np.argmax(valid * (np.arange(scores.size) + 1))
    )  # last True before threshold breaks
    return {
        "sensitivity": float(sensitivity[k]),
        "threshold": float(scores[k]),
        "n_gt": int(n_gt),
    }


def recall_at_fp_per_image(
    detections: List[Dict],
    targets: List[Dict],
    cls_label: int,
    fp_per_image_budget: float = 1.0,
    match_iou: float = 0.5,
) -> Dict[str, float]:
    """Recall achieved at a clinical FP-per-image budget.

    Common operating points in DR screening: 1, 3, 5 FP/img.
    """
    scores, tp, n_gt, n_images = _per_class_tp_fp(
        detections, targets, cls_label, match_iou
    )
    if scores.size == 0 or n_gt == 0 or n_images == 0:
        return {"recall": 0.0, "threshold": 1.0, "n_gt": n_gt}
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(1 - tp)
    fp_per_img = cum_fp / float(max(n_images, 1))
    valid = fp_per_img <= float(fp_per_image_budget)
    if not np.any(valid):
        return {"recall": 0.0, "threshold": float(scores[0]), "n_gt": n_gt}
    k = int(np.argmax(valid * (np.arange(scores.size) + 1)))
    return {
        "recall": float(cum_tp[k] / float(n_gt)),
        "threshold": float(scores[k]),
        "n_gt": int(n_gt),
    }


# ─── calibration ──────────────────────────────────────────────────────────────


def calibration_metrics(
    detections: List[Dict],
    targets: List[Dict],
    cls_label: Optional[int] = None,
    match_iou: float = 0.5,
    n_bins: int = 10,
) -> Dict[str, float]:
    """Brier score + Expected Calibration Error (ECE) of detection confidences.

    If ``cls_label`` is None, aggregates over all classes (overall calibration
    of the detector). Per-detection ground truth is the TP/FP labelling.
    """
    all_scores: List[float] = []
    all_tp: List[int] = []
    for cl in (LESION_CLASSES if cls_label is None else [None]):
        cl_label = (
            FRCNN_LABEL_TO_CLASS_INV().get(cl) if cls_label is None else cls_label
        )
        if cl_label is None:
            continue
        scores, tp, _, _ = _per_class_tp_fp(detections, targets, cl_label, match_iou)
        all_scores.extend(scores.tolist())
        all_tp.extend(tp.tolist())
    if not all_scores:
        return {"brier": 0.0, "ece": 0.0, "n": 0}
    scores = np.asarray(all_scores, dtype=np.float32)
    tp = np.asarray(all_tp, dtype=np.float32)
    brier = float(np.mean((scores - tp) ** 2))
    # ECE
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = float(scores.size)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        in_bin = (scores >= lo) & (scores < hi if i < n_bins - 1 else scores <= hi)
        m = int(in_bin.sum())
        if m == 0:
            continue
        conf = float(scores[in_bin].mean())
        acc = float(tp[in_bin].mean())
        ece += (m / n) * abs(conf - acc)
    return {"brier": float(brier), "ece": float(ece), "n": int(n)}


def FRCNN_LABEL_TO_CLASS_INV() -> Dict[str, int]:
    return {v: k for k, v in FRCNN_LABEL_TO_CLASS.items()}


# ─── pair confusion rate (REV-7 Mode 2A primary metric family) ────────────────


def confusion_pair_rate(
    detections: List[Dict],
    targets: List[Dict],
    src_label: int,
    dst_label: int,
    match_iou: float = 0.5,
) -> Dict[str, float]:
    """Confusion rate from src_label predictions toward dst_label ground truth.

    For each prediction of class src_label, find the best-IoU GT box (across all
    classes). When IoU >= match_iou and the matched GT class == dst_label, count
    a confusion. The rate is confusions / total_src_predictions_with_match.

    Returns:
        {
          "src_class":           int (1-indexed lesion class id),
          "dst_class":           int,
          "n_pred_src":          int,    # total predictions of src_label
          "n_pred_src_matched":  int,    # predictions that had any GT match at IoU>=match_iou
          "n_confusions":        int,    # of which were matched to a dst_label GT
          "confusion_rate":      float,  # n_confusions / max(n_pred_src_matched, 1)
          "match_iou":           float,
        }
    """
    n_pred_src = 0
    n_pred_src_matched = 0
    n_confusions = 0
    for det, tgt in zip(detections, targets):
        d_labels = det.get("labels")
        d_boxes = det.get("boxes")
        if d_labels is None or d_boxes is None:
            continue
        d_labels = np.asarray(d_labels)
        d_boxes = np.asarray(d_boxes, dtype=np.float32)
        mask = d_labels == src_label
        d_boxes_src = d_boxes[mask]
        n_pred_src += int(d_boxes_src.shape[0])
        if d_boxes_src.shape[0] == 0:
            continue
        t_labels = np.asarray(tgt.get("labels", []))
        t_boxes = np.asarray(
            tgt.get("boxes", []), dtype=np.float32
        ).reshape(-1, 4)
        if t_boxes.shape[0] == 0:
            continue
        iou = _box_iou(d_boxes_src, t_boxes)
        for i in range(d_boxes_src.shape[0]):
            best_j = int(np.argmax(iou[i]))
            best_v = float(iou[i, best_j])
            if best_v < match_iou:
                continue
            n_pred_src_matched += 1
            gt_class = int(t_labels[best_j])
            if gt_class == dst_label:
                n_confusions += 1
    rate = float(n_confusions) / float(max(n_pred_src_matched, 1))
    return {
        "src_class": int(src_label),
        "dst_class": int(dst_label),
        "n_pred_src": int(n_pred_src),
        "n_pred_src_matched": int(n_pred_src_matched),
        "n_confusions": int(n_confusions),
        "confusion_rate": float(rate),
        "match_iou": float(match_iou),
    }


def confusion_pair_rates_all(
    detections: List[Dict],
    targets: List[Dict],
    match_iou: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """Compute confusion rates for ALL ordered pairs of lesion classes.

    Returns a dict keyed by ``"<src>__<dst>"`` (e.g. ``"MA__HE"``) and the
    symmetric paired-direction key ``"<a>_X_<b>"`` for pair-level reporting.
    Also exposes the ``ma_he_confusion_rate`` and its reverse + averaged pair
    rate, which are the architecture §C3 gate (vi) primary candidates for
    Mode 2A.
    """
    inv = FRCNN_LABEL_TO_CLASS_INV()
    by_pair: Dict[str, Dict[str, float]] = {}
    for src_name in LESION_CLASSES:
        src_label = inv[src_name]
        for dst_name in LESION_CLASSES:
            if dst_name == src_name:
                continue
            dst_label = inv[dst_name]
            r = confusion_pair_rate(
                detections, targets, src_label, dst_label, match_iou
            )
            by_pair[f"{src_name}__{dst_name}"] = r

    # Symmetric paired-direction summary for MA<->HE specifically (Mode 2A
    # primary metric family per architecture §C3 gate (vi)). Also expose
    # all paired-direction MA<->X rates for diagnostic completeness.
    def _pair_avg(a: str, b: str) -> Dict[str, float]:
        ra = by_pair.get(f"{a}__{b}", {"confusion_rate": 0.0, "n_confusions": 0,
                                       "n_pred_src_matched": 0})
        rb = by_pair.get(f"{b}__{a}", {"confusion_rate": 0.0, "n_confusions": 0,
                                       "n_pred_src_matched": 0})
        # Average rate (matched-prediction-weighted, falls back to plain mean).
        wa = max(int(ra.get("n_pred_src_matched", 0)), 0)
        wb = max(int(rb.get("n_pred_src_matched", 0)), 0)
        total_w = wa + wb
        if total_w > 0:
            avg = (
                float(ra.get("confusion_rate", 0.0)) * wa
                + float(rb.get("confusion_rate", 0.0)) * wb
            ) / float(total_w)
        else:
            avg = (
                float(ra.get("confusion_rate", 0.0))
                + float(rb.get("confusion_rate", 0.0))
            ) / 2.0
        return {
            "pair": f"{a}<->{b}",
            "forward_rate": float(ra.get("confusion_rate", 0.0)),
            "reverse_rate": float(rb.get("confusion_rate", 0.0)),
            "weighted_avg_rate": float(avg),
            "n_forward_matched": int(ra.get("n_pred_src_matched", 0)),
            "n_reverse_matched": int(rb.get("n_pred_src_matched", 0)),
        }

    paired_summary = {
        "MA_X_HE": _pair_avg("MA", "HE"),
        "MA_X_Hard_EX": _pair_avg("MA", "Hard_EX"),
        "MA_X_Soft_EX": _pair_avg("MA", "Soft_EX"),
        "HE_X_Hard_EX": _pair_avg("HE", "Hard_EX"),
        "HE_X_Soft_EX": _pair_avg("HE", "Soft_EX"),
        "Hard_EX_X_Soft_EX": _pair_avg("Hard_EX", "Soft_EX"),
    }
    return {"by_pair": by_pair, "paired_summary": paired_summary}


# ─── top-level aggregator ─────────────────────────────────────────────────────


def compute_all(
    detections: List[Dict],
    targets: List[Dict],
    specificity_targets: Sequence[float] = (0.90, 0.95, 0.99),
    fp_budgets: Sequence[float] = (1.0, 3.0, 5.0),
    match_iou: float = 0.5,
) -> Dict[str, object]:
    """Compute per-class + aggregated clinical metrics dictionary.

    The result is a JSON-serialisable nested dict.
    """
    per_class: Dict[str, Dict[str, object]] = {}
    for cls_name in LESION_CLASSES:
        cl_label = FRCNN_LABEL_TO_CLASS_INV()[cls_name]
        auc = per_class_auroc_auprc(detections, targets, cl_label, match_iou)
        sens = {
            f"sens_at_spec_{s:.2f}": sensitivity_at_specificity(
                detections, targets, cl_label, specificity=s, match_iou=match_iou
            )
            for s in specificity_targets
        }
        rfp = {
            f"recall_at_fp_{b:.0f}": recall_at_fp_per_image(
                detections,
                targets,
                cl_label,
                fp_per_image_budget=b,
                match_iou=match_iou,
            )
            for b in fp_budgets
        }
        cal = calibration_metrics(
            detections, targets, cls_label=cl_label, match_iou=match_iou
        )
        per_class[cls_name] = {
            "auroc_auprc": auc,
            "sensitivity_at_specificity": sens,
            "recall_at_fp_budget": rfp,
            "calibration": cal,
        }
    overall_cal = calibration_metrics(
        detections, targets, cls_label=None, match_iou=match_iou
    )
    # Macro-averages
    macro = {
        "macro_auroc": float(
            np.mean([per_class[c]["auroc_auprc"]["AUROC"] for c in LESION_CLASSES])
        ),
        "macro_auprc": float(
            np.mean([per_class[c]["auroc_auprc"]["AUPRC"] for c in LESION_CLASSES])
        ),
        "macro_sens_at_spec_0p95": float(
            np.mean(
                [
                    per_class[c]["sensitivity_at_specificity"]["sens_at_spec_0.95"][
                        "sensitivity"
                    ]
                    for c in LESION_CLASSES
                ]
            )
        ),
        "macro_recall_at_fp_1": float(
            np.mean(
                [
                    per_class[c]["recall_at_fp_budget"]["recall_at_fp_1"]["recall"]
                    for c in LESION_CLASSES
                ]
            )
        ),
        "overall_brier": float(overall_cal["brier"]),
        "overall_ece": float(overall_cal["ece"]),
    }
    # REV-7: confusion-pair rates (architecture §C3 gate (vi) primary metric
    # family for Mode 2A). MA<->HE is the headline pair; others are reported
    # for completeness so reviewers can verify the gain is not driven by a
    # different clinically distant pair.
    pair_rates = confusion_pair_rates_all(detections, targets, match_iou=match_iou)
    ma_he = pair_rates["paired_summary"].get("MA_X_HE", {})
    macro["ma_he_confusion_rate"] = float(ma_he.get("weighted_avg_rate", 0.0))
    macro["ma_he_forward_rate"] = float(ma_he.get("forward_rate", 0.0))
    macro["ma_he_reverse_rate"] = float(ma_he.get("reverse_rate", 0.0))
    # MA AP50 is already in per_class.MA but expose at macro for selection logic.
    ma_block = per_class.get("MA", {}).get("auroc_auprc", {})
    macro["ma_auroc"] = float(ma_block.get("AUROC", 0.0))
    macro["ma_auprc"] = float(ma_block.get("AUPRC", 0.0))
    return {
        "per_class": per_class,
        "macro": macro,
        "confusion_pair_rates": pair_rates,
        "match_iou": float(match_iou),
        "specificity_targets": list(specificity_targets),
        "fp_budgets": list(fp_budgets),
    }
