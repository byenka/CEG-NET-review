#!/usr/bin/env python
"""
audit_detector_stage1a.py
Phase 19N — Stage 1A: Detector Readiness Audit

Audits FCOS, RetinaNet, or Faster R-CNN checkpoints against the Stage 1
quality floor and writes a model-aware bottleneck summary.

Outputs (all in --output_dir):
  stage1a_readiness_audit.md   — human-readable bottleneck summary
  bbox_stats_per_class.csv     — GT + pred box area distribution per class
  convergence_analysis.csv     — per-epoch training/val metrics
  stage1a_raw.json             — full raw numbers (for diffing later)

Usage:
    python tools/audit_detector_stage1a.py \
        --checkpoint outputs/detector_stage1/stage1c_fasterrcnn_r5b_ext25/fasterrcnn_best.pt \
        --model_type fasterrcnn \
        --output_dir outputs/detector_eval/r5b_baseline \
        --split valid \
        --score_thresh 0.02 \
        --detections_per_img 200
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

if int(os.environ.get("OMP_NUM_THREADS", "0") or "0") <= 0:
    os.environ["OMP_NUM_THREADS"] = "1"
if "MKL_NUM_THREADS" not in os.environ:
    os.environ["MKL_NUM_THREADS"] = "1"

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.join(_here, "..")
if _root not in sys.path:
    sys.path.insert(0, _root)

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from src.data.coco_adapter import (
    DRDetectionDataset,
    LABEL_NAMES,
    NUM_CLASSES,
    collate_fn,
)

_DATA_ROOT = os.path.join(_root, "data")
_DEFAULT_CKPT = os.path.join(
    _root,
    "outputs",
    "detector_stage1",
    "stage1c_fasterrcnn_r5b_ext25",
    "fasterrcnn_best.pt",
)
_DEFAULT_LOG = os.path.join(
    _root,
    "outputs",
    "detector_stage1",
    "stage1c_fasterrcnn_r5b_ext25",
    "training_log.csv",
)

LABEL_MAP: Dict[int, str] = {i + 1: n for i, n in enumerate(LABEL_NAMES)}
CLASS_IDS = list(LABEL_MAP.keys())
DEFAULT_SWEEP_THRESHOLDS = [0.005, 0.01, 0.02, 0.03, 0.05]

# Stage 1C acceptance thresholds (from architecture v3)
ACCEPT_MAP50_MIN = 0.05
ACCEPT_MAP50_TGT = 0.08
ACCEPT_MA_RECALL = 0.20


def _import_attach_noise_aware_fpn():
    try:
        from src.models.noise_aware_fpn import attach_noise_aware_fpn

        return attach_noise_aware_fpn
    except ImportError:
        try:
            from lgovnet.src.models.noise_aware_fpn import attach_noise_aware_fpn

            return attach_noise_aware_fpn
        except ImportError as exc:
            raise RuntimeError(
                "noiseaware_fpn checkpoints require src/models/noise_aware_fpn.py. "
                "That legacy r4a experiment is not mirrored in the active "
                "lgovnetB self-contained toolchain."
            ) from exc


def _import_lgov_fcos():
    try:
        from src.models.lgov_fcos_core import LGovFCOS

        return LGovFCOS
    except ImportError:
        try:
            from lgovnet.src.models.lgov_fcos_core import LGovFCOS

            return LGovFCOS
        except ImportError as exc:
            raise RuntimeError(
                "model_type=fcos requires src/models/lgov_fcos_core.py. "
                "FCOS is a historical baseline and is not part of the active "
                "lgovnetB self-contained detector path."
            ) from exc


def _model_display_name(model_type: str) -> str:
    return {
        "fcos": "FCOS",
        "retinanet": "RetinaNet",
        "fasterrcnn": "Faster R-CNN",
    }.get(model_type, model_type)


def _pick_first_key(row: Dict, *candidates: str) -> Optional[str]:
    for key in candidates:
        if key in row:
            return key
    return None


def _resolve_data_root(data_root: str, split: str) -> str:
    img_dir = os.path.join(data_root, "images", split)
    xml_dir = os.path.join(data_root, "xml", split)
    if os.path.isdir(img_dir) and os.path.isdir(xml_dir):
        return data_root

    if not os.path.isdir(data_root):
        return data_root

    candidates = []
    for name in sorted(os.listdir(data_root)):
        candidate = os.path.join(data_root, name)
        if not os.path.isdir(candidate):
            continue
        cand_img = os.path.join(candidate, "images", split)
        cand_xml = os.path.join(candidate, "xml", split)
        if os.path.isdir(cand_img) and os.path.isdir(cand_xml):
            candidates.append(candidate)

    if len(candidates) == 1:
        resolved = candidates[0]
        print(f"[Stage1A] Auto-resolved data_root: {data_root} -> {resolved}")
        return resolved

    if len(candidates) > 1:
        rel = [os.path.relpath(path, _root) for path in candidates]
        raise FileNotFoundError(
            "Dataset root does not directly contain images/<split> and xml/<split>. "
            f"Found multiple dataset candidates under {data_root}: {', '.join(rel)}. "
            f"Pass --data_root to one dataset directory, e.g. {rel[0]}."
        )

    return data_root


def _to_float(value, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return default
        return float(value.item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ──────────────────────────────────────────────────────────────────────────────
# IoU helper
# ──────────────────────────────────────────────────────────────────────────────


def _iou_matrix(a: Tensor, b: Tensor) -> Tensor:
    if a.shape[0] == 0 or b.shape[0] == 0:
        return torch.zeros(a.shape[0], b.shape[0])
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0.0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=1e-6)


# ──────────────────────────────────────────────────────────────────────────────
# mAP backend
# ──────────────────────────────────────────────────────────────────────────────

try:
    from torchmetrics.detection.mean_ap import MeanAveragePrecision

    _TM_OK = True
except ImportError:
    _TM_OK = False
    warnings.warn("torchmetrics not available — using manual AP50 fallback.")

_TM_RUNTIME_WARNED = False


def _compute_ap50_manual(preds, targets) -> Dict[int, float]:
    ap: Dict[int, float] = {}
    for cls_id in CLASS_IDS:
        det_scores, det_tp = [], []
        n_gt = 0
        for pred, tgt in zip(preds, targets):
            gt_m = tgt["labels"] == cls_id
            gt_b = tgt["boxes"][gt_m].cpu().float()
            n_gt += int(gt_m.sum())

            pd_m = pred["labels"] == cls_id
            if not pd_m.any():
                continue
            pb = pred["boxes"][pd_m].cpu().float()
            ps = pred["scores"][pd_m].cpu().float()

            if gt_b.shape[0] == 0:
                det_scores += ps.tolist()
                det_tp += [0] * pb.shape[0]
                continue

            iou = _iou_matrix(pb, gt_b)
            matched = torch.full((gt_b.shape[0],), False)
            order = ps.argsort(descending=True)
            for idx in order.tolist():
                best_iou, best_j = iou[idx].max(dim=0)
                best_iou = float(best_iou)
                best_j = int(best_j)
                det_scores.append(float(ps[idx]))
                if best_iou >= 0.50 and not matched[best_j]:
                    matched[best_j] = True
                    det_tp.append(1)
                else:
                    det_tp.append(0)

        if n_gt == 0:
            ap[cls_id] = float("nan")
            continue

        order = sorted(range(len(det_scores)), key=lambda i: -det_scores[i])
        tp_cum = 0
        fp_cum = 0
        precs, recs = [], []
        for i in order:
            if det_tp[i]:
                tp_cum += 1
            else:
                fp_cum += 1
            precs.append(tp_cum / (tp_cum + fp_cum))
            recs.append(tp_cum / n_gt)

        # 11-point interpolation
        ap_val = 0.0
        for t in [i / 10 for i in range(11)]:
            prec_at = [p for p, r in zip(precs, recs) if r >= t]
            ap_val += (max(prec_at) if prec_at else 0.0) / 11
        ap[cls_id] = ap_val
    return ap


# ──────────────────────────────────────────────────────────────────────────────
# Recall helper  (TP / n_GT at iou>=0.50)
# ──────────────────────────────────────────────────────────────────────────────


def _compute_recall_per_class(preds, targets, iou_thr=0.50) -> Dict[int, float]:
    recall: Dict[int, float] = {}
    for cls_id in CLASS_IDS:
        n_gt = tp = 0
        for pred, tgt in zip(preds, targets):
            gt_m = tgt["labels"] == cls_id
            gt_b = tgt["boxes"][gt_m].cpu().float()
            n_gt += int(gt_m.sum())
            if gt_b.shape[0] == 0:
                continue

            pd_m = pred["labels"] == cls_id
            if not pd_m.any():
                continue
            pb = pred["boxes"][pd_m].cpu().float()
            iou = _iou_matrix(pb, gt_b)
            matched = torch.full((gt_b.shape[0],), False)
            for j in range(gt_b.shape[0]):
                col = iou[:, j]
                if col.max() >= iou_thr and not matched[j]:
                    matched[j] = True
                    tp += 1
        recall[cls_id] = tp / max(n_gt, 1)
    return recall


def _count_targets_per_class(targets: List[Dict]) -> Dict[int, int]:
    counts = {c: 0 for c in CLASS_IDS}
    for tgt in targets:
        labels = tgt.get("labels", torch.zeros(0, dtype=torch.long))
        for cls_id in CLASS_IDS:
            counts[cls_id] += int((labels == cls_id).sum().item())
    return counts


def _compute_false_positive_stats(
    preds: List[Dict],
    targets: List[Dict],
    iou_thr: float = 0.50,
) -> Dict:
    false_positive_per_class = {c: 0 for c in CLASS_IDS}
    for cls_id in CLASS_IDS:
        fp = 0
        for pred, tgt in zip(preds, targets):
            gt_m = tgt["labels"] == cls_id
            gt_b = tgt["boxes"][gt_m].cpu().float()

            pd_m = pred["labels"] == cls_id
            if not pd_m.any():
                continue
            pb = pred["boxes"][pd_m].cpu().float()
            ps = pred["scores"][pd_m].cpu().float()

            if gt_b.shape[0] == 0:
                fp += int(pb.shape[0])
                continue

            iou = _iou_matrix(pb, gt_b)
            matched = torch.full((gt_b.shape[0],), False)
            for idx in ps.argsort(descending=True).tolist():
                best_iou, best_j = iou[idx].max(dim=0)
                if float(best_iou) >= iou_thr and not matched[int(best_j)]:
                    matched[int(best_j)] = True
                else:
                    fp += 1
        false_positive_per_class[cls_id] = fp

    total_fp = sum(false_positive_per_class.values())
    n_gt_per_class = _count_targets_per_class(targets)
    total_gt = sum(n_gt_per_class.values())
    return {
        "false_positive_per_class": false_positive_per_class,
        "false_positive_count": total_fp,
        "false_positive_pressure": total_fp / max(total_gt, 1),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Box area statistics helper
# ──────────────────────────────────────────────────────────────────────────────


def _area_stats(areas: List[float]) -> Dict:
    if not areas:
        return {
            "n": 0,
            "mean": 0,
            "median": 0,
            "p10": 0,
            "p90": 0,
            "min": 0,
            "max": 0,
            "pct_tiny": 0,
            "pct_small": 0,
        }
    s = sorted(areas)
    n = len(s)
    pct = lambda q: s[max(0, int(q * n) - 1)]
    tiny = sum(1 for a in areas if a < 32**2) / n  # <1024 px²
    small = sum(1 for a in areas if a < 64**2) / n  # <4096 px²
    return {
        "n": n,
        "mean": round(sum(areas) / n, 1),
        "median": round(pct(0.50), 1),
        "p10": round(pct(0.10), 1),
        "p90": round(pct(0.90), 1),
        "min": round(s[0], 1),
        "max": round(s[-1], 1),
        "pct_tiny": round(tiny, 4),
        "pct_small": round(small, 4),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Training log analysis
# ──────────────────────────────────────────────────────────────────────────────


def _parse_log(log_csv: str) -> Optional[List[Dict]]:
    if not log_csv or not os.path.isfile(log_csv):
        return None
    rows = []
    with open(log_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: _try_float(v) for k, v in row.items()})
    return rows


def _try_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def _convergence_analysis(rows: List[Dict]) -> Dict:
    if not rows:
        return {}
    epochs = [int(r.get("epoch", i + 1)) for i, r in enumerate(rows)]
    first = rows[0]
    val_key = _pick_first_key(first, "val_loss_total", "val_loss")
    train_key = _pick_first_key(first, "loss_total", "train_loss")
    map_key = _pick_first_key(first, "val_mAP50", "mAP_50")
    dets_key = _pick_first_key(first, "val_mean_dets_per_image", "mean_dets_per_image")
    score_key = _pick_first_key(first, "val_max_score", "max_score")

    val_losses = [r.get(val_key) for r in rows] if val_key else []
    train_losses = [r.get(train_key) for r in rows] if train_key else []
    map_values = [r.get(map_key) for r in rows] if map_key else []
    row_by_epoch = {int(r.get("epoch", i + 1)): r for i, r in enumerate(rows)}

    valid_val = [(e, v) for e, v in zip(epochs, val_losses) if v is not None]
    best_epoch, best_val = (
        min(valid_val, key=lambda x: x[1]) if valid_val else (None, None)
    )
    valid_map = [(e, v) for e, v in zip(epochs, map_values) if v is not None]
    best_map_epoch, best_map = (
        max(valid_map, key=lambda x: x[1]) if valid_map else (None, None)
    )

    # Plateau detection: val_loss in last 3 epochs vs best
    last3 = [v for _, v in valid_val[-3:]]
    plateau = (
        (max(last3) - min(last3)) / (abs(best_val) + 1e-9) < 0.02
        if len(last3) >= 3
        else False
    )

    # Overfitting signal: val_loss rising while train_loss falling at last epoch
    overfit = False
    if len(rows) >= 3 and len(valid_val) >= 2:
        delta_val = valid_val[-1][1] - valid_val[-2][1] if len(valid_val) >= 2 else 0
        tl = [
            r.get(train_key)
            for r in rows[-2:]
            if train_key and r.get(train_key) is not None
        ]
        delta_train = (tl[-1] - tl[0]) if len(tl) == 2 else 0
        overfit = delta_val > 0.02 and delta_train < 0

    map_at_best_loss = None
    if best_epoch is not None and map_key is not None:
        map_at_best_loss = row_by_epoch.get(best_epoch, {}).get(map_key)

    loss_at_best_map = None
    if best_map_epoch is not None and val_key is not None:
        loss_at_best_map = row_by_epoch.get(best_map_epoch, {}).get(val_key)

    return {
        "total_epochs": len(rows),
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val, 6) if best_val is not None else None,
        "last_val_loss": round(valid_val[-1][1], 6) if valid_val else None,
        "best_map_epoch": best_map_epoch,
        "best_val_mAP50": round(best_map, 6) if best_map is not None else None,
        "map_at_best_loss_epoch": (
            round(map_at_best_loss, 6) if map_at_best_loss is not None else None
        ),
        "val_loss_at_best_map_epoch": (
            round(loss_at_best_map, 6) if loss_at_best_map is not None else None
        ),
        "plateau": plateau,
        "overfit_signal": overfit,
        "log_keys": {
            "train_loss": train_key,
            "val_loss": val_key,
            "val_mAP50": map_key,
            "mean_dets_per_image": dets_key,
            "max_score": score_key,
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# Inference pass
# ──────────────────────────────────────────────────────────────────────────────


def _load_fcos_model(
    ckpt: dict, score_thresh: float, device: torch.device
) -> torch.nn.Module:
    """Load LGovFCOS from checkpoint dict."""
    LGovFCOS = _import_lgov_fcos()
    cfg = ckpt.get("config", {})
    use_p2 = ckpt.get("use_p2", cfg.get("use_p2", True))
    reg_mask_mode = ckpt.get("reg_mask_mode", cfg.get("reg_mask_mode", "strict_inside"))
    reg_relax_margin_ratio = float(
        ckpt.get("reg_relax_margin_ratio", cfg.get("reg_relax_margin_ratio", 0.2))
    )
    model = LGovFCOS(
        num_classes=NUM_CLASSES,
        pretrained_backbone=False,
        matcher_mode=cfg.get("matcher_mode", "lesion_expand"),
        cls_loss_normalizer=cfg.get("cls_loss_normalizer", "pos"),
        cls_loss_weight=cfg.get("cls_loss_weight", 1.0),
        prior_prob=cfg.get("prior_prob", 0.01),
        score_thresh=score_thresh,
        detections_per_img=300,
        use_p2=use_p2,
        reg_mask_mode=reg_mask_mode,
        reg_relax_margin_ratio=reg_relax_margin_ratio,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval()


def _load_retinanet_model(ckpt: dict, device: torch.device) -> torch.nn.Module:
    """Load torchvision RetinaNet from checkpoint dict."""
    try:
        from torchvision.models.detection import retinanet_resnet50_fpn
        from torchvision.models.detection.retinanet import RetinaNetClassificationHead

        try:
            from torchvision.models.detection import RetinaNet_ResNet50_FPN_Weights

            model = retinanet_resnet50_fpn(weights=None)
        except ImportError:
            model = retinanet_resnet50_fpn(pretrained=False)
    except ImportError as e:
        raise ImportError(f"torchvision required for retinanet model_type: {e}")

    cfg = ckpt.get("config", {})
    # Default NUM_CLASSES+1: train script uses 1-indexed labels {1..4}, so head needs 5 nodes
    num_classes = cfg.get("num_classes", NUM_CLASSES + 1)
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=256,
        num_anchors=num_anchors,
        num_classes=num_classes,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    print(
        f"[Stage1A] RetinaNet loaded: num_classes={num_classes} (labels 1-{num_classes-1})"
    )
    return model.to(device).eval()


def _load_fasterrcnn_model(ckpt: dict, device: torch.device) -> torch.nn.Module:
    """Load torchvision Faster R-CNN/FPN from checkpoint dict."""
    try:
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        from torchvision.models.detection.rpn import AnchorGenerator

        try:
            from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

            _weights_v2 = True
        except ImportError:
            _weights_v2 = False
    except ImportError as e:
        raise ImportError(f"torchvision required for fasterrcnn model_type: {e}")

    cfg = ckpt.get("config", {})
    num_classes = cfg.get("num_classes", NUM_CLASSES + 1)
    detections_per_img = int(cfg.get("detections_per_img", 100))
    rpn_pre_nms_top_n_train = int(cfg.get("rpn_pre_nms_top_n_train", 2000))
    rpn_post_nms_top_n_train = int(cfg.get("rpn_post_nms_top_n_train", 2000))
    rpn_pre_nms_top_n_test = int(cfg.get("rpn_pre_nms_top_n_test", 1000))
    rpn_post_nms_top_n_test = int(cfg.get("rpn_post_nms_top_n_test", 1000))

    # Restore small-object anchors if saved in checkpoint config
    anchor_sizes = cfg.get("anchor_sizes", ((8,), (16,), (32,), (64,), (128,)))
    anchor_ratios = cfg.get("anchor_ratios", ((0.5, 1.0, 2.0),) * 5)
    anchor_gen = AnchorGenerator(
        sizes=tuple(tuple(s) for s in anchor_sizes),
        aspect_ratios=tuple(tuple(r) for r in anchor_ratios),
    )
    model_kwargs = {
        "rpn_anchor_generator": anchor_gen,
        "box_detections_per_img": detections_per_img,
        "rpn_pre_nms_top_n_train": rpn_pre_nms_top_n_train,
        "rpn_post_nms_top_n_train": rpn_post_nms_top_n_train,
        "rpn_pre_nms_top_n_test": rpn_pre_nms_top_n_test,
        "rpn_post_nms_top_n_test": rpn_post_nms_top_n_test,
    }

    if _weights_v2:
        model = fasterrcnn_resnet50_fpn(weights=None, **model_kwargs)
    else:
        model = fasterrcnn_resnet50_fpn(pretrained=False, **model_kwargs)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model_variant = cfg.get("model_variant", "baseline")
    if model_variant == "noiseaware_fpn":
        attach_noise_aware_fpn = _import_attach_noise_aware_fpn()
        attach_noise_aware_fpn(
            model,
            alpha=float(cfg.get("noise_gate_alpha", 0.2)),
            hidden_channels=int(cfg.get("noise_hidden_channels", 16)),
            out_bias_init=float(cfg.get("noise_last_bias", -4.0)),
            scatter_kernel_size=int(cfg.get("noise_scatter_kernel", 15)),
        )
    model.load_state_dict(ckpt["model_state_dict"])
    print(
        f"[Stage1A] FasterRCNN loaded: num_classes={num_classes} (labels 1-{num_classes-1})"
        f"  dets_per_img={model.roi_heads.detections_per_img}"
        f"  rpn_pre(train/test)=({model.rpn._pre_nms_top_n['training']}/{model.rpn._pre_nms_top_n['testing']})"
        f"  rpn_post(train/test)=({model.rpn._post_nms_top_n['training']}/{model.rpn._post_nms_top_n['testing']})"
    )
    if model_variant == "noiseaware_fpn":
        print(
            "[Stage1A] FasterRCNN variant: Noise/Artifact-Aware FPN "
            f"(alpha={cfg.get('noise_gate_alpha', 0.2)}, hidden={cfg.get('noise_hidden_channels', 16)}, "
            f"scatter_k={cfg.get('noise_scatter_kernel', 15)})"
        )
    elif model_variant == "roi_focal_loss":
        print(
            "[Stage1A] FasterRCNN variant: ROI focal loss "
            f"(gamma={cfg.get('roi_focal_gamma', 2.0)}, "
            f"alpha_mode={cfg.get('roi_focal_alpha_mode', 'batch_balanced')})"
        )
    return model.to(device).eval()


def _run_inference(
    ckpt_path: str,
    split: str,
    score_thresh: float,
    max_images: int,
    model_type: str = "fcos",
    detections_per_img: int = 0,
    data_root: Optional[str] = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[Stage1A] device={device}  split={split}  score_thresh={score_thresh}"
        + (
            f"  detections_per_img={detections_per_img}"
            if detections_per_img > 0
            else ""
        )
    )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(
        f"[Stage1A] Checkpoint epoch={ckpt.get('epoch','?')}  val_loss={ckpt.get('val_loss','?')}"
    )

    if model_type == "retinanet":
        model = _load_retinanet_model(ckpt, device)
        # torchvision RetinaNet filters at model.score_thresh internally.
        # Without this, all thresholds in a sweep return identical results
        # because the model always uses its default (0.05) regardless of args.
        model.score_thresh = score_thresh
    elif model_type == "fasterrcnn":
        model = _load_fasterrcnn_model(ckpt, device)
        # Faster R-CNN uses roi_heads.score_thresh for post-NMS filtering.
        model.roi_heads.score_thresh = score_thresh
        # detections_per_img cap: default=100, may limit recall on dense lesion images.
        # Set explicitly only when testing cap hypothesis (--detections_per_img > 0).
        if detections_per_img > 0:
            model.roi_heads.detections_per_img = detections_per_img
            print(
                f"[Stage1A] FasterRCNN detections_per_img overridden to {detections_per_img}"
            )
        else:
            print(
                f"[Stage1A] FasterRCNN detections_per_img = {model.roi_heads.detections_per_img} (default)"
            )
    else:
        model = _load_fcos_model(ckpt, score_thresh, device)

    resolved_data_root = _resolve_data_root(data_root or _DATA_ROOT, split)
    ds = DRDetectionDataset(split=split, data_root=resolved_data_root)
    if max_images and max_images < len(ds):
        indices = list(range(max_images))
        ds = torch.utils.data.Subset(ds, indices)
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False,
    )

    all_preds, all_targets = [], []
    gt_areas: Dict[int, List[float]] = {c: [] for c in CLASS_IDS}
    pred_areas: Dict[int, List[float]] = {c: [] for c in CLASS_IDS}

    with torch.no_grad():
        for imgs, targets in loader:
            imgs = [im.to(device) for im in imgs]
            outputs = model(imgs)

            for tgt, pred in zip(targets, outputs):
                # Collect GT areas
                for cls_id in CLASS_IDS:
                    m = tgt["labels"] == cls_id
                    for box in tgt["boxes"][m]:
                        w = float(box[2] - box[0])
                        h = float(box[3] - box[1])
                        if w > 0 and h > 0:
                            gt_areas[cls_id].append(w * h)

                # Collect pred areas
                boxes = pred.get("boxes", torch.zeros(0, 4))
                labels = pred.get("labels", torch.zeros(0, dtype=torch.long))
                scores = pred.get("scores", torch.zeros(0))
                for cls_id in CLASS_IDS:
                    m = labels == cls_id
                    for box in boxes[m]:
                        w = float(box[2] - box[0])
                        h = float(box[3] - box[1])
                        if w > 0 and h > 0:
                            pred_areas[cls_id].append(w * h)

                all_preds.append(
                    {
                        "boxes": boxes.cpu(),
                        "labels": labels.cpu(),
                        "scores": scores.cpu(),
                    }
                )
                all_targets.append(
                    {
                        "boxes": tgt["boxes"].cpu(),
                        "labels": tgt["labels"].cpu(),
                    }
                )

    return all_preds, all_targets, gt_areas, pred_areas


def _count_preds_per_class(all_preds: List[Dict]) -> Dict[int, int]:
    counts = {c: 0 for c in CLASS_IDS}
    for pred in all_preds:
        labels = pred.get("labels", torch.zeros(0, dtype=torch.long))
        for cls_id in CLASS_IDS:
            counts[cls_id] += int((labels == cls_id).sum().item())
    return counts


def _compute_map_metrics(
    all_preds: List[Dict], all_targets: List[Dict]
) -> Tuple[Dict, Optional[Dict]]:
    raw = None
    if _TM_OK:
        global _TM_RUNTIME_WARNED
        try:
            metric = MeanAveragePrecision(
                box_format="xyxy",
                iou_type="bbox",
                max_detection_thresholds=[1, 10, 300],
            )
            metric.warn_on_many_detections = False
            metric.update(all_preds, all_targets)
            raw = metric.compute()
            map_metrics = {
                "backend": "torchmetrics",
                "mAP": round(_to_float(raw.get("map"), 0.0), 6),
                "mAP_50": round(_to_float(raw.get("map_50"), 0.0), 6),
                "mAP_75": round(_to_float(raw.get("map_75"), 0.0), 6),
                "mAP_small": round(_to_float(raw.get("map_small"), -1.0), 6),
                "mAP_medium": round(_to_float(raw.get("map_medium"), -1.0), 6),
                "mAP_large": round(_to_float(raw.get("map_large"), -1.0), 6),
            }
            return map_metrics, raw
        except Exception as e:
            if not _TM_RUNTIME_WARNED:
                warnings.warn(
                    f"torchmetrics MAP backend unavailable at runtime ({e}) — using manual AP50 fallback."
                )
                _TM_RUNTIME_WARNED = True

    ap50_manual = _compute_ap50_manual(all_preds, all_targets)
    valid = [v for v in ap50_manual.values() if not math.isnan(v)]
    mean_ap50 = sum(valid) / max(len(valid), 1)
    map_metrics = {
        "backend": "manual_ap50",
        "mAP_50": round(mean_ap50, 6),
    }
    return map_metrics, raw


def _compute_ap_per_class(
    all_preds: List[Dict], all_targets: List[Dict], raw: Optional[Dict]
) -> Dict[int, float]:
    if _TM_OK and raw is not None:
        raw_per_cls = raw.get("map_per_class", None)
        if (
            raw_per_cls is not None
            and isinstance(raw_per_cls, torch.Tensor)
            and raw_per_cls.ndim == 1
            and raw_per_cls.shape[0] >= len(CLASS_IDS)
        ):
            return {
                cls_id: round(float(raw_per_cls[i].item()), 6)
                for i, cls_id in enumerate(CLASS_IDS)
            }
    return _compute_ap50_manual(all_preds, all_targets)


def _evaluate_predictions(all_preds: List[Dict], all_targets: List[Dict]) -> Dict:
    map_metrics, raw = _compute_map_metrics(all_preds, all_targets)
    ap_per_class = _compute_ap_per_class(all_preds, all_targets, raw)
    recall_per_class = _compute_recall_per_class(all_preds, all_targets)
    fp_stats = _compute_false_positive_stats(all_preds, all_targets)
    n_pred_per_class = _count_preds_per_class(all_preds)
    total_preds = sum(n_pred_per_class.values())
    no_det_images = sum(
        1 for pred in all_preds if pred.get("scores", torch.zeros(0)).numel() == 0
    )
    return {
        "map_metrics": map_metrics,
        "ap_per_class": ap_per_class,
        "recall_per_class": recall_per_class,
        "n_pred_per_class": n_pred_per_class,
        "total_predictions": total_preds,
        "no_det_images": no_det_images,
        "mean_dets_per_image": total_preds / max(len(all_preds), 1),
        **fp_stats,
    }


def _filter_predictions_by_threshold(
    all_preds: List[Dict], score_thresh: float
) -> List[Dict]:
    filtered = []
    for pred in all_preds:
        scores = pred.get("scores", torch.zeros(0))
        keep = scores >= score_thresh
        filtered.append(
            {
                "boxes": pred.get("boxes", torch.zeros(0, 4))[keep],
                "labels": pred.get("labels", torch.zeros(0, dtype=torch.long))[keep],
                "scores": scores[keep],
            }
        )
    return filtered


def _recommend_threshold(threshold_rows: List[Dict]) -> Optional[Dict]:
    if not threshold_rows:
        return None
    best_map = max(row.get("mAP_50", 0.0) for row in threshold_rows)
    best_ma_recall = max(row.get("MA_recall_50", 0.0) for row in threshold_rows)
    candidates = [
        row
        for row in threshold_rows
        if row.get("mAP_50", 0.0) >= best_map - 0.003
        and row.get("MA_recall_50", 0.0) >= best_ma_recall - 0.02
    ]
    pool = candidates or threshold_rows
    return min(
        pool,
        key=lambda row: (
            row.get("total_predictions", 0),
            -row.get("score_thresh", 0.0),
        ),
    )


def _run_threshold_sweep(
    all_preds: List[Dict],
    all_targets: List[Dict],
    n_gt_per_class: Dict[int, int],
    thresholds: List[float],
) -> List[Dict]:
    rows = []
    total_gt = sum(n_gt_per_class.values())
    seen = sorted({round(float(t), 6) for t in thresholds})
    for score_thresh in seen:
        filtered_preds = _filter_predictions_by_threshold(all_preds, score_thresh)
        evaluated = _evaluate_predictions(filtered_preds, all_targets)
        row = {
            "score_thresh": score_thresh,
            "mAP_50": evaluated["map_metrics"].get("mAP_50", 0.0),
            "mAP_75": evaluated["map_metrics"].get("mAP_75", 0.0),
            "MA_AP50": evaluated["ap_per_class"].get(1, 0.0),
            "MA_recall_50": evaluated["recall_per_class"].get(1, 0.0),
            "total_predictions": evaluated["total_predictions"],
            "false_positive_count": evaluated["false_positive_count"],
            "false_positive_pressure": round(
                evaluated["false_positive_pressure"],
                6,
            ),
            "mean_dets_per_image": round(evaluated["mean_dets_per_image"], 2),
            "no_det_images": evaluated["no_det_images"],
            "pred_gt_ratio": round(
                evaluated["total_predictions"] / max(total_gt, 1), 4
            ),
        }
        for cls_id, name in LABEL_MAP.items():
            row[f"pred_{name}"] = evaluated["n_pred_per_class"].get(cls_id, 0)
            row[f"recall_{name}"] = round(
                evaluated["recall_per_class"].get(cls_id, 0.0), 6
            )
        rows.append(row)
    return rows


def _write_threshold_sweep_csv(out_dir: str, rows: List[Dict]) -> Optional[str]:
    if not rows:
        return None
    path = os.path.join(out_dir, "threshold_sweep.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Stage1A] Threshold sweep CSV: {path}")
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Report generation
# ──────────────────────────────────────────────────────────────────────────────


def _high_fp_pressure_note(model_type: str, mean_dets_per_image: float) -> str:
    if model_type == "retinanet":
        if mean_dets_per_image >= 250:
            return (
                "> **High FP pressure.** Validation images are near the RetinaNet detection cap, "
                "so raising score_thresh alone is usually a weak lever. Stronger candidates are prior bias, "
                "anchor matching, focal gamma, and duplicate suppression/NMS tuning."
            )
        return (
            "> **High FP pressure.** Consider: (1) prior bias adjustment, (2) focal gamma tuning, "
            "(3) stricter duplicate suppression or anchor matching review."
        )
    if model_type == "fasterrcnn":
        return (
            "> **High FP pressure.** Consider: (1) score/NMS calibration, (2) detections_per_img and RPN proposal caps, "
            "(3) proposal duplication audit on dense lesion images."
        )
    return (
        "> **High FP pressure.** Consider: (1) raising score_thresh, (2) class-balanced focal loss tuning, "
        "(3) prior bias adjustment."
    )


def _tiny_object_note(model_type: str) -> str:
    if model_type == "fasterrcnn":
        return "Inspect RPN anchor recall and proposal coverage for tiny MA boxes."
    if model_type == "retinanet":
        return (
            "RetinaNet starts from coarser FPN anchors than P2-based detectors; inspect the smallest anchor sizes, "
            "positive-match counts on P3, and whether tiny MA boxes are left unmatched."
        )
    return "FCOS P2 stride=4 should detect these — check if P2 is being utilized (inspect FPN target counts)."


def _oversize_box_note(model_type: str) -> str:
    if model_type == "fasterrcnn":
        return "Proposal localization or ROI box regression may be drifting. Check RPN proposal quality and ROI box loss."
    if model_type == "retinanet":
        return (
            "Anchor assignment or box regression may be drifting on dense tiny lesions. "
            "Check anchor scales, box loss weighting, and post-NMS duplicate suppression."
        )
    return "Regression head may be over-predicting box size. Check IoU loss or centerness quality."


def _fp_bottleneck_note(
    model_type: str, pred_gt_ratio: float, mean_dets_per_image: float
) -> str:
    if model_type == "retinanet":
        if mean_dets_per_image >= 250:
            return (
                f"**[High] FP ratio={pred_gt_ratio:.1f}x.** "
                "Many images are already near the RetinaNet detection cap, so threshold-only tuning is unlikely to recover precision. "
                "Prior bias, anchor matching, focal gamma, or NMS tuning is needed."
            )
        return (
            f"**[High] FP ratio={pred_gt_ratio:.1f}x.** "
            "Score ranking is too flat for a dense anchor detector. Prior bias, focal gamma, anchor matching, or NMS tuning is needed."
        )
    if model_type == "fasterrcnn":
        return (
            f"**[High] FP ratio={pred_gt_ratio:.1f}x.** "
            "Proposal filtering is too permissive. Check score calibration, NMS, detections_per_img, and RPN proposal duplication."
        )
    return (
        f"**[High] FP ratio={pred_gt_ratio:.1f}x.** "
        "Score distribution too flat. Prior bias or focal gamma tuning needed."
    )


def _status(val, tgt, min_):
    if val >= tgt:
        return "PASS (target)"
    if val >= min_:
        return "PASS (minimum)"
    return "FAIL"


def _write_audit_report(
    out_dir: str,
    ckpt_path: str,
    model_type: str,
    split: str,
    score_thresh: float,
    map_metrics: Dict,
    ap_per_class: Dict[int, float],
    recall_per_class: Dict[int, float],
    gt_areas: Dict[int, List[float]],
    pred_areas: Dict[int, List[float]],
    conv: Dict,
    n_images: int,
    n_gt_per_class: Dict[int, int],
    n_pred_per_class: Dict[int, int],
    false_positive_count: int,
    false_positive_pressure: float,
    false_positive_per_class: Dict[int, int],
    threshold_sweep_rows: Optional[List[Dict]] = None,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    map50 = map_metrics.get("mAP_50", 0.0)

    lines = []
    A = lines.append

    model_label = ckpt_path.split(os.sep)[-2] if os.sep in ckpt_path else ckpt_path
    A(
        f"# Stage 1 — {_model_display_name(model_type)} Detector Readiness Audit ({model_label})"
    )
    A(f"Generated: {ts}")
    A(f"Checkpoint: `{ckpt_path}`")
    A(f"Split: {split}  |  Score thresh: {score_thresh}  |  Images: {n_images}")
    A("")

    # ── Acceptance gate ──────────────────────────────────────────────────────
    A("## Acceptance Gate (Stage 1C criteria)")
    A("")
    A("| Metric | Value | Target | Minimum | Status |")
    A("|--------|-------|--------|---------|--------|")
    ma_recall = recall_per_class.get(1, 0.0)
    A(
        f"| mAP_50 | {map50:.4f} | ≥0.08 | ≥0.05 | {_status(map50, ACCEPT_MAP50_TGT, ACCEPT_MAP50_MIN)} |"
    )
    A(
        f"| MA recall@50 | {ma_recall:.4f} | ≥0.20 | ≥0.10 | {_status(ma_recall, ACCEPT_MA_RECALL, 0.10)} |"
    )
    A("")
    if map50 < ACCEPT_MAP50_MIN:
        A(
            "> **GATE: FAIL** — mAP_50 below minimum floor. Stage 1B improvement required."
        )
    elif map50 < ACCEPT_MAP50_TGT:
        A(
            "> **GATE: PARTIAL** — mAP_50 above minimum but below target. Continue Stage 1B."
        )
    else:
        A("> **GATE: PASS** — Ready for Stage 1C freeze decision.")
    A("")

    # ── mAP summary ──────────────────────────────────────────────────────────
    A("## Detection Quality — mAP Summary")
    A("")
    A(f"- mAP (COCO): `{map_metrics.get('mAP', 'n/a')}`")
    A(f"- mAP_50:     `{map50:.4f}`")
    A(f"- mAP_75:     `{map_metrics.get('mAP_75', 'n/a')}`")
    A(f"- mAP_small:  `{map_metrics.get('mAP_small', 'n/a')}`")
    A(f"- mAP_medium: `{map_metrics.get('mAP_medium', 'n/a')}`")
    A(f"- mAP_large:  `{map_metrics.get('mAP_large', 'n/a')}`")
    A("")

    # ── Per-class ─────────────────────────────────────────────────────────────
    A("## Per-Class Metrics")
    A("")
    A(
        "| Class | AP_50 | Recall@50 | GT count | Pred count | FP count | Pred/GT ratio |"
    )
    A(
        "|-------|-------|-----------|----------|------------|----------|---------------|"
    )
    for cls_id, name in LABEL_MAP.items():
        ap = ap_per_class.get(cls_id, float("nan"))
        rec = recall_per_class.get(cls_id, 0.0)
        n_g = n_gt_per_class.get(cls_id, 0)
        n_p = n_pred_per_class.get(cls_id, 0)
        fp_c = false_positive_per_class.get(cls_id, 0)
        ratio = f"{n_p / max(n_g, 1):.1f}x"
        ap_s = f"{ap:.4f}" if not math.isnan(ap) else "n/a"
        A(f"| {name} | {ap_s} | {rec:.4f} | {n_g} | {n_p} | {fp_c} | {ratio} |")
    A("")

    # ── Convergence ───────────────────────────────────────────────────────────
    A("## Training Convergence Analysis")
    A("")
    if conv:
        A(f"- Total epochs trained: {conv.get('total_epochs', 'n/a')}")
        A(f"- Best epoch (lowest val_loss): **{conv.get('best_epoch', 'n/a')}**")
        A(f"- Best val_loss: {conv.get('best_val_loss', 'n/a')}")
        A(f"- Last val_loss: {conv.get('last_val_loss', 'n/a')}")
        if conv.get("best_map_epoch") is not None:
            A(
                f"- Best epoch (highest val_mAP50): **{conv.get('best_map_epoch', 'n/a')}**"
            )
            A(f"- Best val_mAP50: {conv.get('best_val_mAP50', 'n/a')}")
        if conv.get("map_at_best_loss_epoch") is not None:
            A(
                f"- val_mAP50 at best-loss epoch: {conv.get('map_at_best_loss_epoch', 'n/a')}"
            )
        if conv.get("val_loss_at_best_map_epoch") is not None:
            A(
                f"- val_loss at best-mAP epoch: {conv.get('val_loss_at_best_map_epoch', 'n/a')}"
            )
        A(
            f"- Plateau detected (last 3 epochs): {'YES' if conv.get('plateau') else 'no'}"
        )
        A(f"- Overfitting signal: {'YES' if conv.get('overfit_signal') else 'no'}")
        A("")
        if (
            conv.get("best_epoch") is not None
            and conv.get("best_map_epoch") is not None
            and conv.get("best_epoch") != conv.get("best_map_epoch")
        ):
            A(
                "> **Interpretation:** val_loss and val_mAP50 peak at different epochs. "
                "For torchvision detectors this is expected because val_loss is computed in training mode "
                "on sampled proposals, while mAP50 is computed in evaluation mode after ranking, NMS, "
                "and score thresholding."
            )
            A("")
        if (
            conv.get("best_epoch")
            and conv["best_epoch"] <= 3
            and conv.get("best_map_epoch") in (None, conv.get("best_epoch"))
        ):
            A(
                "> **Warning:** Best epoch very early — model may be undertrained or LR too high."
            )
        elif conv.get("plateau"):
            A(
                "> **Info:** Training plateaued. More epochs unlikely to help without schedule change."
            )
        elif conv.get("overfit_signal"):
            A(
                "> **Warning:** Overfitting signal detected. Consider regularization or early stop."
            )
    else:
        A("_Training log not provided or not found. Skipping convergence analysis._")
    A("")

    if threshold_sweep_rows:
        recommended = _recommend_threshold(threshold_sweep_rows)
        A("## Score Threshold Sweep")
        A("")
        A(
            "| Thresh | mAP_50 | MA AP_50 | MA Recall@50 | Total preds | FP count | FP pressure | Dets/img | Pred/GT |"
        )
        A(
            "|--------|--------|----------|--------------|-------------|----------|-------------|----------|---------|"
        )
        for row in threshold_sweep_rows:
            A(
                f"| {row['score_thresh']:.3f} | {row['mAP_50']:.4f} | {row['MA_AP50']:.4f} | "
                f"{row['MA_recall_50']:.4f} | {row['total_predictions']} | {row['false_positive_count']} | "
                f"{row['false_positive_pressure']:.3f} | {row['mean_dets_per_image']:.2f} | "
                f"{row['pred_gt_ratio']:.2f}x |"
            )
        A("")
        if recommended is not None:
            A(
                "> **Suggested operating point:** "
                f"score_thresh={recommended['score_thresh']:.3f} "
                f"(mAP_50={recommended['mAP_50']:.4f}, "
                f"MA recall@50={recommended['MA_recall_50']:.4f}, "
                f"pred/GT={recommended['pred_gt_ratio']:.2f}x)."
            )
            A("")
        map_values = [row["mAP_50"] for row in threshold_sweep_rows]
        pred_values = [row["total_predictions"] for row in threshold_sweep_rows]
        if max(map_values) < ACCEPT_MAP50_MIN:
            A(
                "> **Interpretation:** No tested threshold recovers the minimum mAP_50 floor. Treat score threshold as a secondary lever; the root issue is detector quality or calibration."
            )
            A("")
        if (
            max(map_values) - min(map_values) < 0.002
            and max(pred_values) - min(pred_values) < 1000
        ):
            A(
                "> **Interpretation:** Threshold sweep is nearly flat. Most validation images are already near the detector cap, "
                "so score threshold is a weak lever here; proposal quality, NMS, or detections_per_img are more likely to move mAP/FP tradeoffs."
            )
            A("")

    # ── Box area analysis ─────────────────────────────────────────────────────
    A("## Bounding Box Area Analysis (GT vs Predictions)")
    A("")
    A(
        "| Class | GT median_area | GT %tiny | GT %small | Pred median_area | Pred/GT area ratio |"
    )
    A(
        "|-------|---------------|----------|-----------|-----------------|-------------------|"
    )
    for cls_id, name in LABEL_MAP.items():
        gs = _area_stats(gt_areas.get(cls_id, []))
        ps = _area_stats(pred_areas.get(cls_id, []))
        g_med = gs["median"]
        p_med = ps["median"]
        ratio = f"{p_med / max(g_med, 1):.2f}x"
        A(
            f"| {name} | {g_med:.0f} | {gs['pct_tiny']:.1%} | {gs['pct_small']:.1%} | {p_med:.0f} | {ratio} |"
        )
    A("")
    A(
        "> **Note:** Pred/GT area ratio > 1.5 suggests over-large predictions (box regression drift)."
    )
    if model_type == "retinanet":
        A(
            "> MA %tiny high means small-object difficulty — audit smallest anchor coverage before assuming score-threshold tuning will help."
        )
    elif model_type == "fasterrcnn":
        A(
            "> MA %tiny high means small-object difficulty — inspect RPN anchor recall and proposal survival for tiny lesions."
        )
    else:
        A(
            "> MA %tiny high means small-object difficulty — may need P2 stride tuning or augmentation."
        )
    A("")

    # ── FP pressure ───────────────────────────────────────────────────────────
    A("## False Positive Pressure")
    A("")
    total_preds = sum(n_pred_per_class.values())
    total_gt = sum(n_gt_per_class.values())
    A(f"- Total predictions (score>{score_thresh}): {total_preds}")
    A(f"- Total GT boxes: {total_gt}")
    A(f"- False positive count: {false_positive_count}")
    A(f"- False positive pressure: {false_positive_pressure:.3f}")
    A(f"- Overall pred/GT ratio: {total_preds / max(total_gt, 1):.1f}x")
    A("")
    if false_positive_pressure > 5:
        A(_high_fp_pressure_note(model_type, total_preds / max(n_images, 1)))
    elif false_positive_pressure > 2:
        A(
            "> **Moderate FP pressure.** Tuning score_thresh or NMS may help precision without hurting recall."
        )
    else:
        A("> FP pressure is acceptable.")
    A("")

    # ── Bottleneck summary ────────────────────────────────────────────────────
    A("## Bottleneck Summary")
    A("")
    bottlenecks = []
    if map50 < ACCEPT_MAP50_MIN:
        bottlenecks.append(
            f"**[Critical] mAP_50={map50:.4f} below minimum floor {ACCEPT_MAP50_MIN}.** "
            "Governance layer cannot be evaluated. Stage 1B experiments mandatory."
        )
    if ma_recall < ACCEPT_MA_RECALL:
        bottlenecks.append(
            f"**[High] MA recall@50={ma_recall:.4f} below target {ACCEPT_MA_RECALL}.** "
            "MA (microaneurysm) is the smallest and most clinically critical lesion. "
            + (
                "Consider: (1) anchor/proposal recall on tiny boxes, (2) score calibration for dense small lesions, (3) MA-specific augmentation."
                if model_type == "fasterrcnn"
                else (
                    "Consider: (1) smallest-anchor coverage on P3, (2) prior bias / focal gamma, (3) MA-specific augmentation."
                    if model_type == "retinanet"
                    else "Consider: (1) P2 level weighting, (2) focal loss alpha, (3) MA-specific augmentation."
                )
            )
        )
    if total_preds / max(total_gt, 1) > 5:
        bottlenecks.append(
            _fp_bottleneck_note(
                model_type,
                total_preds / max(total_gt, 1),
                total_preds / max(n_images, 1),
            )
        )

    gs_ma = _area_stats(gt_areas.get(1, []))
    if gs_ma.get("pct_tiny", 0) > 0.30:
        bottlenecks.append(
            f"**[Medium] {gs_ma['pct_tiny']:.1%} of MA GT boxes are tiny (<32px side).** "
            + _tiny_object_note(model_type)
        )

    ps_all = sum(pred_areas.values(), [])
    gs_all = sum(gt_areas.values(), [])
    if ps_all and gs_all:
        p_med = sorted(ps_all)[len(ps_all) // 2]
        g_med = sorted(gs_all)[len(gs_all) // 2]
        if p_med / max(g_med, 1) > 2.5:
            bottlenecks.append(
                f"**[Medium] Pred boxes are ~{p_med/max(g_med,1):.1f}x larger than GT (median).** "
                + _oversize_box_note(model_type)
            )

    if conv.get("best_epoch") and conv.get("total_epochs"):
        if conv["best_epoch"] == conv["total_epochs"]:
            bottlenecks.append(
                "**[Medium] Best epoch is last epoch.** Model may not have converged. "
                "Consider training for more epochs."
            )

    if not bottlenecks:
        A("No critical bottlenecks detected. Proceed to Stage 1C freeze decision.")
    else:
        for i, b in enumerate(bottlenecks, 1):
            A(f"{i}. {b}")
            A("")

    # ── Stage 1B recommendations ──────────────────────────────────────────────
    A("## Stage 1B Experiment Recommendations (priority order)")
    A("")
    A("Change ONE variable per experiment. Record class-wise mAP_50 before and after.")
    A("")
    recs = []

    if ma_recall < ACCEPT_MA_RECALL:
        if model_type == "fasterrcnn":
            recs.append(
                (
                    "Anchor/proposal recall check",
                    "Verify the smallest anchors and RPN proposals cover MA lesions. "
                    "If tiny MA objects rarely survive proposal sampling, review anchor sizes or proposal limits.",
                )
            )
        elif model_type == "retinanet":
            recs.append(
                (
                    "Small-anchor coverage audit",
                    "Measure how many MA boxes are matched by the smallest RetinaNet anchors. "
                    "If positive matches on P3 are scarce, review anchor sizes/aspect ratios before retraining.",
                )
            )
        else:
            recs.append(
                (
                    "P2 utilization check",
                    "Verify that FPN P2 level (stride=4) is generating positives for MA. "
                    "If P2 positives are <5% of total, review matcher expansion radius.",
                )
            )
    if total_preds / max(total_gt, 1) > 5:
        if model_type == "fasterrcnn":
            recs.append(
                (
                    "Proposal/NMS pressure check",
                    "Compare mAP/FP with detections_per_img and RPN caps tightened or loosened. "
                    "If predictions stay saturated, inspect duplicate proposals before changing the detector loss.",
                )
            )
        elif model_type == "retinanet":
            recs.append(
                (
                    "Prior bias and focal calibration",
                    "Audit classification prior bias and try focal gamma 3-4. "
                    "Pair it with stricter duplicate suppression/NMS because the current run sits near the detection cap.",
                )
            )
        else:
            recs.append(
                (
                    "Focal loss gamma",
                    "Increase gamma from 2.0 → 3.0 or 4.0. Reduces FP by down-weighting easy negatives.",
                )
            )
    if gs_ma.get("pct_tiny", 0) > 0.20:
        recs.append(
            (
                "MA-targeted augmentation",
                "Add random zoom-in (scale 0.5–0.8) to oversample small lesion crops. "
                "Keep global augmentation unchanged.",
            )
        )

    recs.append(
        (
            "Learning rate schedule",
            "Try cosine annealing from current LR to 1e-6. "
            "Current schedule may be stepping too aggressively.",
        )
    )
    best_map_epoch = conv.get("best_map_epoch")
    total_epochs = conv.get("total_epochs")
    if best_map_epoch is not None and total_epochs is not None:
        if best_map_epoch >= total_epochs - 2:
            recs.append(
                (
                    "Training duration",
                    f"Best val_mAP50 was at epoch {best_map_epoch} of {total_epochs}. "
                    "A short extension of 3-5 epochs is reasonable if you keep the same schedule.",
                )
            )
        elif conv.get("plateau"):
            recs.append(
                (
                    "Training duration",
                    f"Best val_mAP50 was at epoch {best_map_epoch} of {total_epochs}, and later epochs plateaued. "
                    "More epochs under the current schedule are unlikely to help.",
                )
            )
    else:
        recs.append(
            (
                "Training duration",
                f"Best epoch was {conv.get('best_epoch', '?')} of {conv.get('total_epochs', '?')}. "
                "If converging, train 5 more epochs and re-evaluate.",
            )
        )
    recs.append(
        (
            "Score threshold sweep",
            (
                "Sweep thresh in [0.05, 0.10, 0.20, 0.30, 0.50] and report mAP_50 + recall tradeoff. "
                "If no threshold recovers the minimum floor, treat threshold as a secondary lever and focus on anchor matching / duplicate suppression."
                if model_type == "retinanet"
                else "Sweep thresh in [0.005, 0.01, 0.02, 0.03, 0.05] and report mAP_50 + recall tradeoff. "
                "Current default 0.01 may not be optimal."
            ),
        )
    )

    for i, (title, desc) in enumerate(recs, 1):
        A(f"{i}. **{title}** — {desc}")
        A("")

    A("---")
    A(
        "_Stage 1A audit complete. Run Stage 1B experiments after reviewing bottlenecks above._"
    )
    A(
        f"_Architecture source of truth: `Proposed/architecture-lgov-net-v3_realistic.txt`_"
    )

    report_path = os.path.join(out_dir, "stage1a_readiness_audit.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[Stage1A] Report: {report_path}")
    return report_path


def _write_bbox_csv(out_dir: str, gt_areas: Dict, pred_areas: Dict):
    path = os.path.join(out_dir, "bbox_stats_per_class.csv")
    fields = [
        "class",
        "source",
        "n",
        "mean_area",
        "median_area",
        "p10_area",
        "p90_area",
        "min_area",
        "max_area",
        "pct_tiny_lt1024",
        "pct_small_lt4096",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for cls_id, name in LABEL_MAP.items():
            for src, areas in [
                ("gt", gt_areas.get(cls_id, [])),
                ("pred", pred_areas.get(cls_id, [])),
            ]:
                s = _area_stats(areas)
                w.writerow(
                    {
                        "class": name,
                        "source": src,
                        "n": s["n"],
                        "mean_area": s["mean"],
                        "median_area": s["median"],
                        "p10_area": s["p10"],
                        "p90_area": s["p90"],
                        "min_area": s["min"],
                        "max_area": s["max"],
                        "pct_tiny_lt1024": s["pct_tiny"],
                        "pct_small_lt4096": s["pct_small"],
                    }
                )
    print(f"[Stage1A] BBox stats: {path}")


def _write_convergence_csv(out_dir: str, rows: List[Dict]):
    if not rows:
        return
    first = rows[0]
    train_key = _pick_first_key(first, "loss_total", "train_loss")
    val_key = _pick_first_key(first, "val_loss_total", "val_loss")
    map_key = _pick_first_key(first, "val_mAP50", "mAP_50")
    balanced_key = _pick_first_key(first, "val_balanced", "balanced")
    dets_key = _pick_first_key(first, "val_mean_dets_per_image", "mean_dets_per_image")
    score_key = _pick_first_key(first, "val_max_score", "max_score")

    keys = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_mAP50",
        "val_balanced",
        "mean_dets_per_image",
        "max_score",
    ]
    path = os.path.join(out_dir, "convergence_analysis.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(
                {
                    "epoch": row.get("epoch", ""),
                    "train_loss": row.get(train_key, "") if train_key else "",
                    "val_loss": row.get(val_key, "") if val_key else "",
                    "val_mAP50": row.get(map_key, "") if map_key else "",
                    "val_balanced": row.get(balanced_key, "") if balanced_key else "",
                    "mean_dets_per_image": row.get(dets_key, "") if dets_key else "",
                    "max_score": row.get(score_key, "") if score_key else "",
                }
            )
    print(f"[Stage1A] Convergence CSV: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Stage 1A — Detector Readiness Audit")
    parser.add_argument("--checkpoint", default=_DEFAULT_CKPT)
    parser.add_argument(
        "--log_csv",
        default=None,
        help="Training log CSV. If omitted, auto-detected from checkpoint directory.",
    )
    parser.add_argument(
        "--output_dir", default=os.path.join(_root, "outputs", "detector_stage1")
    )
    parser.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    parser.add_argument(
        "--data_root",
        default=_DATA_ROOT,
        help="Dataset root containing images/ and xml/ subdirs (default: data/).",
    )
    parser.add_argument("--score_thresh", type=float, default=0.01)
    parser.add_argument(
        "--max_images",
        type=int,
        default=0,
        help="Limit images for fast testing (0 = all)",
    )
    parser.add_argument(
        "--model_type",
        default="fasterrcnn",
        choices=["fcos", "retinanet", "fasterrcnn"],
        help="Detector type to load from checkpoint (default: fasterrcnn).",
    )
    parser.add_argument(
        "--detections_per_img",
        type=int,
        default=0,
        help="Override Faster R-CNN roi_heads.detections_per_img (default: 0 = use model default=100). "
        "Use 300 to test detection cap hypothesis on dense lesion images.",
    )
    parser.add_argument(
        "--sweep_thresholds",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Optional score thresholds for post-audit sweep. "
            f"Example: {' '.join(str(t) for t in DEFAULT_SWEEP_THRESHOLDS)}"
        ),
    )
    parser.add_argument(
        "--require_min_map50",
        type=float,
        default=None,
        help=(
            "Optional hard gate: exit non-zero if measured mAP_50 is below this value. "
            "Useful for enforcing baseline parity checks in automated runbooks."
        ),
    )
    args = parser.parse_args()

    # Auto-detect log CSV from checkpoint directory if not provided
    if args.log_csv is None:
        ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint))
        # Prefer logs in the same checkpoint directory to avoid cross-run contamination.
        for candidate_name in (
            "training_log_full_mode2_detector.csv",
            "training_log.csv",
            "head_baseline_log.csv",
        ):
            candidate = os.path.join(ckpt_dir, candidate_name)
            if os.path.isfile(candidate):
                args.log_csv = candidate
                break
        if args.log_csv is None:
            print(
                "[Stage1A] No local training log found near checkpoint; "
                "convergence section will be omitted."
            )

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[Stage1A] === Phase 19N Detector Readiness Audit ===")
    print(f"[Stage1A] model_type={args.model_type}  Checkpoint: {args.checkpoint}")

    # Inference
    all_preds, all_targets, gt_areas, pred_areas = _run_inference(
        args.checkpoint,
        args.split,
        args.score_thresh,
        args.max_images,
        model_type=args.model_type,
        detections_per_img=args.detections_per_img,
        data_root=args.data_root,
    )
    n_images = len(all_preds)

    evaluated = _evaluate_predictions(all_preds, all_targets)
    map_metrics = evaluated["map_metrics"]
    ap_per_class = evaluated["ap_per_class"]
    recall_per_class = evaluated["recall_per_class"]

    # GT / pred counts
    n_gt_per_class = {c: len(gt_areas.get(c, [])) for c in CLASS_IDS}
    n_pred_per_class = evaluated["n_pred_per_class"]

    threshold_sweep_rows = None
    if args.sweep_thresholds:
        sweep_thresholds = sorted({round(float(t), 6) for t in args.sweep_thresholds})
        sweep_base_thresh = min(sweep_thresholds)
        sweep_preds = all_preds
        sweep_targets = all_targets
        if sweep_base_thresh < args.score_thresh:
            print(
                f"[Stage1A] Re-running inference for threshold sweep at min threshold {sweep_base_thresh}"
            )
            sweep_preds, sweep_targets, _, _ = _run_inference(
                args.checkpoint,
                args.split,
                sweep_base_thresh,
                args.max_images,
                model_type=args.model_type,
                detections_per_img=args.detections_per_img,
                data_root=args.data_root,
            )
        threshold_sweep_rows = _run_threshold_sweep(
            sweep_preds,
            sweep_targets,
            n_gt_per_class,
            sweep_thresholds,
        )
        _write_threshold_sweep_csv(args.output_dir, threshold_sweep_rows)

    # Training log
    log_rows = _parse_log(args.log_csv)
    conv = _convergence_analysis(log_rows or [])
    _write_convergence_csv(args.output_dir, log_rows or [])

    # Outputs
    _write_audit_report(
        args.output_dir,
        args.checkpoint,
        args.model_type,
        args.split,
        args.score_thresh,
        map_metrics,
        ap_per_class,
        recall_per_class,
        gt_areas,
        pred_areas,
        conv,
        n_images,
        n_gt_per_class,
        n_pred_per_class,
        evaluated["false_positive_count"],
        evaluated["false_positive_pressure"],
        evaluated["false_positive_per_class"],
        threshold_sweep_rows=threshold_sweep_rows,
    )
    _write_bbox_csv(args.output_dir, gt_areas, pred_areas)

    # Raw JSON
    raw_out = {
        "run_ts": datetime.now().isoformat(),
        "checkpoint": args.checkpoint,
        "split": args.split,
        "score_thresh": args.score_thresh,
        "n_images": n_images,
        "map_metrics": map_metrics,
        "ap_per_class": {LABEL_MAP[k]: v for k, v in ap_per_class.items()},
        "recall_per_class": {LABEL_MAP[k]: v for k, v in recall_per_class.items()},
        "n_gt_per_class": {LABEL_MAP[k]: v for k, v in n_gt_per_class.items()},
        "n_pred_per_class": {LABEL_MAP[k]: v for k, v in n_pred_per_class.items()},
        "false_positive_count": evaluated["false_positive_count"],
        "false_positive_pressure": evaluated["false_positive_pressure"],
        "false_positive_per_class": {
            LABEL_MAP[k]: v for k, v in evaluated["false_positive_per_class"].items()
        },
        "convergence": conv,
        "threshold_sweep": threshold_sweep_rows,
    }
    raw_path = os.path.join(args.output_dir, "stage1a_raw.json")
    with open(raw_path, "w") as f:
        json.dump(raw_out, f, indent=2)
    print(f"[Stage1A] Raw JSON: {raw_path}")

    # Console summary
    map50 = map_metrics.get("mAP_50", 0.0)
    print("\n[Stage1A] ══════════════════════════ SUMMARY ══════════════════")
    print(
        f"  mAP_50 = {map50:.4f}  (min={ACCEPT_MAP50_MIN}, target={ACCEPT_MAP50_TGT})"
    )
    for cls_id, name in LABEL_MAP.items():
        ap = ap_per_class.get(cls_id, float("nan"))
        rec = recall_per_class.get(cls_id, 0.0)
        print(
            f"  {name:<10} AP50={ap:.4f}  recall@50={rec:.4f}  "
            f"gt={n_gt_per_class[cls_id]}  pred={n_pred_per_class[cls_id]}"
        )
    gate = (
        "FAIL"
        if map50 < ACCEPT_MAP50_MIN
        else ("PARTIAL" if map50 < ACCEPT_MAP50_TGT else "PASS")
    )
    print(f"\n  GATE: {gate}")
    if args.require_min_map50 is not None:
        hard_gate = float(args.require_min_map50)
        if map50 + 1e-12 < hard_gate:
            print(
                f"[Stage1A] HARD GATE FAIL: mAP_50={map50:.4f} "
                f"< required {hard_gate:.4f}"
            )
            raise SystemExit(2)
        print(
            f"[Stage1A] HARD GATE PASS: mAP_50={map50:.4f} "
            f">= required {hard_gate:.4f}"
        )
    print("[Stage1A] ═════════════════════════════════════════════════════\n")


if __name__ == "__main__":
    main()
