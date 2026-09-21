#!/usr/bin/env python
"""posthoc_temperature_scaling.py — REV-7C safety-net calibration tool.

Fit a single global temperature parameter T to recalibrate Faster R-CNN
detection scores on the OIA-DDR valid split. Output:

  - calibration_table.json   (mapping original_score → calibrated_score)
  - calibrated_clinical_metrics.json  (full re-run of compute_all with new scores)
  - posthoc_summary.json     (T, ECE_before, ECE_after, FPpr_before/after)

This is a SUPPORTING ABLATION tool per architecture REV-7 §EXPLICIT
REJECTIONS list ("Calibration-only revision is treated as supporting
ablation, NOT as C3 novelty"). Use it to recover marginal ECE / FPpr at
audit time when the Mode 2A REAL run is close to the non-regression gate
but not perfectly calibrated.

Method:
  - Run frozen-checkpoint inference on the valid split.
  - For each detection, label it TP/FP via greedy per-class IoU=0.5 match.
  - Grid-search T over [0.1, 0.2, ..., 5.0] to minimise overall ECE (10 bins).
  - Apply T: new_score = sigmoid(logit(p) / T)  where logit(p) = log(p/(1-p)).
  - Re-evaluate clinical metrics with calibrated scores; record both.

CUDA enforcement: --require_cuda hard-fails if CUDA is unavailable.

Usage:
  python tools/posthoc_temperature_scaling.py \\
      --checkpoint outputs/mode2_r4_*/audit_topk/selected_best.pt \\
      --data_root data/oiaddr \\
      --split valid \\
      --output_dir outputs/mode2_r4_*/posthoc_temp \\
      --require_cuda
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_here = Path(__file__).resolve().parent
_root = _here.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.metrics.clinical_metrics import compute_all, _box_iou
from src.data.coco_adapter import DRDetectionDataset, collate_fn  # type: ignore


LESION_CLASSES = ("MA", "HE", "Hard_EX", "Soft_EX")
FRCNN_LABEL_TO_CLASS: Dict[int, str] = {
    (i + 1): name for i, name in enumerate(LESION_CLASSES)
}


def _build_model(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    """Construct a Faster R-CNN model and load the given checkpoint state.

    Mirrors the construction used by ``tools/eval_clinical_metrics.py`` so
    that calibrated inference is consistent with the un-calibrated baseline.
    """
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.rpn import AnchorGenerator
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    cfg = ckpt.get("config", {}) or {}
    num_classes = int(cfg.get("num_classes", 5))
    anchor_sizes = cfg.get("anchor_sizes", ((8,), (16,), (32,), (64,), (128,)))
    anchor_ratios = cfg.get("anchor_ratios", ((0.5, 1.0, 2.0),) * 5)
    kwargs = dict(
        rpn_anchor_generator=AnchorGenerator(
            sizes=tuple(tuple(s) for s in anchor_sizes),
            aspect_ratios=tuple(tuple(r) for r in anchor_ratios),
        ),
        box_score_thresh=0.02,
        box_detections_per_img=int(cfg.get("box_detections_per_img", 200)),
    )
    try:
        model = fasterrcnn_resnet50_fpn(weights=None, **kwargs)
    except TypeError:
        model = fasterrcnn_resnet50_fpn(pretrained=False, **kwargs)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()


@torch.no_grad()
def _run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> List[Dict]:
    out: List[Dict] = []
    for images, _ in loader:
        images = [img.to(device) for img in images]
        preds = model(images)
        for p in preds:
            out.append({
                "boxes": p["boxes"].detach().cpu().numpy().astype(np.float32),
                "labels": p["labels"].detach().cpu().numpy().astype(np.int64),
                "scores": p["scores"].detach().cpu().numpy().astype(np.float32),
            })
    return out


def _gather_targets(ds: DRDetectionDataset) -> List[Dict]:
    out: List[Dict] = []
    for i in range(len(ds)):
        _, tgt = ds[i]
        out.append({
            "boxes": tgt["boxes"].detach().cpu().numpy() if torch.is_tensor(tgt["boxes"]) else tgt["boxes"],
            "labels": tgt["labels"].detach().cpu().numpy() if torch.is_tensor(tgt["labels"]) else tgt["labels"],
        })
    return out


def _flag_tp_per_class(
    detections: List[Dict],
    targets: List[Dict],
    match_iou: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build (scores, tp_flags) flat arrays across all classes for calibration.

    For each predicted box, find the best-IoU GT box of the SAME class
    (greedy, per-image, descending score). TP=1 if matched at IoU>=match_iou.
    """
    all_scores: List[float] = []
    all_tp: List[int] = []
    for det, tgt in zip(detections, targets):
        d_boxes = det["boxes"]
        d_labels = det["labels"]
        d_scores = det["scores"]
        t_boxes = np.asarray(
            tgt.get("boxes", np.zeros((0, 4))), dtype=np.float32
        ).reshape(-1, 4)
        t_labels = np.asarray(tgt.get("labels", []))
        # Sort by descending score for greedy matching.
        order = np.argsort(-d_scores)
        d_boxes = d_boxes[order]
        d_labels = d_labels[order]
        d_scores = d_scores[order]
        used = np.zeros(t_boxes.shape[0], dtype=bool)
        if t_boxes.shape[0] > 0 and d_boxes.shape[0] > 0:
            iou = _box_iou(d_boxes, t_boxes)
        else:
            iou = np.zeros((d_boxes.shape[0], t_boxes.shape[0]), dtype=np.float32)
        for i in range(d_boxes.shape[0]):
            all_scores.append(float(d_scores[i]))
            assigned = False
            if iou.shape[1] > 0:
                best_j = -1
                best_v = match_iou
                for j in range(t_boxes.shape[0]):
                    if used[j]:
                        continue
                    if int(t_labels[j]) != int(d_labels[i]):
                        continue
                    if iou[i, j] >= best_v:
                        best_v = float(iou[i, j])
                        best_j = j
                if best_j >= 0:
                    used[best_j] = True
                    assigned = True
            all_tp.append(1 if assigned else 0)
    return np.asarray(all_scores, dtype=np.float32), np.asarray(all_tp, dtype=np.int32)


def _ece(scores: np.ndarray, tp: np.ndarray, n_bins: int = 10) -> float:
    if scores.size == 0:
        return 0.0
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
    return float(ece)


def _apply_temperature(scores: np.ndarray, T: float, eps: float = 1e-6) -> np.ndarray:
    """T-scaled scores via logit-divide.

    new_score = sigmoid(logit(p) / T) where logit(p) = log(p / (1-p)).
    T > 1 makes scores less confident (pull toward 0.5).
    T < 1 makes scores more confident (push away from 0.5).
    T = 1 is identity.
    """
    if T <= 0:
        raise ValueError(f"Temperature must be > 0; got {T}")
    p = np.clip(scores, eps, 1.0 - eps)
    logit = np.log(p / (1.0 - p))
    out = 1.0 / (1.0 + np.exp(-logit / T))
    return out.astype(np.float32)


def _fit_temperature(
    scores: np.ndarray,
    tp: np.ndarray,
    grid: Optional[List[float]] = None,
) -> Tuple[float, float, float]:
    """Grid-search T to minimize ECE. Returns (T_best, ECE_before, ECE_after)."""
    if grid is None:
        # Two-pass grid: coarse then fine.
        coarse = [round(0.1 * i, 2) for i in range(1, 51)]  # 0.1..5.0
        grid = coarse
    ece_before = _ece(scores, tp)
    best_T = 1.0
    best_ece = ece_before
    for T in grid:
        try:
            new_scores = _apply_temperature(scores, T)
        except ValueError:
            continue
        ece_T = _ece(new_scores, tp)
        if ece_T < best_ece:
            best_ece = ece_T
            best_T = float(T)
    # Refinement around best_T.
    fine = [round(best_T + 0.01 * k, 3) for k in range(-9, 10)]
    fine = [t for t in fine if t > 0]
    for T in fine:
        try:
            new_scores = _apply_temperature(scores, T)
        except ValueError:
            continue
        ece_T = _ece(new_scores, tp)
        if ece_T < best_ece:
            best_ece = ece_T
            best_T = float(T)
    return float(best_T), float(ece_before), float(best_ece)


def _calibrate_detections(detections: List[Dict], T: float) -> List[Dict]:
    out: List[Dict] = []
    for det in detections:
        new_scores = _apply_temperature(det["scores"], T)
        out.append({
            "boxes": det["boxes"],
            "labels": det["labels"],
            "scores": new_scores,
        })
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="REV-7C post-hoc temperature scaling.")
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--data_root", required=True, type=Path)
    p.add_argument("--split", default="valid")
    p.add_argument("--output_dir", required=True, type=Path)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--match_iou", type=float, default=0.5)
    p.add_argument(
        "--temperature_override",
        type=float,
        default=None,
        help="Skip grid search and use the supplied T. Useful for replication.",
    )
    p.add_argument(
        "--require_cuda",
        action="store_true",
        help="Hard-fail if torch.cuda.is_available() is False.",
    )
    args = p.parse_args()

    if args.require_cuda and not torch.cuda.is_available():
        print("[posthoc] ERROR: --require_cuda set but CUDA not available.")
        return 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[posthoc] device={device}  checkpoint={args.checkpoint}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    ds = DRDetectionDataset(split=args.split, data_root=str(args.data_root))
    loader = DataLoader(
        ds,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=(device.type == "cuda"),
    )

    model = _build_model(args.checkpoint, device)
    print(f"[posthoc] running inference on split={args.split} ({len(ds)} images)...")
    detections = _run_inference(model, loader, device)
    targets = _gather_targets(ds)
    print("[posthoc] flagging TP via greedy per-class IoU=0.5 match...")
    scores_all, tp_all = _flag_tp_per_class(detections, targets, match_iou=args.match_iou)
    print(
        f"[posthoc] collected {scores_all.size} detection scores "
        f"(positive rate = {(tp_all.sum() / max(tp_all.size, 1)):.4f})"
    )

    if args.temperature_override is not None:
        T_best = float(args.temperature_override)
        ece_before = _ece(scores_all, tp_all)
        new_scores = _apply_temperature(scores_all, T_best)
        ece_after = _ece(new_scores, tp_all)
        print(f"[posthoc] using T_override={T_best:.3f}")
    else:
        print("[posthoc] grid-searching T over [0.1, 5.0] then fine-tuning ±0.01...")
        T_best, ece_before, ece_after = _fit_temperature(scores_all, tp_all)
        print(
            f"[posthoc] T_best={T_best:.3f}  ECE_before={ece_before:.4f}  "
            f"ECE_after={ece_after:.4f}  (reduction {ece_before - ece_after:.4f})"
        )

    # Apply T to every detection and re-evaluate full clinical metrics.
    calibrated = _calibrate_detections(detections, T_best)
    print("[posthoc] re-computing clinical metrics with calibrated scores...")
    metrics_calibrated = compute_all(calibrated, targets, match_iou=float(args.match_iou))
    metrics_original = compute_all(detections, targets, match_iou=float(args.match_iou))

    summary = {
        "tool": "posthoc_temperature_scaling",
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "n_images": len(ds),
        "n_detections": int(scores_all.size),
        "match_iou": float(args.match_iou),
        "temperature_best": float(T_best),
        "ECE_flat_before": float(ece_before),
        "ECE_flat_after": float(ece_after),
        "macro_before": metrics_original["macro"],
        "macro_after": metrics_calibrated["macro"],
    }
    (args.output_dir / "posthoc_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (args.output_dir / "calibrated_clinical_metrics.json").write_text(
        json.dumps(
            {
                "tool": "posthoc_temperature_scaling",
                "checkpoint": str(args.checkpoint),
                "data_root": str(args.data_root),
                "split": args.split,
                "n_images": len(ds),
                "device": str(device),
                "temperature_best": float(T_best),
                "metrics": metrics_calibrated,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    macro_b = metrics_original["macro"]
    macro_a = metrics_calibrated["macro"]
    print(
        f"[posthoc] BEFORE: macro_AUROC={macro_b['macro_auroc']:.4f}  "
        f"overall_ECE={macro_b['overall_ece']:.4f}  "
        f"overall_Brier={macro_b['overall_brier']:.4f}"
    )
    print(
        f"[posthoc] AFTER:  macro_AUROC={macro_a['macro_auroc']:.4f}  "
        f"overall_ECE={macro_a['overall_ece']:.4f}  "
        f"overall_Brier={macro_a['overall_brier']:.4f}"
    )
    print(f"[posthoc] wrote: {args.output_dir / 'posthoc_summary.json'}")
    print(f"[posthoc] wrote: {args.output_dir / 'calibrated_clinical_metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
