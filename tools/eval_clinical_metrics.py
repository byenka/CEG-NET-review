#!/usr/bin/env python
"""eval_clinical_metrics.py — clinical decision metrics for Mode 2 v2.1 checkpoints.

Loads a Faster R-CNN checkpoint (Mode 1 LOCKED detector, Mode 2 v2 best, or any
compatible checkpoint), runs inference on the requested split, and computes
clinical metrics suitable for MedIA submission:

- Per-class AUROC / AUPRC
- Sensitivity @ specificity = 0.90 / 0.95 / 0.99
- Recall @ FP-per-image budget = 1 / 3 / 5
- Brier score + Expected Calibration Error (ECE, 10 bins)
- Macro aggregates across MA / HE / Hard_EX / Soft_EX

CUDA enforcement: ``--require_cuda`` (default True) hard-fails if cuda is not
available, since checkpoint inference on large evaluation sets is impractical
on CPU.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

_here = Path(__file__).resolve().parent
_root = _here.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import torch
from torch.utils.data import DataLoader

from src.metrics.clinical_metrics import compute_all
from src.data.coco_adapter import DRDetectionDataset, collate_fn  # type: ignore


def _build_model(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
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
                "boxes": p["boxes"].detach().cpu().numpy(),
                "labels": p["labels"].detach().cpu().numpy(),
                "scores": p["scores"].detach().cpu().numpy(),
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


def main() -> int:
    p = argparse.ArgumentParser(description="Clinical metrics for Mode 2 v2.1.")
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--data_root", required=True, type=Path)
    p.add_argument("--split", default="valid")
    p.add_argument("--output_json", required=True, type=Path)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--max_images", type=int, default=0,
                   help="0 = use full split.")
    p.add_argument("--match_iou", type=float, default=0.5)
    p.add_argument("--require_cuda", action="store_true",
                   help="Hard-fail if CUDA unavailable (recommended on server).")
    args = p.parse_args()

    if args.require_cuda and not torch.cuda.is_available():
        print("[clinical] ERROR: --require_cuda set but torch.cuda.is_available() is False.")
        return 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[clinical] device={device}  checkpoint={args.checkpoint}")

    max_images = args.max_images if args.max_images > 0 else None
    ds = DRDetectionDataset(split=args.split, data_root=str(args.data_root), max_images=max_images)
    loader = DataLoader(
        ds, batch_size=max(1, int(args.batch_size)),
        shuffle=False, collate_fn=collate_fn,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=(device.type == "cuda"),
    )

    model = _build_model(args.checkpoint, device)
    print(f"[clinical] running inference on split={args.split} ({len(ds)} images)...")
    detections = _run_inference(model, loader, device)
    targets = _gather_targets(ds)
    print(f"[clinical] computing clinical metrics (match_iou={args.match_iou})...")
    metrics = compute_all(detections, targets, match_iou=float(args.match_iou))

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tool": "eval_clinical_metrics",
        "checkpoint": str(args.checkpoint),
        "data_root": str(args.data_root),
        "split": args.split,
        "n_images": len(ds),
        "device": str(device),
        "metrics": metrics,
    }
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[clinical] wrote {args.output_json}")
    macro = metrics["macro"]
    print(f"[clinical] macro_AUROC={macro['macro_auroc']:.4f}  "
          f"macro_AUPRC={macro['macro_auprc']:.4f}  "
          f"macro_sens@spec0.95={macro['macro_sens_at_spec_0p95']:.4f}  "
          f"macro_recall@fp1={macro['macro_recall_at_fp_1']:.4f}  "
          f"overall_ECE={macro['overall_ece']:.4f}  "
          f"overall_Brier={macro['overall_brier']:.4f}")
    # REV-7: surface Mode 2A primary metric family (architecture §C3 gate (vi)).
    ma_he_rate = float(macro.get("ma_he_confusion_rate", 0.0))
    ma_he_fwd = float(macro.get("ma_he_forward_rate", 0.0))
    ma_he_rev = float(macro.get("ma_he_reverse_rate", 0.0))
    ma_auroc = float(macro.get("ma_auroc", 0.0))
    ma_auprc = float(macro.get("ma_auprc", 0.0))
    print(f"[clinical] MA_AUROC={ma_auroc:.4f}  MA_AUPRC={ma_auprc:.4f}  "
          f"MA<->HE confusion (weighted_avg)={ma_he_rate:.4f}  "
          f"(forward={ma_he_fwd:.4f}, reverse={ma_he_rev:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
