#!/usr/bin/env python
"""
train_stage1c_retinanet.py
Phase 19N Stage 1C — RetinaNet/FPN solid detector integration.

Historical FCOS floor was superseded by Faster R-CNN/FPN `r5b_ext25` as the
current primary Mode-1 detector. This RetinaNet script remains useful as a
second established detector for diversity experiments and auxiliary-source runs.
This script uses torchvision RetinaNet/FPN with pretrained COCO backbone
as a non-novelty supporting component per architecture v3.

Experiment order (one variable per run):
  R1a: RetinaNet pretrained, NO hflip  ← this run
  R1b: RetinaNet pretrained + hflip    ← only if R1a passes mAP50>=0.05
    R2:  Faster R-CNN/FPN primary path   ← already available if R1a fails

Label mapping (locked, matches coco_adapter.py):
  ma  → 1 → MA
  he  → 2 → HE
  ex  → 3 → Hard_EX
  se  → 4 → Soft_EX
  background → 0 (torchvision convention)

Usage:
  # Smoke test (1 epoch, 8 images)
  python tools/train_stage1c_retinanet.py --max_images 8 --epochs 1 --smoke_test

  # Stage1C-R1a full run
  python tools/train_stage1c_retinanet.py \\
    --epochs 15 --batch_size 2 --lr 1e-4 --seed 42 \\
    --freeze_backbone_epochs 5 \\
    --output_dir outputs/detector_stage1/stage1c_retina_r1a
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from typing import Dict, List, Optional

if int(os.environ.get("OMP_NUM_THREADS", "0") or "0") <= 0:
    os.environ["OMP_NUM_THREADS"] = "1"
if "MKL_NUM_THREADS" not in os.environ:
    os.environ["MKL_NUM_THREADS"] = "1"

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.join(_here, "..")
if _root not in sys.path:
    sys.path.insert(0, _root)

import torch
from torch.utils.data import DataLoader

from lgovnet.src.data.coco_adapter import (
    DRDetectionDataset,
    LABEL_NAMES,
    NUM_CLASSES,
    collate_fn,
)

# ── RetinaNet imports ─────────────────────────────────────────────────────────

try:
    from torchvision.models.detection import retinanet_resnet50_fpn
    from torchvision.models.detection.retinanet import RetinaNetClassificationHead

    try:
        from torchvision.models.detection import RetinaNet_ResNet50_FPN_Weights

        _WEIGHTS_V2 = True
    except ImportError:
        _WEIGHTS_V2 = False
except ImportError as e:
    raise ImportError(
        "torchvision not found. Install: pip install torchvision\n" f"Original: {e}"
    )

_DATA_ROOT = os.path.join(_root, "data")
_DEFAULT_OUT = os.path.join(_root, "outputs", "detector_stage1", "stage1c_retina_r1a")

LABEL_MAP: Dict[int, str] = {i + 1: n for i, n in enumerate(LABEL_NAMES)}
CLASS_IDS = list(LABEL_MAP.keys())  # [1, 2, 3, 4]


# ── Model factory ─────────────────────────────────────────────────────────────


def _build_retinanet(num_classes: int, pretrained: bool) -> torch.nn.Module:
    """Load pretrained RetinaNet and replace classification head."""
    if pretrained:
        if _WEIGHTS_V2:
            weights = RetinaNet_ResNet50_FPN_Weights.COCO_V1
            model = retinanet_resnet50_fpn(weights=weights)
        else:
            model = retinanet_resnet50_fpn(pretrained=True)
        print("[RetinaNet] Loaded pretrained COCO_V1 weights")
    else:
        if _WEIGHTS_V2:
            model = retinanet_resnet50_fpn(weights=None)
        else:
            model = retinanet_resnet50_fpn(pretrained=False)
        print("[RetinaNet] No pretrained weights")

    # Replace classification head.
    # num_classes must be >= max(label_ids)+1 because torchvision indexes the
    # one-hot target tensor with label values directly.
    # Our labels are 1-indexed {1,2,3,4}, so we need num_classes=5 (indices 0-4).
    # Class-0 output node is never trained positively and gets filtered at inference.
    num_anchors = model.head.classification_head.num_anchors
    in_channels = 256  # standard FPN output channels
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes,  # caller passes NUM_CLASSES+1
    )
    print(
        f"[RetinaNet] Head replaced: 91 COCO → {num_classes} nodes"
        f"  (labels 1-4: {', '.join(LABEL_NAMES)}; node 0 unused)"
    )
    return model


def _set_backbone_frozen(model: torch.nn.Module, frozen: bool) -> None:
    """Freeze or unfreeze the ResNet50 backbone body."""
    requires_grad = not frozen
    for param in model.backbone.body.parameters():
        param.requires_grad = requires_grad
    state = "FROZEN" if frozen else "TRAINABLE"
    n_params = sum(1 for p in model.backbone.body.parameters())
    print(f"[RetinaNet] Backbone ({n_params} param groups): {state}")


# ── Training utilities ────────────────────────────────────────────────────────


def _train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler,
    epoch: int,
    total_epochs: int,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = len(loader)
    t0 = time.time()

    for batch_idx, (images, targets) in enumerate(loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == n_batches:
            cls_l = loss_dict.get("classification", torch.tensor(0.0)).item()
            reg_l = loss_dict.get("bbox_regression", torch.tensor(0.0)).item()
            elapsed = time.time() - t0
            print(
                f"  [epoch {epoch}] batch {batch_idx+1}/{n_batches}"
                f"  cls={cls_l:.4f}  reg={reg_l:.4f}"
                f"  total={loss.item():.4f}  [{elapsed:.0f}s]"
            )

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def _val_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    scaler,
) -> float:
    """Compute validation loss by temporarily switching to train mode."""
    model.train()
    total_loss = 0.0
    n_batches = len(loader)
    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                loss_dict = model(images, targets)
        else:
            loss_dict = model(images, targets)
        total_loss += sum(loss_dict.values()).item()
    model.eval()
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def _val_det_stats(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    score_thresh: float,
) -> Dict:
    """Quick detection stats for monitoring (no mAP, fast)."""
    model.eval()
    total_dets = 0
    max_score = 0.0
    n_images = 0
    per_class_dets = {c: 0 for c in CLASS_IDS}

    for images, _ in loader:
        images = [img.to(device) for img in images]
        preds = model(images)
        for pred in preds:
            n_images += 1
            scores = pred["scores"]
            labels = pred["labels"]
            keep = scores >= score_thresh
            n_det = int(keep.sum().item())
            total_dets += n_det
            if scores.numel() > 0:
                max_score = max(max_score, float(scores.max().item()))
            for c in CLASS_IDS:
                per_class_dets[c] += int(((labels == c) & keep).sum().item())

    mean_dets = total_dets / max(n_images, 1)
    return {
        "total_dets": total_dets,
        "mean_dets_per_image": mean_dets,
        "max_score": max_score,
        "per_class_dets": per_class_dets,
        "n_images": n_images,
    }


try:
    from torchmetrics.detection.mean_ap import MeanAveragePrecision as _TM_MAP

    _TM_OK = True
except ImportError:
    _TM_OK = False

_TM_RUNTIME_WARNED = False


def _iou_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
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


def _ap50_manual(cls_id: int, preds: List[Dict], targets: List[Dict]) -> float:
    det_scores, det_tp = [], []
    n_gt = 0
    for pred, tgt in zip(preds, targets):
        gt_m = tgt["labels"] == cls_id
        gt_b = tgt["boxes"][gt_m].float()
        n_gt += int(gt_m.sum())
        pd_m = pred["labels"] == cls_id
        if not pd_m.any():
            continue
        pb = pred["boxes"][pd_m].float()
        ps = pred["scores"][pd_m].float()
        if gt_b.shape[0] == 0:
            det_scores += ps.tolist()
            det_tp += [0] * pb.shape[0]
            continue
        iou = _iou_matrix(pb, gt_b)
        matched = torch.full((gt_b.shape[0],), False)
        for idx in ps.argsort(descending=True).tolist():
            best_iou, best_j = iou[idx].max(dim=0)
            det_scores.append(float(ps[idx]))
            if float(best_iou) >= 0.50 and not matched[int(best_j)]:
                matched[int(best_j)] = True
                det_tp.append(1)
            else:
                det_tp.append(0)
    if n_gt == 0:
        return float("nan")
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
    ap_val = 0.0
    for t in [i / 10 for i in range(11)]:
        at = [p for p, r in zip(precs, recs) if r >= t]
        ap_val += (max(at) if at else 0.0) / 11
    return ap_val


def _manual_map50_result(preds: List[Dict], targets: List[Dict]) -> Dict:
    result = {}
    valid = []
    for c in CLASS_IDS:
        ap = _ap50_manual(c, preds, targets)
        value = round(ap, 6) if not math.isnan(ap) else -1.0
        result[f"AP50_{LABEL_MAP[c]}"] = value
        if value >= 0.0:
            valid.append(value)
    result["mAP50"] = round(sum(valid) / max(len(valid), 1), 6)
    return result


@torch.no_grad()
def _val_map50(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    score_thresh: float,
) -> Dict:
    """Compute mAP50 and per-class AP50 on validation set.

    Uses score_thresh=0.01 regardless of training threshold so results
    are always comparable across runs.
    """
    model.eval()
    _eval_thresh = 0.01  # fixed low threshold for fair mAP comparison
    all_preds, all_targets = [], []

    for images, targets in loader:
        images = [img.to(device) for img in images]
        preds = model(images)
        for pred, tgt in zip(preds, targets):
            scores = pred["scores"]
            keep = scores >= _eval_thresh
            all_preds.append(
                {
                    "boxes": pred["boxes"][keep].cpu(),
                    "labels": pred["labels"][keep].cpu(),
                    "scores": scores[keep].cpu(),
                }
            )
            all_targets.append(
                {
                    "boxes": tgt["boxes"].cpu(),
                    "labels": tgt["labels"].cpu(),
                }
            )

    if not all_preds:
        return {"mAP50": 0.0, **{f"AP50_{LABEL_MAP[c]}": 0.0 for c in CLASS_IDS}}

    global _TM_RUNTIME_WARNED
    if _TM_OK:
        try:
            # class_metrics=True required for map_per_class tensor to be populated.
            # Without it, torchmetrics returns scalar -1 regardless of iou_thresholds.
            try:
                metric = _TM_MAP(
                    box_format="xyxy",
                    iou_type="bbox",
                    max_detection_thresholds=[1, 10, 300],
                    class_metrics=True,
                )
            except TypeError:
                metric = _TM_MAP(
                    box_format="xyxy",
                    iou_type="bbox",
                    max_detection_thresholds=[1, 10, 300],
                )
            try:
                metric.warn_on_many_detections = False
            except AttributeError:
                pass

            metric.update(all_preds, all_targets)
            raw = metric.compute()

            map50 = float(raw.get("map_50", raw.get("map", torch.tensor(0.0))).item())
            result = {"mAP50": round(map50, 6)}

            per_cls = raw.get("map_per_class", None)
            if (
                per_cls is not None
                and isinstance(per_cls, torch.Tensor)
                and per_cls.ndim == 1
                and per_cls.shape[0] >= len(CLASS_IDS)
            ):
                for i, c in enumerate(CLASS_IDS):
                    result[f"AP50_{LABEL_MAP[c]}"] = round(float(per_cls[i].item()), 6)
                return result
        except Exception as e:
            if not _TM_RUNTIME_WARNED:
                print(
                    "[Stage1C-RetinaNet] WARN: torchmetrics mAP backend unavailable "
                    f"at runtime ({e}); using manual AP50 fallback."
                )
                _TM_RUNTIME_WARNED = True

    return _manual_map50_result(all_preds, all_targets)


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1C — RetinaNet/FPN DR lesion detector fine-tune"
    )
    parser.add_argument("--data_root", default=_DATA_ROOT)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--freeze_backbone_epochs",
        type=int,
        default=5,
        help="Freeze ResNet50 backbone for first N epochs, then unfreeze.",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=0,
        help="Limit images (0=all). Use small value for smoke test.",
    )
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--score_thresh", type=float, default=0.05)
    parser.add_argument(
        "--no_pretrained",
        action="store_true",
        help="Train from scratch (no COCO pretrained weights).",
    )
    parser.add_argument(
        "--no_amp", action="store_true", help="Disable automatic mixed precision."
    )
    parser.add_argument(
        "--hflip_prob",
        type=float,
        default=0.0,
        help="Horizontal flip probability. Keep 0.0 for R1a/v3.",
    )
    parser.add_argument(
        "--lr_schedule",
        default="none",
        choices=["none", "cosine"],
        help="LR schedule for phase 2 (after backbone unfreeze). "
        "cosine=CosineAnnealingLR from lr*0.1 to lr*1e-3.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Short run to verify pipeline. Sets max_images=8 if not set.",
    )
    parser.add_argument("--output_dir", default=_DEFAULT_OUT)
    args = parser.parse_args()

    # Smoke test override
    if args.smoke_test and args.max_images == 0:
        args.max_images = 8
        print("[Stage1C] SMOKE TEST — max_images=8")

    if args.hflip_prob > 0.0:
        print(
            f"[Stage1C] WARNING: hflip_prob={args.hflip_prob} is set.\n"
            "  R1a spec requires hflip_prob=0.0 (one variable per experiment).\n"
            "  Use hflip only in R1b after R1a passes mAP50>=0.05."
        )

    if args.seed is not None:
        import random, numpy as np

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"[Stage1C] seed={args.seed}")

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, "retinanet_best.pt")
    log_csv = os.path.join(args.output_dir, "training_log.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available() and not args.no_amp
    try:
        scaler = torch.amp.GradScaler("cuda") if use_amp else None
    except TypeError:
        scaler = torch.cuda.amp.GradScaler() if use_amp else None

    print(f"[Stage1C] device={device}  amp={use_amp}")
    print(f"[Stage1C] output_dir={args.output_dir}")

    # Datasets — no hflip for R1a
    max_img = args.max_images if args.max_images > 0 else None
    train_ds = DRDetectionDataset(
        split="train",
        data_root=args.data_root,
        max_images=max_img,
        hflip_prob=args.hflip_prob,
    )
    val_ds = DRDetectionDataset(
        split="valid",
        data_root=args.data_root,
        max_images=max_img,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    print(f"[Stage1C] train={len(train_ds)}  val={len(val_ds)}")

    # Model — num_classes=NUM_CLASSES+1 because labels are 1-indexed {1..4}
    # torchvision indexes one-hot target by label value directly, so max label (4)
    # must be < num_classes. With num_classes=5, valid column indices are 0-4.
    _num_cls = NUM_CLASSES + 1  # = 5
    model = _build_retinanet(num_classes=_num_cls, pretrained=not args.no_pretrained)
    model.to(device)

    n_total = sum(p.numel() for p in model.parameters())
    print(f"[Stage1C] Total params: {n_total:,}")

    # Optimizer — only non-frozen params at start
    _set_backbone_frozen(model, frozen=True)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )

    # Per-epoch checkpoint epochs (always save these regardless of best)
    # epoch 13 included: known peak epoch from R1a_v2 run
    _SAVE_EPOCHS = {2, 3, 5, 10, 13, args.epochs}

    # Training loop — triple checkpoint: best_val_loss, best_val_mAP50, best_balanced
    # balanced_score = mAP50 + 0.5 * min_class_AP50
    # Prevents optimizing global mAP while collapsing one class (seen in R1a_v2: Hard_EX fell to 0.027)
    best_val_loss = float("inf")
    best_val_map50 = -1.0
    best_val_balanced = -1.0
    ckpt_best_loss = os.path.join(args.output_dir, "retinanet_best_val_loss.pt")
    ckpt_best_map = os.path.join(args.output_dir, "retinanet_best_map50.pt")
    ckpt_best_balanced = os.path.join(args.output_dir, "retinanet_best_balanced.pt")
    # retinanet_best.pt → best_balanced if available, else best_map50, else best_loss
    ckpt_path = os.path.join(args.output_dir, "retinanet_best.pt")
    log_rows = []
    scheduler = None  # set after backbone unfreeze if --lr_schedule cosine

    def _save_ckpt(path, epoch, val_loss, val_map50, val_balanced=-1.0, extra_tag=""):
        obj = {
            "epoch": epoch,
            "val_loss": val_loss,
            "val_mAP50": val_map50,
            "val_balanced": val_balanced,
            "model_state_dict": model.state_dict(),
            "config": {
                "num_classes": _num_cls,
                "pretrained": not args.no_pretrained,
                "hflip_prob": args.hflip_prob,
                "lr_schedule": args.lr_schedule,
                "freeze_backbone_epochs": args.freeze_backbone_epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "score_thresh": args.score_thresh,
                "model_type": "retinanet_resnet50_fpn",
                "label_map": LABEL_MAP,
            },
        }
        torch.save(obj, path)
        if extra_tag:
            print(f"  [ckpt] saved {extra_tag}: {os.path.basename(path)}")

    for epoch in range(1, args.epochs + 1):
        t_epoch = time.time()

        # Unfreeze backbone after freeze_backbone_epochs
        if epoch == args.freeze_backbone_epochs + 1:
            _set_backbone_frozen(model, frozen=False)
            _phase2_lr = args.lr * 0.1
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=_phase2_lr,
            )
            print(f"[Stage1C] Epoch {epoch}: backbone unfrozen, LR={_phase2_lr:.1e}")
            if args.lr_schedule == "cosine":
                _remaining = args.epochs - args.freeze_backbone_epochs
                _eta_min = args.lr * 1e-3
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=_remaining,
                    eta_min=_eta_min,
                )
                print(
                    f"[Stage1C] CosineAnnealingLR: T_max={_remaining}  eta_min={_eta_min:.2e}"
                )

        train_loss = _train_one_epoch(
            model, train_loader, optimizer, device, scaler, epoch, args.epochs
        )
        val_loss_v = _val_loss(model, val_loader, device, scaler)
        map_result = _val_map50(model, val_loader, device, args.score_thresh)
        stats = _val_det_stats(model, val_loader, device, args.score_thresh)

        val_map = map_result["mAP50"]
        # balanced_score = mAP50 + 0.5 * min_class_AP50
        # Guards against optimizing global mAP while collapsing a single class.
        _cls_ap_vals = [map_result.get(f"AP50_{LABEL_MAP[c]}", -1.0) for c in CLASS_IDS]
        _valid_cls = [v for v in _cls_ap_vals if v >= 0.0]
        _min_cls_ap = min(_valid_cls) if _valid_cls else -1.0
        val_balanced = (
            round(val_map + 0.5 * _min_cls_ap, 6) if _min_cls_ap >= 0.0 else -1.0
        )

        elapsed = time.time() - t_epoch

        is_best_loss = val_loss_v < best_val_loss
        is_best_map = val_map > best_val_map50 and val_map >= 0.0
        is_best_balanced = val_balanced > best_val_balanced and val_balanced >= 0.0

        markers = []
        if is_best_loss:
            markers.append("L")
        if is_best_map:
            markers.append("M")
        if is_best_balanced:
            markers.append("B")
        marker = f" [{''.join(markers)}]" if markers else ""

        per_cls_str = "  ".join(
            f"{LABEL_MAP[c]}:{stats['per_class_dets'][c]}" for c in CLASS_IDS
        )
        ap_str = "  ".join(
            f"{LABEL_MAP[c]}:{map_result.get(f'AP50_{LABEL_MAP[c]}', -1):.4f}"
            for c in CLASS_IDS
        )
        print(
            f"Epoch {epoch}/{args.epochs}"
            f"  train={train_loss:.4f}  val_loss={val_loss_v:.4f}"
            f"  val_mAP50={val_map:.4f}"
            f"  dets={stats['mean_dets_per_image']:.1f}/img"
            f"  [{elapsed:.0f}s]{marker}"
        )
        print(f"  per_class dets: {per_cls_str}")
        print(f"  per_class AP50: {ap_str}")

        if is_best_loss:
            best_val_loss = val_loss_v
            _save_ckpt(
                ckpt_best_loss,
                epoch,
                val_loss_v,
                val_map,
                val_balanced,
                "best_val_loss",
            )

        if is_best_map:
            best_val_map50 = val_map
            _save_ckpt(
                ckpt_best_map, epoch, val_loss_v, val_map, val_balanced, "best_mAP50"
            )

        if is_best_balanced:
            best_val_balanced = val_balanced
            _save_ckpt(
                ckpt_best_balanced,
                epoch,
                val_loss_v,
                val_map,
                val_balanced,
                "best_balanced",
            )

        # Per-epoch checkpoints for diagnostic use (epoch 13 = known peak from R1a_v2)
        if epoch in _SAVE_EPOCHS:
            ep_path = os.path.join(args.output_dir, f"retinanet_epoch_{epoch:02d}.pt")
            _save_ckpt(
                ep_path, epoch, val_loss_v, val_map, val_balanced, f"epoch_{epoch:02d}"
            )

        if scheduler is not None:
            scheduler.step()
            _cur_lr = scheduler.get_last_lr()[0]
            print(f"  [sched] LR after step: {_cur_lr:.2e}")

        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss_v, 6),
                "val_mAP50": round(val_map, 6),
                "val_balanced": val_balanced,
                **{
                    f"AP50_{LABEL_MAP[c]}": map_result.get(f"AP50_{LABEL_MAP[c]}", -1.0)
                    for c in CLASS_IDS
                },
                "mean_dets_per_image": round(stats["mean_dets_per_image"], 2),
                "max_score": round(stats["max_score"], 6),
                **{
                    f"dets_{LABEL_MAP[c]}": stats["per_class_dets"][c]
                    for c in CLASS_IDS
                },
            }
        )

    # Best checkpoint alias: balanced > map50 > loss
    import shutil

    if best_val_balanced >= 0.0 and os.path.isfile(ckpt_best_balanced):
        shutil.copy2(ckpt_best_balanced, ckpt_path)
        best_alias = f"best_balanced (score={best_val_balanced:.4f})"
    elif best_val_map50 >= 0.0 and os.path.isfile(ckpt_best_map):
        shutil.copy2(ckpt_best_map, ckpt_path)
        best_alias = f"best_map50 (mAP50={best_val_map50:.4f})"
    else:
        shutil.copy2(ckpt_best_loss, ckpt_path)
        best_alias = f"best_val_loss (fallback)"

    # Write log CSV
    if log_rows:
        with open(log_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
            writer.writeheader()
            writer.writerows(log_rows)

    print(f"\n[Stage1C] Done.")
    print(f"  best_val_loss    : {best_val_loss:.6f}  → {ckpt_best_loss}")
    print(f"  best_val_mAP50   : {best_val_map50:.6f}  → {ckpt_best_map}")
    print(f"  best_val_balanced: {best_val_balanced:.6f}  → {ckpt_best_balanced}")
    print(f"  retinanet_best   : {best_alias}  → {ckpt_path}")
    print(f"  log            : {log_csv}")
    print(f"\n[Stage1C] Next step: run audit")
    print(
        f"  python tools/audit_detector_stage1a.py \\\n"
        f"    --checkpoint {ckpt_path} \\\n"
        f"    --model_type retinanet \\\n"
        f"    --output_dir {os.path.join(args.output_dir + '_audit', '')}"
    )


if __name__ == "__main__":
    main()
