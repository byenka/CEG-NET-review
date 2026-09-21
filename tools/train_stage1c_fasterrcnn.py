#!/usr/bin/env python
"""
train_stage1c_fasterrcnn.py
Phase 19N Stage 1C — Faster R-CNN/FPN established detector.

RetinaNet stopped after R1a_v2 (mAP=0.0451) and R1a_v3 (mAP=0.0394) both failed
the detector gate (mAP50 >= 0.05, all class AP50 >= 0.05).

This script started as the historical R2a baseline and is now reused for the
r4a/r5-series Faster R-CNN variants, including the current strong baseline
`r5b_ext25`.

The detector uses torchvision Faster R-CNN/FPN with:
  - Pretrained COCO backbone (non-novel supporting component)
  - Small-object-aware anchors (MA lesions are ~8x8px at model input)
  - Same class mapping as all prior Stage 1 experiments
  - hflip_prob=0.0 (no augmentation change)
  - Balanced checkpoint: mAP50 + 0.5 * min_class_AP50

Label mapping (locked):
  ma -> 1 -> MA
  he -> 2 -> HE
  ex -> 3 -> Hard_EX
  se -> 4 -> Soft_EX
  background -> 0

Usage:
  # Smoke test
  python tools/train_stage1c_fasterrcnn.py --max_images 8 --epochs 1 --smoke_test

    # Reference r5b_ext25-style run
  python tools/train_stage1c_fasterrcnn.py \\
        --epochs 25 --batch_size 2 --lr 1e-4 \
    --freeze_backbone_epochs 5 \\
    --lr_schedule cosine \\
    --seed 42 \\
        --output_dir outputs/detector_stage1/stage1c_fasterrcnn_r5b_ext25
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

try:
    from src.data.ma_crop_sampler import MAZoomDataset
except ImportError:
    try:
        from lgovnet.src.data.ma_crop_sampler import MAZoomDataset
    except ImportError:
        # r5c MA-zoom augmentation module (src/data/ma_crop_sampler.py) is not
        # present in this tree — it was an isolated r5c experiment. It is only
        # needed when --ma_crop_prob > 0; the default (0.0, the r5b recipe) and
        # the deep-ensemble seeds run without it. See pembahasan/CHANGELOG_r5c.md.
        MAZoomDataset = None

try:
    from src.models.noise_aware_fpn import (
        attach_noise_aware_fpn,
        get_noise_gate_regularization,
    )
except ImportError:
    try:
        from lgovnet.src.models.noise_aware_fpn import (
            attach_noise_aware_fpn,
            get_noise_gate_regularization,
        )
    except ImportError:
        # r4a noise-aware FPN module (src/models/noise_aware_fpn.py) is not
        # present in this tree — the r4a experiment was closed. Only
        # --model_variant noiseaware_fpn needs it. The fallbacks below keep
        # baseline / roi_focal_loss training (incl. the r5b ensemble seeds)
        # behaviourally identical: get_noise_gate_regularization returns None,
        # exactly as the real module does for a non-noise-aware model.
        def attach_noise_aware_fpn(*args, **kwargs):
            raise RuntimeError(
                "--model_variant noiseaware_fpn requires "
                "src/models/noise_aware_fpn.py, which is not present in this "
                "tree (r4a experiment, closed). Use --model_variant baseline."
            )

        def get_noise_gate_regularization(model):
            return None


try:
    from src.models.fasterrcnn_roi_focal import enable_roi_focal_loss
except ImportError:
    from lgovnet.src.models.fasterrcnn_roi_focal import enable_roi_focal_loss

# ── Faster R-CNN imports ───────────────────────────────────────────────────────

try:
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.rpn import AnchorGenerator

    try:
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

        _WEIGHTS_V2 = True
    except ImportError:
        _WEIGHTS_V2 = False
except ImportError as e:
    raise ImportError(
        "torchvision not found. Install: pip install torchvision\n" f"Original: {e}"
    )

_DATA_ROOT = os.path.join(_root, "data")
_DEFAULT_OUT = os.path.join(
    _root, "outputs", "detector_stage1", "stage1c_fasterrcnn_run"
)

# Architecture-level defaults for dense tiny-lesion scenes.
_DEFAULT_DETECTIONS_PER_IMG = 300
_DEFAULT_RPN_PRE_NMS_TOP_N_TRAIN = 4000
_DEFAULT_RPN_POST_NMS_TOP_N_TRAIN = 2000
_DEFAULT_RPN_PRE_NMS_TOP_N_TEST = 3000
_DEFAULT_RPN_POST_NMS_TOP_N_TEST = 1500

LABEL_MAP: Dict[int, str] = {i + 1: n for i, n in enumerate(LABEL_NAMES)}
CLASS_IDS = list(LABEL_MAP.keys())  # [1, 2, 3, 4]

# Small-object anchor sizes for lesion data.
# MA median bbox is ~8x8px at model input. Default torchvision sizes start at 32px
# (too large). Customised to cover small lesions at all FPN levels (P2-P6).
_ANCHOR_SIZES = ((8,), (16,), (32,), (64,), (128,))
_ANCHOR_RATIOS = ((0.5, 1.0, 2.0),) * len(_ANCHOR_SIZES)


# ── Model factory ─────────────────────────────────────────────────────────────


def _build_fasterrcnn(
    num_classes: int,
    pretrained: bool,
    score_thresh: float,
    detections_per_img: int,
    rpn_pre_nms_top_n_train: int,
    rpn_post_nms_top_n_train: int,
    rpn_pre_nms_top_n_test: int,
    rpn_post_nms_top_n_test: int,
    model_variant: str = "baseline",
    noise_gate_alpha: float = 0.2,
    noise_hidden_channels: int = 16,
    noise_last_bias: float = -4.0,
    noise_scatter_kernel: int = 15,
    roi_focal_gamma: float = 2.0,
    roi_focal_alpha_mode: str = "batch_balanced",
) -> torch.nn.Module:
    """Load pretrained Faster R-CNN/FPN and replace classification head."""
    anchor_gen = AnchorGenerator(
        sizes=_ANCHOR_SIZES,
        aspect_ratios=_ANCHOR_RATIOS,
    )
    model_kwargs = {
        "rpn_anchor_generator": anchor_gen,
        "box_score_thresh": score_thresh,
        "box_detections_per_img": detections_per_img,
        "rpn_pre_nms_top_n_train": rpn_pre_nms_top_n_train,
        "rpn_post_nms_top_n_train": rpn_post_nms_top_n_train,
        "rpn_pre_nms_top_n_test": rpn_pre_nms_top_n_test,
        "rpn_post_nms_top_n_test": rpn_post_nms_top_n_test,
    }

    if pretrained:
        if _WEIGHTS_V2:
            weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
            model = fasterrcnn_resnet50_fpn(
                weights=weights,
                **model_kwargs,
            )
        else:
            model = fasterrcnn_resnet50_fpn(
                pretrained=True,
                **model_kwargs,
            )
        print("[FasterRCNN] Loaded pretrained COCO_V1 weights")
    else:
        if _WEIGHTS_V2:
            model = fasterrcnn_resnet50_fpn(
                weights=None,
                **model_kwargs,
            )
        else:
            model = fasterrcnn_resnet50_fpn(
                pretrained=False,
                **model_kwargs,
            )
        print("[FasterRCNN] No pretrained weights")

    # Replace box predictor head (num_classes includes background at index 0).
    # Our labels are 1-indexed {1..4}, background is 0 → num_classes = 5.
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    print(
        f"[FasterRCNN] Box predictor replaced: 91 COCO → {num_classes} classes"
        f"  (labels 1-4: {', '.join(LABEL_NAMES)}; 0=background)"
    )
    print(f"[FasterRCNN] Anchor sizes: {_ANCHOR_SIZES}  ratios: {_ANCHOR_RATIOS[0]}")
    print(
        "[FasterRCNN] Architecture config: "
        f"box_score_thresh={model.roi_heads.score_thresh}  "
        f"box_detections_per_img={model.roi_heads.detections_per_img}  "
        f"rpn_pre_nms(train/test)=({model.rpn._pre_nms_top_n['training']}/{model.rpn._pre_nms_top_n['testing']})  "
        f"rpn_post_nms(train/test)=({model.rpn._post_nms_top_n['training']}/{model.rpn._post_nms_top_n['testing']})"
    )
    if model_variant == "roi_focal_loss":
        enable_roi_focal_loss(
            gamma=roi_focal_gamma,
            alpha_mode=roi_focal_alpha_mode,
        )
        print(
            "[FasterRCNN] Variant: ROI focal loss "
            f"(gamma={roi_focal_gamma:.2f}, alpha_mode={roi_focal_alpha_mode})"
        )
    if model_variant == "noiseaware_fpn":
        attach_noise_aware_fpn(
            model,
            alpha=noise_gate_alpha,
            hidden_channels=noise_hidden_channels,
            out_bias_init=noise_last_bias,
            scatter_kernel_size=noise_scatter_kernel,
        )
        print(
            "[FasterRCNN] Variant: Noise/Artifact-Aware FPN "
            f"(alpha={noise_gate_alpha:.3f}, hidden={noise_hidden_channels}, "
            f"bias={noise_last_bias:.2f}, scatter_k={noise_scatter_kernel})"
        )
    return model


def _set_backbone_frozen(model: torch.nn.Module, frozen: bool) -> None:
    requires_grad = not frozen
    for param in model.backbone.body.parameters():
        param.requires_grad = requires_grad
    state = "FROZEN" if frozen else "TRAINABLE"
    n = sum(1 for _ in model.backbone.body.parameters())
    print(f"[FasterRCNN] Backbone ({n} param groups): {state}")


# ── Training utilities ────────────────────────────────────────────────────────


def _train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    scaler,
    epoch,
    total_epochs,
    noise_reg_weight,
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
                gate_reg = get_noise_gate_regularization(model)
                if gate_reg is not None and noise_reg_weight > 0.0:
                    loss = loss + noise_reg_weight * gate_reg
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            gate_reg = get_noise_gate_regularization(model)
            if gate_reg is not None and noise_reg_weight > 0.0:
                loss = loss + noise_reg_weight * gate_reg
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == n_batches:
            elapsed = time.time() - t0
            cls_l = loss_dict.get("loss_classifier", torch.tensor(0.0)).item()
            box_l = loss_dict.get("loss_box_reg", torch.tensor(0.0)).item()
            rpn_l = loss_dict.get("loss_rpn_box_reg", torch.tensor(0.0)).item()
            obj_l = loss_dict.get("loss_objectness", torch.tensor(0.0)).item()
            print(
                f"  [epoch {epoch}] batch {batch_idx+1}/{n_batches}"
                f"  cls={cls_l:.4f}  box={box_l:.4f}"
                f"  rpn_obj={obj_l:.4f}  rpn_box={rpn_l:.4f}"
                f"  total={loss.item():.4f}  [{elapsed:.0f}s]"
            )
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def _val_loss(model, loader, device, scaler, noise_reg_weight) -> float:
    # Torchvision detection losses are only available in train() mode.
    # Fix the RNG during validation to reduce proposal / ROI subsampling noise.
    rng_devices = []
    if device.type == "cuda":
        rng_devices = [
            device.index if device.index is not None else torch.cuda.current_device()
        ]

    with torch.random.fork_rng(devices=rng_devices):
        torch.manual_seed(12345)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(12345)

        model.train()
        total_loss = 0.0
        n_batches = len(loader)
        for images, targets in loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    loss_dict = model(images, targets)
                    loss = sum(loss_dict.values())
                    gate_reg = get_noise_gate_regularization(model)
                    if gate_reg is not None and noise_reg_weight > 0.0:
                        loss = loss + noise_reg_weight * gate_reg
            else:
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())
                gate_reg = get_noise_gate_regularization(model)
                if gate_reg is not None and noise_reg_weight > 0.0:
                    loss = loss + noise_reg_weight * gate_reg
            total_loss += loss.item()
    model.eval()
    return total_loss / max(n_batches, 1)


def _noise_reg_weight_for_epoch(base_weight: float, epoch: int) -> float:
    if base_weight <= 0.0:
        return 0.0
    if epoch <= 5:
        return base_weight
    if epoch <= 10:
        return base_weight * 0.5
    return base_weight * 0.1


@torch.no_grad()
def _val_det_stats(model, loader, device, score_thresh) -> Dict:
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
            total_dets += int(keep.sum().item())
            if scores.numel() > 0:
                max_score = max(max_score, float(scores.max().item()))
            for c in CLASS_IDS:
                per_class_dets[c] += int(((labels == c) & keep).sum().item())
    return {
        "total_dets": total_dets,
        "mean_dets_per_image": total_dets / max(n_images, 1),
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
    """11-point interpolated AP50 for a single class."""
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
    tp_cum = fp_cum = 0
    precs, recs = [], []
    for i in order:
        tp_cum += det_tp[i]
        fp_cum += 1 - det_tp[i]
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
def _val_map50(model, loader, device, score_thresh) -> Dict:
    """Compute mAP50 and per-class AP50. Uses score_thresh=0.01 for comparability."""
    model.eval()
    _eval_thresh = 0.01
    all_preds, all_targets = [], []
    for images, targets in loader:
        images = [img.to(device) for img in images]
        preds = model(images)
        for pred, tgt in zip(preds, targets):
            keep = pred["scores"] >= _eval_thresh
            all_preds.append(
                {
                    "boxes": pred["boxes"][keep].cpu(),
                    "labels": pred["labels"][keep].cpu(),
                    "scores": pred["scores"][keep].cpu(),
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
                    "[Stage1C-FasterRCNN] WARN: torchmetrics mAP backend unavailable "
                    f"at runtime ({e}); using manual AP50 fallback."
                )
                _TM_RUNTIME_WARNED = True

    return _manual_map50_result(all_preds, all_targets)


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1C — Faster R-CNN/FPN DR lesion detector"
    )
    parser.add_argument("--data_root", default=_DATA_ROOT)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--freeze_backbone_epochs",
        type=int,
        default=5,
        help="Freeze ResNet50 body for first N epochs, then unfreeze.",
    )
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--score_thresh", type=float, default=0.01)
    parser.add_argument(
        "--detections_per_img",
        type=int,
        default=_DEFAULT_DETECTIONS_PER_IMG,
        help="ROI head output cap per image (architecture-level).",
    )
    parser.add_argument(
        "--rpn_pre_nms_top_n_train",
        type=int,
        default=_DEFAULT_RPN_PRE_NMS_TOP_N_TRAIN,
    )
    parser.add_argument(
        "--rpn_post_nms_top_n_train",
        type=int,
        default=_DEFAULT_RPN_POST_NMS_TOP_N_TRAIN,
    )
    parser.add_argument(
        "--rpn_pre_nms_top_n_test",
        type=int,
        default=_DEFAULT_RPN_PRE_NMS_TOP_N_TEST,
    )
    parser.add_argument(
        "--rpn_post_nms_top_n_test",
        type=int,
        default=_DEFAULT_RPN_POST_NMS_TOP_N_TEST,
    )
    parser.add_argument(
        "--model_variant",
        default="baseline",
        choices=["baseline", "noiseaware_fpn", "roi_focal_loss"],
        help=(
            "Baseline Faster R-CNN/FPN, r4a Noise/Artifact-Aware FPN, "
            "or r5a ROI focal loss."
        ),
    )
    parser.add_argument("--noise_gate_alpha", type=float, default=0.2)
    parser.add_argument("--noise_hidden_channels", type=int, default=16)
    parser.add_argument("--noise_last_bias", type=float, default=-4.0)
    parser.add_argument(
        "--noise_scatter_kernel",
        type=int,
        default=15,
        help=(
            "HF scatter window size (must be odd) for MA-vs-noise disambiguation. "
            "Larger = more separation between isolated blobs (MA) and continuous texture (noise). "
            "Recommended: 11 / 15 / 21. Default: 15."
        ),
    )
    parser.add_argument(
        "--noise_reg_weight",
        type=float,
        default=1e-3,
        help="Base gate identity regularization weight for noise-aware FPN.",
    )
    parser.add_argument(
        "--roi_focal_gamma",
        type=float,
        default=2.0,
        help="Focusing parameter for r5a ROI focal loss.",
    )
    parser.add_argument(
        "--roi_focal_alpha_mode",
        default="batch_balanced",
        choices=["none", "batch_balanced"],
        help="Per-batch class balancing mode for r5a ROI focal loss.",
    )
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument(
        "--hflip_prob",
        type=float,
        default=0.0,
        help="Keep 0.0 for the baseline recipe (no augmentation variable change).",
    )
    parser.add_argument(
        "--ma_crop_prob",
        type=float,
        default=0.0,
        help=(
            "r5c: probability of replacing a full-image sample with an "
            "MA-centric zoom-in crop. 0.0 = off (default). 0.5 = recommended."
        ),
    )
    parser.add_argument(
        "--ma_crop_size",
        type=int,
        default=512,
        help=(
            "r5c: side length (px) of the MA-centric crop square. "
            "Smaller crops give stronger zoom; 512 is recommended for OIADDR."
        ),
    )
    parser.add_argument(
        "--lr_schedule",
        default="cosine",
        choices=["none", "cosine"],
        help="LR schedule for phase 2. Default: cosine.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--output_dir", default=_DEFAULT_OUT)
    args = parser.parse_args()

    if args.smoke_test and args.max_images == 0:
        args.max_images = 8
        print("[Stage1C-FasterRCNN] SMOKE TEST — max_images=8")

    if args.hflip_prob > 0.0:
        print(
            f"[Stage1C-FasterRCNN] WARNING: hflip_prob={args.hflip_prob}. "
            "Baseline recipe keeps hflip_prob=0.0 (one variable per experiment)."
        )

    if args.seed is not None:
        import random, numpy as np

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"[Stage1C-FasterRCNN] seed={args.seed}")

    os.makedirs(args.output_dir, exist_ok=True)
    log_csv = os.path.join(args.output_dir, "training_log.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available() and not args.no_amp
    try:
        scaler = torch.amp.GradScaler("cuda") if use_amp else None
    except TypeError:
        scaler = torch.cuda.amp.GradScaler() if use_amp else None

    print(f"[Stage1C-FasterRCNN] device={device}  amp={use_amp}")
    print(f"[Stage1C-FasterRCNN] output_dir={args.output_dir}")
    print(f"[Stage1C-FasterRCNN] model_variant={args.model_variant}")

    max_img = args.max_images if args.max_images > 0 else None
    train_ds = DRDetectionDataset(
        split="train",
        data_root=args.data_root,
        max_images=max_img,
        hflip_prob=args.hflip_prob,
    )
    if args.ma_crop_prob > 0.0:
        if MAZoomDataset is None:
            raise RuntimeError(
                "--ma_crop_prob > 0 requires src/data/ma_crop_sampler.py "
                "(MAZoomDataset, the r5c experiment), which is not present in "
                "this tree. Run with --ma_crop_prob 0.0 (the r5b recipe), or "
                "restore the module — full spec in pembahasan/CHANGELOG_r5c.md."
            )
        train_ds = MAZoomDataset(
            train_ds,
            ma_crop_prob=args.ma_crop_prob,
            ma_crop_size=args.ma_crop_size,
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
    print(f"[Stage1C-FasterRCNN] train={len(train_ds)}  val={len(val_ds)}")

    # num_classes=NUM_CLASSES+1: labels are 1-indexed {1..4}, background=0
    _num_cls = NUM_CLASSES + 1  # = 5
    model = _build_fasterrcnn(
        num_classes=_num_cls,
        pretrained=not args.no_pretrained,
        score_thresh=args.score_thresh,
        detections_per_img=args.detections_per_img,
        rpn_pre_nms_top_n_train=args.rpn_pre_nms_top_n_train,
        rpn_post_nms_top_n_train=args.rpn_post_nms_top_n_train,
        rpn_pre_nms_top_n_test=args.rpn_pre_nms_top_n_test,
        rpn_post_nms_top_n_test=args.rpn_post_nms_top_n_test,
        model_variant=args.model_variant,
        noise_gate_alpha=args.noise_gate_alpha,
        noise_hidden_channels=args.noise_hidden_channels,
        noise_last_bias=args.noise_last_bias,
        noise_scatter_kernel=args.noise_scatter_kernel,
        roi_focal_gamma=args.roi_focal_gamma,
        roi_focal_alpha_mode=args.roi_focal_alpha_mode,
    )
    model.to(device)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[Stage1C-FasterRCNN] Total params: {n_total:,}")

    _set_backbone_frozen(model, frozen=True)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )

    # Per-epoch diagnostic checkpoints (epoch 13 included: known RetinaNet peak)
    _SAVE_EPOCHS = {2, 3, 5, 10, 13, args.epochs}

    best_val_loss = float("inf")
    best_val_map50 = -1.0
    best_val_balanced = -1.0
    ckpt_best_loss = os.path.join(args.output_dir, "fasterrcnn_best_val_loss.pt")
    ckpt_best_map = os.path.join(args.output_dir, "fasterrcnn_best_map50.pt")
    ckpt_best_balanced = os.path.join(args.output_dir, "fasterrcnn_best_balanced.pt")
    ckpt_path = os.path.join(args.output_dir, "fasterrcnn_best.pt")
    log_rows = []
    scheduler = None

    if args.model_variant == "noiseaware_fpn":
        architecture_label = "Noise/Artifact-Aware FPN"
    elif args.model_variant == "roi_focal_loss":
        architecture_label = "Faster R-CNN/FPN + ROI focal loss"
    elif args.ma_crop_prob > 0.0:
        architecture_label = f"Faster R-CNN/FPN + MA-zoom (p={args.ma_crop_prob:.2f} sz={args.ma_crop_size})"
    else:
        architecture_label = "Faster R-CNN/FPN baseline"

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
                "detections_per_img": args.detections_per_img,
                "rpn_pre_nms_top_n_train": args.rpn_pre_nms_top_n_train,
                "rpn_post_nms_top_n_train": args.rpn_post_nms_top_n_train,
                "rpn_pre_nms_top_n_test": args.rpn_pre_nms_top_n_test,
                "rpn_post_nms_top_n_test": args.rpn_post_nms_top_n_test,
                "anchor_sizes": _ANCHOR_SIZES,
                "anchor_ratios": _ANCHOR_RATIOS,
                "model_type": "fasterrcnn_resnet50_fpn",
                "model_variant": args.model_variant,
                "architecture_label": architecture_label,
                "noise_gate_alpha": args.noise_gate_alpha,
                "noise_hidden_channels": args.noise_hidden_channels,
                "noise_last_bias": args.noise_last_bias,
                "noise_scatter_kernel": args.noise_scatter_kernel,
                "noise_reg_weight": args.noise_reg_weight,
                "roi_focal_gamma": args.roi_focal_gamma,
                "roi_focal_alpha_mode": args.roi_focal_alpha_mode,
                "ma_crop_prob": args.ma_crop_prob,
                "ma_crop_size": args.ma_crop_size,
                "label_map": LABEL_MAP,
            },
        }
        torch.save(obj, path)
        if extra_tag:
            print(f"  [ckpt] saved {extra_tag}: {os.path.basename(path)}")

    for epoch in range(1, args.epochs + 1):
        t_epoch = time.time()

        if epoch == args.freeze_backbone_epochs + 1:
            _set_backbone_frozen(model, frozen=False)
            _phase2_lr = args.lr * 0.1
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=_phase2_lr,
            )
            print(
                f"[Stage1C-FasterRCNN] Epoch {epoch}: backbone unfrozen, LR={_phase2_lr:.1e}"
            )
            if args.lr_schedule == "cosine":
                _remaining = args.epochs - args.freeze_backbone_epochs
                _eta_min = args.lr * 1e-3
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=_remaining,
                    eta_min=_eta_min,
                )
                print(
                    f"[Stage1C-FasterRCNN] CosineAnnealingLR: T_max={_remaining}  eta_min={_eta_min:.2e}"
                )

        noise_reg_weight = _noise_reg_weight_for_epoch(args.noise_reg_weight, epoch)
        train_loss = _train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler,
            epoch,
            args.epochs,
            noise_reg_weight,
        )
        val_loss_v = _val_loss(model, val_loader, device, scaler, noise_reg_weight)
        map_result = _val_map50(model, val_loader, device, args.score_thresh)
        stats = _val_det_stats(model, val_loader, device, args.score_thresh)

        val_map = map_result["mAP50"]
        _cls_ap = [map_result.get(f"AP50_{LABEL_MAP[c]}", -1.0) for c in CLASS_IDS]
        _valid = [v for v in _cls_ap if v >= 0.0]
        _min_cls = min(_valid) if _valid else -1.0
        val_balanced = round(val_map + 0.5 * _min_cls, 6) if _min_cls >= 0.0 else -1.0

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
            f"  val_mAP50={val_map:.4f}  balanced={val_balanced:.4f}"
            f"  dets={stats['mean_dets_per_image']:.1f}/img"
            f"  gate_reg={noise_reg_weight:.1e}"
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
        if epoch in _SAVE_EPOCHS:
            ep_path = os.path.join(args.output_dir, f"fasterrcnn_epoch_{epoch:02d}.pt")
            _save_ckpt(
                ep_path, epoch, val_loss_v, val_map, val_balanced, f"epoch_{epoch:02d}"
            )

        if scheduler is not None:
            scheduler.step()
            print(f"  [sched] LR after step: {scheduler.get_last_lr()[0]:.2e}")

        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss_v, 6),
                "val_mAP50": round(val_map, 6),
                "val_balanced": val_balanced,
                "noise_reg_weight": round(noise_reg_weight, 8),
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

    import shutil

    if best_val_balanced >= 0.0 and os.path.isfile(ckpt_best_balanced):
        shutil.copy2(ckpt_best_balanced, ckpt_path)
        best_alias = f"best_balanced (score={best_val_balanced:.4f})"
    elif best_val_map50 >= 0.0 and os.path.isfile(ckpt_best_map):
        shutil.copy2(ckpt_best_map, ckpt_path)
        best_alias = f"best_map50 (mAP50={best_val_map50:.4f})"
    else:
        shutil.copy2(ckpt_best_loss, ckpt_path)
        best_alias = "best_val_loss (fallback)"

    if log_rows:
        with open(log_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
            writer.writeheader()
            writer.writerows(log_rows)

    print(f"\n[Stage1C-FasterRCNN] Done.")
    print(f"  best_val_loss    : {best_val_loss:.6f}  → {ckpt_best_loss}")
    print(f"  best_val_mAP50   : {best_val_map50:.6f}  → {ckpt_best_map}")
    print(f"  best_val_balanced: {best_val_balanced:.6f}  → {ckpt_best_balanced}")
    print(f"  fasterrcnn_best  : {best_alias}  → {ckpt_path}")
    print(f"  log              : {log_csv}")
    print(
        "  note             : val_loss is a train-mode surrogate for detector models; "
        "use best_val_mAP50 for deployment checkpoint selection"
    )
    print(f"\n[Stage1C-FasterRCNN] Next step: run audit")
    print(
        f"  python tools/audit_detector_stage1a.py \\\n"
        f"    --checkpoint {ckpt_path} \\\n"
        f"    --model_type fasterrcnn \\\n"
        f"    --output_dir {args.output_dir + '_audit/'}"
    )


if __name__ == "__main__":
    main()
