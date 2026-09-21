"""
multiclass_inference.py
Phase 19N Pilar 1 — REAL CLASS PROBABILITIES (gap G1)

Standard torchvision Faster R-CNN inference discards the per-class probability
vector and keeps only `(box, label, score)` after per-class NMS. Downstream
SDCA then reads a runner-up class that is purely synthetic (the export pipeline
fills non-predicted classes with an equal residual `(1-score)/(C-1)`).

Result (see target_gap_implementasi_01.md §2A/2B): SDCA collapses to a
4-valued quantized lookup, and the C1/C2 ablations cannot find KG signal.

This module provides `predict_with_class_probs(model, images, device)` which
runs Faster R-CNN and returns, for each kept box, the FULL multinomial
probability vector over the lesion classes — recovered by tracking which
proposal index each kept box came from through filtering and NMS.

The recovered `class_probs` is what SDCA / e_pred / the LGG need to read to
produce a real, continuous-resolution signal.

Scope: Faster R-CNN (torchvision FasterRCNN, weights >= 0.13). RetinaNet keeps
the synthetic fallback (it is the secondary detector; its impact is small).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import boxes as box_ops

# ── Faster R-CNN: postprocess that retains class_probs ───────────────────────


@torch.no_grad()
def _frcnn_postprocess_with_probs(
    roi_heads,
    class_logits: Tensor,
    box_regression: Tensor,
    proposals: List[Tensor],
    image_shapes: List[Tuple[int, int]],
) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]:
    """Reimplements RoIHeads.postprocess_detections and ALSO returns, for every
    kept box, the full per-lesion-class probability vector.

    The trick: every (proposal, class) candidate is flattened with its proposal
    index tracked in `prop_idx_flat`. After score-threshold + small-box removal
    + per-class NMS, `prop_idx_kept` tells us which proposal each surviving box
    came from. We then index the softmax matrix `scores` by that proposal index
    to recover the full multinomial row.

    Returns four per-image lists:
        boxes_list       [N_kept, 4]
        scores_list      [N_kept]
        labels_list      [N_kept]                 (1-indexed lesion class id)
        class_probs_list [N_kept, num_lesion_classes]   (background col dropped)
    """
    device = class_logits.device
    num_classes = class_logits.shape[-1]  # lesion + background

    boxes_per_image = [p.shape[0] for p in proposals]
    pred_boxes = roi_heads.box_coder.decode(box_regression, proposals)
    pred_scores = F.softmax(class_logits, -1)  # [Σ_props, C]

    pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
    pred_scores_list = pred_scores.split(boxes_per_image, 0)

    out_boxes: List[Tensor] = []
    out_scores: List[Tensor] = []
    out_labels: List[Tensor] = []
    out_probs: List[Tensor] = []

    for boxes, scores, image_shape in zip(
        pred_boxes_list, pred_scores_list, image_shapes
    ):
        # boxes: [N_props, C, 4]   scores: [N_props, C]
        boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

        n_props = scores.shape[0]
        prop_idx = torch.arange(n_props, device=device).view(-1, 1).expand_as(scores)
        labels_full = (
            torch.arange(num_classes, device=device).view(1, -1).expand_as(scores)
        )

        # Drop background column (index 0)
        boxes = boxes[:, 1:]
        scores_fg = scores[:, 1:]
        labels_full = labels_full[:, 1:]
        prop_idx = prop_idx[:, 1:]

        boxes_flat = boxes.reshape(-1, 4)
        scores_flat = scores_fg.reshape(-1)
        labels_flat = labels_full.reshape(-1)
        prop_idx_flat = prop_idx.reshape(-1)

        # Score threshold
        keep = torch.where(scores_flat > roi_heads.score_thresh)[0]
        boxes_flat, scores_flat = boxes_flat[keep], scores_flat[keep]
        labels_flat, prop_idx_flat = labels_flat[keep], prop_idx_flat[keep]

        # Remove tiny boxes
        keep = box_ops.remove_small_boxes(boxes_flat, min_size=1e-2)
        boxes_flat, scores_flat = boxes_flat[keep], scores_flat[keep]
        labels_flat, prop_idx_flat = labels_flat[keep], prop_idx_flat[keep]

        # Per-class NMS
        keep = box_ops.batched_nms(
            boxes_flat, scores_flat, labels_flat, roi_heads.nms_thresh
        )
        keep = keep[: roi_heads.detections_per_img]
        boxes_kept = boxes_flat[keep]
        scores_kept = scores_flat[keep]
        labels_kept = labels_flat[keep]
        prop_idx_kept = prop_idx_flat[keep]

        # Recover full lesion-class probabilities per kept box.
        # scores: [N_props, C]; index by proposal, drop background col.
        probs_full = scores[prop_idx_kept][:, 1:]  # [N_kept, C-1]

        out_boxes.append(boxes_kept)
        out_scores.append(scores_kept)
        out_labels.append(labels_kept)
        out_probs.append(probs_full)

    return out_boxes, out_scores, out_labels, out_probs


def _resize_boxes(
    boxes: Tensor, src_size: Tuple[int, int], dst_size: Tuple[int, int]
) -> Tensor:
    """Linear box rescale from src_size (h, w) → dst_size (h, w)."""
    if boxes.numel() == 0:
        return boxes
    ratio_h = float(dst_size[0]) / float(src_size[0])
    ratio_w = float(dst_size[1]) / float(src_size[1])
    x1, y1, x2, y2 = boxes.unbind(1)
    return torch.stack((x1 * ratio_w, y1 * ratio_h, x2 * ratio_w, y2 * ratio_h), dim=1)


@torch.no_grad()
def predict_with_class_probs(
    model: torch.nn.Module,
    images: List[Tensor],
    device: Optional[torch.device] = None,
) -> List[Dict[str, Tensor]]:
    """Faster R-CNN inference that retains the full per-class probability
    vector for every kept detection.

    Replicates the standard pipeline (transform → backbone → RPN → ROI head)
    but swaps `roi_heads.postprocess_detections` for the probs-retaining
    variant above.

    Returns one dict per image:
        {
          "boxes":       [N, 4]   in ORIGINAL image space,
          "labels":      [N]      1-indexed lesion class id,
          "scores":      [N],
          "class_probs": [N, num_lesion_classes]  (background col dropped),
        }
    """
    if device is None:
        device = next(model.parameters()).device
    if model.training:
        model.eval()

    images_dev = [img.to(device) for img in images]
    original_image_sizes = [img.shape[-2:] for img in images_dev]

    images_t, _ = model.transform(images_dev, None)
    features = model.backbone(images_t.tensors)
    if isinstance(features, torch.Tensor):
        features = {"0": features}

    proposals, _ = model.rpn(images_t, features, None)
    box_features = model.roi_heads.box_roi_pool(
        features, proposals, images_t.image_sizes
    )
    box_features = model.roi_heads.box_head(box_features)
    class_logits, box_regression = model.roi_heads.box_predictor(box_features)

    boxes_l, scores_l, labels_l, probs_l = _frcnn_postprocess_with_probs(
        model.roi_heads, class_logits, box_regression, proposals, images_t.image_sizes
    )

    detections = []
    for boxes, scores, labels, probs, img_size, orig_size in zip(
        boxes_l,
        scores_l,
        labels_l,
        probs_l,
        images_t.image_sizes,
        original_image_sizes,
    ):
        boxes_resized = _resize_boxes(boxes, img_size, orig_size)
        detections.append(
            {
                "boxes": boxes_resized.detach().cpu(),
                "labels": labels.detach().cpu(),
                "scores": scores.detach().cpu(),
                "class_probs": probs.detach().cpu(),
            }
        )
    return detections


@torch.no_grad()
def predict_with_class_probs_tta_hflip(
    model: torch.nn.Module,
    images: List[Tensor],
    device: Optional[torch.device] = None,
) -> List[Dict[str, Tensor]]:
    """Horizontal-flip TTA for Faster R-CNN real class probabilities.

    The model runs on horizontally flipped images, then box coordinates are
    inverse-flipped back to the original image frame. This produces a second
    view with high structural overlap for multi-source matching while keeping
    real ROI-head softmax vectors.
    """
    flipped_images: List[Tensor] = []
    widths: List[float] = []
    for img in images:
        widths.append(float(img.shape[-1]))
        flipped_images.append(torch.flip(img, dims=[-1]))

    flipped_out = predict_with_class_probs(model, flipped_images, device=device)

    inverse_out: List[Dict[str, Tensor]] = []
    for det, width in zip(flipped_out, widths):
        boxes = det["boxes"].clone()
        if boxes.numel() > 0:
            x1 = boxes[:, 0].clone()
            x2 = boxes[:, 2].clone()
            boxes[:, 0] = width - x2
            boxes[:, 2] = width - x1
            boxes[:, 0].clamp_(min=0.0, max=width)
            boxes[:, 2].clamp_(min=0.0, max=width)

        inverse_out.append(
            {
                "boxes": boxes,
                "labels": det["labels"],
                "scores": det["scores"],
                "class_probs": det["class_probs"],
            }
        )
    return inverse_out


@torch.no_grad()
def predict_retinanet_with_class_probs(
    model: torch.nn.Module,
    images: List[Tensor],
    device: Optional[torch.device] = None,
) -> List[Dict[str, Tensor]]:
    """RetinaNet variant — secondary detector. Full per-anchor extraction is
    not implemented; this returns the standard output with a synthetic
    class_probs (predicted class = score, others share the residual).

    Faster R-CNN (predict_with_class_probs) is the primary path; RetinaNet's
    synthetic residual has limited downstream impact because the native
    bridge uses Faster R-CNN as the primary source.
    """
    if device is None:
        device = next(model.parameters()).device
    if model.training:
        model.eval()
    out = model([img.to(device) for img in images])
    c = 4
    detections = []
    for det in out:
        boxes = det["boxes"].detach().cpu()
        labels = det["labels"].detach().cpu()
        scores = det["scores"].detach().cpu()
        n = boxes.shape[0]
        probs = torch.zeros((n, c), dtype=torch.float32)
        if n > 0:
            rest = (1.0 - scores) / (c - 1)
            for i in range(n):
                col = int(labels[i].item()) - 1
                if 0 <= col < c:
                    probs[i, :] = rest[i]
                    probs[i, col] = scores[i]
        detections.append(
            {
                "boxes": boxes,
                "labels": labels,
                "scores": scores,
                "class_probs": probs,
            }
        )
    return detections
