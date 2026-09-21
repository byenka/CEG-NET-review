"""ROI focal-loss patch for torchvision Faster R-CNN.

This keeps the detector architecture unchanged and only swaps the
classification term used by ``roi_heads.fastrcnn_loss`` during training.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.models.detection import roi_heads as tv_roi_heads

_ORIGINAL_FASTRCNN_LOSS = tv_roi_heads.fastrcnn_loss


def _batch_balanced_alpha(
    labels: Tensor, num_classes: int, dtype: torch.dtype
) -> Tensor:
    counts = torch.bincount(labels, minlength=num_classes).to(dtype=dtype)
    alpha = torch.ones(num_classes, device=labels.device, dtype=dtype)
    present = counts > 0
    if present.any():
        alpha_present = counts[present].float().clamp_min(1.0).rsqrt()
        alpha_present = alpha_present / alpha_present.mean().clamp_min(1e-6)
        alpha_present = alpha_present.to(device=labels.device, dtype=dtype)
        alpha[present] = alpha_present
    return alpha


def _focal_classification_loss(
    class_logits: Tensor,
    labels: Tensor,
    gamma: float,
    alpha_mode: str,
) -> Tensor:
    if labels.numel() == 0:
        return class_logits.sum() * 0.0

    log_probs = F.log_softmax(class_logits, dim=-1)
    row_index = torch.arange(labels.numel(), device=labels.device)
    log_pt = log_probs[row_index, labels]
    pt = log_pt.exp()
    loss = -((1.0 - pt).clamp(min=0.0) ** gamma) * log_pt

    if alpha_mode == "batch_balanced":
        alpha = _batch_balanced_alpha(
            labels,
            num_classes=class_logits.shape[1],
            dtype=class_logits.dtype,
        )
        loss = loss * alpha[labels]
    elif alpha_mode != "none":
        raise ValueError(f"Unsupported ROI focal alpha_mode: {alpha_mode}")

    return loss.mean()


def enable_roi_focal_loss(
    gamma: float = 2.0, alpha_mode: str = "batch_balanced"
) -> None:
    def _roi_focal_loss(class_logits, box_regression, labels, regression_targets):
        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        classification_loss = _focal_classification_loss(
            class_logits,
            labels,
            gamma=gamma,
            alpha_mode=alpha_mode,
        )

        sampled_pos_inds_subset = torch.where(labels > 0)[0]
        labels_pos = labels[sampled_pos_inds_subset]
        num_samples = labels.numel()
        box_regression = box_regression.reshape(
            num_samples, box_regression.size(-1) // 4, 4
        )

        box_loss = F.smooth_l1_loss(
            box_regression[sampled_pos_inds_subset, labels_pos],
            regression_targets[sampled_pos_inds_subset],
            beta=1 / 9,
            reduction="sum",
        )
        box_loss = box_loss / max(num_samples, 1)

        return classification_loss, box_loss

    tv_roi_heads.fastrcnn_loss = _roi_focal_loss


def restore_default_roi_loss() -> None:
    tv_roi_heads.fastrcnn_loss = _ORIGINAL_FASTRCNN_LOSS
