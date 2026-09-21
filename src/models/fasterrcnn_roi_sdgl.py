"""ROI SDGL-loss patch for torchvision Faster R-CNN.

This module replaces torchvision RoIHeads classification loss with an
SDGL-shaped objective so detector gradients flow through a clinical-distance
loss (Mode 2 full-detector path).

Scope:
- Replaces only the ROI classification term (loss_classifier)
- Keeps bbox regression term identical to torchvision default
- Designed for Stage 5 full-mode detector fine-tuning
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.models.detection import roi_heads as tv_roi_heads

from lgovnet.src.losses.sdgl import LESION_CLASSES, build_q_sdm

_ORIGINAL_FASTRCNN_LOSS = tv_roi_heads.fastrcnn_loss

_SDGL_CTX: Dict[str, object] = {}
_LAST_STATS: Dict[str, float] = {}


def _compute_dynamic_sdm_norm_matrix(lesion_embeddings: Tensor) -> Tensor:
    """Build differentiable SDM_norm matrix from current lesion embeddings."""
    if lesion_embeddings.ndim != 2 or lesion_embeddings.size(0) != len(LESION_CLASSES):
        raise ValueError(
            "lesion_embeddings must have shape [num_lesion_classes, d] "
            f"with num_lesion_classes={len(LESION_CLASSES)}"
        )

    # Pairwise distances over lesion embeddings; diagonal is zero by definition.
    dist = torch.cdist(lesion_embeddings, lesion_embeddings, p=2)
    max_dist = torch.max(dist)
    if float(max_dist.detach().item()) <= 1e-9:
        return torch.zeros_like(dist)
    return dist / (max_dist + 1e-9)


def _lookup_sdm_norm(ci: str, cj: str, sdm_norm: Dict[Tuple[str, str], Dict]) -> float:
    if ci == cj:
        return 0.0
    entry = sdm_norm.get((ci, cj))
    if entry is None:
        entry = sdm_norm.get((cj, ci))
    if entry is None:
        return 1.0
    return float(entry.get("norm", 1.0))


def _build_close_pairs(
    sdm_norm: Dict[Tuple[str, str], Dict],
    close_threshold: float,
) -> Set[Tuple[int, int]]:
    close_pairs: Set[Tuple[int, int]] = set()
    for i, ci in enumerate(LESION_CLASSES, start=1):
        for j, cj in enumerate(LESION_CLASSES, start=1):
            if i == j:
                continue
            if _lookup_sdm_norm(ci, cj, sdm_norm) < close_threshold:
                close_pairs.add((i, j))
    return close_pairs


def _compute_sdgl_classification_loss(
    class_logits: Tensor,
    labels: Tensor,
) -> Tuple[Tensor, Dict[str, float]]:
    if labels.numel() == 0:
        zero = class_logits.sum() * 0.0
        return zero, {
            "n_samples": 0,
            "n_pos": 0,
            "n_bg": 0,
            "n_r1": 0,
            "n_r2": 0,
            "n_r3": 0,
            "used_fallback": 1.0,
            "used_uniform_temp": 0.0,
        }

    log_probs = F.log_softmax(class_logits, dim=-1)
    probs = log_probs.exp()

    lambda1 = float(_SDGL_CTX.get("lambda1", 1.0))
    lambda2 = float(_SDGL_CTX.get("lambda2", 0.5))
    lambda3 = float(_SDGL_CTX.get("lambda3", 0.3))
    eta = float(_SDGL_CTX.get("eta", 1.0))
    r1_conf = float(_SDGL_CTX.get("r1_conf", 0.70))
    r2_conf = float(_SDGL_CTX.get("r2_conf", 0.40))
    r3_min_weight = float(_SDGL_CTX.get("r3_min_weight", 0.10))
    bg_weight = float(_SDGL_CTX.get("bg_weight", 0.20))
    uniform_temp = bool(_SDGL_CTX.get("uniform_temp", False))
    close_pairs = _SDGL_CTX.get("close_pairs", set())
    sdm_norm = _SDGL_CTX.get("sdm_norm", {})
    close_threshold = float(_SDGL_CTX.get("close_threshold", 0.30))
    lesion_embeddings = _SDGL_CTX.get("lesion_embeddings", None)
    dynamic_sdm_norm = None
    if isinstance(lesion_embeddings, torch.Tensor):
        dynamic_sdm_norm = _compute_dynamic_sdm_norm_matrix(
            lesion_embeddings.to(device=class_logits.device, dtype=class_logits.dtype)
        )

    pos_idx = torch.where(labels > 0)[0]
    bg_idx = torch.where(labels == 0)[0]

    loss_terms: List[Tensor] = []
    n_r1 = n_r2 = n_r3 = 0

    if pos_idx.numel() > 0:
        gt_pos = labels[pos_idx]
        pred_pos = torch.argmax(probs[pos_idx], dim=1)
        conf_pos = torch.max(probs[pos_idx], dim=1).values

        pred_is_fg = pred_pos > 0
        clinically_close = torch.zeros_like(pred_is_fg)
        if pred_is_fg.any():
            fg_locs = torch.where(pred_is_fg)[0]
            if dynamic_sdm_norm is not None:
                pred_idx0 = pred_pos[fg_locs] - 1
                gt_idx0 = gt_pos[fg_locs] - 1
                fg_close = dynamic_sdm_norm[pred_idx0, gt_idx0] < close_threshold
            else:
                fg_pairs = [
                    (int(pred_pos[i].item()), int(gt_pos[i].item())) for i in fg_locs
                ]
                fg_close = torch.tensor(
                    [pair in close_pairs for pair in fg_pairs],
                    device=labels.device,
                    dtype=torch.bool,
                )
            clinically_close[fg_locs] = fg_close

        r1_mask = (pred_pos == gt_pos) & (conf_pos >= r1_conf)
        r2_mask = (
            (pred_pos != gt_pos) & pred_is_fg & clinically_close & (conf_pos >= r2_conf)
        )
        r3_mask = ~(r1_mask | r2_mask)

        n_r1 = int(r1_mask.sum().item())
        n_r2 = int(r2_mask.sum().item())
        n_r3 = int(r3_mask.sum().item())

        if n_r1 > 0:
            abs_r1 = pos_idx[r1_mask]
            gt_r1 = gt_pos[r1_mask]
            l_r1 = -log_probs[abs_r1, gt_r1]
            loss_terms.append(lambda1 * l_r1.mean())

        if n_r2 > 0:
            abs_r2 = pos_idx[r2_mask]
            r2_losses: List[Tensor] = []
            for abs_idx in abs_r2.tolist():
                fg_probs = probs[abs_idx, 1 : 1 + len(LESION_CLASSES)]
                fg_probs = fg_probs / fg_probs.sum().clamp(min=1e-9)
                if uniform_temp:
                    q_sdm = fg_probs.detach()
                elif dynamic_sdm_norm is not None:
                    majority_idx = int(torch.argmax(fg_probs).item())
                    tau = 1.0 + eta * dynamic_sdm_norm[:, majority_idx]
                    scaled_log_p = torch.log(fg_probs.clamp(min=1e-9)) / tau.clamp(
                        min=1.0
                    )
                    q_sdm = F.softmax(scaled_log_p, dim=0)
                else:
                    ens_probs = {
                        cls_name: float(fg_probs[k].detach().cpu().item())
                        for k, cls_name in enumerate(LESION_CLASSES)
                    }
                    q_sdm = build_q_sdm(ens_probs, sdm_norm, eta=eta).to(
                        device=class_logits.device,
                        dtype=class_logits.dtype,
                    )
                q_full = torch.zeros_like(probs[abs_idx])
                q_full[1 : 1 + len(LESION_CLASSES)] = q_sdm
                r2_losses.append(-(q_full * log_probs[abs_idx]).sum())

            if r2_losses:
                loss_terms.append(lambda2 * torch.stack(r2_losses).mean())

        if n_r3 > 0:
            abs_r3 = pos_idx[r3_mask]
            gt_r3 = gt_pos[r3_mask]
            conf_r3 = conf_pos[r3_mask].detach().clamp(min=r3_min_weight, max=1.0)
            l_r3 = -log_probs[abs_r3, gt_r3]
            loss_terms.append(lambda3 * (l_r3 * conf_r3).mean())

    if bg_idx.numel() > 0 and bg_weight > 0.0:
        l_bg = -log_probs[bg_idx, 0].mean()
        loss_terms.append(bg_weight * l_bg)

    if loss_terms:
        classification_loss = torch.stack(loss_terms).sum()
        used_fallback = 0.0
    else:
        classification_loss = F.nll_loss(log_probs, labels, reduction="mean")
        used_fallback = 1.0

    stats = {
        "n_samples": float(labels.numel()),
        "n_pos": float(pos_idx.numel()),
        "n_bg": float(bg_idx.numel()),
        "n_r1": float(n_r1),
        "n_r2": float(n_r2),
        "n_r3": float(n_r3),
        "used_fallback": used_fallback,
        "used_uniform_temp": 1.0 if uniform_temp else 0.0,
        "used_dynamic_sdm": 1.0 if dynamic_sdm_norm is not None else 0.0,
    }
    return classification_loss, stats


def enable_roi_sdgl_loss(
    sdm_norm: Dict[Tuple[str, str], Dict],
    lambda1: float = 1.0,
    lambda2: float = 0.5,
    lambda3: float = 0.3,
    eta: float = 1.0,
    close_threshold: float = 0.30,
    r1_conf: float = 0.70,
    r2_conf: float = 0.40,
    r3_min_weight: float = 0.10,
    bg_weight: float = 0.20,
    uniform_temp: bool = False,
) -> None:
    """Enable ROI SDGL classification loss for torchvision Faster R-CNN."""

    close_pairs = _build_close_pairs(sdm_norm, close_threshold=close_threshold)
    _SDGL_CTX.clear()
    _SDGL_CTX.update(
        {
            "sdm_norm": sdm_norm,
            "close_pairs": close_pairs,
            "lambda1": float(lambda1),
            "lambda2": float(lambda2),
            "lambda3": float(lambda3),
            "eta": float(eta),
            "close_threshold": float(close_threshold),
            "r1_conf": float(r1_conf),
            "r2_conf": float(r2_conf),
            "r3_min_weight": float(r3_min_weight),
            "bg_weight": float(bg_weight),
            "uniform_temp": bool(uniform_temp),
            "lesion_embeddings": None,
        }
    )

    def _roi_sdgl_loss(class_logits, box_regression, labels, regression_targets):
        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        classification_loss, stats = _compute_sdgl_classification_loss(
            class_logits,
            labels,
        )
        _LAST_STATS.clear()
        _LAST_STATS.update(stats)

        sampled_pos_inds_subset = torch.where(labels > 0)[0]
        labels_pos = labels[sampled_pos_inds_subset]
        num_samples = labels.numel()
        box_regression = box_regression.reshape(
            num_samples,
            box_regression.size(-1) // 4,
            4,
        )

        box_loss = F.smooth_l1_loss(
            box_regression[sampled_pos_inds_subset, labels_pos],
            regression_targets[sampled_pos_inds_subset],
            beta=1 / 9,
            reduction="sum",
        )
        box_loss = box_loss / max(num_samples, 1)

        return classification_loss, box_loss

    tv_roi_heads.fastrcnn_loss = _roi_sdgl_loss


def get_last_sdgl_roi_stats() -> Dict[str, float]:
    return dict(_LAST_STATS)


def set_roi_sdgl_dynamic_state(lesion_embeddings: Optional[Tensor]) -> None:
    """Inject current lesion embeddings for differentiable SDM in ROI loss."""
    if lesion_embeddings is None:
        _SDGL_CTX["lesion_embeddings"] = None
        return
    if lesion_embeddings.ndim != 2 or lesion_embeddings.size(0) != len(LESION_CLASSES):
        raise ValueError(
            "lesion_embeddings must have shape [num_lesion_classes, d] "
            f"with num_lesion_classes={len(LESION_CLASSES)}"
        )
    _SDGL_CTX["lesion_embeddings"] = lesion_embeddings


def restore_default_roi_loss() -> None:
    tv_roi_heads.fastrcnn_loss = _ORIGINAL_FASTRCNN_LOSS
