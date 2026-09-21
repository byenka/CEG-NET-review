"""detector_anchor.py
Teacher-student anchor regularizer utilities for Mode 2 full-detector training.

This module captures ROI head logits/box regression tensors during Faster R-CNN
forward passes and computes a lightweight distillation loss:

    L_anchor = alpha * KL(teacher_cls || student_cls)
             + beta  * SmoothL1(teacher_box, student_box)

The loss is intentionally conservative and can be used as a small stabilizing
term to prevent regression while adding governance losses.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Generator, Optional

import torch
import torch.nn.functional as F


@contextmanager
def capture_roi_head_outputs(
    model: torch.nn.Module,
) -> Generator[Dict[str, Optional[torch.Tensor]], None, None]:
    """Capture ROI box_predictor outputs for one forward pass.

    Returns a dict with keys:
      - "cls_logits": Tensor[N, C] or None
      - "box_regression": Tensor[N, 4*C] or None
    """
    state: Dict[str, Optional[torch.Tensor]] = {
        "cls_logits": None,
        "box_regression": None,
    }

    if not hasattr(model, "roi_heads") or not hasattr(model.roi_heads, "box_predictor"):
        yield state
        return

    def _hook(_module, _inputs, outputs):
        if isinstance(outputs, tuple) and len(outputs) >= 2:
            state["cls_logits"] = outputs[0]
            state["box_regression"] = outputs[1]

    handle = model.roi_heads.box_predictor.register_forward_hook(_hook)
    try:
        yield state
    finally:
        handle.remove()


def compute_anchor_loss(
    student_cls_logits: Optional[torch.Tensor],
    student_box_regression: Optional[torch.Tensor],
    teacher_cls_logits: Optional[torch.Tensor],
    teacher_box_regression: Optional[torch.Tensor],
    alpha: float = 1.0,
    beta: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """Compute anchor distillation loss from captured ROI outputs.

    Returns a dict with scalar tensors:
      - l_anchor
      - l_anchor_cls
      - l_anchor_box
      - n_aligned (number of aligned proposal rows)
    """
    ref = student_cls_logits
    if ref is None:
        ref = student_box_regression
    if ref is None:
        ref = teacher_cls_logits
    if ref is None:
        ref = teacher_box_regression

    if ref is None:
        zero = torch.tensor(0.0)
        return {
            "l_anchor": zero,
            "l_anchor_cls": zero,
            "l_anchor_box": zero,
            "n_aligned": torch.tensor(0.0),
        }

    device = ref.device
    dtype = ref.dtype

    if (
        student_cls_logits is None
        or student_box_regression is None
        or teacher_cls_logits is None
        or teacher_box_regression is None
    ):
        zero = torch.zeros((), device=device, dtype=dtype)
        return {
            "l_anchor": zero,
            "l_anchor_cls": zero,
            "l_anchor_box": zero,
            "n_aligned": torch.zeros((), device=device, dtype=dtype),
        }

    n = min(
        int(student_cls_logits.shape[0]),
        int(teacher_cls_logits.shape[0]),
        int(student_box_regression.shape[0]),
        int(teacher_box_regression.shape[0]),
    )
    if n <= 0:
        zero = torch.zeros((), device=device, dtype=dtype)
        return {
            "l_anchor": zero,
            "l_anchor_cls": zero,
            "l_anchor_box": zero,
            "n_aligned": torch.zeros((), device=device, dtype=dtype),
        }

    s_cls = student_cls_logits[:n]
    s_box = student_box_regression[:n]
    t_cls = teacher_cls_logits[:n].detach()
    t_box = teacher_box_regression[:n].detach()

    s_log_prob = F.log_softmax(s_cls, dim=-1)
    t_prob = F.softmax(t_cls, dim=-1)
    l_anchor_cls = F.kl_div(s_log_prob, t_prob, reduction="batchmean")
    l_anchor_box = F.smooth_l1_loss(s_box, t_box, reduction="mean")
    l_anchor = float(alpha) * l_anchor_cls + float(beta) * l_anchor_box

    return {
        "l_anchor": l_anchor,
        "l_anchor_cls": l_anchor_cls,
        "l_anchor_box": l_anchor_box,
        "n_aligned": torch.tensor(float(n), device=device, dtype=dtype),
    }
