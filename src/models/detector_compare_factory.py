"""detector_compare_factory.py — stock torchvision detectors for the MODEL-LEVEL
comparison (E1). Used ONLY by tools/train_detector_compare.py.

Scope + honesty contract:
  - Every detector here is a STOCK torchvision architecture (no custom module).
    Our primary detector (Faster R-CNN/ResNet-50-FPN) is itself stock; the
    project config labels it ``architecture_label="Faster R-CNN/FPN baseline"``.
    We claim NO detector novelty. This factory exists purely so the governance
    contribution can be positioned/validated across detector families on the
    SAME OIA-DDR split (model-level comparison + detector-agnostic test).
  - Does NOT touch Mode 1: no Mode-1 checkpoint, script, or the audit tool is
    modified. Comparison checkpoints live under their own output dirs.

Label convention (matches the existing, proven r5b/r1a pipeline):
  num_classes = NUM_CLASSES + 1 = 5  (labels 1..4, background slot 0).

Supported model_type:
  fasterrcnn       — Faster R-CNN/ResNet-50-FPN (v1)   [+ small-anchor tuning]
  fasterrcnn_v2    — Faster R-CNN/ResNet-50-FPN v2 (improved recipe, 2021)
  retinanet        — RetinaNet/ResNet-50-FPN (v1)
  retinanet_v2     — RetinaNet/ResNet-50-FPN v2 (2021)
  fcos             — FCOS/ResNet-50-FPN (anchor-free, 2019)
  ssd300           — SSD300/VGG16 (one-stage classic)

Anchor note (documented honestly): Faster R-CNN variants receive the
small-anchor generator (sizes 8..128) consistent with our r5b config, because
lesions are tiny. RetinaNet/FCOS/SSD use their standard anchor/point
configurations (re-sizing their heads to match a custom generator is error-prone
and would deviate from each detector's recommended usage). This is the standard
"stock detector" comparison; the anchor config per family is reported.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch

# Small-object anchors used by our Faster R-CNN (matches r5b_ext25 config).
DEFAULT_FRCNN_ANCHOR_SIZES: Tuple = ((8,), (16,), (32,), (64,), (128,))
DEFAULT_FRCNN_ANCHOR_RATIOS: Tuple = ((0.5, 1.0, 2.0),) * 5

SUPPORTED = (
    "fasterrcnn", "fasterrcnn_v2",
    "retinanet", "retinanet_v2",
    "fcos", "ssd300",
)


def _backbone_imagenet_weights():
    """Return ImageNet weights enum for ResNet-50 if available, else None."""
    try:
        from torchvision.models import ResNet50_Weights
        return ResNet50_Weights.IMAGENET1K_V1
    except Exception:
        return None


def build_detector(
    model_type: str,
    num_classes: int = 5,
    score_thresh: float = 0.01,
    detections_per_img: int = 200,
    anchor_sizes: Optional[Sequence] = None,
    anchor_ratios: Optional[Sequence] = None,
    pretrained_backbone: bool = True,
    min_size: Optional[int] = None,
    max_size: Optional[int] = None,
) -> torch.nn.Module:
    """Build a stock torchvision detector with a head sized to ``num_classes``.

    min_size / max_size: shorter-side / longer-side resize for the internal
    GeneralizedRCNNTransform. When None, torchvision default (800/1333) is used —
    the SAME as our Mode-1 r5b detector (verified: skrip asli tidak override).
    Pass explicit values to match the resolution across ALL comparison models.
    SSD300 input is architecturally fixed at 300px; these args are ignored for it.
    """
    mt = model_type.lower()
    if mt not in SUPPORTED:
        raise ValueError(f"unknown model_type='{model_type}'. supported={SUPPORTED}")

    import torchvision.models.detection as D
    from torchvision.models.detection.rpn import AnchorGenerator

    bb_w = _backbone_imagenet_weights() if pretrained_backbone else None
    # Only forward size kwargs when explicitly set; keeps family default otherwise.
    size_kw: dict = {}
    if min_size is not None:
        size_kw["min_size"] = int(min_size)
    if max_size is not None:
        size_kw["max_size"] = int(max_size)

    if mt == "fasterrcnn":
        sizes = tuple(tuple(s) for s in (anchor_sizes or DEFAULT_FRCNN_ANCHOR_SIZES))
        ratios = tuple(tuple(r) for r in (anchor_ratios or DEFAULT_FRCNN_ANCHOR_RATIOS))
        anchor_gen = AnchorGenerator(sizes=sizes, aspect_ratios=ratios)
        return D.fasterrcnn_resnet50_fpn(
            weights=None, weights_backbone=bb_w, num_classes=num_classes,
            rpn_anchor_generator=anchor_gen,
            box_score_thresh=score_thresh, box_detections_per_img=detections_per_img,
            **size_kw,
        )

    if mt == "fasterrcnn_v2":
        return D.fasterrcnn_resnet50_fpn_v2(
            weights=None, weights_backbone=bb_w, num_classes=num_classes,
            box_score_thresh=score_thresh, box_detections_per_img=detections_per_img,
            **size_kw,
        )

    if mt in ("retinanet", "retinanet_v2"):
        ctor = D.retinanet_resnet50_fpn_v2 if mt == "retinanet_v2" else D.retinanet_resnet50_fpn
        return ctor(weights=None, weights_backbone=bb_w, num_classes=num_classes,
                    score_thresh=score_thresh, detections_per_img=detections_per_img,
                    **size_kw)

    if mt == "fcos":
        return D.fcos_resnet50_fpn(weights=None, weights_backbone=bb_w,
                                   num_classes=num_classes,
                                   score_thresh=score_thresh,
                                   detections_per_img=detections_per_img,
                                   **size_kw)

    if mt == "ssd300":
        # SSD input is fixed 300x300 — size_kw not applicable.
        try:
            from torchvision.models import VGG16_Weights
            vgg_w = VGG16_Weights.IMAGENET1K_FEATURES if pretrained_backbone else None
        except Exception:
            vgg_w = None
        return D.ssd300_vgg16(weights=None, weights_backbone=vgg_w,
                              num_classes=num_classes)

    raise ValueError(model_type)  # unreachable


def detector_config(model_type: str, num_classes: int, score_thresh: float,
                    detections_per_img: int, anchor_sizes, anchor_ratios,
                    min_size: Optional[int] = None,
                    max_size: Optional[int] = None) -> dict:
    """Config dict saved into the checkpoint (provenance + reload)."""
    mt = model_type.lower()
    uses_small_anchors = mt == "fasterrcnn"
    # Record the EFFECTIVE resolution so Table 10 can show it per row.
    if mt == "ssd300":
        eff_min, eff_max = 300, 300
    else:
        eff_min = int(min_size) if min_size is not None else 800   # torchvision default = r5b
        eff_max = int(max_size) if max_size is not None else 1333
    return {
        "model_type": mt,
        "model_variant": "stock_comparison",
        "architecture_label": f"{mt} (stock torchvision, comparison)",
        "num_classes": num_classes,
        "score_thresh": score_thresh,
        "detections_per_img": detections_per_img,
        "min_size": eff_min,
        "max_size": eff_max,
        "anchor_sizes": [list(s) for s in (anchor_sizes or DEFAULT_FRCNN_ANCHOR_SIZES)]
        if uses_small_anchors else "model_default",
        "anchor_ratios": [list(r) for r in (anchor_ratios or DEFAULT_FRCNN_ANCHOR_RATIOS)]
        if uses_small_anchors else "model_default",
        "pretrained_backbone": True,
        "no_custom_module": True,
        "note": "stock torchvision detector; no architectural novelty claimed",
    }
