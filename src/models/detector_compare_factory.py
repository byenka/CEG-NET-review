

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
