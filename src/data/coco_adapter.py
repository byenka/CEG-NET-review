"""
coco_adapter.py
FASE 19M-B — Pascal VOC XML → torchvision detection dataset adapter.

Converts the LGov-Net Pascal VOC XML annotations into a PyTorch Dataset
compatible with torchvision detection models (RetinaNet, FCOS, Faster R-CNN).

Torchvision detection contract:
    image:  Tensor [3, H, W]  in [0, 1] range (NO ImageNet normalization)
    target: Dict {
        "boxes":  FloatTensor [N, 4]  pixel coords [x1, y1, x2, y2]
        "labels": Int64Tensor [N]     1-indexed class labels (0 = background)
    }

Class mapping (1-indexed for torchvision):
    1 = MA
    2 = HE
    3 = Hard_EX
    4 = Soft_EX

Usage:
    from src.data.coco_adapter import DRDetectionDataset, LABEL_NAMES

    ds = DRDetectionDataset(split="train", data_root="data")
    image, target = ds[0]
    # image:  FloatTensor [3, H, W]
    # target: {"boxes": FloatTensor[N,4], "labels": Int64Tensor[N]}
"""

from __future__ import annotations

import os
import sys
import importlib
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Environment safeguards (before torch/numpy import) ────────────────────────
_omp = int(os.environ.get("OMP_NUM_THREADS", "0") or "0")
if _omp <= 0:
    os.environ["OMP_NUM_THREADS"] = "1"
if "MKL_NUM_THREADS" not in os.environ:
    os.environ["MKL_NUM_THREADS"] = "1"
# ─────────────────────────────────────────────────────────────────────────────

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.join(_here, "..", "..")
if _root not in sys.path:
    sys.path.insert(0, _root)

import torch
from torch.utils.data import Dataset

try:
    from PIL import Image as PILImage

    _PIL_OK = True
except ImportError:
    _PIL_OK = False

try:
    from src.data.xml_parser import parse_xml_file
    from src.data.ontology import map_raw_to_kg
except ModuleNotFoundError:
    _xml_parser = importlib.import_module("lgovnet.src.data.xml_parser")
    parse_xml_file = _xml_parser.parse_xml_file
    _ontology = importlib.import_module("lgovnet.src.data.ontology")
    map_raw_to_kg = _ontology.map_raw_to_kg

# ── Class catalogue (1-indexed for torchvision) ──────────────────────────────
#: Canonical lesion class names.
LABEL_NAMES: List[str] = ["MA", "HE", "Hard_EX", "Soft_EX"]

#: 1-indexed mapping: torchvision reserves 0 for background.
LABEL_TO_IDX: Dict[str, int] = {name: i + 1 for i, name in enumerate(LABEL_NAMES)}

#: Number of foreground classes.
NUM_CLASSES: int = len(LABEL_NAMES)


def collate_fn(batch):
    """Custom collate for variable-length detection targets."""
    return tuple(zip(*batch))


class DRDetectionDataset(Dataset):
    """Pascal VOC XML dataset adapted for torchvision detection models.

    Returns ALL valid bounding boxes per image (not just the first one).
    Images with no valid boxes are excluded.

    Args:
        split:      'train', 'valid', or 'test'.
        data_root:  Directory containing ``images/`` and ``xml/`` sub-dirs.
        max_images: Cap on number of images (None = no cap).
    """

    def __init__(
        self,
        split: str,
        data_root: str,
        max_images: Optional[int] = None,
        hflip_prob: float = 0.0,
    ) -> None:
        if not _PIL_OK:
            raise ImportError("Pillow required: pip install Pillow")

        self.split = split
        self._hflip_prob = hflip_prob if split == "train" else 0.0

        img_dir = os.path.join(data_root, "images", split)
        xml_dir = os.path.join(data_root, "xml", split)

        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"Image dir not found: {img_dir}")
        if not os.path.isdir(xml_dir):
            raise FileNotFoundError(f"XML dir not found: {xml_dir}")

        # Collect paired stems
        img_stems = {
            os.path.splitext(f)[0]
            for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        }
        xml_stems = {
            os.path.splitext(f)[0]
            for f in os.listdir(xml_dir)
            if f.lower().endswith(".xml")
        }
        paired = sorted(img_stems & xml_stems)
        if max_images is not None:
            paired = paired[:max_images]

        # Build sample list: all valid boxes per image
        self._samples: List[dict] = []
        skipped = 0

        for stem in paired:
            xml_path = os.path.join(xml_dir, stem + ".xml")
            parsed = parse_xml_file(xml_path)
            if parsed is None:
                skipped += 1
                continue

            boxes: List[List[int]] = []
            labels: List[int] = []

            for obj in parsed.get("objects", []):
                canonical = map_raw_to_kg(obj.get("label_raw", ""))
                if canonical is None:
                    continue
                label_idx = LABEL_TO_IDX.get(canonical)
                if label_idx is None:
                    continue

                bbox = obj["bbox"]  # [xmin, ymin, xmax, ymax] pixels
                # Validate box: must have positive area
                if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                    continue

                boxes.append(bbox)
                labels.append(label_idx)

            if len(boxes) == 0:
                skipped += 1
                continue

            img_path = _find_image(img_dir, stem)
            if img_path is None:
                skipped += 1
                continue

            self._samples.append(
                {
                    "stem": stem,
                    "img_path": img_path,
                    "boxes": boxes,
                    "labels": labels,
                }
            )

        print(
            f"[DRDetectionDataset] split={split}"
            f"  loaded={len(self._samples)}"
            f"  skipped={skipped}"
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        s = self._samples[idx]

        img = PILImage.open(s["img_path"]).convert("RGB")

        # Convert to tensor [3, H, W] in [0, 1] — NO ImageNet normalization
        import torchvision.transforms.functional as F
        import random

        image = F.to_tensor(img)  # [3, H, W], float32, [0, 1]

        boxes = torch.as_tensor(s["boxes"], dtype=torch.float32)  # [N, 4]
        labels = torch.as_tensor(s["labels"], dtype=torch.int64)  # [N]

        # Horizontal flip augmentation (train split only, controlled by hflip_prob)
        if self._hflip_prob > 0.0 and random.random() < self._hflip_prob:
            image = F.hflip(image)  # [3, H, W]
            W = image.shape[2]
            # bbox transform: new_x1 = W - old_x2,  new_x2 = W - old_x1
            boxes = boxes.clone()
            boxes[:, 0], boxes[:, 2] = W - boxes[:, 2].clone(), W - boxes[:, 0].clone()

        # REV-7 (Sprint 2): expose stem as image_id so downstream Mode 2 closed-loop
        # can key per-image secondary-detector caches without changing the loader API.
        # Note: image_id is a plain str (not a tensor) — callers must skip it when
        # moving targets to a device (see ``_targets_to_device`` in train_stage5_mode2).
        target = {"boxes": boxes, "labels": labels, "image_id": s["stem"]}

        return image, target

    def get_class_counts(self) -> Dict[str, int]:
        """Count total annotations per class across dataset."""
        counts = {name: 0 for name in LABEL_NAMES}
        for s in self._samples:
            for lbl in s["labels"]:
                name = LABEL_NAMES[lbl - 1]  # 1-indexed back to 0-indexed
                counts[name] += 1
        return counts


def _find_image(img_dir: str, stem: str) -> Optional[str]:
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        p = os.path.join(img_dir, stem + ext)
        if os.path.isfile(p):
            return p
    return None
