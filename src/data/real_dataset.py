"""
real_dataset.py
PyTorch Dataset for LGov-Net real-data single-head detector pilot — FASE 19A.

Loads paired image/XML files from a data split directory, resizes images to
a fixed size, scales bounding boxes accordingly, and maps raw XML labels to
canonical KG class indices.

For the pilot, only the FIRST VALID bounding box per image is returned as the
detection target.  Images with no valid canonical box are excluded entirely.

Class index mapping (LESION_CLASS_INDEX):
    MA       -> 0
    HE       -> 1
    Hard_EX  -> 2
    Soft_EX  -> 3

Usage:
    from src.data.real_dataset import RealLesionDataset, LESION_CLASSES, NUM_CLASSES

    ds = RealLesionDataset(split="train", data_root="data", img_size=512, max_images=32)
    image, bbox_norm, class_idx, stem = ds[0]
    # image:     FloatTensor [3, 512, 512]  (ImageNet-normalised)
    # bbox_norm: FloatTensor [4]            (x1, y1, x2, y2 in [0, 1])
    # class_idx: LongTensor  scalar
    # stem:      str          image file stem (no extension)
"""

from __future__ import annotations

import os
import sys
import warnings
from typing import List, Optional, Tuple

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.join(_here, "..", "..")
if _root not in sys.path:
    sys.path.insert(0, _root)

import torch
from torch.utils.data import Dataset
from torchvision import transforms

try:
    from src.data.xml_parser import parse_xml_file
    from src.data.ontology import map_raw_to_kg
except ModuleNotFoundError:
    from lgovnet.src.data.xml_parser import parse_xml_file
    from lgovnet.src.data.ontology import map_raw_to_kg

try:
    from PIL import Image as _PilImage

    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


# ── Class catalogue ────────────────────────────────────────────────────────────
#: Ordered canonical lesion class names (index = class_idx in targets).
LESION_CLASSES: List[str] = ["MA", "HE", "Hard_EX", "Soft_EX"]

#: Mapping from canonical label string to integer class index.
LESION_CLASS_INDEX: dict = {name: idx for idx, name in enumerate(LESION_CLASSES)}

#: Number of lesion classes.
NUM_CLASSES: int = len(LESION_CLASSES)


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════


class RealLesionDataset(Dataset):
    """Paired image + XML annotation dataset for lesion detection.

    Each item exposes:
        image:     FloatTensor [3, img_size, img_size]  (ImageNet-normalised)
        bbox_norm: FloatTensor [4]  (x1, y1, x2, y2 normalised to [0, 1])
        class_idx: LongTensor  scalar
        stem:      str  image file stem

    Only images with at least one valid, canonically-mappable bounding box are
    included.  The **first such box** per image is the pilot training target.

    Args:
        split:      'train', 'valid', or 'test'.
        data_root:  Directory that contains ``images/`` and ``xml/`` sub-dirs.
        img_size:   Square resize target in pixels (default 512).
        max_images: Maximum number of images to load (None = no cap).

    Raises:
        ImportError:      if Pillow is not installed.
        FileNotFoundError: if image or XML directory for the split is missing.
    """

    _IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    _IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    def __init__(
        self,
        split: str,
        data_root: str,
        img_size: int = 512,
        max_images: Optional[int] = None,
    ) -> None:
        if not _PIL_AVAILABLE:
            raise ImportError(
                "Pillow is required for RealLesionDataset: pip install Pillow"
            )

        self.split = split
        self.img_size = img_size

        img_dir = os.path.join(data_root, "images", split)
        xml_dir = os.path.join(data_root, "xml", split)

        if not os.path.isdir(img_dir):
            raise FileNotFoundError(
                f"[RealLesionDataset] Image directory not found: {img_dir}"
            )
        if not os.path.isdir(xml_dir):
            raise FileNotFoundError(
                f"[RealLesionDataset] XML directory not found: {xml_dir}"
            )

        # ── Collect paired stems ───────────────────────────────────────────
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
        paired: List[str] = sorted(img_stems & xml_stems)
        if max_images is not None:
            paired = paired[:max_images]

        # ── Build sample list ──────────────────────────────────────────────
        self._samples: List[dict] = []
        skipped_no_box = 0
        skipped_xml_err = 0

        for stem in paired:
            xml_path = os.path.join(xml_dir, stem + ".xml")
            parsed = parse_xml_file(xml_path)

            if parsed is None:
                skipped_xml_err += 1
                continue

            # Find first valid box with a known canonical label
            first_box: Optional[dict] = None
            for obj in parsed.get("objects", []):
                canonical = map_raw_to_kg(obj.get("label_raw", ""))
                if canonical is None:
                    continue
                class_idx = LESION_CLASS_INDEX.get(canonical)
                if class_idx is None:
                    continue
                first_box = {
                    "class_idx": class_idx,
                    "class_name": canonical,
                    "bbox_orig": obj["bbox"],  # [xmin, ymin, xmax, ymax] pixels
                }
                break

            if first_box is None:
                skipped_no_box += 1
                continue

            img_path = _find_image_path(img_dir, stem)
            if img_path is None:
                continue

            self._samples.append(
                {
                    "stem": stem,
                    "img_path": img_path,
                    **first_box,
                }
            )

        print(
            f"[RealLesionDataset] split={split}"
            f"  loaded={len(self._samples)}"
            f"  skipped_no_box={skipped_no_box}"
            f"  skipped_xml_err={skipped_xml_err}"
        )
        if len(self._samples) == 0:
            warnings.warn(
                f"[RealLesionDataset] No valid samples found for split='{split}'. "
                "Check image/XML directories and label mapping."
            )

        # ── Image transform ────────────────────────────────────────────────
        self._transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self._IMAGENET_MEAN,
                    std=self._IMAGENET_STD,
                ),
            ]
        )

    # ── Dataset interface ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        """Return (image, bbox_norm, class_idx, stem)."""
        sample = self._samples[idx]

        # Load and transform image
        img = _PilImage.open(sample["img_path"]).convert("RGB")
        orig_w, orig_h = img.size  # PIL reports (width, height)
        image_tensor = self._transform(img)

        # Normalise bbox from original pixel space to [0, 1]
        xmin, ymin, xmax, ymax = sample["bbox_orig"]
        scale_x = 1.0 / max(orig_w, 1)
        scale_y = 1.0 / max(orig_h, 1)
        bbox_norm = torch.tensor(
            [
                xmin * scale_x,
                ymin * scale_y,
                xmax * scale_x,
                ymax * scale_y,
            ],
            dtype=torch.float32,
        ).clamp(0.0, 1.0)

        class_idx = torch.tensor(sample["class_idx"], dtype=torch.long)

        return image_tensor, bbox_norm, class_idx, sample["stem"]


# ── Private helpers ────────────────────────────────────────────────────────────


def _find_image_path(img_dir: str, stem: str) -> Optional[str]:
    """Return the full path of the image file for *stem*, or None."""
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        p = os.path.join(img_dir, stem + ext)
        if os.path.isfile(p):
            return p
    return None
