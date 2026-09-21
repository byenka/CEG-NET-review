"""
xml_parser.py
Parse Pascal VOC XML annotation files for LGov-Net dataset.

Assumed XML structure (standard Pascal VOC):
    <annotation>
      <filename>image001.jpg</filename>
      <size>
        <width>512</width>
        <height>512</height>
      </size>
      <object>
        <name>MA</name>
        <bndbox>
          <xmin>45</xmin>
          <ymin>120</ymin>
          <xmax>60</xmax>
          <ymax>135</ymax>
        </bndbox>
      </object>
      ...
    </annotation>

Object dict contract (per LGov-Net Fase 2):
    {
        "label_raw": str,               # raw string from <name>, no mapping
        "bbox": [xmin, ymin, xmax, ymax],  # integers
        "cx": float,                    # bbox center x
        "cy": float,                    # bbox center y
        "w":  float,                    # width  = xmax - xmin
        "h":  float,                    # height = ymax - ymin
        "area": float,                  # w * h
    }

Malformed or unreadable files emit a warning and return None.
No exception is ever raised to the caller.
"""

import warnings
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_xml_file(xml_path: "str | Path") -> "dict | None":
    """Parse a single Pascal VOC XML annotation file.

    Args:
        xml_path: Path to the .xml annotation file.

    Returns:
        Dict with keys ``image_name`` (str), ``width`` (int), ``height`` (int),
        and ``objects`` (list[dict]) using the per-object contract above.
        Returns ``None`` if the file is malformed or unreadable; a warning
        is emitted instead of raising an exception.
    """
    path = Path(xml_path)

    try:
        tree = ET.parse(path)
    except ET.ParseError as exc:
        warnings.warn(f"[xml_parser] Malformed XML, skipping {path.name}: {exc}")
        return None
    except OSError as exc:
        warnings.warn(f"[xml_parser] Cannot read file, skipping {path.name}: {exc}")
        return None

    root = tree.getroot()

    # --- image name ---
    image_name: str = _get_text(root, "filename") or path.stem

    # --- image size ---
    size_node = root.find("size")
    width: int = _parse_int(size_node, "width") if size_node is not None else 0
    height: int = _parse_int(size_node, "height") if size_node is not None else 0

    # --- objects ---
    objects: list = []
    for obj in root.findall("object"):
        name_node = obj.find("name")
        if name_node is None or not (name_node.text or "").strip():
            warnings.warn(
                f"[xml_parser] Missing or empty <name> tag in {path.name}, "
                "object skipped."
            )
            continue

        label_raw: str = name_node.text.strip()

        bbox_node = obj.find("bndbox")
        if bbox_node is None:
            warnings.warn(
                f"[xml_parser] Missing <bndbox> for '{label_raw}' in "
                f"{path.name}, object skipped."
            )
            continue

        xmin = _parse_int(bbox_node, "xmin")
        ymin = _parse_int(bbox_node, "ymin")
        xmax = _parse_int(bbox_node, "xmax")
        ymax = _parse_int(bbox_node, "ymax")

        w = float(xmax - xmin)
        h = float(ymax - ymin)

        if w <= 0 or h <= 0:
            warnings.warn(
                f"[xml_parser] Degenerate bbox ({xmin},{ymin},{xmax},{ymax}) "
                f"for '{label_raw}' in {path.name}, object skipped."
            )
            continue

        objects.append(
            {
                "label_raw": label_raw,
                "bbox": [xmin, ymin, xmax, ymax],
                "cx": xmin + w / 2.0,
                "cy": ymin + h / 2.0,
                "w": w,
                "h": h,
                "area": w * h,
            }
        )

    return {
        "image_name": image_name,
        "width": width,
        "height": height,
        "objects": objects,
    }


def collect_all_labels(xml_dir: "str | Path") -> list:
    """Scan all XML files in xml_dir and return sorted list of unique raw label strings.

    Useful for auditing before any label-to-class mapping is applied.
    Files that fail to parse are skipped with a warning.

    Args:
        xml_dir: Directory containing Pascal VOC XML files.

    Returns:
        Sorted list of unique label strings found across all files.
    """
    labels: set = set()
    for xml_path in Path(xml_dir).glob("*.xml"):
        result = parse_xml_file(xml_path)
        if result is None:
            continue
        for obj in result["objects"]:
            labels.add(obj["label_raw"])
    return sorted(labels)


def count_annotations_per_class(xml_dir: "str | Path") -> dict:
    """Return mapping of {label_string: total_bbox_count} across all XML files.

    Args:
        xml_dir: Directory containing Pascal VOC XML files.

    Returns:
        Dict mapping raw label string to total number of bboxes.
    """
    counts: dict = {}
    for xml_path in Path(xml_dir).glob("*.xml"):
        result = parse_xml_file(xml_path)
        if result is None:
            continue
        for obj in result["objects"]:
            label = obj["label_raw"]
            counts[label] = counts.get(label, 0) + 1
    return counts


# ── internal helpers ──────────────────────────────────────────────────────────


def _get_text(node: ET.Element, tag: str) -> "str | None":
    """Return stripped text of a direct child tag, or None if absent/empty."""
    child = node.find(tag)
    if child is not None and child.text:
        return child.text.strip()
    return None


def _parse_int(node: ET.Element, tag: str, default: int = 0) -> int:
    """Return integer value of a direct child tag, or default on any failure."""
    child = node.find(tag)
    try:
        return (
            int(float(child.text.strip()))
            if child is not None and child.text
            else default
        )
    except (ValueError, AttributeError):
        return default
