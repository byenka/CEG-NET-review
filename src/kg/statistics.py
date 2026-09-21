"""
statistics.py
Compute dataset-derived statistics for initializing trainable KG edges.

Three main computations:
  1. Lesion counts per image — prevalence and annotation density.
  2. Co-occurrence matrix — image-level lesion pair co-occurrence.
     Used to initialize co_occurrence_pattern edge weights.
  3. Lesion-zone affinity — how often each lesion class appears in each zone.
     Used to initialize spatial_distribution edge weights.

Design decisions:
  - All statistics use CANONICAL KG lesion node names (MA, HE, Hard_EX, Soft_EX),
    obtained by mapping raw XML labels through ontology.map_raw_to_kg().
  - If a node is absent from the dataset (zero observations), its row/column
    is retained in output dicts but marked as unobserved.  A RuntimeWarning is
    emitted for every CONCEPTUAL_ONLY_NODE not observed.
  - Empty XML directories return empty stats — no crash, informational print only.
  - Zone assignment uses caller-supplied landmark_map when available, and
    estimate_landmarks_approx() as fallback (emits UserWarning).
  - Unobserved nodes in the zone-affinity CSV are written with empty cells,
    not zeros, to avoid fabricating statistics.
"""

import csv
import json
import warnings
from pathlib import Path
from typing import Optional

from lgovnet.src.data.xml_parser import parse_xml_file
from lgovnet.src.data.ontology import (
    LESION_NODES,
    CONCEPTUAL_ONLY_NODES,
    map_raw_to_kg,
)
from lgovnet.src.data.zoning import (
    assign_bbox_to_zone,
    estimate_landmarks_approx,
    ZoneThresholds,
    ZONE_NAMES,
)

# Canonical lesion list ordered by KG index — must match LESION_NODES order.
LESION_ORDER: list = list(LESION_NODES.keys())  # ["MA", "HE", "Hard_EX", "Soft_EX"]


# ── Internal helpers ──────────────────────────────────────────────────────────


def _warn_conceptual_nodes(observed_nodes: set) -> None:
    """Emit RuntimeWarning for any conceptual-only node absent from dataset."""
    for node in sorted(CONCEPTUAL_ONLY_NODES):
        if node not in observed_nodes:
            warnings.warn(
                f"[statistics] Conceptual-only node '{node}' has ZERO observations "
                f"in this dataset. It is retained in the KG for structural reasons "
                f"(pathophysiological_mechanism and severity_transition edges) but "
                f"its trainable edges will not be data-initialized. "
                f"Consider using clinical priors as fallback.",
                RuntimeWarning,
                stacklevel=3,
            )


# ── Public API ────────────────────────────────────────────────────────────────


def compute_lesion_counts_per_image(xml_dir: "str | Path") -> list:
    """Compute per-image lesion bbox counts using canonical KG class names.

    Args:
        xml_dir: Directory of Pascal VOC XML annotation files.

    Returns:
        List of dicts, one per successfully parsed XML file.
        Each dict: {"image_name": str, "MA": int, "HE": int,
                    "Hard_EX": int, "Soft_EX": int,
                    "total": int, "unmapped": int}
        "unmapped" counts bboxes whose raw label could not be mapped to any
        KG lesion node.  Returns [] if no XML files are found.
    """
    xml_dir = Path(xml_dir)
    xml_files = sorted(xml_dir.glob("*.xml"))

    if not xml_files:
        print(f"[statistics] No XML files found in '{xml_dir}'. Returning empty stats.")
        return []

    rows = []
    for xml_path in xml_files:
        parsed = parse_xml_file(xml_path)
        if parsed is None:
            continue  # warning already emitted by xml_parser

        counts = {cls: 0 for cls in LESION_ORDER}
        unmapped = 0

        for obj in parsed["objects"]:
            kg_name = map_raw_to_kg(obj["label_raw"])
            if kg_name is not None:
                counts[kg_name] = counts.get(kg_name, 0) + 1
            else:
                unmapped += 1

        row = {"image_name": parsed["image_name"]}
        row.update(counts)
        row["total"] = sum(counts.values())
        row["unmapped"] = unmapped
        rows.append(row)

    return rows


def compute_cooccurrence_matrix(
    xml_dir: "str | Path",
    image_counts: Optional[list] = None,
) -> dict:
    """Compute image-level co-occurrence counts between canonical lesion class pairs.

    Co-occurrence is image-level: an image contributes 1 to cell (A, B) if it
    contains at least one bbox of class A AND at least one bbox of class B.
    Diagonal entries equal total images where that class appears (prevalence).

    Args:
        xml_dir:      Directory of Pascal VOC XML files.  Only used when
                      image_counts is None.
        image_counts: Pre-computed output of compute_lesion_counts_per_image().
                      If provided, xml_dir is not re-scanned.

    Returns:
        {
          "counts": {cls_a: {cls_b: int}},        # raw co-occurrence matrix
          "n_images_total": int,
          "n_images_with_any_annotation": int,
          "observed_nodes": list[str],             # nodes with at least 1 image
          "unobserved_nodes": list[str],
        }
    """
    if image_counts is None:
        image_counts = compute_lesion_counts_per_image(xml_dir)

    n_total = len(image_counts)
    n_with_ann = sum(1 for row in image_counts if row["total"] > 0)

    matrix = {a: {b: 0 for b in LESION_ORDER} for a in LESION_ORDER}

    for row in image_counts:
        present = {cls for cls in LESION_ORDER if row.get(cls, 0) > 0}
        for cls_a in present:
            for cls_b in present:
                matrix[cls_a][cls_b] += 1

    observed = [cls for cls in LESION_ORDER if matrix[cls][cls] > 0]
    unobserved = [cls for cls in LESION_ORDER if matrix[cls][cls] == 0]

    _warn_conceptual_nodes(set(observed))

    return {
        "counts": matrix,
        "n_images_total": n_total,
        "n_images_with_any_annotation": n_with_ann,
        "observed_nodes": observed,
        "unobserved_nodes": unobserved,
    }


def compute_lesion_zone_affinity(
    xml_dir: "str | Path",
    landmark_map: Optional[dict] = None,
) -> dict:
    """Compute raw and normalized lesion-zone frequency matrix.

    Each bbox centroid is assigned a zone via assign_bbox_to_zone(). For each
    (lesion_class, zone) pair, the raw count is the number of bboxes in that
    combination. Normalization divides each observed lesion row by the total
    bbox count for that class.

    Args:
        xml_dir:      Directory of Pascal VOC XML files.
        landmark_map: Optional dict {image_name: AnatomicalLandmarks}.
                      If None, or if an image is missing from the map,
                      estimate_landmarks_approx() is used as fallback (warns).

    Returns:
        {
          "raw_counts":  {lesion_class: {zone_name: int}},
          "normalized":  {lesion_class: {zone_name: float}},   # observed only
          "observed_nodes": list[str],
          "unobserved_nodes": list[str],
          "total_bboxes_per_class": {lesion_class: int},
        }
        Unobserved lesion classes are absent from "normalized" — no fake zeros.
    """
    xml_dir = Path(xml_dir)
    xml_files = sorted(xml_dir.glob("*.xml"))

    empty_result = {
        "raw_counts": {cls: {z: 0 for z in ZONE_NAMES} for cls in LESION_ORDER},
        "normalized": {},
        "observed_nodes": [],
        "unobserved_nodes": list(LESION_ORDER),
        "total_bboxes_per_class": {cls: 0 for cls in LESION_ORDER},
    }

    if not xml_files:
        print(
            f"[statistics] No XML files found in '{xml_dir}'. Returning empty affinity."
        )
        _warn_conceptual_nodes(set())
        return empty_result

    thresholds = ZoneThresholds()
    raw_counts = {cls: {z: 0 for z in ZONE_NAMES} for cls in LESION_ORDER}
    total_bboxes = {cls: 0 for cls in LESION_ORDER}

    for xml_path in xml_files:
        parsed = parse_xml_file(xml_path)
        if parsed is None:
            continue

        img_name = parsed["image_name"]
        w = parsed.get("width", 512) or 512
        h = parsed.get("height", 512) or 512

        # Resolve landmarks — caller-supplied preferred, approx as fallback.
        if landmark_map is not None and img_name in landmark_map:
            lm = landmark_map[img_name]
        else:
            lm = estimate_landmarks_approx(image_w=w, image_h=h)

        for obj in parsed["objects"]:
            kg_name = map_raw_to_kg(obj["label_raw"])
            if kg_name is None:
                continue
            zone = assign_bbox_to_zone(obj, lm, thresholds)
            raw_counts[kg_name][zone] += 1
            total_bboxes[kg_name] += 1

    observed = [cls for cls in LESION_ORDER if total_bboxes[cls] > 0]
    unobserved = [cls for cls in LESION_ORDER if total_bboxes[cls] == 0]

    normalized = {}
    for cls in observed:
        total = total_bboxes[cls]
        normalized[cls] = {z: raw_counts[cls][z] / total for z in ZONE_NAMES}

    _warn_conceptual_nodes(set(observed))

    return {
        "raw_counts": raw_counts,
        "normalized": normalized,
        "observed_nodes": observed,
        "unobserved_nodes": unobserved,
        "total_bboxes_per_class": total_bboxes,
    }


def compute_dataset_summary(
    xml_dir: "str | Path",
    image_counts: Optional[list] = None,
    cooccurrence: Optional[dict] = None,
    affinity: Optional[dict] = None,
) -> dict:
    """Aggregate summary statistics as a flat, JSON-serializable dict.

    Any argument passed as None is computed from xml_dir automatically.

    Returns:
        Dict suitable for writing to kg_dataset_stats.json.  Includes
        prevalence counts, prevalence rates, bbox counts per class,
        the full co-occurrence matrix, and the normalized zone affinity.
    """
    xml_dir = Path(xml_dir)
    if image_counts is None:
        image_counts = compute_lesion_counts_per_image(xml_dir)
    if cooccurrence is None:
        cooccurrence = compute_cooccurrence_matrix(xml_dir, image_counts)
    if affinity is None:
        affinity = compute_lesion_zone_affinity(xml_dir)

    n_images = cooccurrence["n_images_total"]
    total_bboxes = sum(r["total"] for r in image_counts)
    total_unmapped = sum(r.get("unmapped", 0) for r in image_counts)

    prevalence = {cls: cooccurrence["counts"][cls][cls] for cls in LESION_ORDER}
    prevalence_rate = {
        cls: (prevalence[cls] / n_images if n_images > 0 else 0.0)
        for cls in LESION_ORDER
    }

    return {
        "xml_dir": str(xml_dir),
        "n_images_total": n_images,
        "n_images_with_any_annotation": cooccurrence["n_images_with_any_annotation"],
        "total_bboxes": total_bboxes,
        "total_bboxes_unmapped": total_unmapped,
        "lesion_order": LESION_ORDER,
        "observed_nodes": cooccurrence["observed_nodes"],
        "unobserved_nodes": cooccurrence["unobserved_nodes"],
        "conceptual_only_nodes": sorted(CONCEPTUAL_ONLY_NODES),
        "ontology_frozen": True,
        "prevalence_count": prevalence,
        "prevalence_rate": prevalence_rate,
        "total_bboxes_per_class": affinity["total_bboxes_per_class"],
        "cooccurrence_counts": cooccurrence["counts"],
        "zone_affinity_normalized": affinity["normalized"],
    }


# ── IO helpers ────────────────────────────────────────────────────────────────


def save_image_counts_csv(image_counts: list, out_path: "str | Path") -> None:
    """Write image_level_counts.csv — one row per image."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_name"] + LESION_ORDER + ["total", "unmapped"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(image_counts)
    print(f"  Saved: {out_path}  ({len(image_counts)} rows)")


def save_cooccurrence_csv(cooccurrence: dict, out_path: "str | Path") -> None:
    """Write cooccurrence_matrix.csv — 4×4 symmetric matrix, lesion vs lesion."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = cooccurrence["counts"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + LESION_ORDER)
        for cls_a in LESION_ORDER:
            writer.writerow([cls_a] + [matrix[cls_a][cls_b] for cls_b in LESION_ORDER])
    print(f"  Saved: {out_path}  ({len(LESION_ORDER)}x{len(LESION_ORDER)} matrix)")


def save_zone_affinity_csv(affinity: dict, out_path: "str | Path") -> None:
    """Write lesion_zone_affinity.csv — rows: lesion classes, cols: ETDRS zones.

    Unobserved lesion nodes are written with empty cells (not zeros) to avoid
    fabricating statistics for nodes with no data.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = affinity["normalized"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["lesion"] + ZONE_NAMES + ["observed"])
        for cls in LESION_ORDER:
            is_obs = cls in normalized
            if is_obs:
                zone_vals = [f"{normalized[cls][z]:.6f}" for z in ZONE_NAMES]
            else:
                zone_vals = [""] * len(ZONE_NAMES)  # blank — no fake data
            writer.writerow([cls] + zone_vals + [str(is_obs)])
    print(
        f"  Saved: {out_path}  ({len(LESION_ORDER)} lesion rows × {len(ZONE_NAMES)} zones)"
    )


def save_stats_json(summary: dict, out_path: "str | Path") -> None:
    """Write kg_dataset_stats.json — full serialized summary."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {out_path}")


def load_stats(stats_path: "str | Path") -> dict:
    """Load previously computed stats from kg_dataset_stats.json."""
    with open(stats_path, "r", encoding="utf-8") as f:
        return json.load(f)
