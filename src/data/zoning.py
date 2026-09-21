"""
zoning.py
ETDRS-style anatomical zone assignment for retinal bounding box centroids.

Five zones following the ETDRS grid convention (distances in disc diameters, DD):

  Priority  Zone               Condition
  --------  -----------------  -----------------------------------------------
     1      Foveal             dist_fovea ≤ 1 DD
     2      Peripapillary      dist_disc  ≤ 1 DD  (and not Foveal)
     3      Parafoveal         1 DD < dist_fovea ≤ 2 DD  (and not Peripapillary)
     4      Vascular_Arcades   within the arcade corridor  (and not above)
     5      Peripheral         default (all remaining points)

Arcade corridor approximation (no eye laterality needed):
  - perpendicular distance to the fovea–disc line ≤ arcade_vband_dd DD
  - dist_fovea in (parafoveal_dd, arcade_outer_dd] DD

Landmark source is caller-supplied — no automatic OD/fovea detector here.
Use estimate_landmarks_approx() for development/unit testing only (warns).

Reference:  ETDRS Report No. 2 (1991); ICDR scale (Wilkinson 2003).
"""

import math
import warnings
from dataclasses import dataclass, field


# ── Constants ─────────────────────────────────────────────────────────────────

ZONE_NAMES: list = [
    "Foveal",
    "Parafoveal",
    "Peripapillary",
    "Vascular_Arcades",
    "Peripheral",
]

# Colour hint for visualisation (zone -> hex), used by inspect_zones.py
ZONE_COLORS: dict = {
    "Foveal": "#e74c3c",  # red
    "Parafoveal": "#e67e22",  # orange
    "Peripapillary": "#8e44ad",  # purple
    "Vascular_Arcades": "#2980b9",  # blue
    "Peripheral": "#27ae60",  # green
}


# ── Dataclasses ───────────────────────────────────────────────────────────────


@dataclass
class AnatomicalLandmarks:
    """Optic disc and fovea positions required for zone assignment.

    Attributes:
        fovea:   (x, y) fovea centre in pixels, image coordinate system.
        disc:    (x, y) optic disc centre in pixels.
        dd_px:   Disc diameter in pixels. For a 512×512 retinal fundus image
                 ~45 px is a typical value (1 DD ≈ 1500 µm).
        image_w: Image width in pixels.
        image_h: Image height in pixels.
    """

    fovea: "tuple[float, float]"
    disc: "tuple[float, float]"
    dd_px: float = 45.0
    image_w: int = 512
    image_h: int = 512


@dataclass
class ZoneThresholds:
    """Distance thresholds for ETDRS zone assignment, all in disc diameters (DD).

    Attributes:
        foveal_dd:         Max dist from fovea for Foveal zone (default 1.0 DD).
        parafoveal_dd:     Outer radius from fovea for Parafoveal (default 2.0 DD).
        peripapillary_dd:  Max dist from optic disc for Peripapillary (default 1.0 DD).
        arcade_outer_dd:   Max dist from fovea that still counts as Vascular_Arcades
                           (default 5.0 DD).
        arcade_vband_dd:   Half-height of the arcade corridor, measured perpendicular
                           to the fovea–disc line (default 2.0 DD).
    """

    foveal_dd: float = 1.0
    parafoveal_dd: float = 2.0
    peripapillary_dd: float = 1.0
    arcade_outer_dd: float = 5.0
    arcade_vband_dd: float = 2.0


_DEFAULT_THRESHOLDS = ZoneThresholds()


# ── Landmark utilities ─────────────────────────────────────────────────────────


def estimate_landmarks_approx(
    image_w: int = 512,
    image_h: int = 512,
    dd_px: float = 45.0,
    disc_offset_dd: float = 3.5,
) -> AnatomicalLandmarks:
    """Return a coarse landmark estimate from image dimensions.

    Places fovea at image centre, optic disc offset to the right by
    ``disc_offset_dd`` disc diameters (nasal side for a right eye in standard
    fundus orientation).

    WARNING: This is an approximation for development and unit tests only.
    Eye laterality is ignored.  Use actual annotated landmarks for experiments.

    Args:
        image_w:         Image width in pixels.
        image_h:         Image height in pixels.
        dd_px:           Disc diameter in pixels.
        disc_offset_dd:  How many DDs to offset the disc from the fovea
                         (default 3.5 DD, typical nasal offset).

    Returns:
        AnatomicalLandmarks with approximate fovea and disc positions.
    """
    warnings.warn(
        "[zoning] estimate_landmarks_approx() is a coarse approximation "
        "(right-eye, disc right of fovea). Provide actual landmark coordinates "
        "for real experiments.",
        UserWarning,
        stacklevel=2,
    )
    cx = image_w / 2.0
    cy = image_h / 2.0
    disc_x = cx + disc_offset_dd * dd_px
    return AnatomicalLandmarks(
        fovea=(cx, cy),
        disc=(disc_x, cy),
        dd_px=dd_px,
        image_w=image_w,
        image_h=image_h,
    )


# ── Core zone assignment ──────────────────────────────────────────────────────


def assign_point_to_zone(
    cx: float,
    cy: float,
    landmarks: AnatomicalLandmarks,
    thresholds: "ZoneThresholds | None" = None,
) -> str:
    """Assign a 2D point to one of the 5 ETDRS-style anatomical zones.

    Priority order: Foveal > Peripapillary > Parafoveal > Vascular_Arcades
    > Peripheral.

    Args:
        cx, cy:     Point coordinates in pixels.
        landmarks:  Optic disc and fovea positions.
        thresholds: Zone distance thresholds (uses defaults if None).

    Returns:
        One of ZONE_NAMES.
    """
    th = thresholds or _DEFAULT_THRESHOLDS
    dd = landmarks.dd_px

    fx, fy = landmarks.fovea
    dx, dy = landmarks.disc

    dist_fovea_dd = math.hypot(cx - fx, cy - fy) / dd
    dist_disc_dd = math.hypot(cx - dx, cy - dy) / dd

    # ── Priority 1: Foveal ────────────────────────────────────────────────
    if dist_fovea_dd <= th.foveal_dd:
        return "Foveal"

    # ── Priority 2: Peripapillary ─────────────────────────────────────────
    if dist_disc_dd <= th.peripapillary_dd:
        return "Peripapillary"

    # ── Priority 3: Parafoveal ────────────────────────────────────────────
    if dist_fovea_dd <= th.parafoveal_dd:
        return "Parafoveal"

    # ── Priority 4: Vascular_Arcades ─────────────────────────────────────
    if dist_fovea_dd <= th.arcade_outer_dd:
        perp_dd = _perp_dist_to_line_dd(cx, cy, landmarks)
        if perp_dd is not None and perp_dd <= th.arcade_vband_dd:
            return "Vascular_Arcades"

    # ── Priority 5: Peripheral (default) ─────────────────────────────────
    return "Peripheral"


def assign_bbox_to_zone(
    obj: dict,
    landmarks: AnatomicalLandmarks,
    thresholds: "ZoneThresholds | None" = None,
) -> str:
    """Assign a bbox annotation dict (from xml_parser) to a zone using its centroid.

    Args:
        obj:       Object dict from xml_parser.parse_xml_file(), must contain
                   keys ``cx`` and ``cy``.
        landmarks: Optic disc and fovea positions.
        thresholds: Zone distance thresholds (uses defaults if None).

    Returns:
        One of ZONE_NAMES.
    """
    return assign_point_to_zone(obj["cx"], obj["cy"], landmarks, thresholds)


# ── Dataset-level distribution ────────────────────────────────────────────────


def compute_zone_distribution(
    annotations: list,
    landmark_map: dict,
    thresholds: "ZoneThresholds | None" = None,
) -> dict:
    """Compute per-zone bbox count across a list of parsed annotation dicts.

    Args:
        annotations:  List of dicts from xml_parser.parse_xml_file().
                      Each dict must have key ``image_name`` and ``objects``.
        landmark_map: Mapping of image_name -> AnatomicalLandmarks (or dict with
                      keys ``fovea``, ``disc``, ``dd_px``).
        thresholds:   Zone thresholds (uses defaults if None).

    Returns:
        {zone_name: count} for all ZONE_NAMES.
    """
    counts: dict = {z: 0 for z in ZONE_NAMES}
    missing_landmarks: list = []

    for ann in annotations:
        name = ann.get("image_name", "")
        lm = landmark_map.get(name)
        if lm is None:
            missing_landmarks.append(name)
            continue

        # Accept both AnatomicalLandmarks dataclass and plain dict
        if isinstance(lm, dict):
            lm = AnatomicalLandmarks(
                fovea=tuple(lm["fovea"]),
                disc=tuple(lm["disc"]),
                dd_px=float(lm.get("dd_px", 45.0)),
            )

        for obj in ann.get("objects", []):
            zone = assign_bbox_to_zone(obj, lm, thresholds)
            counts[zone] += 1

    if missing_landmarks:
        warnings.warn(
            f"[zoning] {len(missing_landmarks)} image(s) had no landmark entry "
            f"in landmark_map and were skipped: {missing_landmarks[:5]}"
            + (" ..." if len(missing_landmarks) > 5 else ""),
            UserWarning,
            stacklevel=2,
        )

    return counts


# ── Internal helpers ──────────────────────────────────────────────────────────


def _perp_dist_to_line_dd(
    cx: float,
    cy: float,
    landmarks: AnatomicalLandmarks,
) -> "float | None":
    """Return perpendicular distance (in DD) from point to the fovea–disc line.

    Returns None if fovea == disc (degenerate case).
    """
    fx, fy = landmarks.fovea
    dx, dy = landmarks.disc
    fd_dist = math.hypot(dx - fx, dy - fy)
    if fd_dist < 1e-6:
        return None
    # Perpendicular distance via cross product
    perp_px = abs((dx - fx) * (fy - cy) - (fx - cx) * (dy - fy)) / fd_dist
    return perp_px / landmarks.dd_px
