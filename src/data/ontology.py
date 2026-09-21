"""
ontology.py
Clinical class definitions, fixed attributes, and label mapping for LGov-Net.

Two layers are kept strictly separated:
  Layer 1 — KG NODES (canonical): the 9 nodes of G = (V, E).
            These never change regardless of dataset.
  Layer 2 — DATASET LABELS (raw): whatever strings appear in XML <name> tags.
            These are dataset-specific and may vary across sources.

Between layers sits the MAPPING:
  raw XML label  →  normalize_raw_label()  →  map_raw_to_kg()  →  canonical KG node

Key design decision — Soft_EX as a confirmed-observed node:
  Soft_EX is a defined KG lesion node (idx=3) with full clinical attributes,
  edges, and ICDR weight. It MUST remain in the KG regardless of whether it
  appears in the training dataset, because:
    (a) pathophysiological_mechanism edges (MA→Soft_EX) and
        severity_transition edges (MA↔Soft_EX:2.0, HE↔Soft_EX:2.0) define
        the geometric structure of SDM_learned that governs the SDGL loss.
    (b) absence from labeled data ≠ absence from clinical knowledge.
  Confirmed via visual audit (seed=42, 20-sample): `se` in OIA-DDR labels
  fluffy/ill-defined cotton-wool type lesions → Soft_EX. FROZEN 2025.
  Use get_observed_kg_nodes() to find which nodes appear in actual XML files.

Fixed clinical knowledge sourced from:
  - ICDR 5-level grading scale
  - Inter-observer kappa studies
  - Architecture spec: architecture-lgov-net-v2_rev.txt (Contribution 1)

Do NOT modify numeric values without clinical justification.
"""

# ── Node Indices ─────────────────────────────────────────────────────────────
# Order must be consistent across schema.py, kg.yaml, and all downstream code.

LESION_NODES: dict[str, int] = {
    "MA": 0,
    "HE": 1,
    "Hard_EX": 2,
    "Soft_EX": 3,
}

ZONE_NODES: dict[str, int] = {
    "Foveal": 4,
    "Parafoveal": 5,
    "Peripapillary": 6,
    "Vascular_Arcades": 7,
    "Peripheral": 8,
}

ALL_NODES: dict[str, int] = {**LESION_NODES, **ZONE_NODES}
NUM_NODES: int = 9  # must equal len(ALL_NODES)

# Reverse lookup: index -> name
IDX_TO_NODE: dict[int, str] = {v: k for k, v in ALL_NODES.items()}

# ── Clinical Attributes (8-dim per lesion node) ───────────────────────────────
# [size_min_um, size_max_um, shape_code, brightness_code,
#  edge_sharpness_code, freq_code, icdr_weight, severity_rank]
#
# shape_code:         1=round, 2=dot/blot/flame, 3=angular/waxy, 4=round/fluffy
# brightness_code:    0=dark, 1=bright
# edge_sharpness:     0=ill-defined, 1=well-defined, 2=moderate
# freq_code:          1=low, 2=moderate, 3=high, 4=very_high
# severity_rank:      0=lowest DR impact, 2=highest in this set

CLINICAL_ATTRS: dict[str, list] = {
    "MA": [10, 125, 1, 0, 1, 4, 1.0, 0],
    "HE": [50, 1000, 2, 0, 2, 3, 2.0, 1],
    "Hard_EX": [20, 500, 3, 1, 1, 2, 1.5, 1],
    "Soft_EX": [200, 1500, 4, 1, 0, 1, 3.0, 2],
}

# ── Clinical Attribute Normalization ─────────────────────────────────────────
# Per-dimension (min, max) bounds derived from the 4 lesion node values above.
# Dimension order mirrors CLINICAL_ATTRS:
#   [size_min_um, size_max_um, shape_code, brightness_code,
#    edge_sharpness_code, freq_code, icdr_weight, severity_rank]
_CLINICAL_ATTR_BOUNDS: list = [
    (10.0, 200.0),  # size_min_um:     MA=10 .. Soft_EX=200
    (125.0, 1500.0),  # size_max_um:     MA=125 .. Soft_EX=1500
    (1.0, 4.0),  # shape_code:      1=round .. 4=round/fluffy
    (0.0, 1.0),  # brightness_code: 0=dark .. 1=bright
    (0.0, 2.0),  # edge_sharpness:  0=ill-defined .. 2=moderate
    (1.0, 4.0),  # freq_code:       1=low .. 4=very_high
    (1.0, 3.0),  # icdr_weight:     1.0=MA .. 3.0=Soft_EX
    (0.0, 2.0),  # severity_rank:   0=lowest .. 2=highest
]


def normalize_clinical_attrs(
    raw_attrs: dict,
    bounds: "list | None" = None,
) -> "dict[str, list[float]]":
    """Min-max normalize clinical attribute vectors per dimension.

    Formula per dimension d: (x - min_d) / (max_d - min_d).
    If max_d == min_d (degenerate), result is 0.0 for that dimension.
    All output values are in [0.0, 1.0].

    Args:
        raw_attrs: {node_name: [raw_value, ...]} e.g. CLINICAL_ATTRS.
        bounds:    Per-dim (min, max) list. Defaults to _CLINICAL_ATTR_BOUNDS.

    Returns:
        {node_name: [normalized_float, ...]} — same keys, all values in [0, 1].
    """
    if bounds is None:
        bounds = _CLINICAL_ATTR_BOUNDS
    result: dict = {}
    for name, attrs in raw_attrs.items():
        norm: list = []
        for i, val in enumerate(attrs):
            lo, hi = bounds[i]
            denom = hi - lo
            norm.append(0.0 if denom == 0.0 else (float(val) - lo) / denom)
        result[name] = norm
    return result


# Pre-computed normalized attributes — USE THIS in get_node_features().
# Raw values are preserved in CLINICAL_ATTRS for documentation and auditing.
CLINICAL_ATTRS_NORMALIZED: dict = normalize_clinical_attrs(CLINICAL_ATTRS)

# ── Fixed Edge Weights ────────────────────────────────────────────────────────

# severity_transition: ICDR grade cost of confusing two lesion classes (undirected)
# Stored as frozenset keys so order doesn't matter for lookup.
SEVERITY_TRANSITION: dict = {
    frozenset({"MA", "HE"}): 0.0,
    frozenset({"MA", "Hard_EX"}): 1.0,
    frozenset({"MA", "Soft_EX"}): 2.0,
    frozenset({"HE", "Hard_EX"}): 1.0,
    frozenset({"HE", "Soft_EX"}): 2.0,
    frozenset({"Hard_EX", "Soft_EX"}): 1.5,
}

# morphological_confusability: inter-observer confusion (undirected)
MORPHOLOGICAL_CONFUSABILITY: dict = {
    frozenset({"MA", "HE"}): 0.8,
    frozenset({"Hard_EX", "Soft_EX"}): 0.5,
    frozenset({"MA", "Hard_EX"}): 0.1,
}

# pathophysiological_mechanism: directed, topology fixed (MA is upstream)
# Format: (src_name, dst_name) -> init_weight (weights are trainable)
PATHOPHYSIOLOGICAL_MECHANISM: dict = {
    ("MA", "HE"): 1.0,  # rupture -> hemorrhage
    ("MA", "Hard_EX"): 1.0,  # leakage -> lipid deposition
    ("MA", "Soft_EX"): 0.3,  # weak upstream prior
}


# ── Label Mapping ─────────────────────────────────────────────────────────────
# RAW_TO_KG_MAP: normalized raw string -> canonical KG node name.
#
# Normalization step (applied before lookup):
#   1. strip whitespace
#   2. lowercase
#   3. replace space / hyphen / dot with underscore
#   e.g. "Hard EX" -> "hard_ex" -> "Hard_EX"
#
# Ambiguous entries are marked [AMBIGUOUS] — verify against actual XML audit.
# If the dataset uses a different short form for Hard_EX, override via the
# custom_map argument of map_raw_to_kg() without editing this constant.
#
# Sources cross-checked against:
#   OIA-DDR, IDRiD (Indian Diabetic Retinopathy Image Dataset),
#   Kaggle EyePACS DR annotation conventions.

RAW_TO_KG_MAP: dict[str, str] = {
    # ── MA (Microaneurysm) ──────────────────────────────────────────────────
    "ma": "MA",
    "microaneurysm": "MA",
    "microaneurysms": "MA",
    "micro_aneurysm": "MA",
    # ── HE (Hemorrhage) ─────────────────────────────────────────────────────
    "he": "HE",
    "hemorrhage": "HE",
    "hemorrhages": "HE",
    "haemorrhage": "HE",  # British spelling variant
    "haemorrhages": "HE",
    # ── Hard_EX (Hard Exudate) ───────────────────────────────────────────────
    "hard_ex": "Hard_EX",
    "hardex": "Hard_EX",
    "hard_exudate": "Hard_EX",
    "hard_exudates": "Hard_EX",
    "he_ex": "Hard_EX",
    "ex": "Hard_EX",  # Confirmed: OIA-DDR uses `ex` for Hard_EX (waxy/discrete
    # bright spots). Verified via visual audit (seed=42, 20-sample). FROZEN 2025.
    "exudate": "Hard_EX",  # [AMBIGUOUS] if dataset uses generic "exudate"
    "exudates": "Hard_EX",  # confirm via audit whether soft/hard is merged
    # ── Soft_EX (Soft Exudate / Cotton-Wool Spot) ───────────────────────────
    "soft_ex": "Soft_EX",
    "softex": "Soft_EX",
    "soft_exudate": "Soft_EX",
    "soft_exudates": "Soft_EX",
    "cotton_wool": "Soft_EX",
    "cotton_wool_spot": "Soft_EX",
    "cws": "Soft_EX",
    "se": "Soft_EX",  # common short form in several DR datasets
}

# Labels explicitly marked ambiguous — downstream code may warn on these.
# `ex` removed after visual audit confirmed ex→Hard_EX in OIA-DDR (2025).
# `exudate`/`exudates` remain ambiguous as generic terms across other datasets.
AMBIGUOUS_RAW_LABELS: frozenset = frozenset({"exudate", "exudates"})

# KG nodes that may be absent from a dataset's label set but must remain in
# the graph for structural/clinical reasons (see module docstring).
# Soft_EX was formerly listed here because `se` → Soft_EX was unconfirmed.
# CONFIRMED via visual audit (seed=42, 20-sample, April 2025): `se` in OIA-DDR
# labels fluffy/ill-defined lesions → Soft_EX (111 images, 1 460 bboxes).
# Frozenset is now empty; retained as frozenset() for API compatibility.
CONCEPTUAL_ONLY_NODES: frozenset = frozenset()


# ── Mapping Functions ─────────────────────────────────────────────────────────


def normalize_raw_label(label: str) -> str:
    """Normalize a raw XML label string to a consistent intermediate form.

    Steps: strip → lowercase → replace [ -./] with underscore.

    Args:
        label: raw string from XML <name> tag, e.g. "Hard EX", "SOFT-EX".

    Returns:
        Normalized string, e.g. "hard_ex", "soft_ex".

    Examples:
        >>> normalize_raw_label("Hard EX")
        'hard_ex'
        >>> normalize_raw_label("  MA  ")
        'ma'
        >>> normalize_raw_label("Cotton-Wool")
        'cotton_wool'
    """
    result = label.strip().lower()
    for ch in (" ", "-", ".", "/"):
        result = result.replace(ch, "_")
    return result


def map_raw_to_kg(
    label: str,
    custom_map: "dict[str, str] | None" = None,
) -> "str | None":
    """Map a raw XML label to its canonical KG lesion node name.

    Applies normalize_raw_label() first, then looks up in RAW_TO_KG_MAP
    (or custom_map if provided). Emits a RuntimeWarning for ambiguous labels.

    Args:
        label:      Raw label string from XML <name> tag.
        custom_map: Optional override mapping (normalized_str -> canonical_name).
                    Merged over RAW_TO_KG_MAP; custom_map takes precedence.

    Returns:
        Canonical KG node name (e.g. "MA", "HE", "Hard_EX", "Soft_EX"),
        or None if the normalized label is not found in any map.

    Examples:
        >>> map_raw_to_kg("hard_ex")
        'Hard_EX'
        >>> map_raw_to_kg("unknown_lesion") is None
        True
    """
    import warnings

    norm = normalize_raw_label(label)
    active_map: dict = {**RAW_TO_KG_MAP, **(custom_map or {})}
    canonical = active_map.get(norm)

    if canonical is None:
        return None

    if norm in AMBIGUOUS_RAW_LABELS:
        warnings.warn(
            f"[ontology] Ambiguous label '{label}' (normalized: '{norm}') "
            f"mapped to '{canonical}'. Verify against XML audit output. "
            "Override via custom_map if needed.",
            RuntimeWarning,
            stacklevel=2,
        )

    return canonical


def get_all_kg_lesion_nodes() -> list:
    """Return ordered list of all C=4 canonical KG lesion node names.

    Includes conceptual-only nodes (e.g. Soft_EX) even if unobserved in data.
    Order matches LESION_NODES index assignment.

    Returns:
        ['MA', 'HE', 'Hard_EX', 'Soft_EX']
    """
    return sorted(LESION_NODES.keys(), key=lambda n: LESION_NODES[n])


def get_observed_kg_nodes(raw_labels: "list[str]") -> list:
    """Return canonical KG lesion node names that appear in the dataset.

    Maps each raw label to a KG node; deduplicates and sorts by node index.
    Unmappable labels are silently ignored (caller should audit separately).

    Args:
        raw_labels: List of raw label strings from XML audit output,
                    e.g. from xml_parser.collect_all_labels().

    Returns:
        Sorted list of canonical KG node names present in the dataset.
        Conceptual-only nodes (e.g. Soft_EX) are included ONLY if at least
        one raw label maps to them.

    Examples:
        >>> get_observed_kg_nodes(["MA", "HE", "hard_ex"])
        ['MA', 'HE', 'Hard_EX']
        >>> get_observed_kg_nodes(["MA", "HE"])  # Soft_EX absent
        ['MA', 'HE']
    """
    observed: set = set()
    for raw in raw_labels:
        canonical = map_raw_to_kg(raw)
        if canonical is not None and canonical in LESION_NODES:
            observed.add(canonical)
    return sorted(observed, key=lambda n: LESION_NODES[n])
