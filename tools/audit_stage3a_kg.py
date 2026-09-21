#!/usr/bin/env python
"""
audit_stage3a_kg.py
Phase 19N — Stage 3A: KG / C1 Readiness Audit

Validates the clinical knowledge graph (Contribution 1) before governance
gate training (Stage 3D). This audit must PASS before SDCA and compat_scores
are used in any governance experiment.

Pipeline executed:
  1. Build static KG from schema.py (clinical priors only)
  2. Initialize LGovNetEncoder (2-layer HeteroGAT)
  3. Run forward pass → node embeddings [9, 64]
  4. Compute SDM (pairwise lesion L2 distances)
  5. Normalize via normalize_sdca() → sdm_norm in [0,1]
  6. Compute compat_scores → {(lesion, zone): float ∈ (0,1)}
  7. Validate against gate criteria (professor Stage 3 acceptance spec)
  8. Save frozen KG state for Stage 3B evidence recomputation

Gate criteria (professor, 4diskgpt1.txt §5):
  1. All 4 lesion nodes present: MA, HE, Hard_EX, Soft_EX
  2. SDM_norm: shape 4×4 equivalent (6 unique pairs), diagonal=0, in [0,1], no NaN, not all same
  3. compat_scores: in [0,1], not all 0.5, high-affinity pairs clearly distinct
     Hard_EX↔Foveal > Hard_EX↔Peripheral (compat_prior biases this at init)
     Soft_EX↔Peripapillary > Soft_EX↔Peripheral (compat_prior biases this at init)
  4. Evidence vector re-run with real KG (replaces Stage 2 stubs)
  5. Report saved: outputs/stage3_audit/kg_evidence_audit.json

NOT in scope here:
  - LGG training (Stage 3D)
  - Mode 1 governance claims (Stage 3D+)
  - Dataset co-occurrence update (build_dataset_stats.py — can run before Stage 3D)

Outputs:
  outputs/stage3_audit/
    kg_evidence_audit.json      — full audit report with gate result
    sdm_norm.json               — frozen sdm_norm for Stage 3B evidence
    compat_scores.json          — frozen compat_scores for Stage 3B evidence
    kg_encoder_init.pt          — encoder init state_dict (reproducibility)
    nodes.json                  — 9 node specs (saved for traceability)
    edges.json                  — edge specs (saved for traceability)

Usage:
    python tools/audit_stage3a_kg.py
    python tools/audit_stage3a_kg.py --output_dir outputs/stage3_audit
    python tools/audit_stage3a_kg.py --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.join(_here, "..")
if _root not in sys.path:
    sys.path.insert(0, _root)

import torch

from lgovnet.src.kg.schema import export_nodes_edges, build_clinical_prior_sdm
from lgovnet.src.kg.builder import build_graph_from_artifacts
from lgovnet.src.kg.encoder import (
    build_encoder_inputs,
    build_gat_encoder,
    compute_sdm,
    compute_compat,
    normalize_sdca,
    LESION_NAMES,
    ZONE_NAMES,
)

_LESION_CLASSES = ["MA", "HE", "Hard_EX", "Soft_EX"]
_ZONE_NAMES = [
    "Foveal",
    "Parafoveal",
    "Peripapillary",
    "Vascular_Arcades",
    "Peripheral",
]

# Gate thresholds (professor spec)
_SDM_GATE_MIN_DISTINCT_VALUES = 3  # sdm_norm values must not all be the same
_COMPAT_GATE_HIGH_AFFINITY_MARGIN = (
    0.05  # high-affinity compat > neutral compat by at least this
)

# High-affinity pairs (from schema.py HIGH_AFFINITY + compat_prior)
_HIGH_AFFINITY_PAIRS = {
    ("Hard_EX", "Foveal"),
    ("Hard_EX", "Parafoveal"),
    ("Soft_EX", "Peripapillary"),
}

# Comparison pairs for clinical ordering check (soft gate)
_CLINICAL_ORDER_CHECKS = [
    # (high_affinity, low_affinity, description)
    (
        ("Hard_EX", "Foveal"),
        ("Hard_EX", "Peripheral"),
        "Hard_EX more plausible near fovea than periphery",
    ),
    (
        ("Hard_EX", "Parafoveal"),
        ("Hard_EX", "Peripheral"),
        "Hard_EX more plausible in parafoveal than periphery",
    ),
    (
        ("Soft_EX", "Peripapillary"),
        ("Soft_EX", "Peripheral"),
        "Soft_EX more plausible near optic disc than periphery",
    ),
]

# C2b: required clinical ordering checks for SDM (ICDR-derived).
# Each entry: ((lo_a, lo_b), (hi_a, hi_b), description)
# lo pair must have strictly smaller SDM than hi pair.
_SDM_CLINICAL_ORDER_CHECKS = [
    (
        ("MA", "HE"),
        ("MA", "Hard_EX"),
        "SDM(MA,HE) < SDM(MA,Hard_EX): ICDR grade-0 < grade-1 confusion cost",
    ),
    (
        ("MA", "Hard_EX"),
        ("MA", "Soft_EX"),
        "SDM(MA,Hard_EX) < SDM(MA,Soft_EX): ICDR grade-1 < grade-2",
    ),
    (
        ("HE", "Hard_EX"),
        ("HE", "Soft_EX"),
        "SDM(HE,Hard_EX) < SDM(HE,Soft_EX): ICDR grade-1 < grade-2",
    ),
    (
        ("MA", "HE"),
        ("HE", "Soft_EX"),
        "SDM(MA,HE) < SDM(HE,Soft_EX): grade-0 < grade-2 (clinically safest vs dangerous)",
    ),
]


def _key_to_str(pair: tuple) -> str:
    """Convert (ci, cj) tuple to JSON-safe key string."""
    return f"{pair[0]}__{pair[1]}"


def _save_sdm_norm(sdm_norm: dict, path: Path) -> None:
    """Save sdm_norm as JSON with string keys."""
    serializable = {_key_to_str(k): v for k, v in sdm_norm.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)


def _save_compat_scores(compat: dict, path: Path) -> None:
    """Save compat_scores as JSON with string keys."""
    serializable = {_key_to_str(k): round(v, 6) for k, v in compat.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)


def _validate_sdm_norm(sdm_norm: dict) -> Tuple[bool, List[str], List[str]]:
    """Check gate criteria 2: SDM_norm validity.

    Returns (passed, errors, notes).
    """
    errors: List[str] = []
    notes: List[str] = []

    # Check all 6 lesion pairs present
    expected_pairs = set()
    for i in range(len(_LESION_CLASSES)):
        for j in range(i + 1, len(_LESION_CLASSES)):
            expected_pairs.add((_LESION_CLASSES[i], _LESION_CLASSES[j]))
    missing = expected_pairs - set(sdm_norm.keys())
    if missing:
        errors.append(f"Missing SDM pairs: {missing}")

    # Check values
    norm_values = [v["norm"] for v in sdm_norm.values()]
    for pair, entry in sdm_norm.items():
        raw = entry["raw"]
        norm = entry["norm"]
        if raw < 0:
            errors.append(f"SDM raw({pair}) < 0: {raw:.4f}")
        if not (0.0 <= norm <= 1.0):
            errors.append(f"SDM norm({pair}) out of [0,1]: {norm:.4f}")
        if raw != raw:  # NaN check
            errors.append(f"SDM raw({pair}) is NaN")

    # Not all same
    unique_vals = set(round(v, 4) for v in norm_values)
    if len(unique_vals) < _SDM_GATE_MIN_DISTINCT_VALUES:
        errors.append(
            f"SDM_norm has < {_SDM_GATE_MIN_DISTINCT_VALUES} distinct values "
            f"(all near same value → KG not providing clinical distance signal)"
        )

    # Diagonal is 0 (implicitly: no pair is self)
    # No (ci, ci) pairs in our output, so diagonal=0 is structural ✓

    # Soft check: clinical severity ordering
    # SDM(MA, Hard_EX) should > SDM(MA, HE) — confusing MA↔HardEX changes ICDR grade
    ma_he = sdm_norm.get(("MA", "HE"), {}).get("norm", 0)
    ma_hex = sdm_norm.get(("MA", "Hard_EX"), {}).get("norm", 0)
    ma_sex = sdm_norm.get(("MA", "Soft_EX"), {}).get("norm", 0)
    if ma_he < ma_hex:
        notes.append(
            f"GOOD: SDM_norm(MA,HE)={ma_he:.4f} < SDM_norm(MA,Hard_EX)={ma_hex:.4f} "
            "(matches ICDR severity cost: MA↔HE cost=0 < MA↔Hard_EX cost=1)"
        )
    else:
        notes.append(
            f"NOTE: SDM_norm(MA,HE)={ma_he:.4f} ≥ SDM_norm(MA,Hard_EX)={ma_hex:.4f} "
            "at init — ordering will improve with training (soft check, not gate failure)"
        )
    if ma_hex < ma_sex:
        notes.append(
            f"GOOD: SDM_norm(MA,Hard_EX)={ma_hex:.4f} < SDM_norm(MA,Soft_EX)={ma_sex:.4f} "
            "(Soft_EX has highest ICDR severity cost)"
        )
    else:
        notes.append(
            f"NOTE: SDM_norm(MA,Hard_EX)={ma_hex:.4f} ≥ SDM_norm(MA,Soft_EX)={ma_sex:.4f} "
            "at init — will adapt with training (soft check)"
        )

    return len(errors) == 0, errors, notes


def _validate_compat_scores(compat: dict) -> Tuple[bool, List[str], List[str]]:
    """Check gate criteria 3: compat_scores validity.

    Returns (passed, errors, notes).
    """
    errors: List[str] = []
    notes: List[str] = []

    # Check all 20 pairs present (4 lesions × 5 zones)
    expected_pairs = set()
    for l in _LESION_CLASSES:
        for z in _ZONE_NAMES:
            expected_pairs.add((l, z))
    missing = expected_pairs - set(compat.keys())
    if missing:
        errors.append(f"Missing compat pairs: {missing}")

    # Check values in [0,1]
    for pair, val in compat.items():
        if not (0.0 < val < 1.0):
            errors.append(
                f"Compat{pair}={val:.4f} not in (0,1) — sigmoid should guarantee this"
            )
        if val != val:
            errors.append(f"Compat{pair} is NaN")

    # Not all 0.5
    unique_vals = set(round(v, 3) for v in compat.values())
    if unique_vals == {0.5}:
        errors.append("All compat_scores = 0.5 — clinical prior not influencing APS")

    # High-affinity pairs should be clearly above neutral
    clinical_ok = True
    for hi_pair, lo_pair, desc in _CLINICAL_ORDER_CHECKS:
        hi_val = compat.get(hi_pair, 0.5)
        lo_val = compat.get(lo_pair, 0.5)
        if hi_val > lo_val + _COMPAT_GATE_HIGH_AFFINITY_MARGIN:
            notes.append(
                f"GOOD: Compat{hi_pair}={hi_val:.4f} > Compat{lo_pair}={lo_val:.4f} "
                f"(+{hi_val-lo_val:.4f}) — {desc}"
            )
        else:
            notes.append(
                f"WARN: Compat{hi_pair}={hi_val:.4f} not clearly > Compat{lo_pair}={lo_val:.4f} "
                f"({hi_val-lo_val:+.4f}) — {desc} — compat_prior may not be overcoming random init"
            )
            clinical_ok = False

    if not clinical_ok:
        errors.append(
            "High-affinity compat pairs not clearly distinct from neutral pairs. "
            "Check compat_init_bias_mode='high_affinity_prior' is set in encoder."
        )

    return len(errors) == 0, errors, notes


def _validate_sdm_clinical_ordering(sdm: dict) -> Tuple[bool, List[str], List[str]]:
    """Check C2b: SDM values are clinically ordered per ICDR grading scale.

    This gate is meaningless on random-init encoder SDM (expected to WARN/FAIL).
    It should always PASS on build_clinical_prior_sdm() output by construction.
    Used here to confirm the clinical-prior SDM is correctly built.

    Returns (passed, errors, notes).
    """
    errors: List[str] = []
    notes: List[str] = []

    idx_order = {c: i for i, c in enumerate(_LESION_CLASSES)}

    def _get_norm(a: str, b: str) -> float:
        pair = (a, b) if idx_order[a] < idx_order[b] else (b, a)
        entry = sdm.get(pair)
        return entry["norm"] if entry else None

    for (lo_a, lo_b), (hi_a, hi_b), desc in _SDM_CLINICAL_ORDER_CHECKS:
        lo_val = _get_norm(lo_a, lo_b)
        hi_val = _get_norm(hi_a, hi_b)
        if lo_val is None or hi_val is None:
            errors.append(f"Missing pair for clinical order check: {desc}")
        elif lo_val < hi_val:
            notes.append(
                f"GOOD: SDM({lo_a},{lo_b})={lo_val:.4f} < SDM({hi_a},{hi_b})={hi_val:.4f} — {desc}"
            )
        else:
            errors.append(
                f"FAIL: SDM({lo_a},{lo_b})={lo_val:.4f} >= SDM({hi_a},{hi_b})={hi_val:.4f}"
                f" — {desc}"
            )

    return len(errors) == 0, errors, notes


def run_kg_audit(args) -> dict:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Stage3A] === Phase 19N Stage 3A KG Audit ===")
    print(f"[Stage3A] output_dir={out_dir}  seed={args.seed}")

    torch.manual_seed(args.seed)
    device = torch.device("cpu")  # KG is tiny (9 nodes), CPU sufficient

    # ── Step 1: Build static KG ────────────────────────────────────────────────
    nodes_path = out_dir / "nodes.json"
    edges_path = out_dir / "edges.json"
    print(f"[Stage3A] Step 1: Building static KG from schema.py...")
    export_nodes_edges(nodes_path=str(nodes_path), edges_path=str(edges_path))
    print(f"[Stage3A]   Saved: {nodes_path.name}, {edges_path.name}")

    # ── Step 2: Assemble BuiltGraph ─────────────────────────────────────────────
    print(f"[Stage3A] Step 2: Assembling BuiltGraph (static_only mode)...")
    kg = build_graph_from_artifacts(
        nodes_json=nodes_path,
        edges_json=edges_path,
        cooccurrence_csv=None,  # static_only — no dataset stats yet
        zone_affinity_csv=None,
    )
    meta = kg.get("meta", {})
    n_edges_by_type = {
        etype: len(elist) for etype, elist in kg.get("edges", {}).items()
    }
    print(f"[Stage3A]   Nodes: {len(kg['nodes'])}  Edges by type: {n_edges_by_type}")

    # ── Step 3: Build encoder inputs ────────────────────────────────────────────
    print(f"[Stage3A] Step 3: Building encoder inputs (x, edge_index, ...)...")
    enc_inputs = build_encoder_inputs(kg, node_feat_dim=17, device=device)
    print(
        f"[Stage3A]   x.shape={enc_inputs['x'].shape}  "
        f"edge_index.shape={enc_inputs['edge_index'].shape}"
    )

    # ── Step 4: Initialize encoder ──────────────────────────────────────────────
    print(f"[Stage3A] Step 4: Initializing LGovNetEncoder (seed={args.seed})...")
    encoder = build_gat_encoder(
        node_feat_dim=17,
        hidden_dim=64,
        embedding_dim=64,
        edge_type_dim=16,
        dropout=0.0,  # no dropout during init-phase audit
        compat_init_bias_mode="high_affinity_prior",
    ).to(device)
    encoder.eval()
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"[Stage3A]   Encoder parameters: {n_params:,}")

    # Save init state_dict for reproducibility
    ckpt_path = out_dir / "kg_encoder_init.pt"
    torch.save(
        {
            "state_dict": encoder.state_dict(),
            "seed": args.seed,
            "node_feat_dim": 17,
            "hidden_dim": 64,
            "embedding_dim": 64,
            "edge_type_dim": 16,
            "compat_init_bias_mode": "high_affinity_prior",
            "stage": "3A_init",
        },
        ckpt_path,
    )
    print(f"[Stage3A]   Saved encoder init: {ckpt_path.name}")

    # ── Step 5: Forward pass → embeddings ──────────────────────────────────────
    print(f"[Stage3A] Step 5: Forward pass → node embeddings...")
    with torch.no_grad():
        embeddings = encoder.forward_node_embeddings(
            x=enc_inputs["x"],
            edge_index=enc_inputs["edge_index"],
            edge_type_ids=enc_inputs["edge_type_ids"],
            edge_weights=enc_inputs["edge_weights"],
        )
    node_names = enc_inputs["node_names"]
    print(f"[Stage3A]   Embeddings: {embeddings.shape}  node_names={node_names}")

    # ── Step 6: Compute SDM and normalize ──────────────────────────────────────
    print(f"[Stage3A] Step 6: Computing SDM_learned (pairwise L2 distances)...")
    with torch.no_grad():
        sdm_raw = compute_sdm(embeddings, node_names=node_names)
        sdm_norm = normalize_sdca(sdm_raw, method="log_scaled")

    print(f"[Stage3A]   SDM pairs: {len(sdm_raw)}")
    for pair in sorted(sdm_raw.keys()):
        raw = sdm_raw[pair]
        norm = sdm_norm[pair]["norm"]
        print(f"[Stage3A]     SDM({pair[0]},{pair[1]}): raw={raw:.4f}  norm={norm:.4f}")

    # Encoder SDM: saved as diagnostic artifact only (random-init, not clinical-prior)
    enc_sdm_path = out_dir / "sdm_norm_encoder_diagnostic.json"
    _save_sdm_norm(sdm_norm, enc_sdm_path)
    print(
        f"[Stage3A]   Saved (encoder diagnostic, NOT for governance): {enc_sdm_path.name}"
    )

    # ── Step 6b: Build clinical-prior SDM (ICDR-derived, deterministic) ─────────
    print(
        f"[Stage3A] Step 6b: Building clinical-prior SDM from SEVERITY_TRANSITION (ICDR)..."
    )
    sdm_prior = build_clinical_prior_sdm()
    sdm_prior_path = out_dir / "sdm_norm_clinical_prior.json"
    _save_sdm_norm(sdm_prior, sdm_prior_path)
    print(
        f"[Stage3A]   SDM clinical-prior ({len(sdm_prior)} pairs) — grading-aware C1 claim:"
    )
    for pair in sorted(sdm_prior.keys()):
        raw_cost = sdm_prior[pair]["raw"]
        norm_val = sdm_prior[pair]["norm"]
        print(
            f"[Stage3A]     SDM_prior({pair[0]},{pair[1]}): icdr_cost={raw_cost:.2f}  norm={norm_val:.4f}"
        )
    print(f"[Stage3A]   Saved: {sdm_prior_path.name}")

    # ── Step 7: Compute compat_scores ───────────────────────────────────────────
    print(
        f"[Stage3A] Step 7: Computing compat_scores (Compat(c,a) = σ(e_c W e_a + prior))..."
    )
    with torch.no_grad():
        compat_scores = compute_compat(
            embeddings=embeddings,
            W_compat=encoder.W_compat,
            node_names=node_names,
            lesion_names=LESION_NAMES,
            zone_names=ZONE_NAMES,
            compat_prior=encoder.compat_prior,
        )

    print(f"[Stage3A]   Compat scores ({len(compat_scores)} pairs):")
    for lesion in LESION_NAMES:
        row_str = "  ".join(f"{z}={compat_scores[(lesion, z)]:.4f}" for z in ZONE_NAMES)
        print(f"[Stage3A]     {lesion:10s}: {row_str}")

    compat_path = out_dir / "compat_scores.json"
    _save_compat_scores(compat_scores, compat_path)
    print(f"[Stage3A]   Saved: {compat_path.name}")

    # ── Step 8: Gate validation ─────────────────────────────────────────────────
    print(f"\n[Stage3A] Step 8: Gate validation...")

    # Criterion 1: all 4 lesion nodes
    has_all_nodes = all(n in node_names for n in _LESION_CLASSES)
    c1_pass = has_all_nodes
    print(f"[Stage3A]   C1 (all lesion nodes): {'PASS' if c1_pass else 'FAIL'}")

    # Criterion 2: SDM_norm validity
    c2_pass, c2_errors, c2_notes = _validate_sdm_norm(sdm_norm)
    print(f"[Stage3A]   C2 (SDM_norm validity): {'PASS' if c2_pass else 'FAIL'}")
    for e in c2_errors:
        print(f"[Stage3A]     ERROR: {e}")
    for n in c2_notes:
        print(f"[Stage3A]     {n}")

    # Criterion 3: compat_scores validity
    c3_pass, c3_errors, c3_notes = _validate_compat_scores(compat_scores)
    print(f"[Stage3A]   C3 (compat_scores validity): {'PASS' if c3_pass else 'FAIL'}")
    for e in c3_errors:
        print(f"[Stage3A]     ERROR: {e}")
    for n in c3_notes:
        print(f"[Stage3A]     {n}")

    # ── Step 8c: C2b — Clinical ordering gate on clinical-prior SDM ─────────────
    c2b_pass, c2b_errors, c2b_notes = _validate_sdm_clinical_ordering(sdm_prior)
    print(f"[Stage3A]   C2b (SDM clinical ordering): {'PASS' if c2b_pass else 'FAIL'}")
    for e in c2b_errors:
        print(f"[Stage3A]     ERROR: {e}")
    for n in c2b_notes:
        print(f"[Stage3A]     {n}")

    # Also run C2b on encoder SDM to expose the inversion (informational only)
    enc_c2b_pass, enc_c2b_errors, _ = _validate_sdm_clinical_ordering(sdm_norm)
    if not enc_c2b_pass:
        print(
            f"[Stage3A]   INFO: Encoder SDM C2b = WARN (expected at random init — "
            f"clinical ordering will improve with training):"
        )
        for e in enc_c2b_errors:
            print(f"[Stage3A]     {e}")

    gate_pass = c1_pass and c2_pass and c2b_pass and c3_pass

    # ── Step 9: Evidence recomputation sanity check ─────────────────────────────
    print(
        f"\n[Stage3A] Step 9: Evidence recomputation sanity check (using clinical-prior SDM)..."
    )
    ev_ok = _sanity_check_evidence(sdm_prior, compat_scores)
    print(f"[Stage3A]   Evidence vector format: {'PASS' if ev_ok else 'FAIL'}")

    # ── Summary ─────────────────────────────────────────────────────────────────
    summary = {
        "stage": "3A",
        "seed": args.seed,
        "compat_init_bias_mode": "high_affinity_prior",
        "n_nodes": len(kg["nodes"]),
        "n_edges_by_type": n_edges_by_type,
        "n_encoder_params": n_params,
        "sdm_encoder_diagnostic": {
            _key_to_str(k): round(v["norm"], 6) for k, v in sdm_norm.items()
        },
        "sdm_clinical_prior": {
            _key_to_str(k): round(v["norm"], 6) for k, v in sdm_prior.items()
        },
        "compat_scores": {
            _key_to_str(k): round(v, 6) for k, v in compat_scores.items()
        },
        "gate_c1_all_lesion_nodes": c1_pass,
        "gate_c2a_sdm_numeric": c2_pass,
        "gate_c2a_errors": c2_errors,
        "gate_c2a_notes": c2_notes,
        "gate_c2b_sdm_clinical_ordering": c2b_pass,
        "gate_c2b_errors": c2b_errors,
        "gate_c2b_notes": c2b_notes,
        "gate_c2b_encoder_sdm_pass": enc_c2b_pass,
        "gate_c3_compat": c3_pass,
        "gate_c3_errors": c3_errors,
        "gate_c3_notes": c3_notes,
        "gate_c4_evidence_format": ev_ok,
        "gate_overall": gate_pass and ev_ok,
        "sdm_framing": {
            "encoder_sdm": "random_init_diagnostic — clinical ordering NOT guaranteed",
            "clinical_prior_sdm": "ICDR_severity_transition — grading-aware, use for Stage 3B",
        },
        "artifacts": {
            "nodes_json": str(nodes_path),
            "edges_json": str(edges_path),
            "sdm_norm_encoder_diagnostic_json": str(enc_sdm_path),
            "sdm_norm_clinical_prior_json": str(sdm_prior_path),
            "compat_scores_json": str(compat_path),
            "encoder_init_pt": str(ckpt_path),
        },
    }

    report_path = out_dir / "kg_evidence_audit.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[Stage3A] ══════════════════ SUMMARY ══════════════════")
    print(f"  C1  (lesion nodes):        {'PASS' if c1_pass else 'FAIL'}")
    print(f"  C2a (SDM numeric):         {'PASS' if c2_pass else 'FAIL'}")
    print(
        f"  C2b (SDM clinical order):  {'PASS' if c2b_pass else 'FAIL'}"
        f"  [encoder SDM C2b: {'PASS' if enc_c2b_pass else 'WARN — expected at random init'}]"
    )
    print(f"  C3  (compat_scores):       {'PASS' if c3_pass else 'FAIL'}")
    print(f"  C4  (evidence format):     {'PASS' if ev_ok else 'FAIL'}")
    print(f"  GATE:                      {'PASS' if gate_pass and ev_ok else 'FAIL'}")
    print(f"[Stage3A] ═══════════════════════════════════════════════")
    print(f"[Stage3A] Report: {report_path}")
    print(
        "\n[Stage3A] Stage 3B artifacts ready:"
        f"\n  sdm_norm_clinical_prior.json  ← use this for governance evidence (ICDR-grading-aware)"
        f"\n  compat_scores.json            ← use this for APS (prior-calibrated)"
        f"\n  sdm_norm_encoder_diagnostic.json ← diagnostic only, NOT for governance claims"
        f"\n\n[Stage3A] Next: python tools/audit_stage3b_evidence.py"
        f"\n  --kg_dir outputs/stage3_audit"
        f"\n  --cache_dir outputs/detector_cache/primary/oiaddr/valid"
        f"\n  --split valid"
    )

    return summary


def _sanity_check_evidence(sdm_norm: dict, compat_scores: dict) -> bool:
    """Evidence vector format check with two synthetic DetectorBoxes.

    Checks (C4 spec):
      1. evidence_vector exists and has length 4
      2. all components are finite
      3. all components are in [0.0, 1.0]
      4. APS = 0.5 when anatomy_context=None (od_confidence='n/a' fail-safe)
      5. anatomy_uncertain = True under no anatomy context

    NOTE: SDCA = 0.0 is VALID for clinical-prior SDM when majority=MA and
    runner_up=HE (SDM_clinical_prior(MA,HE) = 0.0 by ICDR design — zero-cost
    confusion pair). Do NOT treat SDCA=0.0 as a format error.

    Two boxes are tested:
      box_A: MA(0.70) vs HE(0.10) runner-up → SDCA should be 0.0 (MA↔HE cost)
      box_B: MA(0.70) vs Soft_EX(0.20) runner-up → SDCA should be 1.0 (MA↔Soft_EX cost)
    """
    import math

    try:
        from lgovnet.src.governance.evidence import build_evidence_for_bbox
        from lgovnet.src.governance.contracts import make_placeholder_pss

        def _check_box(box: dict, label: str) -> bool:
            pss = make_placeholder_pss(box["bbox_id"])
            ev = build_evidence_for_bbox(
                box=box,
                pss_result=pss,
                sdm_norm=sdm_norm,
                compat_scores=compat_scores,
                anatomy_context=None,
            )
            vec = ev.get("evidence_vector", [])
            if len(vec) == 4:
                vec_str = (
                    f"SAS={vec[0]:.4f}, SDCA={vec[1]:.4f}, "
                    f"APS={vec[2]:.4f}, PSS={vec[3]:.4f}"
                )
            else:
                vec_str = f"MALFORMED len={len(vec)}"
            print(f"[Stage3A]   {label}:")
            print(
                f"[Stage3A]     ev=[{vec_str}]  anatomy_uncertain={ev.get('anatomy_uncertain')}"
            )

            ok = (
                len(vec) == 4
                and all(math.isfinite(v) for v in vec)
                and all(0.0 <= v <= 1.0 for v in vec)
                and ev.get("anatomy_uncertain") is True
                and abs(ev.get("aps", -1) - 0.5) < 0.01
            )
            sdca = ev.get("sdca_norm", -1)
            if sdca == 0.0:
                print(
                    f"[Stage3A]     NOTE: SDCA=0.0 — clinically correct for zero-cost pair (MA↔HE under ICDR)"
                )
            return ok

        # Box A: MA runner-up=HE → SDM_clinical(MA,HE)=0.0 expected
        box_a = {
            "bbox_id": "stage3a_sanity_b0000",
            "head_predictions": [],
            "ensemble_probs": {
                "MA": 0.70,
                "HE": 0.10,
                "Hard_EX": 0.10,
                "Soft_EX": 0.10,
            },
            "assigned_zone": "Foveal",
        }
        # Box B: MA runner-up=Soft_EX → SDM_clinical(MA,Soft_EX)=1.0 expected
        box_b = {
            "bbox_id": "stage3a_sanity_b0001",
            "head_predictions": [],
            "ensemble_probs": {
                "MA": 0.70,
                "HE": 0.05,
                "Hard_EX": 0.05,
                "Soft_EX": 0.20,
            },
            "assigned_zone": "Peripapillary",
        }
        ok_a = _check_box(box_a, "SanityA (MA→HE runner-up, expect SDCA=0.0)")
        ok_b = _check_box(box_b, "SanityB (MA→Soft_EX runner-up, expect SDCA=1.0)")
        return ok_a and ok_b

    except Exception as exc:
        print(f"[Stage3A]   Evidence sanity check failed: {exc}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Stage 3A KG audit — validate SDM_norm and compat_scores."
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/stage3_audit",
        help="Directory for all audit outputs (default: outputs/stage3_audit).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for encoder initialization (default: 42).",
    )
    args = parser.parse_args()

    summary = run_kg_audit(args)
    gate = summary.get("gate_overall", False)
    sys.exit(0 if gate else 1)


if __name__ == "__main__":
    main()
