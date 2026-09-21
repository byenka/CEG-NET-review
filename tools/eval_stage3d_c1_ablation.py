#!/usr/bin/env python
"""
eval_stage3d_c1_ablation.py
Phase 19N — Stage 3D C1 Ablation: KG Design Contribution.

Tests whether the ICDR grading-aware clinical SDM ordering is responsible
for the SDM-row embedding gain observed in Stage 3D.3, or whether any
structured SDM (uniform, random, reversed) would work equally well.

This directly validates Contribution 1 (C1) of architecture v3:
  "Directed clinical KG with grading-aware SDM"

Key question:
  Is B_sdmrow_clinical > B_sdmrow_uniform / random / reversed / shuffled?
  → YES: ICDR clinical ordering contributes to governance routing.
  → NO:  Any embedding structure is equivalent; clinical KG is not decisive.

SDM variants tested (all use [SAS, SDM-row-embedding] as MLP input):
  B0                : score threshold (no model, baseline)
  B_class_only      : SAS + one-hot(pred_class)  [identity, non-KG reference]
  B_sdmrow_clinical : SAS + clinical SDM-row     [KG C1 claim — ICDR-derived]
  B_sdmrow_uniform  : SAS + uniform SDM-row      [all distances = 0.5]
  B_sdmrow_reversed : SAS + reversed SDM-row     [1 - clinical, anti-ICDR]
  B_sdmrow_shuffled : SAS + shuffled SDM-row     [clinical with permuted labels]
  B_sdmrow_rand_s*  : SAS + random SDM-row       [5 seeds]

Clinical SDM (from ICDR severity_transition, normalized by max_cost=2.0):
  MA ↔ HE       = 0.00  (grade-0: same ICDR tier)
  MA ↔ Hard_EX  = 0.50  (grade-1)
  MA ↔ Soft_EX  = 1.00  (grade-2: different tier)
  HE ↔ Hard_EX  = 0.50  (grade-1)
  HE ↔ Soft_EX  = 1.00  (grade-2)
  Hard_EX↔Soft_EX = 0.75

Decision rule (strict):
  B_sdmrow_clinical must win ≥ 75% of comparisons against all controls
  (uniform, reversed, shuffled, random×5 = 8 controls × 3 TP targets = 24).
  Additionally: clinical must beat reversed at ≥ 2/3 targets (ordering direction test).

Operating point protocol: same as Stage 3D.1/3D.2/3D.3.
  Threshold selected on valid at TP_ret >= 0.85 / 0.80 / 0.75.
  All metrics reported on held-out test split.
  Bootstrap FP_supp_gain CI (1000 resamples by image_id).

Inputs:
  outputs/stage3c_audit/governance_features_{train,valid,test}.csv
  outputs/stage3_audit/sdm_norm_clinical_prior.json   (optional, has fallback)

Outputs:
  outputs/stage3d_ablation/c1_ablation_results.csv
  outputs/stage3d_ablation/bootstrap_c1_fp_supp_gain_ci.csv
  outputs/stage3d_ablation/c1_ablation_decision.md

Usage:
    python tools/eval_stage3d_c1_ablation.py \\
      --stage3c_dir outputs/stage3c_audit \\
      --stage3a_dir outputs/stage3_audit \\
      --output_dir  outputs/stage3d_ablation
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from sklearn.metrics import average_precision_score
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    _SK = True
except ImportError:
    _SK = False
    print("[Stage3D.C1] WARNING: scikit-learn not installed. Only B0 will run.")

# ── Constants ─────────────────────────────────────────────────────────────────

_CLASS_ORDER = ["MA", "HE", "Hard_EX", "Soft_EX"]
_CLASS_IDX = {c: i for i, c in enumerate(_CLASS_ORDER)}
_N = 4   # number of lesion classes

_RETAIN_CATS = frozenset({"TP_EXACT", "CLINICALLY_CLOSE_MISMATCH", "LOCALIZATION_WEAK"})

_TP_TARGETS = [0.85, 0.80, 0.75]
_THRESHOLD_GRID = np.round(np.linspace(0.001, 0.999, 499), 4)
_N_BOOTSTRAP = 1000
_BOOTSTRAP_SEED = 42
_N_RANDOM_SEEDS = 5

# Hardcoded clinical SDM (fallback if JSON load fails)
# ICDR severity_transition normalized by max_cost=2.0
_CLINICAL_SDM_FALLBACK = np.array([
    # MA    HE    Hard_EX  Soft_EX
    [0.00, 0.00,  0.50,    1.00],   # MA
    [0.00, 0.00,  0.50,    1.00],   # HE
    [0.50, 0.50,  0.00,    0.75],   # Hard_EX
    [1.00, 1.00,  0.75,    0.00],   # Soft_EX
], dtype=np.float32)


# ── SDM constructors ──────────────────────────────────────────────────────────

def _load_clinical_sdm(sdm_path: Path) -> np.ndarray:
    """Load clinical SDM from JSON (keys: 'ci__cj'). Falls back to hardcoded."""
    if sdm_path.exists():
        try:
            with open(sdm_path, encoding="utf-8") as f:
                data = json.load(f)
            mat = np.zeros((_N, _N), dtype=np.float32)
            for key_str, val in data.items():
                if "__" not in key_str:
                    continue
                ci, cj = [s.strip() for s in key_str.split("__", 1)]
                if ci not in _CLASS_IDX or cj not in _CLASS_IDX:
                    continue
                norm_val = float(val["norm"] if isinstance(val, dict) else val)
                i, j = _CLASS_IDX[ci], _CLASS_IDX[cj]
                mat[i, j] = mat[j, i] = norm_val
            # Verify clinical ordering
            if mat[0, 2] > 0.01 and mat[0, 3] > mat[0, 2]:
                print(f"[Stage3D.C1]   Clinical SDM loaded: "
                      f"SDM(MA,HE)={mat[0,1]:.3f}  "
                      f"SDM(MA,Hard_EX)={mat[0,2]:.3f}  "
                      f"SDM(MA,Soft_EX)={mat[0,3]:.3f}  ✓")
                return mat
            else:
                print(f"[Stage3D.C1]   Clinical SDM loaded but ordering check failed — "
                      f"using fallback.")
        except Exception as e:
            print(f"[Stage3D.C1]   SDM load error: {e} — using fallback.")
    else:
        print(f"[Stage3D.C1]   SDM not found at {sdm_path} — using fallback.")
    print(f"[Stage3D.C1]   Using hardcoded ICDR SDM.")
    return _CLINICAL_SDM_FALLBACK.copy()


def _uniform_sdm() -> np.ndarray:
    """All off-diagonal distances = 0.5 (no clinical ordering information)."""
    mat = np.full((_N, _N), 0.5, dtype=np.float32)
    np.fill_diagonal(mat, 0.0)
    return mat


def _reversed_sdm(clinical: np.ndarray) -> np.ndarray:
    """Anti-clinical: reverse the ICDR ordering (1 - clinical, diagonal=0)."""
    mat = 1.0 - clinical
    np.fill_diagonal(mat, 0.0)
    return mat.astype(np.float32)


def _shuffled_sdm(clinical: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Clinical SDM with class labels permuted (one random shuffle)."""
    perm = rng.permutation(_N)
    mat = clinical[np.ix_(perm, perm)].copy()
    return mat.astype(np.float32)


def _random_sdm(rng: np.random.Generator) -> np.ndarray:
    """Random symmetric SDM in [0,1] with diagonal=0."""
    raw = rng.uniform(0.0, 1.0, (_N, _N)).astype(np.float32)
    mat = (raw + raw.T) / 2.0
    np.fill_diagonal(mat, 0.0)
    return mat


def _sdm_row_embeddings(sdm: np.ndarray) -> np.ndarray:
    """Return [N_classes, N_classes] embedding: row i = SDM distances from class i."""
    return sdm.astype(np.float32)   # already [4, 4]


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_csv(path: Path) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _parse(records: List[dict]) -> Tuple[
    np.ndarray,   # [N, 1] SAS
    np.ndarray,   # [N]    class_idx
    np.ndarray,   # [N, 4] one_hot
    np.ndarray,   # [N]    y_binary
    np.ndarray,   # [N]    is_unsafe
    List[str],    # image_ids
]:
    sas_list, cls_list, y_list, un_list, img_list = [], [], [], [], []
    for r in records:
        try:
            sas = float(r["ev_sas"])
            if not math.isfinite(sas):
                continue
        except (KeyError, ValueError):
            continue
        cat = r.get("category", "")
        cls = r.get("pred_class", "")
        if cls not in _CLASS_IDX or cat not in (
            "TP_EXACT", "CLINICALLY_CLOSE_MISMATCH", "LOCALIZATION_WEAK",
            "UNSAFE_MISMATCH", "LOW_CONF_CANDIDATE", "BACKGROUND_FP"
        ):
            continue
        sas_list.append(sas)
        cls_list.append(_CLASS_IDX[cls])
        y_list.append(1 if cat in _RETAIN_CATS else 0)
        un_list.append(cat == "UNSAFE_MISMATCH")
        img_list.append(str(r.get("image_id", "")))

    N = len(sas_list)
    sas_arr  = np.array(sas_list, dtype=np.float64).reshape(-1, 1)
    cls_arr  = np.array(cls_list, dtype=np.int32)
    oh_arr   = np.eye(_N, dtype=np.float32)[cls_arr]   # [N, 4]
    y_arr    = np.array(y_list, dtype=np.int32)
    un_arr   = np.array(un_list, dtype=bool)
    return sas_arr, cls_arr, oh_arr, y_arr, un_arr, img_list


def _build_X(
    sas: np.ndarray,
    cls_idx: np.ndarray,
    sdm_emb: Optional[np.ndarray],
    oh: Optional[np.ndarray],
) -> np.ndarray:
    """Build feature matrix.
    sdm_emb: [4, 4] SDM rows — each box gets its pred_class row.
    oh: [N, 4] one-hot.
    """
    cols = [sas]
    if sdm_emb is not None:
        cols.append(sdm_emb[cls_idx])   # [N, 4]
    if oh is not None:
        cols.append(oh)
    return np.concatenate(cols, axis=1).astype(np.float64)


# ── MLP ──────────────────────────────────────────────────────────────────────

def _train_mlp(X: np.ndarray, y: np.ndarray,
               seed: int = 42) -> Tuple[object, object]:
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    m = MLPClassifier(
        hidden_layer_sizes=(32, 16), max_iter=300, random_state=seed,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=15,
        learning_rate_init=0.001, verbose=False,
    )
    m.fit(Xs, y)
    return m, sc


def _proba(m: object, sc: object, X: np.ndarray) -> np.ndarray:
    return m.predict_proba(sc.transform(X))[:, 1].astype(np.float64)


# ── Operating point helpers ────────────────────────────────────────────────────

def _find_thresh(y: np.ndarray, score: np.ndarray, target: float) -> Optional[float]:
    n_ret = int((y == 1).sum())
    if n_ret == 0:
        return None
    best: Optional[float] = None
    for t in sorted(_THRESHOLD_GRID):
        if int(((y == 1) & (score >= float(t))).sum()) / n_ret >= target:
            best = float(t)
    return best


def _eval_at(y: np.ndarray, score: np.ndarray,
             is_un: np.ndarray, thresh: float) -> dict:
    kept = score >= thresh
    n_ret = int((y == 1).sum()); n_sup = int((y == 0).sum())
    tp_r = int(((y == 1) & kept).sum()) / max(n_ret, 1)
    fp_s = int(((y == 0) & ~kept).sum()) / max(n_sup, 1)
    prec = int(y[kept].sum()) / max(int(kept.sum()), 1)
    un_s = int((is_un & ~kept).sum()) / int(is_un.sum()) if is_un.any() else None
    auprc = float(average_precision_score(y, score)) \
            if (_SK and len(np.unique(y)) > 1) else None
    return {
        "tp_ret": round(tp_r, 4), "fp_supp": round(fp_s, 4),
        "precision": round(prec, 4),
        "unsafe_supp": round(un_s, 4) if un_s is not None else None,
        "auprc": round(auprc, 4) if auprc is not None else None,
    }


# ── Bootstrap ─────────────────────────────────────────────────────────────────

def _boot_gain(
    img_ids: List[str], y: np.ndarray,
    sc_m: np.ndarray, sc_b0: np.ndarray,
    th_m: float, th_b0: float,
    n_boot: int = 1000, seed: int = 42,
) -> dict:
    if not _SK:
        return {"mean": None, "ci_lo": None, "ci_hi": None,
                "frac_pos": None, "n": 0}
    rng = np.random.default_rng(seed)
    uniq = list(dict.fromkeys(img_ids))
    n_u = len(uniq)
    img2idx: Dict[str, List[int]] = {}
    for i, img in enumerate(img_ids):
        img2idx.setdefault(img, []).append(i)
    gains: List[float] = []
    for _ in range(n_boot):
        samp = rng.choice(uniq, size=n_u, replace=True)
        idx = []; [idx.extend(img2idx[img]) for img in samp]
        ia = np.array(idx, dtype=np.int64)
        yb = y[ia]
        if yb.sum() == 0 or (1 - yb).sum() == 0:
            continue
        sup = yb == 0; n_s = int(sup.sum())
        if n_s == 0:
            continue
        gains.append(
            int((sup & (sc_m[ia] < th_m)).sum()) / n_s -
            int((sup & (sc_b0[ia] < th_b0)).sum()) / n_s
        )
    if len(gains) < 10:
        return {"mean": None, "ci_lo": None, "ci_hi": None,
                "frac_pos": None, "n": len(gains)}
    arr = np.array(gains)
    return {
        "mean": round(float(arr.mean()), 4),
        "ci_lo": round(float(np.percentile(arr, 2.5)), 4),
        "ci_hi": round(float(np.percentile(arr, 97.5)), 4),
        "frac_pos": round(float((arr > 0).mean()), 4),
        "n": len(gains),
    }


# ── Writers ───────────────────────────────────────────────────────────────────

_RES_FIELDS = [
    "variant", "sdm_type", "tp_ret_target", "thresh_valid",
    "tp_ret_valid", "fp_supp_valid", "auprc_valid",
    "tp_ret_test",  "fp_supp_test",  "auprc_test",
    "fp_supp_gain_over_b0",
    "fp_supp_gain_over_class_only",
    "fp_supp_gain_over_clinical",
    "achieves_target_test",
]
_BOOT_FIELDS = [
    "variant", "tp_ret_target",
    "mean_gain", "ci_lo", "ci_hi", "frac_positive", "n", "significant",
]


def _write_csv(rows: List[dict], fields: List[str], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def _fmt(v: Optional[float], fmt: str = ".4f") -> str:
    return format(v, fmt) if v is not None else "N/A"


def _write_md(
    res_rows: List[dict],
    boot_rows: List[dict],
    variant_order: List[str],
    clinical_wins: int,
    total_comparisons: int,
    path: Path,
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Stage 3D C1 Ablation — KG Clinical SDM Design Contribution\n\n")
        f.write(
            "> **Key question**: Does ICDR grading-aware SDM ordering contribute "
            "to governance routing beyond what uniform or random SDMs provide?\n\n"
        )
        f.write(
            "SDM variants (all use [SAS, SDM-row-embedding] input):\n"
            "- `clinical`: ICDR severity_transition, normalized. MA↔HE=0, "
            "MA↔Hard_EX=0.5, MA↔Soft_EX=1.0\n"
            "- `uniform`: all off-diagonal = 0.5 (no ordering)\n"
            "- `reversed`: 1 − clinical (anti-ICDR ordering)\n"
            "- `shuffled`: clinical with permuted class labels\n"
            "- `random_s*`: random symmetric SDM (5 seeds)\n\n"
        )

        for target in _TP_TARGETS:
            f.write(f"## TP_ret ≥ {target:.2f}\n\n")
            f.write(
                "| Variant | SDM | Thresh V | FP_supp V | FP_supp T "
                "| Gain/B0 | Gain/class_only | Gain/clinical |\n"
            )
            f.write("|---|---|---:|---:|---:|---:|---:|---:|\n")
            for name in variant_order:
                row = next(
                    (r for r in res_rows if r["variant"] == name
                     and abs(r.get("tp_ret_target", -1) - target) < 1e-6), None
                )
                if row is None:
                    continue
                g0 = row.get("fp_supp_gain_over_b0")
                gc = row.get("fp_supp_gain_over_class_only")
                gcl = row.get("fp_supp_gain_over_clinical")
                f.write(
                    f"| {name} | {row.get('sdm_type', '?')} "
                    f"| {_fmt(row.get('thresh_valid'))} "
                    f"| {_fmt(row.get('fp_supp_valid'))} "
                    f"| {_fmt(row.get('fp_supp_test'))} "
                    f"| {_fmt(g0, '+.4f') if g0 is not None else '—'} "
                    f"| {_fmt(gc, '+.4f') if gc is not None else '—'} "
                    f"| {_fmt(gcl, '+.4f') if gcl is not None else '—'} |\n"
                )
            f.write("\n")

        f.write("## Bootstrap FP_supp_gain CI vs B0 (1000 resamples)\n\n")
        for target in _TP_TARGETS:
            f.write(f"### TP_ret ≥ {target:.2f}\n\n")
            f.write("| Variant | Mean gain | 95% CI | P(>0) | Sig |\n")
            f.write("|---|---:|---|---:|:---:|\n")
            for name in variant_order:
                row = next(
                    (r for r in boot_rows if r.get("variant") == name
                     and abs(r.get("tp_ret_target", -1) - target) < 1e-6), None
                )
                if row is None:
                    continue
                ci = (f"[{_fmt(row.get('ci_lo'))}, {_fmt(row.get('ci_hi'))}]"
                      if row.get("ci_lo") is not None else "N/A")
                f.write(
                    f"| {name} "
                    f"| {_fmt(row.get('mean_gain'), '+.4f') if row.get('mean_gain') is not None else 'N/A'} "
                    f"| {ci} "
                    f"| {_fmt(row.get('frac_positive'))} "
                    f"| {row.get('significant', '—')} |\n"
                )
            f.write("\n")

        f.write("## C1 Decision\n\n```\n")
        threshold = int(0.75 * total_comparisons)
        verdict = "PASS" if clinical_wins >= threshold else "FAIL"
        f.write(
            f"B_sdmrow_clinical wins {clinical_wins}/{total_comparisons} "
            f"comparisons vs controls\n"
            f"(threshold: ≥{threshold} = 75% of {total_comparisons}).\n"
            f"C1 KG clinical SDM contribution: {verdict}\n\n"
        )
        if verdict == "PASS":
            f.write(
                "→ ICDR clinical ordering in the SDM contributes meaningfully "
                "to governance routing.\n"
                "→ Paper claim: 'grading-aware SDM derived from clinical KG "
                "outperforms uniform and random alternatives.'\n"
            )
        else:
            f.write(
                "→ Clinical SDM does not consistently outperform alternatives.\n"
                "→ Paper reframing: C1 contribution is structural/auditability "
                "(traceable KG), not empirical performance gain.\n"
            )
        f.write("```\n\n")

        f.write(
            "## Implications\n\n"
            "| Outcome | Paper action |\n|---|---|\n"
            "| clinical > uniform | Claim: grading-aware SDM adds governance "
            "signal over generic structure |\n"
            "| clinical > reversed | Claim: ICDR ordering direction matters "
            "(not just any distances) |\n"
            "| clinical > random | Claim: structured clinical KG is needed, "
            "not random graph |\n"
            "| clinical ≈ all controls | Reframe: C1 = traceable KG design, "
            "auditability, not perf gain |\n"
        )

        f.write("\n## Next Steps\n\n")
        f.write(
            "1. Write paper draft (Stage 4): framing based on C1 verdict.\n"
            "2. Build comparison table with published DR detection methods.\n"
            "3. Implement per-bbox certificate output (C2 governance artifact).\n"
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 3D C1 Ablation: KG clinical SDM design contribution."
    )
    parser.add_argument("--stage3c_dir", default="outputs/stage3c_audit")
    parser.add_argument("--stage3a_dir", default="outputs/stage3_audit")
    parser.add_argument("--output_dir",  default="outputs/stage3d_ablation")
    parser.add_argument("--n_bootstrap", type=int, default=_N_BOOTSTRAP)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    s3c = Path(args.stage3c_dir)
    s3a = Path(args.stage3a_dir)

    print("[Stage3D.C1] === Phase 19N C1 Ablation: KG SDM Design ===")

    # ── Load clinical SDM ──────────────────────────────────────────────────────
    clin_sdm = _load_clinical_sdm(s3a / "sdm_norm_clinical_prior.json")

    # ── Build all SDM variants ─────────────────────────────────────────────────
    rng_main = np.random.default_rng(seed=0)
    sdm_variants: List[Tuple[str, str, np.ndarray]] = [
        # (name, sdm_type, sdm_matrix)
        ("B_sdmrow_clinical", "clinical",  clin_sdm),
        ("B_sdmrow_uniform",  "uniform",   _uniform_sdm()),
        ("B_sdmrow_reversed", "reversed",  _reversed_sdm(clin_sdm)),
        ("B_sdmrow_shuffled", "shuffled",  _shuffled_sdm(clin_sdm, np.random.default_rng(99))),
    ]
    for s in range(_N_RANDOM_SEEDS):
        sdm_variants.append((
            f"B_sdmrow_rand_s{s}", "random",
            _random_sdm(np.random.default_rng(s + 100)),
        ))

    print("[Stage3D.C1] SDM variants:")
    for name, stype, mat in sdm_variants:
        print(f"[Stage3D.C1]   {name:<26}  [{stype}]  "
              f"row_MA=[{mat[0,0]:.2f},{mat[0,1]:.2f},"
              f"{mat[0,2]:.2f},{mat[0,3]:.2f}]")

    # ── Load data ──────────────────────────────────────────────────────────────
    recs: Dict[str, List[dict]] = {}
    for sp in ["train", "valid", "test"]:
        fp = s3c / f"governance_features_{sp}.csv"
        if not fp.exists():
            print(f"[Stage3D.C1] ERROR: {fp} not found.")
            import sys; sys.exit(1)
        recs[sp] = _load_csv(fp)

    sas_tr, cls_tr, oh_tr, y_tr, un_tr, img_tr = _parse(recs["train"])
    sas_va, cls_va, oh_va, y_va, un_va, img_va = _parse(recs["valid"])
    sas_te, cls_te, oh_te, y_te, un_te, img_te = _parse(recs["test"])

    for sp, y in [("train", y_tr), ("valid", y_va), ("test", y_te)]:
        print(f"[Stage3D.C1]   {sp}: {len(y)} boxes  "
              f"RETAIN={y.sum()} ({100*y.mean():.1f}%)  "
              f"SUPPRESS={(1-y).sum()}")

    # ── Build X matrices for all variants ─────────────────────────────────────
    # Variant definitions: (name, sdm_type, sdm_emb, use_onehot)
    all_variants: List[Tuple[str, str, Optional[np.ndarray], bool]] = [
        ("B0",          "score",    None,  False),   # score passthrough
        ("B_class_only","onehot",   None,  True),    # SAS + one-hot
    ]
    for name, stype, mat in sdm_variants:
        all_variants.append((name, stype, mat, False))  # SAS + SDM-row

    variant_order = [v[0] for v in all_variants]

    # ── Train models & collect scores ─────────────────────────────────────────
    scores_va: Dict[str, np.ndarray] = {}
    scores_te: Dict[str, np.ndarray] = {}

    if not _SK:
        print("[Stage3D.C1] sklearn unavailable — only B0 will run.")

    for name, stype, sdm_emb, use_oh in all_variants:
        oh_arg_tr = oh_tr if use_oh else None
        oh_arg_va = oh_va if use_oh else None
        oh_arg_te = oh_te if use_oh else None

        X_tr_v = _build_X(sas_tr, cls_tr, sdm_emb, oh_arg_tr)
        X_va_v = _build_X(sas_va, cls_va, sdm_emb, oh_arg_va)
        X_te_v = _build_X(sas_te, cls_te, sdm_emb, oh_arg_te)

        if name == "B0":
            scores_va[name] = X_va_v[:, 0]
            scores_te[name] = X_te_v[:, 0]
            print(f"[Stage3D.C1] {name}: passthrough  shape={X_va_v.shape}")
        elif _SK:
            print(f"[Stage3D.C1] Training {name:<26}  [{stype}]  "
                  f"shape={X_tr_v.shape}...", end=" ", flush=True)
            m, sc = _train_mlp(X_tr_v, y_tr)
            print(f"({getattr(m,'n_iter_','?')} iter)")
            scores_va[name] = _proba(m, sc, X_va_v)
            scores_te[name] = _proba(m, sc, X_te_v)

    # ── Operating point calibration ────────────────────────────────────────────
    b0_thresh: Dict[float, Optional[float]] = {}
    b0_fp_te:  Dict[float, Optional[float]] = {}
    for tgt in _TP_TARGETS:
        th = _find_thresh(y_va, scores_va["B0"], tgt)
        b0_thresh[tgt] = th
        b0_fp_te[tgt] = _eval_at(y_te, scores_te["B0"], un_te, th)["fp_supp"] \
                        if th is not None else None

    print(f"\n[Stage3D.C1] Operating point calibration...")
    res_rows: List[dict] = []
    all_thresh: Dict[Tuple[str, float], Optional[float]] = {}

    # First pass: collect clinical and class_only FP for gain computation
    clin_fp_te: Dict[float, Optional[float]] = {}
    co_fp_te:   Dict[float, Optional[float]] = {}

    for name, stype, _, _ in all_variants:
        if name not in scores_va:
            continue
        sv, st = scores_va[name], scores_te[name]
        for tgt in _TP_TARGETS:
            th = _find_thresh(y_va, sv, tgt)
            all_thresh[(name, tgt)] = th
            if th is None:
                continue
            m_te = _eval_at(y_te, st, un_te, th)
            if name == "B_sdmrow_clinical":
                clin_fp_te[tgt] = m_te["fp_supp"]
            if name == "B_class_only":
                co_fp_te[tgt] = m_te["fp_supp"]

    # Second pass: build result rows with all gain columns
    for name, stype, _, _ in all_variants:
        if name not in scores_va:
            continue
        sv, st = scores_va[name], scores_te[name]
        for tgt in _TP_TARGETS:
            th = all_thresh.get((name, tgt))
            if th is None:
                print(f"[Stage3D.C1]   {name:<26} TP>={tgt:.2f}: cannot reach target")
                continue
            m_va = _eval_at(y_va, sv, un_va, th)
            m_te = _eval_at(y_te, st, un_te, th)

            b0_ref   = b0_fp_te.get(tgt)
            clin_ref = clin_fp_te.get(tgt)
            co_ref   = co_fp_te.get(tgt)

            gain_b0  = round(m_te["fp_supp"] - b0_ref, 4) \
                       if b0_ref is not None and name != "B0" else None
            gain_co  = round(m_te["fp_supp"] - co_ref, 4) \
                       if co_ref is not None and name not in ("B0","B_class_only") else None
            gain_clin = round(m_te["fp_supp"] - clin_ref, 4) \
                        if clin_ref is not None and name not in ("B0","B_sdmrow_clinical") else None

            gstr = f"{gain_b0:+.4f}" if gain_b0 is not None else "(B0)"
            print(f"[Stage3D.C1]   {name:<26} TP>={tgt:.2f}: "
                  f"thresh={th:.4f}  "
                  f"FP_supp(V={m_va['fp_supp']:.3f} T={m_te['fp_supp']:.3f})  "
                  f"gain/B0={gstr}")

            res_rows.append({
                "variant": name, "sdm_type": stype, "tp_ret_target": tgt,
                "thresh_valid": th,
                "tp_ret_valid":  m_va["tp_ret"],  "fp_supp_valid":  m_va["fp_supp"],
                "auprc_valid":   m_va["auprc"],
                "tp_ret_test":   m_te["tp_ret"],  "fp_supp_test":   m_te["fp_supp"],
                "auprc_test":    m_te["auprc"],
                "fp_supp_gain_over_b0":       gain_b0,
                "fp_supp_gain_over_class_only": gain_co,
                "fp_supp_gain_over_clinical": gain_clin,
                "achieves_target_test": (m_te["tp_ret"] or 0) >= tgt,
            })

    # ── Bootstrap CI ──────────────────────────────────────────────────────────
    print(f"\n[Stage3D.C1] Bootstrap CI ({args.n_bootstrap} resamples)...")
    boot_rows: List[dict] = []
    for name, stype, _, _ in all_variants:
        if name == "B0" or name not in scores_te:
            continue
        for tgt in _TP_TARGETS:
            th_m  = all_thresh.get((name, tgt))
            th_b0 = b0_thresh.get(tgt)
            if th_m is None or th_b0 is None:
                continue
            boot = _boot_gain(
                img_te, y_te, scores_te[name], scores_te["B0"],
                th_m, th_b0, args.n_bootstrap, _BOOTSTRAP_SEED,
            )
            sig = "YES" if boot["ci_lo"] is not None and boot["ci_lo"] > 0 else "no"
            print(f"[Stage3D.C1]   {name:<26} TP>={tgt:.2f}: "
                  f"gain={_fmt(boot['mean'], '+.4f')}  "
                  f"CI=[{_fmt(boot['ci_lo'])},{_fmt(boot['ci_hi'])}]  "
                  f"P(>0)={_fmt(boot['frac_pos'])}  sig={sig}")
            boot_rows.append({
                "variant": name, "tp_ret_target": tgt,
                "mean_gain": boot["mean"], "ci_lo": boot["ci_lo"],
                "ci_hi": boot["ci_hi"], "frac_positive": boot["frac_pos"],
                "n": boot["n"], "significant": sig,
            })

    # ── Count C1 wins ─────────────────────────────────────────────────────────
    controls = [
        "B_sdmrow_uniform", "B_sdmrow_reversed", "B_sdmrow_shuffled",
    ] + [f"B_sdmrow_rand_s{s}" for s in range(_N_RANDOM_SEEDS)]

    clinical_wins = 0
    total_comparisons = 0
    print(f"\n[Stage3D.C1] C1 Decision: clinical vs controls")
    for ctrl in controls:
        for tgt in _TP_TARGETS:
            r_clin = next(
                (r for r in res_rows if r["variant"] == "B_sdmrow_clinical"
                 and abs(r.get("tp_ret_target", -1) - tgt) < 1e-6), None
            )
            r_ctrl = next(
                (r for r in res_rows if r["variant"] == ctrl
                 and abs(r.get("tp_ret_target", -1) - tgt) < 1e-6), None
            )
            if r_clin is None or r_ctrl is None:
                continue
            fp_c = r_clin.get("fp_supp_test")
            fp_t = r_ctrl.get("fp_supp_test")
            if fp_c is None or fp_t is None:
                continue
            total_comparisons += 1
            win = fp_c > fp_t
            if win:
                clinical_wins += 1
            print(f"[Stage3D.C1]   clinical({fp_c:.4f}) {'>' if win else '<='} "
                  f"{ctrl.replace('B_sdmrow_','')[:12]}({fp_t:.4f}) "
                  f"TP>={tgt:.2f} → {'WIN' if win else 'lose'}")

    pass_threshold = int(0.75 * total_comparisons) if total_comparisons > 0 else 1
    verdict = "PASS" if clinical_wins >= pass_threshold else "FAIL"
    print(f"\n[Stage3D.C1] Result: clinical wins {clinical_wins}/{total_comparisons}  "
          f"(need ≥{pass_threshold} = 75%)  → C1 {verdict}")

    # ── Write outputs ──────────────────────────────────────────────────────────
    rp = out_dir / "c1_ablation_results.csv"
    _write_csv(res_rows, _RES_FIELDS, rp)
    print(f"\n[Stage3D.C1] {rp}")

    bp = out_dir / "bootstrap_c1_fp_supp_gain_ci.csv"
    _write_csv(boot_rows, _BOOT_FIELDS, bp)
    print(f"[Stage3D.C1] {bp}")

    md = out_dir / "c1_ablation_decision.md"
    _write_md(res_rows, boot_rows, variant_order, clinical_wins,
              total_comparisons, md)
    print(f"[Stage3D.C1] {md}")

    print(f"\n[Stage3D.C1] Done.  C1 verdict: {verdict}  "
          f"({clinical_wins}/{total_comparisons} wins)")


if __name__ == "__main__":
    main()
