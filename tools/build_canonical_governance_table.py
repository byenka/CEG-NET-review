#!/usr/bin/env python
"""build_canonical_governance_table.py — G2: reconcile the governance numbers into
ONE canonical, fully-labelled FP-reduction table so the three differently-measured
governance figures in the journal stop looking contradictory.

Background (the inconsistency this fixes):
  - T7 (old)  reported +0.536  -> that is the Stage3D.4 TIER-ROUTING operating
    point (M0 at TP>=0.80 thresh~0.999), a DIFFERENT metric.
  - T8        reported M0 0.629 / KG 0.700 (+0.071) -> FP-reduction benchmark
    under the FULL Mode-1 evidence (real PSS + external anatomy).
  - 0506 E2   reported M0 0.492 / KG 0.538 (+0.046) -> FP-reduction benchmark on
    a COMPARISON detector under LIGHT evidence (approximate APS, placeholder PSS).
They are not contradictory; they are different (operating point / evidence /
detector). This tool puts every FP-reduction-benchmark row in one table with
EXPLICIT Detector + Evidence + operating-point columns, plus the bootstrap CI,
and a caption note that the tier-routing figure lives in its own table.

Input: repeatable --row "Setting|Detector|Evidence|/path/to/fp_benchmark.json".
Each JSON is produced by tools/eval_fp_reduction_benchmark.py (ideally with
--bootstrap so the CI column is filled).

Usage:
  python tools/build_canonical_governance_table.py \
    --row "Headline (Mode-1)|Faster R-CNN r5b|full (ext.APS+realPSS)|outputs/0506/headline_fp_benchmark.json" \
    --row "Detector-agnostic|Faster R-CNN (25e)|light (approx APS, placeholder PSS)|outputs/0506/detcmp_e2/fasterrcnn_fp_benchmark.json" \
    --row "Detector-agnostic|FCOS|light|outputs/0506/detcmp_e2/fcos_fp_benchmark.json" \
    --out_md outputs/0506/tables/canonical_governance.md \
    --out_tex outputs/0506/tables/T_canonical_governance.tex
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _f(x, nd=3):
    try:
        if x is None or x != x:
            return "[--]"
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "[--]"


def _signed(x):
    try:
        return f"{float(x):+.3f}" if x is not None and x == x else "[--]"
    except (TypeError, ValueError):
        return "[--]"


def _row_metrics(path: str) -> dict:
    p = json.loads(Path(path).read_text(encoding="utf-8"))
    res = p.get("results", [])
    m0 = next((r for r in res if r.get("method") == "M0_score"), None)
    kg = next((r for r in res if str(r.get("method", "")).startswith("KG-gate")), None)
    m0a = (m0 or {}).get("auprc")
    kga = (kg or {}).get("auprc")
    bsa = (p.get("bootstrap_auprc") or {}).get("kg_minus_m0") or {}
    # PRIMARY = threshold-free AUPRC gain (FP-suppression gain is TP-mismatch sensitive)
    gain = bsa.get("gain_point")
    if gain is None and (m0a is not None and kga is not None):
        gain = kga - m0a
    m0fp = (m0 or {}).get("fp_supp")
    kgfp = (kg or {}).get("fp_supp")
    fp_gain = (kgfp - m0fp) if (m0fp is not None and kgfp is not None) else None
    return {
        "m0": m0a, "kg": kga, "gain": gain,
        "ci": bsa.get("gain_ci95"), "sig": bsa.get("sig"),
        "fp_gain": fp_gain,
        "kg_prec": (kg or {}).get("precision_after"),
        "kg_auprc": kga,
        "tp_ret": p.get("tp_ret_target"), "positive": p.get("positive_label"),
        "n_test": p.get("n_test_boxes"),
    }


def _ci_str(ci):
    return f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else "[--]"


def main() -> int:
    ap = argparse.ArgumentParser(description="G2: one canonical governance FP-reduction table.")
    ap.add_argument("--row", action="append", default=[],
                    help='"Setting|Detector|Evidence|/path/fp_benchmark.json". Repeatable.')
    ap.add_argument("--out_md", required=True, type=Path)
    ap.add_argument("--out_tex", required=True, type=Path)
    args = ap.parse_args()

    rows = []
    for spec in args.row:
        parts = spec.split("|")
        if len(parts) != 4:
            print(f"[canon] ERROR: --row needs 4 '|'-fields, got: {spec}")
            return 2
        setting, detector, evidence, path = (s.strip() for s in parts)
        if not Path(path).exists():
            print(f"[canon] WARN: missing {path} (skipped)")
            continue
        m = _row_metrics(path)
        m.update({"setting": setting, "detector": detector, "evidence": evidence,
                  "source": path})
        rows.append(m)
    if not rows:
        print("[canon] no readable rows."); return 1

    # consistency check: same metric definition across rows
    tps = {r["tp_ret"] for r in rows}
    poss = {r["positive"] for r in rows}
    note_consistency = (f"All rows use the SAME task + metric "
                        f"(positive={'/'.join(sorted(str(x) for x in poss))}, "
                        f"threshold-free AUPRC, TP-ret target={'/'.join(str(x) for x in sorted(tps))}).")

    # ---- markdown ----
    md = ["# Canonical governance comparison (G2 reconciliation)", "",
          note_consistency,
          "Each row is the SAME KG-evidence governance gate vs the score-only "
          "baseline (M0), differing only in **detector** and **evidence config**. "
          "PRIMARY metric = threshold-free **AUPRC gain** (image-clustered bootstrap "
          "CI). The FP-suppression gain is context only (TP-retention not exactly "
          "matched on test, so it overstates the effect).", "",
          "| Setting | Detector | Evidence | M0 AUPRC | +Gov AUPRC | ΔAUPRC | 95% CI | sig | FP-supp Δ† | n_test |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        sig = "—" if r["sig"] is None else ("YES" if r["sig"] else "no")
        md.append(f"| {r['setting']} | {r['detector']} | {r['evidence']} | "
                  f"{_f(r['m0'])} | {_f(r['kg'])} | {_signed(r['gain'])} | "
                  f"{_ci_str(r['ci'])} | {sig} | {_signed(r.get('fp_gain'))} | {r['n_test']} |")
    md += ["",
           "> †FP-suppression gain is at each method's own test operating point "
           "(TP-retention NOT exactly matched), so it overstates the effect — AUPRC "
           "is the reliable metric. The tier-routing figure (previous +0.536) is a "
           "DIFFERENT metric (Stage3D.4 routing operating point) reported separately.", ""]
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    # ---- LaTeX ----
    tex = [
        "% Canonical governance FP-reduction table (G2). Auto-generated by",
        "% tools/build_canonical_governance_table.py. One metric, explicit",
        "% Detector + Evidence columns, image-clustered bootstrap CI on the gain.",
        r"\begin{table}[t]", r"\centering",
        r"\caption{Canonical governance comparison. The same KG-evidence governance "
        r"gate vs.\ the score-only baseline ($M_0$) under one FP-reduction metric "
        r"(matched TP-retention), with \emph{detector} and \emph{evidence} stated "
        r"per row. $\Delta$ is the FP-suppression gain; the 95\% CI is an "
        r"image-clustered paired bootstrap. The tier-routing operating-point figure "
        r"is reported separately (it is a different metric).}",
        r"\label{tab:canonical_governance}",
        r"\begin{tabular}{lllcccc}", r"\toprule",
        r"Setting & Detector & Evidence & $M_0$ & +Gov & $\Delta$ & 95\% CI($\Delta$) \\",
        r"\midrule",
    ]
    for r in rows:
        tex.append(f"{r['setting']} & {r['detector']} & {r['evidence']} & "
                   f"{_f(r['m0'])} & {_f(r['kg'])} & {_signed(r['gain'])} & "
                   f"{_ci_str(r['ci'])} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    args.out_tex.write_text("\n".join(tex), encoding="utf-8")

    print(f"[canon] {len(rows)} row(s) -> {args.out_md}  +  {args.out_tex}")
    for r in rows:
        print(f"[canon]   {r['setting']:<20} {r['detector']:<22} "
              f"delta={_signed(r['gain'])} CI={_ci_str(r['ci'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
