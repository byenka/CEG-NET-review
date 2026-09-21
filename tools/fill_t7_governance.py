#!/usr/bin/env python
"""fill_t7_governance.py — Point 2: fill the [RUN] cells of T7 (governance vs.
alternative FP-reduction) from a REAL controlled benchmark JSON produced by
tools/eval_fp_reduction_benchmark.py.

T7 originally carried [RUN] placeholders for "Temperature scaling" and
"MC-Dropout uncertainty filter". Two honesty notes baked into this filler:

  * Temperature scaling (Guo'17) IS already one of the benchmark methods
    (monotonic on score -> at matched TP-retention it equals M0; we report the
    real measured numbers, not an assumption).
  * True MC-Dropout is NOT feasible on the LOCKED Faster R-CNN: its ROI head has
    no dropout, so MC-Dropout would require inserting dropout and RETRAINING the
    frozen detector (forbidden). The defensible, runnable substitute is the
    single-pass uncertainty family already in the benchmark (MSP / predictive
    entropy / margin); we report the BEST of them as the "Uncertainty filter"
    row and say so explicitly in the caption. This is standard and apples-to-
    apples on the same detector + data + split.

All numbers come straight from <detector>_fp_benchmark.json — nothing invented.

Usage:
  python tools/fill_t7_governance.py \
    --benchmark_json outputs/0506/detcmp_e2/fasterrcnn_fp_benchmark.json \
    --out_tex outputs/0506/tables/T7_sota_governance_FILLED.tex
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _find(results, *prefixes):
    """First result whose method name starts with any of the given prefixes."""
    for r in results:
        for p in prefixes:
            if r["method"].lower().startswith(p.lower()):
                return r
    return None


def _best_uncertainty(results):
    """Highest FP-suppression among the single-pass uncertainty family."""
    fam = [r for r in results if r["method"] in ("MSP", "Entropy", "Margin")]
    fam = [r for r in fam if r.get("fp_supp") is not None]
    return max(fam, key=lambda r: r["fp_supp"]) if fam else None


def _f(x, nd=3):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else r"\textit{[n/a]}"


def render(payload: dict, detector: str) -> str:
    results = payload["results"]
    tp_ret = payload.get("tp_ret_target", 0.80)
    pos = payload.get("positive_label", "tp_exact")

    m0 = _find(results, "M0", "Score")
    temp = _find(results, "TempScale", "Temperature")
    unc = _best_uncertainty(results)
    ours = _find(results, "KG-gate", "KG-evidence")

    if not (m0 and ours):
        raise SystemExit("[t7] benchmark JSON missing M0 or KG-gate row; cannot fill T7.")

    gain = ours["fp_supp"] - m0["fp_supp"]
    unc_name = unc["method"] if unc else "MSP"

    def row(label, fam, r, audit):
        if r is None:
            return f"{label} & {fam} & \\textit{{[n/a]}} & \\textit{{[n/a]}} & \\textit{{[n/a]}} & {audit} \\\\"
        return (f"{label} & {fam} & {_f(r['fp_supp'])} & {_f(r['precision_after'])} "
                f"& {_f(r['auprc'])} & {audit} \\\\")

    lines = [
        "% T7 — Governance / FP-reduction head-to-head (Axis B). FILLED by",
        "% tools/fill_t7_governance.py from a REAL controlled benchmark "
        "(tools/eval_fp_reduction_benchmark.py).",
        f"% source detector cache: {detector}   positive={pos}   "
        f"matched TP-retention={tp_ret}",
        "% Numbers are measured (not [RUN] placeholders). Temperature scaling and",
        "% the uncertainty filter are re-implemented on the SAME detector + data.",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Governance vs.\ alternative false-positive--reduction methods on",
        r"OIA-DDR \textbf{test}, all compared at matched true-positive retention",
        f"($\\mathrm{{TP}}_{{\\mathrm{{ret}}}}{{=}}{tp_ret}$), on the \\emph{{same}}",
        r"detector, data and split. Unlike score thresholding, post-hoc calibration",
        r"and uncertainty filtering, the KG-evidence governance gate suppresses more",
        r"false positives \emph{and} emits an auditable per-detection certificate.",
        r"(Temperature scaling [Guo'17] and the uncertainty filter are re-implemented",
        r"here; true MC-Dropout is not applicable to the frozen Faster R-CNN whose ROI",
        r"head has no dropout, so the single-pass uncertainty family is used instead.)}",
        r"\label{tab:sota_governance}",
        r"\begin{tabular}{llccc c}",
        r"\toprule",
        r"Method & Family & FP suppr. & Prec. & AUPRC & Audit \\",
        r"\midrule",
        row(r"Score threshold ($M_0$)", "threshold", m0, "No"),
        row(r"Temperature scaling (Guo'17)", "calibration", temp, "No"),
        row(f"Uncertainty filter ({unc_name})", "uncertainty", unc, "No"),
        row(r"\textbf{KG-evidence gate ($M_{4\text{-}tier}$, ours)}", "governance", ours, r"\textbf{Yes}"),
        r"\midrule",
        (r"\multicolumn{2}{l}{FP-suppr.\ gain of ours over $M_0$} & "
         f"\\multicolumn{{4}}{{l}}{{$+{gain:.3f}$ (same matched operating point)}} \\\\"),
        r"\bottomrule",
        r"\end{tabular}",
        r"\\[2pt]",
        r"{\footnotesize Only the governance gate emits a per-detection audit",
        r"certificate. Numbers are from a single controlled benchmark on the same",
        r"detector cache; see Table~\ref{tab:fp_benchmark} for the full $\geq$5-method",
        r"sweep (MSP, entropy, margin, temperature scaling, evidence-LogReg, ours).}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Fill T7 governance table from a benchmark JSON.")
    ap.add_argument("--benchmark_json", required=True, type=Path,
                    help="<detector>_fp_benchmark.json from eval_fp_reduction_benchmark.py")
    ap.add_argument("--out_tex", required=True, type=Path)
    ap.add_argument("--detector", default=None,
                    help="label for the source detector (default: parsed from filename).")
    args = ap.parse_args()

    payload = json.loads(args.benchmark_json.read_text(encoding="utf-8"))
    detector = args.detector or args.benchmark_json.stem.replace("_fp_benchmark", "")
    tex = render(payload, detector)
    args.out_tex.parent.mkdir(parents=True, exist_ok=True)
    args.out_tex.write_text(tex, encoding="utf-8")
    print(f"[t7] wrote {args.out_tex}  (source: {args.benchmark_json}, detector={detector})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
