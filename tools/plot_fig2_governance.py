#!/usr/bin/env python
"""plot_fig2_governance.py — regenerate Figure 2 (governance) with the HONEST,
traceable numbers, replacing the retired +0.536 figure.

Reads the LOCKED B10 metrics:
  outputs/point123_2026-05-30_..._final_clean/stage3d_ab/new/lgg_test_metrics.json
and (optionally) the matched-TP FP-suppression gain + CI from
  outputs/0506/headline_fp_benchmark.json
then draws a threshold-free AUPRC comparison:
  M0 (score threshold) | Score-MLP | KG-evid only | LGov-Net (KG-evid + score)
annotating the KG contribution (LGov-Net - Score-MLP) and governance-over-M0.

Usage (server / anywhere matplotlib is available):
  python tools/plot_fig2_governance.py \
    --metrics outputs/point123_2026-05-30_point123_external_pss_stage3d_final_clean/stage3d_ab/new/lgg_test_metrics.json \
    --headline outputs/0506/headline_fp_benchmark.json \
    --out journal/figures/fig2_governance.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Regenerate honest Figure 2 (governance).")
    ap.add_argument("--metrics", required=True, type=Path,
                    help="lgg_test_metrics.json from the locked Stage3D run.")
    ap.add_argument("--headline", type=Path, default=None,
                    help="optional headline_fp_benchmark.json for matched-TP gain + CI.")
    ap.add_argument("--gov_auprc_ci", default="+0.026,+0.040",
                    help="locked 95%% bootstrap CI for the B4-B0 AUPRC gain "
                         "(from stage3d_new.log; default is the locked value).")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[fig2] ERROR: matplotlib unavailable ({exc}). pip install matplotlib")
        return 2

    from decimal import Decimal, ROUND_HALF_UP
    def r3(x):  # round-half-up to 3 dp (matches the locked Table 2 rounding, e.g. 0.6815 -> 0.682)
        return f"{Decimal(repr(float(x))).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)}"
    b = json.loads(args.metrics.read_text(encoding="utf-8"))["baselines"]
    def auc(k): return float(b[k]["auprc"])
    bars = [
        ("$M_0$\n(score thr.)",     auc("B0_threshold"),  "#9e9e9e"),
        ("Score-MLP\n(score only)", auc("B_score_MLP"),   "#9e9e9e"),
        ("KG-evid.\nonly",          auc("B3_LGG_TORCH"),  "#bdbdbd"),
        ("CEG-Net\n(KG+score)",     auc("B4_MLP"),        "#2e7d32"),
    ]
    labels = [x[0] for x in bars]; vals = [x[1] for x in bars]; cols = [x[2] for x in bars]

    kg_contrib = auc("B4_MLP") - auc("B_score_MLP")   # isolated KG contribution
    gov_over_m0 = auc("B4_MLP") - auc("B0_threshold")  # governance vs score thr.

    fp_txt = ""
    if args.headline and args.headline.exists():
        hp = json.loads(args.headline.read_text(encoding="utf-8"))
        bs = hp.get("bootstrap") or {}
        if bs.get("gain_ci95"):
            ci = bs["gain_ci95"]
            fp_txt = (f"matched-TP FP-suppression gain = {bs.get('gain_point', 0):+.3f} "
                      f"(95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}])")

    fig, axfig = plt.subplots(figsize=(6.2, 4.2))
    xs = range(len(bars))
    axfig.bar(xs, vals, color=cols, edgecolor="black", linewidth=0.6, width=0.62)
    for i, v in enumerate(vals):
        axfig.text(i, v + 0.006, r3(v), ha="center", va="bottom", fontsize=10,
                   fontweight=("bold" if i == len(vals) - 1 else "normal"))
    axfig.set_ylabel("AUPRC (threshold-free)")
    axfig.set_xticks(list(xs)); axfig.set_xticklabels(labels, fontsize=9)
    axfig.set_ylim(0, max(vals) * 1.25)
    axfig.set_title("Clinical evidence gate: KG evidence is complementary to detector score",
                    fontsize=10)
    ci_lo, ci_hi = (s.strip() for s in args.gov_auprc_ci.split(","))
    cap = (f"KG contribution (CEG-Net $-$ Score-MLP) = {kg_contrib:+.3f} AUPRC   |   "
           f"vs $M_0$ = {gov_over_m0:+.3f}  (95% CI [{ci_lo}, {ci_hi}], P>0=1.0)")
    if fp_txt:
        cap += "\n" + fp_txt
    axfig.text(0.5, -0.30, cap, transform=axfig.transAxes, ha="center", va="top",
               fontsize=8.5)
    fig.subplots_adjust(bottom=0.30)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"[fig2] wrote {args.out}")
    print(f"[fig2]   AUPRC: M0={auc('B0_threshold'):.3f} ScoreMLP={auc('B_score_MLP'):.3f} "
          f"KGonly={auc('B3_LGG_TORCH'):.3f} LGovNet={auc('B4_MLP'):.3f}")
    print(f"[fig2]   KG contribution={kg_contrib:+.3f}  governance-over-M0={gov_over_m0:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
