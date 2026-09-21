#!/usr/bin/env python
"""fig7_detector_agnostic.py — two-panel detector-agnostic analysis (CEG-Net).

The same clinical evidence gate is placed on top of seven frozen detectors. Two
panels make the detector-agnostic claim concrete:

  (a) Forest plot of the threshold-free AUPRC gain (gate - M0) with 95%
      image-clustered bootstrap CI. The gain is positive on every detector and
      its CI excludes zero on five of seven.

  (b) Absolute AUPRC of the detector score (M0) and of the gate, drawn as a
      dumbbell per detector and sorted by M0. The base detectors span a wide
      quality range (AUPRC 0.39 to 0.82), and the gate lifts the area across the
      whole range, which is the substance of the detector-agnostic claim. The
      lift does not require the detector to expose real per-ROI class
      probabilities: it persists on the synthetic-proxy detectors (FCOS, YOLO).

Reads per-detector benchmark JSONs from tools/eval_fp_reduction_benchmark.py
(governance_retain task): each carries results[] (M0_score, KG-gate-MLP AUPRC) and
bootstrap_auprc.kg_minus_m0 (gain point + CI + significance).

Command, run from the repository root:
  python journal/figures/scripts/fig7_detector_agnostic.py

  (override inputs with one or more --glob and --out)

Output: journal/figures/fig7_detector_agnostic.{png,pdf}
"""
from __future__ import annotations

import argparse
import glob as _glob
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import C, apply_style, fmt, panel_tags_bottom, save  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GLOBS = [
    str(ROOT / "outputs/0506/detcmp_e2/*_fp_benchmark.json"),
    str(ROOT / "outputs/0506/yolo_e2/*_fp_benchmark.json"),
]
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "fig7_detector_agnostic.png"

# synthetic-proxy detectors (no real per-ROI class distribution)
_SYNTH = {"fcos", "yolov8s", "yolo11s", "rtdetr-l"}
_LABEL = {
    "fasterrcnn": "Faster R-CNN", "fasterrcnn_v2": "Faster R-CNN v2",
    "retinanet": "RetinaNet", "retinanet_v2": "RetinaNet v2", "fcos": "FCOS",
    "yolov8s": "YOLOv8-s", "yolo11s": "YOLO11-s", "rtdetr-l": "RT-DETR-L",
    # rtdetr_r18 (outputs/0506/rtdetr_e2) BUKAN rtdetr-l. Yang ini fine-tune HF
    # PekingU/rtdetr_r18vd dengan real_prob_coverage = 1.0, jadi TIDAK masuk _SYNTH.
    "rtdetr_r18": "RT-DETR-R18",
}


def load_rows(globs):
    rows = []
    for g in globs:
        for path in sorted(_glob.glob(g)):
            p = json.loads(Path(path).read_text(encoding="utf-8"))
            bsa = (p.get("bootstrap_auprc") or {}).get("kg_minus_m0") or {}
            gain, ci = bsa.get("gain_point"), bsa.get("gain_ci95")
            if gain is None or ci is None:
                continue
            m0 = kg = None
            for r in p.get("results", []):
                if r["method"] == "M0_score":
                    m0 = r["auprc"]
                if "KG-gate-MLP" in r["method"]:
                    kg = r["auprc"]
            tag = Path(path).stem.replace("_fp_benchmark", "")
            rows.append({
                "name": _LABEL.get(tag, tag), "tag": tag,
                "gain": gain, "lo": ci[0], "hi": ci[1],
                "sig": bool(bsa.get("sig")), "m0": m0, "kg": kg,
                "synth": tag in _SYNTH,
            })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", action="append", default=[])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    globs = args.glob or DEFAULT_GLOBS

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load_rows(globs)
    if not rows:
        print("[fig7] no rows."); return 1

    apply_style()
    fig, (axf, axd) = plt.subplots(1, 2, figsize=(7.3, 3.1),
                                   gridspec_kw={"width_ratios": [1.0, 1.15]})

    # ── Panel (a): forest plot of gain ───────────────────────────────────────
    rf = sorted(rows, key=lambda r: r["gain"])  # ascending, bottom-up
    ys = list(range(len(rf)))
    for y, r in zip(ys, rf):
        col = C["b4"] if r["sig"] else C["neutral"]
        axf.plot([r["lo"], r["hi"]], [y, y], color=col, lw=2.2, solid_capstyle="round")
        axf.plot([r["gain"]], [y], "o", color=col, ms=4.5,
                 markeredgecolor="white", markeredgewidth=0.5)
        axf.text(r["hi"], y, "  $+$" + fmt(r["gain"]) + ("*" if r["sig"] else ""),
                 va="center", ha="left", fontsize=7.2)
    axf.axvline(0.0, color="black", lw=0.8, ls="--", alpha=0.6)
    axf.set_yticks(ys); axf.set_yticklabels([r["name"] for r in rf])
    axf.set_ylim(-0.6, len(rf) - 0.4)
    lo_min = min(r["lo"] for r in rf); hi_max = max(r["hi"] for r in rf)
    span = max(hi_max - lo_min, 1e-3)
    axf.set_xlim(lo_min - 0.05 * span, hi_max + 0.32 * span)
    axf.set_xlabel(r"$\Delta$AUPRC (gate $-$ $M_0$)")
    n_sig = sum(r["sig"] for r in rf)
    axf.grid(axis="x"); axf.grid(axis="y", visible=False)

    # ── Panel (b): absolute AUPRC dumbbell ───────────────────────────────────
    rd = rf  # urutan SAMA dengan panel (a) agar satu detektor mudah ditelusuri
    ysd = list(range(len(rd)))
    for y, r in zip(ysd, rd):
        axd.plot([r["m0"], r["kg"]], [y, y], color="#CCCCCC", lw=1.6, zorder=1)
        axd.plot([r["m0"]], [y], "o", color=C["m0"], ms=4.5, zorder=2,
                 markeredgecolor="white", markeredgewidth=0.5)
        gcol = C["b4"] if r["sig"] else C["neutral"]
        axd.plot([r["kg"]], [y], "D", color=gcol, ms=4.5, zorder=3,
                 markeredgecolor="white", markeredgewidth=0.5)
        # Jarak label 0.008 satuan-data hanya ~3 pt pada rentang sumbu ini, lebih
        # sempit daripada penanda 4.5 pt itu sendiri, sehingga angka menempel pada
        # penanda saat M0 dan gate berdekatan (YOLO11-s: 0.822 vs 0.828).
        axd.text(r["kg"] + 0.017, y, fmt(r["kg"]), va="center", ha="left",
                 fontsize=7.2)
        axd.text(r["m0"] - 0.017, y, fmt(r["m0"]), va="center", ha="right",
                 fontsize=7.2, color=C["m0"])
        if r["synth"]:
            axd.text(-0.012, y, "†", va="center", ha="right", fontsize=8,
                     color=C["b6"], transform=axd.get_yaxis_transform())
    axd.set_yticks(ysd)
    axd.set_yticklabels([])
    axd.set_ylim(-0.6, len(rd) - 0.4)
    axd.set_xlim(0.30, 0.93)
    axd.set_xlabel("AUPRC (retain vs suppress)")
    axd.grid(axis="x"); axd.grid(axis="y", visible=False)
    # legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C["m0"],
               ms=5, label="$M_0$ (detector score)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=C["b4"],
               ms=5, label="gate (significant)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=C["neutral"],
               ms=5, label="gate (n.s.)"),
    ]
    axd.legend(handles=handles, loc="lower right", handlelength=1.0,
               borderpad=0.25, labelspacing=0.3)

    # Penanda panel di bawah, sejajar satu garis; dipasang setelah tata letak final.
    fig.subplots_adjust(wspace=0.10, bottom=0.24)
    panel_tags_bottom(fig, [axf, axd], ["(a)", "(b)"], y=0.015)
    # Arti * dan † dijelaskan di caption, tidak di dalam gambar.
    save(fig, "fig7_detector_agnostic")
    print(f"[fig7] {len(rows)} detectors, {n_sig} sig")
    for r in sorted(rows, key=lambda x: -x["gain"]):
        print(f"[fig7]   {r['name']:<18} M0={r['m0']:.3f} gate={r['kg']:.3f} "
              f"d={r['gain']:+.3f} {'sig' if r['sig'] else 'ns'} "
              f"{'(synth)' if r['synth'] else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
