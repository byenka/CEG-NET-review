#!/usr/bin/env python
"""Figure 3 — sebaran target rujukan pada empat tier Algorithm 1 (OIA-DDR test).

DITULIS ULANG 2026-09-15. Versi lama membaca tier_breakdown.csv dari stage3d4_ab/new/.
Tier di sana adalah tier_target, peta deterministik dari kategori yang diturunkan dari
GROUND TRUTH (tools/audit_stage3c_governance_targets.py). Figur lama dengan demikian
menampilkan kunci jawaban, bukan keluaran sistem, dan "M0" di sana adalah ambang SAS yang
jenuh pada 0.999, bukan ambang skor detektor.

Versi ini membaca outputs/0506/tier_routing/tier_routing_summary.json yang dibangkitkan
tools/make_tier_routing.py. Tier = keputusan retain pada tau validasi x SDCA (delta 0.30),
tanpa label. Target RETAIN/SUPPRESS dari anotasi hanya dipakai untuk menghitung sebaran:
  (a) deteksi bertarget RETAIN   : fraksi yang dirutekan ke R1..R4
  (b) deteksi bertarget SUPPRESS : fraksi yang dirutekan ke R1..R4
"""
import json
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import C, apply_style, panel_tags_bottom, save  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "outputs/0506/tier_routing/tier_routing_summary.json"
TIERS = ["R1", "R2", "R3", "R4"]
TICK = ["R1\naccepted", "R2\nprovisional", "R3\ncontested", "R4\nrejected"]


def f2(v: float) -> str:
    """Pembulatan setengah-ke-atas dua desimal, konvensi yang sama dengan tabel."""
    return str(Decimal(repr(float(v))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def main():
    s = json.loads(SRC.read_text(encoding="utf-8"))["cohorts"]["oiaddr"]
    series = {
        key: ([s[key]["retain_targets_by_tier"][t] for t in TIERS],
              [s[key]["suppress_targets_by_tier"][t] for t in TIERS])
        # "fused" ditambahkan 2026-09-16: figur ini duduk TEPAT DI ATAS tabel tier pada halaman
        # yang sama, dan tabel itu kini memuat tiga sistem. Dua artefak bersebelahan dengan
        # cakupan berbeda adalah jenis ketidakcocokan yang dicari reviewer.
        for key in ("m0", "fused", "gate")
    }
    if "fused" not in s:
        raise SystemExit("[fig3] blok 'fused' tidak ada di tier_routing_summary.json; "
                         "jalankan ulang python tools/make_tier_routing.py")

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    x = np.arange(4)
    # Tiga batang, jadi lebarnya turun dari 0.36 dan posisinya simetris terhadap pusat tick.
    w = 0.26
    # Peran tetap: M0 abu-abu, gate vermilion, fusi hijau, sama dengan fig6, fig8 dan fig10.
    bars = (("m0", "$M_0$ (score)", C["m0"]),
            ("fused", "score fusion", C["accent"]),
            ("gate", "CEG gate", C["b6"]))
    offs = (-w, 0.0, w)
    ylabels = ["fraction of RETAIN targets", "fraction of SUPPRESS targets"]

    for k, ax in enumerate(axes):
        vals = [series[key][k] for key, _lab, _col in bars]
        for (key, lab, col), off, v in zip(bars, offs, vals):
            ax.bar(x + off, v, w, label=lab, color=col, edgecolor="none")
        ax.set_xticks(x)
        # Ukuran lebih kecil dari label sumbu: "provisional" dan "contested" pada ukuran
        # bawaan saling menempel di antara dua tick yang bersebelahan.
        ax.set_xticklabels(TICK, fontsize=7.4)
        ax.set_ylabel(ylabels[k])
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)
        top = max(sum(vals, []))
        # 12 label per panel, bukan 8, jadi ukurannya diturunkan supaya tidak saling menempel.
        for (key, lab, col), off, v in zip(bars, offs, vals):
            for xi, vi in zip(x + off, v):
                ax.text(xi, vi + 0.015 * top, f2(vi), ha="center", fontsize=6.2, color=col)
        ax.set_ylim(0, top * 1.16)

    # SATU legenda bersama di atas, bukan dua legenda identik di dalam sumbu.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, handlelength=1.0,
               columnspacing=1.2, borderpad=0.2, bbox_to_anchor=(0.5, 1.0))

    fig.subplots_adjust(wspace=0.28, top=0.86, bottom=0.29)
    panel_tags_bottom(fig, list(axes), ["(a)", "(b)"], y=0.012)
    save(fig, "fig3_tier_distribution")


if __name__ == "__main__":
    main()
