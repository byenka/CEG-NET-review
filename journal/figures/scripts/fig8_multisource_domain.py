#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""fig8 — clinical evidence gating lintas empat domain.

DIBANGUN ULANG 2026-09-14. Versi sebelumnya menampilkan tiga batang untuk tiga angka,
memakai palet bawaan matplotlib, memasang judul di dalam gambar yang mengulang caption,
dan TIDAK menampilkan M0 sama sekali sehingga pembaca tak bisa menilai besar-kecilnya
gain. Isinya juga menduplikasi kolom Delta pada Tabel 12. Selain itu angkanya berasal
dari konfigurasi PSS placeholder yang sudah digantikan.

Versi ini menambah informasi, bukan mengulang tabel: menampilkan level absolut ketiga
sistem, margin gate penuh terhadap skor detektor, dan kontribusi multi-source, semuanya
dengan titik per-seed sehingga pembaca melihat sebaran, bukan hanya batang error.

Sumber: outputs/0506/evsds_pss/domain_shift_5seed.json
        (dihasilkan oleh journal/figures/scripts/_figdata_domain_shift.py)
Jalankan dari root repo:
    python journal/figures/scripts/fig8_multisource_domain.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import blended_transform_factory

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import C, apply_style, panel_tags_bottom, save  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "outputs/0506/evsds_pss/domain_shift_5seed.json"
# Baseline fusi skor per kohort. Ditambahkan 2026-09-16: panel (a) tanpa batang ini
# memperlihatkan gate penuh sebagai yang tertinggi di tiap kohort, sementara Sec. 5.5 di
# bawahnya menyatakan fusi mengungguli gate di SETIAP kohort eksternal. Figur dan teks
# tidak boleh saling membantah.
FUS = ROOT / "_septem/scripts_optionB/fusion_external.json"


def jitter(n, w=0.055):
    return np.linspace(-w, w, n)


def gain_panel(ax, cohorts, key, color, ylabel):
    xs = np.arange(len(cohorts))
    means = [c[key]["mean"] for c in cohorts]
    # KONVENSI 2026-09-15: SD sampel (ddof=1) dari nilai per-seed, sama dengan Tabel 2, 3,
    # 12, S2, S3 dan S4. JSON menyimpan SD populasi (ddof=0).
    sds = [float(np.std(c[key]["per_seed"], ddof=1)) for c in cohorts]
    ax.bar(xs, means, yerr=sds, width=0.52, color=color, edgecolor="none",
           error_kw={"ecolor": "#444444", "elinewidth": 0.8, "capsize": 2.5,
                     "capthick": 0.8}, zorder=2)
    for x, c in zip(xs, cohorts):
        pts = c[key]["per_seed"]
        ax.scatter(x + jitter(len(pts)), pts, s=7, color="#222222", zorder=4,
                   linewidths=0)
    ax.axhline(0, color="#444444", lw=0.8, zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels([c["name"] for c in cohorts], rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y"); ax.grid(axis="x", visible=False)
    # Hitungan seed positif dipasang DI ATAS batang, bukan di dasar sumbu. Di dasar
    # sumbu label tertembus batang galat yang menjulur ke bawah nol (FGADR) dan
    # tertimpa titik per-seed (IDRiD). Ruang atas ditambah dulu agar label muat.
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.20 * (hi - lo))
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for x, c in zip(xs, cohorts):
        ax.text(x, 0.99, f"{c[key]['n_pos']}/5", ha="center", va="top",
                fontsize=6.6, color="#555555", transform=trans)


def main() -> int:
    if not SRC.is_file():
        print(f"[fig8] data belum ada: {SRC}\n"
              "       jalankan dulu: python journal/figures/scripts/_figdata_domain_shift.py")
        return 2
    d = json.loads(SRC.read_text(encoding="utf-8"))
    co = d["cohorts"]

    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.3, 2.7),
                             gridspec_kw={"width_ratios": [1.25, 1.0, 1.0]})

    # ---- (a) level absolut ----------------------------------------------
    ax = axes[0]
    xs = np.arange(len(co))
    # Empat batang, jadi lebarnya turun dari 0.26 dan posisinya simetris terhadap pusat.
    # Hijau dipakai untuk keluarga fusi skor, sama seperti fig6. Panel (c) juga hijau,
    # tetapi panel itu tidak berlegenda dan sumbunya sendiri yang menamainya, sehingga
    # tidak ada pembacaan ganda di dalam satu panel.
    fus = {c["name"]: c["fused_ap"] for c in json.loads(FUS.read_text(encoding="utf-8"))["cohorts"]}
    missing = [c["name"] for c in co if c["name"] not in fus]
    if missing:
        raise SystemExit(f"[fig8] kohort tanpa angka fusi: {missing}")
    w = 0.20
    ax.bar(xs - 1.5 * w, [c["m0_auprc"] for c in co], width=w, color=C["m0"],
           label="detector score ($M_0$)", edgecolor="none")
    ax.bar(xs - 0.5 * w, [fus[c["name"]] for c in co], width=w, color=C["accent"],
           label="score fusion", edgecolor="none")
    ax.bar(xs + 0.5 * w, [c["b4_auprc_mean"] for c in co], width=w, color=C["b4"],
           label="KG-gate (B4)", edgecolor="none")
    ax.bar(xs + 1.5 * w, [c["b6_auprc_mean"] for c in co], width=w, color=C["b6"],
           label="+ multi-source (B6)", edgecolor="none")
    ax.set_xticks(xs)
    ax.set_xticklabels([c["name"] for c in co], rotation=20, ha="right")
    ax.set_ylabel("AUPRC (retain vs suppress)")
    ax.set_ylim(0, 0.88)
    ax.grid(axis="y"); ax.grid(axis="x", visible=False)

    # ---- (b) margin gate penuh atas skor --------------------------------
    gain_panel(axes[1], co, "gain_b6_m0", C["b6"], "B6 $-$ $M_0$ (AUPRC)")

    # ---- (c) kontribusi multi-source ------------------------------------
    gain_panel(axes[2], co, "gain_b6_b4", C["accent"], "B6 $-$ B4 (AUPRC)")

    # SATU legenda bersama di atas tengah, satu baris. Ditambatkan pada sumbu (a)
    # legendanya menjulur asimetris ke kolom panel (b), dan dengan ncol=2 barisnya
    # tersisa satu entri sendirian.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, handlelength=1.0,
               columnspacing=1.0, borderpad=0.2, bbox_to_anchor=(0.5, 1.0),
               fontsize=6.9)

    # Penanda panel di bawah, sejajar satu garis; dipasang setelah tata letak final.
    fig.subplots_adjust(wspace=0.52, top=0.84, bottom=0.26)
    panel_tags_bottom(fig, axes, ["(a)", "(b)", "(c)"], y=0.015)
    save(fig, "fig8_multisource_domain")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
