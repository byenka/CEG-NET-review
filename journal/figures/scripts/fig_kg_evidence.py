#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""fig_kg_evidence — struktur KG dan penurunan evidence SDCA / APS.

DIRANCANG ULANG 2026-09-14. Versi sebelumnya bermasalah pada data maupun tampilan.

DATA: nilai Compat di-hardcode di dalam script dan salah satunya sudah melenceng
dari sumbernya (tertulis 0.433 untuk Hard_EX/Peripapillary, padahal
compat_scores.json menyebut 0.43484). Sekarang SELURUH angka dibaca dari berkas.

TAMPILAN: versi lama menabrakkan teks "vessel rupture" dengan "sev=0 (same prior)"
sampai tak terbaca, memasang kotak kuning berisi prosa bergaya salindia, menaruh
tabel kecil di dalam gambar, dan memakai pewarnaan merah/hijau untuk "HIGH"/"low".
Rancangan baru memberi satu tugas per panel:
  (a) struktur: dua kelompok severity, sisi dalam-grade vs lintas-grade, panah
      pathophysiological berarah, dan lima zona ETDRS yang memasok APS;
  (b) penurunan SDCA: satu deteksi, skor detektor SAMA, dua runner-up berbeda,
      biaya klinis berbeda. Inilah inti argumen paper;
  (c) matriks Compat 4x5 penuh, menggantikan empat batang pilihan.

Sumber:
  outputs/stage3_audit/sdm_norm_clinical_prior.json
  outputs/stage3_audit/compat_scores.json

Jalankan dari root repo:
    python journal/figures/scripts/fig_kg_evidence.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import C, apply_style, fmt, panel_tags_bottom, save  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
KG = ROOT / "outputs/stage3_audit"

LESIONS = ["MA", "HE", "Hard_EX", "Soft_EX"]
LESION_LBL = ["MA", "HE", "Hard EX", "Soft EX"]
ZONES = ["Foveal", "Parafoveal", "Peripapillary", "Vascular_Arcades", "Peripheral"]
ZONE_LBL = ["Foveal", "Para-\nfoveal", "Peri-\npapillary", "Vascular\narcades", "Peri-\npheral"]


def load_sdm():
    d = json.loads((KG / "sdm_norm_clinical_prior.json").read_text(encoding="utf-8"))
    out = {}
    for key, v in d.items():
        a, b = key.split("__")
        out[frozenset((a, b))] = v["norm"]
    return out


def load_compat():
    d = json.loads((KG / "compat_scores.json").read_text(encoding="utf-8"))
    return np.array([[d[f"{c}__{z}"] for z in ZONES] for c in LESIONS], dtype=float)


# ── panel (a): struktur ──────────────────────────────────────────────────────
def panel_structure(ax, sdm):
    ax.set_xlim(0, 10); ax.set_ylim(1.05, 6.85)
    ax.axis("off")

    lx = {"MA": 1.5, "HE": 3.6, "Hard_EX": 6.4, "Soft_EX": 8.5}
    ly = 5.0
    # kotak kelompok severity
    # Label kelompok ditaruh di tepi LUAR masing-masing kotak supaya bagian tengah
    # bebas untuk keterangan sisi lintas-grade.
    for x0, x1, name, side in ((0.55, 4.55, "red lesions", "left"),
                               (5.45, 9.45, "bright lesions", "right")):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x0, ly - 0.75), x1 - x0, 1.5, boxstyle="round,pad=0.06",
            fc="#F4F4F4", ec="#BBBBBB", lw=0.8, zorder=1))
        tx = x0 + 0.14 if side == "left" else x1 - 0.14
        ax.text(tx, ly + 0.58, name, ha=side, va="center",
                fontsize=7.2, color="#8A8A8A")

    for key, lbl in zip(LESIONS, LESION_LBL):
        ax.add_patch(mpatches.FancyBboxPatch(
            (lx[key] - 0.72, ly - 0.32), 1.44, 0.64, boxstyle="round,pad=0.04",
            fc="#FFFFFF", ec=C["b4"], lw=1.1, zorder=3))
        ax.text(lx[key], ly, lbl, ha="center", va="center", fontsize=8.0, zorder=4)

    # sisi dalam-grade (biaya 0): garis putus-putus
    for a, b in (("MA", "HE"), ("Hard_EX", "Soft_EX")):
        v = sdm[frozenset((a, b))]
        ax.plot([lx[a] + 0.72, lx[b] - 0.72], [ly - 0.05, ly - 0.05],
                color="#999999", lw=1.0, ls=(0, (3, 2)), zorder=2)
        if a == "MA":
            ax.text((lx[a] + lx[b]) / 2, ly - 0.55, f"same-prior, cost {fmt(v, 2)}",
                    ha="center", va="top", fontsize=6.9, color="#777777")

    # sisi lintas-grade (biaya > 0): garis penuh antar kelompok
    v_cross = sdm[frozenset(("HE", "Hard_EX"))]
    ax.plot([lx["HE"] + 0.72, lx["Hard_EX"] - 0.72], [ly, ly],
            color=C["b6"], lw=1.8, zorder=2)
    ax.text((lx["HE"] + lx["Hard_EX"]) / 2, ly + 0.86,
            f"cross-prior, cost {fmt(v_cross, 2)}", ha="center", va="bottom",
            fontsize=6.9, color=C["b6"])

    # panah pathophysiological berarah, semuanya berpangkal di MA
    ax.annotate("", xy=(lx["Hard_EX"] - 0.74, ly + 0.62), xytext=(lx["MA"], ly + 0.34),
                arrowprops=dict(arrowstyle="-|>", color=C["warn"], lw=1.1,
                                connectionstyle="arc3,rad=-0.28", shrinkA=0, shrinkB=2))
    ax.text(4.0, ly + 1.46, "MA $\\rightarrow$ Hard EX (lipid leakage)", ha="center",
            va="bottom", fontsize=6.9, color=C["warn"])

    # zona ETDRS
    zy = 2.0
    for i, lbl in enumerate(ZONE_LBL):
        zx = 1.0 + i * 2.0
        ax.add_patch(mpatches.FancyBboxPatch(
            (zx - 0.78, zy - 0.42), 1.56, 0.84, boxstyle="round,pad=0.04",
            fc="#FFFFFF", ec="#999999", lw=0.9, zorder=3))
        ax.text(zx, zy, lbl, ha="center", va="center", fontsize=6.9, zorder=4)
    ax.annotate("", xy=(5.0, zy + 0.55), xytext=(5.0, ly - 0.95),
                arrowprops=dict(arrowstyle="-|>", color="#AAAAAA", lw=0.9))
    ax.text(5.25, (zy + ly) / 2 - 0.2, "lesion $\\times$ zone\nexploratory APS",
            ha="left", va="center", fontsize=6.9, color="#777777", linespacing=1.3)
    ax.text(5.0, 1.32, "five ETDRS zones", ha="center", va="top", fontsize=7.2,
            color="#8A8A8A")


# ── panel (b): penurunan SDCA ────────────────────────────────────────────────
def panel_sdca(ax, sdm):
    v_he = sdm[frozenset(("MA", "HE"))]
    v_hex = sdm[frozenset(("MA", "Hard_EX"))]
    cases = [("runner-up HE\n(same prior)", v_he, C["accent"], "retain"),
             ("runner-up Hard EX\n(cross prior)", v_hex, C["b6"], "suppress")]

    ys = [1, 0]
    ax.barh(ys, [c[1] for c in cases], height=0.42,
            color=[c[2] for c in cases], edgecolor="none")
    for y, (lbl, v, col, act) in zip(ys, cases):
        ax.text(v + 0.02, y, fmt(v, 2), va="center", fontsize=8.0, color=col)
    ax.set_yticks(ys)
    ax.set_yticklabels([c[0] for c in cases])
    ax.set_xlim(0, 0.72)
    ax.set_ylim(-0.6, 1.75)
    ax.set_xlabel("SDCA (prior-defined conflict)")
    ax.grid(axis="x"); ax.grid(axis="y", visible=False)
    ax.text(0.02, 0.97, "one detection, identical detector score",
            transform=ax.transAxes, fontsize=7.2, color="#8A8A8A",
            va="top", ha="left")


# ── panel (c): matriks Compat ────────────────────────────────────────────────
def panel_compat(ax, Cm):
    im = ax.imshow(Cm, cmap="Blues", vmin=0.40, vmax=0.72, aspect="auto")
    for i in range(len(LESIONS)):
        for j in range(len(ZONES)):
            v = Cm[i, j]
            ax.text(j, i, fmt(v, 2), ha="center", va="center", fontsize=6.8,
                    color="white" if v > 0.60 else C["text"])
    ax.set_xticks(range(len(ZONES))); ax.set_xticklabels(ZONE_LBL, fontsize=6.6)
    ax.set_yticks(range(len(LESIONS))); ax.set_yticklabels(LESION_LBL)
    ax.set_xlabel("ETDRS zone")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("Exploratory compatibility", fontsize=7.6)
    cb.ax.tick_params(labelsize=6.8)


def main() -> int:
    sdm = load_sdm()
    Cm = load_compat()
    print("[fig_kg_evidence] SDM dibaca:",
          ", ".join(f"MA-{b}={sdm[frozenset(('MA', b))]:.2f}"
                    for b in ("HE", "Hard_EX", "Soft_EX")))
    print("[fig_kg_evidence] Compat dibaca dari compat_scores.json, rentang "
          f"{Cm.min():.3f}..{Cm.max():.3f}")

    apply_style()
    fig = plt.figure(figsize=(7.4, 4.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.92, 1.0],
                          width_ratios=[1.0, 1.0], hspace=0.46, wspace=0.42)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    panel_structure(ax_a, sdm)
    panel_sdca(ax_b, sdm)
    panel_compat(ax_c, Cm)

    # Penanda panel di BAWAH panelnya masing-masing, seperti figur lain. Karena
    # tata letaknya dua baris, penanda (a) diletakkan di celah antar-baris, bukan
    # di dasar gambar: satu garis untuk semua penanda akan memisahkan (a) dari
    # panelnya sendiri. Dipasang setelah tata letak final.
    fig.subplots_adjust(bottom=0.195, top=0.93)
    # Panel (a) skematik: kotak sumbunya jauh lebih besar daripada gambar yang
    # benar-benar tergambar, sehingga penanda yang dipatok ke kotak sumbu tampak
    # mengambang. Yang dipakai kotak-batas hasil render.
    fig.canvas.draw()
    bb = ax_a.get_tightbbox(fig.canvas.get_renderer()).transformed(
        fig.transFigure.inverted())
    panel_tags_bottom(fig, [ax_a], ["(a)"], y=bb.y0 - 0.045)
    panel_tags_bottom(fig, [ax_b, ax_c], ["(b)", "(c)"], y=0.015)

    save(fig, "fig_kg_evidence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
