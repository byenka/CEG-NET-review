#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""fig11 — analisis galat dan mekanisme, OIA-DDR test.

DIBANGUN ULANG 2026-09-14. Masalah versi sebelumnya:
  (a) teks merah "red | bright group boundary" menabrak label sumbu sampai tak terbaca;
  (b) sumbu ganda (SDCA kiri, skor kanan) mengundang salah baca, padahal KEDUA besaran
      berada di [0,1] sehingga satu sumbu sudah cukup; ditambah anotasi bergaris penunjuk;
  (c) label di dalam batang bertumpuk sampai tak terbaca, nama kategori terpotong, dan
      kategori UNSAFE (n=91, 0.14%) tak terlihat sehingga dipasangi PANAH MERAH
      bertuliskan "dangerous UNSAFE here". Kalau sebuah nilai perlu ditunjuk panah,
      encoding-nya yang salah: di sini rentang hitungan 91..20.589 menuntut skala log,
      bukan batang bertumpuk.

Sumber:
  matriks konfusi : outputs/0506/evsds_v2sec/oiaddr/test/governance_features_test.csv
                    (best_iou >= 0.5; baris = best_gt_class, kolom = pred_class)
  rerata & jumlah : outputs/0506/evsds_v2sec/oiaddr/test/governance_target_distribution.csv

Jalankan dari root repo:
    python journal/figures/scripts/fig11_error_analysis.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import C, apply_style, panel_tags_bottom, save  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "outputs/0506/evsds_v2sec/oiaddr/test"
FEATURES = BASE / "governance_features_test.csv"
DIST = BASE / "governance_target_distribution.csv"

CLASSES = ["MA", "HE", "Hard_EX", "Soft_EX"]
CLASS_LBL = ["MA", "HE", "Hard\nEX", "Soft\nEX"]

CATS = [
    ("TP_EXACT", "TP exact", "retain"),
    ("CLINICALLY_CLOSE_MISMATCH", "Clinically close", "retain"),
    ("LOCALIZATION_WEAK", "Localization weak", "retain"),
    ("LOW_CONF_CANDIDATE", "Low-confidence", "suppress"),
    ("BACKGROUND_FP", "Background FP", "suppress"),
    ("UNSAFE_MISMATCH", "Unsafe mismatch", "suppress"),
]


def read_confusion():
    idx = {c: i for i, c in enumerate(CLASSES)}
    M = np.zeros((4, 4), dtype=float)
    with open(FEATURES, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                if float(r["best_iou"]) < 0.5:
                    continue
            except (TypeError, ValueError):
                continue
            g, p = r.get("best_gt_class"), r.get("pred_class")
            if g in idx and p in idx:
                M[idx[g]][idx[p]] += 1
    return M


def read_dist():
    out = {}
    with open(DIST, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["category"]] = r
    return out


def panel_confusion(ax, M):
    """Matriks konfusi, baris dinormalisasi.

    aspect="equal" dipertahankan: matriks konfusi lazimnya persegi, dan sel yang
    tidak persegi menyulitkan pembacaan bobot antar baris. Agar selnya tetap
    lapang, ruang diatur lewat tinggi gambar dan kolom pemisah di main(), bukan
    dengan menggepengkan sel. Sel bernilai nol dikosongkan supaya sel yang berisi
    punya ruang dan terbaca lebih dulu.
    """
    Mn = M / np.clip(M.sum(axis=1, keepdims=True), 1, None)
    ax.imshow(Mn, cmap="Blues", vmin=0, vmax=1, aspect="equal")
    for i in range(4):
        for j in range(4):
            if M[i, j] == 0:
                continue
            light = Mn[i, j] > 0.55
            ax.text(j, i - 0.10, f"{Mn[i, j]:.2f}", ha="center", va="center",
                    fontsize=7.4, color="white" if light else C["text"])
            ax.text(j, i + 0.22, f"{int(M[i, j]):,}", ha="center", va="center",
                    fontsize=6.0,
                    color="#DCE9F5" if light else "#8A8A8A")
    # Pemisah kelompok lesi merah (MA, HE) dan terang (Hard_EX, Soft_EX). Tanpa teks
    # di dalam gambar: penjelasannya ada di caption.
    ax.axhline(1.5, color="#444444", lw=1.0)
    ax.axvline(1.5, color="#444444", lw=1.0)
    ax.set_xticks(range(4)); ax.set_xticklabels(CLASS_LBL)
    ax.set_yticks(range(4)); ax.set_yticklabels(CLASS_LBL)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Ground-truth class")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def panel_signals(ax, dist):
    """SDCA dan skor detektor pada SATU sumbu: keduanya berada di [0,1]."""
    ys = np.arange(len(CATS))
    sdca = [float(dist[k]["avg_sdca"]) for k, _l, _t in CATS]
    score = [float(dist[k]["avg_pred_score"]) for k, _l, _t in CATS]
    h = 0.36
    ax.barh(ys + h / 2, sdca, height=h, color=C["b4"], label="SDCA (KG)",
            edgecolor="none")
    ax.barh(ys - h / 2, score, height=h, color=C["m0"], label="detector score",
            edgecolor="none")
    for y, v in zip(ys + h / 2, sdca):
        ax.text(v + 0.015, y, f"{v:.2f}", va="center", fontsize=6.6, color=C["b4"])
    for y, v in zip(ys - h / 2, score):
        ax.text(v + 0.015, y, f"{v:.2f}", va="center", fontsize=6.6, color=C["m0"])
    ax.set_yticks(ys)
    ax.set_yticklabels([l for _k, l, _t in CATS])
    ax.invert_yaxis()
    ax.set_xlim(0, 0.92)
    ax.set_xlabel("mean value (both in $[0,1]$)")
    ax.grid(axis="x"); ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2,
              handlelength=0.9, columnspacing=0.9, borderpad=0.2, fontsize=7.2)


def panel_counts(ax, dist):
    """Hitungan per kategori pada skala log: rentangnya 91 sampai 20.589."""
    ys = np.arange(len(CATS))
    n = [int(dist[k]["count"]) for k, _l, _t in CATS]
    col = [C["accent"] if t == "retain" else C["warn"] for _k, _l, t in CATS]
    ax.barh(ys, n, height=0.62, color=col, edgecolor="none")
    for y, v in zip(ys, n):
        ax.text(v * 1.18, y, f"{v:,}", va="center", fontsize=6.8)
    ax.set_yticks(ys); ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(45, 90_000)
    ax.set_xlabel("detections (log scale)")
    ax.grid(axis="x"); ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=C["accent"]),
               plt.Rectangle((0, 0), 1, 1, color=C["warn"])]
    # Legenda diletakkan DI DALAM sumbu: panel (c) sempit, sehingga legenda di atas
    # akan menjulur dan menimpa legenda panel (b).
    ax.legend(handles, ["RETAIN", "SUPPRESS"], loc="lower center",
              bbox_to_anchor=(0.5, 1.01), ncol=2, handlelength=0.9,
              columnspacing=0.8, borderpad=0.2, fontsize=6.8)


def main() -> int:
    M = read_confusion()
    dist = read_dist()

    apply_style()
    # Kolom KOSONG disisipkan antara (a) dan (b) sebagai pemisah. subplots_adjust
    # hanya mengenal satu nilai wspace untuk semua celah, padahal yang butuh jarak
    # lebar hanya celah (a)-(b), tempat label kategori panel (b) menjulur ke kiri.
    # Menaikkan wspace global justru memakan lebar yang dibutuhkan matriks.
    # Tinggi gambar dinaikkan karena dengan aspect="equal" sisi persegi dibatasi
    # oleh dimensi terpendek kotak sumbu; kotak yang lebih tinggi memperbesar matriks.
    fig = plt.figure(figsize=(7.4, 3.7))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.10, 0.24, 1.00, 0.72], wspace=0.34)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 2])
    ax_c = fig.add_subplot(gs[0, 3])

    panel_confusion(ax_a, M)
    panel_signals(ax_b, dist)
    panel_counts(ax_c, dist)
    # Penanda panel dipasang SETELAH tata letak final, karena posisinya dibaca dari
    # ax.get_position(). Ruang bawah dilebihkan agar penanda tidak menempel label sumbu.
    fig.subplots_adjust(top=0.88, bottom=0.22)
    panel_tags_bottom(fig, [ax_a, ax_b, ax_c], ["(a)", "(b)", "(c)"], y=0.015)
    save(fig, "fig11_error_analysis")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
