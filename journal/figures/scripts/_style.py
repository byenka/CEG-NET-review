# -*- coding: utf-8 -*-
"""Gaya bersama untuk seluruh figur CEG-Net.

Tujuan: figur terbaca sebagai satu sistem, dan setiap keputusan visual punya alasan.

Aturan yang diberlakukan di sini:
  * TANPA judul di dalam gambar. Caption LaTeX yang mengerjakan itu; judul di dalam
    gambar menduplikasi caption dan memakan ruang kolom.
  * TANPA anotasi panah atau label "best". Itu perangkat presentasi, bukan figur jurnal.
    Kalau sebuah nilai perlu ditunjuk panah, encoding-nya yang salah.
  * Palet aman buta warna (Okabe-Ito) dengan PERAN TETAP, sehingga pembaca tidak perlu
    membaca ulang legenda di tiap figur:
        M0 / skor detektor  -> abu-abu
        KG-gate (B4)        -> biru
        + multi-source (B6) -> vermilion
  * Tanpa spine atas dan kanan, grid hanya pada sumbu nilai, tipis.
  * Label nilai hanya dipasang bila menambah informasi yang tidak terbaca dari sumbu.

Pemakaian:
    from _style import apply_style, C, save
    apply_style()
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ...
    save(fig, "fig6_fp_reduction_benchmark")
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
OUT_DIRS = [ROOT / "_FINAL_Latex/figures", ROOT / "journal/figures"]

# Okabe-Ito, aman untuk deuteranopia dan protanopia.
C = {
    "m0": "#767676",          # skor detektor, baseline
    "b4": "#0072B2",          # KG-gate
    "b6": "#D55E00",          # + multi-source
    "accent": "#009E73",      # hijau, untuk kategori RETAIN
    "warn": "#CC79A7",        # ungu-merah, untuk kategori berbahaya
    "neutral": "#BBBBBB",     # batang konteks
    "grid": "#D9D9D9",
    "text": "#1A1A1A",
}

# Urutan kategori keandalan yang dipakai konsisten di fig11.
CATEGORY_ORDER = [
    ("TP_EXACT", "TP exact", "retain"),
    ("CLINICALLY_CLOSE_MISMATCH", "Clinically close", "retain"),
    ("LOCALIZATION_WEAK", "Localization weak", "retain"),
    ("LOW_CONF_CANDIDATE", "Low-confidence", "suppress"),
    ("BACKGROUND_FP", "Background FP", "suppress"),
    ("UNSAFE_MISMATCH", "Unsafe mismatch", "suppress"),
]


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8.5,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "axes.titleweight": "regular",
        "axes.labelcolor": C["text"],
        "text.color": C["text"],
        "xtick.color": C["text"],
        "ytick.color": C["text"],
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": C["grid"],
        "grid.linewidth": 0.5,
        "grid.alpha": 0.9,
        "legend.frameon": False,
        "lines.linewidth": 1.4,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
    })


def fmt(v: float, nd: int = 3) -> str:
    """Pembulatan setengah-ke-atas, sama dengan yang dipakai tabel LaTeX.

    Perlu eksplisit: format bawaan Python memakai pembulatan biner, sehingga
    0.6815 menjadi "0.681" sedangkan Tabel 2 menulis "0.682". Selisih satu digit
    antara figur dan tabel adalah jenis inkonsistensi yang dicari reviewer.
    """
    from decimal import Decimal, ROUND_HALF_UP
    q = Decimal(1).scaleb(-nd)
    return str(Decimal(repr(float(v))).quantize(q, rounding=ROUND_HALF_UP))


def yonly(ax) -> None:
    """Grid hanya pada sumbu y (untuk bar chart kategorikal)."""
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)


def panel_tag(ax, tag: str, dx: float = -0.14, dy: float = 1.04) -> None:
    """Penanda panel (a)/(b)/(c) di luar area plot, bukan judul."""
    ax.text(dx, dy, tag, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="bottom", ha="left")


def panel_tags_bottom(fig, axes, tags, y: float = 0.015) -> None:
    """Penanda panel (a)/(b)/(c) di BAWAH, sejajar pada satu garis.

    Memakai koordinat FIGUR, bukan koordinat sumbu, supaya semua penanda berada
    pada ketinggian yang sama walau tiap panel punya label sumbu berbeda panjang.
    Panggil SETELAH tata letak final (setelah subplots_adjust / tight_layout),
    karena posisinya dibaca dari ax.get_position().

    Satu entri `axes` boleh berupa DAFTAR sumbu, bukan satu sumbu: penandanya lalu
    dipusatkan pada keseluruhan kelompok. Itu dipakai ketika satu penanda menaungi
    sebaris panel kecil (fig12 panel (c) berisi empat domain); dipusatkan pada panel
    pertama saja, penandanya terbaca seolah hanya menamai panel itu.

    Untuk tata letak lebih dari satu baris, panggil sekali per baris dengan `y`
    masing-masing: satu garis untuk semua penanda akan memisahkan penanda baris atas
    dari panelnya sendiri.
    """
    for item, tag in zip(axes, tags):
        group = item if isinstance(item, (list, tuple)) else [item]
        boxes = [a.get_position() for a in group]
        x0 = min(b.x0 for b in boxes)
        x1 = max(b.x1 for b in boxes)
        fig.text((x0 + x1) / 2.0, y, tag, ha="center", va="bottom",
                 fontsize=9, fontweight="bold")


def save(fig, stem: str) -> None:
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(d / f"{stem}.{ext}")
    plt.close(fig)
    print(f"[fig] {stem}.png/.pdf -> " + ", ".join(str(d.relative_to(ROOT)) for d in OUT_DIRS))
