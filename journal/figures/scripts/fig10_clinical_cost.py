#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""fig10 — kepercayaan detektor buta terhadap bahaya lintas-grade.

DIBANGUN ULANG 2026-09-14. Nilainya benar, tetapi encoding-nya membuat temuan
terkuat paper tampil lemah:
  * Panel kanan lama memakai histogram densitas untuk SDCA. SDCA hanya punya empat
    nilai diskrit (0, 0.50, 0.75, 1.00), sehingga panelnya nyaris kosong dengan
    sumbu densitas sampai 20 yang tidak bermakna.
  * Panel kiri lama menumpuk dua histogram bertransparansi. Itu cara paling lemah
    untuk menunjukkan "kedua distribusi berimpit": warna campurannya justru
    menciptakan warna ketiga yang mengaburkan.
  * Klaimnya adalah tentang AUROC, jadi kurva ROC adalah encoding yang tepat:
    pembaca langsung melihat satu kurva menempel diagonal dan satu menempel sudut.

Sumber: outputs/0506/evsds_v2sec/oiaddr/test/governance_features_test.csv
  UNSAFE_MISMATCH (positif, grade-shifting) vs CLINICALLY_CLOSE_MISMATCH (negatif).

Jalankan dari root repo:
    python journal/figures/scripts/fig10_clinical_cost.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import C, apply_style, fmt, panel_tags_bottom, save  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "outputs/0506/evsds_v2sec/oiaddr/test/governance_features_test.csv"
# Titik per-deteksi skor fusi, dari _septem/scripts_optionB/clinical_cost_fusion.py.
# Ditambahkan 2026-09-16: tanpa kurva ini panel (a) menampilkan dua sinyal sementara
# Tabel 13 di bawahnya memuat tiga, dan justru sinyal ketiga itulah buktinya bahwa
# ensembling memenangkan AUPRC tanpa membeli satu pun dimensi biaya klinis.
PTS = ROOT / "_septem/scripts_optionB/clinical_cost_points.csv"

UNSAFE = "UNSAFE_MISMATCH"
CLOSE = "CLINICALLY_CLOSE_MISMATCH"


def ecdf(v):
    x = np.sort(np.asarray(v, dtype=float))
    return x, np.arange(1, x.size + 1) / x.size


def main() -> int:
    df = pd.read_csv(SRC, usecols=["category", "pred_score", "ev_sdca"],
                     low_memory=False)
    u = df[df.category == UNSAFE]
    c = df[df.category == CLOSE]
    y = np.r_[np.ones(len(u)), np.zeros(len(c))]
    s_score = np.r_[u.pred_score.to_numpy(), c.pred_score.to_numpy()]
    s_sdca = np.r_[u.ev_sdca.to_numpy(), c.ev_sdca.to_numpy()]
    a_score = roc_auc_score(y, s_score)
    a_sdca = roc_auc_score(y, s_sdca)

    pts = pd.read_csv(PTS)
    if len(pts) != len(u) + len(c):
        raise SystemExit(f"[fig10] titik fusi {len(pts)} != {len(u) + len(c)}")
    y_f = pts.unsafe.to_numpy()
    s_fused = pts.fused_three_pass.to_numpy()
    a_fused = roc_auc_score(y_f, s_fused)
    # Kedua sumber harus sepakat pada skor detektor sebelum kurvanya digambar bersama.
    if abs(roc_auc_score(y_f, pts.detector_score.to_numpy()) - a_score) > 1e-9:
        raise SystemExit("[fig10] AUROC skor detektor berbeda antara dua sumber")

    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.9, 3.2))

    # ---- (a) ROC: satu kurva di diagonal, satu di sudut --------------------
    ax1.plot([0, 1], [0, 1], color="#AAAAAA", lw=0.8, ls=(0, (2, 2)), zorder=1)
    for yy, s, col, z in ((y_f, s_fused, C["accent"], 2), (y, s_score, C["m0"], 3),
                          (y, s_sdca, C["b4"], 4)):
        fpr, tpr, _ = roc_curve(yy, s)
        ax1.plot(fpr, tpr, color=col, lw=1.6, zorder=z)
    # Kurva diberi label LANGSUNG, tanpa kotak legenda. Kedua kurva berada di posisi
    # yang sangat berbeda (satu menempel sudut, satu di diagonal), sehingga kotak
    # legenda hanya akan menutupi salah satunya di mana pun ia diletakkan.
    ax1.text(0.30, 0.955, f"KG clinical cost (SDCA)\nAUROC {fmt(a_sdca)}",
             color=C["b4"], fontsize=7.2, ha="left", va="top", linespacing=1.35)
    # Kedua kurva berbasis skor berimpit di diagonal, jadi labelnya ditumpuk di ruang
    # kosong kanan-bawah, bukan ditempel pada kurva masing-masing yang akan bertabrakan.
    ax1.text(0.52, 0.33, f"detector score\nAUROC {fmt(a_score)}  (chance)",
             color=C["m0"], fontsize=7.2, ha="left", va="top", linespacing=1.35)
    ax1.text(0.52, 0.16, f"three-pass score mean\nAUROC {fmt(a_fused)}  (chance)",
             color=C["accent"], fontsize=7.2, ha="left", va="top", linespacing=1.35)
    ax1.set_xlabel("False-positive rate")
    ax1.set_ylabel("True-positive rate")
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)

    # ---- (b) ECDF skor: kedua kelas praktis berimpit -----------------------
    # Panel (b) menggambar DUA KELAS, bukan dua metode. Hijau sengaja tidak dipakai di sini:
    # sejak kurva fusi masuk ke panel (a), hijau sudah punya arti tetap "fusi skor" di fig6,
    # fig8 dan panel (a) figur ini, sehingga memakainya lagi untuk kelas within-grade membuat
    # satu warna berarti dua hal di dalam satu gambar. Merah muda untuk kelas berbahaya adalah
    # peran tetap C["warn"] di _style.py; pasangannya dibuat netral gelap.
    for v, col, lab in ((c.pred_score, "#444444", f"within-grade (n={len(c):,})"),
                        (u.pred_score, C["warn"], f"grade-shifting (n={len(u)})")):
        x, p = ecdf(v)
        ax2.step(x, p, where="post", color=col, lw=1.5, label=lab)
    ax2.set_xlabel("detector confidence score")
    ax2.set_ylabel("cumulative fraction")
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
    ax2.legend(loc="lower right", handlelength=1.2, handletextpad=0.5,
               labelspacing=0.3, borderpad=0.25, fontsize=7.2)
    ax2.text(0.03, 0.965,
             f"mean {u.pred_score.mean():.3f} vs {c.pred_score.mean():.3f}",
             transform=ax2.transAxes, ha="left", va="top", fontsize=7.2,
             color="#555555")

    # Penanda panel dipasang SETELAH tata letak final; posisinya dibaca dari
    # ax.get_position(). Ruang bawah dilebihkan supaya penanda tidak menempel
    # pada label sumbu.
    fig.subplots_adjust(wspace=0.34, top=0.95, bottom=0.24)
    panel_tags_bottom(fig, [ax1, ax2], ["(a)", "(b)"], y=0.015)
    save(fig, "fig10_clinical_cost")
    print(f"[fig10] AUROC score={a_score:.4f}  SDCA={a_sdca:.4f}  "
          f"n_unsafe={len(u)}  n_close={len(c)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
