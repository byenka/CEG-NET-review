#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""fig12 — peringkat bebas-ambang: kurva PR dan ROC.

DIBANGUN ULANG 2026-09-14. Dua masalah pada versi sebelumnya:
  1. Panel (c) memakai data PSS placeholder, sehingga garis putus-putus M0 untuk IDRiD
     justru berada DI ATAS garis gate. Itu berlawanan dengan caption-nya sendiri.
  2. Delapan kurva bertumpuk dalam satu sumbu membuat panel (c) sulit dibaca, ditambah
     judul di dalam gambar yang mengulang caption.

Versi ini memakai PSS nyata untuk kohort eksternal dan memecah panel (c) menjadi empat
panel kecil, satu per domain, sehingga tiap perbandingan gate-versus-skor dapat dibaca
tanpa menelusuri legenda.

Sumber:
  in-domain : outputs/0506/evsds_v2sec/oiaddr/{valid,test}/governance_features_*.csv
  eksternal : outputs/0506/evsds_pss/{idrid,tjdr,fgadr}/*/governance_features_*.csv
Kurva memakai seed gate 0; luas yang dikutip caption berasal dari tabel.

Jalankan dari root repo (butuh beberapa menit karena memuat CSV besar):
    python journal/figures/scripts/fig12_pr_roc.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import C, apply_style, panel_tags_bottom, save  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.eval_fp_reduction_benchmark import build_methods, load_csv  # noqa: E402

BASE = ROOT / "outputs/0506"
POS = "governance_retain"
M0, B4, B6 = "M0_score", "KG-gate-MLP (ours)", "KG-gate+SDS+corrob (B6)"

DOMAINS = [
    ("OIA-DDR", "in-domain", BASE / "evsds_v2sec/oiaddr/test/governance_features_test.csv"),
    ("IDRiD", "zero-shot", BASE / "evsds_pss/idrid/test/governance_features_test.csv"),
    ("TJDR", "zero-shot", BASE / "evsds_pss/tjdr/test/governance_features_test.csv"),
    ("FGADR", "zero-shot", BASE / "evsds_pss/fgadr/train/governance_features_train.csv"),
]


def thin(x, y, n=600):
    """Kurangi titik kurva agar PDF tidak membengkak, bentuk kurva tetap sama."""
    if len(x) <= n:
        return x, y
    idx = np.unique(np.linspace(0, len(x) - 1, n).astype(int))
    return x[idx], y[idx]


def scores_for(valid, test):
    out = {}
    for name, _fam, _sv, st in build_methods(valid, test, mlp_seed=0):
        if name in (M0, B4, B6):
            out[name] = st
    return out


def main() -> int:
    valid = load_csv(BASE / "evsds_v2sec/oiaddr/valid/governance_features_valid.csv", POS)

    apply_style()
    fig = plt.figure(figsize=(7.3, 4.9))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.25, 1.0], hspace=0.62, wspace=0.42)
    ax_pr = fig.add_subplot(gs[0, :2])
    ax_roc = fig.add_subplot(gs[0, 2:])
    small = [fig.add_subplot(gs[1, i]) for i in range(4)]

    for i, (name, role, path) in enumerate(DOMAINS):
        if not path.is_file():
            print(f"[fig12] LEWAT {name}: {path} tidak ada")
            continue
        print(f"[fig12] memuat {name} ...", flush=True)
        test = load_csv(path, POS)
        y = test["label"]
        s = scores_for(valid, test)

        if name == "OIA-DDR":
            # Label sengaja dipendekkan: label panjang membuat kotak legenda melebar
            # sampai menimpa kurva. Kepanjangannya dijelaskan di caption.
            for key, col, lab in ((M0, C["m0"], "$M_0$ score"),
                                  (B4, C["b4"], "KG-gate"),
                                  (B6, C["b6"], "+ multi-source")):
                pr_p, pr_r, _ = precision_recall_curve(y, s[key])
                a = auc(pr_r, pr_p)
                xx, yy = thin(pr_r, pr_p)
                ax_pr.plot(xx, yy, color=col, label=f"{lab}  {a:.3f}")
                fpr, tpr, _ = roc_curve(y, s[key])
                xx, yy = thin(fpr, tpr)
                ax_roc.plot(xx, yy, color=col,
                            label=f"{lab}  {roc_auc_score(y, s[key]):.3f}")
            chance = float((y == 1).mean())
            ax_pr.axhline(chance, color="#999999", lw=0.8, ls=(0, (2, 2)))
            ax_pr.text(0.99, chance + 0.012, f"chance {chance:.3f}", ha="right",
                       fontsize=6.8, color="#777777")
            ax_roc.plot([0, 1], [0, 1], color="#999999", lw=0.8, ls=(0, (2, 2)))

        ax = small[i]
        for key, col, ls in ((M0, C["m0"], (0, (3, 2))), (B6, C["b6"], "-")):
            pr_p, pr_r, _ = precision_recall_curve(y, s[key])
            xx, yy = thin(pr_r, pr_p, 400)
            ax.plot(xx, yy, color=col, ls=ls, lw=1.2)
        pr_p, pr_r, _ = precision_recall_curve(y, s[M0])
        a0 = auc(pr_r, pr_p)
        pr_p6, pr_r6, _ = precision_recall_curve(y, s[B6])
        a6 = auc(pr_r6, pr_p6)
        ax.set_title(f"{name}\n{role}", fontsize=7.6, pad=3)
        ax.text(0.97, 0.95, f"B6 {a6:.3f}\n$M_0$ {a0:.3f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=6.6, linespacing=1.3)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.5, 1.0])
        if i == 0:
            ax.set_ylabel("Precision")
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("Recall")
        del test

    ax_pr.set_xlabel("Recall (TP retention)"); ax_pr.set_ylabel("Precision")
    ax_pr.set_xlim(0, 1); ax_pr.set_ylim(0.3, 1.0)
    # Sudut KANAN-ATAS yang kosong pada kurva PR menurun ini: di recall tinggi
    # presisinya justru rendah. Di kiri-bawah legendanya menabrak garis chance
    # beserta labelnya, yang memang duduk di dasar sumbu.
    ax_pr.legend(loc="upper right", title="AUPRC", title_fontsize=7.0,
                 fontsize=6.9, handlelength=1.1, handletextpad=0.5,
                 labelspacing=0.28, borderpad=0.25)
    ax_roc.set_xlabel("False-positive rate"); ax_roc.set_ylabel("True-positive rate")
    ax_roc.set_xlim(0, 1); ax_roc.set_ylim(0, 1)
    ax_roc.legend(loc="lower right", title="AUROC", title_fontsize=7.0,
                  fontsize=6.9, handlelength=1.1, handletextpad=0.5,
                  labelspacing=0.28, borderpad=0.25)

    # Penanda panel di bawah panelnya masing-masing. Tata letaknya dua baris, jadi
    # penanda baris atas diletakkan tepat di bawah kotak-batas hasil render kedua
    # panel besar, bukan di dasar gambar. Penanda (c) menaungi seluruh baris bawah,
    # karena keempat panel kecil itu satu panel (c) yang sama.
    fig.subplots_adjust(bottom=0.155)
    fig.canvas.draw()
    bb = ax_pr.get_tightbbox(fig.canvas.get_renderer()).transformed(
        fig.transFigure.inverted())
    panel_tags_bottom(fig, [ax_pr, ax_roc], ["(a)", "(b)"], y=bb.y0 - 0.032)
    bb2 = small[0].get_tightbbox(fig.canvas.get_renderer()).transformed(
        fig.transFigure.inverted())
    panel_tags_bottom(fig, [small], ["(c)"], y=bb2.y0 - 0.032)
    save(fig, "fig12_pr_roc")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
