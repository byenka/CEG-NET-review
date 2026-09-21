#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""fig6 — perbandingan terkendali keluarga FP-reduction, OIA-DDR test.

DIBANGUN ULANG 2026-09-15 (Opsi B). Sumber sebelumnya evsds_full_benchmark.json
(konfigurasi full, seed 0). Kini membaca hasil protokol tunggal:
  outputs/0506/indomain_5seed/indomain_results.json   (tools/make_indomain_results.py)
konfigurasi deployed; baris gate adalah rerata lima seed. Pembanding linear kini memakai
masukan yang SAMA dengan knowledge-graph gate (evidence + skor), sehingga selisih linear
vs nonlinear tidak lagi terkonfound oleh perbedaan masukan.

Jalankan dari root repo:
    python journal/figures/scripts/fig6_fp_reduction_benchmark.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import C, apply_style, panel_tags_bottom, save  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "outputs/0506/indomain_5seed/indomain_results.json"
# Baseline fusi skor dan gate yang membacanya. Ditambahkan 2026-09-16: tanpa baris ini figur
# menampilkan gate penuh (0.699) sebagai batang tertinggi sementara Tabel 8 menempatkan gate
# di atas skor fusi (0.733) di puncak, yaitu persis jenis inkonsistensi yang dicari reviewer.
FUS = ROOT / "_septem/scripts_optionB/baselines_nokg.json"
GOF = ROOT / "_septem/scripts_optionB/gate_on_fusion.json"

FUSION = [
    ("tta_mean0", "Flip TTA, score mean", "ens"),
    ("ens2_mean0", "Two-detector ensemble, score mean", "ens"),
    ("ens3_mean0", "Three-pass score mean", "ens"),
]

FIXED = [
    ("M0", "Detector score ($M_0$)", "base"),
    ("MSP", "Max-softmax prob.", "unc"),
    ("Entropy", "Predictive entropy", "unc"),
    ("Margin", "Top-1 $-$ top-2 margin", "unc"),
    ("TempScale", "Temperature scaling", "base"),
    ("LogReg_evidence_score", "Knowledge-graph gate, linear", "lin"),
]
GATES = [
    ("score_only", "Score-only gate", "base"),
    ("kg_gate", "Knowledge-graph gate, MLP", "b4"),
    ("full_gate", "Full gate, CEG-Net", "b6"),
]
# Nama sama dengan baris Tabel 8. Warna mengikuti fig2 dan fig8: biru = knowledge-graph gate
# (B4), jingga = gate penuh (B6); bentuk linear dari gate yang sama memakai biru muda. Versi
# sebelumnya memberi regresi logistik warna B4 dan gate B4 warna B6, bertentangan dengan fig2/fig8.
# Dua peran baru, dipilih supaya tidak menabrak peran tetap di _style.py:
#   ens  hijau  = keluarga fusi skor (bukan gate sama sekali);
#   abl  jingga muda = gate penuh dengan knowledge graph dibuang, jadi tint dari b6, mengikuti
#        logika yang sama seperti biru muda untuk bentuk linear dari b4;
#   b6f  jingga tua = gate penuh yang membaca skor fusi, yaitu b6 dengan satu masukan tambahan.
GROUP_COLOR = {"base": C["m0"], "unc": C["neutral"], "lin": "#8FC1E3", "b4": C["b4"],
               "b6": C["b6"], "ens": C["accent"], "abl": "#F0A868", "b6f": "#8C3D00"}


def main() -> int:
    d = json.loads(SRC.read_text(encoding="utf-8"))
    rows = []
    for key, name, grp in FIXED:
        r = d["fixed"][key]
        rows.append((name, grp, {"auprc": r["auprc"], "tp_ret": r["tp_ret"], "fp_supp": r["fp_supp"]}))
    for key, name, grp in GATES:
        g = d["summary"][key]
        rows.append((name, grp, {"auprc": g["auprc"]["mean"], "tp_ret": g["tp_ret"]["mean"],
                                 "fp_supp": g["fp_supp"]["mean"]}))
    # Baseline fusi dan gate yang membacanya; sumbernya dua JSON terpisah dengan protokol sama.
    bas = json.loads(FUS.read_text(encoding="utf-8"))
    gof = json.loads(GOF.read_text(encoding="utf-8"))
    for key, name, grp in FUSION:
        r = bas["fusion"][key]
        rows.append((name, grp, {"auprc": r["ap"], "tp_ret": r["tp_ret"], "fp_supp": r["fp_supp"]}))
    for src, key, name, grp in ((bas, "nokg_gate", "Full gate without knowledge graph", "abl"),
                                (gof, "gate_fused", "Full gate on the three-pass mean", "b6f")):
        r = src["gates"][key]
        rows.append((name, grp, {"auprc": r["ap_mean"], "tp_ret": r["tp_ret_mean"],
                                 "fp_supp": r["fp_supp_mean"]}))
    rows.sort(key=lambda x: x[2]["auprc"])
    m0 = d["fixed"]["M0"]["auprc"]

    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0),
                                   gridspec_kw={"width_ratios": [1.30, 1.0]})

    # ---- (a) AUPRC per metode --------------------------------------------
    ys = list(range(len(rows)))
    ax1.barh(ys, [r["auprc"] for _n, _g, r in rows],
             color=[GROUP_COLOR[g] for _n, g, _r in rows], height=0.68, edgecolor="none")
    ax1.set_yticks(ys)
    ax1.set_yticklabels([n for n, _g, _r in rows])
    ax1.set_xlabel("AUPRC (retain vs suppress)")
    ax1.set_xlim(0, 0.82)
    ax1.grid(axis="x")
    ax1.grid(axis="y", visible=False)
    ax1.axvline(m0, color=C["m0"], lw=0.9, ls=(0, (4, 3)), zorder=3)
    for i, (_n, _g, r) in enumerate(rows):
        ax1.text(r["auprc"] + 0.009, i, f"{r['auprc']:.3f}", va="center", fontsize=7.4)
    ax1.set_ylim(-0.6, len(rows) - 0.4)

    # ---- (b) operating point ---------------------------------------------
    # Titik gate hampir berimpit: B4 digambar di atas bentuk linear, B6 paling atas, supaya garis
    # penunjuk "KG gate, MLP" berakhir pada titik biru yang terlihat.
    zorder = {"base": 3, "unc": 3, "lin": 3, "b4": 4, "b6": 5, "ens": 3, "abl": 4, "b6f": 6}
    for _name, grp, r in rows:
        ax2.scatter(r["tp_ret"], r["fp_supp"], s=34, color=GROUP_COLOR[grp], zorder=zorder[grp],
                    edgecolor="white", linewidth=0.6)
    by = {n: r for n, _g, r in rows}
    groups = [
        ("$M_0$, temp. scaling,\nscore-only gate", by["Detector score ($M_0$)"], (8, 6), None),
        ("max-softmax, entropy,\nmargin", by["Max-softmax prob."], (8, -4), None),
        # Tiga titik gate hampir berimpit. Labelnya dipasang di ruang kosong dengan garis
        # penunjuk; dengan offset biasa "KG gate, MLP" menimpa label sumbu 0.6 dan urutan
        # label tidak jelas milik titik yang mana.
        # Koordinat label ditetapkan dari posisi titik yang sebenarnya, bukan dikira-kira:
        #   KG linear (0.802, 0.573)   KG MLP (0.808, 0.575)   CEG-Net (0.819, 0.578)
        #   gate-atas-fusi (0.806, 0.662)   fusi tiga pass (0.846, 0.600)
        #   tanpa-KG (0.857, 0.521) dan ensemble dua pass (0.857, 0.523) hampir berimpit
        # Percobaan pertama menempatkan "KG gate, linear" pada (0.745, 0.690) dan "CEG-Net"
        # pada (0.880, 0.505); keduanya salah tunjuk, yang pertama berakhir di titik
        # gate-atas-fusi dan yang kedua bersebelahan dengan titik Flip TTA. Label kini memakai
        # ruang kosong di kiri atas dan di kanan, dan tiap garis penunjuk dicek agar tidak
        # melewati titik lain.
        ("KG gate, linear", by["Knowledge-graph gate, linear"], None, (0.735, 0.700)),
        ("KG gate, MLP", by["Knowledge-graph gate, MLP"], None, (0.735, 0.450)),
        ("CEG-Net", by["Full gate, CEG-Net"], None, (0.900, 0.530)),
        # Dua titik yang menjadi pokok pembandingnya diberi label; dua baseline fusi lain
        # tidak, karena panel (a) sudah mengidentifikasi setiap metode dan menambah label
        # di sini hanya menumpuk teks pada wilayah yang sudah padat.
        ("Three-pass mean", by["Three-pass score mean"], None, (0.900, 0.645)),
        ("Gate on three-pass mean", by["Full gate on the three-pass mean"], None, (0.855, 0.760)),
    ]
    for label, r, off, txt in groups:
        if txt is None:
            dx, dy = off
            ax2.annotate(label, (r["tp_ret"], r["fp_supp"]), textcoords="offset points",
                         xytext=(dx, dy), fontsize=6.9, ha="left" if dx > 0 else "right",
                         va="center", color=C["text"], linespacing=1.25)
        else:
            ax2.annotate(label, (r["tp_ret"], r["fp_supp"]), xytext=txt, textcoords="data",
                         fontsize=6.9, ha="left", va="center", color=C["text"],
                         arrowprops={"arrowstyle": "-", "color": "#777777", "lw": 0.6,
                                     "shrinkA": 2.0, "shrinkB": 4.0})
    ax2.set_xlabel("TP retention at selected threshold")
    ax2.set_ylabel("FP suppression")
    ax2.set_xlim(0.73, 1.00)
    ax2.set_ylim(0.0, 0.82)

    fig.subplots_adjust(wspace=0.46, bottom=0.24)
    panel_tags_bottom(fig, [ax1, ax2], ["(a)", "(b)"], y=0.015)
    save(fig, "fig6_fp_reduction_benchmark")
    for n, _g, r in rows:
        print(f"[fig6] {n:32s} AUPRC={r['auprc']:.4f} TP={r['tp_ret']:.4f} FP={r['fp_supp']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
