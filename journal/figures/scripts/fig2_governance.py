#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""fig2 — hasil utama in-domain dengan SATU protokol.

DIBANGUN ULANG 2026-09-15 (Opsi B). Versi sebelumnya membaca lgg_test_metrics.json dari run
Stage3D terkunci. Di run itu gate dipasang pada split TRAIN dengan sklearn MLP (32,16), dan
baris "KG-only" adalah LGG torch berlatih Gumbel-Softmax; naskah justru mendeskripsikan gate
numpy yang dipasang pada VALID. Versi ini membaca hasil protokol yang ditulis naskah:
  outputs/0506/indomain_5seed/indomain_results.json   (tools/make_indomain_results.py)
konfigurasi deployed, lima seed gate, bootstrap image-clustered 2.000 per seed.

Gaya dipertahankan dari revisi 2026-09-14: label pendek, garis acuan M0 tipis dan samar,
nilai AUPRC di atas batang, SATU kontras di dalam gambar.

Jalankan dari root repo:
    python journal/figures/scripts/fig2_governance.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import C, apply_style, fmt, save  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "outputs/0506/indomain_5seed/indomain_results.json"

# Nama pendek saja. Definisi tiap model ada di caption dan di Tabel 2.
BARS = [
    ("$M_0$", "M0", C["m0"]),
    ("Score-only", "score_only", C["m0"]),
    ("Evidence-only", "evidence_only", C["neutral"]),
    ("KG gate", "kg_gate", C["b4"]),
    ("CEG-Net", "full_gate", C["b6"]),
]


def main() -> int:
    d = json.loads(SRC.read_text(encoding="utf-8"))
    s = d["summary"]
    m0 = d["fixed"]["M0"]["auprc"]
    vals, sds = [], []
    for _lbl, key, _c in BARS:
        if key == "M0":
            vals.append(m0)
            sds.append(0.0)
        else:
            vals.append(s[key]["auprc"]["mean"])
            sds.append(s[key]["auprc"]["sd"])

    apply_style()
    fig, ax = plt.subplots(figsize=(4.9, 2.9))
    xs = list(range(len(BARS)))
    ax.bar(xs, vals, width=0.56, color=[c for _l, _k, c in BARS], edgecolor="none", zorder=3)
    # Galat = SD antar-seed, tipis; M0 tidak bergantung seed sehingga tanpa galat.
    ax.errorbar([x for x, e in zip(xs, sds) if e > 0], [v for v, e in zip(vals, sds) if e > 0],
                yerr=[e for e in sds if e > 0], fmt="none", ecolor="#333333", elinewidth=0.7,
                capsize=2.0, zorder=4)
    for x, v, e in zip(xs, vals, sds):
        ax.text(x, v + e + 0.012, fmt(v), ha="center", va="bottom", fontsize=7.4)

    ax.axhline(m0, color=C["m0"], lw=0.6, ls=(0, (5, 4)), alpha=0.55, zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([l for l, _k, _c in BARS], fontsize=7.6)
    ax.set_ylabel("AUPRC (retain vs suppress)")
    ax.set_ylim(0, 0.88)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)

    full = s["full_gate"]["delta_vs_m0"]
    n_ci = s["boot_full_gate_minus_M0"]["n_ci_above_0"]
    ax.text(0.015, 0.99,
            "CEG-Net Δ vs $M_0$ = $+$" + fmt(full["mean"]) + " $\\pm$ " + fmt(full["sd"])
            + f"  ({full['n_pos']}/5 seeds; 95% CI > 0 in {n_ci}/5)",
            transform=ax.transAxes, va="top", ha="left", fontsize=7.0, color="#555555")

    save(fig, "fig2_governance")
    print("[fig2] " + "  ".join(f"{k}={v:.4f}" for (_l, k, _c), v in zip(BARS, vals)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
