#!/usr/bin/env python
"""fig_kg_matrix.py — matriks jarak semantik berjangkar ICDR + inventaris jenis sisi.

Memperlihatkan Clinical Knowledge Graph (Kontribusi 1) sebagai artefak konkret, dengan
membaca LANGSUNG berkas knowledge-graph yang terkunci dan terpakai (anti-halusinasi
secara konstruksi):
  - outputs/stage3_audit/sdm_norm_clinical_prior.json  (SDM_norm yang benar-benar
    dipakai SDCA; dimuat tools/audit_stage3c_governance_targets.py, sdm_source =
    severity_transition_icdr).
  - outputs/stage3_audit/edges.json  (5 jenis sisi, dengan flag berarah).

DIRAPIKAN 2026-09-14. Tiga hal yang membuatnya tampil beda sendiri dari figur lain:
  * Ukuran huruf dipatok manual (9, 10, 8, 8.5 pt) sehingga jauh lebih besar daripada
    figur lain yang mengikuti _style. Setelah diperkecil ke kolom jurnal, figur ini
    tampak seperti hasil perbesaran. Sekarang ukurannya mengikuti _style.
  * Label kategori panel (b) digambar manual pada koordinat DATA di x = -0.4, jadi
    menjulur ke luar sumbu tanpa diketahui mesin tata letak, dan nyaris menabrak
    label colorbar panel (a). Sekarang memakai label tik-y yang sesungguhnya.
  * Kotak penanda blok satu-grade memakai hijau #1a9850 yang di luar palet.

Panel (a): heatmap SDM_norm atas empat kelas lesi. Pasangan satu-grade yang berbagi
kelompok keparahan ICDR (MA-HE) berjarak 0.00; pasangan lintas-grade (MA-Hard_EX =
0.50, MA-Soft_EX = 1.00) berjarak lebih besar. Struktur sadar-grade inilah yang
memberi SDCA daya pisah biaya-klinisnya.
Panel (b): INVENTARIS lima jenis sisi klinis beserta jumlahnya. Panjang batang di
sini adalah cacah, BUKAN kepentingan: jumlahnya nyaris seluruhnya ditentukan
kelengkapan skema (4 kelas x 5 zona = 20 sisi spatial_distribution; C(4,2) = 6 sisi
severity_transition dan co_occurrence). Batang terpanjang justru milik
spatial_distribution, yang keseluruhan bobot awalnya seragam 1.0, sedangkan jenis
sisi yang benar-benar dibaca SDCA adalah severity_transition dengan bobot berjenjang
0.05 sampai 2.0. Karena itu severity_transition diberi warna biru: panel (a) adalah
baris itu setelah diskalakan Eq. 1.

Jalankan dari root repo:
    python journal/figures/scripts/fig_kg_matrix.py

Keluaran: journal/figures/fig_kg_matrix.{png,pdf}
"""
import collections
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import C, apply_style, panel_tags_bottom, save  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
SDM_JSON = ROOT / "outputs/stage3_audit/sdm_norm_clinical_prior.json"
EDGES_JSON = ROOT / "outputs/stage3_audit/edges.json"

CLASSES = ["MA", "HE", "Hard_EX", "Soft_EX"]
LABELS = ["MA", "HE", "Hard\nEX", "Soft\nEX"]


def load_sdm():
    raw = json.loads(SDM_JSON.read_text())
    idx = {c: i for i, c in enumerate(CLASSES)}
    M = np.zeros((4, 4))
    for key, v in raw.items():
        a, b = key.split("__")
        if a in idx and b in idx:
            M[idx[a]][idx[b]] = v["norm"]
            M[idx[b]][idx[a]] = v["norm"]
    return M


def load_edges():
    edges = json.loads(EDGES_JSON.read_text())
    by_type = collections.defaultdict(lambda: {"n": 0, "directed": False})
    for e in edges:
        t = e["edge_type"]
        by_type[t]["n"] += 1
        if e.get("directed"):
            by_type[t]["directed"] = True
    path_dir = [(e["src_name"], e["dst_name"], e.get("init_weight")) for e in edges
                if e["edge_type"] == "pathophysiological_mechanism" and e.get("directed")]
    return dict(by_type), len(edges), path_dir


def panel_heatmap(ax, M):
    im = ax.imshow(M, cmap="YlOrRd", vmin=0, vmax=1, aspect="equal")
    ax.set_xticks(range(4)); ax.set_xticklabels(LABELS)
    ax.set_yticks(range(4)); ax.set_yticklabels(LABELS)
    for i in range(4):
        for j in range(4):
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.6,
                    color="white" if v > 0.62 else C["text"])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\mathrm{SDM_{norm}}$ (author-defined lesion-pair prior)", fontsize=7.6)
    cbar.ax.tick_params(labelsize=7.0)
    cbar.outline.set_visible(False)


def panel_edges(ax, by_type, n_total):
    order = ["severity_transition", "pathophysiological_mechanism",
             "morphological_confusability", "spatial_distribution",
             "co_occurrence_pattern"]
    names = {"severity_transition": "severity transition",
             "pathophysiological_mechanism": "pathophysiological",
             "morphological_confusability": "morphological",
             "spatial_distribution": "spatial distribution",
             "co_occurrence_pattern": "co-occurrence"}
    order = [t for t in order if t in by_type]
    counts = [by_type[t]["n"] for t in order]
    directed = [by_type[t]["directed"] for t in order]
    y = np.arange(len(order))[::-1]
    # Biru menandai jenis sisi yang dibaca SDCA dan menjadi isi panel (a).
    colors = [C["b4"] if t == "severity_transition" else C["neutral"] for t in order]
    ax.barh(y, counts, color=colors, edgecolor="none", height=0.58)
    for yi, c in zip(y, counts):
        ax.text(c + 0.5, yi, str(c), ha="left", va="center", fontsize=7.2)
    # Label kategori sebagai tik-y yang sebenarnya, bukan teks pada koordinat data:
    # dengan begitu mesin tata letak tahu lebarnya dan tidak ada yang menjulur menabrak
    # panel (a).
    ax.set_yticks(y)
    ax.set_yticklabels([names[t] + ("  (directed)" if d else "")
                        for t, d in zip(order, directed)])
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlim(0, max(counts) + 3)
    ax.set_xlabel(f"number of edges ({n_total} total)")
    ax.grid(axis="x"); ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.spines[["left"]].set_visible(False)


def main():
    M = load_sdm()
    by_type, n_total, path_dir = load_edges()
    # verifikasi ke konsol (anti-halusinasi)
    print("[fig_kg_matrix] SDM_norm loaded:")
    for i, ci in enumerate(CLASSES):
        for j, cj in enumerate(CLASSES):
            if i < j:
                print(f"    {ci:8}-{cj:8} = {M[i,j]:.2f}")
    print(f"[fig_kg_matrix] edges: {n_total} total, "
          f"types={ {k: v['n'] for k, v in by_type.items()} }")

    apply_style()
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.0, 2.9),
                                   gridspec_kw={"width_ratios": [1.0, 1.12]})
    panel_heatmap(axa, M)
    panel_edges(axb, by_type, n_total)
    # Ruang bawah disisakan lewat rect agar penanda panel tidak menempel label sumbu;
    # penanda dipasang setelah tata letak final karena posisinya dibaca dari sumbu.
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=1.6)
    panel_tags_bottom(fig, [axa, axb], ["(a)", "(b)"], y=0.015)
    save(fig, "fig_kg_matrix")
    if path_dir:
        print("[fig_kg_matrix] directed pathophysiological edges: "
              + ", ".join(f"{a}->{b} (w={w:g})" for a, b, w in path_dir))


if __name__ == "__main__":
    main()
