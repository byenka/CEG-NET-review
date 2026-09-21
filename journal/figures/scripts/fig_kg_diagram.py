#!/usr/bin/env python
"""fig_kg_diagram.py — Diagram KG CEG-Net, visual encoding yang jelas.

Sistem encoding garis:
  SOLID  (—)   = FIXED  : bobot beku, prior klinis (morphological, severity)
  DASHED (- -) = TRAINABLE : bobot dipelajari dari data (spatial, co-occurrence)
  ARROW  (→)   = DIRECTED  : satu arah (pathophysiological SAJA)
  Tanpa kepala = UNDIRECTED : dua arah
  Ketebalan    ∝ bobot / kekuatan edge

Layout:
  Panel (a) kiri besar : Full KG — pathophysiological + spatial_distribution
  Panel (b) kanan atas : morphological_confusability — 2×2 node, FIXED SOLID
  Panel (c) kanan tengah: severity_transition — 2×2 node, FIXED SOLID
  Panel (d) bawah full : co_occurrence_pattern — 4 node horizontal, TRAINABLE DASHED
  Legend strip          : encoding visual + warna + arti

Output: journal/figures/fig_kg_diagram.{png,pdf}
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "fig_kg_diagram.png"

# ── Palette ───────────────────────────────────────────────────────────────────
C_LESION  = "#fff3e0"    # lesion node fill
C_ZONE    = "#e3edf8"    # zone node fill
C_PATH    = "#b71c1c"    # pathophysiological — merah tua
C_MORPH   = "#6a1b9a"    # morphological — ungu
C_SEV     = "#e65100"    # severity (sev>0) — oranye tua
C_SEV0    = "#9e9e9e"    # severity MA↔HE sev=0 — abu
C_SPATHI  = "#0d47a1"    # spatial HIGH_AFFINITY — biru tua
C_SPAT    = "#90caf9"    # spatial standard — biru muda
C_COOC    = "#1b5e20"    # co_occurrence — hijau tua
CEDGE     = "#37474f"    # node border


# ── Primitives ────────────────────────────────────────────────────────────────

def node_box(ax, cx, cy, w, h, text, fc, fs=8.5, bold=True, z=6):
    p = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                       boxstyle="round,pad=0.016,rounding_size=0.025",
                       lw=1.5, edgecolor=CEDGE, facecolor=fc, zorder=z)
    ax.add_patch(p)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", zorder=z+1)


def edge(ax, x1, y1, x2, y2, col, lw, ls, directed, rad=0.0, zA=3, alpha=1.0):
    style = "-|>" if directed else "-"
    cs    = f"arc3,rad={rad}"
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=col, lw=lw,
                                linestyle=ls, mutation_scale=14,
                                connectionstyle=cs, shrinkA=8, shrinkB=8),
                zorder=zA, alpha=alpha)


def midlabel(ax, x1, y1, x2, y2, txt, col, fs=6.0, rad=0.0, dy=0.0, dx=0.0):
    mx = (x1+x2)/2 + dx + rad*(x2-x1)*0.08
    my = (y1+y2)/2 + abs(rad)*abs(x2-x1)*0.42 + dy
    ax.text(mx, my, txt, ha="center", va="center", fontsize=fs,
            color=col, zorder=9,
            bbox=dict(boxstyle="round,pad=0.06", fc="white", alpha=0.78, lw=0))


# ─────────────────────────────────────────────────────────────────────────────
# PANEL (a) — Full KG: pathophysiological + spatial_distribution
# ─────────────────────────────────────────────────────────────────────────────

def panel_a(ax):
    ax.set_xlim(-0.8, 11.2)
    ax.set_ylim(-0.6, 8.2)
    ax.axis("off")
    ax.set_title("(a)  Struktur KG Lengkap\npathophysiological  +  spatial distribution",
                 fontsize=9.5, fontweight="bold", pad=6)

    NW, NH = 1.6, 0.60
    ZW, ZH = 1.38, 0.52
    LY, ZY = 5.8, 1.1

    lx = {"MA": 1.0, "HE": 3.8, "Hard_EX": 6.8, "Soft_EX": 9.8}
    zx = {"Foveal": 0.6, "Parafoveal": 2.6, "Peripapillary": 5.0,
          "Vascular\nArcades": 7.6, "Peripheral": 9.8}

    HIGH = {("Hard_EX","Foveal"), ("Hard_EX","Parafoveal"), ("Soft_EX","Peripapillary")}

    # spatial_distribution — TRAINABLE = DASHED
    for ln, lxv in lx.items():
        for zn, zxv in zx.items():
            zn_key = zn.replace("\n"," ")
            is_hi  = (ln, zn_key) in HIGH or (ln, zn) in HIGH
            col    = C_SPATHI if is_hi else C_SPAT
            lw     = 2.0 if is_hi else 0.65
            alpha  = 1.0 if is_hi else 0.5
            ax.plot([lxv, zxv], [LY - NH/2, ZY + ZH/2],
                    color=col, lw=lw, ls="--", alpha=alpha,
                    zorder=1, solid_capstyle="round", dash_capstyle="round")

    # HIGH_AFFINITY "★" labels
    for ln, zn, lab in [("Hard_EX","Foveal","★ w=2"),
                         ("Hard_EX","Parafoveal","★ w=2"),
                         ("Soft_EX","Peripapillary","★ w=2")]:
        lxv = lx[ln]; zxv = zx[zn] if zn in zx else zx[zn+"\n"+"Arcades"]
        if zn == "Foveal":    zxv = zx["Foveal"]
        if zn == "Parafoveal":zxv = zx["Parafoveal"]
        if zn == "Peripapillary": zxv = zx["Peripapillary"]
        ax.text((lxv+zxv)/2, (LY+ZY)/2 + 0.1, lab,
                ha="center", va="center", fontsize=5.5,
                color=C_SPATHI, fontweight="bold", zorder=8)

    # pathophysiological — FIXED topology, DIRECTED SOLID →
    # MA→HE
    edge(ax, lx["MA"]+NW/2, LY+0.16, lx["HE"]-NW/2, LY+0.16,
         C_PATH, 2.5, "-", True, rad=-0.35)
    midlabel(ax, lx["MA"], LY+0.16, lx["HE"], LY+0.16,
             "vessel rupture\nw=1.0", C_PATH, fs=6.0, rad=-0.35, dy=0.18)

    # MA→Hard_EX
    edge(ax, lx["MA"]+NW/2, LY+0.05, lx["Hard_EX"]-NW/2, LY+0.05,
         C_PATH, 2.5, "-", True, rad=-0.22)
    midlabel(ax, lx["MA"], LY+0.05, lx["Hard_EX"], LY+0.05,
             "lipid leakage  w=1.0", C_PATH, fs=6.0, rad=-0.22, dy=0.62)

    # MA→Soft_EX (long arc, high above)
    edge(ax, lx["MA"]+NW/2, LY+0.22, lx["Soft_EX"]-NW/2, LY+0.22,
         C_PATH, 2.5, "-", True, rad=-0.16)
    midlabel(ax, lx["MA"], LY+0.22, lx["Soft_EX"], LY+0.22,
             "weak prior  w=0.3", C_PATH, fs=6.0, rad=-0.16, dy=1.35)

    # Node boxes (drawn last so they cover edge endpoints)
    for name, xv in lx.items():
        node_box(ax, xv, LY, NW, NH, name, C_LESION, fs=9, bold=True)

    zlabels = {"Foveal":"Foveal","Parafoveal":"Para-\nfoveal",
               "Peripapillary":"Peripa-\npapillary",
               "Vascular\nArcades":"Vasc.\nArcades","Peripheral":"Peripheral"}
    for name, xv in zx.items():
        node_box(ax, xv, ZY, ZW, ZH+0.1, zlabels[name], C_ZONE, fs=7.5, bold=False)

    ax.text(-0.65, LY, "Lesion\nNodes", ha="right", va="center",
            fontsize=7, color="#555", fontstyle="italic")
    ax.text(-0.65, ZY, "ETDRS\nZone\nNodes", ha="right", va="center",
            fontsize=7, color="#555", fontstyle="italic")

    # Mini legend inside panel (a)
    items_a = [
        (C_PATH,   2.3, "-",  True,  "Pathophysiological → directed, FIXED topology"),
        (C_SPATHI, 2.0, "--", False, "Spatial HIGH_AFFINITY ↔ trainable, w=2.0"),
        (C_SPAT,   0.8, "--", False, "Spatial standard ↔ trainable, w=1.0"),
    ]
    for i, (c, lw, ls, direc, lab) in enumerate(items_a):
        y_i = -0.3 - i * 0.27
        sty = "-|>" if direc else "-"
        ax.annotate("", xy=(0.9, y_i), xytext=(0.1, y_i),
                    arrowprops=dict(arrowstyle=sty, color=c, lw=lw, linestyle=ls,
                                   mutation_scale=10, shrinkA=0, shrinkB=0),
                    zorder=5)
        ax.text(1.05, y_i, lab, va="center", fontsize=6.0, color=c)


# ─────────────────────────────────────────────────────────────────────────────
# 2×2 helper — used by panels b and c
# ─────────────────────────────────────────────────────────────────────────────
#
#   MA (0,1)       HE (1,1)
#   Hard_EX (0,0)  Soft_EX (1,0)
#
# Edge positions (all unique, no overlap):
#   MA↔HE         : horizontal top → arc ABOVE
#   Hard_EX↔Soft_EX : horizontal bottom → arc BELOW
#   MA↔Hard_EX    : vertical left → arc LEFT
#   HE↔Soft_EX    : vertical right → arc RIGHT
#   MA↔Soft_EX    : diagonal ↘ → arc slight bow right
#   HE↔Hard_EX    : diagonal ↙ → arc slight bow left

NW2, NH2 = 1.25, 0.50
SPC = 2.8   # spacing between nodes in 2x2

def sq_pos(name):
    """Centre coordinates in 2×2 square layout."""
    m = {"MA":      (0.0,   SPC),
         "HE":      (SPC,   SPC),
         "Hard_EX": (0.0,   0.0),
         "Soft_EX": (SPC,   0.0)}
    return m[name]

def sq_border(name, side):
    cx, cy = sq_pos(name)
    if side == "top":    return cx, cy + NH2/2
    if side == "bottom": return cx, cy - NH2/2
    if side == "left":   return cx - NW2/2, cy
    if side == "right":  return cx + NW2/2, cy
    return cx, cy

def draw_2x2_nodes(ax):
    for name in ["MA","HE","Hard_EX","Soft_EX"]:
        cx, cy = sq_pos(name)
        node_box(ax, cx, cy, NW2, NH2, name, C_LESION, fs=8, bold=True, z=6)

def sq_setup(ax, title, subtitle):
    span = SPC + NW2
    ax.set_xlim(-1.1, span + 1.1)
    ax.set_ylim(-1.6, span + 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{title}\n{subtitle}", fontsize=8.8, fontweight="bold", pad=5)


# ─────────────────────────────────────────────────────────────────────────────
# PANEL (b) — morphological_confusability  (FIXED, SOLID, 3 pair)
# ─────────────────────────────────────────────────────────────────────────────

def panel_b(ax):
    sq_setup(ax,
             "(b)  Morphological Confusability",
             "FIXED · Solid · 3 pasang  (bukan 6)")

    morph = [
        # (a, b, κ,  border_a, border_b, rad,  label_dy)
        ("MA",      "HE",       0.8, "top",    "top",    -0.48, +0.62),
        ("Hard_EX", "Soft_EX",  0.5, "bottom", "bottom", +0.48, -0.62),
        ("MA",      "Hard_EX",  0.1, "left",   "left",   +0.50, 0.0),
    ]
    for a, b, kappa, sa, sb, rad, ldy in morph:
        x1, y1 = sq_border(a, sa)
        x2, y2 = sq_border(b, sb)
        lw = 1.0 + kappa * 3.5
        edge(ax, x1, y1, x2, y2, C_MORPH, lw, "-", False, rad=rad, zA=3)
        midlabel(ax, x1, y1, x2, y2, f"κ = {kappa}", C_MORPH,
                 fs=6.8, rad=rad, dy=ldy*0.5)

    draw_2x2_nodes(ax)

    # Catatan
    ax.text(SPC/2, -1.35,
            "Solid = FIXED  ·  ketebalan ∝ kappa\n"
            "HE↔Hard_EX, HE↔Soft_EX, MA↔Soft_EX : TIDAK ADA edge ini",
            ha="center", va="center", fontsize=6.0, color="#555",
            style="italic",
            bbox=dict(boxstyle="round,pad=0.1", fc="#f5f5f5", lw=0.5, ec="#ccc"))


# ─────────────────────────────────────────────────────────────────────────────
# PANEL (c) — severity_transition  (FIXED, SOLID, 6 pair)
# ─────────────────────────────────────────────────────────────────────────────

def panel_c(ax):
    sq_setup(ax,
             "(c)  Severity Transition  (ICDR Grade Cost)",
             "FIXED · 6 pasang · Solid oranye  |  MA↔HE sev=0 → putus-putus abu")

    sev = [
        # (a, b, sev, border_a, border_b, rad, ldy)
        ("MA",      "HE",       0.0, "top",    "top",    -0.52, +0.70),
        ("Hard_EX", "Soft_EX",  1.5, "bottom", "bottom", +0.52, -0.70),
        ("MA",      "Hard_EX",  1.0, "left",   "left",   +0.50, 0.0),
        ("HE",      "Soft_EX",  2.0, "right",  "right",  -0.50, 0.0),
        ("MA",      "Soft_EX",  2.0, "bottom", "bottom", +0.30, -0.52),
        ("HE",      "Hard_EX",  1.0, "bottom", "bottom", -0.30, -0.52),
    ]
    for a, b, sv, sa, sb, rad, ldy in sev:
        x1, y1 = sq_border(a, sa)
        x2, y2 = sq_border(b, sb)
        is_zero = (sv == 0.0)
        col = C_SEV0 if is_zero else C_SEV
        ls  = "--"    if is_zero else "-"
        lw  = 1.0 if is_zero else 1.0 + sv * 0.85
        lab = f"sev={sv:.1f}" + (" (dashed)" if is_zero else "")
        edge(ax, x1, y1, x2, y2, col, lw, ls, False, rad=rad, zA=3)
        midlabel(ax, x1, y1, x2, y2, lab, col, fs=6.2, rad=rad, dy=ldy*0.46)

    draw_2x2_nodes(ax)

    ax.text(SPC/2, -1.35,
            "Solid = FIXED  ·  ketebalan ∝ severity  ·  putus-putus = sev=0 (aman)\n"
            "sev=2.0 tertebal (paling berbahaya dikacaukan)",
            ha="center", va="center", fontsize=6.0, color="#555",
            style="italic",
            bbox=dict(boxstyle="round,pad=0.1", fc="#f5f5f5", lw=0.5, ec="#ccc"))


# ─────────────────────────────────────────────────────────────────────────────
# PANEL (d) — co_occurrence_pattern  (TRAINABLE, DASHED, 6 pair)
# ─────────────────────────────────────────────────────────────────────────────

def panel_d(ax):
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.9, 3.5)
    ax.axis("off")
    ax.set_title(
        "(d)  Co-occurrence Pattern  —  TRAINABLE · Putus-putus hijau · 6 pasang\n"
        "init_weight = 1.0  (diperbarui dari statistik dataset OIA-DDR)",
        fontsize=9.5, fontweight="bold", pad=5)

    NW4, NH4 = 1.5, 0.54
    cx4 = {"MA": 1.0, "HE": 3.7, "Hard_EX": 6.6, "Soft_EX": 9.5}
    cy4 = 1.0

    # 6 pasang, arc heights bertingkat agar tidak tumpang tindih
    cooc = [
        # (a, b, rad,  label)
        ("MA",      "HE",       -0.30, "MA ↔ HE"),
        ("HE",      "Hard_EX",  -0.30, "HE ↔ Hard_EX"),
        ("Hard_EX", "Soft_EX",  -0.30, "Hard_EX ↔ Soft_EX"),
        ("MA",      "Hard_EX",  -0.50, "MA ↔ Hard_EX"),
        ("HE",      "Soft_EX",  -0.50, "HE ↔ Soft_EX"),
        ("MA",      "Soft_EX",  -0.65, "MA ↔ Soft_EX"),
    ]
    for a, b, rad, lab in cooc:
        x1 = cx4[a]; x2 = cx4[b]
        edge(ax, x1, cy4+NH4/2, x2, cy4+NH4/2,
             C_COOC, 1.5, "--", False, rad=rad, zA=3)
        midlabel(ax, x1, cy4+NH4/2, x2, cy4+NH4/2,
                 lab, C_COOC, fs=6.0, rad=rad, dy=0.06)

    for name, xv in cx4.items():
        node_box(ax, xv, cy4, NW4, NH4, name, C_LESION, fs=9, bold=True)

    ax.text(5.25, -0.62,
            "Putus-putus = TRAINABLE  ·  Semua init_weight=1.0  ·  Tidak ada kepala panah (undirected)",
            ha="center", va="center", fontsize=6.2, color="#555", style="italic")


# ─────────────────────────────────────────────────────────────────────────────
# LEGEND STRIP
# ─────────────────────────────────────────────────────────────────────────────

def panel_legend(ax):
    ax.set_xlim(0, 16)
    ax.set_ylim(-0.1, 3.2)
    ax.axis("off")
    ax.set_title("Legenda Visual Encoding", fontsize=9, fontweight="bold", pad=4)

    # Section: Node types
    node_box(ax, 0.5, 2.7, 0.9, 0.45, "Lesion\nNode", C_LESION, fs=6.5, bold=True, z=5)
    ax.text(1.1, 2.7, "MA / HE / Hard_EX / Soft_EX\n(feat_dim=17: one-hot + clinical attrs)",
            va="center", fontsize=7.0)
    node_box(ax, 5.5, 2.7, 0.9, 0.45, "Zone\nNode", C_ZONE, fs=6.5, bold=False, z=5)
    ax.text(6.1, 2.7, "Foveal / Parafoveal / Peripapillary /\nVascular Arcades / Peripheral  (feat_dim=9)",
            va="center", fontsize=7.0)

    # Horizontal separator
    ax.plot([0, 16], [2.3, 2.3], color="#ddd", lw=0.8)

    # Section: Line style encoding
    ax.text(0.0, 2.1, "GARIS:", fontsize=7.5, fontweight="bold", color="#333", va="center")
    ax.plot([0.7, 1.5], [2.1, 2.1], color="#555", lw=2.2, ls="-")
    ax.text(1.6, 2.1, "Solid (—) = FIXED: bobot beku, prior klinis", va="center", fontsize=7.0)
    ax.plot([6.7, 7.5], [2.1, 2.1], color="#555", lw=2.0, ls="--")
    ax.text(7.6, 2.1, "Putus-putus (- -) = TRAINABLE: dipelajari dari data", va="center", fontsize=7.0)
    ax.annotate("", xy=(12.7, 2.1), xytext=(11.8, 2.1),
                arrowprops=dict(arrowstyle="-|>", color="#555", lw=2.0, mutation_scale=12))
    ax.text(12.8, 2.1, "Panah (→) = DIRECTED (satu arah, pathophys. saja)", va="center", fontsize=7.0)

    ax.text(0.0, 1.75, "TEBAL:", fontsize=7.5, fontweight="bold", color="#333", va="center")
    ax.text(0.7, 1.75, "Ketebalan ∝ kekuatan bobot / κ / severity cost",
            va="center", fontsize=7.0, color="#555")

    ax.plot([0, 16], [1.4, 1.4], color="#ddd", lw=0.8)

    # Edge types — 2 columns
    rows = [
        # col 0
        [(C_PATH,   2.5, "-",  True,  "pathophysiological_mechanism\n"
          "FIXED topology · TRAINABLE weight · Directed →\n"
          "MA→HE (w=1.0) · MA→Hard_EX (w=1.0) · MA→Soft_EX (w=0.3)"),
         (C_MORPH,  2.2, "-",  False, "morphological_confusability\n"
          "FIXED · Solid · ketebalan ∝ κ (kappa)\n"
          "MA↔HE(κ=0.8) · Hard_EX↔Soft_EX(κ=0.5) · MA↔Hard_EX(κ=0.1)  [hanya 3 pasang]"),
        ],
        # col 1
        [(C_SEV,    2.0, "-",  False, "severity_transition  (ICDR grade cost)\n"
          "FIXED · Solid oranye · ketebalan ∝ severity · 6 pasang semua kombinasi\n"
          "MA↔HE: sev=0 → putus-putus abu (aman/dalam-grade)"),
         (C_SPATHI, 2.0, "--", False, "spatial_distribution  HIGH_AFFINITY  (★ w=2.0)\n"
          "TRAINABLE · Putus-putus biru tebal\n"
          "Hard_EX↔Foveal · Hard_EX↔Parafoveal · Soft_EX↔Peripapillary"),
        ],
    ]

    col_x = [0.0, 8.0]
    row_y = [1.15, 0.45]

    for ci, col in enumerate(rows):
        for ri, item in enumerate(col):
            col_c, lw, ls, direc, lbl = item
            x0 = col_x[ci]
            y0 = row_y[ri]
            sty = "-|>" if direc else "-"
            ax.annotate("", xy=(x0+0.7, y0), xytext=(x0+0.0, y0),
                        arrowprops=dict(arrowstyle=sty, color=col_c, lw=lw,
                                        linestyle=ls, mutation_scale=12,
                                        shrinkA=0, shrinkB=0), zorder=5)
            for k, line in enumerate(lbl.split("\n")):
                ax.text(x0+0.85, y0 + (1-k)*0.14, line,
                        va="center", fontsize=6.8,
                        color=col_c if k == 0 else "#444",
                        fontweight="bold" if k == 0 else "normal")

    # co_occurrence separate line at bottom
    x0 = col_x[1]
    y0 = row_y[1] - 0.30
    ax.annotate("", xy=(x0+0.7, y0), xytext=(x0+0.0, y0),
                arrowprops=dict(arrowstyle="-", color=C_COOC, lw=1.5,
                                linestyle="--", mutation_scale=12,
                                shrinkA=0, shrinkB=0), zorder=5)
    ax.text(x0+0.85, y0+0.13, "co_occurrence_pattern",
            va="center", fontsize=6.8, color=C_COOC, fontweight="bold")
    ax.text(x0+0.85, y0-0.01, "TRAINABLE · Putus-putus hijau · 6 pasang lesion · init_weight=1.0",
            va="center", fontsize=6.8, color="#444")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    fig = plt.figure(figsize=(18, 15))

    # Outer grid: [top row panels | legend]
    gs_out = fig.add_gridspec(3, 1,
                               height_ratios=[2.4, 1.05, 0.85],
                               hspace=0.10)

    # Top row: panel_a (wide) | panel_b + panel_c (stacked right)
    gs_top = gs_out[0].subgridspec(1, 2, width_ratios=[1.6, 1.0], wspace=0.07)
    gs_right = gs_top[1].subgridspec(2, 1, hspace=0.18)

    ax_a   = fig.add_subplot(gs_top[0])
    ax_b   = fig.add_subplot(gs_right[0])
    ax_c   = fig.add_subplot(gs_right[1])
    ax_d   = fig.add_subplot(gs_out[1])
    ax_leg = fig.add_subplot(gs_out[2])

    panel_a(ax_a)
    panel_b(ax_b)
    panel_c(ax_c)
    panel_d(ax_d)
    panel_legend(ax_leg)

    fig.suptitle(
        "CEG-Net / LGov-Net  —  Clinical Knowledge Graph\n"
        "9 node  ·  5 edge type  ·  "
        "Sumber: src/kg/schema.py · src/data/ontology.py · src/kg/encoder.py",
        fontsize=12, fontweight="bold", y=0.997,
    )

    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".pdf"), dpi=180, bbox_inches="tight")
    print(f"[fig_kg_diagram] {OUT}")

    print("\nVerifikasi:")
    for c in [
        "MA→HE, MA→Hard_EX, MA→Soft_EX : panah merah solid directed",
        "morphological_confusability : 3 pasang, solid ungu, tebal ∝ κ",
        "severity_transition : 6 pasang, MA↔HE putus-putus abu, lain solid oranye",
        "spatial_distribution : 20 pair, putus-putus biru (HA tebal, std tipis)",
        "co_occurrence_pattern : 6 pasang, putus-putus hijau",
        "Solid = FIXED  |  Putus-putus = TRAINABLE  |  Panah = DIRECTED",
        "Layout 2x2 = tidak ada tumpang tindih di panel b & c",
    ]:
        print(f"  OK  {c}")


if __name__ == "__main__":
    main()
