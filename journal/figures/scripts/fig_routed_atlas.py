#!/usr/bin/env python
"""fig_routed_atlas.py — atlas deteksi terutekan per kohort (CEG-Net).

KOREKSI 2026-09-15 — CACAT SERIUS PADA VERSI SEBELUMNYA.
Versi lama mewarnai kotak menurut kolom `tier_target` CSV governance. Kolom itu adalah
peta deterministik dari `category`, dan `category` diturunkan dari GROUND TRUTH
(tools/audit_stage3c_governance_targets.py baris 488-509). Jadi atlas berjudul
"routed detections" menampilkan kunci jawaban, bukan keluaran gate, dan tidak dapat
diproduksi pada gambar tanpa anotasi.

Versi ini membaca tier keluaran gate dari outputs/0506/tier_routing/{kohort}_gate_tiers.csv
(tools/make_tier_routing.py): keputusan retain pada tau* dari OIA-DDR valid dikali SDCA pada
delta 0.30, persis Algorithm 1. Kategori rujukan tidak dipakai untuk mewarnai.

Palet Okabe-Ito, aman buta warna (lihat TIER_COLOR):
  R1 Accepted     - hijau kebiruan (#009E73)
  R2 Provisional  - biru langit    (#56B4E9)
  R3 Contested    - jingga         (#E69F00)
  R4 Rejected     - ungu kemerahan (#CC79A7)

Keluaran:
  mode --grid_only (dipakai paper):
    journal/figures/fig_routed_atlas_combined.{png,pdf}   empat gambar teratas per kohort
    journal/figures/fig_routed_atlas_sample.{png,pdf}     empat gambar acak berseed
  mode penuh (per kohort):
    outputs/0506/tier_routing/routed_images/<kohort>/<image_id>.png
    outputs/0506/tier_routing/<kohort>_routing_manifest.json
    journal/figures/fig_routed_atlas_grid[_<kohort>].{png,pdf}

Perintah (dari root repo, setelah tools/make_tier_routing.py):
  python journal/figures/scripts/fig_routed_atlas.py --grid_only
  python journal/figures/scripts/fig_routed_atlas.py --grid_only --random_sample --seed 0
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import numpy as np

ROOT    = Path(__file__).resolve().parents[3]
FIG_DIR = Path(__file__).resolve().parents[1]
TIER_DIR = ROOT / "outputs/0506/tier_routing"

DATASETS = {
    "oiaddr": "OIA-DDR",
    "idrid":  "IDRiD",
    "tjdr":   "TJDR",
    "fgadr":  "FGADR",
}
# FGADR dipakai sebagai kohort eksternal zero-shot lewat split train-nya; kohort itu
# tidak membawa pembagian test/valid sendiri.
SPLITS = {"oiaddr": "test", "idrid": "test", "tjdr": "test", "fgadr": "train"}
# Konfigurasi deployed yang sama dengan Tabel 12: kohort eksternal memakai evsds_pss
# (PSS terukur), bukan evsds_v2sec yang PSS-nya placeholder pada kohort eksternal.
FEATURE_ROOT = {"oiaddr": "evsds_v2sec", "idrid": "evsds_pss",
                "tjdr": "evsds_pss", "fgadr": "evsds_pss"}
GRID_NCOLS = {k: 4 for k in DATASETS}

TIER_COLOR = {
    "R1": "#009E73",   # hijau kebiruan
    "R2": "#56B4E9",   # biru langit
    "R3": "#E69F00",   # jingga
    "R4": "#CC79A7",   # ungu kemerahan
}
TIER_LABEL = {
    "R1": "Accepted",
    "R2": "Provisional",
    "R3": "Contested",
    "R4": "Rejected",
}
TIER_ORDER = ["R1", "R2", "R3", "R4"]


# ── helpers ───────────────────────────────────────────────────────────────────

def _paths(ds_key: str):
    sp = SPLITS[ds_key]
    base = ROOT / "outputs/0506" / FEATURE_ROOT[ds_key] / ds_key / sp
    suffix = f"_{ds_key}" if ds_key != "oiaddr" else ""
    return dict(
        split        = sp,
        features_csv = base / f"governance_features_{sp}.csv",
        gate_tiers   = TIER_DIR / f"{ds_key}_gate_tiers.csv",
        cache_dir    = ROOT / "outputs/detector_cache/primary" / ds_key / sp,
        image_dir    = ROOT / "data" / ds_key / f"images/{sp}",
        out_dir      = TIER_DIR / "routed_images" / ds_key,
        manifest     = TIER_DIR / f"{ds_key}_routing_manifest.json",
        grid_png     = FIG_DIR / f"fig_routed_atlas_grid{suffix}.png",
    )


def load_gate_tiers(gate_csv: Path):
    """Baca tier gate per kotak. Kembalikan (tiers[image][bbox] = baris, stats per gambar)."""
    if not gate_csv.is_file():
        raise SystemExit(f"[atlas] {gate_csv} tidak ada; jalankan tools/make_tier_routing.py")
    rows: dict[str, dict[str, dict]] = defaultdict(dict)
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    with open(gate_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[row["image_id"]][row["bbox_id"]] = row
            counts[row["image_id"]][row["tier"]] += 1
    stats = [(img, dict(tc)) for img, tc in counts.items()]
    return rows, stats


def load_features(features_csv: Path) -> dict[str, dict[str, dict]]:
    data: dict[str, dict[str, dict]] = defaultdict(dict)
    with open(features_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            data[row["image_id"]][row["bbox_id"]] = row
    return data


def load_cache(cache_dir: Path, image_id: str) -> list[dict]:
    p = cache_dir / f"{image_id}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


def load_image(image_dir: Path, image_id: str):
    from PIL import Image
    for ext in (".jpg", ".jpeg", ".png", ".tif"):
        p = image_dir / f"{image_id}{ext}"
        if p.exists():
            return Image.open(p).convert("RGB")
    return None


def get_bbox(box: dict):
    raw = box.get("detector_raw") or box.get("r2a_raw") or {}
    return raw.get("bbox"), raw.get("score")


def draw_annotated(image_dir, image_id, tiers, boxes, badge: bool = True):
    """tiers: peta bbox_id -> baris gate_tiers (kolom 'tier' = keluaran gate)."""
    img = load_image(image_dir, image_id)
    if img is None:
        return None, {}
    W, H = img.size
    fig, ax = plt.subplots(1, 1, figsize=(W / 200, H / 200))
    ax.imshow(np.array(img))
    ax.axis("off")
    tier_counts: dict[str, int] = defaultdict(int)
    for box in boxes:
        row = tiers.get(box.get("bbox_id"))
        if row is None:
            continue
        bbox, _ = get_bbox(box)
        if not bbox:
            continue
        tier = row["tier"]
        col = TIER_COLOR.get(tier, "#888888")
        x1, y1, x2, y2 = [float(v) for v in bbox]
        ax.add_patch(Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            fill=False, edgecolor=col, linewidth=max(0.8, W / 800),
        ))
        tier_counts[tier] += 1
    # Badge sudut dan judul hanya untuk berkas per-gambar yang berdiri sendiri. Pada
    # mozaik gabungan keduanya ikut mengecil sampai tak terbaca, sementara hitungan tier
    # dan nama gambar sudah dicetak ulang oleh judul selnya sendiri.
    if badge:
        for i, t in enumerate(TIER_ORDER):
            ax.text(0.01, 0.99 - i * 0.05,
                    f"{t} {TIER_LABEL[t]}: {tier_counts.get(t, 0)}",
                    transform=ax.transAxes, fontsize=10.5, va="top",
                    color=TIER_COLOR[t],
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.7, lw=0))
        ax.set_title(image_id, fontsize=10.5, pad=2)
    fig.tight_layout(pad=0)
    return fig, dict(tier_counts)


def make_grid(per_image_stats, out_dir, grid_png, ds_name, split, ncols):
    from PIL import Image as PILImage
    per_image_stats = sorted(per_image_stats, key=lambda x: -x[1].get("R1", 0))
    nrows = math.ceil(len(per_image_stats) / ncols)
    fig_g, axes_g = plt.subplots(nrows, ncols,
                                  figsize=(ncols * 1.7, nrows * 1.65 + 0.8))
    axes_g = np.array(axes_g).ravel()
    for i, (image_id, tc) in enumerate(per_image_stats):
        ax = axes_g[i]
        thumb_p = out_dir / f"{image_id}.png"
        if thumb_p.exists():
            ax.imshow(np.array(PILImage.open(thumb_p).convert("RGB")))
        ax.axis("off")
        ax.set_title(
            f"R1:{tc.get('R1',0)} R2:{tc.get('R2',0)}\n"
            f"R3:{tc.get('R3',0)} R4:{tc.get('R4',0)}",
            fontsize=10.5, pad=1,
        )
    for ax in axes_g[len(per_image_stats):]:
        ax.axis("off")
    handles = [mpatches.Patch(color=TIER_COLOR[t], label=f"{t} {TIER_LABEL[t]}")
               for t in TIER_ORDER]
    fig_g.legend(handles=handles, loc="lower center", ncol=4,
                 fontsize=10.5, frameon=True, bbox_to_anchor=(0.5, 0.0))
    n_imgs = len(per_image_stats)
    n_det  = sum(sum(tc.values()) for _, tc in per_image_stats)
    fig_g.suptitle(
        f"CEG-Net Routed Detection Atlas - {ds_name} {split} "
        f"({n_imgs} images, {n_det} detections, sorted by R1 desc)",
        fontsize=12, fontweight="bold",
    )
    fig_g.tight_layout(rect=[0, 0.025, 1, 0.97])
    fig_g.savefig(grid_png, dpi=120, bbox_inches="tight")
    fig_g.savefig(grid_png.with_suffix(".pdf"), dpi=120, bbox_inches="tight")
    plt.close(fig_g)


def make_combined_grid(all_ds_data, out_path):
    """Mozaik gabungan, satu baris per kohort.

    DIRAPIKAN 2026-09-14:
      * Judul di dalam gambar dibuang; caption LaTeX yang mengerjakannya.
      * Nama kohort dicetak sekali sebagai label baris di kiri, bersama penanda panelnya.
      * Sumbu ditambatkan ke sisi atas, supaya judul sel tidak turun seperti anak tangga.
      * Sel 1.73 in: textwidth elsarticle preprint hanya 5.40 in, dan dengan sel ini PDF
        selebar 6.92 in tercetak pada skala 0.78.
      * Kolomnya EMPAT: tinggi CETAK hanya ditentukan rasio sisi, dan mozaik tiga kolom
        untuk empat kohort tercetak 8.28 in, melampaui textheight 7.60 in.
    """
    from PIL import Image as PILImage
    nrows = len(all_ds_data)
    ncols = 4
    cell_w = 1.73

    # Tinggi tiap baris mengikuti rasio sisi fundus kohortnya. Daftar dipakai APA ADANYA:
    # pemilihannya dikerjakan pemanggil, untuk mode teratas maupun mode sampel acak.
    picks, row_asp = [], []
    for _ds_key, (per_image_stats, out_dir) in all_ds_data.items():
        top = list(per_image_stats)[:ncols]
        asp = []
        for image_id, _tc in top:
            thumb_p = out_dir / f"{image_id}.png"
            if thumb_p.exists():
                with PILImage.open(thumb_p) as im:
                    asp.append(im.height / im.width)
        picks.append((top, out_dir))
        row_asp.append(max(asp) if asp else 1.0)

    # +0.47 in per baris untuk judul TIGA baris, +0.38 in untuk legenda di bawah.
    fig_h = sum(cell_w * a + 0.47 for a in row_asp) + 0.38
    fig_g, axes_g = plt.subplots(nrows, ncols, figsize=(ncols * cell_w, fig_h),
                                 gridspec_kw={"height_ratios": row_asp})
    if nrows == 1:
        axes_g = np.expand_dims(axes_g, axis=0)

    row_axes = []
    for row_idx, (sorted_stats, out_dir) in enumerate(picks):
        for col_idx in range(ncols):
            ax = axes_g[row_idx, col_idx]
            ax.axis("off")
            ax.set_anchor("N")
            if col_idx >= len(sorted_stats):
                continue
            image_id, tc = sorted_stats[col_idx]
            thumb_p = out_dir / f"{image_id}.png"
            if thumb_p.exists():
                ax.imshow(np.array(PILImage.open(thumb_p).convert("RGB")))
            ax.set_title(
                f"{image_id}\n"
                f"R1 {tc.get('R1',0)}   R2 {tc.get('R2',0)}\n"
                f"R3 {tc.get('R3',0)}   R4 {tc.get('R4',0)}",
                fontsize=7.9, pad=2.5, linespacing=1.25,
            )
        row_axes.append(axes_g[row_idx, 0])

    handles = [mpatches.Patch(color=TIER_COLOR[t], label=f"{t} {TIER_LABEL[t]}")
               for t in TIER_ORDER]
    fig_g.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.6,
                 frameon=False, handlelength=1.0, columnspacing=1.6,
                 bbox_to_anchor=(0.5, 0.004))
    fig_g.tight_layout(rect=[0.04, 0.045, 1, 1], h_pad=2.4)

    # Label baris ditulis SETELAH tata letak final: posisinya dibaca dari sumbu.
    for (ds_key, _), ax0, tag in zip(all_ds_data.items(), row_axes, "abcdefgh"):
        box = ax0.get_position()
        fig_g.text(0.014, (box.y0 + box.y1) / 2.0, f"({tag})  {DATASETS[ds_key]}",
                   rotation=90, ha="center", va="center",
                   fontsize=9.2, fontweight="bold")

    # Ditulis ke KEDUA folder keluaran, seperti _style.save().
    for d in (out_path.parent, ROOT / "_FINAL_Latex/figures"):
        d.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            fig_g.savefig(d / f"{out_path.stem}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig_g)
    print(f"  Combined grid -> {out_path.stem}.png/.pdf (journal/figures + _FINAL_Latex/figures)")


def rebuild_grid_only(random_sample: bool = False, seed: int = 0,
                      out_stem: str = "fig_routed_atlas_combined"):
    """Bangun mozaik gabungan saja, tanpa merender seluruh kohort.

    random_sample=False memilih empat gambar dengan deteksi R1 terbanyak (atlas utama).
    random_sample=True menarik empat gambar acak berseed per kohort, untuk atlas pelengkap
    yang memeriksa apakah pola tier bertahan di luar pemilihan yang menguntungkan itu.
    """
    import random
    all_ds_data = {}
    for ds_key in DATASETS:
        p = _paths(ds_key)
        tiers, stats = load_gate_tiers(p["gate_tiers"])
        if random_sample:
            rng = random.Random(f"{seed}:{ds_key}")
            top = rng.sample(sorted(stats), k=min(4, len(stats)))
        else:
            top = sorted(stats, key=lambda x: (-x[1].get("R1", 0), x[0]))[:4]
        p["out_dir"].mkdir(parents=True, exist_ok=True)
        for image_id, _tc in top:
            boxes = load_cache(p["cache_dir"], image_id)
            fig, _ = draw_annotated(p["image_dir"], image_id,
                                    tiers.get(image_id, {}), boxes, badge=False)
            if fig is None:
                print(f"  [warn] gambar tidak ada: {image_id}")
                continue
            fig.savefig(p["out_dir"] / f"{image_id}.png", dpi=150,
                        bbox_inches="tight")
            plt.close(fig)
        mode = "acak" if random_sample else "teratas"
        print(f"[atlas:{ds_key}] {len(top)} gambar {mode}:")
        for image_id, tc in top:
            print(f"    {image_id}: " + "  ".join(f"{t}={tc.get(t, 0)}" for t in TIER_ORDER))
        all_ds_data[ds_key] = (top, p["out_dir"])
    make_combined_grid(all_ds_data, FIG_DIR / f"{out_stem}.png")


# ── per-dataset runner ────────────────────────────────────────────────────────

def run_dataset(ds_key: str):
    ds_name = DATASETS[ds_key]
    p = _paths(ds_key)
    p["out_dir"].mkdir(parents=True, exist_ok=True)

    tiers, _stats = load_gate_tiers(p["gate_tiers"])
    features = load_features(p["features_csv"])
    image_ids = sorted(tiers.keys())
    total = len(image_ids)
    n_det_total = sum(len(v) for v in tiers.values())
    print(f"\n[atlas:{ds_key}] {total} images, {n_det_total} detections")

    manifest: dict[str, dict] = {}
    per_image_stats: list[tuple[str, dict]] = []

    for idx, image_id in enumerate(image_ids, 1):
        tier_rows = tiers[image_id]
        ev = features.get(image_id, {})
        boxes = load_cache(p["cache_dir"], image_id)

        # manifest: keluaran gate dipisah tegas dari kategori rujukan
        manifest[image_id] = {
            bid: {
                "tier":       row["tier"],
                "decision":   row["decision"],
                "g":          float(row["g"]),
                "pred_class": ev.get(bid, {}).get("pred_class", ""),
                "pred_score": float(ev.get(bid, {}).get("pred_score") or 0),
                "ev_sas":     float(ev.get(bid, {}).get("ev_sas") or 0),
                "ev_sdca":    float(ev.get(bid, {}).get("ev_sdca") or 0),
                "ev_aps":     float(ev.get(bid, {}).get("ev_aps") or 0),
                "ev_pss":     float(ev.get(bid, {}).get("ev_pss") or 0),
                "reference_category": row["category"],
            }
            for bid, row in tier_rows.items()
        }

        fig, tc = draw_annotated(p["image_dir"], image_id, tier_rows, boxes)
        if fig is not None:
            fig.savefig(p["out_dir"] / f"{image_id}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            per_image_stats.append((image_id, tc))
        else:
            print(f"  [warn] image not found: {image_id}")
            per_image_stats.append((image_id, {}))

        if idx % 25 == 0 or idx == total:
            print(f"  [{idx}/{total}] annotated")

    p["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  manifest -> {p['manifest'].name}")

    make_grid(per_image_stats, p["out_dir"], p["grid_png"],
              ds_name, p["split"], GRID_NCOLS.get(ds_key, 12))
    print(f"  grid     -> {p['grid_png'].name}")

    all_t: dict[str, int] = defaultdict(int)
    for _, tc in per_image_stats:
        for t, c in tc.items():
            all_t[t] += c
    print(f"  Tier summary ({n_det_total} detections):")
    for t in TIER_ORDER:
        n = all_t.get(t, 0)
        print(f"    {t} {TIER_LABEL[t]:12s}: {n:6d} ({100*n/max(n_det_total,1):.1f}%)")

    return per_image_stats, p["out_dir"]


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS), default=None,
                    help="Run only this dataset (default: all four)")
    ap.add_argument("--grid_only", action="store_true",
                    help="Bangun mozaik gabungan saja dari berkas tier gate")
    ap.add_argument("--random_sample", action="store_true",
                    help="Pilih empat gambar ACAK per kohort, bukan empat teratas "
                         "menurut cacah R1 (atlas pelengkap)")
    ap.add_argument("--seed", type=int, default=0,
                    help="Seed pemilihan untuk --random_sample")
    ap.add_argument("--out_stem", default=None,
                    help="Nama dasar berkas keluaran mozaik")
    args = ap.parse_args()

    if args.grid_only:
        stem = args.out_stem or ("fig_routed_atlas_sample" if args.random_sample
                                 else "fig_routed_atlas_combined")
        rebuild_grid_only(random_sample=args.random_sample, seed=args.seed,
                          out_stem=stem)
        print("\n[atlas] Selesai (grid_only).")
        return

    targets = [args.dataset] if args.dataset else list(DATASETS)
    all_ds_data = {}
    for ds in targets:
        per_image_stats, out_dir = run_dataset(ds)
        all_ds_data[ds] = (sorted(per_image_stats, key=lambda x: (-x[1].get("R1", 0), x[0])),
                           out_dir)

    if len(all_ds_data) == len(DATASETS):
        make_combined_grid(all_ds_data, FIG_DIR / "fig_routed_atlas_combined.png")

    print("\n[atlas] All done.")


if __name__ == "__main__":
    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        print("[atlas] ERROR: Pillow not installed. Run: pip install pillow")
        sys.exit(1)
    main()
