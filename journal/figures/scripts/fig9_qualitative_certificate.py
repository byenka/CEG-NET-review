#!/usr/bin/env python
"""fig9_qualitative_certificate.py — figur auditabilitas (multi-contoh, multi-sertifikat).

KOREKSI 2026-09-15 — CACAT SERIUS PADA VERSI SEBELUMNYA.
Sertifikat lama mencetak tiga baris berjudul "Gating category", "Routing tier" dan
"Decision", seluruhnya dibaca dari kolom `category` dan `tier_target` CSV governance.
Kedua kolom itu diturunkan dari GROUND TRUTH: tools/audit_stage3c_governance_targets.py
baris 488-509 menghitung `category` dari `best_iou` terhadap kotak anotasi dan
kecocokan `pred_class == best_gt_class`, dan `tier_target` adalah peta deterministik
dari `category`. Keduanya adalah TARGET SUPERVISI, bukan keluaran sistem.

Naskah mendefinisikan gate sebagai g(x) pada Eq. gate, dengan RETAIN bila g(x) >= tau*.
Jadi figur auditabilitas — yang mewujudkan Kontribusi 3 — menampilkan kunci jawaban
dengan judul yang terbaca seperti keluaran sistem. Pada gambar tanpa anotasi ketiga
baris itu tidak dapat diproduksi sama sekali, sehingga sertifikat versi lama tidak bisa
diterbitkan dalam pemakaian nyata. Diperiksa: dari 12 kotak sampel, 5 di antaranya
keputusan gate yang sebenarnya BERTENTANGAN dengan baris "Decision" versi lama.

Versi ini memasang gate yang sesungguhnya dan memisahkan dua blok secara tegas:
  * "Gate output"          — g(x) dan keputusannya pada tau*, yang benar-benar dihitung
                             sistem dan tersedia tanpa anotasi.
  * "Reference (annotation)" — kategori dan tier dari anotasi, dinyatakan apa adanya
                             sebagai rujukan, bukan sebagai keluaran.

Gate dipasang persis seperti "KG-gate-MLP (ours)" pada
tools/eval_fp_reduction_benchmark.py: MLP satu lapis tersembunyi atas
[SAS, SDCA, APS, PSS] + skor detektor, dipasang pada split valid, dan tau* dipilih pada
titik kerja retensi-TP 0.80 di split valid (protokol yang sama dengan naskah).

Kotak-kotak deteksi diwarnai menurut KATEGORI gating dengan palet Okabe-Ito: keluarga
hijau-biru untuk RETAIN, keluarga jingga-ungu untuk SUPPRESS (lihat _COLOR).

Masukan (digabung lewat bbox_id dalam tiap gambar):
  --image_ids       daftar image id dipisah koma
  --image_dir       folder berisi <image_id>.jpg/.png
  --cache_dir       folder cache detektor (<image_id>.json = DAFTAR kotak)
  --features_csv    governance_features_*.csv split test
  --valid_csv       governance_features_valid.csv, dipakai memasang gate dan memilih
                    tau*. Bila tidak diberikan, blok "Gate output" dihilangkan dan
                    hanya blok rujukan yang dicetak.
  --certify_bbox_ids  satu atau dua bbox_id yang diberi sertifikat

Jalankan dari root repo (butuh beberapa menit: memuat dua CSV besar dan memasang gate):
  python journal/figures/scripts/fig9_qualitative_certificate.py \
    --image_ids 007-3748-200,007-1853-100 \
    --image_dir data/oiaddr/images/test \
    --cache_dir outputs/detector_cache/primary/oiaddr/test \
    --features_csv outputs/0506/evsds_v2sec/oiaddr/test/governance_features_test.csv \
    --valid_csv outputs/0506/evsds_v2sec/oiaddr/valid/governance_features_valid.csv \
    --certify_bbox_ids 007-3748-200_b0160,007-3748-200_b0161 \
    --out journal/figures/fig9_qualitative_certificate.png
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import OUT_DIRS, apply_style, panel_tags_bottom  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]

_RETAIN = {"TP_EXACT", "CLINICALLY_CLOSE_MISMATCH", "LOCALIZATION_WEAK"}
# Palet Okabe-Ito, sama dengan figur lain. Palet lama memakai hijau #2e7d32 dan
# merah #c62828 berdampingan: justru pasangan yang tidak terbaca oleh deuteranopia
# dan protanopia, padahal paper mengklaim seluruh figurnya aman buta warna.
_COLOR = {
    "TP_EXACT": "#009E73",                   # hijau kebiruan  (RETAIN)
    "CLINICALLY_CLOSE_MISMATCH": "#F0E442",  # kuning          (RETAIN, ditandai)
    "LOCALIZATION_WEAK": "#56B4E9",          # biru langit     (RETAIN)
    "LOW_CONF_CANDIDATE": "#E69F00",         # jingga          (SUPPRESS)
    "BACKGROUND_FP": "#D55E00",              # vermilion       (SUPPRESS)
    "UNSAFE_MISMATCH": "#CC79A7",            # ungu kemerahan  (SUPPRESS)
}
_LEGEND = [
    ("TP_EXACT", "RETAIN: TP exact"),
    ("CLINICALLY_CLOSE_MISMATCH", "RETAIN: clinically close"),
    ("LOCALIZATION_WEAK", "RETAIN: localization weak"),
    ("LOW_CONF_CANDIDATE", "SUPPRESS: low confidence"),
    ("BACKGROUND_FP", "SUPPRESS: background FP"),
    ("UNSAFE_MISMATCH", "SUPPRESS: unsafe mismatch"),
]
_GATE_FEATS = ("ev_sas", "ev_sdca", "ev_aps", "ev_pss")
_TP_RET_TARGET = 0.80


def _bbox(box):
    raw = box.get("detector_raw") or box.get("r2a_raw") or {}
    return raw.get("bbox"), raw.get("score")


def _load_image(image_dir, image_id):
    from PIL import Image
    for ext in (".jpg", ".jpeg", ".png", ".tif"):
        p = Path(image_dir) / f"{image_id}{ext}"
        if p.exists():
            return Image.open(p).convert("RGB")
    return None


def _ev_for_image(features_csv, image_id):
    ev = {}
    with open(features_csv, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("image_id") == image_id:
                ev[r.get("bbox_id")] = r
    return ev


def compute_gate(valid_csv: Path, test_csv: Path):
    """Pasang gate pada split valid, kembalikan (peta bbox_id -> g(x), tau*).

    Sengaja memakai fungsi benchmark apa adanya, supaya angka pada sertifikat berasal
    dari kode yang sama dengan yang menghasilkan tabel, bukan dari salinan terpisah
    yang bisa menyimpang diam-diam.
    """
    import numpy as np
    sys.path.insert(0, str(ROOT))
    from tools.eval_fp_reduction_benchmark import (
        apply_mlp, fit_mlp, load_csv, threshold_for_tp_ret,
    )

    valid = load_csv(valid_csv, "governance_retain")
    test = load_csv(test_csv, "governance_retain")
    Xv = np.stack([valid[f] for f in _GATE_FEATS] + [valid["pred_score"]], 1)
    Xt = np.stack([test[f] for f in _GATE_FEATS] + [test["pred_score"]], 1)
    mlp = fit_mlp(Xv, valid["label"].astype(np.float64), seed=0)
    tau = threshold_for_tp_ret(apply_mlp(Xv, mlp), valid["label"], _TP_RET_TARGET)
    g_test = apply_mlp(Xt, mlp)

    bids = []
    with open(test_csv, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            bids.append(r["bbox_id"])
    if len(bids) != len(g_test):
        raise SystemExit("[fig9] urutan baris CSV test tidak sejajar dengan gate")
    return {b: float(v) for b, v in zip(bids, g_test)}, float(tau)


# Algorithm 1: tier = keputusan gate x SDCA pada delta. Nilai delta sama dengan
# tools/make_tier_routing.py; SDM_norm hanya bernilai {0, 0.50, 0.75, 1.00}, sehingga setiap
# delta di (0, 0.50] memberi tier yang sama.
_DELTA = 0.30
_TIER_NAME = {"R1": "accepted", "R2": "provisional", "R3": "contested", "R4": "rejected"}
_PROB_COLS = (("prob_ma", "MA"), ("prob_he", "HE"),
              ("prob_hardex", "Hard_EX"), ("prob_softex", "Soft_EX"))


def _runner_up(row):
    """Kelas runner-up = argmax distribusi per-ROI di luar kelas prediksi."""
    pred = row.get("pred_class", "")
    best, best_p = "", -1.0
    for col, name in _PROB_COLS:
        if name == pred:
            continue
        try:
            p = float(row.get(col, ""))
        except (TypeError, ValueError):
            continue
        if p > best_p:
            best, best_p = name, p
    return best


def _tier(g, tau, sdca):
    hi = sdca >= _DELTA
    if g >= tau:
        return "R2" if hi else "R1"
    return "R3" if hi else "R4"


def render_certificate(ax, plt, row, score, g, tau, tag):
    """Satu panel sertifikat. Blok gate dan blok rujukan DIPISAH tegas.

    Blok evidence hanya memuat masukan yang benar-benar dikonsumsi gate B4
    ([SAS, SDCA, APS, PSS] + skor). Baris ev_sds dan cacah sumber versi lama dibuang
    karena gate sertifikat ini tidak memakainya. Baris "routing tier" rujukan juga dibuang:
    tier kini keluaran gate (Algorithm 1), bukan tier_target dari anotasi.
    """
    ax.axis("off")

    def gv(k):
        v = row.get(k, "")
        try:
            return f"{float(v):.3f}"
        except (TypeError, ValueError):
            return str(v)

    cat = row.get("category", "")
    ref_dec = "RETAIN" if cat in _RETAIN else "SUPPRESS"
    lines = [
        ("detection", f"{row.get('pred_class','?')}  score {score:.3f}"),
        ("", ""),
        ("Evidence vector", ""),
        ("  SAS  source agreement", gv("ev_sas")),
        ("  SDCA ICDR-aware conflict", gv("ev_sdca")),
        ("  runner-up class", _runner_up(row)),
        ("  APS  anatomical plausibility", gv("ev_aps")),
        ("  PSS  perturbation stability", gv("ev_pss")),
        ("", ""),
    ]
    if g is not None:
        gate_dec = "RETAIN" if g >= tau else "SUPPRESS"
        t = _tier(g, tau, float(row.get("ev_sdca") or 0.0))
        lines += [
            ("Gate output", ""),
            ("  g(x) retain probability", f"{g:.3f}"),
            (f"  decision at $\\tau^\\star$ = {tau:.3f}", gate_dec),
            (f"  tier ($\\delta$ = {_DELTA:.2f})", f"{t} {_TIER_NAME[t]}"),
            ("", ""),
        ]
    else:
        gate_dec = None
    lines += [
        ("Reference (annotation)", ""),
        ("  category", cat),
        ("  triage label", ref_dec),
    ]

    HEADERS = {"Evidence vector", "Gate output", "Reference (annotation)"}
    DEC_COLOR = {"RETAIN": _COLOR["TP_EXACT"], "SUPPRESS": _COLOR["BACKGROUND_FP"]}

    ax.text(0.02, 0.985, f"AUDIT CERTIFICATE {tag}", fontsize=9.0,
            fontweight="bold", transform=ax.transAxes, va="top")
    y = 0.915
    for k, v in lines:
        if not k:
            y -= 0.022
            continue
        if k in HEADERS:
            ax.text(0.02, y, k, fontsize=7.8, fontweight="bold",
                    transform=ax.transAxes, va="top")
            y -= 0.052
            continue
        bold = v in DEC_COLOR
        ax.text(0.03, y, k, fontsize=7.6, transform=ax.transAxes, va="top")
        ax.text(0.97, y, v, fontsize=7.6, family="monospace", ha="right",
                transform=ax.transAxes, va="top",
                color=DEC_COLOR.get(v, "black"),
                fontweight="bold" if bold else "normal")
        y -= 0.050
    ax.add_patch(plt.Rectangle((0.0, 0.0), 1.0, 1.0, fill=False,
                               edgecolor="#bbb", lw=1, transform=ax.transAxes))
    return gate_dec


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Figur sertifikat per-deteksi (multi-contoh, multi-sertifikat).")
    ap.add_argument("--image_ids", default=None, help="image id dipisah koma")
    ap.add_argument("--image_id", default=None, help="satu image id (kompatibilitas)")
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--features_csv", required=True, type=Path)
    ap.add_argument("--valid_csv", default=None, type=Path,
                    help="split valid untuk memasang gate dan memilih tau*")
    ap.add_argument("--certify_bbox_ids", default=None,
                    help="satu atau dua bbox_id yang diberi sertifikat")
    ap.add_argument("--certify_bbox_id", default=None, help="kompatibilitas")
    ap.add_argument("--certify_image_id", default=None)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    ids = [s.strip() for s in (args.image_ids or args.image_id or "").split(",")
           if s.strip()]
    if not ids:
        print("[fig9] beri --image_ids atau --image_id")
        return 1
    cert_ids = [s.strip() for s in
                (args.certify_bbox_ids or args.certify_bbox_id or "").split(",")
                if s.strip()]

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.patches import Patch, Rectangle
        apply_style()
    except Exception as exc:  # pragma: no cover
        print(f"[fig9] matplotlib/pillow tidak tersedia ({exc}).")
        return 2

    gate_map, tau = ({}, 0.0)
    if args.valid_csv is not None:
        print("[fig9] memasang gate pada split valid ...", flush=True)
        gate_map, tau = compute_gate(args.valid_csv, args.features_csv)
        print(f"[fig9] tau* = {tau:.6f} (retensi TP {_TP_RET_TARGET})", flush=True)
    else:
        print("[fig9] --valid_csv tidak diberikan: blok 'Gate output' dilewati")

    n_img = len(ids)
    n_cert = max(len(cert_ids), 1)
    n_panels = n_img + n_cert
    ncols = 2 if n_panels == 4 else n_panels
    nrows = -(-n_panels // ncols)
    fig, axes_grid = plt.subplots(nrows, ncols,
                                  figsize=(3.45 * ncols, 3.25 * nrows))
    axes = list(np.atleast_1d(axes_grid).ravel())
    for ax in axes[n_panels:]:
        ax.axis("off")

    # ── panel fundus ────────────────────────────────────────────────────────
    cert_rows = {}   # bbox_id -> (row, score, image_id, bbox)
    for ax, image_id in zip(axes[:n_img], ids):
        img = _load_image(args.image_dir, image_id)
        ax.axis("off")
        # Ditambatkan ke sisi atas: fundus berbeda rasio sisi, dan dengan penambatan
        # tengah judul tiap panel turun sendiri-sendiri seperti anak tangga.
        ax.set_anchor("N")
        if img is None:
            ax.set_title(f"{image_id}\n(image not found)", fontsize=8, color="red")
            continue
        ax.imshow(img)
        cache_f = Path(args.cache_dir) / f"{image_id}.json"
        if not cache_f.exists():
            ax.set_title(f"{image_id}\n(cache missing)", fontsize=8, color="red")
            continue
        boxes = json.loads(cache_f.read_text(encoding="utf-8"))
        ev = _ev_for_image(args.features_csv, image_id)
        n_ret = n_sup = 0
        for box in boxes:
            bid = box.get("bbox_id")
            row = ev.get(bid)
            if row is None:
                continue
            bbox, score = _bbox(box)
            if not bbox:
                continue
            cat = row.get("category", "")
            x1, y1, x2, y2 = [float(v) for v in bbox]
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                   edgecolor=_COLOR.get(cat, "#9e9e9e"),
                                   linewidth=1.4))
            n_ret += 1 if cat in _RETAIN else 0
            n_sup += 0 if cat in _RETAIN else 1
            if bid in cert_ids:
                cert_rows[bid] = (row,
                                  float(score or row.get("pred_score") or 0.0),
                                  image_id, (x1, y1, x2, y2))
        # Hitungan ini dari kategori RUJUKAN (anotasi), bukan keputusan gate.
        ax.set_title(f"{image_id}\nreference: RETAIN {n_ret} · SUPPRESS {n_sup}",
                     fontsize=8.0)

    # ── tandai kotak bersertifikat pada panelnya, bernomor ──────────────────
    for k, bid in enumerate(cert_ids, 1):
        if bid not in cert_rows:
            print(f"[fig9] PERINGATAN: {bid} tidak ditemukan pada gambar manapun")
            continue
        _row, _sc, cimg, (x1, y1, x2, y2) = cert_rows[bid]
        if cimg not in ids:
            continue
        ax = axes[ids.index(cimg)]
        pad = max(18.0, 0.35 * max(x2 - x1, y2 - y1))
        ax.add_patch(Rectangle((x1 - pad, y1 - pad),
                               (x2 - x1) + 2 * pad, (y2 - y1) + 2 * pad,
                               fill=False, edgecolor="black", linewidth=1.8,
                               linestyle=(0, (4, 2))))
        # Penanda diselang-seling kiri-atas / kanan-bawah. Kotak bersertifikat sering
        # bertetangga rapat, dan dua penanda di sudut yang sama akan bertumpuk.
        if k % 2 == 1:
            tx, ty, ha, va = x1 - pad, y1 - pad - 8, "right", "bottom"
        else:
            tx, ty, ha, va = x2 + pad, y2 + pad + 8, "left", "top"
        ax.text(tx, ty, f"{k}", fontsize=9, fontweight="bold",
                color="black", ha=ha, va=va,
                bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="black",
                          lw=0.8))

    # ── panel sertifikat ────────────────────────────────────────────────────
    for k, bid in enumerate(cert_ids, 1):
        ax = axes[n_img + k - 1]
        if bid not in cert_rows:
            ax.axis("off")
            continue
        row, sc, _cimg, _bb = cert_rows[bid]
        g = gate_map.get(bid) if gate_map else None
        dec = render_certificate(ax, plt, row, sc, g, tau, f"{k}")
        print(f"[fig9] sertifikat {k}: {bid} skor={sc:.4f} "
              f"g={g if g is None else round(g, 4)} gate={dec} "
              f"ref={row.get('category')}")

    # ── legenda + penanda panel ─────────────────────────────────────────────
    legend_handles = [Patch(color=_COLOR[k], label=lab) for k, lab in _LEGEND]
    leg_ncol = 6 if ncols > 2 else 3
    leg_rows = -(-len(legend_handles) // leg_ncol)
    foot = 0.055 * leg_rows + 0.045
    fig.tight_layout(rect=(0, foot, 1, 1), h_pad=2.2)
    fig.legend(handles=legend_handles, loc="lower center", ncol=leg_ncol,
               bbox_to_anchor=(0.5, 0.030), frameon=False, fontsize=7.6,
               handlelength=1.0, columnspacing=1.6, handletextpad=0.5,
               labelspacing=0.35)

    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    tags = [f"({c})" for c in "abcdefgh"[:n_panels]]
    for r in range(nrows):
        row_axes = axes[r * ncols:(r + 1) * ncols][:n_panels - r * ncols]
        if not row_axes:
            continue
        y0 = min(a.get_tightbbox(rend).transformed(
            fig.transFigure.inverted()).y0 for a in row_axes)
        panel_tags_bottom(fig, row_axes, tags[r * ncols:(r + 1) * ncols],
                          y=max(y0 - 0.030, 0.004))

    # Ditulis ke KEDUA folder keluaran, seperti _style.save().
    written = []
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            target = d / f"{args.out.stem}.{ext}"
            fig.savefig(target, dpi=200, bbox_inches="tight")
            written.append(target)
    print(f"[fig9] ditulis {len(written)} berkas ({n_img} gambar, "
          f"{len(cert_rows)} sertifikat)")
    for t in written:
        print(f"        {t}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
