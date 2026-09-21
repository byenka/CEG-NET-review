#!/usr/bin/env python
"""emit_certificates.py — emisikan sertifikat audit per-deteksi untuk keempat kohort.

Sertifikat memuat persis apa yang dijanjikan naskah pada Section 3.3: deteksi itu sendiri,
vektor bukti yang dikonsumsi gate, pasangan kelas di balik SDCA (argmax ROI dan runner-up),
probabilitas retain, keputusan pada tau*, dan tier.

Skrip ini TIDAK memasang model dan TIDAK menghitung ulang bukti. Ia merakit catatan per
deteksi dari artefak beku yang sudah menghasilkan tabel naskah:
  - governance_features_*.csv : bukti, distribusi per-ROI, label teremisi
  - {cohort}_gate_tiers.csv   : g, keputusan, tier dari tools/make_tier_routing.py
  - tier_routing_summary.json : tau_gate, delta, resep gate
  - outputs/stage3_audit/kg_graph.json : prior pasangan lesi

Determinisme: tidak ada stempel waktu jam dinding di dalam catatan. Nilai yang berasal dari
CSV sumber diteruskan apa adanya sebagai string, sehingga dua eksekusi menghasilkan berkas
yang identik bita-per-bita. Provenans direkam sebagai sha256 berkas sumber di manifest.

Keluaran (outputs/0506/certificates/):
  {cohort}_certificates.csv  — satu baris per deteksi
  {cohort}_sample.json       — lima catatan pertama dalam bentuk JSON terbaca
  manifest.json              — jumlah baris, sha256 keluaran, sha256 sumber, parameter

Jalankan dari root repo:
  python tools/emit_certificates.py
  python tools/emit_certificates.py --out-dir <dir>   # untuk uji determinisme
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LESIONS = ("MA", "HE", "Hard_EX", "Soft_EX")
PROB_COL = {"MA": "prob_ma", "HE": "prob_he", "Hard_EX": "prob_hardex", "Soft_EX": "prob_softex"}

SOURCES = {
    "oiaddr": ("outputs/0506/evsds_v2sec/oiaddr/test/governance_features_test.csv",
               "outputs/0506/tier_routing/oiaddr_gate_tiers.csv"),
    "idrid":  ("outputs/0506/evsds_pss/idrid/test/governance_features_test.csv",
               "outputs/0506/tier_routing/idrid_gate_tiers.csv"),
    "tjdr":   ("outputs/0506/evsds_pss/tjdr/test/governance_features_test.csv",
               "outputs/0506/tier_routing/tjdr_gate_tiers.csv"),
    "fgadr":  ("outputs/0506/evsds_pss/fgadr/train/governance_features_train.csv",
               "outputs/0506/tier_routing/fgadr_gate_tiers.csv"),
}
SUMMARY = "outputs/0506/tier_routing/tier_routing_summary.json"
GRAPH = "outputs/stage3_audit/kg_graph.json"

TIER_NAME = {"R1": "accepted", "R2": "provisional", "R3": "contested", "R4": "rejected"}

FIELDS = [
    "cohort", "bbox_id", "image_id",
    "emitted_class", "detector_score",
    "argmax_class", "runnerup_class", "argmax_prob", "runnerup_prob",
    "emitted_equals_argmax",
    "ev_sas", "ev_sdca", "ev_aps", "ev_pss", "ev_sds", "ev_sds_nsrc", "ev_pss_status",
    "retain_probability", "tau_star", "decision", "delta", "tier", "tier_name",
]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_sdm(graph_path: Path):
    """SDM_norm dari clinical_cost, bukan weight yang berlantai epsilon."""
    g = json.loads(graph_path.read_text(encoding="utf-8"))
    idx = {n: i for i, n in enumerate(LESIONS)}
    raw = [[0.0] * 4 for _ in range(4)]
    for e in g["edges"]["severity_transition"]:
        a, b = idx[e["src_name"]], idx[e["dst_name"]]
        raw[a][b] = raw[b][a] = float(e["clinical_cost"])
    mx = max(max(r) for r in raw)
    return [[v / mx for v in r] for r in raw]


def top_two(row: dict):
    """Argmax dan runner-up dari distribusi per-ROI. Seri dipecah oleh urutan LESIONS."""
    probs = [(c, float(row[PROB_COL[c]])) for c in LESIONS]
    order = sorted(range(4), key=lambda i: (-probs[i][1], i))
    return probs[order[0]], probs[order[1]]


def emit(out_dir: Path) -> dict:
    summary = json.loads((ROOT / SUMMARY).read_text(encoding="utf-8"))
    tau = summary["tau_gate"]
    delta = summary["delta"]
    sdm = load_sdm(ROOT / GRAPH)

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "tool": "tools/emit_certificates.py",
        "schema": FIELDS,
        "tau_star": tau,
        "delta": delta,
        "gate_recipe": summary["gate"],
        "tier_rule": summary["rule"],
        "sdm_norm_off_diagonal": [sdm[i][j] for i in range(4) for j in range(i + 1, 4)],
        "lesion_order": list(LESIONS),
        "sources": {},
        "outputs": {},
    }

    for cohort, (feat_rel, tier_rel) in SOURCES.items():
        feat_p, tier_p = ROOT / feat_rel, ROOT / tier_rel
        tiers = {}
        with tier_p.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                tiers[r["bbox_id"]] = r

        out_p = out_dir / f"{cohort}_certificates.csv"
        n = 0
        sample = []
        with feat_p.open(newline="", encoding="utf-8") as fin, \
             out_p.open("w", newline="", encoding="utf-8") as fout:
            w = csv.DictWriter(fout, fieldnames=FIELDS, lineterminator="\n")
            w.writeheader()
            for row in csv.DictReader(fin):
                bid = row["bbox_id"]
                t = tiers[bid]
                (a_cls, _), (r_cls, _) = top_two(row)
                rec = {
                    "cohort": cohort,
                    "bbox_id": bid,
                    "image_id": row["image_id"],
                    "emitted_class": row["pred_class"],
                    "detector_score": row["pred_score"],
                    "argmax_class": a_cls,
                    "runnerup_class": r_cls,
                    "argmax_prob": row[PROB_COL[a_cls]],
                    "runnerup_prob": row[PROB_COL[r_cls]],
                    "emitted_equals_argmax": "1" if row["pred_class"] == a_cls else "0",
                    "ev_sas": row["ev_sas"],
                    "ev_sdca": row["ev_sdca"],
                    "ev_aps": row["ev_aps"],
                    "ev_pss": row["ev_pss"],
                    "ev_sds": row.get("ev_sds", ""),
                    "ev_sds_nsrc": row.get("ev_sds_nsrc", ""),
                    "ev_pss_status": row.get("ev_pss_status", ""),
                    "retain_probability": t["g"],
                    "tau_star": f"{tau:.17g}",
                    "decision": t["decision"],
                    "delta": f"{delta:.17g}",
                    "tier": t["tier"],
                    "tier_name": TIER_NAME[t["tier"]],
                }
                w.writerow(rec)
                if len(sample) < 5:
                    sample.append(rec)
                n += 1

        (out_dir / f"{cohort}_sample.json").write_text(
            json.dumps(sample, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        manifest["sources"][cohort] = {
            "features": feat_rel, "features_sha256": sha256_file(feat_p),
            "tiers": tier_rel, "tiers_sha256": sha256_file(tier_p),
        }
        manifest["outputs"][cohort] = {
            "file": out_p.name, "rows": n, "sha256": sha256_file(out_p),
        }
        print(f"  {cohort:7s} {n:7d} sertifikat -> {out_p.name}")

    manifest["sources"]["_graph"] = {"file": GRAPH, "sha256": sha256_file(ROOT / GRAPH)}
    manifest["sources"]["_summary"] = {"file": SUMMARY, "sha256": sha256_file(ROOT / SUMMARY)}
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/0506/certificates")
    args = ap.parse_args()
    out = Path(args.out_dir)
    if not out.is_absolute():
        out = ROOT / out
    print(f"[certificates] menulis ke {out}")
    m = emit(out)
    total = sum(v["rows"] for v in m["outputs"].values())
    print(f"[certificates] total {total} sertifikat di {len(m['outputs'])} kohort")


if __name__ == "__main__":
    sys.exit(main())
