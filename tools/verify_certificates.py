#!/usr/bin/env python
"""verify_certificates.py — validasi auditability teknis atas sertifikat per-deteksi.

Lima pemeriksaan, semuanya deterministik dan tanpa anotasi:

  C1 determinisme        emisi ulang ke direktori sementara menghasilkan berkas identik
                         bita-per-bita (sha256 sama) untuk setiap kohort.
  C2 rekonstruksi        keputusan dan tier dapat dihitung ulang HANYA dari field
                         sertifikat (retain_probability, tau_star, ev_sdca, delta)
                         memakai aturan yang diterbitkan, dan cocok dengan yang tersimpan.
  C3 konsistensi prior   ev_sdca sama dengan SDM_norm[argmax, runner-up] yang dihitung
                         ulang dari kg_graph.json terbitan.
  C4 pengungkapan label  laju ketidaksamaan antara label teremisi dan argmax ROI, angka
                         yang dinyatakan naskah untuk OIA-DDR test.
  C5 kebebasan ranker    field prior sertifikat (argmax, runner-up, ev_sdca) dihitung dari
                         distribusi per-ROI saja, sehingga tidak berubah bila keputusan
                         diambil oleh skor detektor alih-alih gate.

Ini adalah validasi auditability TEKNIS. Ia tidak menunjukkan manfaat klinis dan bukan
pengganti reader study.

Keluaran: outputs/0506/certificates/verification.json
Jalankan dari root repo:
  python tools/verify_certificates.py
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.emit_certificates import (  # noqa: E402
    LESIONS, PROB_COL, SOURCES, emit, load_sdm, sha256_file, top_two, GRAPH,
)

CERT_DIR = ROOT / "outputs/0506/certificates"
OUT = CERT_DIR / "verification.json"

csv.field_size_limit(1 << 24)


def rebuild_decision(g: float, tau: float, sdca: float, delta: float):
    decision = "RETAIN" if g >= tau else "SUPPRESS"
    if decision == "RETAIN":
        tier = "R1" if sdca < delta else "R2"
    else:
        tier = "R3" if sdca >= delta else "R4"
    return decision, tier


def main() -> None:
    manifest = json.loads((CERT_DIR / "manifest.json").read_text(encoding="utf-8"))
    tau = manifest["tau_star"]
    delta = manifest["delta"]
    sdm = load_sdm(ROOT / GRAPH)
    idx = {c: i for i, c in enumerate(LESIONS)}

    report = {"tau_star": tau, "delta": delta, "checks": {}, "per_cohort": {}}

    # ---- C1 determinisme -----------------------------------------------------
    tmp = Path(tempfile.mkdtemp(prefix="certverify_"))
    try:
        print("[C1] emisi ulang untuk uji determinisme ...")
        emit(tmp)
        c1 = {}
        for cohort in SOURCES:
            a = manifest["outputs"][cohort]["sha256"]
            b = sha256_file(tmp / f"{cohort}_certificates.csv")
            c1[cohort] = {"stored_sha256": a, "reemitted_sha256": b, "identical": a == b}
        report["checks"]["C1_determinism"] = {
            "per_cohort": c1,
            "passed": all(v["identical"] for v in c1.values()),
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # ---- C2..C5 sekali lintas per kohort -------------------------------------
    tot = {"n": 0, "c2": 0, "c3": 0, "c4_mismatch": 0, "c5": 0}
    for cohort, (feat_rel, _) in SOURCES.items():
        print(f"[C2-C5] {cohort} ...")
        # distribusi per-ROI dari CSV bukti, untuk C3 dan C5 (tanpa gate)
        roi = {}
        with (ROOT / feat_rel).open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                (a_cls, _), (r_cls, _) = top_two(row)
                roi[row["bbox_id"]] = (a_cls, r_cls, row["ev_sdca"])

        n = c2 = c3 = c4 = c5 = 0
        with (CERT_DIR / f"{cohort}_certificates.csv").open(newline="", encoding="utf-8") as fh:
            for rec in csv.DictReader(fh):
                n += 1
                d, t = rebuild_decision(float(rec["retain_probability"]),
                                        float(rec["tau_star"]),
                                        float(rec["ev_sdca"]),
                                        float(rec["delta"]))
                if d == rec["decision"] and t == rec["tier"]:
                    c2 += 1
                recomputed = sdm[idx[rec["argmax_class"]]][idx[rec["runnerup_class"]]]
                if abs(recomputed - float(rec["ev_sdca"])) <= 1e-9:
                    c3 += 1
                if rec["emitted_equals_argmax"] == "0":
                    c4 += 1
                a_cls, r_cls, sdca_src = roi[rec["bbox_id"]]
                if (a_cls == rec["argmax_class"] and r_cls == rec["runnerup_class"]
                        and sdca_src == rec["ev_sdca"]):
                    c5 += 1

        report["per_cohort"][cohort] = {
            "n": n,
            "C2_decision_tier_match": c2,
            "C2_rate": c2 / n,
            "C3_sdca_match": c3,
            "C3_rate": c3 / n,
            "C4_emitted_differs_from_argmax": c4,
            "C4_rate": c4 / n,
            "C5_prior_fields_ranker_free": c5,
            "C5_rate": c5 / n,
        }
        tot["n"] += n
        tot["c2"] += c2
        tot["c3"] += c3
        tot["c4_mismatch"] += c4
        tot["c5"] += c5

    report["checks"]["C2_decision_reconstruction"] = {
        "matched": tot["c2"], "of": tot["n"], "passed": tot["c2"] == tot["n"]}
    report["checks"]["C3_prior_consistency"] = {
        "matched": tot["c3"], "of": tot["n"], "passed": tot["c3"] == tot["n"]}
    report["checks"]["C4_label_disclosure"] = {
        "differs": tot["c4_mismatch"], "of": tot["n"],
        "rate_overall": tot["c4_mismatch"] / tot["n"],
        "rate_oiaddr": report["per_cohort"]["oiaddr"]["C4_rate"]}
    report["checks"]["C5_ranker_independence"] = {
        "matched": tot["c5"], "of": tot["n"], "passed": tot["c5"] == tot["n"]}

    report["all_passed"] = all(
        report["checks"][k].get("passed", True)
        for k in ("C1_determinism", "C2_decision_reconstruction",
                  "C3_prior_consistency", "C5_ranker_independence"))
    report["scope_note"] = ("Validasi auditability teknis. Tidak menunjukkan manfaat klinis "
                            "dan bukan pengganti reader study.")

    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n=== RINGKASAN ===")
    print(f"C1 determinisme      : {'LOLOS' if report['checks']['C1_determinism']['passed'] else 'GAGAL'}")
    print(f"C2 rekonstruksi      : {tot['c2']}/{tot['n']}")
    print(f"C3 konsistensi prior : {tot['c3']}/{tot['n']}")
    print(f"C4 label != argmax   : {tot['c4_mismatch']}/{tot['n']} "
          f"(OIA-DDR {report['per_cohort']['oiaddr']['C4_rate']*100:.2f}%)")
    print(f"C5 bebas ranker      : {tot['c5']}/{tot['n']}")
    print(f"SELURUHNYA           : {'LOLOS' if report['all_passed'] else 'GAGAL'}")
    print(f"ditulis {OUT}")


if __name__ == "__main__":
    sys.exit(main())
