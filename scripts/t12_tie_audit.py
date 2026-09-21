"""Audit estimator AUPRC untuk Tabel 12 (domain shift): trapezoid vs average precision.

Pertanyaan: (1) apakah sel Tabel 12 (tiga desimal) bergantung pada urutan skor kembar;
(2) apakah sel itu berubah bila AUPRC dihitung sebagai average precision.
Mereplikasi journal/figures/scripts/_figdata_domain_shift.py: gate B4 dan B6 (build_methods)
dipasang pada OIA-DDR valid untuk seed 0..4 lalu diterapkan pada tiap kohort.
"""
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.eval_fp_reduction_benchmark import apply_mlp, fit_mlp, load_csv  # noqa: E402
from tools.make_indomain_results import metric, q  # noqa: E402

SCR = Path(__file__).resolve().parent  # keluaran ditulis di samping skrip
REF = ROOT / "outputs/0506/evsds_pss/domain_shift_5seed.json"
POS = "governance_retain"
EV = ("ev_sas", "ev_sdca", "ev_aps", "ev_pss")


def feats(d, b6):
    x = np.stack([d[c] for c in EV] + [d["pred_score"]], 1)
    if b6:
        x = np.concatenate([x, d["ev_sds"].reshape(-1, 1), d["ev_sds_nsrc"].reshape(-1, 1)], 1)
    return x


def disp_gain(vals):
    m, s = statistics.mean(vals), statistics.stdev(vals)
    return f"{'+' if m >= 0 else '-'}{q(abs(m))}+-{q(s)} ({sum(1 for v in vals if v > 0)}/5)"


def main():
    t0 = time.perf_counter()
    ref = json.loads(REF.read_text(encoding="utf-8"))
    valid = load_csv(ROOT / ref["valid_csv"], POS)
    yv = valid["label"].astype(np.float64)
    models = {}
    for seed in range(5):
        for b6 in (False, True):
            models[(seed, b6)] = fit_mlp(feats(valid, b6), yv, seed=seed)
        print(f"fit seed {seed} ({time.perf_counter() - t0:.0f}s)", flush=True)
    out = []
    for c in ref["cohorts"]:
        te = load_csv(ROOT / c["csv"], POS)
        y = te["label"]
        m0 = metric(te["pred_score"], y)
        rec = {"name": c["name"], "n": int(y.size), "m0": m0, "m0_json": c["m0_auprc"],
               "m0_abs_diff_json": abs(m0["auprc_trapezoid"] - c["m0_auprc"]), "b4": [], "b6": []}
        for seed in range(5):
            for b6, key, jkey in ((False, "b4", "b4_auprc_per_seed"), (True, "b6", "b6_auprc_per_seed")):
                r = metric(apply_mlp(feats(te, b6), models[(seed, b6)]), y)
                r["json_tz"] = c[jkey][seed]
                r["abs_diff_json"] = abs(r["auprc_trapezoid"] - c[jkey][seed])
                rec[key].append(r)

        def cells(k):
            m0v = m0[k]
            b4 = [r[k] for r in rec["b4"]]
            b6 = [r[k] for r in rec["b6"]]
            return {"m0": q(m0v), "b4_mean": q(statistics.mean(b4)), "b6_mean": q(statistics.mean(b6)),
                    "b6_b4": disp_gain([b - a for a, b in zip(b4, b6)]),
                    "b6_m0": disp_gain([b - m0v for b in b6])}

        rec["cells_tz"] = cells("auprc_trapezoid")
        rec["cells_ap"] = cells("auprc")
        lo4 = statistics.mean(r["tie_bounds"][0] for r in rec["b4"])
        hi4 = statistics.mean(r["tie_bounds"][1] for r in rec["b4"])
        lo6 = statistics.mean(r["tie_bounds"][0] for r in rec["b6"])
        hi6 = statistics.mean(r["tie_bounds"][1] for r in rec["b6"])
        m0lo, m0hi = m0["tie_bounds"]
        rec["tz_display_range"] = {
            "m0": [q(m0lo), q(m0hi)], "b4_mean": [q(lo4), q(hi4)], "b6_mean": [q(lo6), q(hi6)],
            "b6_m0_mean": [q(lo6 - m0hi), q(hi6 - m0lo)], "b6_b4_mean": [q(lo6 - hi4), q(hi6 - lo4)]}
        rec["tie_width_max"] = {
            "m0": m0hi - m0lo,
            "b4": max(r["tie_bounds"][1] - r["tie_bounds"][0] for r in rec["b4"]),
            "b6": max(r["tie_bounds"][1] - r["tie_bounds"][0] for r in rec["b6"])}
        rec["max_abs_diff_json"] = max(r["abs_diff_json"] for r in rec["b4"] + rec["b6"])
        out.append(rec)
        print(f"{c['name']}: n={rec['n']} json_diff m0={rec['m0_abs_diff_json']:.2e} gates={rec['max_abs_diff_json']:.2e}\n"
              f"  TZ cells {rec['cells_tz']}\n  AP cells {rec['cells_ap']}\n"
              f"  TZ display range under tie order {rec['tz_display_range']}\n"
              f"  tie width max {rec['tie_width_max']} ({time.perf_counter() - t0:.0f}s)", flush=True)
        del te
    SCR.joinpath("t12_tie_audit.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote t12_tie_audit.json")


if __name__ == "__main__":
    main()
