#!/usr/bin/env python

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.eval_fp_reduction_benchmark import (  # noqa: E402
    apply_mlp, auprc, eval_at_threshold, fit_mlp, load_csv, open_csv, threshold_for_tp_ret,
)

DELTA = 0.30
TP_RET_TARGET = 0.80
GATE_SEED = 0
EV = ("ev_sas", "ev_sdca", "ev_aps", "ev_pss")
TIERS = ("R1", "R2", "R3", "R4")
TIER_NAME = {"R1": "accepted", "R2": "provisional", "R3": "contested", "R4": "rejected"}
SDM_LEVELS = {0.0, 0.5, 0.75, 1.0}

BASE = ROOT / "outputs/0506"
EVIDENCE = ROOT / "results/evidence"


def _first(workspace: Path, repo_name: str) -> Path:

    return workspace if workspace.exists() else EVIDENCE / repo_name


VALID = _first(BASE / "evsds_v2sec/oiaddr/valid/governance_features_valid.csv",
               "oiaddr_valid_evidence.csv.gz")
COHORTS = {
    "oiaddr": _first(BASE / "evsds_v2sec/oiaddr/test/governance_features_test.csv",
                     "oiaddr_test_evidence.csv.gz"),
    "idrid": _first(BASE / "evsds_pss/idrid/test/governance_features_test.csv",
                    "idrid_test_evidence.csv.gz"),
    "tjdr": _first(BASE / "evsds_pss/tjdr/test/governance_features_test.csv",
                   "tjdr_test_evidence.csv.gz"),
    "fgadr": _first(BASE / "evsds_pss/fgadr/train/governance_features_train.csv",
                    "fgadr_train_evidence.csv.gz"),
}
# The per-detection tier CSVs and the summary go beside the workspace outputs, or into
# _cache/ in the repository so that the bundled results/ files are never overwritten.
OUT = BASE / "tier_routing" if BASE.is_dir() else ROOT / "_cache/tier_routing"
TAB = ROOT / "_FINAL_Latex/tables/T_tier_routing.tex"
# Skor fusi tiga pass OIA-DDR, dibangkitkan _septem/scripts_optionB/export_fused_scores.py.
# Dicari di dua tempat supaya skrip ini jalan baik di pohon proyek maupun di repositori kode
# anonim, yang menaruh berkas hasil di results/experiments/.
FUSED_CANDIDATES = (ROOT / "_septem/scripts_optionB/fused_scores_oiaddr.csv",
                    ROOT / "results/experiments/fused_scores_oiaddr.csv")


def load_fused() -> dict:

    for p in FUSED_CANDIDATES:
        if p.is_file():
            out = {}
            with open(p, newline="", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    out[(r["split"], r["bbox_id"])] = float(r["fused"])
            print(f"[tier] skor fusi dari {p.name} ({len(out)} kotak)", flush=True)
            return out
    raise SystemExit("[tier] fused_scores_oiaddr.csv tidak ditemukan; jalankan dulu "
                     "python _septem/scripts_optionB/export_fused_scores.py")


def fused_vector(fused: dict, split: str, ids) -> np.ndarray:

    missing = [b for b in ids if (split, b) not in fused]
    if missing:
        raise SystemExit(f"[tier] {len(missing)} bbox_id {split} tidak ada di skor fusi")
    return np.array([fused[(split, b)] for b in ids], dtype=np.float64)


def rel(p: Path) -> str:
    return p.resolve().relative_to(ROOT).as_posix()


def q(v: float, nd: int = 3) -> str:

    return str(Decimal(repr(float(v))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def gate_matrix(d: dict) -> np.ndarray:
    return np.stack([d[f] for f in EV] + [d["pred_score"]], 1)


def read_ids(path: Path):
    ids, imgs, cats = [], [], []
    with open_csv(path) as fh:
        for r in csv.DictReader(fh):
            ids.append(r["bbox_id"])
            imgs.append(r["image_id"])
            cats.append(r["category"])
    return ids, imgs, np.array(cats)


def assign(keep: np.ndarray, sdca: np.ndarray, delta: float = DELTA) -> np.ndarray:
    hi = sdca >= delta
    return np.where(keep & ~hi, "R1",
                    np.where(keep & hi, "R2", np.where(~keep & hi, "R3", "R4")))


def composition(tier: np.ndarray, label: np.ndarray, cats: np.ndarray) -> dict:
    n = len(label)
    out = {"tiers": {}}
    for t in TIERS:
        m = tier == t
        k = int(m.sum())
        out["tiers"][t] = {
            "name": TIER_NAME[t],
            "n": k,
            "share": k / n,
            "retain_target_share": float(label[m].mean()) if k else None,
            "tp_exact_share": float((cats[m] == "TP_EXACT").mean()) if k else None,
            "categories": dict(sorted(Counter(cats[m].tolist()).items())),
        }
    pos, neg = label == 1, label == 0
    out["retain_targets_by_tier"] = {t: float((tier[pos] == t).mean()) for t in TIERS}
    out["suppress_targets_by_tier"] = {t: float((tier[neg] == t).mean()) for t in TIERS}
    out["flagged_share_R2_R3"] = float(np.isin(tier, ("R2", "R3")).mean())
    for cat, key in (("UNSAFE_MISMATCH", "grade_shifting"),
                     ("CLINICALLY_CLOSE_MISMATCH", "within_grade")):
        m = cats == cat
        out[key] = {"n": int(m.sum()), **{t: int((tier[m] == t).sum()) for t in TIERS}}
    return out


def build_table(oia: dict) -> str:
    m0, fu, gt = oia["m0"], oia["fused"], oia["gate"]
    rule = {"R1": r"retain, SDCA $<\delta$", "R2": r"retain, SDCA $\geq\delta$",
            "R3": r"suppress, SDCA $\geq\delta$", "R4": r"suppress, SDCA $<\delta$"}
    rows = []
    for t in TIERS:
        a, b, c = m0["tiers"][t], fu["tiers"][t], gt["tiers"][t]
        rows.append(
            f"{t} {TIER_NAME[t]} & {rule[t]} & {q(a['share'])} & {q(a['retain_target_share'])}"
            f" & {q(b['share'])} & {q(b['retain_target_share'])}"
            f" & {q(c['share'])} & {q(c['retain_target_share'])} \\\\")
    gs_m, gs_f, gs_g = m0["grade_shifting"], fu["grade_shifting"], gt["grade_shifting"]
    wg_m, wg_f, wg_g = m0["within_grade"], fu["within_grade"], gt["within_grade"]
    n_gs, n_wg = gs_m["n"], wg_m["n"]
    lines = [
        "% Generated by tools/make_tier_routing.py from frozen JSON; do not edit by hand.",
        "% Source: outputs/0506/tier_routing/tier_routing_summary.json",
        "% Tier = retain decision (tau* from validation) x SDCA (delta 0.30), for three",
        "% decision sources: detector score, three-pass fusion score and the gate; the",
        "% reference categories from the annotations are used only to evaluate the tiers,",
        "% never to assign them.",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Four-tier routing on OIA-DDR test under Algorithm~\ref{alg:cegnet} (knowledge-graph "
        r"gate, seed 0, without the multi-source signals). The tier crosses the retain decision with SDCA at "
        r"$\delta=0.30$, and each of the three systems supplies that decision at its own "
        r"threshold selected on validation for a true-positive retention of 0.80. Share is the fraction of all detections in the tier, "
        r"and RETAIN target is the fraction of the tier whose annotation-derived target "
        r"is RETAIN.}",
        r"\label{tab:tier_routing}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}llcccccc@{}}",
        r"\toprule",
        (r" & & \multicolumn{2}{c}{$M_0$ (score)} & \multicolumn{2}{c}{Score fusion} & "
         r"\multicolumn{2}{c}{KG gate (B4)} \\"),
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
        (r"Tier & Rule & Share & RETAIN target & Share & RETAIN target & Share & "
         r"RETAIN target \\"),
        r"\midrule",
        *rows,
        r"\midrule",
        (r"\multicolumn{2}{@{}l}{TP retention / FP suppression} & "
         rf"\multicolumn{{2}}{{c}}{{{q(m0['tp_ret'])} / {q(m0['fp_supp'])}}} & "
         rf"\multicolumn{{2}}{{c}}{{{q(fu['tp_ret'])} / {q(fu['fp_supp'])}}} & "
         rf"\multicolumn{{2}}{{c}}{{{q(gt['tp_ret'])} / {q(gt['fp_supp'])}}} \\"),
        (rf"\multicolumn{{2}}{{@{{}}l}}{{Grade-shifting confusions ($n={n_gs}$) in R1 / R2 / R3}} & "
         rf"\multicolumn{{2}}{{c}}{{{gs_m['R1']} / {gs_m['R2']} / {gs_m['R3']}}} & "
         rf"\multicolumn{{2}}{{c}}{{{gs_f['R1']} / {gs_f['R2']} / {gs_f['R3']}}} & "
         rf"\multicolumn{{2}}{{c}}{{{gs_g['R1']} / {gs_g['R2']} / {gs_g['R3']}}} \\"),
        (rf"\multicolumn{{2}}{{@{{}}l}}{{Within-grade confusions ($n={n_wg:,}$) in R1 / R4}} & "
         rf"\multicolumn{{2}}{{c}}{{{wg_m['R1']:,} / {wg_m['R4']}}} & "
         rf"\multicolumn{{2}}{{c}}{{{wg_f['R1']:,} / {wg_f['R4']}}} & "
         rf"\multicolumn{{2}}{{c}}{{{wg_g['R1']:,} / {wg_g['R4']}}} \\").replace(",", "{,}"),
        r"\bottomrule",
        r"\end{tabular}}",
        r"\\[2pt]",
        r"{\footnotesize Each system uses its own validation threshold, so the three do not "
        r"reach the same retention on test.}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    valid = load_csv(VALID, "governance_retain")
    Xv, yv = gate_matrix(valid), valid["label"]
    mlp = fit_mlp(Xv, yv.astype(np.float64), seed=GATE_SEED)
    tau_gate = threshold_for_tp_ret(apply_mlp(Xv, mlp), yv, TP_RET_TARGET)
    tau_m0 = threshold_for_tp_ret(valid["pred_score"], yv, TP_RET_TARGET)
    fused = load_fused()
    ids_v, _imgs_v, _cats_v = read_ids(VALID)
    tau_fused = threshold_for_tp_ret(fused_vector(fused, "valid", ids_v), yv, TP_RET_TARGET)
    print(f"[tier] tau* gate = {tau_gate:.6f}   tau M0 = {tau_m0:.6f}   "
          f"tau fusi = {tau_fused:.6f}", flush=True)

    summary = {
        "tool": "tools/make_tier_routing.py",
        "rule": "R1 RETAIN&SDCA<delta, R2 RETAIN&SDCA>=delta, R3 SUPPRESS&SDCA>=delta, "
                "R4 SUPPRESS&SDCA<delta",
        "delta": DELTA,
        "tp_ret_target": TP_RET_TARGET,
        "gate": "fit_mlp([ev_sas, ev_sdca, ev_aps, ev_pss, pred_score]), hidden 16, "
                "1500 iters, lr 0.1, seed 0",
        "fused": "rata-rata skor primer + flip + detektor kedua, kotak tanpa pasangan = 0; "
                 "keputusan retain memakai tau-nya sendiri dari valid pada retensi TP 0.80",
        "valid_csv": rel(VALID),
        "tau_gate": tau_gate,
        "tau_m0": tau_m0,
        "tau_fused": tau_fused,
        "cohorts": {},
    }

    for key, path in COHORTS.items():
        d = load_csv(path, "governance_retain")
        ids, imgs, cats = read_ids(path)
        if len(ids) != len(d["label"]):
            raise SystemExit(f"[tier] {key}: urutan baris tidak sejajar")
        sd, y = d["ev_sdca"], d["label"]
        levels = sorted(set(np.round(sd, 6).tolist()))
        g = apply_mlp(gate_matrix(d), mlp)
        keep = g >= tau_gate
        tier = assign(keep, sd)
        invariance = {str(dl): bool(np.array_equal(assign(keep, sd, dl), tier))
                      for dl in (1e-6, 0.25, 0.5)}
        entry = {
            "csv": rel(path),
            "n": int(len(y)),
            "n_images": len(set(imgs)),
            "sdca_levels": levels,
            "sdca_levels_within_expected": set(levels) <= SDM_LEVELS,
            "delta_invariance": invariance,
            "gate": {"auprc": auprc(g, y), **eval_at_threshold(g, y, tau_gate),
                     **composition(tier, y, cats)},
        }
        if key == "oiaddr":
            s = d["pred_score"]
            entry["m0"] = {"auprc": auprc(s, y), **eval_at_threshold(s, y, tau_m0),
                           **composition(assign(s >= tau_m0, sd), y, cats)}
            # Keputusan retain dari skor fusi, tier tetap dari SDCA. Ini menunjukkan aturan
            # tier komposabel dengan peringkat mana pun, termasuk yang terkuat, dan bahwa
            # SDCA-lah yang membuat tier ada, bukan gate-nya.
            fs = fused_vector(fused, "test", ids)
            entry["fused"] = {"auprc": auprc(fs, y), **eval_at_threshold(fs, y, tau_fused),
                              **composition(assign(fs >= tau_fused, sd), y, cats)}

        per_img: dict = {}
        for im, t in zip(imgs, tier):
            per_img.setdefault(im, Counter())[t] += 1
        r1 = [c.get("R1", 0) for c in per_img.values()]
        entry["per_image_R1"] = {"min": int(min(r1)), "max": int(max(r1))}

        with open(OUT / f"{key}_gate_tiers.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["bbox_id", "image_id", "g", "decision", "ev_sdca", "tier", "category"])
            for i in range(len(ids)):
                w.writerow([ids[i], imgs[i], f"{g[i]:.6f}",
                            "RETAIN" if keep[i] else "SUPPRESS",
                            f"{sd[i]:.4f}", tier[i], cats[i]])
        summary["cohorts"][key] = entry
        gt = entry["gate"]
        print(f"[tier] {key}: n={entry['n']} img={entry['n_images']} levels={levels} "
              f"inv={invariance} AUPRC={gt['auprc']:.4f} tp_ret={gt['tp_ret']:.4f} "
              f"fp_supp={gt['fp_supp']:.4f} "
              + " ".join(f"{t}={gt['tiers'][t]['share']:.3f}/{gt['tiers'][t]['retain_target_share']:.3f}"
                         for t in TIERS), flush=True)

    (OUT / "tier_routing_summary.json").write_text(json.dumps(summary, indent=2),
                                                   encoding="utf-8")
    TAB.write_text(build_table(summary["cohorts"]["oiaddr"]), encoding="utf-8")
    print(f"[tier] ditulis {rel(OUT / 'tier_routing_summary.json')} dan {rel(TAB)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
