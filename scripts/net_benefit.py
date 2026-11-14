
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.eval_fp_reduction_benchmark import apply_mlp, fit_mlp, load_csv  # noqa: E402
from tools.make_indomain_results import (  # noqa: E402
    POSITIVE, SEEDS, T_DEP, V_DEP, gate_matrix,
)

FUSED = ROOT / "_septem/scripts_optionB/fused_scores_oiaddr.csv"
OUT = ROOT / "_septem/scripts_optionB/net_benefit.json"
EV = ("ev_sas", "ev_sdca", "ev_aps", "ev_pss", "ev_sds", "ev_sds_nsrc")
PT = [round(x, 3) for x in np.arange(0.05, 0.80, 0.05)]
t0 = time.perf_counter()


def log(m):
    print(f"[nb] {m} ({time.perf_counter() - t0:.0f}s)", flush=True)


def load_fused(split, bbox_ids):
    if not FUSED.is_file():
        return None
    m = {}
    with open(FUSED, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["split"] == split:
                m[r["bbox_id"]] = float(r["fused"])
    if any(b not in m for b in bbox_ids):
        return None
    return np.array([m[b] for b in bbox_ids], dtype=np.float64)


def bbox_ids(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as fh:
        return [r["bbox_id"] for r in csv.DictReader(fh)]


def net_benefit(score, y, pt):

    n = len(y)
    tau = float(np.quantile(score, 1.0 - pt))
    keep = score >= tau
    tp = float((keep & (y == 1)).sum())
    fp = float((keep & (y == 0)).sum())
    return tp / n - (fp / n) * (pt / (1.0 - pt))


va, te = load_csv(V_DEP, POSITIVE), load_csv(T_DEP, POSITIVE)
yv, yt = va["label"], te["label"]
fv = load_fused("valid", bbox_ids(V_DEP))
ft = load_fused("test", bbox_ids(T_DEP))
log(f"test n={len(yt)} positif={int(yt.sum())} ({yt.mean():.3f}); "
    f"skor fusi {'ada' if ft is not None else 'tidak ada'}")

systems = {"detector_score": te["pred_score"]}
if ft is not None:
    systems["score_fusion"] = ft

Xv, Xt = gate_matrix(va, "full_gate", None), gate_matrix(te, "full_gate", None)
gate_scores = []
for s in SEEDS:
    m = fit_mlp(Xv, yv.astype(np.float64), seed=s)
    gate_scores.append(apply_mlp(Xt, m))
systems["full_gate"] = np.mean(gate_scores, axis=0)
log("gate penuh dipasang, 5 seed (skor dirata-ratakan antar seed untuk kurva)")

if ft is not None:
    Xvf = np.stack([va[c] for c in EV] + [fv], 1)
    Xtf = np.stack([te[c] for c in EV] + [ft], 1)
    gf = [apply_mlp(Xtf, fit_mlp(Xvf, yv.astype(np.float64), seed=s)) for s in SEEDS]
    systems["gate_on_fusion"] = np.mean(gf, axis=0)
    log("gate di atas skor fusi dipasang, 5 seed")

prev = float(yt.mean())
curves = {"treat_all": [prev - (1 - prev) * (p / (1 - p)) for p in PT],
          "treat_none": [0.0 for _ in PT]}
for name, s in systems.items():
    curves[name] = [net_benefit(np.asarray(s, dtype=np.float64), yt, p) for p in PT]

best = {}
for i, p in enumerate(PT):
    vals = {k: v[i] for k, v in curves.items()}
    best[str(p)] = max(vals, key=vals.get)

res = {"pt_grid": PT, "prevalence": prev, "n": int(len(yt)),
       "curves": curves, "best_system_per_pt": best,
       "note": "NB = TP/n - (FP/n) * pt/(1-pt); ambang = kuantil skor yang menyamai pt"}
OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")

log("manfaat bersih pada beberapa p_t:")
for i, p in enumerate(PT):
    if i % 3:
        continue
    row = "  ".join(f"{k}={curves[k][i]:+.4f}" for k in curves if k != "treat_none")
    log(f"  pt={p:.2f}  {row}  -> terbaik: {best[str(p)]}")
log(f"selesai -> {OUT.name}")
