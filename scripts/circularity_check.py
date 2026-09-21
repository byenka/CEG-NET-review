"""circularity_check.py — apakah gain gate bertahan pada target yang tidak didefinisikan lewat SDM?

Target deployed (governance_retain) memakai SDM_norm untuk memisahkan clinically close (RETAIN)
dari unsafe mismatch (SUPPRESS), sedangkan gate membaca SDCA yang juga berasal dari SDM.
Pemeriksaan (hanya membaca, tidak mengubah naskah):
  A. gate dipasang seperti naskah (target deployed, OIA-DDR valid), dievaluasi pada test TANPA
     baris CLINICALLY_CLOSE_MISMATCH dan UNSAFE_MISMATCH (label murni IoU/skor);
  B. gate dipasang ulang pada valid dengan target clean yang sama, dievaluasi pada test clean.
Lima seed, average precision, rerata dan SD sampel; M0 pada subset yang sama.
"""
import json
import statistics
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.eval_fp_reduction_benchmark import apply_mlp, fit_mlp, load_csv  # noqa: E402
from tools.make_indomain_results import T_DEP, V_DEP, gate_matrix  # noqa: E402

OUT = ROOT / "_septem/scripts_optionB/circularity_check.json"
SEEDS = (0, 1, 2, 3, 4)

v_full = load_csv(V_DEP, "governance_retain")
v_clean = load_csv(V_DEP, "governance_retain_clean")
t_clean = load_csv(T_DEP, "governance_retain_clean")
y_t = t_clean["label"]
m0 = float(average_precision_score(y_t, t_clean["pred_score"]))
print(f"valid full {len(v_full['label'])}, valid clean {len(v_clean['label'])}, "
      f"test clean {len(y_t)} (pos {int(y_t.sum())}); M0 AP {m0:.4f}", flush=True)

res = {"m0_ap_test_clean": m0, "n_test_clean": int(len(y_t)), "n_test_clean_pos": int(y_t.sum())}
for mode, vdata in (("A_fit_deployed_eval_clean", v_full), ("B_fit_clean_eval_clean", v_clean)):
    block = {}
    for g in ("kg_gate", "full_gate"):
        vals = []
        for seed in SEEDS:
            m = fit_mlp(gate_matrix(vdata, g, None), vdata["label"].astype(np.float64), seed=seed)
            vals.append(float(average_precision_score(y_t, apply_mlp(gate_matrix(t_clean, g, None), m))))
            print(f"{mode} {g} seed {seed}: AP {vals[-1]:.4f}", flush=True)
        gains = [v - m0 for v in vals]
        block[g] = {"ap_per_seed": vals, "ap_mean": statistics.mean(vals), "ap_sd": statistics.stdev(vals),
                    "gain_mean": statistics.mean(gains), "gain_sd": statistics.stdev(gains),
                    "n_gain_pos": sum(1 for x in gains if x > 0)}
        print(f"== {mode} {g}: AP {block[g]['ap_mean']:.4f} +- {block[g]['ap_sd']:.4f}, gain "
              f"{block[g]['gain_mean']:+.4f} +- {block[g]['gain_sd']:.4f} ({block[g]['n_gain_pos']}/5)", flush=True)
    res[mode] = block
OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
print(f"wrote {OUT}")
