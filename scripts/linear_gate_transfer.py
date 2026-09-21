"""linear_gate_transfer.py — bentuk linear gate B4/B6 pada keempat kohort (bahan saran poin 1).

Regresi logistik (fit_logreg, 500 iterasi, lr 0.5) dipasang sekali pada OIA-DDR valid dengan
masukan yang sama dengan gate MLP, lalu diterapkan tanpa perubahan ke OIA-DDR test, IDRiD, TJDR
dan FGADR. Dibandingkan dengan rerata lima seed MLP (average precision) dari
_septem/scripts_optionB/t12_tie_audit.json. Hanya membaca; tidak mengubah naskah.
"""
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.eval_fp_reduction_benchmark import apply_logreg, fit_logreg, load_csv  # noqa: E402
from tools.make_indomain_results import gate_matrix, metric  # noqa: E402

REF = ROOT / "outputs/0506/evsds_pss/domain_shift_5seed.json"
AUD = ROOT / "_septem/scripts_optionB/t12_tie_audit.json"
OUT = ROOT / "_septem/scripts_optionB/linear_gate_transfer.json"
POS = "governance_retain"

ref = json.loads(REF.read_text(encoding="utf-8"))
aud = {c["name"]: c for c in json.loads(AUD.read_text(encoding="utf-8"))}
valid = load_csv(ROOT / ref["valid_csv"], POS)
yv = valid["label"].astype(np.float64)
params = {g: fit_logreg(gate_matrix(valid, g, None), yv) for g in ("kg_gate", "full_gate")}
rows = []
for c in ref["cohorts"]:
    te = load_csv(ROOT / c["csv"], POS)
    y = te["label"]
    a = aud[c["name"]]
    m0 = a["m0"]["auprc"]
    mlp_b4 = statistics.mean(r["auprc"] for r in a["b4"])
    mlp_b6 = statistics.mean(r["auprc"] for r in a["b6"])
    mlp_b6_pos = sum(1 for r in a["b6"] if r["auprc"] > m0)
    lin = {}
    for g in ("kg_gate", "full_gate"):
        lin[g] = metric(apply_logreg(gate_matrix(te, g, None), params[g]), y)["auprc"]
    rec = {"cohort": c["name"], "m0_ap": m0, "mlp_b4_ap_mean": mlp_b4, "mlp_b6_ap_mean": mlp_b6,
           "mlp_b6_seeds_above_m0": mlp_b6_pos, "lin_b4_ap": lin["kg_gate"], "lin_b6_ap": lin["full_gate"],
           "gain_mlp_b6": mlp_b6 - m0, "gain_lin_b6": lin["full_gate"] - m0,
           "gain_mlp_b4": mlp_b4 - m0, "gain_lin_b4": lin["kg_gate"] - m0}
    rows.append(rec)
    print(f"{c['name']:8s} M0 {m0:.4f} | B4 MLP {mlp_b4:.4f} lin {lin['kg_gate']:.4f} | "
          f"B6 MLP {mlp_b6:.4f} lin {lin['full_gate']:.4f} | gain B6 MLP {mlp_b6 - m0:+.4f} "
          f"lin {lin['full_gate'] - m0:+.4f} | gain B4 MLP {mlp_b4 - m0:+.4f} lin {lin['kg_gate'] - m0:+.4f}",
          flush=True)
    del te
OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
print(f"wrote {OUT}")
