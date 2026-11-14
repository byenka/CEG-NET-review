
import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

def _resolve(*candidates):

    for base in Path(__file__).resolve().parents:
        for rel in candidates:
            p = base / rel
            if p.exists():
                return p, base
    raise SystemExit("input not found: " + " or ".join(candidates))


SRC, ROOT = _resolve("_septem/scripts_optionB/circularity_check.json",
                     "results/experiments/circularity_check.json")
OUT = ROOT / "_FINAL_Latex/tables/S5_label_circularity.tex"


def q(v, nd=3):
    return str(Decimal(repr(float(v))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def sgn(v, nd=3):
    return ("$+$" if v >= 0 else "$-$") + q(abs(v), nd)


d = json.loads(SRC.read_text(encoding="utf-8"))
m0, n, npos = d["m0_ap_test_clean"], d["n_test_clean"], d["n_test_clean_pos"]
rows = [rf"Detector score threshold ($M_0$) & -- & {q(m0)} & -- & -- \\", r"\midrule"]
blocks = (("A_fit_deployed_eval_clean", "as in the main results"),
          ("B_fit_clean_eval_clean", "refitted on this label"))
names = (("kg_gate", "Knowledge-graph gate (B4)"), ("full_gate", "Full gate, CEG-Net (B6)"))
for bkey, how in blocks:
    for gkey, label in names:
        g = d[bkey][gkey]
        rows.append(rf"{label} & {how} & {q(g['ap_mean'])} $\pm$ {q(g['ap_sd'])} & "
                    rf"{sgn(g['gain_mean'])} $\pm$ {q(g['gain_sd'])} & {g['n_gain_pos']}/5 \\")
    if bkey == blocks[0][0]:
        rows.append(r"\midrule")
tex = "\n".join([
    r"\begin{table}[htbp]",
    r"\centering",
    (r"\caption{Gating on a label that does not use $\mathrm{SDM}_{\mathrm{norm}}$. The clinically close "
     rf"and unsafe-mismatch detections are removed from OIA-DDR test, leaving {n:,} of 64{{,}}214 detections "
     rf"and {npos:,} positives. The first block keeps the gates of the main results and the second refits "
     r"them on this label. AUPRC is mean $\pm$ SD over five seeds.}").replace(",", "{,}", 0),
    r"\label{tab:label_circularity}",
    r"\footnotesize",
    r"\setlength{\tabcolsep}{4pt}",
    r"\resizebox{\textwidth}{!}{%",
    r"\begin{tabular}{@{}llccc@{}}",
    r"\toprule",
    r"Model & Gate fit & AUPRC & $\Delta$AUPRC vs.\ $M_0$ & Seeds $\Delta>0$ \\",
    r"\midrule",
    *rows,
    r"\bottomrule",
    r"\end{tabular}}",
    r"\end{table}",
    "",
])
OUT.write_text(tex.replace(f"{n:,}", f"{n:,}".replace(",", "{,}")).replace(f"{npos:,}", f"{npos:,}".replace(",", "{,}")),
               encoding="utf-8")
print(f"ditulis {OUT}")
