"""make_s5_circularity.py — tabel suplemen S5 dari circularity_check.json.

Isi: gating pada subset test yang labelnya TIDAK memakai matriks jarak semantik (kategori
clinically-close dan unsafe-mismatch dibuang), untuk dua cara pemasangan gate:
  A. gate dipasang seperti hasil utama (target deployed pada valid), dievaluasi pada label bersih;
  B. gate dipasang ulang pada target bersih, dievaluasi pada label bersih.
Keluaran: _FINAL_Latex/tables/S5_label_circularity.tex. Jalankan dari root repo.
"""
import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "_septem/scripts_optionB/circularity_check.json"
OUT = ROOT / "_FINAL_Latex/tables/S5_label_circularity.tex"


def q(v, nd=3):
    return str(Decimal(repr(float(v))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def sgn(v, nd=3):
    return ("$+$" if v >= 0 else "$-$") + q(abs(v), nd)


d = json.loads(SRC.read_text(encoding="utf-8"))
m0, n, npos = d["m0_ap_test_clean"], d["n_test_clean"], d["n_test_clean_pos"]
rows = [rf"Detector score threshold ($M_0$) & -- & {q(m0)} & -- & -- \\", r"\midrule"]
blocks = (("A_fit_deployed_eval_clean", "as in Table~\\ref{tab:governance}"),
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
    "% S5 — DIBANGKITKAN _septem/scripts_optionB/make_s5_circularity.py dari",
    "% _septem/scripts_optionB/circularity_check.json. JANGAN diedit tangan.",
    r"\begin{table}[htbp]",
    r"\centering",
    (r"\caption{Gating when the triage label does not use the semantic-distance matrix. The "
     rf"clinically close and unsafe-mismatch categories, which are defined through "
     rf"$\mathrm{{SDM}}_{{\mathrm{{norm}}}}$, are removed from the OIA-DDR test split, leaving "
     rf"{n:,} of 64{{,}}214 detections and {npos:,} positives. The first block keeps the gates "
     r"fitted as in the main results; the second refits them on the same label definition. AUPRC "
     r"is average precision, mean $\pm$ SD over five initialization seeds.}").replace(",", "{,}", 0),
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
