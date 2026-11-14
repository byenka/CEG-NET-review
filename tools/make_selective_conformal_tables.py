#!/usr/bin/env python

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _resolve(*candidates):
    for base in Path(__file__).resolve().parents:
        for rel in candidates:
            p = base / rel
            if p.exists():
                return p, base
    raise SystemExit("masukan tidak ditemukan: " + " atau ".join(candidates))


SRC, ROOT = _resolve("outputs/0506/selective_conformal/baselines.json",
                     "results/selective_conformal/baselines.json")
FUS = SRC.parent / "fusion_headtohead.json"
GCAL, _ = _resolve("_LGOVNET_NEW_SEP/results/step4_gate_calibration.json",
                   "results/conditional_coverage/step4_gate_calibration.json")
# The three-pass score mean row (added 2026-09-30) comes from the locked step 5 run for OIA-DDR, IDRiD and TJDR and
# from the locked step 6 run for FGADR, whose secondary pass step 5 did not read.
STEP5, _ = _resolve("_LGOVNET_NEW_SEP/results/step5_fusion_external_coverage.json",
                    "results/conditional_coverage/step5_fusion_external_coverage.json")
STEP6, _ = _resolve("_LGOVNET_NEW_SEP/results/step6_fgadr_three_pass.json",
                    "results/conditional_coverage/step6_fgadr_three_pass.json")
MEAN_LABEL = "Three-pass score mean"
_TABLES = ROOT / "_FINAL_Latex/tables" if (ROOT / "_FINAL_Latex/tables").is_dir() else ROOT / "tables"

COHORTS = ["OIA-DDR test", "IDRiD", "TJDR", "FGADR"]
ROWS = [
    ("M0_score", "Detector score ($M_0$)"),
    ("Margin", "Top-1$-$top-2 margin"),
    ("Evidence-LogReg", "Evidence, linear"),
    ("KG-gate-MLP (ours)", "Knowledge-graph gate"),
    ("KG-gate+SDS+corrob (B6)", "Full gate, CEG-Net"),
]
GATES = [
    ("KG-gate-MLP (ours)", "KG gate (B4)"),
    ("KG-gate+SDS+corrob (B6)", "Full gate (B6)"),
]


def f4(x):
    return f"{x:.4f}"


def sg(x):
    return f"$+{x:.3f}$" if x >= 0 else f"$-{abs(x):.3f}$"


def main() -> None:
    d = json.loads(SRC.read_text(encoding="utf-8"))
    dig = hashlib.sha256(SRC.read_bytes()).hexdigest()[:16]
    fus = json.loads(FUS.read_text(encoding="utf-8")) if FUS.exists() else None
    gcal = json.loads(GCAL.read_text(encoding="utf-8"))
    gdig = hashlib.sha256(GCAL.read_bytes()).hexdigest()[:16]
    alpha = d["alpha"]
    nominal = 1.0 - alpha

    # ---------------- S15
    # Three-pass score mean per cohort, with a control that the locked runs scored the same population
    s5 = json.loads(STEP5.read_text(encoding="utf-8"))
    s6 = json.loads(STEP6.read_text(encoding="utf-8"))
    mean = {c: (s5["cohorts"][c]["methods"]["F"]["aurc"], s5["cohorts"][c]["methods"]["F"]["gap"])
            for c in COHORTS[:3]}
    f3 = s6["fgadr_three_pass"]["D_operating_point"]["F3"]
    mean["FGADR"] = (f3["aurc"], f3["gap"])
    for c in COHORTS:
        ref = d["cohorts"][c]["methods"]
        run = s5["cohorts"][c]["methods"] if c != "FGADR" else s6["fgadr_three_pass"]["D_operating_point"]
        assert abs(run["M0"]["gap"] - ref["M0_score"]["conformal_coverage_gap"]) < 1e-9, c
        assert abs(run["M0"]["aurc"] - ref["M0_score"]["aurc"]) < 1e-9, c
        assert abs(run["B6"]["aurc"] - ref["KG-gate+SDS+corrob (B6)"]["aurc"]) < 1e-6, c
    assert s5["cohorts"]["OIA-DDR test"]["fusion_passes"] == 3 and s5["cohorts"]["TJDR"]["fusion_passes"] == 3

    rows = []
    # Bold marks the best AURC in each cohort column, whichever method holds it. The coverage gap is
    # not bolded, because the text requires it to be read together with ranking.
    best_aurc = {c: min([d["cohorts"][c]["methods"][k]["aurc"] for k, _ in ROWS] + [mean[c][0]]) for c in COHORTS}

    def row(label, aurcs, gaps):
        cells = [label] + [r"\textbf{" + f4(v) + "}" if v == best_aurc[c] else f4(v) for c, v in zip(COHORTS, aurcs)]
        return " & ".join(cells + [sg(g) for g in gaps]) + r" \\"

    for key, label in ROWS:
        if key == "KG-gate-MLP (ours)":
            rows.append(row(MEAN_LABEL, [mean[c][0] for c in COHORTS], [mean[c][1] for c in COHORTS]))
        rows.append(row(label, [d["cohorts"][c]["methods"][key]["aurc"] for c in COHORTS],
                        [d["cohorts"][c]["methods"][key]["conformal_coverage_gap"] for c in COHORTS]))

    note = ""
    if fus:
        m = fus["methods"]
        p = fus["paired_vs_fusion"]
        # The three-pass mean is now a row of the table, so the note keeps only the paired in-domain
        # AURC differences to it, which appear nowhere else.
        assert abs(m["Three-pass mean (F)"]["aurc"] - mean["OIA-DDR test"][0]) < 1e-6
        note = (
            "On OIA-DDR, the paired AURC difference to the three-pass mean is "
            f"{p['Full gate (S+E)']['delta']:+.4f} for the full gate "
            f"(95\\% CI [{p['Full gate (S+E)']['ci95'][0]:+.4f}, "
            f"{p['Full gate (S+E)']['ci95'][1]:+.4f}]) and "
            f"{p['Fusion-input gate (F+E)']['delta']:+.4f} for the gate that reads the averaged score "
            f"(95\\% CI [{p['Fusion-input gate (F+E)']['ci95'][0]:+.4f}, "
            f"{p['Fusion-input gate (F+E)']['ci95'][1]:+.4f}])."
        )

    caption = (
        "Selective prediction and conformal operating point on the retain-versus-suppress task. "
        "AURC is the area under the risk-coverage curve, where risk is the share of detections "
        "that should have been suppressed among those retained, and lower is better. "
        "The coverage gap is the empirical minus the nominal coverage of a split-conformal "
        f"threshold calibrated on the OIA-DDR validation split at a nominal {nominal:.2f}. "
        "Best AURC per dataset in bold. "
        "The three-pass score mean averages the detector scores of the primary, flipped and secondary "
        "passes that the evidence also uses. "
        "Learned rows average five gate seeds, and maximum-softmax probability and predictive "
        "entropy are omitted."
    )

    s15 = (
        "% Generated by tools/make_selective_conformal_tables.py from frozen JSON; no refitting.\n"
        "% Source: outputs/0506/selective_conformal/baselines.json; three-pass mean from step5/step6 JSON\n"
        f"% sha256[:16] = {dig}\n"
        "\\begin{table}[htbp]\n\\centering\n"
        "\\caption{" + caption + "}\n"
        "\\label{tab:selective_conformal}\n\\footnotesize\n"
        "\\setlength{\\tabcolsep}{4pt}\n\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{lrrrrcccc}\n\\toprule\n"
        "\\multirow{2}{*}{Method} & \\multicolumn{4}{c}{AURC (lower is better)} "
        "& \\multicolumn{4}{c}{Conformal coverage gap} \\\\\n"
        "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\n"
        " & OIA-DDR & IDRiD & TJDR & FGADR & OIA-DDR & IDRiD & TJDR & FGADR \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}\n"
        + (f"\\par\\smallskip{{\\footnotesize {note}}}\n" if note else "")
        + "\\end{table}\n"
    )
    (_TABLES / "S15_selective_conformal.tex").write_text(s15, encoding="utf-8")

    # ---------------- S16
    order = None
    crows = []
    for c in COHORTS:
        cal = d["cohorts"][c]["calibration"]
        gm = gcal["cohorts"][c]["methods"]
        gpub = gcal["cohorts"][c]["published_variants"]
        if order is None:
            order = list(cal.keys())
        for k in order:  # the two sources have to agree on the three published variants
            for metric in ("ece", "brier"):
                if abs(gpub[k][metric] - cal[k][metric]) > 1e-9:
                    raise SystemExit(f"S16: {c} {k} {metric} berbeda antara baselines.json dan "
                                     "step4_gate_calibration.json")
        cells = [c]
        cells += [f"{cal[k]['ece']:.4f}" for k in order]
        cells += [f"{gm[k]['ece_mean']:.4f}" for k, _ in GATES]
        cells += [f"{cal[k]['brier']:.4f}" for k in order]
        cells += [f"{gm[k]['brier_mean']:.4f}" for k, _ in GATES]
        crows.append(" & ".join(cells) + r" \\")

    names = [k.split(" (")[0].capitalize() if k.startswith("temperature") else k.capitalize()
             for k in order] + [lab for _, lab in GATES]
    nc = len(names)
    ccap = (
        "Expected calibration error and Brier score of the detector score and of the two gates. Temperature "
        "scaling uses the parameter of the benchmark table of the main article. The gate columns are five-seed "
        "means, and lower is better for both metrics."
    )
    cnote = (
        "The target is the RETAIN versus SUPPRESS governance label of Section 4.4 of the main article, and every "
        "fitted mapping comes from the OIA-DDR validation split. Raw and temperature-scaled scores are min-max "
        "rescaled to the unit interval, which is not a probability transform, while the gate sigmoids are read "
        "directly."
    )
    s16 = (
        "% Generated by tools/make_selective_conformal_tables.py from frozen JSON; no refitting.\n"
        f"% sha256[:16] = {dig}\n"
        f"% gate calibration source sha256[:16] = {gdig}\n"
        "\\begin{table}[htbp]\n\\centering\n"
        "\\caption{" + ccap + "}\n"
        "\\label{tab:calibration_family}\n\\footnotesize\n"
        "\\setlength{\\tabcolsep}{4pt}\n\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{l" + "r" * (2 * nc) + "}\n\\toprule\n"
        f"\\multirow{{2}}{{*}}{{Dataset}} & \\multicolumn{{{nc}}}{{c}}{{Expected calibration error}} "
        f"& \\multicolumn{{{nc}}}{{c}}{{Brier score}} \\\\\n"
        f"\\cmidrule(lr){{2-{nc + 1}}}\\cmidrule(lr){{{nc + 2}-{2 * nc + 1}}}\n"
        " & " + " & ".join(names) + " & " + " & ".join(names) + " \\\\\n"
        "\\midrule\n" + "\n".join(crows) + "\n\\bottomrule\n\\end{tabular}}\n"
        f"\\par\\smallskip{{\\footnotesize {cnote}}}\n"
        "\\end{table}\n"
    )
    (_TABLES / "S16_calibration_family.tex").write_text(s16, encoding="utf-8")

    print(f"ditulis {_TABLES / 'S15_selective_conformal.tex'}")
    print(f"ditulis {_TABLES / 'S16_calibration_family.tex'}")
    print(f"sumber sha256[:16] = {dig}")


if __name__ == "__main__":
    main()
