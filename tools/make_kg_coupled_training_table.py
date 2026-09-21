#!/usr/bin/env python
"""make_kg_coupled_training_table.py — Tabel S4: knowledge graph di dalam pelatihan detektor.

Hasil negatif yang dirujuk Sec. 3.1, Limitations dan Conclusion. Sebelum tabel ini ada,
naskah menyatakan "coupling the knowledge graph into detector training does not improve
over standard fine-tuning" tanpa rujukan ke bukti.

Desain (alpha-PHASE 1.5, 2026-06-03): detektor beku r5b_ext25 epoch 18 di-fine-tune 20
epoch lagi pada OIA-DDR train dalam empat konfigurasi berpasangan, tiga seed masing-masing:
  B_none        : loss deteksi standar saja
  D_kg_only     : + lambda_KG * L_KG + lambda_gate * L_gate (encoder KG ikut dilatih)
  C_anchor_only : + lambda_anchor * L_anchor (distilasi ke detektor beku)
  A_full        : + ketiganya
Evaluasi pada OIA-DDR valid (149 gambar). Checkpoint tiap run dipilih pada split yang sama.

Sumber, semuanya dibaca, tidak diketik:
  outputs/_alpha_phase1p5_decomp/per_seed.csv
  <run_dir>/mode2_summary.json                          bobot loss, epoch, batch per run
  <run_dir>/audit_topk/clinical_metrics.json            dicatat untuk provenans
  outputs/detector_eval/r5b_baseline/stage1a_raw.json   detektor beku: mAP50
  outputs/mode2_v2_1_2026-06-01_crossdataset/baseline_r5b/oiaddr_valid/clinical_metrics.json
                                                        detektor beku: ECE
Keluaran:
  _FINAL_Latex/tables/S4_kg_coupled_training.tex
  outputs/_alpha_phase1p5_decomp/s4_table_values.json

Jalankan dari root repo:
  python tools/make_kg_coupled_training_table.py
"""
from __future__ import annotations

import csv
import json
import statistics
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DECOMP = ROOT / "outputs/_alpha_phase1p5_decomp"
PER_SEED = DECOMP / "per_seed.csv"
BASE_MAP = ROOT / "outputs/detector_eval/r5b_baseline/stage1a_raw.json"
BASE_CLIN = (ROOT / "outputs/mode2_v2_1_2026-06-01_crossdataset/baseline_r5b"
             / "oiaddr_valid/clinical_metrics.json")
TAB = ROOT / "_FINAL_Latex/tables/S4_kg_coupled_training.tex"
OUT_JSON = DECOMP / "s4_table_values.json"

ORDER = [
    ("B_none", "Standard fine-tuning"),
    ("D_kg_only", r"$+$ knowledge-graph losses"),
    ("C_anchor_only", r"$+$ distillation anchor"),
    ("A_full", r"$+$ knowledge-graph losses and anchor"),
]


def q(v, nd=3):
    return str(Decimal(repr(float(v))).quantize(Decimal(1).scaleb(-nd),
                                                rounding=ROUND_HALF_UP))


def signed(v, nd=3):
    return ("$+$" if v >= 0 else "$-$") + q(abs(v), nd)


def lam(v) -> str:
    v = float(v)
    return "0" if v == 0 else f"{v:g}"


def find_key(obj, key):
    """Cari kunci pertama secara rekursif; kembalikan (nilai, jalur)."""
    stack = [(obj, "")]
    while stack:
        cur, path = stack.pop(0)
        if isinstance(cur, dict):
            if key in cur:
                return cur[key], f"{path}/{key}"
            stack.extend((v, f"{path}/{k}") for k, v in cur.items())
        elif isinstance(cur, list):
            stack.extend((v, f"{path}[{i}]") for i, v in enumerate(cur))
    return None, None


def main() -> int:
    rows = list(csv.DictReader(open(PER_SEED, newline="", encoding="utf-8")))
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r["group"], []).append(r)

    values: dict = {"source": PER_SEED.relative_to(ROOT).as_posix(), "groups": {}}
    for g, rs in groups.items():
        rs.sort(key=lambda r: int(r["seed"]))
        cfg = set()
        clin_meta = set()
        for r in rs:
            s = json.loads((ROOT / r["run_dir"] / "mode2_summary.json").read_text(encoding="utf-8"))
            cfg.add((s["lambda_kg"], s["lambda_gate"], s["lambda_anchor"], s["epochs"],
                     s["batch_size"], s["freeze_kg"], s["kg_encoder_trainable"],
                     s.get("mode2a_preserve_standard_roi_loss"), s["data_root"],
                     s["train_images"], s["valid_images"]))
            cm = json.loads((ROOT / r["run_dir"] / "audit_topk/clinical_metrics.json")
                            .read_text(encoding="utf-8"))
            clin_meta.add((str(find_key(cm, "split")[0]),
                           str(find_key(cm, "box_score_thresh")[0])))
        if len(cfg) != 1:
            raise SystemExit(f"[s4] konfigurasi {g} tidak seragam antar-seed: {cfg}")
        (lkg, lgate, lanc, ep, bs, frz, trainable, std_roi, droot, ntr, nva) = cfg.pop()
        maps = [float(r["mAP_50"]) for r in rs]
        eces = [float(r["overall_ece"]) for r in rs]
        values["groups"][g] = {
            "seeds": [int(r["seed"]) for r in rs],
            "run_dirs": [r["run_dir"] for r in rs],
            "lambda_kg": lkg, "lambda_gate": lgate, "lambda_anchor": lanc,
            "epochs": ep, "batch_size": bs, "freeze_kg": frz,
            "kg_encoder_trainable": trainable, "standard_roi_loss": std_roi,
            "data_root": droot, "train_images": ntr, "valid_images": nva,
            "clinical_metrics_split_thresh": sorted(clin_meta),
            "mAP_50": maps, "mAP_50_mean": statistics.mean(maps),
            "mAP_50_sd": statistics.stdev(maps),
            "ece": eces, "ece_mean": statistics.mean(eces), "ece_sd": statistics.stdev(eces),
        }

    base = json.loads(BASE_MAP.read_text(encoding="utf-8"))
    bclin = json.loads(BASE_CLIN.read_text(encoding="utf-8"))
    b_ece, b_ece_path = find_key(bclin, "overall_ece")
    values["frozen"] = {
        "mAP_50": base["map_metrics"]["mAP_50"], "split": base.get("split"),
        "n_images": base.get("n_images"), "score_thresh": base.get("score_thresh"),
        "ece": b_ece, "ece_json_path": b_ece_path,
        "ece_split_thresh": (str(find_key(bclin, "split")[0]),
                             str(find_key(bclin, "box_score_thresh")[0])),
    }

    gb, gd = values["groups"]["B_none"], values["groups"]["D_kg_only"]
    if gb["seeds"] != gd["seeds"]:
        raise SystemExit("[s4] seed B dan D tidak berpasangan")
    d_map = [d - b for d, b in zip(gd["mAP_50"], gb["mAP_50"])]
    d_ece = [d - b for d, b in zip(gd["ece"], gb["ece"])]
    values["d_minus_b"] = {"mAP_50": d_map, "ece": d_ece,
                           "mAP_50_mean": statistics.mean(d_map),
                           "ece_mean": statistics.mean(d_ece)}

    epochs = {v["epochs"] for v in values["groups"].values()}
    nva = {v["valid_images"] for v in values["groups"].values()}
    if len(epochs) != 1 or len(nva) != 1:
        raise SystemExit(f"[s4] epoch/valid tidak seragam: {epochs} {nva}")
    n_ep, n_val = epochs.pop(), nva.pop()
    seeds = gb["seeds"]
    n_seed_txt = {2: "two", 3: "three", 4: "four", 5: "five"}.get(len(seeds), str(len(seeds)))

    body = [
        rf"Frozen detector (no fine-tuning) & -- & -- & -- & {q(values['frozen']['mAP_50'])} & "
        rf"{q(values['frozen']['ece'])} \\",
        r"\midrule",
    ]
    for g, label in ORDER:
        v = values["groups"][g]
        body.append(
            rf"{label} & {lam(v['lambda_kg'])} & {lam(v['lambda_gate'])} & "
            rf"{lam(v['lambda_anchor'])} & {q(v['mAP_50_mean'])} $\pm$ {q(v['mAP_50_sd'])} & "
            rf"{q(v['ece_mean'])} $\pm$ {q(v['ece_sd'])} \\")
    seed_txt = " / ".join(str(s) for s in seeds)
    body += [
        r"\midrule",
        (rf"\multicolumn{{4}}{{@{{}}l}}{{Knowledge-graph losses $-$ standard, seed {seed_txt}}} & "
         + " / ".join(signed(x) for x in d_map) + " & "
         + " / ".join(signed(x) for x in d_ece) + r" \\"),
    ]
    tex = "\n".join([
        "% S4 — DIBANGKITKAN tools/make_kg_coupled_training_table.py dari",
        "% outputs/_alpha_phase1p5_decomp/per_seed.csv dan mode2_summary.json tiap run.",
        "% JANGAN diedit tangan. SD = simpangan baku sampel atas seed.",
        r"\begin{table}[htbp]",
        r"\centering",
        (r"\caption{Knowledge graph inside detector training, on OIA-DDR validation "
         rf"({n_val} images). Each configuration fine-tunes the frozen detector for {n_ep} "
         rf"further epochs with {n_seed_txt} seeds; values are mean $\pm$ SD over seeds. ECE "
         r"is the expected calibration error of detection scores against IoU $\geq 0.5$ "
         r"matches, with ten bins. Lower ECE is better.}"),
        r"\label{tab:kg_coupled}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}lccccc@{}}",
        r"\toprule",
        (r"Configuration & $\lambda_{\mathrm{KG}}$ & $\lambda_{\mathrm{gate}}$ & "
         r"$\lambda_{\mathrm{anchor}}$ & $\mathrm{mAP}_{50}$ & ECE \\"),
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table}",
        "",
    ])
    TAB.write_text(tex, encoding="utf-8")
    OUT_JSON.write_text(json.dumps(values, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in values.items() if k != "groups"}, indent=1))
    for g, _ in ORDER:
        v = values["groups"][g]
        print(f"{g:14s} kg={v['lambda_kg']} gate={v['lambda_gate']} anchor={v['lambda_anchor']} "
              f"ep={v['epochs']} bs={v['batch_size']} std_roi={v['standard_roi_loss']} "
              f"clin={v['clinical_metrics_split_thresh']} "
              f"mAP={v['mAP_50_mean']:.4f}+-{v['mAP_50_sd']:.4f} "
              f"ECE={v['ece_mean']:.4f}+-{v['ece_sd']:.4f}")
    print(f"[s4] ditulis {TAB.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
