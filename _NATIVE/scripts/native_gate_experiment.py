"""native_gate_experiment.py — eksperimen PRA-REGISTRASI gate graph-attention KG-native.

Pertanyaan: apakah gate yang membaca knowledge graph secara native (graph attention atas
9 node KG, lalu attention yang dikondisikan pada tiap deteksi) mengungguli gate MLP penuh
yang dipakai naskah, dan apakah keunggulan itu datang dari STRUKTUR graf?

SELURUH HIPERPARAMETER, KRITERIA, DAN ATURAN TAFSIR DIPATOK DI SINI SEBELUM DIJALANKAN.
Tidak ada penyetelan pada data test. Hasil dilaporkan apa pun keluarannya.

Varian (semua dipasang pada OIA-DDR validasi, 5 seed, dievaluasi pada OIA-DDR test dan
tiga kohort eksternal zero-shot):
  G0   gate MLP penuh naskah (fit_mlp atas SAS, SDCA, APS, PSS, skor, sds, nsrc).
       KONTROL POSITIF: rerata AP OIA-DDR test harus mereproduksi 0.69855 (toleransi 0.001).
  G0+  MLP atas masukan per-deteksi YANG SAMA dengan G1 (7 skalar, distribusi 4 kelas,
       one-hot top-1 dan top-2), TANPA graf. Memisahkan efek masukan dari efek graf.
  G1   gate KG-native: HeteroGAT 2 lapis atas 38 edge KG (5 tipe + self-loop; bobot edge
       masuk ke logit attention), lalu attention berkondisi-deteksi atas 9 node, lalu head.
  G2   G1 identik, tetapi endpoint tujuan edge DIACAK di dalam tiap tipe edge (berbeda per
       seed). Menguji apakah struktur graf berpengaruh.

Hiperparameter terkunci (G0+, G1, G2): d=16, 2 head, 2 lapis GAT, head 16 tanh; G0+ memakai
dua lapis tersembunyi 32 dan 16 agar kekalahannya tidak dapat dijelaskan oleh kapasitas.
Adam lr 0.01, weight decay 1e-4, 400 epoch full-batch, BCE, tanpa early stopping.
Skalar distandardisasi dengan rerata dan simpangan baku VALIDASI, sama seperti fit_mlp.

KETERBATASAN DATA YANG DIKETAHUI SEBELUM DIJALANKAN:
  - radial_zone KOSONG pada setiap deteksi di setiap kohort. Lima dari sembilan node (zona)
    tidak dapat dikaitkan ke deteksi; mereka hanya berperan sebagai konteks message passing.
  - distribusi kelas detektor SEKUNDER tidak tersimpan; hanya ev_sds (JSD skalar) dan
    ev_sds_nsrc. Attention multi-source atas distribusi dua detektor tidak dapat dibangun.

KRITERIA LOLOS (seluruhnya wajib):
  C1  rerata(AP G1 - AP G0) >= +0.010 pada OIA-DDR test, DAN CI bootstrap > 0 pada >= 4/5 seed
  C2  G1 > G2 dengan CI bootstrap > 0 pada >= 4/5 seed
  C3  rerata AP G1 >= rerata AP G0 pada >= 2 dari 3 kohort eksternal
ATURAN TAFSIR (dipatok di muka): bila LOLOS tetapi G1 tidak mengungguli G0+ dengan CI > 0
pada >= 4/5 seed, keunggulan itu milik MASUKAN tambahan, bukan graf, dan TIDAK BOLEH
diklaim sebagai kontribusi KG-native.

Bootstrap: berklaster-citra, berpasangan, 2.000 resample, benih 0 (sama dengan naskah).

Isolasi: skrip hanya MEMBACA CSV bukti, graf dan modul tools di lokasi aslinya. Seluruh
keluaran ditulis di bawah _NATIVE/. Hash md5 kode dan graf yang dipakai dicatat di JSON.

Jalankan dari root repositori:
  python _NATIVE/scripts/native_gate_experiment.py           # eksperimen resmi
  python _NATIVE/scripts/native_gate_experiment.py --quick   # uji asap, berkas terpisah
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

HERE = Path(__file__).resolve()
NATIVE = HERE.parents[1]
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from tools.eval_fp_reduction_benchmark import apply_mlp, fit_mlp, load_csv  # noqa: E402
from tools.make_indomain_results import (  # noqa: E402
    N_BOOT, POSITIVE, SEEDS, T_DEP, V_DEP, gate_matrix, paired_bootstrap_ap,
)

BOOT_SEED = 0
GRAPH = ROOT / "outputs/stage3_audit/kg_graph.json"
EXTERNAL = [
    ("IDRiD", ROOT / "outputs/0506/evsds_pss/idrid/test/governance_features_test.csv"),
    ("TJDR", ROOT / "outputs/0506/evsds_pss/tjdr/test/governance_features_test.csv"),
    ("FGADR", ROOT / "outputs/0506/evsds_pss/fgadr/train/governance_features_train.csv"),
]
PROB_COLS = ("prob_ma", "prob_he", "prob_hardex", "prob_softex")
LESIONS = ("MA", "HE", "Hard_EX", "Soft_EX")
SCALARS = ("ev_sas", "ev_sdca", "ev_aps", "ev_pss", "pred_score", "ev_sds", "ev_sds_nsrc")
POS_CONTROL_G0 = 0.69855
POS_CONTROL_TOL = 1e-3
REF_FUSION = 0.7224
REF_GATE_ON_FUSION = 0.7332

HP = {"d": 16, "heads": 2, "gat_layers": 2, "head_hidden": 16, "g0plus_hidden": [32, 16],
      "lr": 1e-2, "weight_decay": 1e-4, "epochs": 400, "shuffle_seed_offset": 1000,
      "torch_threads": 8}
CRIT = {"c1_min_mean_gain": 0.010, "c1_min_seeds_ci_pos": 4, "c2_min_seeds_ci_pos": 4,
        "c3_min_external_cohorts": 2, "graph_attribution_min_seeds_ci_pos": 4}

T0 = time.perf_counter()


def log(msg):
    print(f"[native] {msg} ({time.perf_counter() - T0:.0f}s)", flush=True)


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------------- data
def read_probs(path, n_expected):
    """Kolom probabilitas dibaca dalam urutan baris yang sama dengan load_csv.
    load_csv(governance_retain) tidak membuang baris; jumlahnya tetap diasersi."""
    probs = np.empty((n_expected, 4), dtype=np.float64)
    n = 0
    with open(path, newline="", encoding="utf-8") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            if i >= n_expected:
                raise SystemExit(f"BERHENTI: {path} punya baris lebih banyak dari label")
            probs[i] = [float(row[c]) for c in PROB_COLS]
            n = i + 1
    if n != n_expected:
        raise SystemExit(f"BERHENTI: baris probabilitas {n} != label {n_expected} ({path})")
    if not np.isfinite(probs).all():
        raise SystemExit(f"BERHENTI: probabilitas tidak hingga di {path}")
    return probs / np.clip(probs.sum(1, keepdims=True), 1e-12, None)


def load_cohort(path):
    d = load_csv(path, POSITIVE)
    d["probs"] = read_probs(path, len(d["label"]))
    d["scalars"] = np.stack([d[c] for c in SCALARS], 1)
    return d


def validate_graph(graph):
    """Graph contract check: ordered nodes, expected edge types, and feature dims."""
    nodes = sorted(graph["nodes"], key=lambda n: n["idx"])
    names = [n["name"] for n in nodes]
    if tuple(names[:4]) != LESIONS:
        raise SystemExit("BERHENTI: urutan node KG tidak sesuai (4 lesi lalu 5 zona)")
    if any(n["node_type"] != "zone" for n in nodes[4:]):
        raise SystemExit("BERHENTI: node zona dalam KG tidak konsisten")
    required_edges = [
        "morphological_confusability",
        "pathophysiological_mechanism",
        "spatial_distribution",
        "severity_transition",
        "co_occurrence_pattern",
    ]
    missing = [t for t in required_edges if t not in graph["edges"]]
    if missing:
        raise SystemExit(f"BERHENTI: edge type KG tidak lengkap: {missing}")
    for t, edges in graph["edges"].items():
        if not edges:
            raise SystemExit(f"BERHENTI: edge type KG kosong: {t}")
    for nm in names[:4]:
        if nm not in graph["node_features"]:
            raise SystemExit(f"BERHENTI: feature node lesi hilang: {nm}")
    for nm in names[4:]:
        if nm not in graph["node_features"]:
            raise SystemExit(f"BERHENTI: feature node zona hilang: {nm}")
    if any(len(graph["node_features"][nm]) != (17 if i < 4 else 9) for i, nm in enumerate(names)):
        raise SystemExit("BERHENTI: dimensi fitur node KG tidak sesuai kontrak 17/9")
    return True


def load_graph(shuffle_seed=None):
    g = json.loads(GRAPH.read_text(encoding="utf-8"))
    validate_graph(g)
    nodes = sorted(g["nodes"], key=lambda n: n["idx"])
    names = [n["name"] for n in nodes]
    xl = np.array([g["node_features"][nm] for nm in names[:4]], dtype=np.float32)
    xz = np.array([g["node_features"][nm] for nm in names[4:]], dtype=np.float32)
    etypes = list(g["edges"].keys())
    rng = np.random.default_rng(shuffle_seed) if shuffle_seed is not None else None
    src, dst, et, w = [], [], [], []
    n_edges, n_rewired = 0, 0
    for ti, t in enumerate(etypes):
        edges = g["edges"][t]
        dsts = [int(e["dst"]) for e in edges]
        new = [int(x) for x in rng.permutation(dsts)] if rng is not None else dsts
        for e, d_old, d_new in zip(edges, dsts, new):
            s_, wt = int(e["src"]), float(e["weight"])
            n_edges += 1
            n_rewired += int(d_old != d_new)
            src.append(s_); dst.append(d_new); et.append(ti); w.append(wt)
            if not e["directed"]:
                src.append(d_new); dst.append(s_); et.append(ti); w.append(wt)
    for v in range(len(names)):
        src.append(v); dst.append(v); et.append(len(etypes)); w.append(1.0)
    return {"names": names, "etypes": etypes, "xl": xl, "xz": xz,
            "src": torch.tensor(src), "dst": torch.tensor(dst), "et": torch.tensor(et),
            "w": torch.tensor(w, dtype=torch.float32), "n_etypes": len(etypes) + 1,
            "n_edges": n_edges, "n_rewired": n_rewired}


# ------------------------------------------------------------------------ model
def scatter_softmax(logit, index, n):
    ex = torch.exp(logit.clamp(-30.0, 30.0))
    denom = torch.zeros(n, logit.shape[1]).index_add(0, index, ex)
    return ex / denom[index].clamp_min(1e-12)


class HeteroGATLayer(nn.Module):
    def __init__(self, d, heads, n_etypes):
        super().__init__()
        self.d, self.heads = d, heads
        self.proj = nn.Linear(d, d * heads, bias=False)
        self.att_src = nn.Parameter(torch.empty(heads, d))
        self.att_dst = nn.Parameter(torch.empty(heads, d))
        nn.init.normal_(self.att_src, std=0.1)
        nn.init.normal_(self.att_dst, std=0.1)
        self.etype = nn.Embedding(n_etypes, heads)
        self.out = nn.Linear(d * heads, d)

    def forward(self, h, src, dst, et, w):
        n = h.shape[0]
        z = self.proj(h).view(n, self.heads, self.d)
        logit = (z[src] * self.att_src).sum(-1) + (z[dst] * self.att_dst).sum(-1)
        logit = F.leaky_relu(logit + self.etype(et) + torch.log(w.clamp_min(1e-3))[:, None], 0.2)
        att = scatter_softmax(logit, dst, n)
        agg = torch.zeros(n, self.heads, self.d).index_add(0, dst, z[src] * att[..., None])
        return h + F.elu(self.out(agg.reshape(n, self.heads * self.d)))


class NativeGate(nn.Module):
    """G1/G2. Graf ikut per deteksi lewat (a) embedding lesi berbobot distribusi kelas,
    (b) selisih embedding top-1 dan top-2, (c) attention berkondisi-deteksi atas 9 node."""

    def __init__(self, graph, n_scalars):
        super().__init__()
        d = HP["d"]
        self.d = d
        self.register_buffer("xl", torch.tensor(graph["xl"]))
        self.register_buffer("xz", torch.tensor(graph["xz"]))
        self.register_buffer("e_src", graph["src"].clone())
        self.register_buffer("e_dst", graph["dst"].clone())
        self.register_buffer("e_type", graph["et"].clone())
        self.register_buffer("e_w", graph["w"].clone())
        self.in_lesion = nn.Linear(graph["xl"].shape[1], d)
        self.in_zone = nn.Linear(graph["xz"].shape[1], d)
        self.gat = nn.ModuleList([HeteroGATLayer(d, HP["heads"], graph["n_etypes"])
                                  for _ in range(HP["gat_layers"])])
        self.q = nn.Linear(n_scalars + d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.head = nn.Sequential(nn.Linear(n_scalars + 3 * d, HP["head_hidden"]), nn.Tanh(),
                                  nn.Linear(HP["head_hidden"], 1))

    def node_embeddings(self):
        h = torch.cat([F.elu(self.in_lesion(self.xl)), F.elu(self.in_zone(self.xz))], 0)
        for layer in self.gat:
            h = layer(h, self.e_src, self.e_dst, self.e_type, self.e_w)
        return h

    def attention(self, S, P):
        h = self.node_embeddings()
        e = P @ h[:4]
        return torch.softmax(self.q(torch.cat([S, e], 1)) @ self.k(h).T / self.d ** 0.5, dim=1)

    def forward(self, S, P):
        h = self.node_embeddings()
        hl = h[:4]
        e = P @ hl
        top2 = P.topk(2, dim=1).indices
        diff = hl[top2[:, 0]] - hl[top2[:, 1]]
        att = torch.softmax(self.q(torch.cat([S, e], 1)) @ self.k(h).T / self.d ** 0.5, dim=1)
        ctx = att @ self.v(h)
        return self.head(torch.cat([S, e, diff, ctx], 1)).squeeze(1)


class InputMLP(nn.Module):
    """G0+. Informasi per deteksi identik dengan G1, tanpa graf."""

    def __init__(self, n_scalars, hidden):
        super().__init__()
        layers, prev = [], n_scalars + 12
        for hdim in hidden:
            layers += [nn.Linear(prev, hdim), nn.Tanh()]
            prev = hdim
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, S, P):
        top2 = P.topk(2, dim=1).indices
        oh1 = F.one_hot(top2[:, 0], 4).float()
        oh2 = F.one_hot(top2[:, 1], 4).float()
        return self.net(torch.cat([S, P, oh1, oh2], 1)).squeeze(1)


def train_torch(build, S, P, y, seed, epochs):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build()
    opt = torch.optim.Adam(model.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"])
    yt = torch.tensor(y, dtype=torch.float32)
    model.train()
    loss = None
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(S, P), yt)
        loss.backward()
        opt.step()
    model.eval()
    return model, float(loss.detach())


@torch.no_grad()
def score(model, S, P, chunk=100_000):
    out = [torch.sigmoid(model(S[i:i + chunk], P[i:i + chunk])) for i in range(0, len(S), chunk)]
    return torch.cat(out).numpy().astype(np.float64)


# ------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="uji asap: 1 seed, 30 epoch, tanpa bootstrap, tanpa FGADR")
    args = ap.parse_args()
    torch.set_num_threads(HP["torch_threads"])

    seeds = (0,) if args.quick else tuple(SEEDS)
    epochs = 30 if args.quick else HP["epochs"]
    do_boot = not args.quick
    externals = [c for c in EXTERNAL if not (args.quick and c[0] == "FGADR")]
    out_path = NATIVE / "results" / ("native_gate_quick.json" if args.quick else "native_gate_experiment.json")
    log(f"mode={'QUICK' if args.quick else 'RESMI'} seeds={seeds} epochs={epochs} bootstrap={do_boot}")

    va, te = load_cohort(V_DEP), load_cohort(T_DEP)
    ext = {name: load_cohort(p) for name, p in externals}
    log(f"data: valid {len(va['label'])}, test {len(te['label'])}, "
        + ", ".join(f"{k} {len(v['label'])}" for k, v in ext.items()))

    mu = va["scalars"].mean(0)
    sd = va["scalars"].std(0) + 1e-8

    def tens(d):
        return (torch.tensor((d["scalars"] - mu) / sd, dtype=torch.float32),
                torch.tensor(d["probs"], dtype=torch.float32))

    Sv, Pv = tens(va)
    St, Pt = tens(te)
    ext_t = {k: tens(v) for k, v in ext.items()}
    yv = va["label"].astype(np.float64)
    yt = te["label"]
    Xv0, Xt0 = gate_matrix(va, "full_gate", None), gate_matrix(te, "full_gate", None)
    Xe0 = {k: gate_matrix(v, "full_gate", None) for k, v in ext.items()}

    graph = load_graph(None)
    log(f"graf: {len(graph['names'])} node, {graph['n_edges']} edge, {len(graph['etypes'])} tipe")

    variants = ("G0", "G0+", "G1", "G2")
    rec = {v: {"test_ap": [], "ext_ap": {k: [] for k in ext}, "train_loss": []} for v in variants}
    boots = {"G1_vs_G0": [], "G1_vs_G2": [], "G1_vs_G0+": []}
    shuffle_info, attention_mass = [], []

    for s in seeds:
        m0 = fit_mlp(Xv0, yv, seed=s)
        sc = {"G0": (apply_mlp(Xt0, m0), {k: apply_mlp(Xe0[k], m0) for k in ext})}
        rec["G0"]["train_loss"].append(None)

        mp, lp = train_torch(lambda: InputMLP(len(SCALARS), HP["g0plus_hidden"]), Sv, Pv, yv, s, epochs)
        sc["G0+"] = (score(mp, St, Pt), {k: score(mp, *ext_t[k]) for k in ext})
        rec["G0+"]["train_loss"].append(lp)

        m1, l1 = train_torch(lambda: NativeGate(graph, len(SCALARS)), Sv, Pv, yv, s, epochs)
        sc["G1"] = (score(m1, St, Pt), {k: score(m1, *ext_t[k]) for k in ext})
        rec["G1"]["train_loss"].append(l1)
        with torch.no_grad():
            mass = m1.attention(St, Pt).mean(0).numpy()
        attention_mass.append({"seed": s, "lesion_nodes": float(mass[:4].sum()),
                               "zone_nodes": float(mass[4:].sum()),
                               "per_node": [float(x) for x in mass]})

        g_shuf = load_graph(s + HP["shuffle_seed_offset"])
        shuffle_info.append({"seed": s, "shuffle_seed": s + HP["shuffle_seed_offset"],
                             "n_edges": g_shuf["n_edges"], "n_rewired": g_shuf["n_rewired"]})
        m2, l2 = train_torch(lambda: NativeGate(g_shuf, len(SCALARS)), Sv, Pv, yv, s, epochs)
        sc["G2"] = (score(m2, St, Pt), {k: score(m2, *ext_t[k]) for k in ext})
        rec["G2"]["train_loss"].append(l2)

        for v in variants:
            test_s, ext_s = sc[v]
            rec[v]["test_ap"].append(float(average_precision_score(yt, test_s)))
            for k in ext:
                rec[v]["ext_ap"][k].append(float(average_precision_score(ext[k]["label"], ext_s[k])))
        log(f"seed {s}: test AP " + "  ".join(f"{v} {rec[v]['test_ap'][-1]:.5f}" for v in variants)
            + f" | G2 rewired {g_shuf['n_rewired']}/{g_shuf['n_edges']}")

        if do_boot:
            for key, (a, b) in {"G1_vs_G0": ("G1", "G0"), "G1_vs_G2": ("G1", "G2"),
                                "G1_vs_G0+": ("G1", "G0+")}.items():
                bs = paired_bootstrap_ap(sc[a][0], sc[b][0], yt, te["image_id"], N_BOOT, BOOT_SEED)
                bs["gain_point"] = rec[a]["test_ap"][-1] - rec[b]["test_ap"][-1]
                bs["seed"] = s
                boots[key].append(bs)
            log(f"seed {s}: bootstrap " + "  ".join(
                f"{k} {boots[k][-1]['gain_point']:+.5f} CI[{boots[k][-1]['gain_ci95'][0]:+.4f},"
                f"{boots[k][-1]['gain_ci95'][1]:+.4f}]" for k in boots))

    summary = {v: {"test_ap_mean": float(np.mean(rec[v]["test_ap"])),
                   "test_ap_sd": float(np.std(rec[v]["test_ap"], ddof=1)) if len(seeds) > 1 else None,
                   "ext_ap_mean": {k: float(np.mean(rec[v]["ext_ap"][k])) for k in ext}}
               for v in variants}

    g0_mean = summary["G0"]["test_ap_mean"]
    tol = POS_CONTROL_TOL if not args.quick else 5e-3
    pos_ok = abs(g0_mean - POS_CONTROL_G0) <= tol

    verdict = {"positive_control_g0": {"expected": POS_CONTROL_G0, "observed": g0_mean,
                                       "tolerance": tol, "passed": bool(pos_ok)}}
    if not pos_ok:
        verdict["status"] = "INVALID: kontrol positif G0 gagal, hasil tidak boleh ditafsirkan"
    elif do_boot:
        gains = [a - b for a, b in zip(rec["G1"]["test_ap"], rec["G0"]["test_ap"])]
        gains_g0plus = [a - b for a, b in zip(rec["G1"]["test_ap"], rec["G0+"]["test_ap"])]
        n_sig = {k: int(sum(b["sig"] for b in v)) for k, v in boots.items()}
        c1 = (float(np.mean(gains)) >= CRIT["c1_min_mean_gain"]
              and n_sig["G1_vs_G0"] >= CRIT["c1_min_seeds_ci_pos"])
        c2 = n_sig["G1_vs_G2"] >= CRIT["c2_min_seeds_ci_pos"]
        c3_per = {k: summary["G1"]["ext_ap_mean"][k] >= summary["G0"]["ext_ap_mean"][k] for k in ext}
        c3 = len(ext) == 3 and sum(c3_per.values()) >= CRIT["c3_min_external_cohorts"]
        passed = bool(c1 and c2 and c3)
        graph_attr = bool(
            passed
            and float(np.mean(gains_g0plus)) > 0.0
            and n_sig["G1_vs_G0+"] >= CRIT["graph_attribution_min_seeds_ci_pos"]
        )
        verdict.update({
            "C1": {"mean_gain_g1_g0": float(np.mean(gains)), "n_seeds_ci_pos": n_sig["G1_vs_G0"], "passed": bool(c1)},
            "C2": {"n_seeds_ci_pos_g1_g2": n_sig["G1_vs_G2"], "passed": bool(c2)},
            "C3": {"per_cohort_g1_ge_g0": c3_per, "passed": bool(c3)},
            "g1_vs_g0plus_mean_gain": float(np.mean(gains_g0plus)),
            "g1_vs_g0plus_n_seeds_ci_pos": n_sig["G1_vs_G0+"],
            "PASSED": passed,
            "graph_attributable": graph_attr,
            "status": ("LOLOS dan dapat diatribusikan ke graf" if graph_attr else
                       "LOLOS tetapi keunggulan milik masukan, bukan graf" if passed else
                       "TIDAK LOLOS"),
        })
    else:
        verdict["status"] = "UJI ASAP: tanpa bootstrap, kriteria tidak dievaluasi"

    verdict["descriptive_reference"] = {
        "score_fusion_in_domain": REF_FUSION, "gate_on_fusion_in_domain": REF_GATE_ON_FUSION,
        "g1_minus_fusion": summary["G1"]["test_ap_mean"] - REF_FUSION,
    }

    provenance = {
        "code_md5": {p: md5(ROOT / p) for p in ("tools/eval_fp_reduction_benchmark.py",
                                                  "tools/make_indomain_results.py")},
        "script_md5": md5(HERE),
        "graph_md5": md5(GRAPH),
        "csv": {str(p.relative_to(ROOT)): {"bytes": p.stat().st_size, "mtime": p.stat().st_mtime}
                for p in [V_DEP, T_DEP] + [pp for _, pp in externals]},
        "torch": torch.__version__,
    }
    result = {"mode": "quick" if args.quick else "official", "seeds": list(seeds), "epochs": epochs,
              "hyperparameters": HP, "criteria": CRIT, "per_seed": rec, "summary": summary,
              "bootstrap": boots, "verdict": verdict, "g2_shuffle": shuffle_info,
              "g1_attention_mass_test": attention_mass, "provenance": provenance,
              "known_limitations": ["radial_zone kosong di semua kohort: node zona tidak terkait deteksi",
                                    "distribusi kelas detektor sekunder tidak tersimpan"],
              "elapsed_s": time.perf_counter() - T0}
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    log("RINGKASAN AP OIA-DDR test: " + "  ".join(
        f"{v} {summary[v]['test_ap_mean']:.5f}" for v in variants))
    for k in ext:
        log(f"RINGKASAN AP {k}: " + "  ".join(f"{v} {summary[v]['ext_ap_mean'][k]:.5f}" for v in variants))
    log(f"kontrol positif G0: {g0_mean:.5f} (harapan {POS_CONTROL_G0}) -> {'LOLOS' if pos_ok else 'GAGAL'}")
    log(f"STATUS: {verdict['status']}")
    log(f"ditulis {out_path}")


if __name__ == "__main__":
    main()
