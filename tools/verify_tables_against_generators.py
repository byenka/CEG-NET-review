"""Check that every generated LaTeX table in the canonical submission folder
(_SUBMISSION_ESWA) still matches what its generator produces today.

Each generator is executed with all file writes redirected into a temporary
directory, so the shipped tables are never touched; the redirected output is
then compared byte for byte with the shipped file. The script exits non-zero if
any table has drifted, and reports separately whether the drift reaches the
numbers inside the tabular body or stops at the caption.

    python tools/verify_tables_against_generators.py [--keep DIR] [--against DIR]

--against defaults to _SUBMISSION_ESWA; pass _FINAL_Latex/tables to compare with
the older working copy instead. In the code repository, where _SUBMISSION_ESWA does
not exist, it defaults to results/published_tables/, the tables as typeset in the
paper, and the generators are looked up under scripts/ instead of the workspace
folders (same files, see REPO_PATH).

Hand-maintained tables (no generator) are listed but not checked; they are
listed in _FINAL_Latex/tables/REGENERATE.md.
"""
import argparse
import builtins
import difflib
import os
import re
import runpy
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANON = (ROOT / "_SUBMISSION_ESWA" if (ROOT / "_SUBMISSION_ESWA").is_dir()
         else ROOT / "results/published_tables")
LEGACY = ROOT / "_FINAL_Latex/tables"
# In the legacy working copy T18 lives beside its experiment, not among the tables.
LEGACY_EXTRA = {"T18_honest_calibration.tex":
                ROOT / "_FINAL_Latex/experiments/honest_calibration"}
BS = chr(92)

# Generators that write into _FINAL_Latex/tables, with their arguments.
# render_final_tables.py runs last: besides rendering T15, S10 and S11 it applies
# editorial overlays to tables produced earlier in the list, which are no-ops on
# the current text of T8 and S6.
GENERATORS = [
    "tools/make_dataset_stats_table.py --write",
    "tools/make_indomain_results.py --tables_only",
    "tools/make_core_tables.py",
    "tools/make_kg_coupled_training_table.py",
    "_septem/scripts_optionB/make_s5_circularity.py",
    "tools/make_supplementary_tables.py",
    "tools/make_selective_conformal_tables.py",
    "tools/make_ensemble_bound_table.py",
    "tools/make_certificate_verification_table.py",
    "tools/make_xml_provenance_table.py",
    "_NATIVE/scripts/make_annotation_endpoint_table.py",
    "_LGOVNET_NEW_SEP/scripts/make_mondrian_table.py",
    "_LGOVNET_NEW_SEP/scripts/make_conditional_coverage_tables.py",
    "_LGOVNET_NEW_SEP/scripts/make_alpha_sensitivity_table.py",
    "_septem/scripts_optionB/make_s6_clinical_cost_external.py",
    "_septem/scripts_optionB/make_s7_target_recalibration.py",
    "_septem/scripts_optionB/make_s9_source_domain_sensitivity.py",
    "tools/make_tier_routing.py",
    "tools/make_honest_calibration_table.py",
    "tools/make_exact_tp_table.py",
    "_septem/scripts_optionB/make_s8_net_benefit.py",
    "_septem/finalization_five_priorities/render_final_tables.py",
]
# Where the code repository keeps each workspace script folder.
REPO_PATH = {
    "_NATIVE/scripts/": "scripts/",
    "_LGOVNET_NEW_SEP/scripts/": "scripts/conditional_coverage/",
    "_septem/scripts_optionB/": "scripts/",
    "_septem/finalization_five_priorities/": "scripts/finalization/",
}


def locate(script):
    """Workspace path if it exists, otherwise the repository copy of the same script."""
    if (ROOT / script).exists():
        return script
    for ws, repo in REPO_PATH.items():
        if script.startswith(ws) and (ROOT / (repo + script[len(ws):])).exists():
            return repo + script[len(ws):]
    return script


def run_redirected(script, outdir, args=()):
    """Run a generator with every write sent to outdir, keeping reads intact."""
    outdir.mkdir(parents=True, exist_ok=True)
    o_wt, o_wb, o_po, o_bo = (Path.write_text, Path.write_bytes,
                              Path.open, builtins.open)

    def dest(p):
        return outdir / Path(p).name

    Path.write_text = lambda self, d, *a, **k: o_wt(dest(self), d, *a, **k)
    Path.write_bytes = lambda self, d, *a, **k: o_wb(dest(self), d, *a, **k)
    Path.open = lambda self, mode="r", *a, **k: o_po(
        dest(self) if any(m in mode for m in "wax") else self, mode, *a, **k)

    def bopen(file, mode="r", *a, **k):
        if isinstance(file, (str, os.PathLike)) and any(m in mode for m in "wax"):
            file = dest(file)
        return o_bo(file, mode, *a, **k)

    builtins.open = bopen
    argv, cwd = sys.argv, os.getcwd()
    try:
        os.chdir(ROOT)
        sys.argv = [str(ROOT / script), *args]
        try:
            runpy.run_path(str(ROOT / script), run_name="__main__")
        except SystemExit as code:            # generators end with raise SystemExit(main())
            if code.code not in (None, 0):
                raise RuntimeError(f"generator exited with {code.code}")
    finally:
        Path.write_text, Path.write_bytes = o_wt, o_wb
        Path.open, builtins.open = o_po, o_bo
        sys.argv, _ = argv, os.chdir(cwd)


def body_numbers(t):
    parts = []
    for env in ("tabular", "longtable"):
        parts += re.findall(BS + BS + r"begin\{" + env + r"\}.*?"
                            + BS + BS + r"end\{" + env + r"\}", t, re.S)
    return re.findall(r"[-+]?\d+(?:[.,]\d+)*", "\n".join(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", help="write regenerated tables here instead of a temp dir")
    ap.add_argument("--against", default=str(CANON), help="folder of shipped tables to compare with")
    args = ap.parse_args()
    tables = Path(args.against)
    if not tables.is_absolute():
        tables = ROOT / tables
    extra = LEGACY_EXTRA if tables.resolve() == LEGACY.resolve() else {}
    print(f"comparing against {tables}")

    base = Path(args.keep) if args.keep else Path(tempfile.mkdtemp(prefix="tabcheck_"))
    produced, drift = {}, []
    for entry in GENERATORS:
        script, *args = entry.split()
        script = locate(script)
        out = base / Path(script).stem
        try:
            run_redirected(script, out, args)
        except Exception as exc:                      # noqa: BLE001
            print(f"[FAIL ] {script}: {type(exc).__name__}: {exc}")
            drift.append(script)
            continue
        for g in sorted(out.glob("*.tex")):
            produced.setdefault(g.name, []).append(g)

    print()
    for name in sorted(produced):
        shipped = extra.get(name, tables) / name
        if not shipped.exists():
            print(f"[MISS ] {name}: generated but not shipped")
            drift.append(name)
            continue
        st = shipped.read_text(encoding="utf-8").replace("\r\n", "\n")
        for g in produced[name]:
            gt = g.read_text(encoding="utf-8").replace("\r\n", "\n")
            if gt == st:
                print(f"[ok   ] {name}  ({g.parent.name})")
                continue
            kind = ("DATA" if body_numbers(gt) != body_numbers(st) else "CAPTION/WORDING")
            print(f"[DRIFT] {name}  ({g.parent.name})  {kind}")
            for line in list(difflib.unified_diff(
                    gt.split("\n"), st.split("\n"),
                    fromfile="generated", tofile="shipped", lineterm="", n=0))[2:]:
                print("        " + line[:200])
            drift.append(name)

    hand = sorted(p.name for p in tables.glob("*.tex")
                  if p.name not in produced and p.name not in ("main.tex", "supplementary.tex"))
    print("\nhand-maintained, not checked here:", ", ".join(hand) or "none")
    if drift:
        print(f"\n{len(drift)} table(s) drifted from their generator.")
        return 1
    print(f"\nall {len(produced)} generated tables match their generator.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
