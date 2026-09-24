#!/usr/bin/env python3
"""
Rebuild Paper D supplementary zip (compiled manuscript PDF + scripts + data + figures).
=============================================================================
Type:       CLI
Reads:      paper_d/PAPER_D.pdf, scripts/s18..s40 (+ shared host modules),
            data/s18..s40, data/corpora/, figures/paperD_*.pdf
Produces:   paper_d/Supplementary_Material_PaperD.zip (local-only, gitignored)
=============================================================================
Mirrors scripts/rebuild_paperE_supp_zip.py / rebuild_paperF_supp_zip.py:
explicit allow-list, no internal-process artifacts, safety scan, and a
manuscript sha cross-check against the folder copy.
"""
from pathlib import Path
import hashlib
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = ROOT / "paper_d" / "Supplementary_Material_PaperD.zip"
SKIP_NAME_PARTS = ("__pycache__", ".pyc", ".nbc", ".nbi", ".DS_Store", "Thumbs.db")
# Prefer full-run data; skip smoke/screening artifacts if present
SKIP_PATTERNS = ("_quick.", "_quick_", ".bak")

README = """# Supplementary Code and Data — Paper D

> Paper D — "REDEM-SSM: A State-Space Architecture with Native Online
> Learning, Meta-Adaptation, and Structural Plasticity"

Contents: the compiled manuscript (PDF), the Paper D experiment scripts (s18–s35, plus
the P4 factor-ablation and ESN-baseline pair s38/s40), the shared host
modules they import (`recurrent_substrate.py`, `streaming_tasks.py`,
`per_token_io.py`, `shallow_trap_array_simulator.py`), committed result
tables used by the manuscript (including the public-domain real-text
corpora under `data/corpora/`), and figure generators.

Reproduce from the repository root (CPU-only; torch CPU):

    python scripts/s19_ssm_rls_readout.py --sequential
    python scripts/s20_ssm_m3_routing.py --sequential
    python scripts/s21_ssm_m4_m5.py --sequential
    python scripts/s21_ssm_m4_m5.py --frozen-p-probe
    python scripts/s22_ssm_p4_benchmark.py --sequential
    python scripts/s23_ssm_p4_realtext.py --sequential
    python scripts/s26_ssm_p4_fair_tf.py
    python scripts/s31_char_bigram_oracle.py
    python scripts/s33_ssm_p4_m5.py
    python scripts/s35_readout_boundary_probe.py --workers 4
    python scripts/s38_ssm_p4_m3m4_ablation.py
    python scripts/s40_esn_rls_p4_baseline.py
    python scripts/gen_paperD_fig1_p1_arms.py
    python scripts/gen_paperD_fig2_routing.py
    python scripts/gen_paperD_fig3_benchmark.py

`--sequential` disables the multiprocessing pool; `--workers N` caps it.
Experiment scripts accept `--quick` for a reduced smoke run. Each run
regenerates the committed `data/` files.

The manuscript is included as a compiled PDF (`manuscript/PAPER_D.pdf`) for
reference only. The LaTeX source is submitted separately through the
journal's 'LaTeX source files' item type: Elsevier does not allow LaTeX
files as Supplementary items.

Every headline number in the manuscript maps to a committed `data/*`
artefact (see `scripts/verify_claims.py` in the full repository
<https://github.com/huyamingc/REDEM>).

Requirements: Python 3.11+; numpy and torch (CPU) for the experiment
chain; matplotlib for the figure generators. Exact package pins for
archived runs live in `requirements-lock.txt` in the full repository.

License: MIT, same as the companion repository.
"""

D_SCRIPTS = [
    # experiment chain
    "s18_llm_drift_gate.py",
    "s19_ssm_rls_readout.py",
    "s20_ssm_m3_routing.py",
    "s21_ssm_m4_m5.py",
    "s22_ssm_p4_benchmark.py",
    "s23_ssm_p4_realtext.py",
    "s26_ssm_p4_fair_tf.py",
    "s31_char_bigram_oracle.py",
    "s33_ssm_p4_m5.py",
    "s35_readout_boundary_probe.py",
    "s38_ssm_p4_m3m4_ablation.py",
    "s40_esn_rls_p4_baseline.py",
    # shared host modules the chain imports
    "recurrent_substrate.py",
    "streaming_tasks.py",
    "per_token_io.py",
    "shallow_trap_array_simulator.py",
    # figure generators
    "gen_paperD_fig1_p1_arms.py",
    "gen_paperD_fig2_routing.py",
    "gen_paperD_fig3_benchmark.py",
]

D_DATA_STEMS = [
    "s18_llm_drift_gate_v1",
    "s19_ssm_rls_readout_v1",
    "s20_ssm_m3_routing_v1",
    "s21_ssm_m4_m5_v1",
    "s22_ssm_p4_benchmark_v1",
    "s23_ssm_p4_realtext_v1",
    "s26_ssm_p4_fair_tf_v1",
    "s31_char_bigram_oracle_v1",
    "s33_ssm_p4_m5_v1",
    "s35_readout_boundary_probe_v1",
    "s38_ssm_p4_m3m4_ablation_v1",
    "s40_esn_rls_p4_baseline_v1",
]
# s37 exports only the JSON (no CSV companion)
D_DATA_EXTRA = ["s37_dormant_p_probe_v1.json"]

D_FIGURES = [
    "paperD_fig1_p1_arms.pdf",
    "paperD_fig2_routing.pdf",
    "paperD_fig3_benchmark.pdf",
]

D_CORPORA = ["README.md", "alice.txt", "dickens.txt"]


def should_skip(arcname: str) -> bool:
    name = arcname.replace("\\", "/")
    if name in SKIP_NAME_PARTS:
        return True
    base = Path(name).name
    for part in SKIP_NAME_PARTS:
        if part in base:
            return True
    for pat in SKIP_PATTERNS:
        if pat in name:
            return True
    return False


def add_file(zf: zipfile.ZipFile, src: Path, arcname: str) -> None:
    if should_skip(arcname):
        print(f"  skip  {arcname}")
        return
    if not src.is_file():
        raise SystemExit(f"MISSING input: {arcname}  ({src})")
    zf.write(src, arcname)
    print(f"  add   {arcname}  ({src.stat().st_size} bytes)")


def main() -> None:
    print(f"ROOT={ROOT}")
    print(f"ZIP={ZIP_PATH}")

    items: list[tuple[Path, str]] = []

    # compiled manuscript, PDF only: Elsevier forbids LaTeX files as
    # Supplementary items, so the .tex source goes through the journal's
    # 'LaTeX source files' item type instead
    items.append((ROOT / "paper_d" / "PAPER_D.pdf", "manuscript/PAPER_D.pdf"))

    # scripts
    for name in D_SCRIPTS:
        items.append((ROOT / "scripts" / name, f"scripts/{name}"))

    # data (csv+json pairs, plus the json-only stem)
    data_dir = ROOT / "data"
    for stem in D_DATA_STEMS:
        for ext in (".csv", ".json"):
            items.append((data_dir / f"{stem}{ext}", f"data/{stem}{ext}"))
    for name in D_DATA_EXTRA:
        items.append((data_dir / name, f"data/{name}"))

    # public-domain corpora used by s23
    for name in D_CORPORA:
        items.append((data_dir / "corpora" / name, f"data/corpora/{name}"))

    # figures
    fig_dir = ROOT / "figures"
    for name in D_FIGURES:
        items.append((fig_dir / name, f"figures/{name}"))

    # unique by arcname
    seen: set[str] = set()
    unique_items: list[tuple[Path, str]] = []
    for src, arc in items:
        if arc in seen:
            continue
        seen.add(arc)
        unique_items.append((src, arc))

    print(f"planned entries: {len(unique_items) + 1}")  # +1 zip README

    tmp = ZIP_PATH.with_suffix(".zip.tmp")
    if tmp.exists():
        tmp.unlink()
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("README.md", README)
        print("  add   README.md")
        for src, arc in unique_items:
            add_file(zf, src, arc)

    # safety scan
    print("\n=== safety scan of new zip ===")
    bad: list[str] = []
    with zipfile.ZipFile(tmp) as zf:
        for name in zf.namelist():
            if should_skip(name):
                bad.append(f"unexpected skip-list member: {name}")
            if name.endswith((".tex", ".bib", ".bbl")):
                bad.append(f"LaTeX source in Supplementary (Elsevier forbids it): {name}")
            if name.endswith((".md", ".py", ".txt", ".json", ".csv")):
                try:
                    text = zf.read(name).decode("utf-8", "replace")
                except Exception:
                    continue
                for key in ("review_workspace", "modification_log", "git history",
                            "clean-P0", "desk-reject"):
                    if key in text:
                        bad.append(f"{name}: contains '{key}'")
        print(f"entries: {len(zf.namelist())}")
        print(f"manuscript sha: {hashlib.sha256(zf.read('manuscript/PAPER_D.pdf')).hexdigest()[:16]}")
    folder_sha = hashlib.sha256((ROOT / "paper_d" / "PAPER_D.pdf").read_bytes()).hexdigest()[:16]
    print(f"folder  sha: {folder_sha}")

    if bad:
        print("PROBLEMS:")
        for b in bad:
            print(" ", b)
        print("NOT replacing zip")
        return

    tmp.replace(ZIP_PATH)
    print(f"\nOK wrote {ZIP_PATH} ({ZIP_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
