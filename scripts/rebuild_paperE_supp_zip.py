#!/usr/bin/env python3
"""Rebuild Paper E supplementary zip without internal process traces."""
from pathlib import Path
import zipfile
import hashlib

ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = ROOT / "paper_e" / "Supplementary_Material_PaperE.zip"
SKIP_NAME_PARTS = ("__pycache__", ".pyc", ".nbc", ".nbi", ".DS_Store", "Thumbs.db")
SKIP_EXACT = {
    "data/s58b_trace_seed4.json",
}
# Prefer full-run data; skip smoke/screening artifacts if present
SKIP_PATTERNS = ("_quick.", "_quick_", ".bak")

README = """# Supplementary Code and Data — Paper E

Self-contained reproduction package for:

> Paper E — "Self-evolution under ±1 reward: sign-flip correction,
> reconstructed-label capacity sensing, and snapshot-gated memory in a
> recurrent relaxation reservoir" (target: *Neurocomputing*)

This archive carries the compiled manuscript (PDF), the self-evolution experiment
chain (`s39`–`s65`), the frozen `deps/` modules the chain imports, the result
files behind the numbers in the paper, and the six vector figures.

## Layout

| Path | Contents |
|---|---|
| `manuscript/` | `PAPER_E.pdf` — compiled manuscript for reference. The LaTeX source is submitted separately through the journal's 'LaTeX source files' item type (Elsevier does not allow LaTeX files as Supplementary items). |
| `scripts/` | Chain `s39_*`–`s65_*` (incl. `s43b`, `s58a`–`s58f`, `s64b`) plus `gen_fig_self_evolution.py`. |
| `data/` | Committed full-run CSV/JSON results cited by the manuscript. |
| `figures/` | Six vector PDFs used in the paper. |
| `paper_e/deps/` | Frozen dependencies: `recurrent_substrate.py`, `online_readout.py`, `streaming_tasks.py`, `shallow_trap_array_simulator.py` + `README.md`. Chain scripts insert this directory first on `sys.path`. |

## Requirements

- Python 3.11+ recommended
- numpy, scipy, scikit-learn (ridge/PCA probes), matplotlib
- numba optional: modules keep `try/except ImportError` fallbacks

Exact package pins for archived runs live in the companion repository file
`requirements-lock.txt` at <https://github.com/huyamingc/REDEM>.

## Usage

From the repository root (or after extracting this archive so that
`scripts/` and `paper_e/deps/` keep their relative layout):

```bash
PYTHONUNBUFFERED=1 python scripts/s58e_d2_timescale.py
PYTHONUNBUFFERED=1 python scripts/s64b_weak_source_reliability.py --quick
```

Scripts write CSV/JSON under `data/`. Full 10-seed runs are the values
reported in the paper.

## License

MIT, same as the companion repository.
"""


def should_skip(arcname: str) -> bool:
    name = arcname.replace("\\", "/")
    if name in SKIP_EXACT:
        return True
    base = Path(name).name
    for part in SKIP_NAME_PARTS:
        if part in name:
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
        print(f"  MISS  {arcname}  ({src})")
        return
    zf.write(src, arcname)
    print(f"  add   {arcname}  ({src.stat().st_size} bytes)")


def main() -> None:
    print(f"ROOT={ROOT}")
    print(f"ZIP={ZIP_PATH}")

    # Collect files that belong in the package
    items: list[tuple[Path, str]] = []

    # compiled manuscript, PDF only: Elsevier forbids LaTeX files as
    # Supplementary items, so the .tex source goes through the journal's
    # 'LaTeX source files' item type instead
    items.append((ROOT / "paper_e" / "PAPER_E.pdf", "manuscript/PAPER_E.pdf"))

    # scripts — Paper E chain only
    script_dir = ROOT / "scripts"
    e_scripts = [
        "s39_prediction_reward_test.py",
        "s40_meta_flip_ablation.py",
        "s41_meta_reward_self_tuning.py",
        "s42_corrupt_signal_test.py",
        "s43_hypothesis_population.py",
        "s43b_synthetic_rotation.py",
        "s44_regime_zoo.py",
        "s45_population_r2r3.py",
        "s46_rule_adjudication.py",
        "s47_self_correction_gain.py",
        "s48_rotation2d.py",
        "s49_capacity_probe.py",
        "s50_nonlinear_capacity.py",
        "s51_quadratic_circle.py",
        "s52_dynamic_memory.py",
        "s53_auto_basis.py",
        "s54_memory_boundaries.py",
        "s55_sphere_capacity.py",
        "s56_joint_self_evolution.py",
        "s57_dimension_stress.py",
        "s58a_complexity_trigger.py",
        "s58b_relative_sense_stress.py",
        "s58c_multi_level_expansion.py",
        "s58d_kernel_capacity_sweep.py",
        "s58e_d2_timescale.py",
        "s58f_margin_sensitivity.py",
        "s59_self_tuned_gate.py",
        "s60_adaptive_window.py",
        "s61_cross_family_transfer.py",
        "s62_adaptive_meta_timescale.py",
        "s63_content_rendering.py",
        "s64_multi_source_validation.py",
        "s64b_weak_source_reliability.py",
        "s65_ewc_baseline.py",
        "gen_fig_self_evolution.py",
    ]
    for name in e_scripts:
        items.append((script_dir / name, f"scripts/{name}"))

    # data — Paper E result stems only (exclude Paper D/F companions)
    data_dir = ROOT / "data"
    e_data_prefixes = (
        "s39_prediction_reward_v2",
        "s40_meta_flip", "s41_meta_reward", "s42_corrupt", "s43_population",
        "s43b_rotation", "s44_regime", "s45_population", "s46_rule",
        "s47_self_correction", "s48_rotation2d", "s49_capacity",
        "s50_nonlinear_capacity", "s51_quadratic_circle", "s52_dynamic_memory",
        "s53_auto_basis", "s54_memory_boundaries", "s55_sphere",
        "s56_joint", "s57_dimension", "s58a_", "s58b_relative",
        "s58c_", "s58d_", "s58e_", "s58f_", "s59_self", "s60_adaptive_window",
        "s61_cross", "s62_adaptive_meta", "s63_content", "s64_multi",
        "s64b_weak", "s65_ewc", "s65_tuning",
    )
    for p in sorted(data_dir.glob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".csv", ".json"}:
            continue
        if p.name.startswith(e_data_prefixes):
            items.append((p, f"data/{p.name}"))

    # figures
    fig_dir = ROOT / "figures"
    for p in sorted(fig_dir.glob("self_evo_f*.pdf")):
        items.append((p, f"figures/{p.name}"))

    # frozen deps
    deps = ROOT / "paper_e" / "deps"
    for p in sorted(deps.iterdir()):
        if p.is_file() and p.suffix in {".py", ".md"}:
            items.append((p, f"paper_e/deps/{p.name}"))

    # unique by arcname
    seen = set()
    unique_items = []
    for src, arc in items:
        if arc in seen:
            continue
        seen.add(arc)
        unique_items.append((src, arc))

    print(f"planned entries: {len(unique_items)}")

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
    bad = []
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
        print(f"manuscript sha: {hashlib.sha256(zf.read('manuscript/PAPER_E.pdf')).hexdigest()[:16]}")
    folder_sha = hashlib.sha256((ROOT / "paper_e" / "PAPER_E.pdf").read_bytes()).hexdigest()[:16]
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
