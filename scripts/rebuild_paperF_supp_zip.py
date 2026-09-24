#!/usr/bin/env python3
"""
Build Paper F supplementary zip (compiled manuscript PDF + scripts + data + figures).
=============================================================================
Type:           IO
Produces:       paper_f/Supplementary_Material_PaperF.zip
Reads:          paper_f/PAPER_F.pdf, scripts/s50..s54,s66,s67*, data/s50.., figures/paperF*
Role:           submission tooling (local-only artefact)
=============================================================================
"""
from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_f" / "Supplementary_Material_PaperF.zip"

SCRIPT_STEMS = [
    "s50_paper_f_pilot_nl_readout",
    "s51_paper_f_learned_gate",
    "s52_paper_f_soft_route_experts",
    "s53_paper_f_scaling",
    "s53b_paper_f_ksweep",
    "s54_paper_f_mackey_glass",
    "s66_external_ssm_baseline",
    "s66_report",
    "s67_host_freeze",
    "per_token_io",
    "gen_paperF_figs",
]

DATA_STEMS = [
    "s50_paper_f_pilot_nl_readout",
    "s51_paper_f_learned_gate_v2",
    "s52_paper_f_soft_route_experts",
    "s53_paper_f_scaling",
    "s53b_paper_f_ksweep",
    "s54_paper_f_mackey_glass",
    "s66_external_ssm_baseline",
    "s67_host_freeze",
]

README = """# Supplementary Code and Data — Paper F

> Paper F — "Learning the Readout Form: Objective-Level Calibration,
> Sparse Selectivity, and Expert Routing on a Diagonal State-Space Host"

Contents: the compiled manuscript (PDF), the Paper F experiment scripts (s50–s54, s66–s67),
committed result tables used by the manuscript, and figure generators.

Reproduce from the repository root (CPU-only):

    python scripts/s50_paper_f_pilot_nl_readout.py --sequential
    python scripts/s51_paper_f_learned_gate.py --sequential
    python scripts/s52_paper_f_soft_route_experts.py --sequential
    python scripts/s53_paper_f_scaling.py --sequential
    python scripts/s53b_paper_f_ksweep.py --sequential
    python scripts/s54_paper_f_mackey_glass.py --sequential
    python scripts/s66_external_ssm_baseline.py
    python scripts/s67_host_freeze.py
    python scripts/gen_paperF_figs.py

Every headline number in the manuscript maps to a committed `data/*` artefact
(see `scripts/verify_claims.py` in the full repository).
"""


def main() -> int:
    files: list[tuple[Path, str]] = []

    # compiled manuscript, PDF only: Elsevier forbids LaTeX files as
    # Supplementary items, so the .tex source goes through the journal's
    # 'LaTeX source files' item type instead
    pdf = ROOT / "paper_f" / "PAPER_F.pdf"
    if pdf.exists():
        files.append((pdf, "manuscript/PAPER_F.pdf"))

    for stem in SCRIPT_STEMS:
        p = ROOT / "scripts" / f"{stem}.py"
        if p.exists():
            files.append((p, f"scripts/{p.name}"))

    for stem in DATA_STEMS:
        for ext in (".csv", ".json"):
            p = ROOT / "data" / f"{stem}_v1{ext}"
            if not p.exists():
                # allow non-_v1 stems (s51 v2 already includes version)
                p = ROOT / "data" / f"{stem}{ext}"
            if p.exists():
                files.append((p, f"data/{p.name}"))

    for name in (
        "paperF_fig1_arms.pdf",
        "paperF_fig2_floor.pdf",
        "paperF_graphical_abstract.pdf",
        "paperF_fig1_arms.png",
        "paperF_fig2_floor.png",
        "paperF_graphical_abstract.png",
    ):
        p = ROOT / "figures" / name
        if p.exists():
            files.append((p, f"figures/{p.name}"))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("README.md", README)
        for src, arc in files:
            z.write(src, arc)
    print(f"wrote {OUT} ({len(files) + 1} entries)")

    # Elsevier forbids LaTeX sources in Supplementary material
    with zipfile.ZipFile(OUT) as z:
        bad = [n for n in z.namelist() if n.endswith((".tex", ".bib", ".bbl"))]
    if bad:
        raise SystemExit(f"ERROR: LaTeX source in supplementary zip: {bad}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
