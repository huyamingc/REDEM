#!/usr/bin/env python3
"""
Table-cell consistency checks: claim registry vs manuscript tabular cells.
=============================================================================
Type:           AUDIT
Produces:       scripts/table_check_report.md
Reads:          scripts/claims_registry.json, paper_*/PAPER_*.tex
Role:           TABLE
=============================================================================
Upgraded from corpus-string presence to tex *table cell* matching:
- parse every \\begin{tabular}...\\end{tabular} block (with caption/label)
- extract numeric cell tokens from each table
- for table-facing claims (where mentions Table/tab or quantity is a known
  table anchor), require the doc_value to appear in *some table cell*
- for prose claims, fall back to full-corpus presence but record whether the
  hit was a table cell or prose-only
- identity / reproduction_check claims skip manuscript presence
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parents[1]
REG = Path(__file__).resolve().parent / "claims_registry.json"
OUT = Path(__file__).resolve().parent / "table_check_report.md"
TEXES = sorted(ROOT.glob("paper_*/PAPER_*.tex"))

TABULAR_RE = re.compile(
    r"\\begin\{tabular\}.*?\\end\{tabular\}", re.DOTALL
)
CAPTION_RE = re.compile(r"\\caption\{(.*?)\}\s*", re.DOTALL)
LABEL_RE = re.compile(r"\\label\{(tab:[^}]+)\}")
# numeric tokens: 0.898, -1.87, 11.75, 8.65, 2.45, 50, 0.9970
NUM_RE = re.compile(r"(?<![A-Za-z_\\])[-+]?\d+(?:\.\d+)?")


@dataclass
class TableBlock:
    paper: str
    path: str
    label: str
    caption_snip: str
    cells: list[str]  # raw numeric tokens as they appear
    body: str


def _paper_of(path: Path) -> str:
    name = path.parent.name  # paper_e
    return name.replace("paper_", "").upper() if name.startswith("paper_") else name


def parse_tables(tex_path: Path) -> list[TableBlock]:
    text = tex_path.read_text(encoding="utf-8", errors="replace")
    # find captions/labels near each tabular by scanning the whole file
    tables: list[TableBlock] = []
    for m in TABULAR_RE.finditer(text):
        body = m.group(0)
        # look backward for nearest caption/label belonging to this float
        prefix = text[max(0, m.start() - 1200) : m.start()]
        caps = CAPTION_RE.findall(prefix)
        labs = LABEL_RE.findall(prefix)
        caption = caps[-1] if caps else ""
        # strip latex macros lightly
        caption_snip = re.sub(r"\\[a-zA-Z]+\{([^{}]*)\}", r"\1", caption)
        caption_snip = re.sub(r"\\[a-zA-Z]+|[{}$]", " ", caption_snip)
        caption_snip = re.sub(r"\s+", " ", caption_snip).strip()[:120]
        label = labs[-1] if labs else ""
        # cells: only tokens inside the tabular body that look numeric
        cells = NUM_RE.findall(body)
        tables.append(
            TableBlock(
                paper=_paper_of(tex_path),
                path=str(tex_path.relative_to(ROOT)).replace("\\", "/"),
                label=label,
                caption_snip=caption_snip,
                cells=cells,
                body=body,
            )
        )
    return tables


def value_forms(doc_value: float) -> list[str]:
    """Accept common manuscript renderings of a claim doc_value."""
    forms = {f"{doc_value:g}", f"{doc_value:.2f}", f"{doc_value:.1f}", f"{doc_value:.3f}"}
    # trailing-zero stripped
    forms.add(f"{doc_value:g}".rstrip("0").rstrip("."))
    # percent-style sometimes written without leading 0? keep simple
    # 0.897925 -> 0.897925 / 0.898 / 0.90 already covered
    forms.add(f"{doc_value:.4f}".rstrip("0").rstrip("."))
    forms.add(f"{doc_value:.5f}".rstrip("0").rstrip("."))
    forms.add(f"{doc_value:.6f}".rstrip("0").rstrip("."))
    return [f for f in forms if f]


def cell_matches(cells: list[str], doc_value: float) -> tuple[bool, str]:
    forms = set(value_forms(doc_value))
    for c in cells:
        if c in forms:
            return True, c
        # also allow cell parsed as float equal within rounding of forms
        try:
            cv = float(c)
        except ValueError:
            continue
        for f in forms:
            try:
                if abs(cv - float(f)) <= 1e-9:
                    return True, c
            except ValueError:
                continue
        # relative closeness for rounded cells (0.898 matches 0.8980)
        if abs(cv - doc_value) <= max(5e-4, 0.005 * max(abs(doc_value), 0.01)):
            return True, c
    return False, ""


def is_table_facing(c: dict) -> bool:
    where = (c.get("where") or "").lower()
    qty = (c.get("quantity") or "").lower()
    # only force a *table cell* hit when the claim explicitly anchors a table
    if "tab:" in where or "table cell" in where or "table tab:" in where:
        return True
    if "tables" in where and "abstract" not in where and "readme" not in where:
        return True
    if "zoo" in qty or "multisource" in qty or "table cell" in qty:
        return True
    return False


def is_identity(c: dict) -> bool:
    if c.get("kind") == "identity":
        return True
    sec = (c.get("section") or "").lower()
    qty = (c.get("quantity") or "").lower()
    return sec == "reproduction_check" or "bit-matches" in qty


def main() -> int:
    if not REG.exists():
        print("check_tables: missing claims_registry.json")
        return 1
    reg = json.loads(REG.read_text(encoding="utf-8"))
    claims = reg.get("claims") or []

    all_tables: list[TableBlock] = []
    corpora: dict[str, str] = {}
    for t in TEXES:
        all_tables.extend(parse_tables(t))
        corpora[str(t.relative_to(ROOT)).replace("\\", "/")] = t.read_text(
            encoding="utf-8", errors="replace"
        )
    corpus = "\n".join(corpora.values())
    all_cells = [cell for tb in all_tables for cell in tb.cells]

    lines = [
        "# table_check_report",
        "",
        f"claims: {len(claims)}",
        f"tables parsed: {len(all_tables)}",
        f"numeric table cells: {len(all_cells)}",
        "",
        "| paper | quantity | doc | computed | in_table | table_hit | prose | ok |",
        "|---|---|---|---|---|---|---|---|",
    ]
    fail = 0
    n_table_hits = 0
    n_prose_only = 0
    for c in claims:
        doc = float(c.get("doc_value", 0.0))
        doc_s = f"{doc:g}"
        if is_identity(c):
            ok = bool(c.get("ok"))
            lines.append(
                f"| {c.get('paper')} | {c.get('quantity')} | {doc_s} | "
                f"{c.get('computed')} | skip | identity | skip | "
                f"{'PASS' if ok else 'FAIL'} |"
            )
            if not ok:
                fail += 1
            continue

        # 1) table-cell match (prefer the claim's paper's tables, then all)
        hit_cell = ""
        hit_label = ""
        own = [tb for tb in all_tables if tb.paper == c.get("paper")]
        for pool in (own, all_tables):
            for tb in pool:
                ok_c, tok = cell_matches(tb.cells, doc)
                if ok_c:
                    hit_cell = tok
                    hit_label = tb.label or tb.caption_snip[:40] or tb.path
                    break
            if hit_cell:
                break

        # 2) prose / full corpus presence
        prose = any(f in corpus for f in value_forms(doc))
        if not prose:
            prose = f"{doc:.2f}" in corpus or f"{doc:.1f}" in corpus

        table_facing = is_table_facing(c)
        if table_facing:
            ok = bool(c.get("ok")) and bool(hit_cell)
            if hit_cell:
                n_table_hits += 1
            present_s = "yes" if hit_cell else "no"
        else:
            # prose claims: require corpus hit; table hit is bonus
            ok = bool(c.get("ok")) and (prose or bool(hit_cell))
            if hit_cell:
                n_table_hits += 1
            elif prose:
                n_prose_only += 1
            present_s = "yes" if hit_cell else ("prose" if prose else "no")

        if not ok:
            fail += 1
        lines.append(
            f"| {c.get('paper')} | {c.get('quantity')} | {doc_s} | "
            f"{c.get('computed')} | {present_s} | "
            f"{(hit_cell + ' @ ' + hit_label) if hit_cell else '—'} | "
            f"{'yes' if prose else 'no'} | {'PASS' if ok else 'FAIL'} |"
        )

    lines.append("")
    lines.append(
        f"table-cell hits: **{n_table_hits}**; prose-only: **{n_prose_only}**; "
        f"failures: **{fail}**"
    )
    lines.append("")
    lines.append("## Parsed tables")
    lines.append("")
    lines.append("| paper | label | #cells | caption | path |")
    lines.append("|---|---|---|---|---|")
    for tb in all_tables:
        lines.append(
            f"| {tb.paper} | {tb.label or '—'} | {len(tb.cells)} | "
            f"{tb.caption_snip[:60]} | {tb.path} |"
        )
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"check_tables: {len(claims)} cells / {len(all_tables)} tables / "
        f"{n_table_hits} table hits / {fail} failures"
    )
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
