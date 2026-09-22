#!/usr/bin/env python3
"""
Light reverse / stale audit over paper manuscripts (tex).
=============================================================================
Type:           AUDIT
Produces:       scripts/tex_number_audit.md
Reads:          paper_*/PAPER_*.tex
Role:           STALE / reverse
=============================================================================
Scope: headline magnitudes and forbidden historical wording — not every
auxiliary coefficient (ML papers have few 10^b scientific magnitudes).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "tex_number_audit.md"
TEXES = sorted(ROOT.glob("paper_*/PAPER_*.tex"))

# Forbidden stale wording (regression patterns from known corrections).
FORBIDDEN = [
    (
        r"2\.45\s*<\s*2\.262",
        "false significance: cross-family t=2.45 is ABOVE 2.262 (see MAINTENANCE)",
    ),
    (
        r"requires a preheating simulation",
        "stale pending wording",
    ),
    (
        r"pending dark-matter normalization|pending preheating",
        "stale pending wording (physics-era; must not appear in REDEM papers)",
    ),
    (
        r"per-dimension correction required",
        "retracted structural correction claim",
    ),
    (
        r"1e-67\s*cm\^2",
        "stale physics magnitude from another repo",
    ),
]

# Magnitudes that must appear near a registered/known anchor when quoted as
# absolute headline numbers in D/E/F abstracts or tables.
HEADLINE_NUMS = [
    ("0.307", "F CE drop"),
    ("11.75", "F/D stream anchor"),
    ("8.65", "F softmax stream"),
    ("8.70", "E retention gain"),
    ("3.87", "EWC t"),
]


def main() -> int:
    fail = 0
    lines = ["# tex_number_audit", "", f"scanned: {len(TEXES)} manuscripts", ""]

    lines.append("## Forbidden wording")
    lines.append("")
    lines.append("| pattern | reason | hits |")
    lines.append("|---|---|---|")
    for pat, reason in FORBIDDEN:
        hits = []
        for t in TEXES:
            text = t.read_text(encoding="utf-8", errors="replace")
            if re.search(pat, text, flags=re.I):
                hits.append(t.name)
        if hits:
            fail += 1
        lines.append(f"| `{pat}` | {reason} | {', '.join(hits) or '—'} |")

    lines.append("")
    lines.append("## Headline magnitude presence (D/E/F)")
    lines.append("")
    lines.append("| token | meaning | found in |")
    lines.append("|---|---|---|")
    for token, meaning in HEADLINE_NUMS:
        found = []
        for t in TEXES:
            text = t.read_text(encoding="utf-8", errors="replace")
            if token in text:
                found.append(t.name)
        lines.append(f"| {token} | {meaning} | {', '.join(found) or '—'} |")

    lines.append("")
    lines.append("## Light reverse scan (scientific-looking numbers)")
    lines.append("")
    # count patterns like 0.85\pm0.70, t=..., 10/10, pp values
    pat_mag = re.compile(
        r"(\d+\.\d+\s*\\pm\s*\d+\.\d+)|(\bt\s*=\s*-?\d+\.\d+)|(\d+\s*/\s*10\b)|"
        r"(\d+\.\d+\s*~?(?:nats|pp|ppl))"
    )
    total = 0
    for t in TEXES:
        text = t.read_text(encoding="utf-8", errors="replace")
        n = len(pat_mag.findall(text))
        total += n
        lines.append(f"- `{t.parent.name}/{t.name}`: {n} headline-like tokens")
    lines.append("")
    lines.append(f"total headline-like tokens: **{total}** (manual registry covers a subset; unregistered tokens are not auto-fail in MVP)")
    lines.append("")
    lines.append(f"failures: **{fail}**")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"audit_tex_numbers: FLAG={fail}")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
