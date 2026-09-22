#!/usr/bin/env python3
"""
Audit README.md prose numbers against the claim registry / artefacts.
=============================================================================
Type:           AUDIT
Produces:       scripts/readme_audit.md
Reads:          README.md, scripts/claims_registry.json
Role:           RNUM
=============================================================================
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
REG = Path(__file__).resolve().parent / "claims_registry.json"
OUT = Path(__file__).resolve().parent / "readme_audit.md"

# Anchor patterns that must remain in README and must match a registered claim.
# kind: lit = literal substring that must appear; num = number that must equal claim.
CHECKS = [
    {"id": "S8-full", "pattern": r"0\.996\b", "claim_q": "full-system overall acc @ N=256"},
    {"id": "S8-base", "pattern": r"0\.973\b", "claim_q": "bare-baseline overall acc @ N=256"},
    {"id": "S30-full", "pattern": r"0\.9970\b", "claim_q": "full-system acc @ N=1024"},
    {"id": "S30-base", "pattern": r"0\.9753\b", "claim_q": "baseline acc @ N=1024"},
    {"id": "S11-reg", "pattern": r"8\.47\b", "claim_q": "regulated r3 MC after 3 disturbances"},
    {"id": "S11-fix", "pattern": r"6\.41\b", "claim_q": "fixed-kappa r3 MC"},
    {"id": "F-stream", "pattern": r"11\.75\b", "claim_q": "B-proj / B-lin-clip stream ppl (input path)"},
    {"id": "F-softmax", "pattern": r"8\.65\b", "claim_q": "B-softmax stream ppl"},
    {"id": "F-topk", "pattern": r"7\.03\b", "claim_q": "Gate-C-topk stream ppl"},
    {"id": "E-mem", "pattern": r"\+8\.70\b", "claim_q": "frozen-snapshot retention gain (pp)"},
    {"id": "E-ewc", "pattern": r"−0\.85|-0\.85\b", "claim_q": "EWC whole-run accuracy change (pp)"},
    {"id": "A-kappa", "pattern": r"κ\*∈\(25,30\)|κ\\ast|25\.3", "claim_q": "kappa* lateral_ring (amax 0.1)"},
    {"id": "A-r", "pattern": r"r=0\.97|r = 0\.97", "claim_q": "forgetting-kernel Pearson r"},
    {"id": "C-s14", "pattern": r"-0\.78|−0\.78", "claim_q": "s14 paired MC diff at r1 (dual-fast)"},
    {"id": "B-nov", "pattern": r"14\.59", "claim_q": "novelty-guided rewiring MC_final"},
]


def main() -> int:
    text = README.read_text(encoding="utf-8")
    # strip generated block so RNUM checks hand-written prose only
    begin = "<!-- BEGIN PROVENANCE"
    end = "<!-- END PROVENANCE -->"
    prose = text
    if begin in text and end in text:
        i0 = text.index(begin)
        i1 = text.index(end) + len(end)
        prose = text[:i0] + text[i1:]

    if not REG.exists():
        print("audit_readme: missing claims_registry.json")
        return 1
    reg = json.loads(REG.read_text(encoding="utf-8"))
    by_q = {c["quantity"]: c for c in reg.get("claims") or []}

    lines = ["# readme_audit", "", "| id | pattern | claim | ok | note |", "|---|---|---|---|---|"]
    fail = 0
    for chk in CHECKS:
        found = re.search(chk["pattern"], prose) is not None
        cl = by_q.get(chk["claim_q"])
        claim_ok = bool(cl and cl.get("ok"))
        ok = found and claim_ok
        if not ok:
            fail += 1
        note = []
        if not found:
            note.append("pattern missing in README prose")
        if not cl:
            note.append("claim not registered")
        elif not cl.get("ok"):
            note.append("claim failed")
        lines.append(
            f"| {chk['id']} | `{chk['pattern']}` | {chk['claim_q']} | "
            f"{'PASS' if ok else 'FAIL'} | {'; '.join(note) or 'ok'} |"
        )
    lines.append("")
    lines.append(f"failures: **{fail}**")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"audit_readme: {len(CHECKS)} checks / {fail} failures")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
