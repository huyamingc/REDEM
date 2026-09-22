#!/usr/bin/env python3
"""
Run the B-layer audit chain in order and aggregate exit codes.
=============================================================================
Type:           AUDIT
Produces:       console status; does not re-run heavy experiments
Reads:          data/*.json via verify_claims
Role:           ORCH
=============================================================================
Order: verify_claims -> check_tables -> audit_tex_numbers -> provenance_map
       -> audit_readme
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

HERE = Path(__file__).resolve().parent
STEPS = [
    "verify_claims.py",
    "check_tables.py",
    "audit_tex_numbers.py",
    "provenance_map.py",
    "audit_readme.py",
]


def run_one(name: str) -> int:
    print(f"\n=== {name} ===", flush=True)
    try:
        runpy.run_path(str(HERE / name), run_name="__main__")
        return 0
    except SystemExit as e:
        code = e.code
        if code is None:
            return 0
        return int(code)
    except Exception as e:  # noqa: BLE001
        print(f"{name}: CRASH {e}")
        return 1


def main() -> int:
    codes = {}
    for step in STEPS:
        codes[step] = run_one(step)
    print("\n=== run_all_audits summary ===")
    failed = []
    for step, code in codes.items():
        status = "OK" if code == 0 else f"FAIL({code})"
        print(f"  {step}: {status}")
        if code != 0:
            failed.append(step)
    if failed:
        print(f"RUN_ALL_AUDITS_EXIT=1  failed: {', '.join(failed)}")
        return 1
    print("RUN_ALL_AUDITS_EXIT=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
