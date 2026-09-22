#!/usr/bin/env python3
"""
Claim registry: manuscript / README numbers vs committed data artefacts.
=============================================================================
Type:           AUDIT
Produces:       scripts/claims_registry.json, scripts/verification_report.md
Reads:          data/*.json, data/*.csv (as needed)
Role:           CLAIM — single executable map from quoted quantity to artefact
=============================================================================
"""
from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Callable, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT_JSON = Path(__file__).resolve().parent / "claims_registry.json"
OUT_MD = Path(__file__).resolve().parent / "verification_report.md"

# tolerance classes
# rounding: |doc-computed| / max(|computed|,eps) <= tol  (relative)
# order:    |log10(doc) - log10(computed)| <= tol        (decades)
# identity: exact or near-exact for booleans / integers


@dataclass
class Claim:
    paper: str
    section: str
    quantity: str
    where: str
    doc_value: float
    computed: Any
    tol: float
    kind: str = "rounding"  # rounding | order | identity | precision
    note: str = ""
    ok: bool = field(default=False, init=False)
    detail: str = field(default="", init=False)


CLAIMS: list[Claim] = []


def claim(
    paper: str,
    section: str,
    quantity: str,
    where: str,
    doc_value: float,
    computed: Any,
    tol: float,
    kind: str = "rounding",
    note: str = "",
) -> Claim:
    c = Claim(paper, section, quantity, where, float(doc_value), computed, tol, kind, note)
    CLAIMS.append(c)
    return c


def _j(name: str) -> Optional[dict]:
    p = DATA / name
    if not p.suffix:
        p = p.with_suffix(".json")
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _rows_json(name: str) -> list[dict]:
    d = _j(name)
    if not d:
        return []
    if isinstance(d, dict) and isinstance(d.get("rows"), list):
        return d["rows"]
    if isinstance(d, dict) and isinstance(d.get("results"), list):
        return d["results"]
    if isinstance(d, dict) and isinstance(d.get("aggregates"), list):
        return d["aggregates"]
    return []


def _csv_rows(stem: str) -> list[dict]:
    p = DATA / f"{stem}.csv"
    if not p.exists():
        return []
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _agg(agg: list[dict], **match) -> Optional[dict]:
    for a in agg:
        if all(a.get(k) == v for k, v in match.items()):
            return a
    return None


def _arm_mean(results: list[dict], arm: str, key: str) -> Optional[float]:
    vals = [
        r[key]
        for r in results
        if r.get("arm") == arm and isinstance(r.get(key), (int, float))
    ]
    return mean(vals) if vals else None


def evaluate(c: Claim) -> None:
    computed = c.computed
    if callable(computed):
        try:
            computed = computed()
        except Exception as e:  # noqa: BLE001 — audit must report, not crash
            c.ok = False
            c.detail = f"compute error: {e}"
            return
    c.computed = computed
    if computed is None or (isinstance(computed, float) and math.isnan(computed)):
        c.ok = False
        c.detail = "missing artefact / computed=None"
        return
    try:
        dv = float(c.doc_value)
        cv = float(computed)
    except (TypeError, ValueError):
        c.ok = False
        c.detail = f"non-numeric computed={computed!r}"
        return

    if c.kind == "identity":
        c.ok = abs(dv - cv) <= c.tol
        c.detail = f"doc={dv} computed={cv} |diff|={abs(dv-cv):.4g} tol={c.tol}"
    elif c.kind == "order":
        if cv == 0 or dv == 0:
            c.ok = False
            c.detail = "order kind requires nonzero"
        else:
            decades = abs(math.log10(abs(dv)) - math.log10(abs(cv)))
            c.ok = decades <= c.tol
            c.detail = f"doc={dv} computed={cv} decades={decades:.3f} tol={c.tol}"
    else:  # rounding / precision (relative)
        denom = max(abs(cv), 1e-30)
        rel = abs(dv - cv) / denom
        c.ok = rel <= c.tol
        c.detail = f"doc={dv} computed={cv} rel={rel:.3e} tol={c.tol}"


def register_all() -> None:
    # --- shared / Paper B anchors ---
    s8 = _j("s8_integrated_v1.json") or {}
    s8a = s8.get("aggregates", [])
    full256 = _agg(s8a, arm="full", n_units=256)
    base256 = _agg(s8a, arm="baseline", n_units=256)
    nohomeo = _agg(s8a, arm="no_homeostat", n_units=256)

    claim(
        "B", "S8", "full-system overall acc @ N=256", "abstract / README / tables",
        0.996,
        lambda: full256["overall_acc_mean"] if full256 else None,
        0.002,
        "rounding",
    )
    claim(
        "B", "S8", "bare-baseline overall acc @ N=256", "abstract / README / tables",
        0.973,
        lambda: base256["overall_acc_mean"] if base256 else None,
        0.002,
        "rounding",
    )
    claim(
        "B", "S8", "no-homeostat ties full (acc)", "Paper B ablations",
        0.996,
        lambda: nohomeo["overall_acc_mean"] if nohomeo else None,
        0.002,
        "rounding",
        note="homeostat removal is a tie on this task",
    )

    s30 = _j("s30_integrated_1024_v1.json") or {}
    s30a = s30.get("aggregates", [])
    f1024 = _agg(s30a, arm="full", n_units=1024)
    b1024 = _agg(s30a, arm="baseline", n_units=1024)
    claim(
        "B", "S30", "full-system acc @ N=1024", "abstract / README",
        0.9970,
        lambda: f1024["overall_acc_mean"] if f1024 else None,
        0.0005,
        "rounding",
    )
    claim(
        "B", "S30", "baseline acc @ N=1024", "abstract / README",
        0.9753,
        lambda: b1024["overall_acc_mean"] if b1024 else None,
        0.0005,
        "rounding",
    )

    s11 = _j("s11_disturbance_chain_v1.json") or {}
    s11a = s11.get("aggregates", [])
    reg = _agg(s11a, arm="regulated")
    fix = _agg(s11a, arm="fixed")
    claim(
        "B", "S11", "regulated r3 MC after 3 disturbances", "abstract / Paper B/C",
        8.47,
        lambda: reg["r3_mc_mean"] if reg else None,
        0.01,
        "rounding",
    )
    claim(
        "B", "S11", "fixed-kappa r3 MC", "abstract / Paper B/C",
        6.41,
        lambda: fix["r3_mc_mean"] if fix else None,
        0.01,
        "rounding",
    )
    claim(
        "B", "S11", "relative sequential recovery +32%", "abstract / Paper B/C",
        0.32,
        lambda: (
            (reg["r3_mc_mean"] / fix["r3_mc_mean"] - 1.0) if reg and fix else None
        ),
        0.05,
        "rounding",
    )

    claim(
        "A", "S10", "forgetting-kernel Pearson r", "Paper A",
        0.97,
        lambda: (_j("forgetting_curve_theory_overlay_v1.json") or {}).get("r_pearson"),
        0.01,
        "rounding",
    )

    # --- Paper A: kappa* and homeostat / lambda_target ---
    s27 = _j("s27_clip_kappa_fine_v1.json") or {}
    ks = (s27.get("params") or {}).get("kappa_star_ftle_crossing") or {}
    claim(
        "A", "S27", "kappa* lateral_ring (amax 0.1)", "Paper A",
        25.3,
        lambda: ks.get("lateral_ring_amax0.1"),
        0.01,
        "rounding",
    )
    claim(
        "A", "S27", "kappa* ring_bidir (amax 0.1)", "Paper A",
        27.4,
        lambda: ks.get("ring_bidir_amax0.1"),
        0.01,
        "rounding",
    )
    claim(
        "A", "S27", "kappa* random_graph (amax 0.1)", "Paper A",
        27.9,
        lambda: ks.get("random_graph_amax0.1"),
        0.01,
        "rounding",
    )

    def s6_restore_pct(disturb: str) -> Optional[float]:
        rows = _csv_rows("s6_chaos_regulator_v1")
        by: dict[tuple[str, str], list[float]] = {}
        for r in rows:
            if r.get("part") != "2":
                continue
            v = (r.get("mc_at_settled") or r.get("mc_at_nominal") or "").strip()
            if not v:
                continue
            try:
                by.setdefault((r["disturb"], r["arm"]), []).append(float(v))
            except ValueError:
                continue
        key_f = (disturb, "fixed")
        key_r = (disturb, "regulated")
        if key_f not in by or key_r not in by:
            return None
        f = mean(by[key_f])
        reg = mean(by[key_r])
        return (reg - f) / f * 100.0 if f else None

    claim(
        "A", "S6", "homeostat restore after edge_prune (%)", "Paper A / tables",
        7.9,
        lambda: s6_restore_pct("edge_prune"),
        0.05,
        "rounding",
    )
    claim(
        "A", "S6", "homeostat restore after tau_drift (%)", "Paper A / tables",
        18.0,
        lambda: s6_restore_pct("tau_drift"),
        0.05,
        "rounding",
        note="18.27 measured; abstract quotes 7.9–18%",
    )
    claim(
        "A", "S6", "homeostat restore after noise (%)", "Paper A / tables",
        11.8,
        lambda: s6_restore_pct("noise"),
        0.05,
        "rounding",
    )

    def s12_gain_cv01() -> Optional[float]:
        s12 = _j("s12_lambda_target_sweep_v1.json") or {}
        aggs = s12.get("aggregates") or []
        reg = _agg(aggs, lambda_target=0.0, cv_tau=0.1, arm="regulated")
        fix = _agg(aggs, lambda_target=0.0, cv_tau=0.1, arm="fixed")
        # lambda_target stored as 0.0; also accept 0
        if not reg or not fix:
            for a in aggs:
                if a.get("arm") == "regulated" and a.get("cv_tau") == 0.1 and a.get("lambda_target") in (0, 0.0):
                    reg = a
                if a.get("arm") == "fixed" and a.get("cv_tau") == 0.1 and a.get("lambda_target") in (0, 0.0):
                    fix = a
        if not reg or not fix:
            return None
        base = fix.get("mc_heldout_mean") or fix.get("pre_nmse_mean")
        val = reg.get("mc_heldout_mean")
        if not base or not val:
            return None
        return (val / base - 1.0) * 100.0

    claim(
        "A", "E4", "lambda_target=0 MC gain at CV=0.1 (%)", "Paper A / Paper B table",
        25.0,
        s12_gain_cv01,
        0.05,
        "rounding",
        note="+25% over fixed; main runs use conservative -0.02",
    )

    # --- Paper B extras ---
    def s30_paired_t() -> Optional[float]:
        rows = _csv_rows("s30_integrated_1024_v1")
        if not rows:
            return None
        by: dict[int, dict[str, float]] = {}
        for r in rows:
            try:
                by.setdefault(int(float(r["seed_idx"])), {})[r["arm"]] = float(r["overall_acc"])
            except (KeyError, ValueError, TypeError):
                continue
        diffs = [
            d["full"] - d["baseline"]
            for d in by.values()
            if "full" in d and "baseline" in d
        ]
        if len(diffs) < 2:
            return None
        sd = stdev(diffs)
        return mean(diffs) / (sd / math.sqrt(len(diffs))) if sd > 0 else None

    claim(
        "B", "S30", "N=1024 full-vs-baseline paired t", "Paper B / README",
        15.3,
        s30_paired_t,
        0.05,
        "rounding",
    )

    s25 = _j("s25_reward_gated_plasticity_v1.json") or {}
    s25a = s25.get("aggregates", [])
    claim(
        "B", "S25", "novelty-guided rewiring MC_final", "Paper B",
        14.59,
        lambda: (_agg(s25a, arm="evolve_novelty") or {}).get("mc_final_mean"),
        0.01,
        "rounding",
    )
    claim(
        "B", "S25", "correlation-guided rewiring MC_final", "Paper B",
        12.43,
        lambda: (_agg(s25a, arm="evolve_corr") or {}).get("mc_final_mean"),
        0.01,
        "rounding",
    )

    s24 = _j("s24_homeo_plasticity_coupling_v1.json") or {}
    s24a = s24.get("aggregates", [])
    claim(
        "B", "S24", "homeostat-only r3 MC (no plasticity)", "Paper B / S11 anchor",
        8.47,
        lambda: (_agg(s24a, arm="homeo_no_plasticity") or {}).get("r3_mc_mean"),
        0.01,
        "rounding",
        note="must equal S11 regulated r3 anchor",
    )
    claim(
        "B", "S24", "coupled rewiring r3 MC (harmful)", "Paper B",
        5.27,
        lambda: (_agg(s24a, arm="coupled_churn") or {}).get("r3_mc_mean"),
        0.01,
        "rounding",
    )

    s10 = _j("s10_esn_metadata_v1.json") or {}
    s10a = s10.get("aggregates", [])
    claim(
        "C", "S10", "ESN+meta overall acc", "Paper B/C table",
        0.998,
        lambda: (_agg(s10a, arm="esn_dual") or {}).get("overall_acc_mean"),
        0.002,
        "rounding",
    )
    claim(
        "C", "S10", "ESN-fast overall acc", "Paper B/C table",
        0.996,
        lambda: (_agg(s10a, arm="esn_fast") or {}).get("overall_acc_mean"),
        0.002,
        "rounding",
    )
    claim(
        "C", "S10", "REDEM-full overall acc", "Paper B/C table",
        0.994,
        lambda: (_agg(s10a, arm="redem") or {}).get("overall_acc_mean"),
        0.002,
        "rounding",
    )

    def s14_diff(rnd: str) -> Optional[float]:
        rows = _csv_rows("s14_esn_disturbance_chain_v1")
        if not rows:
            return None
        by: dict[int, dict[str, float]] = {}
        for r in rows:
            try:
                by.setdefault(int(float(r["seed_idx"])), {})[r["arm"]] = float(r[rnd])
            except (KeyError, ValueError, TypeError):
                continue
        diffs = [
            d["esn_dual"] - d["esn_fast"]
            for d in by.values()
            if "esn_dual" in d and "esn_fast" in d
        ]
        return mean(diffs) if diffs else None

    claim(
        "C", "E3-transfer", "s14 paired MC diff at r1 (dual-fast)", "Paper C",
        -0.78,
        lambda: s14_diff("r1_mc"),
        0.05,
        "rounding",
    )
    claim(
        "C", "E3-transfer", "s14 paired MC diff at r2 (dual-fast)", "Paper C",
        -0.76,
        lambda: s14_diff("r2_mc"),
        0.05,
        "rounding",
    )
    claim(
        "C", "E3-transfer", "s14 paired MC diff at r3 (dual-fast)", "Paper C",
        -0.69,
        lambda: s14_diff("r3_mc"),
        0.05,
        "rounding",
        note="metadata robustness transfer falsified; 0/10 seeds positive",
    )

    s2 = _j("s2_online_readout_v1.json") or {}
    s2a = s2.get("aggregates", [])
    # first drift_binary online arm
    claim(
        "B", "S2", "online RLS drift-binary mean acc", "Paper B / README_REDEM",
        0.978,
        lambda: mean(
            [
                a["mean_acc_all_mean"]
                for a in s2a
                if a.get("task") == "drift_binary"
                and a.get("readout") == "rls"
                and isinstance(a.get("mean_acc_all_mean"), (int, float))
            ]
        )
        or None,
        0.01,
        "rounding",
        note="0.974–0.982 band; mean of the two rls arms",
    )

    # --- Paper D ---
    s19 = _rows_json("s19_ssm_rls_readout_v1.json")
    # s19 stores per-run rows; B-proj stream metric may be in params/results
    # Fall back: F re-exports the same B-lin-clip anchor via s50.
    s50 = _j("s50_paper_f_pilot_nl_readout_v1.json") or {}
    s50r = s50.get("results", [])
    claim(
        "D", "P1", "B-proj / B-lin-clip stream ppl (input path)", "abstract / Paper D/F",
        11.75,
        lambda: _arm_mean(s50r, "B-lin-clip", "stream_ppl")
        or _arm_mean(s50r, "B-proj", "stream_ppl"),
        0.005,
        "rounding",
    )
    claim(
        "D", "P1", "oracle / pooled-table ceiling stream ppl", "Paper D",
        7.25,
        lambda: _arm_mean(s50r, "B-lin-clip", "oracle_ppl"),
        0.02,
        "rounding",
    )

    def s20_forget_diff(tau: float) -> Optional[float]:
        rows = _csv_rows("s20_ssm_m3_routing_v1")
        if not rows:
            return None
        by: dict[tuple[str, float], list[float]] = {}
        for r in rows:
            try:
                key = (r["arm"], float(r["tau_m"]))
                by.setdefault(key, []).append(float(r["forgetting_ppl"]))
            except (KeyError, ValueError, TypeError):
                continue
        a1 = by.get(("A1", 0.0))
        a3 = by.get(("A3", tau))
        if not a1 or not a3:
            return None
        return mean(a3) - mean(a1)

    claim(
        "D", "P2", "A3-A1 forgetting diff at tau_m=200", "Paper D",
        -2.05,
        lambda: s20_forget_diff(200.0),
        0.02,
        "rounding",
    )
    claim(
        "D", "P2", "A3-A1 forgetting diff at tau_m=500", "Paper D",
        -1.87,
        lambda: s20_forget_diff(500.0),
        0.02,
        "rounding",
    )
    claim(
        "D", "P2", "A3-A1 forgetting diff at tau_m=1000", "Paper D",
        -1.20,
        lambda: s20_forget_diff(1000.0),
        0.05,
        "rounding",
    )

    s21 = _j("s21_ssm_m4_m5_v1.json") or {}
    s21a = s21.get("aggregates", [])
    claim(
        "D", "P3", "A3-soft stream ppl (E1)", "Paper D",
        8.22,
        lambda: (_agg(s21a, exp="E1", arm="A3-soft") or {}).get("stream_ppl_mean"),
        0.01,
        "rounding",
    )
    claim(
        "D", "P3", "A3-abrupt stream ppl (E1)", "Paper D",
        10.03,
        lambda: (_agg(s21a, exp="E1", arm="A3-abrupt") or {}).get("stream_ppl_mean"),
        0.01,
        "rounding",
    )
    claim(
        "D", "P3", "M5 regulated whitened-state norm mean", "Paper D",
        11.3,
        lambda: (_agg(s21a, exp="E2", arm="REG-clean") or {}).get("norm_mean_mean"),
        0.02,
        "rounding",
    )
    claim(
        "D", "P3", "bare host whitened-state norm mean", "Paper D",
        50.2,
        lambda: (_agg(s21a, exp="E2", arm="BARE-clean") or {}).get("norm_mean_mean"),
        0.02,
        "rounding",
    )

    def s22_arm(key: str) -> Optional[float]:
        rows = _rows_json("s22_ssm_p4_benchmark_v1")
        if not rows:
            return None
        vals = [
            r[key] for r in rows if r.get("arm") == "SSM-REDEM" and isinstance(r.get(key), (int, float))
        ]
        return mean(vals) if vals else None

    claim(
        "D", "P4", "SSM-REDEM stream ppl", "Paper D tables",
        13.18,
        lambda: s22_arm("stream_ppl"),
        0.01,
        "rounding",
    )
    claim(
        "D", "P4", "SSM-REDEM forgetting ppl", "Paper D tables",
        8.93,
        lambda: s22_arm("forgetting_ppl"),
        0.01,
        "rounding",
    )

    def s23_redem_stream() -> Optional[float]:
        rows = _rows_json("s23_ssm_p4_realtext_v1")
        if not rows:
            return None
        vals = [
            r["stream_ppl"]
            for r in rows
            if r.get("arm") == "SSM-REDEM" and isinstance(r.get("stream_ppl"), (int, float))
        ]
        return mean(vals) if vals else None

    claim(
        "D", "P4-text", "REDEM-SSM real-text stream ppl", "Paper D",
        12.07,
        s23_redem_stream,
        0.01,
        "rounding",
    )

    def s31_ceiling() -> Optional[float]:
        d = _j("s31_char_bigram_oracle_v1.json") or {}
        agg = d.get("aggregates") or {}
        return agg.get("stream_ppl_full_mean")

    claim(
        "D", "P4-text", "char-bigram full-book ceiling ppl", "Paper D",
        10.97,
        s31_ceiling,
        0.01,
        "rounding",
    )

    s22 = _j("s22_ssm_p4_benchmark_v1.json") or {}
    cmp22 = (s22.get("params") or {}).get("comparisons") or {}
    claim(
        "D", "P4", "SSM-REDEM vs bare stream diff (ppl, negative=better)", "Paper D tables",
        -2.25,
        lambda: (cmp22.get("SSM-REDEM_vs_SSM-bare") or {}).get("stream_diff_mean"),
        0.02,
        "rounding",
    )
    claim(
        "D", "P4", "SSM-REDEM vs bare forgetting diff", "Paper D tables",
        -4.47,
        lambda: (cmp22.get("SSM-REDEM_vs_SSM-bare") or {}).get("forgetting_diff_mean"),
        0.02,
        "rounding",
    )
    claim(
        "D", "P4", "SSM-REDEM vs TF-A1 stream diff", "Paper D tables",
        -9.28,
        lambda: (cmp22.get("SSM-REDEM_vs_TF-A1") or {}).get("stream_diff_mean"),
        0.02,
        "rounding",
    )

    # --- Paper E ---
    s52 = _j("s52_dynamic_memory_v1.json") or {}
    s52a = s52.get("aggregates", [])
    pop = _agg(s52a, readout="pop_mem")
    rls = _agg(s52a, readout="rls_single")
    claim(
        "E", "R5", "frozen-hypothesis memory mean acc", "abstract / Paper E",
        0.888,
        lambda: pop["mean_acc_all_mean"] if pop else None,
        0.002,
        "rounding",
    )
    claim(
        "E", "R5", "re-adaptation / rls_single mean acc", "Paper E",
        0.838,
        lambda: rls["mean_acc_all_mean"] if rls else None,
        0.002,
        "rounding",
    )

    def retention_gain_pp() -> Optional[float]:
        rows = _csv_rows("s52_dynamic_memory_v1")
        if not rows:
            return None
        by: dict[str, list[float]] = {}
        for r in rows:
            try:
                s3 = float(r["seg3_acc"])
                s4 = float(r["seg4_acc"])
            except (KeyError, ValueError, TypeError):
                continue
            by.setdefault(r.get("readout", ""), []).append((s3 + s4) / 2.0)
        if "pop_mem" not in by or "rls_single" not in by:
            return None
        return (mean(by["pop_mem"]) - mean(by["rls_single"])) * 100.0

    claim(
        "E", "R5", "frozen-snapshot retention gain (pp)", "abstract / Paper E / s65",
        8.70,
        retention_gain_pp,
        0.02,
        "rounding",
        note="mean(seg3,seg4) pop_mem - rls_single",
    )

    s65 = _j("s65_ewc_baseline_v1.json") or {}
    stats = s65.get("stats") or {}
    ewc = (stats.get("rls_ewc_vs_rls_single") or {}).get("mean_acc_all") or {}
    claim(
        "E", "EWC", "EWC whole-run accuracy change (pp)", "abstract / Paper E",
        -0.85,
        lambda: ewc.get("mean_diff", 0.0) * 100.0 if ewc else None,
        0.05,
        "rounding",
    )
    claim(
        "E", "EWC", "EWC whole-run paired t", "Paper E",
        -3.87,
        lambda: ewc.get("t") if ewc else None,
        0.02,
        "rounding",
    )
    claim(
        "E", "EWC", "s65 reference arm bit-matches s52 (max abs diff)", "reproduction_check",
        0.0,
        lambda: (s65.get("reproduction_check_vs_s52") or {}).get("max_abs_difference"),
        0.0,
        "identity",
        note="must stay exactly 0",
    )

    def cross_family_t() -> Optional[float]:
        # coupled ring-segment gain: pop_mem - rls_single on seg4_acc, random_graph_k25
        rows = _csv_rows("s61_cross_family_v1")
        if not rows:
            return None
        by: dict[int, dict[str, float]] = {}
        for r in rows:
            if r.get("substrate") != "random_graph_k25":
                continue
            try:
                seed = int(r["seed_idx"])
                by.setdefault(seed, {})[r["readout"]] = float(r["seg4_acc"])
            except (KeyError, ValueError, TypeError):
                continue
        diffs = [
            d["pop_mem"] - d["rls_single"]
            for _, d in sorted(by.items())
            if "pop_mem" in d and "rls_single" in d
        ]
        if len(diffs) < 2:
            return None
        m = mean(diffs)
        sd = stdev(diffs)
        if sd <= 0:
            return None
        return m / (sd / math.sqrt(len(diffs)))

    claim(
        "E", "R5", "cross-family ring gain paired t (corrected)", "Paper E / MAINTENANCE",
        2.45,
        cross_family_t,
        0.05,
        "rounding",
        note="seg4_acc pop_mem-rls_single on random_graph_k25; 2.45 > 2.262",
    )

    # --- Paper E R1–R8 logic-chain claims ---
    s39 = _j("s39_prediction_reward_v2.json") or _j("s39_prediction_reward_v1.json") or {}
    s39a = s39.get("aggregates", [])
    flip_unc = _agg(s39a, substrate="parallel", readout="probe_directed")
    flip_cpl = _agg(s39a, substrate="random_graph_k25", readout="probe_directed")
    or_unc = _agg(s39a, substrate="parallel", readout="rls_oracle")
    or_cpl = _agg(s39a, substrate="random_graph_k25", readout="rls_oracle")
    claim(
        "E", "R1", "post-inversion flip acc uncoupled", "abstract / Paper E",
        0.898,
        lambda: flip_unc["post_swap_acc_mean"] if flip_unc else None,
        0.002,
        "rounding",
        note="s39 probe_directed / s40 passive_flip (same measurement)",
    )
    claim(
        "E", "R1", "post-inversion flip acc coupled", "abstract / Paper E",
        0.939,
        lambda: flip_cpl["post_swap_acc_mean"] if flip_cpl else None,
        0.002,
        "rounding",
        note="s39 probe_directed post_swap mean on random_graph_k25",
    )
    claim(
        "E", "R1", "oracle post-inversion acc uncoupled", "Paper E",
        0.975,
        lambda: or_unc["post_swap_acc_mean"] if or_unc else None,
        0.002,
        "rounding",
    )
    claim(
        "E", "R1", "oracle post-inversion acc coupled", "Paper E",
        1.000,
        lambda: or_cpl["post_swap_acc_mean"] if or_cpl else None,
        0.002,
        "rounding",
    )

    s40 = _j("s40_meta_flip_v1.json") or {}
    s40a = s40.get("aggregates", [])
    lat = _agg(s40a, substrate="parallel", readout="passive_flip", margin=0.1) or _agg(
        s40a, substrate="parallel", readout="passive_flip", margin=0.2
    )
    claim(
        "E", "R1", "first-flip latency (blocks after inversion)", "Paper E",
        39.0,
        lambda: lat["flip_latency_mean"] if lat else None,
        0.0,
        "identity",
        note="s40 passive_flip; zero seed SD, flip checked every FLIP_CHECK=20",
    )

    s41 = _j("s41_meta_reward_v1.json") or {}
    s41a = s41.get("aggregates", [])
    meta_a = _agg(s41a, substrate="parallel", readout="meta_self", env="A")
    meta_b_unc = _agg(s41a, substrate="parallel", readout="meta_self", env="B")
    meta_b_cpl = _agg(s41a, substrate="random_graph_k25", readout="meta_self", env="B")
    claim(
        "E", "R1", "learned flip value v (post-flip contrast)", "Paper E",
        0.504,
        lambda: meta_a["v_final_mean"] if meta_a else None,
        0.002,
        "rounding",
        note="s41 meta_self env A v_final_mean",
    )
    claim(
        "E", "R1", "stable-stream whole-run acc uncoupled (bit-identical no-meta)", "Paper E",
        0.897925,
        lambda: meta_b_unc["mean_acc_all_mean"] if meta_b_unc else None,
        1e-5,
        "precision",
        note="s41 meta_self env B; matches no-meta arm to six decimals",
    )
    claim(
        "E", "R1", "stable-stream whole-run acc coupled (bit-identical no-meta)", "Paper E",
        0.938241,
        lambda: meta_b_cpl["mean_acc_all_mean"] if meta_b_cpl else None,
        1e-5,
        "precision",
    )

    s44 = _j("s44_regime_zoo_v1.json") or {}
    s44a = s44.get("aggregates", [])
    zoo_items = [
        ("R1 zoo coupled full-swap rmhl post", "R1_full_swap", "random_graph_k25", "rmhl", "post_swap_acc_mean", 0.058),
        ("R1 zoo coupled full-swap flip post", "R1_full_swap", "random_graph_k25", "single_flip", "post_swap_acc_mean", 0.942),
        ("R1 zoo coupled near-inversion rmhl post", "R4_near_inversion", "random_graph_k25", "rmhl", "post_swap_acc_mean", 0.053),
        ("R1 zoo coupled near-inversion flip post", "R4_near_inversion", "random_graph_k25", "single_flip", "post_swap_acc_mean", 0.947),
        ("R1 zoo partial-shift flip post-window", "R2_partial_shift", "random_graph_k25", "single_flip", "post_swap_acc_mean", 0.485),
        ("R1 zoo partial-shift flip steady", "R2_partial_shift", "random_graph_k25", "single_flip", "steady_acc_mean", 0.508),
        ("R1 zoo partial-shift flip mean flips", "R2_partial_shift", "random_graph_k25", "single_flip", "n_flips_mean", 2.7),
    ]
    for qname, regime, substrate, readout, field, docv in zoo_items:
        def _zoo(regime=regime, substrate=substrate, readout=readout, field=field):
            a = _agg(s44a, regime=regime, substrate=substrate, readout=readout)
            return a.get(field) if a else None
        claim(
            "E", "R1", qname, "Paper E / Table tab:zoo",
            docv,
            _zoo,
            0.01,
            "rounding",
            note="s44_regime_zoo",
        )

    s42 = _j("s42_corrupt_signal_v1.json") or {}
    s42a = s42.get("aggregates", [])
    cor_unc = _agg(s42a, substrate="parallel", readout="meta_corrupt")
    cor_cpl = _agg(s42a, substrate="random_graph_k25", readout="meta_corrupt")
    claim(
        "E", "R1", "corrupt-window acc uncoupled (honest negative)", "Paper E",
        0.106,
        lambda: cor_unc["corrupt_acc_mean"] if cor_unc else None,
        0.01,
        "rounding",
        note="s42; self-deception under wrong reward, not self-detectable",
    )
    claim(
        "E", "R1", "corrupt-window acc coupled (honest negative)", "Paper E",
        0.064,
        lambda: cor_cpl["corrupt_acc_mean"] if cor_cpl else None,
        0.01,
        "rounding",
    )

    s46 = _j("s46_rule_adjudication_v1.json") or {}
    s46a = s46.get("aggregates", s46.get("rows", []))
    d_unc = _agg(s46a, regime="R1_full_swap", substrate="parallel", readout="delta_derived")
    d_cpl = _agg(s46a, regime="R1_full_swap", substrate="random_graph_k25", readout="delta_derived")
    rm_unc = _agg(s46a, regime="R1_full_swap", substrate="parallel", readout="rmhl")
    claim(
        "E", "R1", "rule-adjudication delta pre-swap uncoupled", "Paper E",
        0.990,
        lambda: d_unc["pre_swap_acc_mean"] if d_unc else None,
        0.002,
        "rounding",
        note="s46 delta_derived on identity labels (rule selection, not information)",
    )
    claim(
        "E", "R1", "rule-adjudication delta steady uncoupled", "Paper E",
        0.989,
        lambda: d_unc["steady_acc_mean"] if d_unc else None,
        0.002,
        "rounding",
    )
    claim(
        "E", "R1", "rule-adjudication RMHL post-swap uncoupled", "Paper E",
        0.097,
        lambda: rm_unc["post_swap_acc_mean"] if rm_unc else None,
        0.01,
        "rounding",
    )

    # R2 representation / capacity
    s53 = _j("s53_auto_basis_v1.json") or {}
    cap = s53.get("raw_capacity_probe") or {}
    cap_ps = cap.get("per_substrate") or {}
    claim(
        "E", "R2", "raw capacity probe coupled (circle)", "Paper E",
        0.84,
        lambda: (cap_ps.get("random_graph_k25") or {}).get("cap_mean"),
        0.01,
        "rounding",
        note="s53 raw_capacity_probe; population SD over 2550 readings",
    )
    claim(
        "E", "R2", "raw capacity probe uncoupled (circle)", "Paper E",
        0.55,
        lambda: (cap_ps.get("parallel") or {}).get("cap_mean"),
        0.01,
        "rounding",
    )

    s58a = _j("s58a_complexity_trigger_v1.json") or {}
    s58aa = s58a.get("aggregates", [])
    claim(
        "E", "R2", "RLS acc 4D shell2 (two-shell)", "Paper E",
        0.547,
        lambda: (_agg(s58aa, dim="4D", boundary="shell2", substrate="random_graph_k25", readout="rls_raw") or {}).get("mean_acc_all_mean"),
        0.005,
        "rounding",
    )
    claim(
        "E", "R2", "RLS acc 4D shell3 (three-shell)", "Paper E",
        0.559,
        lambda: (_agg(s58aa, dim="4D", boundary="shell3", substrate="random_graph_k25", readout="rls_raw") or {}).get("mean_acc_all_mean"),
        0.005,
        "rounding",
        note="shallow rise +0.015 from two to six quadratic surfaces (flat-to-rising, not saturation)",
    )

    s58d = _j("s58d_kernel_capacity_sweep_v1.json") or {}
    s58da = s58d.get("aggregates", [])

    def _margin(env: str, kappa: float, n_units: int) -> Optional[float]:
        a = _agg(s58da, env=env, kappa=kappa, n_units=n_units)
        return a.get("margin_rls") if a else None

    claim(
        "E", "R2", "4D-shell margin at N=64", "Paper E",
        0.034,
        lambda: _margin("4Dshell", 25.0, 64),
        0.02,
        "rounding",
    )
    claim(
        "E", "R2", "4D-shell margin at N=256", "Paper E",
        0.100,
        lambda: _margin("4Dshell", 25.0, 256),
        0.02,
        "rounding",
    )
    claim(
        "E", "R2", "4D-shell margin at N=512", "Paper E",
        0.107,
        lambda: _margin("4Dshell", 25.0, 512),
        0.02,
        "rounding",
    )
    claim(
        "E", "R2", "4D-shell margin at kappa=20 (sampled peak)", "Paper E",
        0.1185,
        lambda: _margin("4Dshell", 20.0, 256),
        0.01,
        "rounding",
        note="peak at kappa≈20–25, not at deployed 25; both within one seed SD",
    )
    claim(
        "E", "R2", "4D-shell margin at deployed kappa=25", "Paper E",
        0.0995,
        lambda: _margin("4Dshell", 25.0, 256),
        0.01,
        "rounding",
    )
    claim(
        "E", "R2", "2D-circle margin flat across N (N=64)", "Paper E",
        0.401,
        lambda: _margin("2Dcircle", 25.0, 64),
        0.01,
        "rounding",
    )

    # R3 segment-by-segment memory gains (paired pop_mem - rls_single)
    def _seg_gain_pp(seg: str) -> Optional[float]:
        rows = _csv_rows("s52_dynamic_memory_v1")
        by: dict[int, dict[str, float]] = {}
        for r in rows:
            try:
                by.setdefault(int(r["seed_idx"]), {})[r["readout"]] = float(r[seg])
            except (KeyError, ValueError, TypeError):
                continue
        diffs = [
            d["pop_mem"] - d["rls_single"]
            for _, d in sorted(by.items())
            if "pop_mem" in d and "rls_single" in d
        ]
        return mean(diffs) * 100.0 if diffs else None

    def _seg_t(seg: str) -> Optional[float]:
        rows = _csv_rows("s52_dynamic_memory_v1")
        by: dict[int, dict[str, float]] = {}
        for r in rows:
            try:
                by.setdefault(int(r["seed_idx"]), {})[r["readout"]] = float(r[seg])
            except (KeyError, ValueError, TypeError):
                continue
        diffs = [
            d["pop_mem"] - d["rls_single"]
            for _, d in sorted(by.items())
            if "pop_mem" in d and "rls_single" in d
        ]
        if len(diffs) < 2:
            return None
        sd = stdev(diffs)
        if sd <= 0:
            return None
        return mean(diffs) / (sd / math.sqrt(len(diffs)))

    for seg, docv, tdoc in [
        ("seg1_acc", 4.2, 2.72),
        ("seg2_acc", 3.6, 2.73),
        ("seg3_acc", 2.6, 2.02),
        ("seg4_acc", 14.9, 11.5),
    ]:
        claim(
            "E", "R3", f"per-segment memory gain {seg} (pp)", "Paper E",
            docv,
            lambda seg=seg: _seg_gain_pp(seg),
            0.02,
            "rounding",
            note="paired pop_mem - rls_single; only seg4 is 10/10",
        )
        claim(
            "E", "R3", f"per-segment memory gain paired t {seg}", "Paper E",
            tdoc,
            lambda seg=seg: _seg_t(seg),
            0.02,
            "rounding",
            note="df=9; seg3 t=2.02 < 2.262 (n.s.)",
        )

    pop_plain = _agg(s52a, readout="pop_plain")
    claim(
        "E", "R3", "population control (no frozen snapshot) mean acc", "Paper E",
        0.839,
        lambda: pop_plain["mean_acc_all_mean"] if pop_plain else None,
        0.002,
        "rounding",
        note="indistinguishable from rls_single 0.838; gains are frozen memory, not ensemble size",
    )

    # R4 reliability cliff (relative capacity sense)
    s58b = _j("s58b_relative_sense_stress_v1.json") or {}
    s58br = s58b.get("reliability", [])

    def _reliab(seq: str, seg_len: int) -> Optional[float]:
        for r in s58br:
            if r.get("seq_kind") == seq and r.get("seg_len") == seg_len:
                return r.get("reliability")
        return None

    claim(
        "E", "R4", "sense reliability ABAB SEG=500", "Paper E",
        0.90,
        lambda: _reliab("ABAB", 500),
        0.01,
        "rounding",
        note="(1-P_false)*P_correct; 18/20 correct, 0/20 false",
    )
    claim(
        "E", "R4", "sense reliability ABAB SEG=250", "Paper E",
        0.27,
        lambda: _reliab("ABAB", 250),
        0.01,
        "rounding",
    )
    claim(
        "E", "R4", "sense reliability ABAB SEG=125 (cliff floor)", "Paper E",
        0.00,
        lambda: _reliab("ABAB", 125),
        0.0,
        "identity",
        note="fails silently; 0/20 correct, 0/20 false",
    )

    # R5 self-tuned snapshot gate (learned, not hand-set)
    s59 = _j("s59_self_tuned_gate_v1.json") or {}
    s59a = s59.get("aggregates", [])
    gate_dyn = _agg(s59a, env="dyn", readout="mem_vlearn")
    claim(
        "E", "R5", "learned snapshot gate v_snap final (dynamic)", "Paper E",
        0.252,
        lambda: gate_dyn["v_snap_final_mean"] if gate_dyn else None,
        0.01,
        "rounding",
        note="s59 mem_vlearn; stays low when memory is useful",
    )
    claim(
        "E", "R5", "memory selection fraction (dynamic)", "Paper E",
        0.227,
        lambda: gate_dyn["sel_frac_mean"] if gate_dyn else None,
        0.02,
        "rounding",
    )

    # R6 D2 timescale band
    s58e = _j("s58e_d2_timescale_v1.json") or {}
    s58ea = s58e.get("aggregates", [])

    def _d2_post(ratio: float) -> Optional[float]:
        a = _agg(s58ea, substrate="random_graph_k25", readout="meta_self", ratio=ratio)
        return a.get("post_mean_mean") if a else None

    claim(
        "E", "R6", "D2 post-swap at ratio 0.25 (fail band)", "Paper E",
        0.486,
        lambda: _d2_post(0.25),
        0.01,
        "rounding",
        note="tau_env/tau_2; below ratio≈1 no recovery",
    )
    claim(
        "E", "R6", "D2 post-swap at ratio 1.0 (intermediate)", "Paper E",
        0.561,
        lambda: _d2_post(1.0),
        0.01,
        "rounding",
    )
    claim(
        "E", "R6", "D2 post-swap at ratio 1.25 (band interior)", "Paper E",
        0.791,
        lambda: _d2_post(1.25),
        0.01,
        "rounding",
    )
    claim(
        "E", "R6", "D2 post-swap at ratio 1.5 (band interior)", "Paper E",
        0.850,
        lambda: _d2_post(1.5),
        0.01,
        "rounding",
    )
    claim(
        "E", "R6", "D2 post-swap at ratio 1.75 (band interior)", "Paper E",
        0.909,
        lambda: _d2_post(1.75),
        0.01,
        "rounding",
    )
    claim(
        "E", "R6", "D2 post-swap at ratio 2.0 (complete recovery)", "Paper E",
        0.938,
        lambda: _d2_post(2.0),
        0.005,
        "rounding",
        note="transition band inside [1,2]; success rate 1.0 at ratio≥2",
    )

    # R7 content–rendering separation
    s63 = _j("s63_content_rendering_v1.json") or {}
    s63a = s63.get("aggregates", [])

    def _s63(env: str, switch: int, readout: str, field: str = "mean_acc_all_mean") -> Optional[float]:
        a = _agg(s63a, env=env, switch_every=switch, substrate="random_graph_k25", readout=readout)
        return a.get(field) if a else None

    claim(
        "E", "R7", "single-speaker linear acc", "Paper E",
        0.938,
        lambda: _s63("single_A", 0, "rls_single"),
        0.002,
        "rounding",
        note="s63 single_A; A-B-A-B degrades the single map",
    )
    claim(
        "E", "R7", "A-B-A-B linear acc (switch_every=200)", "Paper E",
        0.900,
        lambda: _s63("mixed_AB", 200, "rls_single"),
        0.002,
        "rounding",
    )
    claim(
        "E", "R7", "A-B-A-B memory-recovered acc", "Paper E",
        0.913,
        lambda: _s63("mixed_AB", 200, "pop_mem"),
        0.002,
        "rounding",
        note="R5 memory separates renders; R2 expansion does not",
    )
    claim(
        "E", "R7", "capacity probe single-speaker", "Paper E",
        0.949,
        lambda: _s63("single_A", 0, "rls_single", "capacity_probe_mean"),
        0.002,
        "rounding",
        note="registers render drop below REL_MARGIN=0.10 (max drop 0.084)",
    )
    claim(
        "E", "R7", "capacity probe A-B-A-B", "Paper E",
        0.900,
        lambda: _s63("mixed_AB", 200, "rls_single", "capacity_probe_mean"),
        0.002,
        "rounding",
    )

    # R8 multi-source validation / majority fallacy
    s64 = _j("s64_multi_source_validation_v1.json") or {}
    s64a = s64.get("aggregates", [])

    def _s64(scenario: str, readout: str) -> Optional[float]:
        a = _agg(s64a, scenario=scenario, substrate="random_graph_k25", readout=readout)
        return a.get("mean_acc_all_mean") if a else None

    claim(
        "E", "R8", "minority-good majority-rule acc", "Paper E / Table tab:multisource",
        0.684,
        lambda: _s64("minority_good", "vote_pure"),
        0.002,
        "rounding",
        note="majority fallacy: wrong majority collapses",
    )
    claim(
        "E", "R8", "minority-good validation-selector acc", "Paper E / Table tab:multisource",
        1.000,
        lambda: _s64("minority_good", "emasel_pure"),
        0.002,
        "rounding",
        note="reward-rate-EMA argmax finds the correct source",
    )
    claim(
        "E", "R8", "drifting majority-rule acc", "Paper E / Table tab:multisource",
        0.681,
        lambda: _s64("drifting", "vote_pure"),
        0.002,
        "rounding",
    )
    claim(
        "E", "R8", "drifting validation-selector acc", "Paper E / Table tab:multisource",
        0.991,
        lambda: _s64("drifting", "emasel_pure"),
        0.002,
        "rounding",
        note="selection fraction 0.991",
    )
    claim(
        "E", "R8", "all-bad majority-rule acc (no source to trust)", "Paper E",
        0.494,
        lambda: _s64("all_bad", "vote_pure"),
        0.01,
        "rounding",
    )

    # --- Paper F ---
    claim(
        "F", "C1", "B-softmax stream ppl", "abstract / Paper F",
        8.65,
        lambda: _arm_mean(s50r, "B-softmax-sgd", "stream_ppl"),
        0.005,
        "rounding",
    )
    claim(
        "F", "C1", "stream CE improvement magnitude (nats/token)", "abstract / Paper F",
        0.307,
        lambda: (
            # CE difference from same-token-set is a stored table number; derive
            # from ln(ppl) only as a secondary check with loose tol via order/rounding
            abs(math.log(_arm_mean(s50r, "B-softmax-sgd", "stream_ppl"))
                - math.log(_arm_mean(s50r, "B-lin-clip", "stream_ppl")))
            if s50r else None
        ),
        0.05,
        "rounding",
        note="ln(ppl) proxy; manuscript same-token-set CE is the primary instrument",
    )
    claim(
        "F", "C1", "clip-floor park rate low (~1.6%)", "Paper F",
        1.6,
        lambda: (_arm_mean(s50r, "B-lin-clip", "clip_frac") or 0.0) * 100.0,
        0.15,
        "rounding",
    )
    claim(
        "F", "C1", "clip-floor park rate high (~10.2%)", "Paper F",
        10.2,
        lambda: (_arm_mean(s50r, "Skip-lin-clip", "clip_frac") or 0.0) * 100.0,
        0.15,
        "rounding",
    )
    claim(
        "F", "C1", "simplex does not repair (stream ppl delta vs lin-clip)", "Paper F",
        0.03,
        lambda: abs(
            (_arm_mean(s50r, "B-simplex", "stream_ppl") or 0.0)
            - (_arm_mean(s50r, "B-lin-clip", "stream_ppl") or 0.0)
        ),
        0.10,
        "rounding",
        note="manuscript quotes +0.03; measured 0.0284",
    )

    s51 = _j("s51_paper_f_learned_gate_v2.json") or _j("s51_paper_f_learned_gate_v1.json") or {}
    s51r = s51.get("results", [])
    claim(
        "F", "C4", "Gate-C-topk stream ppl", "abstract / Paper F",
        7.03,
        lambda: _arm_mean(s51r, "Gate-C-topk-softmax", "stream_ppl"),
        0.01,
        "rounding",
    )

    s52f = _j("s52_paper_f_soft_route_experts_v1.json") or {}
    s52fr = s52f.get("results", [])
    claim(
        "F", "C5", "soft routing forgetting ppl", "Paper F",
        7.79,
        lambda: _arm_mean(s52fr, "Soft-topk", "forgetting_ppl"),
        0.01,
        "rounding",
    )
    claim(
        "F", "C5", "hard routing forgetting ppl", "Paper F",
        6.94,
        lambda: _arm_mean(s52fr, "Hard-topk", "forgetting_ppl"),
        0.01,
        "rounding",
    )

    s66 = _j("s66_external_ssm_baseline_v1.json") or {}
    s66r = s66.get("results", [])
    claim(
        "F", "external", "X-SelSSM-frozen stream ppl", "Paper F / Discussion",
        9.03,
        lambda: _arm_mean(s66r, "X-SelSSM-frozen", "stream_ppl"),
        0.01,
        "rounding",
    )
    claim(
        "F", "external", "F Gate-C-topk stream anchor (same host/stream)", "Paper F",
        7.03,
        lambda: _arm_mean(s66r, "F-Gate-C-topk-softmax", "stream_ppl"),
        0.01,
        "rounding",
    )


def main() -> int:
    register_all()
    fail = 0
    lines = [
        "# verification_report",
        "",
        f"claims registered: **{len(CLAIMS)}**",
        "",
        "| paper | section | quantity | where | doc | computed | kind/tol | ok | detail |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c in CLAIMS:
        evaluate(c)
        if not c.ok:
            fail += 1
        comp = c.computed
        if isinstance(comp, float):
            comp_s = f"{comp:.6g}"
        else:
            comp_s = str(comp)
        lines.append(
            f"| {c.paper} | {c.section} | {c.quantity} | {c.where} | "
            f"{c.doc_value:g} | {comp_s} | {c.kind}/{c.tol:g} | "
            f"{'PASS' if c.ok else 'FAIL'} | {c.detail} |"
        )
    lines.append("")
    lines.append(f"failures: **{fail}**")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    OUT_JSON.write_text(
        json.dumps(
            {
                "n_claims": len(CLAIMS),
                "n_fail": fail,
                "claims": [asdict(c) for c in CLAIMS],
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"verify_claims: {len(CLAIMS)} claims / {fail} failures")
    for c in CLAIMS:
        if not c.ok:
            print(f"  FAIL [{c.paper}/{c.section}] {c.quantity}: {c.detail}")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
