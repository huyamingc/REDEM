#!/usr/bin/env python3
"""
Multi-source validation: can a reward-driven learner pick the CORRECT source
when several information sources give CONFLICTING suggestions? (REDEM S64)
=============================================================================
Type:           PAPER
Paper Section:  R5 follow-up (the derivation chain's selection mechanism as
                a multi-source validator; scoping the "majority fallacy" and
                the D4 self-detection boundary)
Experiment:     The user question: "I looked up one piece of information (A)
                but found all other sources contradict it -- can I use a
                majority (少数服从多数) rule to exclude the bad one and find
                the correct one among many sources?" In the derivation chain
                this is NOT a new mechanism: it is the hypothesis-competition
                / reward-rate-selection device of S52 applied to EXTERNAL
                information sources instead of internal frozen hypotheses.
                This experiment tests the claim directly and draws its honest
                boundary.

                Content: the fixed 2-D semantic point stream u with label
                y = g(u) (linear boundary, exactly as s63 single_A; the
                content is decodable from the substrate -- worker_true
                ~0.9 is the oracle upper bound). What changes is the LABEL
                layer: M=5 information sources each emit a block-level
                suggestion hat(y)_i(b):
                  good  : hat(y) = y always (correct source).
                  noise : hat(y) = flip(y) with prob 0.5 (i.i.d. per block;
                          the "seemingly-plausible noise": 50% correct, so
                          its reward-rate EMA sits near 0 -- the hardest
                          kind of bad source, NOT an anti-correlated one).
                  drift_g2n / drift_n2g : a source that is good before block
                          DRIFT_BLOCK and noise after (or vice versa) -- the
                          "source reliability drifts over time" case.
                The learner sees ONLY the suggestions (hat(y)_i), not the
                true y a priori; y is revealed AFTER each block (the reward
                channel, same as every S39-S63 arm).

Arms (the 2x2: {majority vs validation} x {pure aggregation vs worker
learning}, plus the no-source oracle):
  worker_true  : worker RLS trains on the TRUE label y. Upper bound for the
                 substrate decoding (no source layer at all).
  vote_pure    : output = MAJORITY of the 5 suggestions each block, no
                 learning. The user's literal "少数服从多数" rule. No
                 history, no source-memory.
  worker_majority : worker RLS trains on the MAJORITY suggestion each block
                 (a system that blindly trusts the crowd; the crowd is its
                 teacher).
  emasel_pure  : output = suggestion of the source with the highest
                 reward-rate EMA r_ema_i (validation-driven selection, no
                 worker learning). r_ema_i updated each block by comparing
                 the source's suggestion to the revealed y.
  worker_emasel  : worker RLS trains on the suggestion of the r_ema-selected
                 source. The derivation-chain mechanism (S52 argmax-r_ema
                 applied to sources): the system's knowledge is built from
                 what it validates as reliable.

Scenarios (M=5 sources; good/noise counts):
  majority_good : 3 good + 2 noise   (the user's default world)
  minority_good : 1 good + 4 noise   (majority is WRONG -- the stress case)
  all_good      : 5 good             (no conflict)
  all_bad       : 5 noise            (no correct source -- D4 territory)
  drifting      : 1 good->noise, 1 noise->good, +3 noise at DRIFT_BLOCK
                  (the identity of the correct source migrates mid-run)

Predictions (falsifiable):
  P1 (majority works when the majority is right): majority_good -- vote_pure
     ~ high (3 guaranteed-correct votes dominate), worker_majority ~
     worker_true; ema arms also high. The user's intuition holds in the
     default world.
  P2 (majority is a FALLACY when the majority is wrong): minority_good --
     vote_pure collapses to ~0.69 (P(Binomial(4,0.5)>=2)=11/16: 1 good vote
     + 2+ of 4 noise votes), worker_majority degrades, but emasel_pure and
     worker_emasel stay high (the single good source's r_ema ~ +0.9 >> the
     noise sources' ~0, so validation selects it regardless of count).
     THIS is the derivation chain's incremental claim: find the correct
     source by post-hoc validation, not by counting heads.
  P3 (D4 boundary): all_bad -- every arm ~0.5 AND ema selection has no
     signal (all r_ema ~0, argmax picks arbitrarily, n_switches high): the
     system cannot tell it is being fed only noise. Honest boundary: the
     validation channel itself is silent, so "self-detection of poisoning"
     fails exactly as the D4 theorem (s42) states.
  P4 (drift tracking): drifting -- vote_pure stuck at ~0.69 both halves
     (majority never escapes), while the ema arms RE-SELECT the new good
     source after DRIFT_BLOCK (acc recovers to the good-source level,
     n_switches ~1-2): validation tracks a migrating correct source, the
     crowd rule cannot.

Output files:
  data/s64_multi_source_validation_v1.csv   (one row per (scenario,
      substrate, arm, seed) run)
  data/s64_multi_source_validation_v1.json  (params + per-cell aggregates)

Usage: python s64_multi_source_validation.py [--quick] [--sequential]
       (--sequential disables the multiprocessing Pool -- for restricted
       sandboxes where named pipes are unavailable)
"""
import os
import sys
import time
import csv
import json

os.environ.setdefault('PYTHONUNBUFFERED', '1')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

import numpy as np
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'paper_e', 'deps'))
from shallow_trap_array_simulator import gamma, tau0, gen_tau_vec, preprogram_vec
from recurrent_substrate import (
    COUPLING_NONE, COUPLING_CONTRAST_SELF,
    PW, ALPHA0, ALPHA_MIN, ALPHA_MAX, build_topology_csr,
    run_trajectory_nb)
from online_readout import OnlineRLS, running_mean_accuracy
from streaming_tasks import ROT2D_BAND0, ROT2D_BAND1, DB_K_PULSES

# ==================== Fixed parameters (S39-S63-consistent) ====================
N_UNITS = 256
CV_TAU = 0.20
TOPO_SEED = 777
AVG_DEGREE = 8
N_SEEDS = 10
FEATURE_SCALE = 10.0
BIAS = 1.0
KAPPA_RANDOM = 25.0

RLS_FORGETTING = 0.99
RLS_INIT_COV = 1.0

N_BLOCKS = 2000
THETA_STAR = 0.6          # true semantic boundary angle (content, fixed)

M_SOURCES = 5
NOISE_FLIP_P = 0.5        # bad-source flip probability (i.i.d. per block)
WEAK_FLIP_P = 0.30        # intermediate-reliability source: correct with
                          # probability 1 - WEAK_FLIP_P = 0.70 per block
                          # (added 2026-09-09, dedup P1-48)
DRIFT_BLOCK = 1000        # sources change reliability at this block

PROBE_EMA_ALPHA = 0.02    # per-source reward-rate EMA (s52 selection alpha)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's64_multi_source_validation_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's64_multi_source_validation_v1.json')

CURVE_WINDOW = 200

SUBSTRATES = ['random_graph_k25', 'parallel']
ARMS = ['worker_true', 'vote_pure', 'worker_majority',
        'emasel_pure', 'worker_emasel']
SCENARIOS = {
    'majority_good': ['good', 'good', 'good', 'noise', 'noise'],
    'minority_good': ['good', 'noise', 'noise', 'noise', 'noise'],
    'all_good':      ['good', 'good', 'good', 'good', 'good'],
    'all_bad':       ['noise', 'noise', 'noise', 'noise', 'noise'],
    'drifting':      ['drift_g2n', 'drift_n2g', 'noise', 'noise', 'noise'],
    # Intermediate-reliability cells added 2026-09-09 (dedup P1-48): the
    # original design only tested perfectly-good (p=1) and pure-noise (p=0.5)
    # sources, i.e. it never probed the difficult middle where validation-based
    # selection is genuinely challenged. 'weak' = correct with probability
    # 1 - WEAK_FLIP_P (0.70 by default).
    'all_weak':      ['weak', 'weak', 'weak', 'weak', 'weak'],
    'majority_weak': ['weak', 'weak', 'weak', 'noise', 'noise'],
    'minority_weak': ['weak', 'noise', 'noise', 'noise', 'noise'],
}


def build_substrate(topo_name):
    if topo_name == 'parallel':
        ip, idx, wt = build_topology_csr('parallel', N_UNITS)
        return ip, idx, wt, COUPLING_NONE, 0.0
    ip, idx, wt = build_topology_csr('random_graph', N_UNITS,
                                     seed=TOPO_SEED, avg_degree=AVG_DEGREE)
    return ip, idx, wt, COUPLING_CONTRAST_SELF, KAPPA_RANDOM


def gen_content(seed):
    """Fixed 2-D semantic content u and its fixed label y=g(u) (s63 single_A:
    no render -- the substrate-observable, decodable task)."""
    rng = np.random.RandomState(seed)
    u0 = rng.uniform(0.0, 1.0, N_BLOCKS)
    u1 = rng.uniform(0.0, 1.0, N_BLOCKS)
    c, s_ = float(np.cos(THETA_STAR)), float(np.sin(THETA_STAR))
    tau_b = 0.5 * (c + s_)
    labels = (c * u0 + s_ * u1 > tau_b).astype(np.int64)
    lo0, hi0 = float(ROT2D_BAND0[0]), float(ROT2D_BAND0[1])
    lo1, hi1 = float(ROT2D_BAND1[0]), float(ROT2D_BAND1[1])
    dt_seq = np.empty(N_BLOCKS * DB_K_PULSES, dtype=np.float64)
    for b in range(N_BLOCKS):
        seg = np.empty(DB_K_PULSES, dtype=np.float64)
        seg[0::2] = lo0 + u0[b] * (hi0 - lo0)
        seg[1::2] = lo1 + u1[b] * (hi1 - lo1)
        dt_seq[b * DB_K_PULSES:(b + 1) * DB_K_PULSES] = seg
    target_seq = np.repeat(labels, DB_K_PULSES).astype(np.int64)
    return dt_seq, target_seq, labels


def gen_suggestions(scenario, y_blocks, seed_idx):
    """Block-level suggestions hat(y)_i(b) for M sources + per-block
    correctness mask (source i correct on block b?). Deterministic per
    (scenario, seed_idx)."""
    rng = np.random.RandomState(seed_idx * 1000 + 17)
    src_kinds = SCENARIOS[scenario]
    M = len(src_kinds)
    S = np.empty((M, N_BLOCKS), dtype=np.int64)
    good = np.zeros((M, N_BLOCKS), dtype=bool)
    b_idx = np.arange(N_BLOCKS)
    for i, kind in enumerate(src_kinds):
        flips = rng.rand(N_BLOCKS) < NOISE_FLIP_P
        if kind == 'good':
            mask = np.zeros(N_BLOCKS, dtype=bool)
        elif kind == 'weak':
            mask = rng.rand(N_BLOCKS) < WEAK_FLIP_P
        elif kind == 'noise':
            mask = flips
        elif kind == 'drift_g2n':    # good before DRIFT_BLOCK, noise after
            mask = flips & (b_idx >= DRIFT_BLOCK)
        elif kind == 'drift_n2g':    # noise before DRIFT_BLOCK, good after
            mask = flips & (b_idx < DRIFT_BLOCK)
        else:
            raise ValueError(f"unknown source kind: {kind}")
        S[i] = np.where(mask, 1 - y_blocks, y_blocks)
        good[i] = ~mask
    return S, good


def build_features(states, T):
    obs_raw = np.exp(gamma * states) / FEATURE_SCALE
    n_fit = int(0.3 * T)
    mu = obs_raw[:n_fit].mean(axis=0)
    sd = obs_raw[:n_fit].std(axis=0)
    sd[sd < 1e-9] = 1.0
    return np.hstack([(obs_raw - mu) / sd, np.full((T, 1), BIAS)])


def acc_median_in(acc_run, b_lo, b_hi):
    lo, hi = b_lo * DB_K_PULSES, b_hi * DB_K_PULSES
    seg = acc_run[lo:hi]
    seg = seg[np.isfinite(seg)]
    return float(np.nanmedian(seg)) if seg.size else float('nan')


# ========================== Arm kernels (per arm, one block-loop) ==========================

def _worker_train_target(arm, S, y_blocks, b, r_emas):
    """Training label for the worker arms at block b. Returns (target,
    selected_source_idx or -1)."""
    if arm == 'worker_true':
        return y_blocks[b], -1
    if arm == 'worker_majority':
        return int(np.mean(S[:, b]) > 0.5), -1
    # worker_emasel: trust the source with the best reward-rate EMA
    return int(S[int(np.argmax(r_emas)), b]), int(np.argmax(r_emas))


def _ema_update(r_emas, S, y_blocks, b):
    r_emas[:] = ((1.0 - PROBE_EMA_ALPHA) * r_emas
                 + PROBE_EMA_ALPHA * (S[:, b] == y_blocks[b]).astype(np.float64))
    return r_emas


def run_worker_arm(F, S, y_blocks, arm, T):
    """worker_true / worker_majority / worker_emasel. Returns (preds_pulse,
    sel_good_frac, n_switches)."""
    d = F.shape[1]
    rls = OnlineRLS(d, 1, forgetting=RLS_FORGETTING, init_cov=RLS_INIT_COV)
    preds = np.empty(T)
    r_emas = np.zeros(S.shape[0])
    sel_frac = 0.0
    n_switches = 0
    prev_best = -1
    n_good = 0
    for t in range(T):
        preds[t] = float(rls.predict(F[t])[0])
        if (t + 1) % DB_K_PULSES == 0:
            b = (t + 1) // DB_K_PULSES - 1
            y_tgt, sel = _worker_train_target(arm, S, y_blocks, b, r_emas)
            if arm == 'worker_emasel':
                best = int(np.argmax(r_emas))
                sel = best
                if best != prev_best and prev_best != -1:
                    n_switches += 1
                prev_best = best
                n_good += int(S[best, b] == y_blocks[b])
                r_emas = _ema_update(r_emas, S, y_blocks, b)
            x_blk = F[t - DB_K_PULSES + 1:t + 1].mean(axis=0)
            rls.update(x_blk, np.array([float(y_tgt)]))
            blk = preds[t - DB_K_PULSES + 1:t + 1]
            preds[t - DB_K_PULSES + 1:t + 1] = np.mean(blk)
    if arm == 'worker_emasel':
        sel_frac = n_good / N_BLOCKS
    return preds, sel_frac, n_switches if arm == 'worker_emasel' else 0


def run_pure_arm(S, y_blocks, arm, T):
    """vote_pure / emasel_pure: no worker, block-level decisions repeated
    across the K pulses. Returns (preds_pulse, sel_good_frac, n_switches)."""
    preds = np.empty(T)
    r_emas = np.zeros(S.shape[0])
    sel_frac = 0.0
    n_switches = 0
    prev_best = -1
    n_good = 0
    for b in range(N_BLOCKS):
        if arm == 'vote_pure':
            decision = int(np.mean(S[:, b]) > 0.5)
        else:  # emasel_pure
            best = int(np.argmax(r_emas))
            if best != prev_best and prev_best != -1:
                n_switches += 1
            prev_best = best
            decision = int(S[best, b])
            n_good += int(S[best, b] == y_blocks[b])
            r_emas = _ema_update(r_emas, S, y_blocks, b)
        preds[b * DB_K_PULSES:(b + 1) * DB_K_PULSES] = decision
    if arm == 'emasel_pure':
        sel_frac = n_good / N_BLOCKS
    return preds, sel_frac, n_switches if arm == 'emasel_pure' else 0


# ========================== Single run (one trajectory, all 5 arms) ==========================

def run_single(args):
    """(substrate, scenario, seed_idx) -> list of 5 metric dicts (one per
    arm). The trajectory is computed ONCE and shared by all arms."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)

    topo_name, scenario, seed_idx = args
    t0 = time.time()

    tau = gen_tau_vec(N_UNITS, CV_TAU, tau0, seed=seed_idx)
    x0 = preprogram_vec(ALPHA0, tau)
    indptr, indices, wts, mode, kappa = build_substrate(topo_name)

    dt_seq, target_seq, y_blocks = gen_content(seed_idx)
    S, good = gen_suggestions(scenario, y_blocks, seed_idx)
    T = dt_seq.shape[0]
    target = target_seq.astype(np.float64)

    states, _, _ = run_trajectory_nb(
        x0, tau, dt_seq, PW, indptr, indices, wts, kappa,
        ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma, mode, 0)
    F = build_features(states, T)

    outs = []
    for arm in ARMS:
        if arm.startswith('worker'):
            preds, sel_frac, n_sw = run_worker_arm(F, S, y_blocks, arm, T)
        else:
            preds, sel_frac, n_sw = run_pure_arm(S, y_blocks, arm, T)
        acc_run = running_mean_accuracy(preds, target, CURVE_WINDOW)
        outs.append({
            'task': 'multi_source_validation', 'scenario': scenario,
            'substrate': topo_name, 'readout': arm, 'seed_idx': seed_idx,
            'n_units': N_UNITS, 't_total': int(T),
            'runtime_s': time.time() - t0,
            'mean_acc_all': float(np.nanmean(acc_run)),
            'acc_first': acc_median_in(acc_run, 0, 400),
            'acc_mid': acc_median_in(acc_run, 800, 1200),
            'acc_last': acc_median_in(acc_run, 1600, 2000),
            'sel_good_frac': sel_frac,
            'n_switches': n_sw,
        })
    return outs


# ========================== Aggregation ==========================

def agg_list(rs, field):
    vals = np.array([r[field] for r in rs], dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(np.nanmean(vals)) if vals.size else float('nan')


def aggregate(results):
    groups = {}
    for r in results:
        groups.setdefault((r['scenario'], r['substrate'], r['readout']),
                          []).append(r)
    agg = []
    for key, rs in sorted(groups.items()):
        scen, topo, read = key
        entry = {'scenario': scen, 'substrate': topo, 'readout': read,
                 'n_runs': len(rs)}
        for f in ['mean_acc_all', 'acc_first', 'acc_mid', 'acc_last',
                  'sel_good_frac', 'n_switches']:
            entry[f + '_mean'] = agg_list(rs, f)
        agg.append(entry)
    return agg


def print_table(agg):
    print("\n" + "=" * 128)
    print("S64 MULTI-SOURCE VALIDATION (5 sources, conflicting suggestions)")
    print("=" * 128)
    for topo in SUBSTRATES:
        print(f"\n--- {topo} ---")
        print(f"  {'scenario':<14} "
              f"{'worker_true':>12} {'vote_pure':>12} "
              f"{'worker_maj':>12} {'emasel_pure':>12} "
              f"{'worker_emasel':>12}")
        for scen in SCENARIOS:
            line = f"  {scen:<14} "
            for arm in ARMS:
                rs = [e for e in agg if e['scenario'] == scen
                      and e['substrate'] == topo and e['readout'] == arm]
                if rs:
                    line += f"{rs[0]['mean_acc_all_mean']:>12.3f}"
                else:
                    line += f"{'-':>12}"
            print(line)
    print("\n  sel_good_frac / n_switches (ema arms; majority arms: sel=-1):")
    for topo in SUBSTRATES:
        print(f"  --- {topo} ---")
        for scen in SCENARIOS:
            line = f"  {scen:<14} "
            for arm in ['emasel_pure', 'worker_emasel']:
                rs = [e for e in agg if e['scenario'] == scen
                      and e['substrate'] == topo and e['readout'] == arm]
                if rs:
                    line += (f"{arm:<14} sel={rs[0]['sel_good_frac_mean']:.3f} "
                             f"sw={rs[0]['n_switches_mean']:.1f}   ")
                else:
                    line += f"{arm:<14} -   "
            print(line)


# ========================== Main ==========================

def run_sweep(quick=False):
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S64 multi-source validation "
          f"(quick={quick})")

    tau_w = gen_tau_vec(16, CV_TAU, tau0, seed=0)
    x0_w = preprogram_vec(ALPHA0, tau_w)
    dt_w = np.full(16, 10e-6)
    ip_w, idx_w, wt_w = build_topology_csr('random_graph', 16,
                                           seed=TOPO_SEED, avg_degree=AVG_DEGREE)
    run_trajectory_nb(x0_w, tau_w, dt_w, PW, ip_w, idx_w, wt_w, KAPPA_RANDOM,
                      ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma,
                      COUPLING_CONTRAST_SELF, 0)
    print(f"[{time.strftime('%H:%M:%S')}] numba warmup done")

    n_seeds = N_SEEDS if not quick else 2
    all_args = []
    for topo in SUBSTRATES:
        for scen in SCENARIOS:
            for s in range(n_seeds):
                all_args.append((topo, scen, s))
    n_runs = len(all_args)
    n_rows = n_runs * len(ARMS)
    print(f"total trajectories: {n_runs} (substrates={len(SUBSTRATES)}, "
          f"scenarios={len(SCENARIOS)}, seeds={n_seeds}) -> "
          f"{n_rows} arm rows")

    results = []
    n_proc = min(cpu_count(), 8, max(1, n_runs))
    sequential = '--sequential' in sys.argv
    if sequential:
        done = 0
        for args in all_args:
            results.extend(run_single(args))
            done += 1
            if done % max(1, n_runs // 10) == 0 or done == n_runs:
                print(f"[{time.strftime('%H:%M:%S')}] progress "
                      f"{done}/{n_runs}", flush=True)
    else:
        with Pool(n_proc) as pool:
            done = 0
            for res in pool.imap_unordered(run_single, all_args, chunksize=1):
                results.extend(res)
                done += 1
                if done % max(1, n_runs // 10) == 0 or done == n_runs:
                    print(f"[{time.strftime('%H:%M:%S')}] progress "
                          f"{done}/{n_runs}", flush=True)

    os.makedirs(DATA_DIR, exist_ok=True)
    fieldnames = ['task', 'scenario', 'substrate', 'readout', 'seed_idx',
                  'n_units', 't_total', 'runtime_s', 'mean_acc_all',
                  'acc_first', 'acc_mid', 'acc_last', 'sel_good_frac',
                  'n_switches']
    out_csv = CSV_PATH if not quick else CSV_PATH.replace('.csv', '_quick.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(results)

    agg = aggregate(results)
    params = {
        'n_units': N_UNITS, 'cv_tau': CV_TAU, 'alpha0': ALPHA0,
        'gamma': float(gamma), 'tau0': float(tau0),
        'topo_seed': TOPO_SEED, 'avg_degree': AVG_DEGREE,
        'kappa_random': KAPPA_RANDOM, 'theta_star': THETA_STAR,
        'rls_forgetting': RLS_FORGETTING, 'rls_init_cov': RLS_INIT_COV,
        'm_sources': M_SOURCES, 'noise_flip_p': NOISE_FLIP_P,
        'weak_flip_p': WEAK_FLIP_P,
        'good_reliability_scan': {
            'note': ('intermediate-reliability cells (dedup P1-48): all_weak /'
                     ' majority_weak / minority_weak are the p=0.70 analogues '
                     'of all_good / majority_good / minority_good'),
            'reliability_grid': [1.0 - WEAK_FLIP_P, 0.5],
            'cells': ['all_weak', 'majority_weak', 'minority_weak'],
        },
        'sample_budget': {
            'n_blocks': N_BLOCKS,
            'note': ('within-run sample-complexity proxy is already recorded '
                     'per run as acc_first / acc_mid / acc_last '
                     '(block windows 0-400 / 800-1200 / 1600-2000)'),
        },
        'drift_block': DRIFT_BLOCK, 'probe_ema_alpha': PROBE_EMA_ALPHA,
        'n_blocks': N_BLOCKS, 'k_pulses': DB_K_PULSES,
        'scenarios': {k: v for k, v in SCENARIOS.items()},
        'arms': ARMS, 'substrates': SUBSTRATES,
        'n_seeds': n_seeds, 'quick': bool(quick),
        'sequential': '--sequential' in sys.argv,
    }
    out_json = JSON_PATH if not quick else JSON_PATH.replace('.json', '_quick.json')
    with open(out_json, 'w') as f:
        json.dump({'params': params, 'aggregates': agg}, f, indent=2)

    print_table(agg)
    print(f"\nCSV : {out_csv}")
    print(f"JSON: {out_json}")
    print(f"[{time.strftime('%H:%M:%S')}] DONE, total "
          f"{time.time() - t_start:.1f}s")


def main():
    quick = '--quick' in sys.argv
    run_sweep(quick=quick)


if __name__ == '__main__':
    main()
