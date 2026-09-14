#!/usr/bin/env python3
"""
Hypothesis population: prune/seed tracking beats single mirror-flip on a
continuously rotating optimum. (REDEM S43; S39-S42 follow-up)
=============================================================================
Type:           PAPER
Paper Section:  S3/S39-S42 follow-up (self-evolution of the hypothesis set)
Experiment:     Mirror-impossibility corollary: with the {w, -w} mirror set,
                E[r](-w) = -E[r](w) exactly, so a single flip can always
                "fix" an anti-correlated regime -- but the set has only two
                points. When the environment demands a hypothesis in an
                intermediate state (continuously rotating optimum), the
                mirror flip can only overshoot (-w) or lag (w). A population
                of K hypotheses with reward-rate gating, WTA consolidation,
                per-specialist meta-flip, and prune/re-seed from the best
                evolves the hypothesis set itself and should track the
                rotation.

Environment (ramp_binary, local generator -- streaming_tasks.py untouched):
  class intervals ramp linearly over blocks [800, 1600):
    dt0: 10us -> 60us, dt1: 60us -> 40us.
  The optimal readout rotates continuously; at no intermediate block is it
  +/- of the pre-ramp optimum. Labels i.i.d. Bernoulli(0.5) per block, K=20
  pulses/block, T=40000. A standard full-swap drift_binary (swap at block
  1000) is included as a sanity check (no regression expected).

Arms:
  rmhl        : plain RMHL, no meta layer (no-flip baseline).
  single_flip : S41 meta_self -- one hypothesis, learned-value flip.
  pop5        : K=5 specialists; output = argmax r_ema (learned gate);
                WTA consolidation (only the active specialist learns);
                per-specialist meta-flip (S41 logic); every PRUNE_EVERY
                blocks the worst specialist (r_ema < PRUNE_THRESH) is
                pruned and re-seeded from the best + small noise.

Predictions (status checked against the committed data, 10 seeds; the refuted
entry keeps its original claim plus the measured outcome):
  P1 (ramp): pop5 win_mid (crossover window) >= single_flip win_mid -- the
             population tracks the rotation; single_flip has a mid-crossover
             dead zone (flip overshoot).
             [REFUTED by own data: see data/s43_population_v1.csv] measured
             win_mid/win_final (mean over 10 seeds): parallel pop5
             0.51725/0.50550 vs single_flip 0.51925/0.51100 vs rmhl
             0.49300/0.49700; random_graph_k25 pop5 0.50325/0.50550 vs
             single_flip 0.50375/0.50400 vs rmhl 0.49300/0.49700. Every arm
             sits at chance in the measured windows and pop5 is NOT better
             than single_flip. The 'ramp' environment as configured is
             therefore NOT discriminative for this comparison, and no claim
             about population-versus-single-hypothesis tracking can be made
             from this experiment. The pre-ramp accuracy is high (mean_acc_all
             0.685-0.714 across arms), so the chance-level windows are a
             property of the ramp phase, not of a broken run.
  P2 (std) : pop5 recovers after the full swap like single_flip (no
             regression; sanity).
             HOLDS, and the sanity environment is discriminative: pop5
             win_mid/win_final 0.90450/0.90225 (parallel) and 0.94200/0.94250
             (random_graph_k25) vs single_flip 0.90350/0.90075 and
             0.94200/0.94250 (rmhl collapses to 0.09650/0.11050 and
             0.05800/0.06400).
  P3       : prune events occur in pop5 (the set rejects hypotheses whose
             reward-rate estimate decays -- the population-level
             "inhibition").
             [PARTIALLY SUPPORTED] prune events do occur but are rare:
             n_prunes_mean over a whole run is 0.70 (parallel) and 0.50
             (random_graph_k25) in the ramp environment, and 0.10 / 0.00 in
             the std environment (PRUNE_EVERY=50, PRUNE_THRESH=-0.20).

Output files:
  data/s43_population_v1.csv    (one row per run)
  data/s43_population_v1.json   (params + per-cell aggregates)

Usage: python s43_hypothesis_population.py [--quick]
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
from online_readout import ThreeFactorReadout, running_mean_accuracy
from streaming_tasks import gen_drift_binary, DB_K_PULSES

# ========================== Fixed parameters (S3/S39-S42-consistent) ==========================
N_UNITS = 256
CV_TAU = 0.20
TOPO_SEED = 777
AVG_DEGREE = 8
N_SEEDS = 10
FEATURE_SCALE = 10.0
BIAS = 1.0
KAPPA_RANDOM = 25.0

RMHL_ETA = 0.05
RMHL_ELIG_DECAY = 0.9

SWAP_EVERY = 1000        # std env: single swap at block 1000
N_BLOCKS = 2000

# Ramp environment (local generator)
RAMP_LO, RAMP_HI = 800, 1600
RAMP_DT0 = (10e-6, 60e-6)
RAMP_DT1 = (60e-6, 40e-6)
DT_MIN, DT_MAX = 4e-6, 120e-6

# Meta-flip / gate hyperparameters (S41-consistent)
PROBE_EMA_ALPHA = 0.02
FLIP_CHECK = 20
META_V_INIT = 0.15
META_V_THRESH = 0.02
META_BETA = 0.5
META_W_EST = 40
META_WARMUP_BLOCKS = 100

# Population hyperparameters
POP_K = 5
PRUNE_EVERY = 50          # blocks between prune/seed checks
PRUNE_THRESH = -0.20      # prune a specialist whose r_ema is below this
SEED_NOISE = 0.05         # re-seed noise scale (relative to weight scale)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's43_population_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's43_population_v1.json')

CURVE_WINDOW = 200

ARMS = ['rmhl', 'single_flip', 'pop5']
ENVS = ['std', 'ramp']


# ========================== Local ramp generator ==========================

def gen_ramp_binary(seed=0, n_blocks=N_BLOCKS, k_pulses=DB_K_PULSES,
                    ramp_lo=RAMP_LO, ramp_hi=RAMP_HI,
                    dt0_range=RAMP_DT0, dt1_range=RAMP_DT1,
                    dt_min=DT_MIN, dt_max=DT_MAX):
    """Two-class interval stream with a linear class-interval ramp.

    Between blocks ramp_lo and ramp_hi, dt0 ramps from dt0_range[0] to
    dt0_range[1] and dt1 from dt1_range[0] to dt1_range[1]; before/after the
    ramp the intervals are constant. Returns (dt_seq, target_seq).
    """
    rng = np.random.RandomState(seed)
    labels = rng.randint(0, 2, n_blocks)
    block_dt = np.empty(n_blocks)
    for b in range(n_blocks):
        if b < ramp_lo:
            f = 0.0
        elif b >= ramp_hi:
            f = 1.0
        else:
            f = (b - ramp_lo) / float(ramp_hi - ramp_lo)
        dt0 = np.clip(dt0_range[0] + f * (dt0_range[1] - dt0_range[0]),
                      dt_min, dt_max)
        dt1 = np.clip(dt1_range[0] + f * (dt1_range[1] - dt1_range[0]),
                      dt_min, dt_max)
        block_dt[b] = dt0 if labels[b] == 0 else dt1
    dt_seq = np.repeat(block_dt, k_pulses)
    target_seq = np.repeat(labels, k_pulses).astype(np.int64)
    return dt_seq, target_seq


# ========================== Shared helpers ==========================

def build_substrate(topo_name):
    if topo_name == 'parallel':
        ip, idx, wt = build_topology_csr('parallel', N_UNITS)
        return ip, idx, wt, COUPLING_NONE, 0.0
    ip, idx, wt = build_topology_csr('random_graph', N_UNITS,
                                     seed=TOPO_SEED, avg_degree=AVG_DEGREE)
    return ip, idx, wt, COUPLING_CONTRAST_SELF, KAPPA_RANDOM


def acc_median_in(acc_run, b_lo, b_hi):
    """Median running accuracy over pulse range [b_lo*k, b_hi*k)."""
    lo, hi = b_lo * DB_K_PULSES, b_hi * DB_K_PULSES
    seg = acc_run[lo:hi]
    seg = seg[np.isfinite(seg)]
    return float(np.nanmedian(seg)) if seg.size else float('nan')


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# ========================== Single run ==========================

def run_single(args):
    """(topo_name, arm, env, seed_idx) -> metrics dict."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)

    topo_name, arm, env, seed_idx = args
    t0 = time.time()

    tau = gen_tau_vec(N_UNITS, CV_TAU, tau0, seed=seed_idx)
    x0 = preprogram_vec(ALPHA0, tau)
    indptr, indices, wts, mode, kappa = build_substrate(topo_name)

    if env == 'std':
        dt_seq, target_seq, _ = gen_drift_binary(
            seed=seed_idx, n_blocks=N_BLOCKS, k_pulses=DB_K_PULSES,
            swap_every=SWAP_EVERY)
    else:
        dt_seq, target_seq = gen_ramp_binary(seed=seed_idx)
    T = dt_seq.shape[0]
    target = target_seq.astype(np.float64)

    states, _, _ = run_trajectory_nb(
        x0, tau, dt_seq, PW, indptr, indices, wts, kappa,
        ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma, mode, 0)
    obs_raw = np.exp(gamma * states) / FEATURE_SCALE
    n_fit = int(0.3 * T)
    mu = obs_raw[:n_fit].mean(axis=0)
    sd = obs_raw[:n_fit].std(axis=0)
    sd[sd < 1e-9] = 1.0
    F = np.hstack([(obs_raw - mu) / sd, np.full((T, 1), BIAS)])
    d = F.shape[1]

    is_pop = (arm == 'pop5')
    is_flip = (arm in ('single_flip', 'pop5'))
    K = POP_K if is_pop else 1

    # initial weights (same ThreeFactorReadout init rule: uniform(-0.5, 0.5))
    rng_w = np.random.RandomState(seed_idx * 97 + 3)
    W = rng_w.uniform(-0.5, 0.5, (K, d))
    e = np.zeros(d)
    r_ema = np.zeros(K)
    v_flip = np.full(K, META_V_INIT)
    flip_pending = np.zeros(K, dtype=bool)
    blocks_since_flip = np.zeros(K, dtype=np.int64)

    preds = np.empty(T)
    block_o_sum = np.zeros(K)
    active = 0
    n_blocks = 0
    n_flips = 0
    n_prunes = 0
    blocks_since_check = 0

    for t in range(T):
        z = F[t] @ W.T            # (K,)
        o = _sigmoid(z)           # (K,)
        preds[t] = o[active]
        e = RMHL_ELIG_DECAY * e + F[t] * o[active]
        block_o_sum += o
        if (t + 1) % DB_K_PULSES == 0:
            n_blocks += 1
            blocks_since_check += 1
            b = (t + 1) // DB_K_PULSES - 1
            votes = (block_o_sum / DB_K_PULSES) > 0.5   # (K,) bool
            v_out = bool(votes[active])
            r = 1.0 if v_out == bool(target[t]) else -1.0
            # per-specialist reward via the vote-mirror identity:
            # r_k = r if vote_k == v_out else -r
            r_k = np.where(votes == v_out, float(r), float(-r))
            r_ema = (1.0 - PROBE_EMA_ALPHA) * r_ema + PROBE_EMA_ALPHA * r_k

            # WTA consolidation: only the active specialist learns
            W[active] += RMHL_ETA * r * e
            e[:] = 0.0

            # per-specialist meta-flip (S41 logic)
            if is_flip and b >= META_WARMUP_BLOCKS:
                for k in range(K):
                    if flip_pending[k]:
                        blocks_since_flip[k] += 1
                        if blocks_since_flip[k] >= META_W_EST:
                            v_flip[k] = ((1.0 - META_BETA) * v_flip[k]
                                         + META_BETA * r_ema[k])
                            flip_pending[k] = False
                if blocks_since_check >= FLIP_CHECK:
                    blocks_since_check = 0
                    for k in range(K):
                        if (not flip_pending[k] and r_ema[k] < 0.0
                                and v_flip[k] > META_V_THRESH):
                            W[k] = -W[k]
                            n_flips += 1
                            r_ema[k] = 0.0
                            blocks_since_flip[k] = 0
                            flip_pending[k] = True

            # prune / re-seed (population only)
            if is_pop and b >= META_WARMUP_BLOCKS and b % PRUNE_EVERY == 0:
                k_min = int(np.argmin(r_ema))
                if r_ema[k_min] < PRUNE_THRESH:
                    k_best = int(np.argmax(r_ema))
                    W[k_min] = W[k_best] + SEED_NOISE * rng_w.randn(d)
                    r_ema[k_min] = 0.0
                    v_flip[k_min] = META_V_INIT
                    flip_pending[k_min] = False
                    blocks_since_flip[k_min] = 0
                    n_prunes += 1

            # gate for the next block
            active = int(np.argmax(r_ema))
            block_o_sum[:] = 0.0

    acc_run = running_mean_accuracy(preds, target, CURVE_WINDOW)
    res = {'task': env, 'substrate': topo_name, 'readout': arm,
           'seed_idx': seed_idx, 'n_units': N_UNITS, 't_total': int(T),
           'mean_acc_all': float(np.nanmean(acc_run)),
           'n_flips': n_flips, 'n_prunes': n_prunes,
           'runtime_s': time.time() - t0}
    if env == 'std':
        res['win_mid_acc'] = acc_median_in(acc_run, 1100, 1200)   # post-swap
        res['win_final_acc'] = acc_median_in(acc_run, 1800, 2000)  # steady
    else:
        res['win_mid_acc'] = acc_median_in(acc_run, 1300, 1700)   # crossover
        res['win_final_acc'] = acc_median_in(acc_run, 1800, 2000)  # post-ramp
    return res


# ========================== Aggregation ==========================

def aggregate(results):
    groups = {}
    for r in results:
        groups.setdefault((r['substrate'], r['readout'], r['task']), []).append(r)
    agg = []
    for key, rs in sorted(groups.items()):
        topo, read, env = key
        entry = {'substrate': topo, 'readout': read, 'env': env,
                 'n_runs': len(rs)}
        fields = ['mean_acc_all', 'win_mid_acc', 'win_final_acc']
        for f in fields:
            vals = np.array([r[f] for r in rs], dtype=float)
            entry[f + '_mean'] = float(np.nanmean(vals))
            entry[f + '_std'] = float(np.nanstd(vals))
        entry['n_flips_mean'] = float(np.nanmean(
            np.array([r['n_flips'] for r in rs], dtype=float)))
        entry['n_prunes_mean'] = float(np.nanmean(
            np.array([r['n_prunes'] for r in rs], dtype=float)))
        agg.append(entry)
    return agg


def print_table(agg):
    print("\n" + "=" * 112)
    print("S43 HYPOTHESIS POPULATION (mean over seeds)")
    print("=" * 112)
    for env in ['std', 'ramp']:
        label = ('full swap @1000' if env == 'std'
                 else 'interval ramp [800,1600)')
        print(f"\n--- env {env} ({label}) ---")
        print(f"  {'substrate':<17} | {'readout':<12} | {'win_mid':>7} | "
              f"{'win_final':>9} | {'mean':>6} | {'flips':>5} | {'prunes':>6}")
        for a in [x for x in agg if x['env'] == env]:
            print(f"  {a['substrate']:<17} | {a['readout']:<12} | "
                  f"{a['win_mid_acc_mean']:>7.3f} | "
                  f"{a['win_final_acc_mean']:>9.3f} | "
                  f"{a['mean_acc_all_mean']:>6.3f} | "
                  f"{a['n_flips_mean']:>5.1f} | "
                  f"{a['n_prunes_mean']:>6.1f}")


# ========================== Main ==========================

def run_sweep(quick=False):
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S43 hypothesis population "
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
    for topo in ['parallel', 'random_graph_k25']:
        for arm in ARMS:
            for env in ENVS:
                for s in range(n_seeds):
                    all_args.append((topo, arm, env, s))
    n_runs = len(all_args)
    print(f"total runs: {n_runs} (arms={len(ARMS)}, envs={len(ENVS)}, "
          f"seeds={n_seeds})")

    results = []
    with Pool(min(cpu_count(), max(1, n_runs))) as pool:
        done = 0
        for res in pool.imap_unordered(run_single, all_args, chunksize=2):
            results.append(res)
            done += 1
            if done % max(1, n_runs // 10) == 0 or done == n_runs:
                print(f"[{time.strftime('%H:%M:%S')}] progress {done}/{n_runs}",
                      flush=True)

    os.makedirs(DATA_DIR, exist_ok=True)
    fieldnames = ['task', 'substrate', 'readout', 'seed_idx', 'n_units',
                  't_total', 'runtime_s', 'mean_acc_all', 'win_mid_acc',
                  'win_final_acc', 'n_flips', 'n_prunes']
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
        'kappa_random': KAPPA_RANDOM,
        'rmhl_eta': RMHL_ETA, 'rmhl_elig_decay': RMHL_ELIG_DECAY,
        'swap_every': SWAP_EVERY, 'n_blocks': N_BLOCKS,
        'ramp_lo': RAMP_LO, 'ramp_hi': RAMP_HI,
        'ramp_dt0': list(RAMP_DT0), 'ramp_dt1': list(RAMP_DT1),
        'probe_ema_alpha': PROBE_EMA_ALPHA, 'flip_check': FLIP_CHECK,
        'meta_v_init': META_V_INIT, 'meta_v_thresh': META_V_THRESH,
        'meta_beta': META_BETA, 'meta_w_est': META_W_EST,
        'meta_warmup_blocks': META_WARMUP_BLOCKS,
        'pop_k': POP_K, 'prune_every': PRUNE_EVERY,
        'prune_thresh': PRUNE_THRESH, 'seed_noise': SEED_NOISE,
        'n_seeds': n_seeds, 'quick': bool(quick),
    }
    out_json = JSON_PATH if not quick else JSON_PATH.replace('.json', '_quick.json')
    with open(out_json, 'w') as f:
        json.dump({'params': params, 'aggregates': agg}, f, indent=2)

    print_table(agg)
    print(f"\nCSV : {out_csv}")
    print(f"JSON: {out_json}")
    print(f"[{time.strftime('%H:%M:%S')}] DONE, total {time.time() - t_start:.1f}s")


def main():
    quick = '--quick' in sys.argv
    run_sweep(quick=quick)


if __name__ == '__main__':
    main()
