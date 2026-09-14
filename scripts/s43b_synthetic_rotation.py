#!/usr/bin/env python3
"""
Synthetic 2D rotation: does a hypothesis population track a continuously
rotating optimum that a single mirror-flip cannot? (REDEM S43b)
=============================================================================
Type:           EXPLORE
Paper Section:  S3/S39-S43 follow-up (hypothesis-set evolution)
Experiment:     ABSTRACTION TEST (synthetic features, NO substrate). The
                s43 1D result showed: in 1D interval tasks the mirror flip
                is a sufficient recovery mechanism, so a population cannot
                demonstrate an advantage (the optimal readout only ever
                points "toward the higher interval" or its negation). The
                population's value requires a geometry where the optimum
                takes intermediate states: features z in R^2 with label
                sign(w*(theta) . z), where w*(theta) rotates 0 -> pi over
                blocks [800, 1600). At the rotation midpoint the optimum is
                orthogonal to BOTH w and -w -- the mirror set provably
                fails there.

Arms (identical learner machinery to s43, features replaced by z):
  rmhl        : plain RMHL, no meta layer.
  single_flip : S41 meta_self (learned-value flip) on one hypothesis.
  pop5        : K=5 specialists, all-learn from per-specialist derived
                reward r_k (vote-mirror identity), learned gate (argmax
                reward-rate EMA), per-specialist meta-flip, prune/re-seed
                of the worst specialist from the best + gradient-informed
                step (normalized eligibility of the best) + small noise.

Predictions and their status in the committed data
(data/s43b_rotation_v1.csv, 10 seeds, WITH the W_MAX soft norm cap):
  P1: pop5 win_mid (deep-rotation window) clearly above single_flip --
      the population tracks the rotating optimum, single_flip suffers a
      mid-rotation dead zone (its only moves are keep or mirror-flip).
      [PARTIALLY SUPPORTED by own data: pop5 win_mid 0.626 +/- 0.058 is the
      highest of the three arms and above chance, but it leads single_flip
      (0.589 +/- 0.085) by only 0.037 with heavily overlapping seed
      distributions, and rmhl sits at 0.5405 +/- 0.137. The ordering is as
      predicted; the separation is not.]
  P2: win_final: all flip-capable arms recover (~0.9+) once the rotation
      ends at theta=pi (optimum = -wA); rmhl trails or fails.
      [REFUTED by own data: win_final is 0.604 +/- 0.215 (pop5),
      0.6455 +/- 0.203 (single_flip) and 0.6125 +/- 0.202 (rmhl) -- far
      below 0.9 and mutually indistinguishable at these spreads.]
  P3: prune events occur (the population rejects hypotheses whose reward
      rate decays -- population-level inhibition).
      [SUPPORTED by own data: pop5 n_prunes_mean = 2.3 per run.]
  Note on the cap: before the W_MAX = 20 soft L2 norm cap was added, the
  weights ran to |W| ~ 37 and every arm sat at chance in win_mid; the
  numbers above are the post-cap values, so any earlier "all arms at
  chance" annotation in this file describes the pre-cap run and is
  superseded.

Output files:
  data/s43b_rotation_v1.csv    (one row per run)
  data/s43b_rotation_v1.json   (params + per-cell aggregates)

Usage: python s43b_synthetic_rotation.py [--quick]
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
from online_readout import running_mean_accuracy

# ========================== Task parameters ==========================
D_FEAT = 2            # synthetic feature dimension (plus a bias column)
N_BLOCKS = 2000
K_PULSES = 20
T = N_BLOCKS * K_PULSES
BIAS = 1.0

ROT_LO, ROT_HI = 800, 1600     # rotation window (blocks); theta: 0 -> pi

# ========================== Learner parameters (S41/S43-consistent) ==========================
RMHL_ETA = 0.05
RMHL_ELIG_DECAY = 0.9

PROBE_EMA_ALPHA = 0.02
FLIP_CHECK = 20
META_V_INIT = 0.15
META_V_THRESH = 0.02
META_BETA = 0.5
META_W_EST = 40
META_WARMUP_BLOCKS = 100

POP_K = 5
PRUNE_EVERY = 50
PRUNE_THRESH = -0.20
SEED_STEP = 0.30          # gradient-informed seed step (units of w scale)
SEED_NOISE = 0.05
# Soft L2 weight-norm cap, same convention as s45/s46/s47/s48 (dedup P1-91):
# without it |W| grew to ~37 and all three arms sat at chance, which made the
# negative result uninterpretable (bias-runaway regime rather than a property
# of the population).
W_MAX = 20.0

N_SEEDS = 10

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's43b_rotation_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's43b_rotation_v1.json')

CURVE_WINDOW = 200

ARMS = ['rmhl', 'single_flip', 'pop5']


def gen_rotation_stream(seed, noise=0.8):
    """Synthetic stream: class-conditional Gaussians with rotating centroids.

    Per pulse: y ~ Bernoulli(0.5); z = (2y-1)*c(theta) + noise*N(0,I2), where
    c(theta) = [cos, sin] rotates 0 -> pi over blocks [ROT_LO, ROT_HI).
    Class-1 centroid is +c(theta), class-0 is -c(theta) -- the same
    "class-1 features are larger" structure the substrate provides, so RMHL
    eligibility points at the (rotating) class-1 centroid. The optimal
    readout direction is c(theta); at the rotation midpoint it is
    orthogonal to both the pre-rotation optimum and its negation.
    Returns (Z (T,2), target (T,) int 0/1).
    """
    rng = np.random.RandomState(seed)
    Z = np.empty((T, D_FEAT))
    target = np.empty(T, dtype=np.int64)
    for b in range(N_BLOCKS):
        if b < ROT_LO:
            th = 0.0
        elif b >= ROT_HI:
            th = np.pi
        else:
            th = np.pi * (b - ROT_LO) / float(ROT_HI - ROT_LO)
        c = np.array([np.cos(th), np.sin(th)])
        lo, hi = b * K_PULSES, (b + 1) * K_PULSES
        y = rng.randint(0, 2, hi - lo)
        target[lo:hi] = y
        Z[lo:hi] = (2 * y - 1)[:, None] * c[None, :] \
            + noise * rng.randn(hi - lo, D_FEAT)
    return Z, target


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def acc_median_in(acc_run, b_lo, b_hi):
    lo, hi = b_lo * K_PULSES, b_hi * K_PULSES
    seg = acc_run[lo:hi]
    seg = seg[np.isfinite(seg)]
    return float(np.nanmedian(seg)) if seg.size else float('nan')


# ========================== Single run ==========================

def run_single(args):
    """(arm, seed_idx) -> metrics dict."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)

    arm, seed_idx = args
    t0 = time.time()

    Z, target = gen_rotation_stream(seed_idx)
    F = np.hstack([Z, np.full((T, 1), BIAS)])   # (T, 3), bias column
    d = F.shape[1]
    target = target.astype(np.float64)

    is_pop = (arm == 'pop5')
    is_flip = (arm in ('single_flip', 'pop5'))
    K = POP_K if is_pop else 1

    rng_w = np.random.RandomState(seed_idx * 97 + 3)
    W = rng_w.uniform(-0.5, 0.5, (K, d))
    e = np.zeros((K, d))
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
        Ft = F[t]
        o = _sigmoid(Ft @ W.T)          # (K,)
        preds[t] = o[active]
        e = RMHL_ELIG_DECAY * e + Ft[None, :] * o[:, None]   # (K, d)
        block_o_sum += o
        if (t + 1) % K_PULSES == 0:
            n_blocks += 1
            blocks_since_check += 1
            b = (t + 1) // K_PULSES - 1
            votes = (block_o_sum / K_PULSES) > 0.5
            v_out = bool(votes[active])
            r = 1.0 if v_out == bool(target[t]) else -1.0
            r_k = np.where(votes == v_out, float(r), float(-r))
            r_ema = (1.0 - PROBE_EMA_ALPHA) * r_ema + PROBE_EMA_ALPHA * r_k

            # ALL specialists learn from their own derived reward (no WTA)
            W += RMHL_ETA * r_k[:, None] * e
            # soft L2 norm cap per specialist (dedup P1-91)
            for k in range(K):
                nw_k = float(np.linalg.norm(W[k]))
                if nw_k > W_MAX:
                    W[k] *= (W_MAX / nw_k)

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

            # prune / re-seed (population only; gradient-informed seed)
            if is_pop and b >= META_WARMUP_BLOCKS and b % PRUNE_EVERY == 0:
                k_min = int(np.argmin(r_ema))
                if r_ema[k_min] < PRUNE_THRESH:
                    k_best = int(np.argmax(r_ema))
                    eg = e[k_best]
                    nrm = float(np.linalg.norm(eg))
                    if nrm > 1e-12:
                        seed_dir = eg / nrm
                    else:
                        seed_dir = rng_w.randn(d)
                        seed_dir /= float(np.linalg.norm(seed_dir))
                    W[k_min] = (W[k_best] + SEED_STEP * seed_dir
                                + SEED_NOISE * rng_w.randn(d))
                    r_ema[k_min] = 0.0
                    v_flip[k_min] = META_V_INIT
                    flip_pending[k_min] = False
                    blocks_since_flip[k_min] = 0
                    n_prunes += 1

            e[:] = 0.0
            active = int(np.argmax(r_ema))
            block_o_sum[:] = 0.0

    acc_run = running_mean_accuracy(preds, target, CURVE_WINDOW)
    res = {'task': 'rotation2d', 'readout': arm, 'seed_idx': seed_idx,
           'n_units': d, 't_total': int(T),
           'mean_acc_all': float(np.nanmean(acc_run)),
           'win_mid_acc': acc_median_in(acc_run, 1200, 1600),   # deep rotation
           'win_final_acc': acc_median_in(acc_run, 1800, 2000),  # post-rotation
           'n_flips': n_flips, 'n_prunes': n_prunes,
           'w_norm_max_end': float(np.max(np.linalg.norm(W, axis=1))),
           'runtime_s': time.time() - t0}
    return res


# ========================== Aggregation ==========================

def aggregate(results):
    groups = {}
    for r in results:
        groups.setdefault(r['readout'], []).append(r)
    agg = []
    for read, rs in sorted(groups.items()):
        entry = {'readout': read, 'n_runs': len(rs)}
        for f in ['mean_acc_all', 'win_mid_acc', 'win_final_acc']:
            vals = np.array([r[f] for r in rs], dtype=float)
            entry[f + '_mean'] = float(np.nanmean(vals))
            entry[f + '_std'] = float(np.nanstd(vals))
        entry['n_flips_mean'] = float(np.nanmean(
            np.array([r['n_flips'] for r in rs], dtype=float)))
        entry['n_prunes_mean'] = float(np.nanmean(
            np.array([r['n_prunes'] for r in rs], dtype=float)))
        entry['w_norm_max_end_mean'] = float(np.nanmean(
            np.array([r['w_norm_max_end'] for r in rs], dtype=float)))
        agg.append(entry)
    return agg


def print_table(agg):
    print("\n" + "=" * 96)
    print("S43b SYNTHETIC 2D ROTATION (mean over seeds)")
    print("=" * 96)
    print(f"  {'readout':<12} | {'win_mid(rot)':>12} | "
          f"{'win_final':>9} | {'mean':>6} | {'flips':>5} | {'prunes':>6}")
    for a in agg:
        print(f"  {a['readout']:<12} | {a['win_mid_acc_mean']:>12.3f} | "
              f"{a['win_final_acc_mean']:>9.3f} | "
              f"{a['mean_acc_all_mean']:>6.3f} | "
              f"{a['n_flips_mean']:>5.1f} | "
              f"{a['n_prunes_mean']:>6.1f}")


# ========================== Main ==========================

def run_sweep(quick=False):
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S43b synthetic 2D rotation "
          f"(quick={quick})")

    n_seeds = N_SEEDS if not quick else 2
    all_args = [(arm, s) for arm in ARMS for s in range(n_seeds)]
    n_runs = len(all_args)
    print(f"total runs: {n_runs} (arms={len(ARMS)}, seeds={n_seeds})")

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
    fieldnames = ['task', 'readout', 'seed_idx', 'n_units', 't_total',
                  'runtime_s', 'mean_acc_all', 'win_mid_acc',
                  'win_final_acc', 'n_flips', 'n_prunes']
    out_csv = CSV_PATH if not quick else CSV_PATH.replace('.csv', '_quick.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(results)

    agg = aggregate(results)
    params = {
        'd_feat': D_FEAT, 'n_blocks': N_BLOCKS, 'k_pulses': K_PULSES,
        'bias': BIAS, 'rot_lo': ROT_LO, 'rot_hi': ROT_HI,
        'rmhl_eta': RMHL_ETA, 'rmhl_elig_decay': RMHL_ELIG_DECAY,
        'probe_ema_alpha': PROBE_EMA_ALPHA, 'flip_check': FLIP_CHECK,
        'meta_v_init': META_V_INIT, 'meta_v_thresh': META_V_THRESH,
        'meta_beta': META_BETA, 'meta_w_est': META_W_EST,
        'meta_warmup_blocks': META_WARMUP_BLOCKS,
        'pop_k': POP_K, 'prune_every': PRUNE_EVERY,
        'prune_thresh': PRUNE_THRESH, 'seed_step': SEED_STEP,
        'seed_noise': SEED_NOISE, 'w_max': W_MAX,
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
