#!/usr/bin/env python3
"""
Streaming task generators for the recurrent substrate (REDEM S2).
=============================================================================
Type:           CORE
Paper Section:  New-algorithm project Step S2 (see NEW_ALGORITHM_PLAN.md)
Experiment:     Streaming benchmark tasks for online readout evaluation

Tasks (each returns an ordered stream of (dt_seq, target_seq) arrays that
drive the substrate pulse-by-pulse):

  1. drift_binary  : two-class interval stream (blocks of K constant-interval
                     pulses) with continuous interval random walk + abrupt
                     class-interval swaps. The readout must keep classifying
                     under drift (continual learning stress).
  2. narma10       : NARMA-10 benchmark (Atiya-Parlos family). u_t ~ U(0,0.5)
                     mapped to pulse intervals; target y_t is the NARMA-10
                     recurrence. Stationary, memory-demanding.
  3. mackey_glass  : chaotic Mackey-Glass series (a=0.2, b=0.1, tau=17).
                     u_t = x_t mapped to intervals; target = x_{t+1}.
                     Stationary, chaos-forecasting.
  4. context_switch: (reserved for S3) alternating NARMA-10 / Mackey-Glass
                     with a context cue pulse. Generator provided; the
                     three-factor benchmark uses it in S3.

Mapping conventions:
  * Every time step is ONE pulse; the interval dt_t encodes the input value.
  * Values are mapped monotonically into [DT_MAP_LO, DT_MAP_HI] (2us..20us),
    the fast-drive regime characterized in S1 (memory window ~17 pulses).
  * Targets are raw (unscaled); the readout script standardizes as needed.

NOTE: sequence generators are glue code (trivial scalar loops), so numba is
NOT applied here despite the CORE type; the hot physics loop lives in
recurrent_substrate.py.
"""
import os
import sys

import numpy as np

# Unbuffered output (CLAUDE.md 4.5)
os.environ.setdefault('PYTHONUNBUFFERED', '1')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

# Mapping range for scalar inputs -> pulse intervals [s]
DT_MAP_LO = 2e-6
DT_MAP_HI = 20e-6

# Drift-binary task defaults
DB_K_PULSES = 20            # pulses per block (constant interval within block)
DB_N_BLOCKS = 2000          # total blocks
DB_DT0_INIT = 10e-6         # class-0 interval at stream start
DB_DT1_INIT = 60e-6         # class-1 interval at stream start
DB_DT_MIN, DB_DT_MAX = 4e-6, 120e-6
DB_WALK_EVERY = 200         # continuous random-walk step every N blocks
DB_WALK_STD = 0.03          # relative std of walk step
DB_SWAP_EVERY = 1000        # abrupt class-interval swap every N blocks

# Mackey-Glass defaults
MG_A, MG_B, MG_TAU = 0.2, 0.1, 17
MG_N_POINTS = 21000
MG_WARMUP = 300             # map warmup points discarded


def _map_to_intervals(values, lo=DT_MAP_LO, hi=DT_MAP_HI):
    """Monotone map of values in [0,1] to intervals in [lo, hi]."""
    v = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    return lo + v * (hi - lo)


# ========================== Task 1: drifted binary stream ==========================

def gen_drift_binary(seed=0, n_blocks=DB_N_BLOCKS, k_pulses=DB_K_PULSES,
                     dt0_init=DB_DT0_INIT, dt1_init=DB_DT1_INIT,
                     dt_min=DB_DT_MIN, dt_max=DB_DT_MAX,
                     walk_every=DB_WALK_EVERY, walk_std=DB_WALK_STD,
                     swap_every=DB_SWAP_EVERY):
    """Generate a drifted two-class interval stream.

    Returns (dt_seq (n_blocks*k,), target_seq (n_blocks*k,) int 0/1,
             swap_blocks (list of block indices where a swap occurs)).
    Class labels are i.i.d. Bernoulli(0.5) per block.
    """
    rng = np.random.RandomState(seed)
    labels = rng.randint(0, 2, n_blocks)
    dt0, dt1 = float(dt0_init), float(dt1_init)
    block_dt = np.empty(n_blocks)
    swap_blocks = []
    for b in range(n_blocks):
        block_dt[b] = dt0 if labels[b] == 0 else dt1
        if (b + 1) % walk_every == 0:
            dt0 = np.clip(dt0 * (1.0 + rng.randn() * walk_std), dt_min, dt_max)
            dt1 = np.clip(dt1 * (1.0 + rng.randn() * walk_std), dt_min, dt_max)
        if (b + 1) % swap_every == 0 and (b + 1) < n_blocks:
            dt0, dt1 = dt1, dt0
            swap_blocks.append(b + 1)
    dt_seq = np.repeat(block_dt, k_pulses)
    target_seq = np.repeat(labels, k_pulses).astype(np.int64)
    return dt_seq, target_seq, swap_blocks


# ========================== Task 2: NARMA-10 ==========================

def gen_narma10(seed=0, n_points=21000, warmup=100):
    """NARMA-10 stream. Returns (dt_seq, target_seq) of length n_points.

    u_t ~ U(0, 0.5); recurrence:
      y_t = 0.3 y_{t-1} + 0.05 y_{t-1} sum_{i=1..10} y_{t-i}
            + 1.5 u_{t-10} u_{t-1} + 0.1
    First `warmup` points are discarded (map burn-in), targets start at
    index warmup of the internal buffer so every returned target is defined.
    """
    rng = np.random.RandomState(seed)
    u = rng.uniform(0.0, 0.5, n_points + warmup + 10)
    y = np.zeros_like(u)
    for t in range(10, len(u)):
        y[t] = (0.3 * y[t - 1]
                + 0.05 * y[t - 1] * np.sum(y[t - 10:t])
                + 1.5 * u[t - 10] * u[t - 1]
                + 0.1)
    dt_seq = _map_to_intervals(2.0 * u[warmup:warmup + n_points])  # u/0.5 in [0,1]
    target_seq = y[warmup:warmup + n_points]
    return dt_seq, target_seq


# ========================== Task 3: Mackey-Glass ==========================

def gen_mackey_glass(seed=0, n_points=21000, a=MG_A, b=MG_B, tau=MG_TAU,
                     warmup=MG_WARMUP, value_min=0.4, value_max=1.3):
    """Mackey-Glass chaotic series. Returns (dt_seq, target_seq).

    x_{t+1} = (1-b) x_t + a x_{t-tau} / (1 + x_{t-tau}^{10}), step dt=1.
    Values are clipped to [value_min, value_max] before the monotone
    interval map (the attractor lies well inside this box for a=0.2,b=0.1).
    Target at step t is x_{t+1}.
    """
    rng = np.random.RandomState(seed)
    x = np.empty(n_points + warmup + 1)
    x[:tau + 1] = 1.2
    for t in range(tau, n_points + warmup):
        x[t + 1] = ((1.0 - b) * x[t]
                    + a * x[t - tau] / (1.0 + x[t - tau] ** 10.0))
    x_use = x[warmup:warmup + n_points + 1]
    norm = np.clip((x_use - value_min) / (value_max - value_min), 0.0, 1.0)
    dt_seq = _map_to_intervals(norm[:-1])
    target_seq = x_use[1:]
    return dt_seq, target_seq


# Regime-switch task defaults (S5 dual-timescale metadata)
RS_EVENT_RATES = (0.12, 0.20, 0.28)   # probability of a "long interval" event
RS_BASE_RANGE = (2e-6, 8e-6)          # common short-interval range (all regimes)
RS_EVENT_RANGE = (15e-6, 18e-6)       # long-interval range (the rare event)
RS_REGIME_LEN = 1500                  # pulses per regime segment
RS_N_SEGMENTS = 6                     # total regime segments


# ========================== Task 5: regime switch (S5) ==========================

def gen_regime_switch(seed=0, event_rates=RS_EVENT_RATES,
                      base_range=RS_BASE_RANGE, event_range=RS_EVENT_RANGE,
                      seg_len=RS_REGIME_LEN, n_segments=RS_N_SEGMENTS):
    """Rare-event-rate regime stream (long-horizon statistical memory).

    The pulse-interval distribution is IDENTICAL in all regimes (uniform in
    base_range) except for the rate p_r of a rare "long interval" event
    (uniform in event_range). Single pulses are therefore almost
    indistinguishable across regimes; only the ESTIMATED EVENT RATE over a
    long window discriminates them. A ~17-pulse fast-memory window gives a
    noisy rate estimate (binomial sigma ~ 0.07-0.11), while a slow EMA with
    tau ~ 200-1000 pulses gives a reliable estimate.

    Returns (dt_seq, regime_seq) with regime_seq[t] in 0..len(rates)-1.
    """
    rng = np.random.RandomState(seed)
    n_reg = len(event_rates)
    segs = np.empty(n_segments, dtype=np.int64)
    segs[0] = rng.randint(0, n_reg)
    for s in range(1, n_segments):
        opts = [r for r in range(n_reg) if r != segs[s - 1]]
        segs[s] = int(rng.choice(opts))
    dt_parts = []
    for s in range(n_segments):
        p = event_rates[segs[s]]
        n = seg_len
        ev = rng.rand(n) < p
        dt = rng.uniform(base_range[0], base_range[1], n)
        dt[ev] = rng.uniform(event_range[0], event_range[1], int(ev.sum()))
        dt_parts.append(dt)
    dt_seq = np.concatenate(dt_parts)
    regime_seq = np.repeat(segs, seg_len).astype(np.int64)
    return dt_seq, regime_seq

# ========================== Task 6: context switch (reserved S3) ==========================

def gen_context_switch(seed=0, n_points=42000, seg_len=3000):
    """Alternating NARMA-10 / Mackey-Glass with a context cue (reserved S3).

    Returns (dt_seq, target_seq, context_seq) where context_seq is 0
    (NARMA segment) or 1 (MG segment); each segment lasts seg_len pulses
    and alternates. The first pulse of each segment carries a cue (a
    half-window interval DT_MAP_HI) so the readout can identify context.
    """
    rng = np.random.RandomState(seed)
    n_segs = n_points // seg_len
    dt_parts, y_parts, ctx_parts = [], [], []
    for s in range(n_segs):
        ctx = s % 2
        if ctx == 0:
            dt, y = gen_narma10(seed=1000 * seed + s, n_points=seg_len)
        else:
            dt, y = gen_mackey_glass(seed=1000 * seed + s, n_points=seg_len)
        dt = dt.copy()
        dt[0] = DT_MAP_HI  # context cue pulse
        dt_parts.append(dt)
        y_parts.append(y)
        ctx_parts.append(np.full(seg_len, ctx, dtype=np.int64))
    return (np.concatenate(dt_parts), np.concatenate(y_parts),
            np.concatenate(ctx_parts))


# ========================== Self test ==========================

def self_test():
    """Sanity checks for the generators: shapes, ranges, statistics."""
    results = []

    dt, y, swaps = gen_drift_binary(seed=0)
    within_block_dt = np.all(np.diff(dt)[0::DB_K_PULSES] == 0.0)   # intra-block
    within_block_y = np.all(np.diff(y)[0::DB_K_PULSES] == 0.0)
    boundary_changes = np.any(np.diff(dt)[DB_K_PULSES - 1::DB_K_PULSES] != 0.0)
    ok = (dt.shape[0] == DB_N_BLOCKS * DB_K_PULSES
          and y.shape[0] == dt.shape[0]
          and within_block_dt and within_block_y
          and boundary_changes
          and len(swaps) >= 1)
    results.append(("drift_binary_shape_blocks", ok,
                    f"n_pulses={dt.shape[0]}, swaps at blocks {swaps[:3]}"))

    dt, y = gen_narma10(seed=0, n_points=2000)
    ok = (dt.shape == (2000,) and y.shape == (2000,)
          and np.isfinite(y).all()
          and 0.0 <= y.min() <= y.max() < 2.0
          and y.std() > 0.05)
    results.append(("narma10_finite_bounded", ok,
                    f"y in [{y.min():.3f}, {y.max():.3f}], std={y.std():.3f}"))

    dt, y = gen_mackey_glass(seed=0, n_points=2000)
    ok = (dt.shape == (2000,) and y.shape == (2000,)
          and np.isfinite(y).all() and 0.4 < y.mean() < 1.3)
    results.append(("mackey_glass_finite", ok,
                    f"y in [{y.min():.3f}, {y.max():.3f}]"))

    dt, y, ctx = gen_context_switch(seed=0, n_points=6000, seg_len=1500)
    ok = (dt.shape == (6000,) and ctx.shape == (6000,)
          and set(np.unique(ctx)) <= {0, 1} and len(np.unique(ctx)) == 2)
    results.append(("context_switch_structure", ok, "4 segments alternating"))

    dt, reg = gen_regime_switch(seed=0)
    # event-rate structure check: per-regime long-interval fraction ~ p_r
    ok = (dt.shape == (RS_N_SEGMENTS * RS_REGIME_LEN,)
          and reg.shape == dt.shape
          and set(np.unique(reg)) == {0, 1, 2}
          and np.all(np.diff(reg[RS_REGIME_LEN - 1::RS_REGIME_LEN]) != 0))
    # verify the event rate per regime is close to the nominal p_r
    rate_ok = True
    for r in range(len(RS_EVENT_RATES)):
        mask = reg == r
        frac = np.mean(dt[mask] > RS_BASE_RANGE[1])
        rate_ok = rate_ok and abs(frac - RS_EVENT_RATES[r]) < 0.05
    ok = ok and rate_ok
    results.append(("regime_switch_structure", ok,
                    f"n_pulses={dt.shape[0]}, segments={RS_N_SEGMENTS}, "
                    f"rate_err_ok={rate_ok}"))

    # determinism
    dt1, y1, _ = gen_drift_binary(seed=7)
    dt2, y2, _ = gen_drift_binary(seed=7)
    results.append(("deterministic_replay",
                    np.array_equal(dt1, dt2) and np.array_equal(y1, y2),
                    "identical"))
    return results


if __name__ == "__main__":
    print("=" * 64)
    print("streaming_tasks self-test")
    print("=" * 64)
    all_ok = True
    for name, ok, detail in self_test():
        status = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{status}] {name}: {detail}")
    print("=" * 64)
    print("ALL PASS" if all_ok else "SOME CHECKS FAILED")
    raise SystemExit(0 if all_ok else 1)
