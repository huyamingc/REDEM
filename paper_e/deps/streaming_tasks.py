#!/usr/bin/env python3
"""
Streaming task generators for the recurrent substrate (REDEM S2).
=============================================================================
Type:           CORE
Paper Section:  New-algorithm project Step S2
Experiment:     Streaming benchmark tasks for online readout evaluation

Tasks (each returns an ordered stream of (dt_seq, target_seq) arrays that
drive the substrate pulse-by-pulse):

  1. drift_binary  : two-class interval stream (blocks of K constant-interval
                     pulses) with continuous interval random walk + abrupt
                     class-interval swaps. The readout must keep classifying
                     under drift (continual learning stress).
  1b. rotation2d   : 2D input (u0, u1) encoded into alternating channel
                     sub-bands; class boundary rotates smoothly in 2D input
                     space (S48, L2 smooth-rotation regime).
  1c. nonlinear2d  : same 2D dual-band encoding with a STATIC boundary --
                     linear / circle / XOR (S50, L3 capacity probe: is a
                     nonlinear boundary linearly decodable from the
                     substrate's state?).
  1d. nonlinear3d  : 3D input (u0,u1,u2) in three channel sub-bands; static
                     plane or sphere boundary (S55, dimensional escalation of
                     the L3 capacity probe).
  1e. nonlinear4d  : 4D input (u0..u3) in four channel sub-bands; static
                     plane / sphere / shell boundary (S57, completion of the
                     L3 stress-response curve: boundary complexity, not
                     dimension, is the trigger axis).
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

# Unbuffered output
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

# 2D rotation-task defaults (s48)
ROT2D_N_BLOCKS = 2000       # total blocks
ROT2D_K_PULSES = 20         # pulses per block (10 ch0 + 10 ch1, interleaved)
ROT2D_BAND0 = (2e-6, 10e-6) # channel-0 interval band [s] (encodes u0)
ROT2D_BAND1 = (12e-6, 20e-6)  # channel-1 interval band [s] (encodes u1)
ROT2D_THETA0 = 0.6          # initial boundary angle [rad]
ROT2D_DTHETA = 0.5 * np.pi  # total boundary rotation over the run [rad]
ROT2D_ROT_AFTER = 1000      # hold theta0 for the first N blocks, then rotate


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


# ========================== Task 1b: 2D rotation (s48) ==========================

def gen_rotation2d(seed=0, n_blocks=ROT2D_N_BLOCKS, k_pulses=ROT2D_K_PULSES,
                   band0=ROT2D_BAND0, band1=ROT2D_BAND1,
                   theta0=ROT2D_THETA0, dtheta=ROT2D_DTHETA,
                   rot_after=ROT2D_ROT_AFTER, rot_span=None):
    """2D interval-encoded stream with a smoothly rotating linear boundary.

    Each block b draws a 2D input point u = (u0, u1) ~ U(0,1)^2. The class
    label is y_b = 1[cos(theta_b)*u0 + sin(theta_b)*u1 > tau_b], where
    theta_b is the boundary angle at block b: held at theta0 for the first
    `rot_after` blocks, rotating linearly from theta0 to theta0+dtheta over
    the next `rot_span` blocks, then HELD at theta0+dtheta for the rest.
    `rot_span=None` rotates through the end (old behavior). tau_b =
    0.5*(cos+sin) centers the boundary in the unit square. The 2D point is
    encoded into the substrate's 1D interval stream by INTERLEAVING two
    channel sub-bands: even pulses carry channel 0 (u0 -> band0), odd pulses
    carry channel 1 (u1 -> band1). The substrate memory (~17-pulse window,
    S1) makes every pulse's state carry BOTH recent channels, so the readout
    can extract a joint (u0, u1) decision. This is the substrate-decidable
    form of "smooth rotation" (L2): the hypothesis boundary rotates
    continuously in 2D input space instead of inverting discretely (R1/R4)
    or shifting in 1D (R2/R3).

    Returns (dt_seq (n_blocks*k,), target_seq (n_blocks*k,) int 0/1,
             theta_seq (n_blocks,), u0_seq (n_blocks,), u1_seq (n_blocks,)).
    """
    if rot_span is None:
        rot_span = max(0, n_blocks - rot_after)
    rng = np.random.RandomState(seed)
    u0 = rng.uniform(0.0, 1.0, n_blocks)
    u1 = rng.uniform(0.0, 1.0, n_blocks)
    theta = np.full(n_blocks, theta0)
    rot_end = min(n_blocks, rot_after + rot_span)
    if rot_span > 0 and rot_end > rot_after:
        n_rot = rot_end - rot_after
        theta[rot_after:rot_end] = theta0 + dtheta * np.arange(1, n_rot + 1) / n_rot
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    tau_b = 0.5 * (cos_t + sin_t)
    labels = (cos_t * u0 + sin_t * u1 > tau_b).astype(np.int64)
    lo0, hi0 = float(band0[0]), float(band0[1])
    lo1, hi1 = float(band1[0]), float(band1[1])
    dt_seq = np.empty(n_blocks * k_pulses, dtype=np.float64)
    n_even = (k_pulses + 1) // 2
    n_odd = k_pulses // 2
    for b in range(n_blocks):
        seg = np.empty(k_pulses, dtype=np.float64)
        seg[0::2] = lo0 + u0[b] * (hi0 - lo0)
        seg[1::2] = lo1 + u1[b] * (hi1 - lo1)
        dt_seq[b * k_pulses:(b + 1) * k_pulses] = seg
    target_seq = np.repeat(labels, k_pulses).astype(np.int64)
    return dt_seq, target_seq, theta, u0, u1


# ========================== Task 1c: nonlinear 2D boundary (s50) ==========================

def gen_nonlinear2d(seed=0, n_blocks=ROT2D_N_BLOCKS, k_pulses=ROT2D_K_PULSES,
                    band0=ROT2D_BAND0, band1=ROT2D_BAND1, boundary='circle',
                    radius=0.40):
    """2D interval-encoded stream with a STATIC linear or nonlinear boundary.

    Same dual-band interleaved encoding as gen_rotation2d (even pulses ch0 ->
    band0, odd pulses ch1 -> band1; substrate memory makes every pulse's state
    carry both channels). The boundary is FIXED over the whole run (no regime
    change): the L3-capacity question is whether a nonlinear decision boundary
    in 2D input space is linearly decodable from the substrate's state, i.e.
    whether the substrate's nonlinear expansion is sufficient (single linear
    readout suffices, no hypothesis generation needed) or whether a structural
    capacity wall appears (representation-insufficient, L3 trigger).

    Boundary types:
      'linear' : y = 1[cos(0.6)u0 + sin(0.6)u1 > 0.5(cos+sin)] (s48 angle).
      'circle' : y = 1[(u0-0.5)^2 + (u1-0.5)^2 < radius^2] (radius 0.4 ->
                 balanced ~0.50 area). The canonical nonlinear separator.
      'xor'    : y = 1[(u0>0.5) != (u1>0.5)] (XOR in the unit square) -- the
                 canonical linearly-inseparable function.

    Returns (dt_seq (n_blocks*k,), target_seq (n_blocks*k,) int 0/1,
             u0_seq (n_blocks,), u1_seq (n_blocks,)).
    """
    rng = np.random.RandomState(seed)
    u0 = rng.uniform(0.0, 1.0, n_blocks)
    u1 = rng.uniform(0.0, 1.0, n_blocks)
    if boundary == 'linear':
        th = 0.6
        tau_b = 0.5 * (np.cos(th) + np.sin(th))
        labels = (np.cos(th) * u0 + np.sin(th) * u1 > tau_b).astype(np.int64)
    elif boundary == 'circle':
        r = float(radius)
        labels = (((u0 - 0.5) ** 2 + (u1 - 0.5) ** 2) < r * r).astype(np.int64)
    elif boundary == 'xor':
        labels = ((u0 > 0.5) != (u1 > 0.5)).astype(np.int64)
    else:
        raise ValueError(f"unknown boundary: {boundary}")
    lo0, hi0 = float(band0[0]), float(band0[1])
    lo1, hi1 = float(band1[0]), float(band1[1])
    dt_seq = np.empty(n_blocks * k_pulses, dtype=np.float64)
    for b in range(n_blocks):
        seg = np.empty(k_pulses, dtype=np.float64)
        seg[0::2] = lo0 + u0[b] * (hi0 - lo0)
        seg[1::2] = lo1 + u1[b] * (hi1 - lo1)
        dt_seq[b * k_pulses:(b + 1) * k_pulses] = seg
    target_seq = np.repeat(labels, k_pulses).astype(np.int64)
    return dt_seq, target_seq, u0, u1


# ========================== Task 1d: 3D input (s55) ==========================

ROT3D_N_BLOCKS = 2000
ROT3D_K_PULSES = 21          # 3-channel interleave: 7 pulses per channel
ROT3D_BANDS = ((2e-6, 8e-6), (9e-6, 15e-6), (16e-6, 20e-6))  # ch0/ch1/ch2

# ========================== Task 1e: 4D input (s57) ==========================

ROT4D_N_BLOCKS = 2000
ROT4D_K_PULSES = 24          # 4-channel interleave: 6 pulses per channel
ROT4D_BANDS = ((2e-6, 6e-6), (7e-6, 11e-6), (12e-6, 16e-6), (17e-6, 20e-6))
ROT4D_SPHERE_R = 0.570       # calibrated: P(y=1)=0.502 in the unit 4-cube
ROT4D_SHELL_IN = 0.45
ROT4D_SHELL_OUT = 0.640      # calibrated: P(y=1)=0.501 with inner 0.45


def gen_nonlinear3d(seed=0, n_blocks=ROT3D_N_BLOCKS, k_pulses=ROT3D_K_PULSES,
                    bands=ROT3D_BANDS, boundary='sphere', radius=0.50):
    """3D input u=(u0,u1,u2) ~ U(0,1)^3 encoded into THREE channel sub-bands.

    Same philosophy as gen_rotation2d/gen_nonlinear2d but for 3D input: each
    pulse carries one channel (pulse index mod 3 = channel), so the substrate
    memory makes every pulse's state carry all three recent channels. The
    boundary is STATIC:
      'linear' : y = 1[cos(0.6)u0 + sin(0.6)u1 + u2 > tau] (plane in 3D,
                 control: easily linearly decodable).
      'sphere' : y = 1[||u - 0.5||^2 < radius^2] (ball in the unit cube;
                 the stressor: needs the substrate expansion to carry a
                 quadratic form in THREE directions; s55 tests whether the
                 coupled substrate's nonlinear expansion is sufficient or
                 whether representation-insufficiency finally appears).

    Returns (dt_seq (n_blocks*k,), target_seq (n_blocks*k,) int 0/1,
             u0_seq, u1_seq, u2_seq).
    """
    rng = np.random.RandomState(seed)
    u0 = rng.uniform(0.0, 1.0, n_blocks)
    u1 = rng.uniform(0.0, 1.0, n_blocks)
    u2 = rng.uniform(0.0, 1.0, n_blocks)
    if boundary == 'linear':
        th = 0.6
        tau_b = 0.5 * (np.cos(th) + np.sin(th)) + 0.5
        labels = (np.cos(th) * u0 + np.sin(th) * u1 + u2 > tau_b).astype(np.int64)
    elif boundary == 'sphere':
        r = float(radius)
        labels = (((u0 - 0.5) ** 2 + (u1 - 0.5) ** 2 + (u2 - 0.5) ** 2)
                  < r * r).astype(np.int64)
    else:
        raise ValueError(f"unknown boundary: {boundary}")
    los = [float(b[0]) for b in bands]
    his = [float(b[1]) for b in bands]
    dt_seq = np.empty(n_blocks * k_pulses, dtype=np.float64)
    for b in range(n_blocks):
        seg = np.empty(k_pulses, dtype=np.float64)
        uu = (u0[b], u1[b], u2[b])
        for i in range(k_pulses):
            ch = i % 3
            seg[i] = los[ch] + uu[ch] * (his[ch] - los[ch])
        dt_seq[b * k_pulses:(b + 1) * k_pulses] = seg
    target_seq = np.repeat(labels, k_pulses).astype(np.int64)
    return dt_seq, target_seq, u0, u1, u2


def gen_nonlinear4d(seed=0, n_blocks=ROT4D_N_BLOCKS, k_pulses=ROT4D_K_PULSES,
                    bands=ROT4D_BANDS, boundary='sphere',
                    radius=ROT4D_SPHERE_R, shell_in=ROT4D_SHELL_IN,
                    shell_out=ROT4D_SHELL_OUT):
    """4D input u=(u0..u3) ~ U(0,1)^4 encoded into FOUR channel sub-bands.

    Dimensional escalation of gen_nonlinear3d (s57): pulse index mod 4 =
    channel, so the substrate memory makes every pulse's state carry all
    four recent channels. The boundary is STATIC:
      'linear' : y = 1[cos(0.6)u0 + sin(0.6)u1 + u2 + u3 > tau] (hyperplane
                 in 4D; control -- easily linearly decodable if the
                 4-channel encoding carries all four input dims).
      'sphere' : y = 1[||u - 0.5||^2 < radius^2] (4D ball; radius 0.570
                 calibrated so P(y=1)=0.502 in the unit 4-cube -- the ball
                 pokes out of the cube faces (unlike 2D/3D), so the radius
                 is the numerically-balanced value).
      'shell'  : y = 1[shell_in^2 < ||u - 0.5||^2 < shell_out^2] (4D
                 spherical shell / annulus: TWO quadratic boundaries -- the
                 stronger-nonlinearity stressor; calibrated 0.45/0.640 ->
                 P(y=1)=0.501).

    Returns (dt_seq (n_blocks*k,), target_seq (n_blocks*k,) int 0/1,
             u0_seq, u1_seq, u2_seq, u3_seq).
    """
    rng = np.random.RandomState(seed)
    uu = [rng.uniform(0.0, 1.0, n_blocks) for _ in range(4)]
    if boundary == 'linear':
        th = 0.6
        tau_b = 0.5 * (np.cos(th) + np.sin(th)) + 1.0
        labels = (np.cos(th) * uu[0] + np.sin(th) * uu[1]
                  + uu[2] + uu[3] > tau_b).astype(np.int64)
    elif boundary == 'sphere':
        r = float(radius)
        labels = (sum((u - 0.5) ** 2 for u in uu) < r * r).astype(np.int64)
    elif boundary == 'shell':
        ri = float(shell_in)
        ro = float(shell_out)
        d2 = sum((u - 0.5) ** 2 for u in uu)
        labels = ((d2 > ri * ri) & (d2 < ro * ro)).astype(np.int64)
    else:
        raise ValueError(f"unknown boundary: {boundary}")
    los = [float(b[0]) for b in bands]
    his = [float(b[1]) for b in bands]
    dt_seq = np.empty(n_blocks * k_pulses, dtype=np.float64)
    for b in range(n_blocks):
        seg = np.empty(k_pulses, dtype=np.float64)
        for i in range(k_pulses):
            ch = i % 4
            seg[i] = los[ch] + uu[ch][b] * (his[ch] - los[ch])
        dt_seq[b * k_pulses:(b + 1) * k_pulses] = seg
    target_seq = np.repeat(labels, k_pulses).astype(np.int64)
    return dt_seq, target_seq, uu[0], uu[1], uu[2], uu[3]


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

    dt2, y2, theta, u0, u1 = gen_rotation2d(seed=0, n_blocks=200, k_pulses=20)
    # structure: interleaved sub-bands (even in band0, odd in band1),
    # theta holds then rotates, labels balanced, deterministic replay
    ev = dt2[0::2]
    od = dt2[1::2]
    ok2 = (dt2.shape[0] == 200 * 20 and y2.shape[0] == dt2.shape[0]
           and np.all(ev >= ROT2D_BAND0[0]) and np.all(ev <= ROT2D_BAND0[1])
           and np.all(od >= ROT2D_BAND1[0]) and np.all(od <= ROT2D_BAND1[1])
           and np.all(theta[:ROT2D_ROT_AFTER] == ROT2D_THETA0)
           and 0.3 < y2.mean() < 0.7)
    dt2b, y2b, _, _, _ = gen_rotation2d(seed=7, n_blocks=200)
    dt2c, y2c, _, _, _ = gen_rotation2d(seed=7, n_blocks=200)
    ok2 = ok2 and np.array_equal(dt2b, dt2c) and np.array_equal(y2b, y2c)
    results.append(("rotation2d_structure", ok2,
                    f"n_pulses={dt2.shape[0]}, P(y=1)={y2.mean():.3f}, "
                    f"theta0={theta[0]:.3f}, theta_end={theta[-1]:.3f}"))

    # nonlinear2d: shape, label balance, determinism, sub-bands
    ok3 = True
    for bd in ('linear', 'circle', 'xor'):
        dtn, yn, _, _ = gen_nonlinear2d(seed=3, n_blocks=200, boundary=bd)
        ok3 = ok3 and (dtn.shape[0] == 200 * 20 and yn.shape[0] == dtn.shape[0]
                       and 0.3 < yn.mean() < 0.7)
        ok3 = ok3 and np.all(dtn[0::2] >= ROT2D_BAND0[0]) \
            and np.all(dtn[0::2] <= ROT2D_BAND0[1]) \
            and np.all(dtn[1::2] >= ROT2D_BAND1[0]) \
            and np.all(dtn[1::2] <= ROT2D_BAND1[1])
    dn1, yn1, _, _ = gen_nonlinear2d(seed=5, boundary='circle')
    dn2, yn2, _, _ = gen_nonlinear2d(seed=5, boundary='circle')
    ok3 = ok3 and np.array_equal(dn1, dn2) and np.array_equal(yn1, yn2)
    results.append(("nonlinear2d_structure", ok3,
                    "linear/circle/xor balanced, bands ok, deterministic"))

    # nonlinear3d: shape, bands, determinism
    ok4 = True
    for bd in ('linear', 'sphere'):
        d3, y3, ua, ub, uc = gen_nonlinear3d(seed=3, n_blocks=200, boundary=bd)
        ok4 = ok4 and (d3.shape[0] == 200 * 21 and y3.shape[0] == d3.shape[0])
        ok4 = ok4 and (ua.shape[0] == 200 and ub.shape[0] == 200
                       and uc.shape[0] == 200)
    d3a, y3a, _, _, _ = gen_nonlinear3d(seed=7, boundary='sphere')
    d3b, y3b, _, _, _ = gen_nonlinear3d(seed=7, boundary='sphere')
    ok4 = ok4 and np.array_equal(d3a, d3b) and np.array_equal(y3a, y3b)
    # channel bands: pulse index mod 3 = channel
    ok4 = ok4 and np.all(d3a[0::3] >= ROT3D_BANDS[0][0]) \
        and np.all(d3a[0::3] <= ROT3D_BANDS[0][1])
    ok4 = ok4 and np.all(d3a[1::3] >= ROT3D_BANDS[1][0]) \
        and np.all(d3a[1::3] <= ROT3D_BANDS[1][1])
    ok4 = ok4 and np.all(d3a[2::3] >= ROT3D_BANDS[2][0]) \
        and np.all(d3a[2::3] <= ROT3D_BANDS[2][1])
    results.append(("nonlinear3d_structure", ok4,
                    f"n_pulses={d3.shape[0]}, sphere P(y=1)={y3a.mean():.3f}"))

    # nonlinear4d: shape, bands, label balance, determinism
    ok5 = True
    for bd in ('linear', 'sphere', 'shell'):
        d4, y4, *uv = gen_nonlinear4d(seed=3, n_blocks=200, boundary=bd)
        ok5 = ok5 and (d4.shape[0] == 200 * 24 and y4.shape[0] == d4.shape[0])
        ok5 = ok5 and all(u.shape[0] == 200 for u in uv)
        ok5 = ok5 and 0.3 < y4.mean() < 0.7
    d4a, y4a, *_ = gen_nonlinear4d(seed=7, boundary='sphere')
    d4b, y4b, *_ = gen_nonlinear4d(seed=7, boundary='sphere')
    ok5 = ok5 and np.array_equal(d4a, d4b) and np.array_equal(y4a, y4b)
    for ch_i in range(4):
        ok5 = ok5 and np.all(d4a[ch_i::4] >= ROT4D_BANDS[ch_i][0]) \
            and np.all(d4a[ch_i::4] <= ROT4D_BANDS[ch_i][1])
    results.append(("nonlinear4d_structure", ok5,
                    f"n_pulses={d4a.shape[0]}, sphere P(y=1)={y4a.mean():.3f}"))

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
