#!/usr/bin/env python3
"""
Content-rendering separation: the same semantic content rendered by
different "speakers" carries extra information dimensions (REDEM S63)
=============================================================================
Type:           PAPER
Paper Section:  R2/R3/R5 follow-up (content-rendering separation; capacity
                sense + frozen-hypothesis memory under multi-speaker stress)
Experiment:     The user question: the SAME sentence spoken by different
                people (tone / speed / personality) is understood differently
                -- does the same content carry different information
                dimensions depending on the rendering? In this model:
                content C = a fixed 2-D semantic point stream u (labels y =
                g(u) are FIXED, speaker-independent), rendering R = a
                per-speaker affine transform A_s applied to u before it is
                encoded into the substrate's interleaved interval stream
                (band0/band1, same dual-channel encoding as gen_rotation2d).
                The physical signal the readout sees is dt = encode(A_s(u));
                the semantic label is still y = g(u). So "the same sentence"
                (same u, same y) produces physically different signals under
                different speakers.

                Questions:
                  Q1 (is rendering absorbable?): with ONE fixed speaker
                     (single_A), is the render an absorbable fixed affine
                     offset -- a single linear readout already solves it?
                  Q2 (is rendering an EXTRA dimension?): with speakers
                     alternating (mixed_AB, "conversation"), does a single
                     continuously-adapting linear readout degrade (it must
                     hold two affine relations at once), and do the model's
                     separation mechanisms recover it -- (a) basis expansion
                     (rls_quad: hold both relations in a quadratic basis)
                     and (b) frozen-hypothesis memory + reward-rate selection
                     (pop_mem, s52: store one hypothesis per speaker)?
                  Q3 (speed = hard boundary): when the speaker switches
                     faster than the meta observation scale (switch_every <
                     ~50 blocks), does recovery fail for ALL arms (the D2
                     timescale floor applied to the rendering dimension)?

                This is the substrate-decidable content-rendering test: the
                content is invariant (same u, same y), the render is the
                only thing that changes, and the paper's mechanism set
                (capacity sense -> expand, memory -> select) is tested as
                the separation device.

Arms (2-D dual-band encoding, K=20, T=40000; block-vote scoring; both
substrates):
  rls_single  : one block-level RLS on the linear features, continuous
                adaptation (s52/s48 baseline; no separation mechanism).
  rls_quad    : RLS on the quadratic features from block 0 (the basis
                expansion that a capacity sense would trigger; upper bound
                for the expand response).
  pop_mem     : s52 frozen-hypothesis memory + reward-rate selection
                (worker RLS + up to 2 frozen snapshots, functional-identity
                refresh, peak-EMA eviction) -- the "store one hypothesis per
                speaker" separation mechanism.
  meta_ema    : OBSERVATIONAL meta layer (s40/s41/s58e structure): a delta
                readout plus a SLOW reward-rate EMA (tau_2 = 1/alpha = 50
                blocks) that triggers a weight flip when r_ema < 0. The
                control question for P3: can an observer that watches ONLY
                the reward rate detect a RENDER change, when the semantic
                labels y = g(u) are UNCHANGED by the render? Prediction: NO
                -- the reward rate barely moves at a render switch (the
                boundary only shifts a few pp), so the observational meta
                has nothing to react to; it behaves like the no-separation
                baseline. The speed dimension (switch_every) is therefore
                NOT a D2 hard floor for the memory mechanism (which compares
                hypotheses block-by-block with no observation lag), and
                D2's applicability (s58e/s62) is confined to semantic-flip
                observation, not render switching.
  (reported, not a run arm) capacity probe: s53 well-conditioned PCA-50
                ridge on block-mean features vs the TRUE labels, evaluated
                inside each speaker window -- certifies whether the render
                leaves the semantic boundary linearly decodable (Q1/Q2
                decidability).

Envs:
  single_A    : speaker A (identity render) for the whole run.
  mixed_AB    : speakers A/B alternate every `switch_every` blocks
                (A,B,A,B,... revisit), switch_every in {200, 100, 25}
                (vs meta observation scale ~50 blocks: slow / near / fast).

Predictions (falsifiable):
  P1 (render absorbable): single_A -- rls_single >= 0.85 on both substrates;
       capacity probe ~= the single-speaker ridge accuracy; extra
       mechanisms add nothing (pop_mem ~ rls_single; meta_ema ~ rls_single).
  P2 (render = extra dim, and the FIX is memory, not expansion): mixed_AB
       (switch 200) -- a single linear readout must hold TWO different
       boundary relations at once (speaker A's and B's rendered semantics):
       rls_single degrades below single_A; rls_quad ~ rls_single (a single
       quadratic hyperplane is one boundary, it cannot represent the UNION
       of two half-planes either -- expansion does NOT fix a discrete
       hypothesis switch); pop_mem recovers (one frozen hypothesis per
       speaker + reward-rate selection reactivates the returning speaker
       immediately). Capacity probe on the mixed union reads LOWER than
       single_A (the render adds a dimension beyond single-linear capacity
       -> the capacity sense correctly flags insufficiency), but the
       honest fix is rule/hypothesis selection (R5), not basis expansion.
  P3 (observational meta is render-BLIND; speed is not a D2 floor for
       memory): the render does NOT change the semantic labels y=g(u), so
       the reward rate barely moves at a speaker switch -- meta_ema's slow
       EMA observes nothing to correct and behaves like rls_single at every
       switch_every (including fast sw=25: NO collapse, because its
       observation is of SEMANTIC correctness, not render identity). This
       is the honest scoping of D2 (s58e/s62): the timescale floor applies
       to observing SEMANTIC flips, not to render identity changes; the
       memory mechanism separates renders block-by-block without any
       observation lag, so render speed is not a hard boundary for it
       either. The falsifiable contrast is meta_ema ~ rls_single < pop_mem
       at all switch_every.

Output files:
  data/s63_content_rendering_v1.csv    (one row per run)
  data/s63_content_rendering_v1.json   (params + per-cell aggregates)

Usage: python s63_content_rendering.py [--quick] [--sequential]
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
from online_readout import OnlineRLS, ridge_fit, running_mean_accuracy
from streaming_tasks import ROT2D_BAND0, ROT2D_BAND1, DB_K_PULSES

# ==================== Fixed parameters (S3/S39-S62-consistent) ====================
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

# speaker B render: rotate + offset + scale (a different "voice")
SPK_B_THETA = 0.6
SPK_B_BIAS = (0.10, -0.08)
SPK_B_SCALE = 0.85

# memory arm (s52 protocol)
PROBE_EMA_ALPHA = 0.02
SNAP_EMA_MIN = 0.55
SNAP_HOLD = 20
POP_K = 3                 # 1 worker + 2 memory slots
AGREE_W = 50
# functional-identity agreement threshold of the s52 memory dedup: a fresh
# snapshot REFRESHES an existing slot when its block-vote agreement with that
# slot exceeds this value; otherwise it takes a new/evicted slot. Recorded in
# the params JSON (dedup P1-96).
AGREE_THRESH = 0.85

# meta_ema arm (s40/s41/s58e observational structure)
DELTA_ETA = 0.05
W_MAX = 20.0
META_EMA_ALPHA = 0.02     # slow observation EMA: tau_2 = 50 blocks (D2 scale)
META_V_INIT = 0.15
META_V_THRESH = 0.02
META_BETA = 0.5
META_W_EST = 40           # flip-value evaluation window (blocks)
META_WARMUP_BLOCKS = 100
FLIP_CHECK = 20

# capacity probe (s53 protocol)
CAP_N_PC = 50
CAP_THRESH = 0.55

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's63_content_rendering_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's63_content_rendering_v1.json')

CURVE_WINDOW = 200

SUBSTRATES = ['random_graph_k25', 'parallel']
ARMS = ['rls_single', 'rls_quad', 'pop_mem', 'meta_ema']
SWITCH_EVERYS = [200, 100, 25]


def build_substrate(topo_name):
    if topo_name == 'parallel':
        ip, idx, wt = build_topology_csr('parallel', N_UNITS)
        return ip, idx, wt, COUPLING_NONE, 0.0
    ip, idx, wt = build_topology_csr('random_graph', N_UNITS,
                                     seed=TOPO_SEED, avg_degree=AVG_DEGREE)
    return ip, idx, wt, COUPLING_CONTRAST_SELF, KAPPA_RANDOM


def render_speaker(u0, u1, speaker):
    """A_s(u): speaker A = identity; speaker B = rotate/offset/scale.
    Returns rendered (v0, v1) in [0,1]^2 (semantic content u unchanged;
    only the physical encoding changes)."""
    if speaker == 0:
        return u0, u1
    c, s_ = float(np.cos(SPK_B_THETA)), float(np.sin(SPK_B_THETA))
    d0 = u0 - 0.5
    d1 = u1 - 0.5
    r0 = c * d0 - s_ * d1
    r1 = s_ * d0 + c * d1
    v0 = 0.5 + SPK_B_SCALE * r0 + SPK_B_BIAS[0]
    v1 = 0.5 + SPK_B_SCALE * r1 + SPK_B_BIAS[1]
    return np.clip(v0, 0.0, 1.0), np.clip(v1, 0.0, 1.0)


def gen_content_rendered(seed, switch_every=None):
    """Fixed semantic content u (same sentence), per-block speaker render.
    Returns (dt_seq, target_seq, speaker_seq).
    target = g(u) fixed: y = 1[cos(th*)u0 + sin(th*)u1 > 0.5(cos+sin)].
    speaker_seq: 0 (A) whole run if switch_every is None, else A/B
    alternating every switch_every blocks."""
    rng = np.random.RandomState(seed)
    u0 = rng.uniform(0.0, 1.0, N_BLOCKS)
    u1 = rng.uniform(0.0, 1.0, N_BLOCKS)
    c, s_ = float(np.cos(THETA_STAR)), float(np.sin(THETA_STAR))
    tau_b = 0.5 * (c + s_)
    labels = (c * u0 + s_ * u1 > tau_b).astype(np.int64)

    if switch_every is None:
        spk = np.zeros(N_BLOCKS, dtype=np.int64)
    else:
        spk = (np.arange(N_BLOCKS) // switch_every) % 2

    v0 = np.empty(N_BLOCKS)
    v1 = np.empty(N_BLOCKS)
    for b in range(N_BLOCKS):
        v0[b], v1[b] = render_speaker(u0[b], u1[b], spk[b])

    lo0, hi0 = float(ROT2D_BAND0[0]), float(ROT2D_BAND0[1])
    lo1, hi1 = float(ROT2D_BAND1[0]), float(ROT2D_BAND1[1])
    dt_seq = np.empty(N_BLOCKS * DB_K_PULSES, dtype=np.float64)
    for b in range(N_BLOCKS):
        seg = np.empty(DB_K_PULSES, dtype=np.float64)
        seg[0::2] = lo0 + v0[b] * (hi0 - lo0)
        seg[1::2] = lo1 + v1[b] * (hi1 - lo1)
        dt_seq[b * DB_K_PULSES:(b + 1) * DB_K_PULSES] = seg
    target_seq = np.repeat(labels, DB_K_PULSES).astype(np.int64)
    return dt_seq, target_seq, spk, u0, u1


def build_features(states, T, quadratic):
    obs_raw = np.exp(gamma * states) / FEATURE_SCALE
    n_fit = int(0.3 * T)
    mu = obs_raw[:n_fit].mean(axis=0)
    sd = obs_raw[:n_fit].std(axis=0)
    sd[sd < 1e-9] = 1.0
    F = np.hstack([(obs_raw - mu) / sd, np.full((T, 1), BIAS)])
    if quadratic:
        d = F.shape[1]
        F = np.hstack([F, F[:, :d - 1] ** 2])
    return F


def capacity_pca50(Xw, yw):
    """s53 well-conditioned probe: held-out ridge on top-50 PCA components."""
    Xa = np.asarray(Xw, dtype=np.float64)
    ya = np.asarray(yw, dtype=np.float64)
    n = Xa.shape[0]
    n_tr = max(20, n // 2)
    mu = Xa[:n_tr].mean(axis=0)
    sd = Xa[:n_tr].std(axis=0) + 1e-9
    Xs = (Xa - mu) / sd
    U, S, Vt = np.linalg.svd(Xs[:n_tr], full_matrices=False)
    k = min(CAP_N_PC, Vt.shape[0])
    Xp = Xs @ Vt[:k].T
    out = ridge_fit(Xp[:n_tr], ya[:n_tr], Xp[n_tr:], ya[n_tr:],
                    ridge_lambda=1.0)
    pred = out['pred_te']
    return float(np.mean((pred > 0.5).astype(np.float64) == ya[n_tr:]))


def block_means(F, n_blocks):
    return F[:n_blocks * DB_K_PULSES].reshape(n_blocks, DB_K_PULSES,
                                               F.shape[1]).mean(axis=1)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _cap(w, w_max):
    nw = float(np.linalg.norm(w))
    if nw > w_max:
        w = w * (w_max / nw)
    return w


# ========================== Single run ==========================

def run_single(args):
    """(substrate, arm, env, switch_every, seed_idx) -> metrics dict."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)

    topo_name, arm, env, switch_every, seed_idx = args
    t0 = time.time()

    tau = gen_tau_vec(N_UNITS, CV_TAU, tau0, seed=seed_idx)
    x0 = preprogram_vec(ALPHA0, tau)
    indptr, indices, wts, mode, kappa = build_substrate(topo_name)

    se = None if env == 'single_A' else switch_every
    dt_seq, target_seq, spk, u0, u1 = gen_content_rendered(seed_idx, se)
    T = dt_seq.shape[0]
    target = target_seq.astype(np.float64)

    states, _, _ = run_trajectory_nb(
        x0, tau, dt_seq, PW, indptr, indices, wts, kappa,
        ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma, mode, 0)
    F_lin = build_features(states, T, quadratic=False)
    F_quad = build_features(states, T, quadratic=True)

    # capacity probe: PCA-50 on the whole run (Q1/Q2 decidability)
    Bm = block_means(F_lin, N_BLOCKS)
    yb = target[::DB_K_PULSES]
    probe = capacity_pca50(Bm, yb)

    F = F_quad if arm == 'rls_quad' else F_lin
    d = F.shape[1]
    preds = np.empty(T)
    n_snaps = 0

    if arm == 'meta_ema':
        # observational meta (s40/s41/s58e): delta readout + slow reward-rate
        # EMA + weight flip. The render does not change y=g(u), so r_ema
        # barely moves at a speaker switch -> no flip fires -> behaves like
        # the no-separation baseline. Reports n_flips to prove the blindness.
        rng_w = np.random.RandomState(seed_idx * 97 + 3)
        w = rng_w.uniform(-0.5, 0.5, d)
        acc_sF = np.zeros(d)
        acc_oF = np.zeros(d)
        r_ema = 0.0
        v_flip = META_V_INIT
        flip_pending = False
        blocks_since_flip = 0
        blocks_since_check = 0
        n_blocks_run = 0
        n_flips = 0
        for t in range(T):
            o = _sigmoid(float(w @ F[t]))
            preds[t] = o
            acc_sF += F[t]
            acc_oF += F[t] * o
            if (t + 1) % DB_K_PULSES == 0:
                n_blocks_run += 1
                blocks_since_check += 1
                blk = preds[t - DB_K_PULSES + 1:t + 1]
                vote = float(np.mean(blk) > 0.5)
                r = 1.0 if vote == target[t] else -1.0
                r_ema = ((1.0 - META_EMA_ALPHA) * r_ema + META_EMA_ALPHA * r)
                if flip_pending:
                    blocks_since_flip += 1
                    if blocks_since_flip >= META_W_EST:
                        v_flip = ((1.0 - META_BETA) * v_flip
                                  + META_BETA * r_ema)
                        flip_pending = False
                if (not flip_pending
                        and n_blocks_run >= META_WARMUP_BLOCKS
                        and blocks_since_check >= FLIP_CHECK):
                    blocks_since_check = 0
                    if r_ema < 0.0 and v_flip > META_V_THRESH:
                        w = -w
                        n_flips += 1
                        r_ema = 0.0
                        blocks_since_flip = 0
                        flip_pending = True
                y_derived = vote if r > 0.0 else (1.0 - vote)
                w += DELTA_ETA * (y_derived * acc_sF - acc_oF)
                w = _cap(w, W_MAX)
                acc_sF[:] = 0.0
                acc_oF[:] = 0.0
                preds[t - DB_K_PULSES + 1:t + 1] = np.mean(blk)
    elif arm == 'rls_single' or arm == 'rls_quad':
        rls = OnlineRLS(d, 1, forgetting=RLS_FORGETTING, init_cov=RLS_INIT_COV)
        for t in range(T):
            preds[t] = float(rls.predict(F[t])[0])
            if (t + 1) % DB_K_PULSES == 0:
                blk = preds[t - DB_K_PULSES + 1:t + 1]
                vote = float(np.mean(blk) > 0.5)
                r = 1.0 if vote == target[t] else -1.0
                y_derived = vote if r > 0.0 else (1.0 - vote)
                x_blk = F[t - DB_K_PULSES + 1:t + 1].mean(axis=0)
                rls.update(x_blk, np.array([y_derived]))
                preds[t - DB_K_PULSES + 1:t + 1] = np.mean(blk)
    else:  # pop_mem (s52 v4 logic, 1 worker + up to POP_K-1 frozen slots)
        worker = OnlineRLS(d, 1, forgetting=RLS_FORGETTING,
                           init_cov=RLS_INIT_COV)
        mem_w = []
        mem_ema = []
        peak_ema = []
        r_ema_w = 0.0
        hold_cnt = 0
        active_kind = 0
        active_idx = 0
        n_snaps = 0
        recent_blk = []
        for t in range(T):
            if active_kind == 0:
                preds[t] = float(worker.predict(F[t])[0])
            else:
                preds[t] = float(mem_w[active_idx] @ F[t])
            if (t + 1) % DB_K_PULSES == 0:
                seg = F[t - DB_K_PULSES + 1:t + 1]
                x_blk = seg.mean(axis=0)
                recent_blk.append(x_blk)
                if len(recent_blk) > AGREE_W:
                    recent_blk.pop(0)
                wv = float(worker.predict(x_blk)[0])
                r_w = 1.0 if (wv > 0.5) == target[t] else -1.0
                r_ema_w = ((1.0 - PROBE_EMA_ALPHA) * r_ema_w
                           + PROBE_EMA_ALPHA * r_w)
                for i in range(len(mem_w)):
                    mv = float(mem_w[i] @ x_blk)
                    r_m = 1.0 if (mv > 0.5) == target[t] else -1.0
                    mem_ema[i] = ((1.0 - PROBE_EMA_ALPHA) * mem_ema[i]
                                  + PROBE_EMA_ALPHA * r_m)
                v_w = float(wv > 0.5)
                r_w2 = 1.0 if v_w == target[t] else -1.0
                y_d = v_w if r_w2 > 0.0 else (1.0 - v_w)
                worker.update(x_blk, np.array([y_d]))
                if r_ema_w > SNAP_EMA_MIN:
                    hold_cnt += 1
                    if hold_cnt >= SNAP_HOLD:
                        snap = worker.W[:, 0].copy()
                        if recent_blk:
                            rb = np.array(recent_blk)
                            wv_agree = (snap @ rb.T > 0.5)
                            agree = [float(np.mean(
                                wv_agree == (mw @ rb.T > 0.5)))
                                for mw in mem_w]
                            best_agr = max(agree) if agree else 0.0
                            i_best = int(np.argmax(agree)) if agree else 0
                        else:
                            agree = []
                            best_agr, i_best = 0.0, 0
                        if agree and best_agr > AGREE_THRESH:
                            mem_w[i_best] = snap
                            mem_ema[i_best] = r_ema_w
                            peak_ema[i_best] = max(peak_ema[i_best], r_ema_w)
                            n_snaps += 1
                        else:
                            if len(mem_w) < POP_K - 1:
                                mem_w.append(snap)
                                mem_ema.append(r_ema_w)
                                peak_ema.append(r_ema_w)
                                n_snaps += 1
                            elif agree:
                                i_low = int(np.argmin(peak_ema))
                                mem_w[i_low] = snap
                                mem_ema[i_low] = r_ema_w
                                peak_ema[i_low] = r_ema_w
                                n_snaps += 1
                        hold_cnt = 0
                # selection: argmax of worker + memory reward-rate EMAs
                emas = [r_ema_w] + list(mem_ema)
                best = int(np.argmax(emas))
                if best == 0:
                    active_kind, active_idx = 0, 0
                    # block-level rewrite with the ACTIVE predictor's block
                    # vote (s52 protocol): keep preds in [0,1] band so the
                    # accuracy threshold is stable (raw RLS outputs are
                    # unbounded real values that would break the 0.5 cutoff)
                    preds[t - DB_K_PULSES + 1:t + 1] = wv
                else:
                    active_kind, active_idx = 1, best - 1
                    preds[t - DB_K_PULSES + 1:t + 1] = \
                        float(mem_w[active_idx] @ x_blk)

    acc_run = running_mean_accuracy(preds, target, CURVE_WINDOW)

    # per-window accuracy: first/middle/last windows of 400 blocks
    pre_win = acc_median_in(acc_run, 300, 500) if env == 'single_A' else \
        acc_median_in(acc_run, 0, 400)
    post_win = acc_median_in(acc_run, 1600, 2000)

    res = {'task': 'content_rendering', 'env': env,
           'switch_every': int(switch_every if se is not None else 0),
           'substrate': topo_name, 'readout': arm, 'seed_idx': seed_idx,
           'n_units': N_UNITS, 't_total': int(T),
           'mean_acc_all': float(np.nanmean(acc_run)),
           'acc_first': acc_median_in(acc_run, 0, 400),
           'acc_mid': acc_median_in(acc_run, 800, 1200),
           'acc_last': acc_median_in(acc_run, 1600, 2000),
           'pre_win': pre_win, 'post_win': post_win,
           'capacity_probe': float(probe),
           'n_snaps': n_snaps,
           'n_flips': n_flips if arm == 'meta_ema' else -1,
           'runtime_s': time.time() - t0}
    return res


def acc_median_in(acc_run, b_lo, b_hi):
    lo, hi = b_lo * DB_K_PULSES, b_hi * DB_K_PULSES
    seg = acc_run[lo:hi]
    seg = seg[np.isfinite(seg)]
    return float(np.nanmedian(seg)) if seg.size else float('nan')


# ========================== Aggregation ==========================

def agg_list(rs, field):
    vals = np.array([r[field] for r in rs], dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(np.nanmean(vals)) if vals.size else float('nan')


def aggregate(results):
    groups = {}
    for r in results:
        groups.setdefault((r['env'], r['switch_every'], r['substrate'],
                           r['readout']), []).append(r)
    agg = []
    for key, rs in sorted(groups.items()):
        env, sw, topo, read = key
        entry = {'env': env, 'switch_every': sw, 'substrate': topo,
                 'readout': read, 'n_runs': len(rs)}
        for f in ['mean_acc_all', 'acc_first', 'acc_mid', 'acc_last',
                  'pre_win', 'post_win', 'capacity_probe', 'n_snaps',
                  'n_flips']:
            entry[f + '_mean'] = agg_list(rs, f)
        agg.append(entry)
    return agg


def print_table(agg):
    print("\n" + "=" * 130)
    print("S63 CONTENT-RENDERING SEPARATION (same u/same y, per-speaker render)")
    print("=" * 130)
    for topo in SUBSTRATES:
        print(f"\n--- {topo} ---")
        for env in ['single_A', 'mixed_AB']:
            if env == 'single_A':
                rows = [(0, 'single speaker A')]
            else:
                rows = [(sw, f'mixed A/B sw={sw}') for sw in SWITCH_EVERYS]
            for sw, label in rows:
                print(f"  [{env} {label}]  "
                      f"{'rls_single':>18} {'rls_quad':>18} {'pop_mem':>18} "
                      f"{'meta_ema':>18}")
                line = f"    mean : "
                for arm in ARMS:
                    rs = [e for e in agg if e['env'] == env
                          and e['switch_every'] == sw
                          and e['substrate'] == topo and e['readout'] == arm]
                    if rs:
                        line += f"{rs[0]['mean_acc_all_mean']:>18.3f}"
                    else:
                        line += f"{'-':>18}"
                print(line)
                line = f"    probe: "
                for arm in ARMS:
                    rs = [e for e in agg if e['env'] == env
                          and e['switch_every'] == sw
                          and e['substrate'] == topo and e['readout'] == arm]
                    if rs:
                        line += f"{rs[0]['capacity_probe_mean']:>18.3f}"
                    else:
                        line += f"{'-':>18}"
                print(line)
                line = f"    flips: "
                for arm in ARMS:
                    rs = [e for e in agg if e['env'] == env
                          and e['switch_every'] == sw
                          and e['substrate'] == topo and e['readout'] == arm]
                    if rs and arm == 'meta_ema':
                        line += f"{rs[0]['n_flips_mean']:>18.1f}"
                    else:
                        line += f"{'-':>18}"
                print(line)


# ========================== Main ==========================

def run_sweep(quick=False):
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S63 content-rendering "
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
        for env in ['single_A', 'mixed_AB']:
            for arm in ARMS:
                if env == 'single_A':
                    ses = [0]
                else:
                    ses = SWITCH_EVERYS
                for sw in ses:
                    for s in range(n_seeds):
                        all_args.append((topo, arm, env, sw, s))
    n_runs = len(all_args)
    print(f"total runs: {n_runs} (substrates={len(SUBSTRATES)}, "
          f"arms={len(ARMS)}, envs=2, switch={SWITCH_EVERYS}, seeds={n_seeds})")

    results = []
    n_proc = min(cpu_count(), 8, max(1, n_runs))
    sequential = '--sequential' in sys.argv
    if sequential:
        done = 0
        for args in all_args:
            results.append(run_single(args))
            done += 1
            if done % max(1, n_runs // 10) == 0 or done == n_runs:
                print(f"[{time.strftime('%H:%M:%S')}] progress "
                      f"{done}/{n_runs}", flush=True)
    else:
        with Pool(n_proc) as pool:
            done = 0
            for res in pool.imap_unordered(run_single, all_args, chunksize=2):
                results.append(res)
                done += 1
                if done % max(1, n_runs // 10) == 0 or done == n_runs:
                    print(f"[{time.strftime('%H:%M:%S')}] progress "
                          f"{done}/{n_runs}", flush=True)

    os.makedirs(DATA_DIR, exist_ok=True)
    fieldnames = ['task', 'env', 'switch_every', 'substrate', 'readout',
                  'seed_idx', 'n_units', 't_total', 'runtime_s',
                  'mean_acc_all', 'acc_first', 'acc_mid', 'acc_last',
                  'pre_win', 'post_win', 'capacity_probe', 'n_snaps',
                  'n_flips']
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
        'theta_star': THETA_STAR,
        'spk_b_theta': SPK_B_THETA, 'spk_b_bias': list(SPK_B_BIAS),
        'spk_b_scale': SPK_B_SCALE,
        'rls_forgetting': RLS_FORGETTING, 'rls_init_cov': RLS_INIT_COV,
        'probe_ema_alpha': PROBE_EMA_ALPHA, 'snap_ema_min': SNAP_EMA_MIN,
        'snap_hold': SNAP_HOLD, 'pop_k': POP_K, 'agree_w': AGREE_W,
        'agree_thresh': AGREE_THRESH,
        'delta_eta': DELTA_ETA, 'w_max': W_MAX,
        'meta_ema_alpha': META_EMA_ALPHA, 'meta_v_init': META_V_INIT,
        'meta_v_thresh': META_V_THRESH, 'meta_beta': META_BETA,
        'meta_w_est': META_W_EST, 'meta_warmup_blocks': META_WARMUP_BLOCKS,
        'flip_check': FLIP_CHECK,
        'cap_n_pc': CAP_N_PC, 'cap_thresh': CAP_THRESH,
        'n_blocks': N_BLOCKS, 'k_pulses': DB_K_PULSES,
        'substrates': SUBSTRATES, 'arms': ARMS,
        'switch_every': SWITCH_EVERYS,
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
