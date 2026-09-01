#!/usr/bin/env python3
"""
Baseline showdown (REDEM S9): online systems vs batch learners.
=============================================================================
Type:           ML
Paper Section:  New-algorithm project Step S9
Experiment:     Honest ecosystem positioning: our integrated system (REDEM
                core: coupled substrate + online RLS) vs an ESN with the
                same online readout vs batch-trained GRU / tiny transformer
                (torch CPU), on drift_binary (continual learning) and
                mackey_glass (chaos forecasting).

Framing (no "replacement" narrative): REDEM and the ESN are ONLINE learners
(weights adapt through the stream, local rules, no BPTT); the GRU and tiny
transformer are BATCH learners (trained once on the first 30% of the
stream, frozen afterwards) -- they represent the classical offline ceiling
and cannot adapt to drift. The comparison documents the niche.

Systems:
  redem  : random_graph kappa=25 substrate + RLS dense readout (S2 core)
  esn    : ESN-256-hetero (fair_esn_comparison class) + RLS dense readout
           on reservoir states (same online protocol)
  gru    : torch GRU (hidden 64), trained offline on first 30%
  trans  : tiny causal transformer (d=64, 2 layers, context 256), offline

Metrics: drift -> pre/post-swap + mean accuracy; MG -> NMSE on last 30%.

Output files:
  data/s9_baseline_showdown_v1.csv    (one row per run)
  data/s9_baseline_showdown_v1.json   (params + aggregates)

Usage: python baseline_showdown.py [--quick]
"""
import os
import sys
import time
import csv
import json
import warnings
warnings.filterwarnings('ignore')

os.environ.setdefault('PYTHONUNBUFFERED', '1')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

import numpy as np
from multiprocessing import Pool, cpu_count

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shallow_trap_array_simulator import gamma, tau0, gen_tau_vec, preprogram_vec
from recurrent_substrate import (
    COUPLING_CONTRAST_SELF,
    PW, ALPHA0, ALPHA_MIN, ALPHA_MAX, build_topology_csr,
    run_trajectory_nb)
from online_readout import OnlineRLS, running_mean_accuracy
from streaming_tasks import gen_drift_binary, gen_mackey_glass, DB_K_PULSES
from fair_esn_comparison import ESN

# ========================== Fixed parameters ==========================
N_UNITS = 256
CV_TAU = 0.20
TOPO_SEED = 777
AVG_DEGREE = 8
N_SEEDS = 10
N_SEEDS_TRANSFORMER = 10
FEATURE_SCALE = 10.0
BIAS = 1.0
KAPPA_RANDOM = 25.0

RLS_FORGETTING = 0.999
RLS_INIT_COV = 1.0
RLS_TRACE_CAP = 1e8
RLS_REG = 1e-4

# torch model specs
GRU_HIDDEN = 64
TRANS_D_MODEL = 64
TRANS_LAYERS = 2
TRANS_HEADS = 2
TRANS_CONTEXT = 256
TRAIN_EPOCHS = 20
BATCH = 128
LR = 1e-3
TRAIN_FRAC = 0.30

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's9_baseline_showdown_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's9_baseline_showdown_v1.json')


# ========================== torch models ==========================

class GRUModel(nn.Module):
    def __init__(self, hidden=GRU_HIDDEN, out_dim=1):
        super().__init__()
        self.gru = nn.GRU(1, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, out_dim)

    def forward(self, x, h=None):
        out, h = self.gru(x, h)
        return self.fc(out), h


class TinyTransformer(nn.Module):
    def __init__(self, d_model=TRANS_D_MODEL, n_heads=TRANS_HEADS,
                 n_layers=TRANS_LAYERS, context=TRANS_CONTEXT, out_dim=1):
        super().__init__()
        self.embed = nn.Linear(1, d_model)
        self.pos = nn.Parameter(torch.randn(1, context, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=2 * d_model,
            batch_first=True, dropout=0.0, activation='relu')
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.fc = nn.Linear(d_model, out_dim)

    def forward(self, x):
        # x: (B, L, 1); causal mask: position t attends only to tokens <= t
        B, L, _ = x.shape
        e = self.embed(x) + self.pos[:, :L]
        h = self.enc(e, mask=causal_mask(L))
        return self.fc(h)


def causal_mask(size):
    return torch.triu(torch.ones(size, size), diagonal=1).bool()


def train_offline(model, u_tr, y_tr, is_class, seed_idx):
    """Train on the first 30% of the stream (torch, CPU). Returns model.
    Model initialization is seeded in run_single BEFORE construction
    (seed_idx*101+17, the repo rule) so each trial gets its own
    reproducible init; batch composition here is deterministic via
    np.random.RandomState(epoch)."""
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n = u_tr.shape[0]
    if is_class:
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.MSELoss()
    for epoch in range(TRAIN_EPOCHS):
        # random-length segments for GRU (truncated BPTT); fixed windows for
        # the transformer
        idx = np.random.RandomState(epoch).choice(n - 256, BATCH)
        segs = np.stack([u_tr[i:i + 256] for i in idx])[:, :, None]
        tgt = y_tr[idx + 255][:, None] if is_class else np.stack(
            [y_tr[i:i + 256] for i in idx])[:, :, None]
        xs = torch.from_numpy(segs).float()
        opt.zero_grad()
        if isinstance(model, GRUModel):
            out, _ = model(xs)
            if is_class:
                out_t = out[:, -1, :]
                loss = loss_fn(out_t, torch.from_numpy(tgt[:, 0]).long())
            else:
                loss = loss_fn(out, torch.from_numpy(tgt).float())
        else:
            out = model(xs)
            if is_class:
                loss = loss_fn(out[:, -1, :], torch.from_numpy(tgt[:, 0]).long())
            else:
                loss = loss_fn(out, torch.from_numpy(tgt).float())
        loss.backward()
        opt.step()
    return model


# ========================== single run ==========================

def run_single(args):
    """(task, system, seed_idx) -> metrics dict."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
    task, system, seed_idx = args
    t0 = time.time()

    if task == 'drift_binary':
        dt_seq, target_seq, swap_blocks = gen_drift_binary(seed=seed_idx)
        is_class = True
        n_classes = 2
    else:
        dt_seq, target_seq = gen_mackey_glass(seed=seed_idx)
        is_class = False
        n_classes = 1
    T = dt_seq.shape[0]
    target = target_seq.astype(np.float64)
    u_norm = (dt_seq - dt_seq.min()) / max(dt_seq.max() - dt_seq.min(), 1e-12)

    res = {'task': task, 'system': system, 'seed_idx': seed_idx,
           'n_units': N_UNITS, 't_total': int(T)}

    if system == 'redem':
        tau = gen_tau_vec(N_UNITS, CV_TAU, tau0, seed=seed_idx)
        x0 = preprogram_vec(ALPHA0, tau)
        ip, idx, wt = build_topology_csr('random_graph', N_UNITS,
                                         seed=TOPO_SEED, avg_degree=AVG_DEGREE)
        states, _, _ = run_trajectory_nb(x0, tau, dt_seq, PW, ip, idx, wt,
                                         KAPPA_RANDOM, ALPHA0, ALPHA_MIN,
                                         ALPHA_MAX, gamma,
                                         COUPLING_CONTRAST_SELF, 0)
        obs = np.exp(gamma * states) / FEATURE_SCALE
        n_fit = int(0.3 * T)
        mu = obs[:n_fit].mean(axis=0)
        sd = obs[:n_fit].std(axis=0)
        sd[sd < 1e-9] = 1.0
        F = np.hstack([(obs - mu) / sd, np.full((T, 1), BIAS)])
        rls = OnlineRLS(F.shape[1], n_classes,
                        forgetting=RLS_FORGETTING, init_cov=RLS_INIT_COV,
                        trace_cap=RLS_TRACE_CAP, reg=RLS_REG)
        if is_class:
            Y = np.zeros((T, n_classes))
            Y[np.arange(T), target.astype(int)] = 1.0
            _, p = rls.fit_stream(F, Y, n_warmup=200)
            pred = p.argmax(axis=1).astype(np.float64)
        else:
            _, p = rls.fit_stream(F, target[:, None], n_warmup=200)
            pred = p[:, 0]
    elif system == 'esn':
        esn = ESN(n_input=1, n_reservoir=N_UNITS, spectral_radius=0.9,
                  input_scaling=0.5, leaking_rate=0.2, hetero_lr=True,
                  cv_lr=CV_TAU, seed=seed_idx + 999)
        states = esn.process(u_norm[:, None])
        # z-score with first-30% stats, same convention as the REDEM arm
        # (the two online readouts then see comparable feature scales)
        n_fit = int(0.3 * T)
        mu = states[:n_fit].mean(axis=0)
        sd = states[:n_fit].std(axis=0)
        sd[sd < 1e-9] = 1.0
        F = np.hstack([(states - mu) / sd, np.full((T, 1), BIAS)])
        rls = OnlineRLS(F.shape[1], 1, forgetting=RLS_FORGETTING,
                        init_cov=RLS_INIT_COV, trace_cap=RLS_TRACE_CAP,
                        reg=RLS_REG)
        _, p = rls.fit_stream(F, target[:, None], n_warmup=200)
        pred = p[:, 0]
    elif system in ('gru', 'trans'):
        n_train = int(TRAIN_FRAC * T)
        u_tr = u_norm[:n_train]
        y_tr = target[:n_train]
        # Seed BEFORE construction so each trial's weight init is its own
        # per-seed draw (fairness protocol for the frozen deep baselines)
        torch.manual_seed(seed_idx * 101 + 17)
        if system == 'gru':
            model = GRUModel(out_dim=n_classes)
        else:
            model = TinyTransformer(out_dim=n_classes)
        model.train()
        train_offline(model, u_tr, y_tr, is_class, seed_idx)
        model.eval()
        # stateful eval
        pred = np.full(T, np.nan)
        with torch.no_grad():
            if system == 'gru':
                h = None
                for i in range(0, T, 512):
                    x = torch.from_numpy(u_norm[i:i + 512][None, :, None]).float()
                    out, h = model(x, h)
                    # per-position predictions (the GRU output at step t is
                    # the prediction for step t)
                    pred[i:i + out.shape[1]] = out[0].argmax(dim=1).numpy() \
                        if is_class else out[0, :, 0].numpy()
            else:
                # non-overlapping context windows (batched)
                for i in range(0, T, TRANS_CONTEXT):
                    n_actual = min(TRANS_CONTEXT, T - i)
                    seg = u_norm[i:i + n_actual]
                    pad = TRANS_CONTEXT - seg.shape[0]
                    if pad > 0:
                        seg = np.concatenate([seg, np.full(pad, seg[-1])])
                    x = torch.from_numpy(seg[None, :, None]).float()
                    out = model(x)
                    pred[i:i + n_actual] = out[0, :n_actual, :].argmax(
                        dim=1).numpy() if is_class else out[0, :n_actual, 0].numpy()
        pred[:n_train] = np.nan   # training segment excluded from eval
    else:
        raise ValueError(system)

    if task == 'drift_binary':
        acc_run = running_mean_accuracy(pred, target, 200)
        swap_pulse = int(swap_blocks[0] * DB_K_PULSES)
        pre = float(np.nanmedian(acc_run[swap_pulse - 2000:swap_pulse]))
        post = float(np.nanmedian(acc_run[swap_pulse + 4000:swap_pulse + 6000]))
        mean = float(np.nanmean(acc_run))
        res.update({'pre_swap_acc': pre, 'post_swap_acc': post,
                    'mean_acc': mean})
    else:
        n_eval = int(0.3 * T)
        var_eval = float(target[-n_eval:].var())
        mse = float(np.nanmean((pred[-n_eval:] - target[-n_eval:]) ** 2))
        res['nmse_final30'] = mse / var_eval if var_eval > 0 else np.nan
    res['runtime_s'] = time.time() - t0
    return res


def aggregate(results):
    groups = {}
    for r in results:
        groups.setdefault((r['task'], r['system']), []).append(r)
    agg = []
    for key, rs in sorted(groups.items()):
        task, sys_name = key
        entry = {'task': task, 'system': sys_name, 'n_runs': len(rs)}
        fields = (['pre_swap_acc', 'post_swap_acc', 'mean_acc']
                  if task == 'drift_binary' else ['nmse_final30'])
        for f in fields:
            v = np.array([r[f] for r in rs], dtype=float)
            entry[f + '_mean'] = float(np.nanmean(v))
            entry[f + '_std'] = float(np.nanstd(v))
        agg.append(entry)
    return agg


def print_table(agg):
    print("\n" + "=" * 100)
    print("S9 RESULTS (mean over seeds)")
    print("=" * 100)
    for task in ['drift_binary', 'mackey_glass']:
        rows = [a for a in agg if a['task'] == task]
        print(f"\n--- {task} ---")
        if task == 'drift_binary':
            print(f"  {'system':<8} | {'pre_swap':>8} | {'post_swap':>9} | "
                  f"{'mean_acc':>8}")
            for a in sorted(rows, key=lambda x: -x['mean_acc_mean']):
                print(f"  {a['system']:<8} | {a['pre_swap_acc_mean']:>8.3f} | "
                      f"{a['post_swap_acc_mean']:>9.3f} | "
                      f"{a['mean_acc_mean']:>8.3f}")
        else:
            print(f"  {'system':<8} | {'NMSE_final30':>12}")
            for a in sorted(rows, key=lambda x: x['nmse_final30_mean']):
                print(f"  {a['system']:<8} | {a['nmse_final30_mean']:>12.4f}")


def main():
    quick = '--quick' in sys.argv
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S9 baseline showdown (quick={quick})")
    torch.set_num_threads(max(1, cpu_count() // 2))

    tau_w = gen_tau_vec(16, CV_TAU, tau0, seed=0)
    x0_w = preprogram_vec(ALPHA0, tau_w)
    dt_w = np.full(16, 10e-6)
    ip_w, idx_w, wt_w = build_topology_csr('random_graph', 16, seed=TOPO_SEED)
    run_trajectory_nb(x0_w, tau_w, dt_w, PW, ip_w, idx_w, wt_w, KAPPA_RANDOM,
                      ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma,
                      COUPLING_CONTRAST_SELF, 0)
    print(f"[{time.strftime('%H:%M:%S')}] numba warmup done")

    tasks = ['drift_binary', 'mackey_glass']
    if quick:
        tasks = ['drift_binary']
        n_seeds = 2
        n_seeds_t = 2
    else:
        n_seeds = N_SEEDS
        n_seeds_t = N_SEEDS_TRANSFORMER
    all_args = []
    for task in tasks:
        for s in range(n_seeds):
            for sys_name in ['redem', 'esn', 'gru']:
                all_args.append((task, sys_name, s))
        for s in range(n_seeds_t):
            all_args.append((task, 'trans', s))
    n_runs = len(all_args)
    print(f"total runs: {n_runs}")

    results = []
    with Pool(min(cpu_count(), max(1, n_runs))) as pool:
        done = 0
        for res in pool.imap_unordered(run_single, all_args, chunksize=1):
            results.append(res)
            done += 1
            if done % max(1, n_runs // 10) == 0 or done == n_runs:
                print(f"[{time.strftime('%H:%M:%S')}] progress {done}/{n_runs}",
                      flush=True)

    os.makedirs(DATA_DIR, exist_ok=True)
    fieldnames = ['task', 'system', 'seed_idx', 'n_units', 't_total',
                  'pre_swap_acc', 'post_swap_acc', 'mean_acc', 'nmse_final30',
                  'runtime_s']
    out_csv = CSV_PATH if not quick else CSV_PATH.replace('.csv', '_quick.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(results)

    agg = aggregate(results)
    params = {
        'n_units': N_UNITS, 'cv_tau': CV_TAU, 'kappa_random': KAPPA_RANDOM,
        'rls_forgetting': RLS_FORGETTING,
        'gru_hidden': GRU_HIDDEN,
        'trans_d_model': TRANS_D_MODEL, 'trans_layers': TRANS_LAYERS,
        'trans_heads': TRANS_HEADS, 'trans_context': TRANS_CONTEXT,
        'train_epochs': TRAIN_EPOCHS, 'batch': BATCH, 'lr': LR,
        'train_frac': TRAIN_FRAC,
        'n_seeds': n_seeds, 'n_seeds_transformer': n_seeds_t,
        'quick': bool(quick),
    }
    out_json = JSON_PATH if not quick else JSON_PATH.replace('.json', '_quick.json')
    with open(out_json, 'w') as f:
        json.dump({'params': params, 'aggregates': agg}, f, indent=2)

    print_table(agg)
    print(f"\nCSV : {out_csv}")
    print(f"JSON: {out_json}")
    print(f"[{time.strftime('%H:%M:%S')}] DONE, total {time.time() - t_start:.1f}s")


if __name__ == '__main__':
    main()
