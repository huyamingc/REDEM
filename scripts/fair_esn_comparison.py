#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fair ESN comparison: addresses review finding P0#3 / P1#21.
The original baseline_comparison.py used ESN with only 40 units and
uniform leaking rate, vs physical system with 256 distributed-τ devices.

Type:           ML
Paper §:        §4.3 (ESN baseline comparison)
Experiment:     Fair ESN vs Physical system comparison
Output files:   data/fair_esn_results.csv
                data/fair_esn_results.json

This script tests:
  1. ESN-40-uniform  (original unfair config: 40 units, LR=0.2)
  2. ESN-256-uniform (matched unit count: 256 units, LR=0.2)
  3. ESN-256-hetero  (matched units + heterogeneous LR matching τ distribution)
  4. ESN-256-hyperopt (matched units + grid search over SR and LR)

On both binary (20μs vs 200μs) and non-monotonic AB-BA tasks.
"""
import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings, time, sys, csv, json
from pathlib import Path
warnings.filterwarnings('ignore')

# ===== Physical parameters (matching baseline_comparison.py) =====
N = 256
Ea, kB, T0, nu = 0.55, 8.617333262145e-5, 300.0, 1e13
tau_med = (1.0 / nu) * np.exp(Ea / (kB * T0))
CV, I_HRS, gamma_c = 0.20, 50e-15, np.log(100)
alpha0, pre_pulses, pw = 0.02, 50, 1e-6
NOISE_RMS = 10e-12
T_SAMPLE = np.linspace(0, 1000e-6, 30)
N_TRAIN, N_TEST = 2000, 400
N_RUNS = 30  # Now feasible with batch optimization (was 10 when ESN was slow)

BIN_LOW  = [200e-6] * 20
BIN_HIGH = [ 20e-6] * 20
AB_INT   = [20e-6]*10 + [200e-6]*10
BA_INT   = [200e-6]*10 + [20e-6]*10

# ===== ESN class (extended from baseline_comparison.py) =====
class ESN:
    def __init__(self, n_input=1, n_reservoir=40, spectral_radius=0.9,
                 input_scaling=0.5, leaking_rate=0.2, hetero_lr=False,
                 cv_lr=0.20, seed=42):
        self.n_reservoir = n_reservoir
        rng = np.random.RandomState(seed)
        self.W_in = rng.uniform(-input_scaling, input_scaling,
                                (n_reservoir, n_input)).astype(np.float64)
        W_res = rng.randn(n_reservoir, n_reservoir).astype(np.float64) * 0.5
        rho = max(abs(np.linalg.eigvals(W_res)))
        self.W_res = W_res * (spectral_radius / max(rho, 1e-12))
        self.bias = rng.uniform(-0.1, 0.1, n_reservoir).astype(np.float64)

        if hetero_lr:
            s = np.sqrt(np.log(1 + cv_lr**2))
            mu = np.log(leaking_rate) - 0.5 * s**2
            self.lr = rng.lognormal(mu, s, n_reservoir)
            self.lr = np.clip(self.lr, 0.01, 0.9)
        else:
            self.lr = np.full(n_reservoir, leaking_rate, dtype=np.float64)
        # OPTIMIZED: precompute complement for in-place state update
        self.one_minus_lr = 1.0 - self.lr

    def process(self, u_seq):
        """Process input sequence. Returns states (T, n_reservoir).
        NOTE: This is deterministic for same input — call once per class.
        """
        T = u_seq.shape[0]
        r = np.zeros(self.n_reservoir, dtype=np.float64)
        states = np.empty((T, self.n_reservoir), dtype=np.float64)
        lr = self.lr
        omlr = self.one_minus_lr
        W_in, W_res, bias = self.W_in, self.W_res, self.bias
        for t in range(T):
            r = omlr * r + lr * np.tanh(W_in @ u_seq[t] + W_res @ r + bias)
            states[t] = r
        return states

    def relax(self, r0, n_steps):
        """Relaxation with zero input. Returns relaxation states (n_steps, n_res).
        NOTE: Deterministic for same r0 — call once per class.
        """
        r = r0.copy()
        states = np.empty((n_steps, self.n_reservoir), dtype=np.float64)
        lr = self.lr
        omlr = self.one_minus_lr
        W_res, bias = self.W_res, self.bias
        for t in range(n_steps):
            r = omlr * r + lr * np.tanh(W_res @ r + bias)
            states[t] = r
        return states

def intervals_to_pulsetrain(intervals, dt=20e-6, max_len=200):
    ts = np.zeros(max_len)
    pos = 0
    if pos < max_len:
        ts[pos] = 1.0
        pos += 1
    for dt_int in intervals:
        gap_steps = max(1, int(round(dt_int / dt)))
        pos += gap_steps
        if pos >= max_len:
            break
        ts[pos] = 1.0
        pos += 1
    return ts

def esn_gen(esn, intervals_list, n_per_class, noise_rng, noise_std=0.08, dt=20e-6):
    """OPTIMIZED: Compute ESN clean features ONCE per class, batch-add noise.
    
    Same-class samples have identical input → identical deterministic ESN output.
    Only noise differs per sample.
    
    Old: 2400 × (200 process + 50 relax) = 600K timesteps per run
    New: 2 × (200 + 50) = 500 timesteps per run (~1200x speedup)
    """
    n_res = esn.n_reservoir
    n_relax = int(1e-3 / dt)
    sample_idx = np.linspace(0, n_relax - 1, 30).astype(int)
    n_class = len(intervals_list)

    # Step 1: Compute clean features ONCE per class (deterministic)
    clean_feats = []
    for cid, intv in enumerate(intervals_list):
        u1d = intervals_to_pulsetrain(intv, dt=dt)
        u_seq = u1d.reshape(-1, 1)
        states = esn.process(u_seq)          # (T_pulse, n_res)
        r0 = states[-1]                       # final state after pulse train
        relax_states = esn.relax(r0, n_relax)  # (n_relax, n_res)
        feat = relax_states[sample_idx].ravel()  # (30 * n_res,)
        clean_feats.append(feat)

    feat_dim = len(clean_feats[0])
    total_samples = n_per_class * n_class

    # Step 2: Batch generate — same clean feature + different noise per sample
    X = np.empty((total_samples, feat_dim))
    y = np.empty(total_samples, dtype=int)
    for cid in range(n_class):
        start = cid * n_per_class
        end = start + n_per_class
        noise = noise_rng.normal(0, noise_std, (n_per_class, feat_dim))
        X[start:end] = clean_feats[cid][None, :] + noise
        y[start:end] = cid

    return X, y

def esn_run(seed, intervals_list, n_reservoir=40, spectral_radius=0.9,
            leaking_rate=0.2, hetero_lr=False, cv_lr=0.20, n_runs=N_RUNS):
    n_class = len(intervals_list)
    noise_rng = np.random.RandomState(seed + 7777)
    esn = ESN(n_reservoir=n_reservoir, spectral_radius=spectral_radius,
              leaking_rate=leaking_rate, hetero_lr=hetero_lr, cv_lr=cv_lr,
              seed=seed + 999)
    Xtr, ytr = esn_gen(esn, intervals_list, N_TRAIN // n_class, noise_rng)
    Xte, yte = esn_gen(esn, intervals_list, N_TEST // n_class, noise_rng)
    # FIX: use stratified split instead of sequential slicing
    from sklearn.model_selection import train_test_split
    X_all = np.vstack([Xtr, Xte])
    y_all = np.concatenate([ytr, yte])
    Xtr, Xte, ytr, yte = train_test_split(X_all, y_all, train_size=N_TRAIN, test_size=N_TEST,
                                            stratify=y_all, random_state=seed)
    scaler = StandardScaler()
    clf = RidgeClassifier(alpha=0.01)
    clf.fit(scaler.fit_transform(Xtr), ytr)
    return clf.score(scaler.transform(Xte), yte)

def run_config(name, intervals_list, n_reservoir, spectral_radius,
               leaking_rate, hetero_lr=False, cv_lr=0.20, n_runs=N_RUNS):
    print(f"\n  {name} ({n_runs} runs)...", end="", flush=True)
    t0 = time.time()
    accs = []
    for run in range(n_runs):
        seed = run * 100 + 42
        acc = esn_run(seed, intervals_list, n_reservoir, spectral_radius,
                      leaking_rate, hetero_lr, cv_lr)
        accs.append(acc * 100)
    m = np.mean(accs)
    ci = stats.t.interval(0.95, n_runs-1, loc=m, scale=stats.sem(accs))
    print(f" {m:.1f}% [{ci[0]:.1f}, {ci[1]:.1f}]% ({time.time()-t0:.0f}s)")
    return {'name': name, 'mean': m, 'ci_low': ci[0], 'ci_high': ci[1], 'accs': accs}

if __name__ == '__main__':
    results = []
    for task_name, intervals in [("Binary", [BIN_LOW, BIN_HIGH]),
                                  ("AB-BA", [AB_INT, BA_INT])]:
        print(f"\n{'='*60}")
        print(f"Task: {task_name}")
        print(f"{'='*60}")

        # 1. Original unfair ESN (40 units, uniform LR)
        r = run_config("ESN-40-uniform", intervals, 40, 0.9, 0.2)
        r['task'] = task_name; results.append(r)

        # 2. Matched unit count (256 units, uniform LR)
        r = run_config("ESN-256-uniform", intervals, 256, 0.9, 0.2)
        r['task'] = task_name; results.append(r)

        # 3. Matched units + heterogeneous LR
        r = run_config("ESN-256-hetero", intervals, 256, 0.9, 0.2, hetero_lr=True)
        r['task'] = task_name; results.append(r)

        # 4. Hyperparameter grid search (on first 3 seeds)
        print(f"\n  Grid search (3 seeds, AB-BA task)...")
        best_acc, best_sr, best_lr = 0, 0.9, 0.2
        for sr in [0.5, 0.9, 1.2, 1.5]:
            for lr_val in [0.1, 0.2, 0.4, 0.8]:
                accs = []
                for run in range(3):
                    seed = run * 100 + 42
                    acc = esn_run(seed, intervals, 256, sr, lr_val, hetero_lr=True)
                    accs.append(acc * 100)
                m = np.mean(accs)
                if m > best_acc:
                    best_acc, best_sr, best_lr = m, sr, lr_val
                print(f"    SR={sr}, LR={lr_val}: {m:.1f}%")
        print(f"  Best: SR={best_sr}, LR={best_lr} ({best_acc:.1f}%)")

        # 5. Run best config with full N_RUNS
        r = run_config(f"ESN-256-best(SR={best_sr},LR={best_lr})", intervals,
                       256, best_sr, best_lr, hetero_lr=True)
        r['task'] = task_name; results.append(r)

    # Save results
    data_dir = Path(__file__).parent.parent / 'data'
    data_dir.mkdir(exist_ok=True)

    csv_path = data_dir / 'fair_esn_results.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['task','name','mean','ci_low','ci_high'])
        w.writeheader()
        for r in results:
            w.writerow({k: v for k, v in r.items() if k in w.fieldnames})

    json_path = data_dir / 'fair_esn_results.json'
    with open(json_path, 'w') as f:
        json.dump({'results': results}, f, indent=2, default=str)

    print(f"\nResults saved to:\n  {csv_path}\n  {json_path}")
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for r in results:
        print(f"  {r['task']:8s} | {r['name']:35s} | {r['mean']:.1f}% [{r['ci_low']:.1f}, {r['ci_high']:.1f}]%")
