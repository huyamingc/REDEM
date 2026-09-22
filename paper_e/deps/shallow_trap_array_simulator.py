#!/usr/bin/env python3
"""
Vectorized Si3N4 shallow-trap relaxation-reservoir simulator (primary CORE model).
=============================================================================
Type:           CORE
Paper Section:  Methods, substrate model (imported by every s-family script)
Experiment:     square-array (k x k) accuracy / signal / SNR sweep
                (fully vectorized NumPy core, no per-device Python loops)

Purpose:
  Flat vectorized replacement for the earlier ShallowTrapDevice / SquareArray
  class-based simulator, 10-50x faster, with identical physics: log-normal
  time constants around tau_0 ~ 174 us, temperature-scaled interval pair
  (dt_low, dt_high), 50-pulse preprogramming, and per-column current noise.

This module is CORE: it is imported by the s-family PAPER scripts, so its
exported names and their numerical values must not be changed without checking
every importer.

Run as a script it sweeps N over {64, 256, 1024, 4096, 16384} and writes
  data/square_array_results.csv  (one row per N; the run's parameters are
                                  carried as trailing columns, see
                                  PARAM_COLUMNS)
  data/square_array_params.json  (the complete parameter set of that sweep)
Usage:
  python shallow_trap_array_simulator.py              full sweep
  python shallow_trap_array_simulator.py --self-test  noiseless-branch check

Dependencies: numpy, scipy, scikit-learn, numba (optional, with fallback).
"""

import numpy as np
from scipy import stats
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings, csv, math, os, sys, json, time

# Unbuffered output: this module is also run as a script and its sweep prints
# progress over several minutes, which must not sit in a block buffer.
os.environ.setdefault('PYTHONUNBUFFERED', '1')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

# numba JIT acceleration (optional, 9x speedup on core loop functions)
try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    def njit(*args, **kwargs):
        """Fallback: no-op decorator when numba is unavailable."""
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return lambda f: f

warnings.filterwarnings('ignore')

# ========================== Physical constants ==========================
kB = 8.617333262145e-5
T0 = 300.0
Ea = 0.55
nu = 1e13
tau0 = 1.0 / nu * np.exp(Ea / (kB * T0))
I_HRS = 50e-15
ON_OFF_RATIO = 100
gamma = np.log(ON_OFF_RATIO)

DT_LOW_REF = 200e-6
DT_HIGH_REF = 20e-6
N_PRE = 50
N_PULSES = 20
N_SAMPLE = 30
T_SAMPLE_MAX = 1e-3
T_SAMPLES = np.linspace(0, T_SAMPLE_MAX, N_SAMPLE)  # precomputed

# Sweep settings, recorded into the CSV parameter columns and the companion
# JSON so that every reported accuracy/SNR value is traceable without reading
# the source. They are the values the __main__ sweep has always used.
SWEEP_T_K = 300.0            # temperature used for the interval scaling [K]
SWEEP_K_COMP = 0.0           # temperature interval-compression exponent
SWEEP_MEDIAN_TAU = tau0      # median time constant [s]
SWEEP_N_TRAIN = 2000         # training samples per Monte-Carlo run
SWEEP_N_TEST = 400           # held-out samples per Monte-Carlo run
# Per-run time-constant seed rule used by evaluate_accuracy_vec:
#   gen_tau_vec(N, cv_tau, median_tau, seed=run) with run in [0, n_runs)
TAU_SEED_RULE = 'gen_tau_vec(N, cv_tau, median_tau, seed=run_index)'
# Base seed of the train/test RNG: per-run seed = run_index*100 + TAU_NOISE_SEED
TAU_NOISE_SEED = 7777

def compute_tau_T(T, tau_ref=tau0, T_ref=T0, Ea=Ea):
    return tau_ref * np.exp(Ea/kB * (1/T - 1/T_ref))

def adjust_intervals(T, dt_low_ref, dt_high_ref, k_comp, tau_ref=tau0, T_ref=T0, Ea=Ea):
    tau_T = compute_tau_T(T, tau_ref, T_ref, Ea)
    s = (tau_T / tau_ref) ** k_comp
    dt_low = max(dt_low_ref * s, 2e-6)
    dt_high = max(dt_high_ref * s, 2e-6)
    return dt_low, dt_high

# ========================== Vectorized core ==========================

def gen_tau_vec(N, cv, median_tau, seed):
    """Generate tau distribution with independent RandomState."""
    rng = np.random.RandomState(seed)
    sigma = np.sqrt(np.log(1 + cv**2))  # log-normal sigma-CV relation
    mu = np.log(median_tau) - 0.5 * sigma**2
    return rng.lognormal(mu, sigma, N)

@njit(fastmath=True, cache=True)
def preprogram_vec(alpha, tau, dt_pre=2e-6, n_pre=N_PRE, pw=1e-6):
    """Vectorized preprogramming: returns x0 array (N,).
    Applies n_pre write pulses with pulse width pw and interval dt_pre.
    Double relaxation: pulse-width decay + interval decay per iteration,
    consistent with apply_pulse_sequence_vec.
    Numba-JIT compiled for 9x speedup on the n_pre loop."""
    x = np.zeros(len(tau))
    decay_pw = np.exp(-pw / tau)       # (N,) pulse width relaxation
    decay_dt = np.exp(-dt_pre / tau)   # (N,) interval relaxation
    for _ in range(n_pre):
        x = x + alpha * (1.0 - x)
        x = x * decay_pw
        x = x * decay_dt
        x = np.clip(x, 0.0, 1.0)
    return x

@njit(fastmath=True, cache=True)
def apply_pulse_sequence_vec(x0, tau, alpha, dt, n_pulses=N_PULSES, pw=1e-6):
    """Vectorized pulse sequence: returns final x array (N,).
    Applies n_pulses write pulses with pulse width pw and interval dt.
    Numba-JIT compiled for 7x speedup on the n_pulses loop."""
    x = x0.copy()
    decay_pw = np.exp(-pw / tau)    # (N,) pulse width relaxation
    decay_dt = np.exp(-dt / tau)    # (N,) interval relaxation
    for _ in range(n_pulses):
        x = x + alpha * (1.0 - x)
        x = x * decay_pw
        x = x * decay_dt
        x = np.clip(x, 0.0, 1.0)
    return x

def sample_currents_vec(x_final, tau, t_samples=T_SAMPLES, col_noise_rms=0.0,
                         noise_rng=None, k=None):
    """Vectorized current sampling: returns total current (T,).
    x_final: (N,) final state after pulse sequence
    tau: (N,) device taus
    """
    # x_temp: (T, N) — state at each sample time
    x_temp = x_final[None, :] * np.exp(-t_samples[:, None] / tau[None, :])
    np.clip(x_temp, 0.0, 1.0, out=x_temp)
    # I_all: (T, N) — current from each device
    I_all = I_HRS * np.exp(gamma * x_temp)
    # Sum all devices (total current) — column noise added if specified
    total = np.sum(I_all, axis=1)  # (T,)
    if col_noise_rms > 0 and noise_rng is not None and k is not None:
        # Per-column noise: k columns, each with col_noise_rms
        # Total noise = sqrt(k) * col_noise_rms on the sum
        total_noise = col_noise_rms * np.sqrt(k)
        total = total + noise_rng.normal(0, total_noise, len(t_samples))
    return total

def clean_signal_vec(x_final, tau, t_samples=T_SAMPLES):
    """Compute clean (noiseless) total current signal (T,).
    This is identical for all same-class samples, so compute once.
    """
    x_temp = x_final[None, :] * np.exp(-t_samples[:, None] / tau[None, :])
    np.clip(x_temp, 0.0, 1.0, out=x_temp)
    return np.sum(I_HRS * np.exp(gamma * x_temp), axis=1)  # (T,)

def batch_sample_currents(x_final, tau, col_noise_rms, noise_rng, k,
                           t_samples=T_SAMPLES, n_samples=1):
    """OPTIMIZED: Batch generate n_samples waveforms for same class.
    Since x_final is identical for same-class samples, compute clean signal
    once and only vary noise per sample.
    Returns (n_samples, T) array.
    """
    I_clean = clean_signal_vec(x_final, tau, t_samples)  # (T,)
    if col_noise_rms > 0 and k is not None:
        total_noise = col_noise_rms * np.sqrt(k)
        noise = noise_rng.normal(0, total_noise, (n_samples, len(t_samples)))
        return np.maximum(0.0, I_clean[None, :] + noise)  # (n_samples, T)
    else:
        # Noiseless branch: broadcast the single clean waveform to n_samples
        # rows. NOTE: np.broadcast(...) returns a read-only broadcast object
        # without .copy() and raises ValueError; np.broadcast_to(...).copy()
        # is the correct call (bug fix 2026-09-10). Only this
        # col_noise_rms == 0 path is affected; the noise path is unchanged.
        return np.broadcast_to(I_clean[None, :],
                               (n_samples, len(t_samples))).copy()

def generate_dataset_vec(tau, alpha, dt_low, dt_high, col_noise_rms,
                          n_train, n_test, noise_seed, k=None):
    """OPTIMIZED: Generate full train/test dataset using batch generation.
    Same-class samples share identical clean signal; only noise differs.
    Eliminates 2400 per-sample Python loop calls. Uses stratified split.
    """
    from sklearn.model_selection import train_test_split
    noise_rng = np.random.RandomState(noise_seed)
    x0 = preprogram_vec(alpha, tau)

    # Precompute final states for both classes
    x_low = apply_pulse_sequence_vec(x0, tau, alpha, dt_low)
    x_high = apply_pulse_sequence_vec(x0, tau, alpha, dt_high)

    n_tr_half = n_train // 2
    n_te_half = n_test // 2

    # Batch generate: compute clean signal once per class, batch noise
    Xtr = np.empty((n_train, N_SAMPLE))
    ytr = np.empty(n_train, dtype=int)
    Xte = np.empty((n_test, N_SAMPLE))
    yte = np.empty(n_test, dtype=int)

    # Train class 0 (low intervals) — all samples in one call
    Xtr[:n_tr_half] = batch_sample_currents(x_low, tau, col_noise_rms,
                                              noise_rng, k, n_samples=n_tr_half)
    ytr[:n_tr_half] = 0
    # Train class 1 (high intervals)
    Xtr[n_tr_half:] = batch_sample_currents(x_high, tau, col_noise_rms,
                                              noise_rng, k, n_samples=n_tr_half)
    ytr[n_tr_half:] = 1

    # Test class 0
    Xte[:n_te_half] = batch_sample_currents(x_low, tau, col_noise_rms,
                                              noise_rng, k, n_samples=n_te_half)
    yte[:n_te_half] = 0
    # Test class 1
    Xte[n_te_half:] = batch_sample_currents(x_high, tau, col_noise_rms,
                                              noise_rng, k, n_samples=n_te_half)
    yte[n_te_half:] = 1

    # stratified split (not sequential slicing)
    X_all = np.vstack([Xtr, Xte])
    y_all = np.concatenate([ytr, yte])
    Xtr, Xte, ytr, yte = train_test_split(X_all, y_all, train_size=n_train, test_size=n_test,
                                            stratify=y_all, random_state=noise_seed)

    return Xtr, ytr, Xte, yte

def evaluate_accuracy_vec(N, alpha, cv_tau, col_noise_rms, n_train=2000, n_test=400,
                           T=300.0, k_comp=0.0, n_runs=30, median_tau=tau0):
    """Run n_runs MC and return mean, std, CI."""
    k = int(math.isqrt(N))
    dt_low, dt_high = adjust_intervals(T, DT_LOW_REF, DT_HIGH_REF, k_comp)
    acc_list = []

    for run in range(n_runs):
        tau = gen_tau_vec(N, cv_tau, median_tau, seed=run)
        noise_seed = run * 100 + 7777
        Xtr, ytr, Xte, yte = generate_dataset_vec(
            tau, alpha, dt_low, dt_high, col_noise_rms,
            n_train, n_test, noise_seed, k=k)
        scaler = StandardScaler()
        clf = RidgeClassifier(alpha=0.01)
        clf.fit(scaler.fit_transform(Xtr), ytr)
        acc = accuracy_score(yte, clf.predict(scaler.transform(Xte)))
        acc_list.append(acc)

    mean_acc = np.mean(acc_list)
    std_acc = np.std(acc_list)
    if n_runs < 30:
        t_critical = stats.t.ppf(0.975, df=n_runs - 1)
    else:
        t_critical = 1.96
    ci_low = mean_acc - t_critical * std_acc / np.sqrt(n_runs)
    ci_high = mean_acc + t_critical * std_acc / np.sqrt(n_runs)
    return mean_acc, std_acc, ci_low, ci_high

def compute_signal_metrics_vec(N, alpha, cv_tau, col_noise_rms, T=300.0, k_comp=0.0,
                                n_samples=200, median_tau=tau0):
    """OPTIMIZED: Compute signal metrics using batch generation.
    Clean signal computed once per class; noise batch-generated.
    """
    k = int(math.isqrt(N))
    tau = gen_tau_vec(N, cv_tau, median_tau, seed=0)
    x0 = preprogram_vec(alpha, tau)
    dt_low, dt_high = adjust_intervals(T, DT_LOW_REF, DT_HIGH_REF, k_comp)
    x_low = apply_pulse_sequence_vec(x0, tau, alpha, dt_low)
    x_high = apply_pulse_sequence_vec(x0, tau, alpha, dt_high)
    noise_rng = np.random.RandomState(42)

    n_half = n_samples // 2
    # Clean signals: one per class (no noise)
    clean_low = clean_signal_vec(x_low, tau)   # (T,)
    clean_high = clean_signal_vec(x_high, tau)  # (T,)
    # Noisy signals: batch generate
    noisy_low = batch_sample_currents(x_low, tau, col_noise_rms, noise_rng, k,
                                       n_samples=n_half)   # (n_half, T)
    noisy_high = batch_sample_currents(x_high, tau, col_noise_rms, noise_rng, k,
                                        n_samples=n_half)  # (n_half, T)

    # Tile clean signals to match noisy shape for difference computation
    clean_all = np.concatenate([np.broadcast_to(clean_low, (n_half, N_SAMPLE)),
                                np.broadcast_to(clean_high, (n_half, N_SAMPLE))])
    noisy_all = np.concatenate([noisy_low, noisy_high])

    avg_signal_pA = np.mean(clean_all) * 1e12
    signal_rms = np.sqrt(np.mean(clean_all**2)) * 1e12
    noise_std_pA = np.std(noisy_all - clean_all) * 1e12
    snr_db = 20 * np.log10(signal_rms / noise_std_pA) if noise_std_pA > 0 else float('inf')
    return avg_signal_pA, noise_std_pA, snr_db

# ========================== Regression test ==========================
def regression_test_noiseless_batch(n_samples=3, n_units=64, seed=0):
    """Minimal regression test for the NOISELESS batch branch.

    batch_sample_currents(..., col_noise_rms=0, ...) must return a
    (n_samples, N_SAMPLE) array whose rows all equal the clean waveform.
    Covers the np.broadcast -> np.broadcast_to bug fixed 2026-09-09
    (dedup P1-14), which raised ValueError on this path.
    Returns (ok, detail).
    """
    tau = gen_tau_vec(n_units, 0.20, tau0, seed=seed)
    x0 = preprogram_vec(0.02, tau)
    x_fin = apply_pulse_sequence_vec(x0, tau, 0.02, DT_LOW_REF)
    ref = clean_signal_vec(x_fin, tau)
    out = batch_sample_currents(x_fin, tau, 0.0, None, None,
                                n_samples=n_samples)
    shape_ok = out.shape == (n_samples, N_SAMPLE)
    match_ok = bool(np.allclose(
        out, np.broadcast_to(ref, (n_samples, N_SAMPLE))))
    max_err = float(np.max(np.abs(out - ref))) if out.size else float('nan')
    return (shape_ok and match_ok,
            f"shape={out.shape} max|d|={max_err:.3e}")


# ========================== Main ==========================
# Trailing parameter columns appended to every CSV row, so that each recorded
# accuracy/SNR value is traceable to the settings that produced it. The
# pre-existing accuracy/signal columns keep their original names and order and
# come first; these columns are appended after them.
PARAM_COLUMNS = [
    'alpha',                # injection coefficient
    'cv_tau',               # coefficient of variation of the time constants
    'median_tau_s',         # tau_0 in seconds
    'k_comp',               # temperature interval-compression exponent
    'T_K',                  # temperature used for the interval scaling
    'noise_per_col_pA',     # per-column current-noise RMS [pA]
    'noise_total_pA',       # total array current-noise RMS [pA]
    'n_train',              # training samples per Monte-Carlo run
    'n_test',               # held-out samples per Monte-Carlo run
    'tau_seed_rule',        # per-run tau seed rule
    'noise_seed',           # RNG seed passed to the train/test split
]

if __name__ == "__main__":
    if '--self-test' in sys.argv:
        ok, detail = regression_test_noiseless_batch()
        print(f"[{'PASS' if ok else 'FAIL'}] noiseless_batch: {detail}")
        sys.exit(0 if ok else 1)
    print("=" * 60)
    print("VECTORIZED Square array (k×k) with per-column noise")
    print("=" * 60)

    results = []
    alpha = 0.02
    cv_tau = 0.20
    median_tau = SWEEP_MEDIAN_TAU
    k_comp = SWEEP_K_COMP
    T_K = SWEEP_T_K
    n_train = SWEEP_N_TRAIN
    n_test = SWEEP_N_TEST

    for N in [64, 256, 1024, 4096, 16384]:
        k = int(math.isqrt(N))
        total_noise = 10e-12 * np.sqrt(k) / 4.0  # keep ~10pA for 16x16
        col_noise = total_noise / np.sqrt(k)
        n_runs = 30  # All sizes now feasible with batch optimization

        t0 = time.time()
        mean_acc, std_acc, ci_low, ci_high = evaluate_accuracy_vec(
            N, alpha, cv_tau, col_noise, n_train=n_train, n_test=n_test,
            T=T_K, k_comp=k_comp, n_runs=n_runs, median_tau=median_tau)
        elapsed = time.time() - t0

        sig, nse, snr = compute_signal_metrics_vec(
            N, alpha, cv_tau, col_noise, T=T_K, k_comp=k_comp,
            median_tau=median_tau)
        print(f"--- N = {N} ({k}×{k}) [{elapsed:.0f}s, {n_runs} runs] ---")
        print(f"  Accuracy: {mean_acc*100:.1f}% CI=[{ci_low*100:.1f},{ci_high*100:.1f}] | "
              f"Signal mean={sig:.2f} pA | Noise STD={nse:.2f} pA | SNR={snr:.1f} dB")

        results.append({
            'N': N, 'k': k, 'accuracy_pct': mean_acc*100,
            'ci_low_pct': ci_low*100, 'ci_high_pct': ci_high*100,
            'signal_pA': sig, 'noise_pA': nse, 'snr_dB': snr, 'n_runs': n_runs,
            'alpha': alpha, 'cv_tau': cv_tau, 'median_tau_s': median_tau,
            'k_comp': k_comp, 'T_K': T_K,
            'noise_per_col_pA': col_noise * 1e12,
            'noise_total_pA': total_noise * 1e12,
            'n_train': n_train, 'n_test': n_test,
            'tau_seed_rule': TAU_SEED_RULE,
            'noise_seed': TAU_NOISE_SEED,
        })

    # Save CSV: the original columns first, then the parameter columns.
    # Output goes to the repository-level data/ directory, the same place every
    # s-family script writes its results, so the sweep's output sits with the
    # rest of the committed data rather than in an unused paper_e/data/ path.
    data_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             '..', '..', 'data'))
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, 'square_array_results.csv')
    fieldnames = [c for c in results[0].keys() if c not in PARAM_COLUMNS] + PARAM_COLUMNS
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)
    print(f"\nResults saved to {csv_path}")

    # Companion JSON: the complete parameter set of this sweep, so the CSV can
    # be read without the source. Written next to the CSV.
    params_path = os.path.join(data_dir, 'square_array_params.json')
    params = {
        'script': 'shallow_trap_array_simulator.py',
        'type': 'CORE',
        'experiment': 'square_array_sweep',
        'N_list': [int(r['N']) for r in results],
        'alpha': alpha,
        'cv_tau': cv_tau,
        'median_tau_s': median_tau,
        'k_comp': k_comp,
        'T_K': T_K,
        'dt_low_ref_s': DT_LOW_REF,
        'dt_high_ref_s': DT_HIGH_REF,
        'n_pre_pulses': N_PRE,
        'n_pulses': N_PULSES,
        'n_sample': N_SAMPLE,
        'n_train': n_train,
        'n_test': n_test,
        'n_runs': n_runs,
        'tau_seed_rule': TAU_SEED_RULE,
        'noise_seed': TAU_NOISE_SEED,
        'noise_model': 'per-column Gaussian current noise, RMS = total/sqrt(k)',
        'classifier': 'StandardScaler + RidgeClassifier(alpha=0.01)',
        'outputs': ['data/square_array_results.csv'],
        'physical_constants': {
            'kB_eV_per_K': kB, 'T0_K': T0, 'Ea_eV': Ea, 'nu_Hz': nu,
            'tau0_s': tau0, 'I_HRS_A': I_HRS, 'ON_OFF_RATIO': ON_OFF_RATIO,
            'gamma': gamma,
        },
    }
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=2)
    print(f"Parameters saved to {params_path}")
