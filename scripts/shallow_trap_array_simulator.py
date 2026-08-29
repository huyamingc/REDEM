#!/usr/bin/env python3
"""
Vectorized simulator for Si3N4 shallow-trap relaxation reservoir.
OPTIMIZED: fully vectorized NumPy operations, no per-device Python loops.
This replaces the slow ShallowTrapDevice/SquareArray class-based version
with flat vectorized functions that achieve 10-50x speedup.
"""

import numpy as np
from scipy import stats
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings, csv, math, os, time

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
        return np.broadcast(I_clean[None, :], (n_samples, len(t_samples))).copy()

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

# ========================== Main ==========================
if __name__ == "__main__":
    print("=" * 60)
    print("VECTORIZED Square array (k×k) with per-column noise")
    print("=" * 60)

    results = []
    alpha = 0.02
    cv_tau = 0.20

    for N in [64, 256, 1024, 4096, 16384]:
        k = int(math.isqrt(N))
        total_noise = 10e-12 * np.sqrt(k) / 4.0  # keep ~10pA for 16x16
        col_noise = total_noise / np.sqrt(k)
        n_runs = 30  # All sizes now feasible with batch optimization

        t0 = time.time()
        mean_acc, std_acc, ci_low, ci_high = evaluate_accuracy_vec(
            N, alpha, cv_tau, col_noise, n_train=2000, n_test=400, n_runs=n_runs)
        elapsed = time.time() - t0

        sig, nse, snr = compute_signal_metrics_vec(N, alpha, cv_tau, col_noise)
        print(f"--- N = {N} ({k}×{k}) [{elapsed:.0f}s, {n_runs} runs] ---")
        print(f"  Accuracy: {mean_acc*100:.1f}% CI=[{ci_low*100:.1f},{ci_high*100:.1f}] | "
              f"Signal mean={sig:.2f} pA | Noise STD={nse:.2f} pA | SNR={snr:.1f} dB")

        results.append({
            'N': N, 'k': k, 'accuracy_pct': mean_acc*100,
            'ci_low_pct': ci_low*100, 'ci_high_pct': ci_high*100,
            'signal_pA': sig, 'noise_pA': nse, 'snr_dB': snr, 'n_runs': n_runs
        })

    # Save CSV
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'square_array_results.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print(f"\nResults saved to {csv_path}")
