#!/usr/bin/env python3
"""
Recurrent relaxation substrate with per-pulse topology coupling (REDEM S1).
=============================================================================
Type:           CORE
Paper Section:  New-algorithm project Step S1 (see NEW_ALGORITHM_PLAN.md)
Experiment:     REDEM recurrent substrate core dynamics

Purpose:
  Generalizes the parallel shallow-trap array (shallow_trap_array_simulator)
  into a recurrently coupled substrate: the injection coefficient of each
  unit is modulated every pulse by a topology-dependent contrast against its
  neighbors' current ratios. This is the per-pulse temporal generalization of
  the quasi-static alpha_eff coupling in topology_comparison.py.

Physics (per pulse step t with interval dt_t after the pulse):
  1. Current ratios  i_j = exp(gamma * x_j)   (scale-free observable, in [1, 100])
  2. Coupling contrast g_i from CSR topology neighbors:
       CONTRAST_SELF (mode 1): g_i = (wmean(i_nbrs) - i_i) / i_i        (v4 style)
       CONTRAST_NBR  (mode 2): g_i = (i_i - wmean(i_nbrs)) / wmean(i_nbrs)
                                (Phase3 ring_unidir / hub_star style)
       ADDITIVE      (mode 3): control condition, non-physical ESN-style
                                drive: x_i += kappa * wmean(x_nbrs) applied
                                after the standard uncoupled update, clipped.
  3. alpha_eff_i = clip(alpha0 * (1 + kappa * g_i), alpha_min, alpha_max)
  4. Injection:           x_i = x_i + alpha_eff_i * (1 - x_i)
  5. Pulse-width relax.:  x_i *= exp(-pw / tau_i)
  6. Interval relaxation: x_i *= exp(-dt_t / tau_i)
  7. Clip to [0, 1]

  The two-pass structure (compute all contrasts from the pre-update state,
  then update all units) makes the step order-independent.

Downstream importers (do not break this API without notice):
  scripts/substrate_recurrence_characterization.py (PAPER, S1)

Dependencies: numpy, numba (optional), shallow_trap_array_simulator (CORE).
NOTE: shallow_trap_array_simulator.py is NEVER modified; only imported.
"""
import os
import sys

import numpy as np

# numba JIT acceleration (optional), same fallback pattern as existing CORE
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


# Physics anchors imported from the existing CORE (never modified).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shallow_trap_array_simulator import gamma, tau0, gen_tau_vec, preprogram_vec

# Coupling mode codes
COUPLING_NONE = 0           # parallel baseline (kappa ignored)
COUPLING_CONTRAST_SELF = 1  # v4 style: (neighbor_mean - self) / self
COUPLING_CONTRAST_NBR = 2   # Phase3/hub style: (self - neighbor_mean) / neighbor_mean
COUPLING_ADDITIVE = 3       # control: additive state drive (non-physical)

# Default physical parameters (consistent with shallow_trap_array_simulator)
PW = 1e-6                    # pulse width [s], matches CORE default
ALPHA0 = 0.02                # baseline injection coefficient
ALPHA_MIN = 0.001            # physical alpha_eff clip lower bound
ALPHA_MAX = 0.10             # physical alpha_eff clip upper bound


# ========================== Topology builders (CSR) ==========================
# CSR format: indptr (N+1,) int64, indices (E,) int64, wts (E,) float64.
# Row i lists unit i's neighbors; weights are row-normalized (each row sums
# to 1) so the weighted neighbor mean is a plain weighted sum.

def build_topology_csr(name, n_units, seed=777, lateral_radius=4, avg_degree=8):
    """Build CSR adjacency for a named topology.

    Supported names:
      parallel     : no edges (uncoupled baseline)
      ring_bidir   : bidirectional ring, neighbors i-1 / i+1
      ring_unidir  : directed ring, each unit watches its predecessor
      hub_star     : unit 0 is the hub (unmodified), all others watch the hub
      lateral_ring : ring with distance-decay weights within +/- lateral_radius
      random_graph : Erdos-Renyi undirected graph with mean degree ~ avg_degree

    Returns (indptr, indices, wts).
    """
    n = int(n_units)
    if n < 2:
        raise ValueError("n_units must be >= 2")

    if name == "parallel":
        return (np.zeros(n + 1, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64))

    if name == "ring_bidir":
        indptr = (np.arange(n + 1, dtype=np.int64) * 2)
        indices = np.empty(2 * n, dtype=np.int64)
        for i in range(n):
            indices[2 * i] = (i - 1) % n
            indices[2 * i + 1] = (i + 1) % n
        wts = np.full(2 * n, 0.5, dtype=np.float64)
        return indptr, indices, wts

    if name == "ring_unidir":
        indptr = np.arange(n + 1, dtype=np.int64)
        indices = np.array([(i - 1) % n for i in range(n)], dtype=np.int64)
        wts = np.ones(n, dtype=np.float64)
        return indptr, indices, wts

    if name == "hub_star":
        # unit 0 = hub with no neighbors; units 1..n-1 each watch unit 0
        indices = np.zeros(n - 1, dtype=np.int64)  # all neighbors are unit 0
        indptr = np.zeros(n + 1, dtype=np.int64)
        for i in range(1, n):
            indptr[i] = i - 1
        indptr[n] = n - 1
        wts = np.ones(n - 1, dtype=np.float64)
        return indptr, indices, wts

    if name == "lateral_ring":
        r = int(lateral_radius)
        r = max(1, min(r, (n // 2) - 1 if n > 3 else 1))
        nbr_lists = []
        wt_lists = []
        for i in range(n):
            nbrs = []
            wsum = 0.0
            for d in range(1, r + 1):
                w = np.exp(-d / 2.0)
                nbrs.append(((i - d) % n, w))
                nbrs.append(((i + d) % n, w))
                wsum += 2.0 * w
            nbr_lists.append(nbrs)
            wt_lists.append(wsum)
        counts = [len(nb) for nb in nbr_lists]
        indptr = np.zeros(n + 1, dtype=np.int64)
        for i in range(n):
            indptr[i + 1] = indptr[i] + counts[i]
        total_edges = indptr[n]
        indices = np.empty(total_edges, dtype=np.int64)
        wts = np.empty(total_edges, dtype=np.float64)
        for i in range(n):
            base = indptr[i]
            wsum = wt_lists[i]
            for e, (j, w) in enumerate(nbr_lists[i]):
                indices[base + e] = j
                wts[base + e] = w / wsum
        return indptr, indices, wts

    if name == "random_graph":
        rng = np.random.RandomState(int(seed))
        p = min(1.0, float(avg_degree) / max(n - 1, 1))
        mask = rng.rand(n, n) < p
        np.fill_diagonal(mask, False)
        mask = np.triu(mask, 1)
        rows, cols = np.nonzero(mask)
        # symmetrize: each undirected edge appears in both rows
        src = np.concatenate([rows, cols])
        dst = np.concatenate([cols, rows])
        order = np.lexsort((dst, src))
        src, dst = src[order], dst[order]
        counts = np.bincount(src, minlength=n)
        indptr = np.zeros(n + 1, dtype=np.int64)
        for i in range(n):
            indptr[i + 1] = indptr[i] + counts[i]
        indices = dst.astype(np.int64)
        wts = np.empty(indices.shape[0], dtype=np.float64)
        for i in range(n):
            k = indptr[i + 1] - indptr[i]
            if k > 0:
                wts[indptr[i]:indptr[i + 1]] = 1.0 / k
            # k == 0 -> isolated unit; dynamics falls back to alpha0
        return indptr, indices, wts

    raise ValueError(f"unknown topology name: {name}")


def adjacency_to_csr(mask):
    """Convert a boolean adjacency mask (N x N, undirected, no self-loops)
    to row-normalized CSR (indptr, indices, wts). Used by the structure-
    plasticity experiments (REDEM S7) to rebuild topologies after edge
    rewiring. Isolated units keep empty rows (dynamics fall back to alpha0).
    """
    mask = np.asarray(mask, dtype=bool)
    n = mask.shape[0]
    if mask.shape != (n, n):
        raise ValueError("mask must be square")
    rows, cols = np.nonzero(mask)
    src = np.concatenate([rows, cols])
    dst = np.concatenate([cols, rows])
    order = np.lexsort((dst, src))
    src, dst = src[order], dst[order]
    counts = np.bincount(src, minlength=n)
    indptr = np.zeros(n + 1, dtype=np.int64)
    for i in range(n):
        indptr[i + 1] = indptr[i] + counts[i]
    indices = dst.astype(np.int64)
    wts = np.empty(indices.shape[0], dtype=np.float64)
    for i in range(n):
        k = indptr[i + 1] - indptr[i]
        if k > 0:
            wts[indptr[i]:indptr[i + 1]] = 1.0 / k
    return indptr, indices, wts


# ========================== Numba core dynamics ==========================

@njit(fastmath=True, cache=True)
def _couple_and_update_nb(x, tau, dt, decay_pw, indptr, indices, wts,
                          kappa, alpha0, alpha_min, alpha_max, gamma_, mode, diag):
    """One pulse step: topology coupling + injection + double relaxation.

    Updates x in place. diag is a (3,) accumulator array:
      diag[0] += 1 per unit whose alpha_eff hit a clip bound
      diag[1] += |g_i| per coupled unit
      diag[2] += 1 per coupled unit
    Two-pass (compute coupling from pre-update state, then update all units)
    so the result is independent of unit iteration order.
    """
    n = x.shape[0]
    if mode == 3:
        # --- additive control: standard update then neighbor-mean drive ---
        drive = np.empty(n)
        for i in range(n):
            s = 0.0
            for e in range(indptr[i], indptr[i + 1]):
                s += wts[e] * x[indices[e]]
            drive[i] = s
        for i in range(n):
            xi = x[i] + alpha0 * (1.0 - x[i])
            xi = xi * decay_pw[i] * np.exp(-dt / tau[i])
            xi = xi + kappa * drive[i]
            if xi < 0.0:
                xi = 0.0
            elif xi > 1.0:
                xi = 1.0
            x[i] = xi
    else:
        # --- contrast-coupled alpha_eff ---
        alpha_eff = np.empty(n)
        for i in range(n):
            a = alpha0
            if indptr[i + 1] > indptr[i]:
                s = 0.0
                for e in range(indptr[i], indptr[i + 1]):
                    s += wts[e] * np.exp(gamma_ * x[indices[e]])
                i_self = np.exp(gamma_ * x[i])
                if mode == 1:
                    g = (s - i_self) / i_self
                else:
                    g = (i_self - s) / s
                diag[1] += abs(g)
                diag[2] += 1.0
                a = alpha0 * (1.0 + kappa * g)
                if a < alpha_min:
                    a = alpha_min
                    diag[0] += 1.0
                elif a > alpha_max:
                    a = alpha_max
                    diag[0] += 1.0
            alpha_eff[i] = a
        for i in range(n):
            xi = x[i] + alpha_eff[i] * (1.0 - x[i])
            xi = xi * decay_pw[i] * np.exp(-dt / tau[i])
            if xi < 0.0:
                xi = 0.0
            elif xi > 1.0:
                xi = 1.0
            x[i] = xi


@njit(fastmath=True, cache=True)
def run_trajectory_nb(x0, tau, dt_seq, pw, indptr, indices, wts, kappa,
                      alpha0, alpha_min, alpha_max, gamma_, mode, n_washout):
    """Run one trajectory of the coupled substrate.

    Returns (states, clip_frac, g_abs_mean) where states is
    (T - n_washout, N): the state AFTER each pulse step (post-decay),
    for pulses n_washout .. T-1.
    """
    n = x0.shape[0]
    T = dt_seq.shape[0]
    x = x0.copy()
    decay_pw = np.exp(-pw / tau)
    states = np.empty((T - n_washout, n))
    diag = np.zeros(3)
    for t in range(T):
        _couple_and_update_nb(x, tau, dt_seq[t], decay_pw, indptr, indices, wts,
                              kappa, alpha0, alpha_min, alpha_max, gamma_, mode, diag)
        if t >= n_washout:
            states[t - n_washout, :] = x
    clip_frac = diag[0] / max(diag[2], 1.0)
    g_abs_mean = diag[1] / max(diag[2], 1.0)
    return states, clip_frac, g_abs_mean


@njit(fastmath=True, cache=True)
def run_trajectory_kappa_nb(x0, tau, dt_seq, kappa_seq, pw, indptr, indices,
                            wts, alpha0, alpha_min, alpha_max, gamma_, mode,
                            n_washout):
    """Run a trajectory with per-pulse coupling strength kappa_seq[t].

    Same physics as run_trajectory_nb but kappa varies over time (used by
    the chaos regulator homeostat, REDEM S6). Returns
    (states, clip_frac, g_abs_mean) as in run_trajectory_nb.
    """
    n = x0.shape[0]
    T = dt_seq.shape[0]
    x = x0.copy()
    decay_pw = np.exp(-pw / tau)
    states = np.empty((T - n_washout, n))
    diag = np.zeros(3)
    for t in range(T):
        _couple_and_update_nb(x, tau, dt_seq[t], decay_pw, indptr, indices, wts,
                              kappa_seq[t], alpha0, alpha_min, alpha_max,
                              gamma_, mode, diag)
        if t >= n_washout:
            states[t - n_washout, :] = x
    clip_frac = diag[0] / max(diag[2], 1.0)
    g_abs_mean = diag[1] / max(diag[2], 1.0)
    return states, clip_frac, g_abs_mean


@njit(fastmath=True, cache=True)
def run_pair_ftle_nb(x0, tau, dt_seq, pw, indptr, indices, wts, kappa,
                     alpha0, alpha_min, alpha_max, gamma_, mode, eps, renorm_every):
    """Benettin-style finite-time Lyapunov exponent (per pulse step).

    The twin trajectory starts at x0 + eps (uniform perturbation, norm
    eps * sqrt(N)) and evolves under its own full nonlinear dynamics with
    the same drive. Every renorm_every steps the separation is renormalized
    back to eps * sqrt(N). Returns (ftle_per_pulse, final_distance).
    """
    n = x0.shape[0]
    T = dt_seq.shape[0]
    x = x0.copy()
    xt = x0 + eps
    decay_pw = np.exp(-pw / tau)
    d0 = eps * np.sqrt(n)
    log_sum = 0.0
    d = d0
    diag = np.zeros(3)  # required accumulator; values unused here
    for t in range(T):
        _couple_and_update_nb(x, tau, dt_seq[t], decay_pw, indptr, indices, wts,
                              kappa, alpha0, alpha_min, alpha_max, gamma_, mode, diag)
        _couple_and_update_nb(xt, tau, dt_seq[t], decay_pw, indptr, indices, wts,
                              kappa, alpha0, alpha_min, alpha_max, gamma_, mode, diag)
        if (t + 1) % renorm_every == 0:
            acc = 0.0
            for i in range(n):
                diff = xt[i] - x[i]
                acc += diff * diff
            d = np.sqrt(acc)
            if d > 1e-300:
                log_sum += np.log(d / d0)
                scale = d0 / d
                for i in range(n):
                    xt[i] = x[i] + (xt[i] - x[i]) * scale
            else:
                # numerically fully contracted; count one window of maximal
                # observed contraction to avoid log(0), then reset the twin
                log_sum += np.log(1e-300 / d0)
                for i in range(n):
                    xt[i] = x[i] + eps
    ftle = log_sum / T
    return ftle, d


# ========================== Self test ==========================

def self_test():
    """Consistency checks for the recurrent substrate.

    1. Uncoupled trajectory must reproduce apply_pulse_sequence_vec
       (the existing CORE) to numerical precision.
    2. Topology CSR builders: degree sanity for every topology.
    3. Parallel-system FTLE must be negative and bounded (contraction).
    4. Deterministic replay: identical inputs give identical outputs.
    Returns a list of (check_name, passed, detail) tuples.
    """
    from shallow_trap_array_simulator import apply_pulse_sequence_vec

    results = []

    # --- check 1: uncoupled step == existing CORE step ---
    n, cv, seed = 64, 0.2, 3
    tau = gen_tau_vec(n, cv, tau0, seed=seed)
    x0 = preprogram_vec(ALPHA0, tau)
    dt_const = 10e-6
    n_pulses = 20
    x_ref = apply_pulse_sequence_vec(x0, tau, ALPHA0, dt_const, n_pulses)
    dt_seq = np.full(n_pulses, dt_const)
    indptr, indices, wts = build_topology_csr("parallel", n)
    states, _, _ = run_trajectory_nb(x0, tau, dt_seq, PW, indptr, indices, wts,
                                     0.0, ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma,
                                     COUPLING_NONE, 0)
    err = float(np.max(np.abs(states[-1] - x_ref)))
    results.append(("uncoupled_matches_core", err < 1e-10, f"max|dx|={err:.3e}"))

    # --- check 2: topology degree sanity ---
    ok = True
    details = []
    for name, exp_deg in (("ring_bidir", 2), ("ring_unidir", 1),
                          ("hub_star", None), ("lateral_ring", 8),
                          ("random_graph", None)):
        ip, idx, w = build_topology_csr(name, n, seed=777)
        degs = np.diff(ip)
        wsum_ok = True
        for i in range(n):
            if degs[i] > 0:
                if abs(w[ip[i]:ip[i + 1]].sum() - 1.0) > 1e-9:
                    wsum_ok = False
        if name == "hub_star":
            cond = (degs[0] == 0) and np.all(degs[1:] == 1) and wsum_ok
        elif name == "random_graph":
            mean_deg = degs.mean()
            cond = (0 < degs.max() <= n - 1) and (mean_deg > 2) and wsum_ok
        else:
            cond = np.all(degs == exp_deg) and wsum_ok
        ok = ok and cond
        details.append(f"{name}: deg mean={degs.mean():.1f}")
    results.append(("topology_csr_sanity", ok, "; ".join(details)))

    # --- check 3: parallel FTLE is contracting ---
    T = 400
    rng = np.random.RandomState(11)
    dt_rand = rng.uniform(2e-6, 20e-6, T)
    ftle, _ = run_pair_ftle_nb(x0, tau, dt_rand, PW, indptr, indices, wts,
                               0.0, ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma,
                               COUPLING_NONE, 1e-8, 10)
    cond = (-1.0 < ftle < -0.001)
    results.append(("parallel_ftle_negative", cond, f"ftle={ftle:.4f}/pulse"))

    # --- check 4: determinism ---
    ip2, idx2, w2 = build_topology_csr("ring_bidir", n)
    s1, c1, g1 = run_trajectory_nb(x0, tau, dt_rand, PW, ip2, idx2, w2, 0.1,
                                   ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma,
                                   COUPLING_CONTRAST_SELF, 0)
    s2, c2, g2 = run_trajectory_nb(x0, tau, dt_rand, PW, ip2, idx2, w2, 0.1,
                                   ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma,
                                   COUPLING_CONTRAST_SELF, 0)
    det_err = float(np.max(np.abs(s1 - s2)))
    cond = (det_err == 0.0) and (c1 == c2) and (g1 == g2)
    results.append(("deterministic_replay", cond, f"max|dx|={det_err:.3e}"))

    return results


if __name__ == "__main__":
    print("=" * 64)
    print("recurrent_substrate self-test")
    print(f"numba available: {_HAS_NUMBA}")
    print("=" * 64)
    all_ok = True
    for name, ok, detail in self_test():
        status = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{status}] {name}: {detail}")
    print("=" * 64)
    print("ALL PASS" if all_ok else "SOME CHECKS FAILED")
    sys.exit(0 if all_ok else 1)
