#!/usr/bin/env python3
"""
Online readouts for the recurrent substrate (REDEM S2).
=============================================================================
Type:           CORE
Paper Section:  New-algorithm project Step S2
Experiment:     Online learning readout: RLS with forgetting factor

Provides:
  * OnlineRLS       : recursive least squares with exponential forgetting,
                      multi-output, inverse-covariance trace cap. This is the
                      minimal "training == inference" readout: every pulse
                      updates the weights from the live prediction error.
  * ridge_fit       : offline multi-output ridge (numpy), used as the static
                      baseline and for S1-style memory capacity fits.
  * sliding_metrics : vectorized running-mean helpers for accuracy/MSE curves.

NOTE: RLS update uses numba @njit for the O(N^2) inner loop to eliminate
per-step Python overhead and temporary array allocations.
"""
import numpy as np

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return lambda f: f


@njit(fastmath=True, cache=True)
def _rls_update_nb(P, W, x, y, forgetting, inv_forgetting, reg,
                   eye, kz_work, z_work, e_work, trace_cap,
                   n_features, n_outputs):
    """In-place RLS update (numba). Modifies P, W, kz_work, z_work, e_work."""
    for i in range(n_features):
        s = 0.0
        for j in range(n_features):
            s += P[i, j] * x[j]
        z_work[i] = s
    dot_xz = 0.0
    for i in range(n_features):
        dot_xz += x[i] * z_work[i]
    denom = forgetting + dot_xz
    for j in range(n_outputs):
        s = 0.0
        for i in range(n_features):
            s += x[i] * W[i, j]
        e_work[j] = y[j] - s
    for i in range(n_features):
        ki = z_work[i] / denom
        for j in range(n_outputs):
            W[i, j] += ki * e_work[j]
        for j in range(n_features):
            kz_val = ki * z_work[j]
            P[i, j] = (P[i, j] - kz_val) * inv_forgetting + reg * eye[i, j]
    tr = 0.0
    for i in range(n_features):
        tr += P[i, i]
    if tr > trace_cap:
        scale = trace_cap / tr
        for i in range(n_features):
            for j in range(n_features):
                P[i, j] *= scale


class OnlineRLS:
    """Recursive least squares readout with exponential forgetting.

    Update (standard RLS with Tikhonov regularization):
        z      = P x
        k      = z / (forgetting + x^T z)
        e      = y - x^T W
        W     += k e^T
        P     := (P - k z^T) / forgetting + reg * I
    The `reg` term keeps P bounded away from singularity when features are
    near-collinear (e.g., homogenized substrate states at high coupling).
    A trace cap additionally prevents covariance blow-up.

    x is expected to be a (n_features,) feature vector (bias column
    included by the caller); y is (n_outputs,).
    """

    def __init__(self, n_features, n_outputs, forgetting=0.995,
                 init_cov=1.0, trace_cap=1e8, reg=1e-4):
        if not 0.0 < forgetting <= 1.0:
            raise ValueError("forgetting must be in (0, 1]")
        self.n_features = int(n_features)
        self.n_outputs = int(n_outputs)
        self.forgetting = float(forgetting)
        self.trace_cap = float(trace_cap)
        self.reg = float(reg)
        self.P = np.eye(n_features, dtype=np.float64) * float(init_cov)
        self.W = np.zeros((n_features, n_outputs), dtype=np.float64)
        self._eye = np.eye(n_features, dtype=np.float64)
        self._inv_forgetting = 1.0 / float(forgetting)
        self._kz = np.empty((n_features, n_features), dtype=np.float64)
        self._z = np.empty(n_features, dtype=np.float64)
        self._e = np.empty(n_outputs, dtype=np.float64)
        self.steps = 0

    def predict(self, x):
        """Readout output for feature vector x (no weight update)."""
        return x @ self.W

    def update(self, x, y):
        """Online update from one (x, y) pair."""
        x = np.ascontiguousarray(x, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        if _HAS_NUMBA:
            _rls_update_nb(self.P, self.W, x, y, self.forgetting,
                           self._inv_forgetting, self.reg, self._eye,
                           self._kz, self._z, self._e, self.trace_cap,
                           self.n_features, self.n_outputs)
            self.steps += 1
            return self._e.copy()
        z = self.P @ x
        denom = self.forgetting + float(x @ z)
        k = z / denom
        e = y - (x @ self.W)
        self.W += np.outer(k, e)
        np.outer(k, z, out=self._kz)
        self.P -= self._kz
        np.multiply(self.P, self._inv_forgetting, out=self.P)
        if self.reg > 0.0:
            self.P += self.reg * self._eye
        tr = np.trace(self.P)
        if tr > self.trace_cap:
            self.P *= (self.trace_cap / tr)
        self.steps += 1
        return e

    def fit_stream(self, X, Y, n_warmup=0):
        """Fit on a stream of feature rows X (T, n) and targets Y (T, m).

        ONLINE protocol: predict BEFORE the update at each step (the
        prediction at time t never uses the target at time t), then update.
        Returns (errors (T, m), predictions (T, m)); the first n_warmup
        steps' errors are NaN-marked (transient, weight initialization).
        """
        T = X.shape[0]
        errs = np.full((T, self.n_outputs), np.nan)
        preds = np.empty((T, self.n_outputs))
        for t in range(T):
            pred = self.predict(X[t])
            preds[t] = pred
            e = self.update(X[t], Y[t])
            if t >= n_warmup:
                errs[t] = e
        return errs, preds


class ThreeFactorReadout:
    """Three-factor online readout (M1 eligibility + M2 gating, S3).

    Two modes:
      'reward' : sigmoid output o = sigmoid(w x) in (0,1); eligibility
                 e += x * o accumulates during inference; consolidate(R)
                 applies w += eta * R * e with R a scalar reward (+1/-1).
                 Learns from sparse / delayed reward only. NOTE: a pure
                 scalar reward carries no class information, so this rule
                 cannot credit-assign through mapping inversion (see the
                 S3 gate findings).
      'error'  : linear output yhat = w x; eligibility is the input trace
                 e = elig_decay * e + x; consolidate(delta) applies
                 w += eta * delta * e (delta rule with eligibility trace;
                 elig_decay=0 reduces to plain LMS). Dense supervision.

    Eligibility is zeroed on consolidation (reset=True, default).
    """

    def __init__(self, n_features, mode='reward', learning_rate=0.05,
                 elig_decay=0.9, seed=0):
        self.n_features = int(n_features)
        self.mode = mode
        if mode not in ('reward', 'error'):
            raise ValueError(f"unknown mode: {mode}")
        self.learning_rate = float(learning_rate)
        self.elig_decay = float(elig_decay)
        self.rng = np.random.RandomState(seed)
        scale = 0.1 if mode == 'error' else 0.5
        self.w = self.rng.uniform(-scale, scale, n_features)
        self.e = np.zeros(n_features)

    def predict(self, x):
        """Readout output (sigmoid for 'reward', linear for 'error')."""
        z = float(x @ self.w)
        if self.mode == 'reward':
            return 1.0 / (1.0 + np.exp(-z))
        return z

    def observe(self, x):
        """Accumulate eligibility during inference (no consolidation)."""
        if self.mode == 'reward':
            o = self.predict(x)
            self.e = self.elig_decay * self.e + np.asarray(x) * o
            return o
        self.e = self.elig_decay * self.e + np.asarray(x)
        return self.predict(x)

    def consolidate(self, third_factor, reset=True):
        """Apply the gating factor: w += eta * third_factor * e."""
        self.w += self.learning_rate * float(third_factor) * self.e
        if reset:
            self.e[:] = 0.0

    def fit_stream(self, X, mode='dense', targets=None, n_warmup=0):
        """Dense-supervision pass: predict, accumulate eligibility, then
        consolidate with the per-pulse error delta = y - yhat.
        Returns (predictions (T,), errors (T,) with NaN in warmup)."""
        T = X.shape[0]
        y = np.asarray(targets, dtype=np.float64)
        preds = np.empty(T)
        errs = np.full(T, np.nan)
        for t in range(T):
            preds[t] = self.predict(X[t])
            delta = y[t] - preds[t]
            if self.mode == 'reward':
                self.e = self.elig_decay * self.e + X[t] * preds[t]
            else:
                self.e = self.elig_decay * self.e + X[t]
            self.consolidate(delta, reset=True)
            if t >= n_warmup:
                errs[t] = delta
        return preds, errs


def ridge_fit(Xtr, Ytr, Xte=None, Yte=None, ridge_lambda=1.0):
    """Offline multi-output ridge regression (with intercept via Y centering).

    Xtr (n1, p), Ytr (n1, m); features standardized with training stats and
    targets centered with training means (the intercept is absorbed into the
    centering, so prediction = Xs @ W + ymu). Y is left on its raw scale.
    Returns dict with W, mu, sd, ymu, pred_tr, pred_te (or None).
    """
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd[sd < 1e-12] = 1.0
    Xs = (Xtr - mu) / sd
    ymu = Ytr.mean(axis=0)
    Yc = Ytr - ymu
    A = Xs.T @ Xs + ridge_lambda * np.eye(Xs.shape[1])
    B = Xs.T @ Yc
    W = np.linalg.solve(A, B)
    pred_tr = Xs @ W + ymu
    pred_te = None
    if Xte is not None:
        pred_te = ((Xte - mu) / sd) @ W + ymu
    return {'W': W, 'mu': mu, 'sd': sd, 'ymu': ymu,
            'pred_tr': pred_tr, 'pred_te': pred_te}


def memory_capacity_heldout(obs, dt_seq, k_max=50, ridge_lambda=1.0):
    """Held-out Jaeger memory capacity (70/30 chronological split, k_max
    leakage buffer). obs (T, N), dt_seq (T,) the driving intervals.
    Returns mc_total_test = sum of corr^2 over lags 1..k_max on the test
    segment (the S1 convention). Shared by the S6/S7 substrate-quality
    probes.
    """
    S, n = obs.shape
    X = obs[k_max:, :]
    T = X.shape[0]
    n_train = int(0.7 * T)
    mu = X[:n_train].mean(axis=0)
    sd = X[:n_train].std(axis=0)
    sd[sd < 1e-12] = 1.0
    Xs = (X - mu) / sd
    Y = np.empty((T, k_max + 1))
    for k in range(k_max + 1):
        Y[:, k] = dt_seq[k_max + np.arange(T) - k]
    ymu = Y[:n_train].mean(axis=0)
    Yc = Y - ymu
    Xtr, Xte = Xs[:n_train], Xs[n_train + k_max:]
    Ytr, Yte = Yc[:n_train], Yc[n_train + k_max:]
    A = Xtr.T @ Xtr + ridge_lambda * np.eye(n)
    W = np.linalg.solve(A, Xtr.T @ Ytr)
    Pte = Xte @ W
    pc2 = []
    for k in range(k_max + 1):
        a = Pte[:, k] - Pte[:, k].mean()
        b = Yte[:, k]
        denom = np.sqrt(float((a * a).sum()) * float((b * b).sum()))
        r = float((a * b).sum()) / denom if denom > 0 else 0.0
        pc2.append(r * r)
    return float(np.sum(pc2[1:]))


def running_mean_accuracy(preds, targets, window=200):
    """Running mean accuracy for binary targets, vectorized.

    preds: (T,) real-valued; target: (T,) int 0/1. Accuracy at t uses the
    window [t-window+1, t] of hard-thresholded predictions.
    Returns (T,) accuracy array (first window-1 entries NaN).
    """
    T = len(targets)
    hits = ((preds > 0.5).astype(np.float64) == targets).astype(np.float64)
    cum = np.cumsum(np.concatenate([[0.0], hits]))
    run = (cum[window:] - cum[:-window]) / window
    out = np.full(T, np.nan)
    out[window - 1:] = run
    return out


def running_mean_mse(preds, targets, window=200):
    """Running mean squared error, vectorized (same window convention)."""
    err2 = (np.asarray(preds) - np.asarray(targets)) ** 2
    cum = np.cumsum(np.concatenate([[0.0], err2]))
    run = (cum[window:] - cum[:-window]) / window
    out = np.full(len(targets), np.nan)
    out[window - 1:] = run
    return out


if __name__ == "__main__":
    # quick self-test: RLS tracks a drifting linear target
    rng = np.random.RandomState(0)
    T = 4000
    x = rng.randn(T, 5)
    w_true = np.linspace(1.0, -1.0, 5)
    y = x @ w_true
    rls = OnlineRLS(5, 1, forgetting=0.99, init_cov=100.0)
    errs, preds = rls.fit_stream(x, y[:, None], n_warmup=200)
    final_mse = float(np.mean(errs[-500:] ** 2))
    assert final_mse < 1e-6, f"RLS tracking failed: final MSE={final_mse}"
    print(f"OnlineRLS self-test PASS (final MSE={final_mse:.2e})")
