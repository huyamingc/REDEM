#!/usr/bin/env python3
"""
S35: P1 readout boundary probes (Paper D)
=============================================================================
Type:           PAPER
Paper §:        Paper D §P1 (readout boundary analysis)
Experiment:     Readout-boundary probes on the two-domain protocol:
                (1) full-window in-sample ridge oracle on the whitened
                    state features (must reproduce the s19 CV-whiten
                    oracle values);
                (2) first-half-window in-sample oracle (window
                    dependence of the same instrument);
                (3) nested-feature check: oracle([state; proj; 1]) vs
                    oracle([proj; 1]) (metric monotonicity);
                (4) out-of-sample linear decoding of the current token
                    from the fast channels (tau <= 8, tau <= 30) with
                    the slow-channel (tau >= 100) contrast.
Host / task / metrics: s19 verbatim (sample_substrate 'CV',
                gen_drift_stream, whiten_scale, CE with clip 1e-12).
Outputs:        data/s35_readout_boundary_probe_v1.csv, .json
Dependencies:   numpy, torch (via the s19 module import).
=============================================================================
"""
import os
import sys
import json
import time
from multiprocessing import Pool, cpu_count

os.environ.setdefault('PYTHONUNBUFFERED', '1')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s19_ssm_rls_readout as s19

N_SEEDS = s19.N_SEEDS
TAU_CUT_FAST = 8.0          # paper M3 fast-channel definition (tau <= 8)
TAU_CUT_FAST30 = 30.0       # robustness check with a wider fast band
TAU_CUT_SLOW = 100.0        # slow-channel contrast
CSV_OUT = os.path.join(s19.DATA_DIR, 's35_readout_boundary_probe_v1.csv')
JSON_OUT = os.path.join(s19.DATA_DIR, 's35_readout_boundary_probe_v1.json')


def ppl_from_pred(yhat, targets):
    """Stream perplexity of a linear readout: CE on the clipped target
    component, no softmax (s19 metric verbatim)."""
    p = np.clip(np.take_along_axis(yhat, targets[:, None], axis=1).ravel(),
                1e-12, 1.0)
    return float(np.exp(np.mean(-np.log(p))))


def ridge_oracle_ppl(X, Y, targets, delta):
    """In-sample ridge oracle (s19 formula verbatim): W = Y^T X (X^T X +
    delta I)^{-1}; report the clip-log ppl on the same rows."""
    F = X.shape[1]
    XtX = X.T @ X + delta * np.eye(F)
    Wr = Y.T @ X @ np.linalg.inv(XtX)
    return ppl_from_pred(X @ Wr.T, targets)


def run_probe_seed(seed):
    """All probes for one seed (top-level for Pool). Returns a dict."""
    import torch  # imported inside the worker; s19 needs it
    A, B = s19.sample_substrate(seed, 'CV')
    scale = s19.whiten_scale(A)
    stream, domains = s19.gen_drift_stream(seed)
    T = stream.shape[0]
    N, V = s19.N_STATE, s19.VOCAB
    tau = -1.0 / np.log(np.asarray(A, dtype=np.float64))

    # Whitened state features + bias (s19 build verbatim)
    h = torch.zeros(N, dtype=torch.float64)
    hs = np.empty((T - 1, N + 1))
    hs[:, -1] = 1.0
    Bp = np.empty((T - 1, N))
    for t in range(1, T):
        h = A * h + B[:, stream[t - 1]]
        hs[t - 1, :N] = (h * scale).numpy()
        Bp[t - 1] = B[:, stream[t - 1]].numpy()

    Y = np.zeros((T - 1, V))
    Y[np.arange(T - 1), stream[1:]] = 1.0
    Eprev = np.zeros((T - 1, V))
    Eprev[np.arange(T - 1), stream[:-1]] = 1.0
    tgt = stream[1:]
    n_tr = (T - 1) // 2
    itr, ite = np.arange(n_tr), np.arange(n_tr, T - 1)

    # (1) full-window oracle on the state features (s19 formula)
    oracle_full = ridge_oracle_ppl(hs, Y, tgt, s19.RLS_DELTA)
    # (2) half-window in-sample oracle (same formula, first half)
    oracle_half = ridge_oracle_ppl(hs[itr], Y[itr], tgt[itr],
                                   s19.RLS_DELTA)
    # (3) nested check: [state; proj; 1] vs [proj; 1]
    Xskip = np.hstack([hs[:, :N], Bp, np.ones((T - 1, 1))])
    Xproj = np.hstack([Bp, np.ones((T - 1, 1))])
    oracle_skip = ridge_oracle_ppl(Xskip, Y, tgt, s19.RLS_DELTA)
    oracle_proj = ridge_oracle_ppl(Xproj, Y, tgt, s19.RLS_DELTA)

    # (4) out-of-sample token decoding from fast / slow channels
    def decode_acc(Xf):
        R, *_ = np.linalg.lstsq(Xf[itr], Eprev[itr], rcond=None)
        return float(np.mean(np.argmax(Xf[ite] @ R, axis=1)
                             == stream[:-1][ite]))

    fast8 = tau <= TAU_CUT_FAST
    fast30 = tau <= TAU_CUT_FAST30
    slow = tau >= TAU_CUT_SLOW
    dec_fast8 = decode_acc(hs[:, np.where(fast8)[0]])
    dec_fast30 = decode_acc(hs[:, np.where(fast30)[0]])
    dec_slow = decode_acc(hs[:, np.where(slow)[0]])

    return dict(seed=seed, oracle_full=oracle_full, oracle_half=oracle_half,
                oracle_skip=oracle_skip, oracle_proj=oracle_proj,
                decode_fast8_te=dec_fast8, decode_fast30_te=dec_fast30,
                decode_slow_te=dec_slow, n_fast8=int(fast8.sum()),
                n_fast30=int(fast30.sum()), n_slow=int(slow.sum()))


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workers', type=int, default=4,
                    help='Pool size (default 4; each worker imports torch, '
                         'so keep this small on RAM-limited machines)')
    args = ap.parse_args()

    t0 = time.time()
    print(f'[s35] START {time.strftime("%H:%M:%S")} | {N_SEEDS} seeds, '
          f'probes: full/half oracle, skip/proj nested check, '
          f'fast/slow token decode', flush=True)

    n_proc = max(1, min(args.workers, N_SEEDS))
    with Pool(n_proc) as pool:
        rows = pool.map(run_probe_seed, range(N_SEEDS), chunksize=1)

    rows.sort(key=lambda r: r['seed'])

    # CSV
    cols = ['seed', 'oracle_full', 'oracle_half', 'oracle_skip',
            'oracle_proj', 'decode_fast8_te', 'decode_fast30_te',
            'decode_slow_te', 'n_fast8', 'n_fast30', 'n_slow']
    with open(CSV_OUT, 'w', newline='', encoding='utf-8') as f:
        f.write(','.join(cols) + '\n')
        for r in rows:
            f.write(','.join(str(r[c]) for c in cols) + '\n')

    # Cross-check against the committed s19 CV-whiten oracle values
    s19_vals = {}
    try:
        with open(s19.CSV_PATH, 'r', encoding='utf-8') as f:
            for line in f.read().splitlines()[1:]:
                parts = line.split(',')
                if parts[0] == 'CV-whiten':
                    s19_vals[int(parts[1])] = float(parts[5])
    except OSError:
        pass
    diff = [abs(r['oracle_full'] - s19_vals[r['seed']])
            for r in rows if r['seed'] in s19_vals]

    def agg(key):
        vals = [r[key] for r in rows]
        return dict(mean=float(np.mean(vals)), min=float(np.min(vals)),
                    max=float(np.max(vals)))

    summary = {
        'params': {
            'spectrum': 'CV', 'n_state': s19.N_STATE, 'vocab': s19.VOCAB,
            'tau_min': s19.TAU_MIN, 'tau_max': s19.TAU_MAX,
            'tau_cut_fast8': TAU_CUT_FAST, 'tau_cut_fast30': TAU_CUT_FAST30,
            'tau_cut_slow': TAU_CUT_SLOW,
            'seg_len': s19.SEG_LEN, 'n_segments': s19.N_SEGMENTS,
            'rls_delta': s19.RLS_DELTA, 'n_seeds': N_SEEDS,
            'metric': 'stream CE on clipped linear outputs (no softmax)',
            's19_csv_cross_check_max_abs_diff': (
                float(np.max(diff)) if diff else None),
        },
        'aggregates': {k: agg(k) for k in
                       ['oracle_full', 'oracle_half', 'oracle_skip',
                        'oracle_proj', 'decode_fast8_te',
                        'decode_fast30_te', 'decode_slow_te']},
        'per_seed': rows,
    }
    with open(JSON_OUT, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f'[s35] DONE {time.strftime("%H:%M:%S")} '
          f'({time.time() - t0:.1f}s) | CSV {CSV_OUT}', flush=True)
    print(f'[s35] full-window oracle mean {summary["aggregates"]["oracle_full"]["mean"]:.2f} '
          f'(s19 cross-check max |diff| {summary["params"]["s19_csv_cross_check_max_abs_diff"]:.2e})',
          flush=True)
    print(f'[s35] half-window oracle mean {summary["aggregates"]["oracle_half"]["mean"]:.2f} | '
          f'skip {summary["aggregates"]["oracle_skip"]["mean"]:.2f} vs proj '
          f'{summary["aggregates"]["oracle_proj"]["mean"]:.2f}', flush=True)
    print(f'[s35] token decode (out-of-sample): fast8 '
          f'{summary["aggregates"]["decode_fast8_te"]["min"]:.3f}..'
          f'{summary["aggregates"]["decode_fast8_te"]["max"]:.3f} | slow '
          f'{summary["aggregates"]["decode_slow_te"]["min"]:.3f}..'
          f'{summary["aggregates"]["decode_slow_te"]["max"]:.3f} '
          f'(chance 0.031)', flush=True)


if __name__ == '__main__':
    main()
