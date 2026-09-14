#!/usr/bin/env python3
"""Per-token CE dump for same-token-set rescoring (Paper D metric audit).

Type: EXPLORE (IO helper; no training logic).
"""
from pathlib import Path
import numpy as np

PER_TOKEN_DIR = Path(__file__).resolve().parent.parent / 'data' / 'per_token'


def save_stream_and_holdout(script_tag, arm, seed, extra_key,
                            stream_ce, stream_ce_unc,
                            hold_ce=None, hold_ce_unc=None):
    """Write per-token arrays for one SSM run.

    stream_ce:     reported clipped CE over t=0..T-1 (index 0 unused/nan)
    stream_ce_unc: unclipped CE; nan where y_hat <= 0
    hold_ce / hold_ce_unc: 1-D concatenation over holdout windows
    """
    PER_TOKEN_DIR.mkdir(parents=True, exist_ok=True)
    safe_arm = str(arm).replace('/', '_').replace(' ', '_')
    safe_extra = str(extra_key).replace('/', '_').replace(' ', '_')
    path = PER_TOKEN_DIR / f'{script_tag}__{safe_arm}__{safe_extra}__s{seed}.npz'
    payload = {
        'stream_ce': np.asarray(stream_ce, dtype=np.float64),
        'stream_ce_unc': np.asarray(stream_ce_unc, dtype=np.float64),
    }
    if hold_ce is not None:
        payload['hold_ce'] = np.asarray(hold_ce, dtype=np.float64)
    if hold_ce_unc is not None:
        payload['hold_ce_unc'] = np.asarray(hold_ce_unc, dtype=np.float64)
    np.savez_compressed(path, **payload)
    return path


def load_pair(script_tag, arm_a, arm_b, seed, extra_key='0'):
    """Load two arms' per-token arrays for the same seed/stream."""
    safe_a = str(arm_a).replace('/', '_').replace(' ', '_')
    safe_b = str(arm_b).replace('/', '_').replace(' ', '_')
    safe_extra = str(extra_key).replace('/', '_').replace(' ', '_')
    pa = PER_TOKEN_DIR / f'{script_tag}__{safe_a}__{safe_extra}__s{seed}.npz'
    pb = PER_TOKEN_DIR / f'{script_tag}__{safe_b}__{safe_extra}__s{seed}.npz'
    if not pa.exists() or not pb.exists():
        return None
    A = np.load(pa)
    B = np.load(pb)
    return A, B


def same_token_set_mean(ce_unc_a, ce_unc_b):
    """Mean CE of each arm on tokens where BOTH arms have y_hat > 0.

    Returns (mean_a, mean_b, n_common, n_a, n_b) or None if empty.
    """
    a = np.asarray(ce_unc_a, dtype=np.float64).ravel()
    b = np.asarray(ce_unc_b, dtype=np.float64).ravel()
    n = min(a.size, b.size)
    a, b = a[1:n], b[1:n]  # drop t=0
    both = np.isfinite(a) & np.isfinite(b)
    n_common = int(both.sum())
    if n_common == 0:
        return None
    return (float(a[both].mean()), float(b[both].mean()),
            n_common, int(np.isfinite(a).sum()), int(np.isfinite(b).sum()))
