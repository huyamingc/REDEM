#!/usr/bin/env python3
"""S66 reporting helper: arm summary + paired comparisons from the s66 CSV.

Type:           EXPLORE (read-only audit of committed s66 rows; no training)
Paper Section:  Paper F Discussion / Limitations (external selective-SSM check)
Experiment:     S66 audit view

What it verifies (against data/s66_external_ssm_baseline_v1.csv):
  - per-arm 10-seed means of stream_ppl / forgetting_ppl
  - paired external-vs-F-anchor deltas, t, and seed win counts
  - lr_scale separation (matched 1.0 vs divergent 4.0), so the paper's
    "matched-lr 10.19" quote is not silently mixed with the 4x rows
  - trainable parameter totals (readout+gate+host)

Coupling: Paper F Discussion quotes Gate-C-topk 7.03 vs frozen 9.03
(0/10), joint 10.19 / forget 115.24, and the lr-pooling disclosure.
This script is the offline view of those numbers. It does not train
or emit new result artifacts.

Usage: python s66_report.py
"""
import csv
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, '..', 'data', 's66_external_ssm_baseline_v1.csv')
if not os.path.exists(CSV):
    CSV = CSV.replace('.csv', '_quick.csv')

EXT = ('X-SelSSM-full', 'X-SelSSM-frozen')
BASE = ('F-Gate-C-topk-softmax', 'F-Skip-sub-softmax', 'F-B-softmax-sgd')


def load():
    with open(CSV, newline='') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r['seed'] = int(float(r['seed']))
        r['lr_scale'] = float(r.get('lr_scale', 1.0) or 1.0)
        for k in ('stream_ppl', 'forgetting_ppl', 'oracle_ppl', 'clip_frac',
                  'neg_frac', 'floor_frac', 'forget_floor_frac'):
            r[k] = float(r[k])
        for k in ('n_params_readout', 'n_params_gate', 'n_params_host'):
            r[k] = int(float(r[k]))
    return rows


def key(r):
    return (r['arm'], r['lr_scale'], r['seed'])


def main():
    rows = load()
    print(f'file: {os.path.basename(CSV)}   rows: {len(rows)}')
    groups = {}
    for r in rows:
        groups.setdefault((r['arm'], r['lr_scale']), []).append(r)

    print('\n' + '=' * 104)
    print('ARM SUMMARY')
    print('=' * 104)
    print(f"{'arm':26s} {'lr':>5} {'n':>3} {'stream':>9} {'forget':>10} "
          f"{'clip%':>7} {'neg%':>7} {'floor%':>7} {'oracle':>8} {'npar':>7}")
    for (arm, lr) in sorted(groups, key=lambda t: (t[0], t[1])):
        rs = groups[(arm, lr)]
        npar = rs[0]['n_params_readout'] + rs[0]['n_params_gate'] \
            + rs[0]['n_params_host']
        print(f"{arm:26s} {lr:5.1f} {len(rs):3d} "
              f"{np.mean([r['stream_ppl'] for r in rs]):9.4f} "
              f"{np.mean([r['forgetting_ppl'] for r in rs]):10.4f} "
              f"{100 * np.mean([r['clip_frac'] for r in rs]):7.3f} "
              f"{100 * np.mean([r['neg_frac'] for r in rs]):7.3f} "
              f"{100 * np.mean([r['floor_frac'] for r in rs]):7.3f} "
              f"{np.mean([r['oracle_ppl'] for r in rs]):8.3f} {npar:7d}")

    print('\n' + '=' * 104)
    print('PAIRED COMPARISONS  (external - F anchor; negative = external better;'
          ' wins = seeds external is better)')
    print('=' * 104)
    for col in ('stream_ppl', 'forgetting_ppl'):
        print(f'\n-- {col} --')
        print(f"{'external':18s} {'lr':>5} {'vs F anchor':24s} {'delta':>9} "
              f"{'t':>7} {'wins':>7} {'n':>3}")
        for e in EXT:
            for lr in sorted({r['lr_scale'] for r in rows if r['arm'] == e}):
                for b in BASE:
                    a = {r['seed']: r[col] for r in rows
                         if r['arm'] == e and r['lr_scale'] == lr}
                    bb = {r['seed']: r[col] for r in rows if r['arm'] == b}
                    seeds = sorted(set(a) & set(bb))
                    if not seeds:
                        continue
                    d = np.array([a[s] - bb[s] for s in seeds])
                    n = len(d)
                    sd = d.std(ddof=1) if n > 1 else float('nan')
                    t = (d.mean() / (sd / np.sqrt(n))
                         if n > 1 and sd > 0 else float('nan'))
                    print(f'{e:18s} {lr:5.1f} {b:24s} {d.mean():+9.4f} '
                          f'{t:+7.2f} {int((d < 0).sum()):4d}/{n:<2d} {n:3d}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
