#!/usr/bin/env python3
"""
S31: Char-bigram oracle for the real-text benchmark (Paper D follow-up).
=============================================================================
Type:           PAPER
Paper Section:  Paper D, Sec. P4 (real-text transfer) follow-up
Experiment:     The honest scope of the input-path RLS readout is
                first-order: it can at best reach the char-bigram ceiling
                of each source. S23 reports ppl ~12-13 for the SSM arms
                and states this is the bigram regime, but no oracle was
                run. This script computes the count-based bigram-model
                ceiling on the EXACT S23 protocol (same per-seed streams,
                same per-seed reference/holdout windows), under two fits:

  fit_full : P(c'|c) estimated from the ENTIRE source book (the
             information-theoretic first-order ceiling; the readout could
             never beat this with finite windows)
  fit_ref  : P(c'|c) estimated from the same per-seed 1500-char
             reference windows the M3 metadata uses (REF_LEN, s23
             ref_windows) - the ceiling FAIR to the readout's training
             signal

  Metrics mirror S23 verbatim: stream ppl (predict with the active
  source's model) and forgetting ppl (500-char held-out window of the
  previous source, domain-matched model). Uniform baseline ppl = 32.

  Verdict rule: if the SSM-bare stream ppl (~13.3) is at or above the
  fit_ref oracle, the readout has reached its first-order ceiling and
  the "higher-order structure is out of reach" scope statement is
  quantitatively confirmed; if the oracle is clearly lower, the scope
  statement needs revision.

Output files:
  data/s31_char_bigram_oracle_v1.csv    (per seed, both fits)
  data/s31_char_bigram_oracle_v1.json   (params + aggregates)

Usage: python s31_char_bigram_oracle.py [--quick]
"""
import os
import sys
import time
import csv
import json
from collections import Counter

os.environ.setdefault('PYTHONUNBUFFERED', '1')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s18_llm_drift_gate import VOCAB, HOLDOUT_LEN
from s20_ssm_m3_routing import REF_LEN
from s23_ssm_p4_realtext import (load_corpus, gen_real_stream,
                                 ref_windows, SEG_LEN, N_SEGMENTS)

N_SEEDS = 10
ALPHA = 1.0                 # Laplace smoothing for the bigram model

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's31_char_bigram_oracle_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's31_char_bigram_oracle_v1.json')


def fit_bigram(tokens):
    """P(c'|c) with add-alpha smoothing; returns log P (V, V)."""
    V = VOCAB
    counts = np.zeros((V, V))
    for i in range(len(tokens) - 1):
        counts[tokens[i], tokens[i + 1]] += 1.0
    row_sums = counts.sum(axis=1, keepdims=True)
    prob = (counts + ALPHA) / (row_sums + ALPHA * V)
    return np.log(np.clip(prob, 1e-12, 1.0))


def stream_ppl_oracle(stream, domains, log_p_by_domain):
    ces = []
    for t in range(1, len(stream)):
        c, c_next = stream[t - 1], stream[t]
        ces.append(-log_p_by_domain[domains[t]][c, c_next])
    return float(np.exp(np.mean(ces)))


def forgetting_ppl_oracle(sources, log_p_by_domain, seed):
    """Mirror S23: per switch, a random 500-char window of the PREVIOUS
    source, scored by that source's model."""
    rng = np.random.RandomState(seed * 41 + 7)
    forgets = []
    for si in range(1, N_SEGMENTS):
        prev_dom = (si - 1) % 2
        src = sources[prev_dom]
        start = int(rng.randint(0, len(src) - HOLDOUT_LEN))
        hold = src[start:start + HOLDOUT_LEN]
        ces = [-log_p_by_domain[prev_dom][hold[i], hold[i + 1]]
               for i in range(HOLDOUT_LEN - 1)]
        forgets.append(float(np.exp(np.mean(ces))))
    return float(np.mean(forgets)) if forgets else float('nan')


def main():
    quick = '--quick' in sys.argv
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S31 char-bigram oracle "
          f"(quick={quick})")

    alice, dickens, mapping, unk = load_corpus()
    sources = [alice, dickens]
    n_seeds = 1 if quick else N_SEEDS

    # Full-book fits (information-theoretic ceiling)
    log_full = [fit_bigram(alice), fit_bigram(dickens)]

    rows = []
    for s in range(n_seeds):
        stream, domains, _ = gen_real_stream(s, sources)
        windows = ref_windows(s, sources)   # per-seed 1500-char windows
        log_ref = [fit_bigram(w) for w in windows]

        row = {'seed': s,
               'stream_ppl_full': stream_ppl_oracle(stream, domains, log_full),
               'stream_ppl_ref': stream_ppl_oracle(stream, domains, log_ref),
               'forgetting_ppl_full': forgetting_ppl_oracle(
                   sources, log_full, s),
               'forgetting_ppl_ref': forgetting_ppl_oracle(
                   sources, log_ref, s)}
        rows.append(row)
        if not quick:
            print(f"[{time.strftime('%H:%M:%S')}] seed {s}: "
                  f"stream full/ref {row['stream_ppl_full']:.3f}/"
                  f"{row['stream_ppl_ref']:.3f}", flush=True)

    os.makedirs(DATA_DIR, exist_ok=True)
    out_csv = CSV_PATH if not quick else CSV_PATH.replace('.csv', '_quick.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def agg(name):
        vals = np.array([r[name] for r in rows], dtype=float)
        return float(np.mean(vals)), float(np.std(vals))

    agg_out = {}
    for m in ['stream_ppl_full', 'stream_ppl_ref',
              'forgetting_ppl_full', 'forgetting_ppl_ref']:
        mu, sd = agg(m)
        agg_out[m + '_mean'] = mu
        agg_out[m + '_std'] = sd

    print("\n" + "=" * 96)
    print("S31 RESULTS (Paper D real-text oracle): char-bigram ceiling")
    print("=" * 96)
    print(f"  uniform baseline ppl: {VOCAB}")
    for m in ['stream_ppl_full', 'stream_ppl_ref',
              'forgetting_ppl_full', 'forgetting_ppl_ref']:
        print(f"  {m:>22}: {agg_out[m + '_mean']:8.3f} +/- "
              f"{agg_out[m + '_std']:.3f}")
    print("-" * 96)
    print("  S23 committed: SSM-bare stream 13.34, forgetting 12.31; "
          "SSM-REDEM 12.07/11.85")
    print("  Verdict: readout stream 13.3 vs oracle (ref) -> "
          "first-order ceiling reached iff readout >= oracle")

    params = {
        'experiment': 'char-bigram oracle on the S23 real-text protocol',
        'protocol': 'mirrors S23 verbatim (per-seed streams, reference '
                    'windows REF_LEN=1500, 500-char holdouts)',
        'fits': {'full': 'P(c\'|c) from the entire source book',
                 'ref': 'P(c\'|c) from the per-seed reference windows '
                        '(the metadata\'s training signal)'},
        'alpha': ALPHA, 'vocab': VOCAB, 'n_seeds': n_seeds,
        's23_reference': {'SSM-bare': '13.34 stream / 12.31 forget',
                          'SSM-REDEM': '12.07 / 11.85'},
        'quick': bool(quick),
    }
    out_json = JSON_PATH if not quick else JSON_PATH.replace('.json', '_quick.json')
    with open(out_json, 'w') as f:
        json.dump({'params': params, 'aggregates': agg_out, 'rows': rows},
                  f, indent=2)

    print(f"\nCSV : {out_csv}")
    print(f"JSON: {out_json}")
    print(f"[{time.strftime('%H:%M:%S')}] DONE, total {time.time() - t_start:.1f}s")


if __name__ == '__main__':
    main()
