# table_check_report

claims: 121
tables parsed: 22
numeric table cells: 747

| paper | quantity | doc | computed | in_table | table_hit | prose | ok |
|---|---|---|---|---|---|---|---|
| B | full-system overall acc @ N=256 | 0.996 | 0.9962111111111112 | yes | 1.000 @ tab:bench | yes | PASS |
| B | bare-baseline overall acc @ N=256 | 0.973 | 0.973411111111111 | yes | 0.97 @ tab:neg | yes | PASS |
| B | no-homeostat ties full (acc) | 0.996 | 0.9962222222222221 | yes | 1.000 @ tab:bench | yes | PASS |
| B | full-system acc @ N=1024 | 0.997 | 0.9970222222222223 | yes | 1.000 @ tab:bench | yes | PASS |
| B | baseline acc @ N=1024 | 0.9753 | 0.9753444444444443 | yes | 0.98 @ tab:neg | yes | PASS |
| B | regulated r3 MC after 3 disturbances | 8.47 | 8.474948883555967 | yes | 8.47 @ tab:s10 | yes | PASS |
| B | fixed-kappa r3 MC | 6.41 | 6.414554704263663 | yes | 6.41 @ tab:s10 | yes | PASS |
| B | relative sequential recovery +32% | 0.32 | 0.32120611239355235 | yes | +0.30 @ Transformer drift-gate proof of concept  | yes | PASS |
| A | forgetting-kernel Pearson r | 0.97 | 0.9706856810206829 | yes | 1 @ tab:phase | yes | PASS |
| A | kappa* lateral_ring (amax 0.1) | 25.3 | 25.30001187678561 | prose | — | yes | PASS |
| A | kappa* ring_bidir (amax 0.1) | 27.4 | 27.44774101354044 | prose | — | yes | PASS |
| A | kappa* random_graph (amax 0.1) | 27.9 | 27.942895773298936 | prose | — | yes | PASS |
| A | homeostat restore after edge_prune (%) | 7.9 | 7.939536844045607 | yes | +7.9 @ tab:homeostat | yes | PASS |
| A | homeostat restore after tau_drift (%) | 18 | 18.26650841757927 | yes | +18 @ tab:homeostat | yes | PASS |
| A | homeostat restore after noise (%) | 11.8 | 11.804522490631356 | yes | +11.8 @ tab:homeostat | yes | PASS |
| A | lambda_target=0 MC gain at CV=0.1 (%) | 25 | 24.55144147223043 | yes | 25 @ tab:phase | yes | PASS |
| B | N=1024 full-vs-baseline paired t | 15.3 | 15.256005856493127 | prose | — | yes | PASS |
| B | novelty-guided rewiring MC_final | 14.59 | 14.585449704460018 | yes | 14.58 @ Transformer drift-gate proof of concept  | yes | PASS |
| B | correlation-guided rewiring MC_final | 12.43 | 12.434047269794494 | yes | 12.46 @ tab:homeostat | yes | PASS |
| B | homeostat-only r3 MC (no plasticity) | 8.47 | 8.474948883555967 | yes | 8.47 @ tab:s10 | yes | PASS |
| B | coupled rewiring r3 MC (harmful) | 5.27 | 5.271525363305836 | prose | — | yes | PASS |
| C | ESN+meta overall acc | 0.998 | 0.9979111111111111 | yes | 0.9955 @ paper_c/PAPER_C.tex | yes | PASS |
| C | ESN-fast overall acc | 0.996 | 0.995511111111111 | yes | 0.9955 @ paper_c/PAPER_C.tex | yes | PASS |
| C | REDEM-full overall acc | 0.994 | 0.9942333333333334 | yes | 0.9955 @ paper_c/PAPER_C.tex | yes | PASS |
| C | s14 paired MC diff at r1 (dual-fast) | -0.78 | -0.7785058668741548 | prose | — | yes | PASS |
| C | s14 paired MC diff at r2 (dual-fast) | -0.76 | -0.7591889315236324 | prose | — | yes | PASS |
| C | s14 paired MC diff at r3 (dual-fast) | -0.69 | -0.6931466416395877 | yes | -0.693 @ Post-disturbance recovery vs.\ metadata  | yes | PASS |
| B | online RLS drift-binary mean acc | 0.978 | 0.977939448757569 | yes | 0.98 @ tab:neg | yes | PASS |
| D | B-proj / B-lin-clip stream ppl (input path) | 11.75 | 11.754124283044472 | yes | 11.75 @ paper_d/PAPER_D.tex | yes | PASS |
| D | oracle / pooled-table ceiling stream ppl | 7.25 | 7.247809995539264 | prose | — | yes | PASS |
| D | A3-A1 forgetting diff at tau_m=200 | -2.05 | -2.0526782634737275 | yes | -2.05 @ paper_d/PAPER_D.tex | yes | PASS |
| D | A3-A1 forgetting diff at tau_m=500 | -1.87 | -1.869466006154214 | yes | -1.87 @ paper_d/PAPER_D.tex | yes | PASS |
| D | A3-A1 forgetting diff at tau_m=1000 | -1.2 | -1.2015268901595109 | yes | -1.20 @ paper_d/PAPER_D.tex | yes | PASS |
| D | A3-soft stream ppl (E1) | 8.22 | 8.218244292247023 | prose | — | yes | PASS |
| D | A3-abrupt stream ppl (E1) | 10.03 | 10.032775778209635 | yes | 10 @ paper_d/PAPER_D.tex | yes | PASS |
| D | M5 regulated whitened-state norm mean | 11.3 | 11.313732258887935 | yes | 11.29 @ tab:phase | yes | PASS |
| D | bare host whitened-state norm mean | 50.2 | 50.18823344016563 | prose | — | yes | PASS |
| D | SSM-REDEM stream ppl | 13.18 | 13.179586276981818 | yes | 13.18 @ P4 (10 seeds, four-domain irregular-swit | yes | PASS |
| D | SSM-REDEM forgetting ppl | 8.93 | 8.93112298460743 | yes | 8.91 @ paper_d/PAPER_D.tex | yes | PASS |
| D | REDEM-SSM real-text stream ppl | 12.07 | 12.07099417985203 | prose | — | yes | PASS |
| D | char-bigram full-book ceiling ppl | 10.97 | 10.973725062300073 | yes | 0011 @ paper_c/PAPER_C.tex | yes | PASS |
| D | SSM-REDEM vs bare stream diff (ppl, negative=better) | -2.25 | -2.246721214020686 | yes | -2.25 @ P4 (10 seeds, four-domain irregular-swit | yes | PASS |
| D | SSM-REDEM vs bare forgetting diff | -4.47 | -4.473469096705111 | yes | -4.47 @ P4 (10 seeds, four-domain irregular-swit | yes | PASS |
| D | SSM-REDEM vs TF-A1 stream diff | -9.28 | -9.278820106388663 | yes | -9.28 @ P4 (10 seeds, four-domain irregular-swit | yes | PASS |
| E | frozen-hypothesis memory mean acc | 0.888 | 0.8880186427476696 | yes | 0.89 @ tab:neg | yes | PASS |
| E | re-adaptation / rls_single mean acc | 0.838 | 0.8376248084218989 | prose | — | yes | PASS |
| E | frozen-snapshot retention gain (pp) | 8.7 | 8.699999999999996 | prose | — | yes | PASS |
| E | EWC whole-run accuracy change (pp) | -0.85 | -0.8519509560061345 | prose | — | yes | PASS |
| E | EWC whole-run paired t | -3.87 | -3.870591931997619 | prose | — | yes | PASS |
| E | s65 reference arm bit-matches s52 (max abs diff) | 0 | 0.0 | skip | identity | skip | PASS |
| E | cross-family ring gain paired t (corrected) | 2.45 | 2.449489742783178 | prose | — | yes | PASS |
| E | post-inversion flip acc uncoupled | 0.898 | 0.898 | yes | 0.897 @ tab:zoo | yes | PASS |
| E | post-inversion flip acc coupled | 0.939 | 0.93875 | yes | 0.938 @ tab:zoo | yes | PASS |
| E | oracle post-inversion acc uncoupled | 0.975 | 0.975 | yes | 1.000 @ tab:zoo | yes | PASS |
| E | oracle post-inversion acc coupled | 1 | 1.0 | yes | 1.000 @ tab:zoo | yes | PASS |
| E | first-flip latency (blocks after inversion) | 39 | 39.0 | skip | identity | skip | PASS |
| E | learned flip value v (post-flip contrast) | 0.504 | 0.5037621465429181 | yes | 0.505 @ tab:zoo | yes | PASS |
| E | stable-stream whole-run acc uncoupled (bit-identical no-meta) | 0.897925 | 0.8979251149468606 | yes | 0.897 @ tab:zoo | yes | PASS |
| E | stable-stream whole-run acc coupled (bit-identical no-meta) | 0.938241 | 0.9382407854074017 | yes | 0.938 @ tab:zoo | yes | PASS |
| E | R1 zoo coupled full-swap rmhl post | 0.058 | 0.05800000000000001 | yes | 0.058 @ tab:zoo | yes | PASS |
| E | R1 zoo coupled full-swap flip post | 0.942 | 0.942 | yes | 0.938 @ tab:zoo | yes | PASS |
| E | R1 zoo coupled near-inversion rmhl post | 0.053 | 0.053000000000000005 | yes | 0.053 @ tab:zoo | yes | PASS |
| E | R1 zoo coupled near-inversion flip post | 0.947 | 0.9469999999999998 | yes | 0.947 @ tab:zoo | yes | PASS |
| E | R1 zoo partial-shift flip post-window | 0.485 | 0.48475 | yes | 0.485 @ tab:zoo | yes | PASS |
| E | R1 zoo partial-shift flip steady | 0.508 | 0.508 | yes | 0.509 @ tab:neg | yes | PASS |
| E | R1 zoo partial-shift flip mean flips | 2.7 | 2.7 | yes | +2.71 @ paper_d/PAPER_D.tex | yes | PASS |
| E | corrupt-window acc uncoupled (honest negative) | 0.106 | 0.10600000000000001 | yes | 0.1 @ tab:lambda_sweep | yes | PASS |
| E | corrupt-window acc coupled (honest negative) | 0.064 | 0.06400000000000002 | yes | 0.06 @ tab:neg | yes | PASS |
| E | rule-adjudication delta pre-swap uncoupled | 0.99 | 0.99 | yes | 1.000 @ tab:zoo | yes | PASS |
| E | rule-adjudication delta steady uncoupled | 0.989 | 0.9889999999999999 | yes | 1.000 @ tab:zoo | yes | PASS |
| E | rule-adjudication RMHL post-swap uncoupled | 0.097 | 0.0965 | yes | 0.1 @ tab:lambda_sweep | yes | PASS |
| E | raw capacity probe coupled (circle) | 0.84 | 0.8414411764705884 | prose | — | yes | PASS |
| E | raw capacity probe uncoupled (circle) | 0.55 | 0.546843137254902 | prose | — | yes | PASS |
| E | RLS acc 4D shell2 (two-shell) | 0.547 | 0.5467976820568607 | prose | — | yes | PASS |
| E | RLS acc 4D shell3 (three-shell) | 0.559 | 0.559026673082153 | prose | — | yes | PASS |
| E | 4D-shell margin at N=64 | 0.034 | 0.03400000000000003 | yes | 0 @ tab:zoo | yes | PASS |
| E | 4D-shell margin at N=256 | 0.1 | 0.09949999999999992 | yes | 0.1 @ tab:lambda_sweep | yes | PASS |
| E | 4D-shell margin at N=512 | 0.107 | 0.10699999999999998 | yes | 0.1 @ tab:lambda_sweep | yes | PASS |
| E | 4D-shell margin at kappa=20 (sampled peak) | 0.1185 | 0.11850000000000005 | yes | 0.12 @ tab:phase | yes | PASS |
| E | 4D-shell margin at deployed kappa=25 | 0.0995 | 0.09949999999999992 | yes | 0.1 @ tab:lambda_sweep | yes | PASS |
| E | 2D-circle margin flat across N (N=64) | 0.401 | 0.40100000000000013 | yes | 0.4 @ tab:lambda_sweep | yes | PASS |
| E | per-segment memory gain seg1_acc (pp) | 4.2 | 4.1499999999999995 | prose | — | yes | PASS |
| E | per-segment memory gain paired t seg1_acc | 2.72 | 2.7231416669378654 | yes | +2.71 @ paper_d/PAPER_D.tex | yes | PASS |
| E | per-segment memory gain seg2_acc (pp) | 3.6 | 3.600000000000001 | prose | — | yes | PASS |
| E | per-segment memory gain paired t seg2_acc | 2.73 | 2.7308973820905384 | prose | — | yes | PASS |
| E | per-segment memory gain seg3_acc (pp) | 2.6 | 2.5500000000000003 | prose | — | yes | PASS |
| E | per-segment memory gain paired t seg3_acc | 2.02 | 2.021398875151293 | yes | +2 @ tab:multisource | yes | PASS |
| E | per-segment memory gain seg4_acc (pp) | 14.9 | 14.850000000000001 | yes | 14.87 @ tab:homeostat | yes | PASS |
| E | per-segment memory gain paired t seg4_acc | 11.5 | 11.46365971771185 | prose | — | yes | PASS |
| E | population control (no frozen snapshot) mean acc | 0.839 | 0.8387389764076278 | prose | — | yes | PASS |
| E | sense reliability ABAB SEG=500 | 0.9 | 0.9 | yes | 0.897 @ tab:zoo | yes | PASS |
| E | sense reliability ABAB SEG=250 | 0.27 | 0.27 | yes | +0.30 @ Transformer drift-gate proof of concept  | yes | PASS |
| E | sense reliability ABAB SEG=125 (cliff floor) | 0 | 0.0 | skip | identity | skip | PASS |
| E | learned snapshot gate v_snap final (dynamic) | 0.252 | 0.25199999999999995 | yes | +0.30 @ Transformer drift-gate proof of concept  | yes | PASS |
| E | memory selection fraction (dynamic) | 0.227 | 0.22695000000000004 | yes | 0.2 @ tab:lambda_sweep | yes | PASS |
| E | D2 post-swap at ratio 0.25 (fail band) | 0.486 | 0.48616124497991964 | yes | 0.485 @ tab:zoo | yes | PASS |
| E | D2 post-swap at ratio 1.0 (intermediate) | 0.561 | 0.5614381282051282 | prose | — | yes | PASS |
| E | D2 post-swap at ratio 1.25 (band interior) | 0.791 | 0.7914655046826222 | prose | — | yes | PASS |
| E | D2 post-swap at ratio 1.5 (band interior) | 0.85 | 0.8498780578480224 | prose | — | yes | PASS |
| E | D2 post-swap at ratio 1.75 (band interior) | 0.909 | 0.9094356885450701 | yes | 0.908 @ tab:neg | yes | PASS |
| E | D2 post-swap at ratio 2.0 (complete recovery) | 0.938 | 0.9375090789473683 | yes | 0.938 @ tab:zoo | yes | PASS |
| E | single-speaker linear acc | 0.938 | 0.937713248410844 | yes | 0.938 @ tab:zoo | yes | PASS |
| E | A-B-A-B linear acc (switch_every=200) | 0.9 | 0.8995859400517574 | yes | 0.897 @ tab:zoo | yes | PASS |
| E | A-B-A-B memory-recovered acc | 0.913 | 0.912894776513153 | yes | 0.9 @ paper_d/PAPER_D.tex | yes | PASS |
| E | capacity probe single-speaker | 0.949 | 0.9492999999999998 | yes | 0.947 @ tab:zoo | yes | PASS |
| E | capacity probe A-B-A-B | 0.9 | 0.9003 | yes | 0.897 @ tab:zoo | yes | PASS |
| E | minority-good majority-rule acc | 0.684 | 0.6836862139142232 | yes | 0.684 @ tab:multisource | yes | PASS |
| E | minority-good validation-selector acc | 1 | 1.0 | yes | 1.000 @ tab:zoo | yes | PASS |
| E | drifting majority-rule acc | 0.681 | 0.6808148036481495 | yes | 0.684 @ tab:multisource | yes | PASS |
| E | drifting validation-selector acc | 0.991 | 0.991306751086656 | yes | 1.000 @ tab:zoo | yes | PASS |
| E | all-bad majority-rule acc (no source to trust) | 0.494 | 0.49363357704580296 | yes | 0.494 @ tab:multisource | yes | PASS |
| F | B-softmax stream ppl | 8.65 | 8.648338148738018 | yes | 8.648 @ tab:c1c2 | yes | PASS |
| F | stream CE improvement magnitude (nats/token) | 0.307 | 0.30683700086248233 | yes | +0.30 @ Transformer drift-gate proof of concept  | yes | PASS |
| F | clip-floor park rate low (~1.6%) | 1.6 | 1.5817545419189956 | prose | — | yes | PASS |
| F | clip-floor park rate high (~10.2%) | 10.2 | 10.228901605644758 | yes | 10.23 @ tab:c1c2 | yes | PASS |
| F | simplex does not repair (stream ppl delta vs lin-clip) | 0.03 | 0.02838957053185709 | yes | 0 @ tab:c1c2 | yes | PASS |
| F | Gate-C-topk stream ppl | 7.03 | 7.032655791397405 | yes | 7.033 @ tab:c4 | yes | PASS |
| F | soft routing forgetting ppl | 7.79 | 7.788825326133671 | yes | 7.789 @ tab:c5 | yes | PASS |
| F | hard routing forgetting ppl | 6.94 | 6.939440481059913 | yes | 6.939 @ tab:c5 | yes | PASS |
| F | X-SelSSM-frozen stream ppl | 9.03 | 9.029671573863947 | yes | 9.07 @ tab:phase | yes | PASS |
| F | F Gate-C-topk stream anchor (same host/stream) | 7.03 | 7.032655791397405 | yes | 7.033 @ tab:c4 | yes | PASS |

table-cell hits: **88**; prose-only: **30**; failures: **0**

## Parsed tables

| paper | label | #cells | caption | path |
|---|---|---|---|---|
| A | tab:phase | 27 | Per-topology summary of the sweep (held-out MC; mean s.d.\ o | paper_a/PAPER_A.tex |
| A | tab:homeostat | 20 | Post-disturbance held-out MC of the -homeostat (random\_grap | paper_a/PAPER_A.tex |
| B | tab:neg | 23 | Reward-only readout learners fail at the class-interval inve | paper_b/PAPER_B.tex |
| B | tab:lambda_sweep | 40 | Criticality target sweep: post-disturbance held-out MC (5 se | paper_b/PAPER_B.tex |
| B | tab:bench | 8 | Standard-task comparison (10 seeds). REDEM and the ESN run o | paper_b/PAPER_B.tex |
| C | — | 20 |  | paper_c/PAPER_C.tex |
| C | tab:s10 | 20 | Equalization on regime\_switch (10 seeds). Accuracy from the | paper_c/PAPER_C.tex |
| C | — | 24 | Post-disturbance recovery vs.\ metadata timescale (10 seeds) | paper_c/PAPER_C.tex |
| C | — | 24 |  | paper_c/PAPER_C.tex |
| C | tab:s16b | 18 | Probe-protocol stress test (10 seeds, all four _m ; _m = | paper_c/PAPER_C.tex |
| C | — | 74 | Transformer drift-gate proof of concept (10 seeds). Left: st | paper_c/PAPER_C.tex |
| D | — | 74 |  | paper_d/PAPER_D.tex |
| D | — | 21 | P4 (10 seeds, four-domain irregular-switch protocol): the fu | paper_d/PAPER_D.tex |
| D | tab:p4 | 36 |  | paper_d/PAPER_D.tex |
| D | — | 18 |  | paper_d/PAPER_D.tex |
| D | — | 170 |  | paper_d/PAPER_D.tex |
| E | tab:zoo | 34 | The R1 regime zoo ( s44\_regime\_zoo | paper_e/PAPER_E.tex |
| E | tab:multisource | 33 | Multi-source validation (supporting experiment R8): mean acc | paper_e/PAPER_E.tex |
| F | tab:c1c2 | 22 | Readout form (10-seed means). Post-hoc simplex removes negat | paper_f/PAPER_F.tex |
| F | tab:c3 | 10 | Dimension/optimizer controls under softmax (10-seed means). | paper_f/PAPER_F.tex |
| F | tab:c4 | 16 | Gates under CE (10-seed means). Unconstrained Gate-C is the  | paper_f/PAPER_F.tex |
| F | tab:c5 | 15 | Routing on the Gate-C-topk base, _m=500 (10-seed means; med  | paper_f/PAPER_F.tex |
