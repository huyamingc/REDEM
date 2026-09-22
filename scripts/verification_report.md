# verification_report

claims registered: **121**

| paper | section | quantity | where | doc | computed | kind/tol | ok | detail |
|---|---|---|---|---|---|---|---|---|
| B | S8 | full-system overall acc @ N=256 | abstract / README / tables | 0.996 | 0.996211 | rounding/0.002 | PASS | doc=0.996 computed=0.9962111111111112 rel=2.119e-04 tol=0.002 |
| B | S8 | bare-baseline overall acc @ N=256 | abstract / README / tables | 0.973 | 0.973411 | rounding/0.002 | PASS | doc=0.973 computed=0.973411111111111 rel=4.223e-04 tol=0.002 |
| B | S8 | no-homeostat ties full (acc) | Paper B ablations | 0.996 | 0.996222 | rounding/0.002 | PASS | doc=0.996 computed=0.9962222222222221 rel=2.231e-04 tol=0.002 |
| B | S30 | full-system acc @ N=1024 | abstract / README | 0.997 | 0.997022 | rounding/0.0005 | PASS | doc=0.997 computed=0.9970222222222223 rel=2.229e-05 tol=0.0005 |
| B | S30 | baseline acc @ N=1024 | abstract / README | 0.9753 | 0.975344 | rounding/0.0005 | PASS | doc=0.9753 computed=0.9753444444444443 rel=4.557e-05 tol=0.0005 |
| B | S11 | regulated r3 MC after 3 disturbances | abstract / Paper B/C | 8.47 | 8.47495 | rounding/0.01 | PASS | doc=8.47 computed=8.474948883555967 rel=5.839e-04 tol=0.01 |
| B | S11 | fixed-kappa r3 MC | abstract / Paper B/C | 6.41 | 6.41455 | rounding/0.01 | PASS | doc=6.41 computed=6.414554704263663 rel=7.101e-04 tol=0.01 |
| B | S11 | relative sequential recovery +32% | abstract / Paper B/C | 0.32 | 0.321206 | rounding/0.05 | PASS | doc=0.32 computed=0.32120611239355235 rel=3.755e-03 tol=0.05 |
| A | S10 | forgetting-kernel Pearson r | Paper A | 0.97 | 0.970686 | rounding/0.01 | PASS | doc=0.97 computed=0.9706856810206829 rel=7.064e-04 tol=0.01 |
| A | S27 | kappa* lateral_ring (amax 0.1) | Paper A | 25.3 | 25.3 | rounding/0.01 | PASS | doc=25.3 computed=25.30001187678561 rel=4.694e-07 tol=0.01 |
| A | S27 | kappa* ring_bidir (amax 0.1) | Paper A | 27.4 | 27.4477 | rounding/0.01 | PASS | doc=27.4 computed=27.44774101354044 rel=1.739e-03 tol=0.01 |
| A | S27 | kappa* random_graph (amax 0.1) | Paper A | 27.9 | 27.9429 | rounding/0.01 | PASS | doc=27.9 computed=27.942895773298936 rel=1.535e-03 tol=0.01 |
| A | S6 | homeostat restore after edge_prune (%) | Paper A / tables | 7.9 | 7.93954 | rounding/0.05 | PASS | doc=7.9 computed=7.939536844045607 rel=4.980e-03 tol=0.05 |
| A | S6 | homeostat restore after tau_drift (%) | Paper A / tables | 18 | 18.2665 | rounding/0.05 | PASS | doc=18.0 computed=18.26650841757927 rel=1.459e-02 tol=0.05 |
| A | S6 | homeostat restore after noise (%) | Paper A / tables | 11.8 | 11.8045 | rounding/0.05 | PASS | doc=11.8 computed=11.804522490631356 rel=3.831e-04 tol=0.05 |
| A | E4 | lambda_target=0 MC gain at CV=0.1 (%) | Paper A / Paper B table | 25 | 24.5514 | rounding/0.05 | PASS | doc=25.0 computed=24.55144147223043 rel=1.827e-02 tol=0.05 |
| B | S30 | N=1024 full-vs-baseline paired t | Paper B / README | 15.3 | 15.256 | rounding/0.05 | PASS | doc=15.3 computed=15.256005856493127 rel=2.884e-03 tol=0.05 |
| B | S25 | novelty-guided rewiring MC_final | Paper B | 14.59 | 14.5854 | rounding/0.01 | PASS | doc=14.59 computed=14.585449704460018 rel=3.120e-04 tol=0.01 |
| B | S25 | correlation-guided rewiring MC_final | Paper B | 12.43 | 12.434 | rounding/0.01 | PASS | doc=12.43 computed=12.434047269794494 rel=3.255e-04 tol=0.01 |
| B | S24 | homeostat-only r3 MC (no plasticity) | Paper B / S11 anchor | 8.47 | 8.47495 | rounding/0.01 | PASS | doc=8.47 computed=8.474948883555967 rel=5.839e-04 tol=0.01 |
| B | S24 | coupled rewiring r3 MC (harmful) | Paper B | 5.27 | 5.27153 | rounding/0.01 | PASS | doc=5.27 computed=5.271525363305836 rel=2.894e-04 tol=0.01 |
| C | S10 | ESN+meta overall acc | Paper B/C table | 0.998 | 0.997911 | rounding/0.002 | PASS | doc=0.998 computed=0.9979111111111111 rel=8.907e-05 tol=0.002 |
| C | S10 | ESN-fast overall acc | Paper B/C table | 0.996 | 0.995511 | rounding/0.002 | PASS | doc=0.996 computed=0.995511111111111 rel=4.911e-04 tol=0.002 |
| C | S10 | REDEM-full overall acc | Paper B/C table | 0.994 | 0.994233 | rounding/0.002 | PASS | doc=0.994 computed=0.9942333333333334 rel=2.347e-04 tol=0.002 |
| C | E3-transfer | s14 paired MC diff at r1 (dual-fast) | Paper C | -0.78 | -0.778506 | rounding/0.05 | PASS | doc=-0.78 computed=-0.7785058668741548 rel=1.919e-03 tol=0.05 |
| C | E3-transfer | s14 paired MC diff at r2 (dual-fast) | Paper C | -0.76 | -0.759189 | rounding/0.05 | PASS | doc=-0.76 computed=-0.7591889315236324 rel=1.068e-03 tol=0.05 |
| C | E3-transfer | s14 paired MC diff at r3 (dual-fast) | Paper C | -0.69 | -0.693147 | rounding/0.05 | PASS | doc=-0.69 computed=-0.6931466416395877 rel=4.540e-03 tol=0.05 |
| B | S2 | online RLS drift-binary mean acc | Paper B / README_REDEM | 0.978 | 0.977939 | rounding/0.01 | PASS | doc=0.978 computed=0.977939448757569 rel=6.192e-05 tol=0.01 |
| D | P1 | B-proj / B-lin-clip stream ppl (input path) | abstract / Paper D/F | 11.75 | 11.7541 | rounding/0.005 | PASS | doc=11.75 computed=11.754124283044472 rel=3.509e-04 tol=0.005 |
| D | P1 | oracle / pooled-table ceiling stream ppl | Paper D | 7.25 | 7.24781 | rounding/0.02 | PASS | doc=7.25 computed=7.247809995539264 rel=3.022e-04 tol=0.02 |
| D | P2 | A3-A1 forgetting diff at tau_m=200 | Paper D | -2.05 | -2.05268 | rounding/0.02 | PASS | doc=-2.05 computed=-2.0526782634737275 rel=1.305e-03 tol=0.02 |
| D | P2 | A3-A1 forgetting diff at tau_m=500 | Paper D | -1.87 | -1.86947 | rounding/0.02 | PASS | doc=-1.87 computed=-1.869466006154214 rel=2.856e-04 tol=0.02 |
| D | P2 | A3-A1 forgetting diff at tau_m=1000 | Paper D | -1.2 | -1.20153 | rounding/0.05 | PASS | doc=-1.2 computed=-1.2015268901595109 rel=1.271e-03 tol=0.05 |
| D | P3 | A3-soft stream ppl (E1) | Paper D | 8.22 | 8.21824 | rounding/0.01 | PASS | doc=8.22 computed=8.218244292247023 rel=2.136e-04 tol=0.01 |
| D | P3 | A3-abrupt stream ppl (E1) | Paper D | 10.03 | 10.0328 | rounding/0.01 | PASS | doc=10.03 computed=10.032775778209635 rel=2.767e-04 tol=0.01 |
| D | P3 | M5 regulated whitened-state norm mean | Paper D | 11.3 | 11.3137 | rounding/0.02 | PASS | doc=11.3 computed=11.313732258887935 rel=1.214e-03 tol=0.02 |
| D | P3 | bare host whitened-state norm mean | Paper D | 50.2 | 50.1882 | rounding/0.02 | PASS | doc=50.2 computed=50.18823344016563 rel=2.344e-04 tol=0.02 |
| D | P4 | SSM-REDEM stream ppl | Paper D tables | 13.18 | 13.1796 | rounding/0.01 | PASS | doc=13.18 computed=13.179586276981818 rel=3.139e-05 tol=0.01 |
| D | P4 | SSM-REDEM forgetting ppl | Paper D tables | 8.93 | 8.93112 | rounding/0.01 | PASS | doc=8.93 computed=8.93112298460743 rel=1.257e-04 tol=0.01 |
| D | P4-text | REDEM-SSM real-text stream ppl | Paper D | 12.07 | 12.071 | rounding/0.01 | PASS | doc=12.07 computed=12.07099417985203 rel=8.236e-05 tol=0.01 |
| D | P4-text | char-bigram full-book ceiling ppl | Paper D | 10.97 | 10.9737 | rounding/0.01 | PASS | doc=10.97 computed=10.973725062300073 rel=3.395e-04 tol=0.01 |
| D | P4 | SSM-REDEM vs bare stream diff (ppl, negative=better) | Paper D tables | -2.25 | -2.24672 | rounding/0.02 | PASS | doc=-2.25 computed=-2.246721214020686 rel=1.459e-03 tol=0.02 |
| D | P4 | SSM-REDEM vs bare forgetting diff | Paper D tables | -4.47 | -4.47347 | rounding/0.02 | PASS | doc=-4.47 computed=-4.473469096705111 rel=7.755e-04 tol=0.02 |
| D | P4 | SSM-REDEM vs TF-A1 stream diff | Paper D tables | -9.28 | -9.27882 | rounding/0.02 | PASS | doc=-9.28 computed=-9.278820106388663 rel=1.272e-04 tol=0.02 |
| E | R5 | frozen-hypothesis memory mean acc | abstract / Paper E | 0.888 | 0.888019 | rounding/0.002 | PASS | doc=0.888 computed=0.8880186427476696 rel=2.099e-05 tol=0.002 |
| E | R5 | re-adaptation / rls_single mean acc | Paper E | 0.838 | 0.837625 | rounding/0.002 | PASS | doc=0.838 computed=0.8376248084218989 rel=4.479e-04 tol=0.002 |
| E | R5 | frozen-snapshot retention gain (pp) | abstract / Paper E / s65 | 8.7 | 8.7 | rounding/0.02 | PASS | doc=8.7 computed=8.699999999999996 rel=4.084e-16 tol=0.02 |
| E | EWC | EWC whole-run accuracy change (pp) | abstract / Paper E | -0.85 | -0.851951 | rounding/0.05 | PASS | doc=-0.85 computed=-0.8519509560061345 rel=2.290e-03 tol=0.05 |
| E | EWC | EWC whole-run paired t | Paper E | -3.87 | -3.87059 | rounding/0.02 | PASS | doc=-3.87 computed=-3.870591931997619 rel=1.529e-04 tol=0.02 |
| E | EWC | s65 reference arm bit-matches s52 (max abs diff) | reproduction_check | 0 | 0 | identity/0 | PASS | doc=0.0 computed=0.0 |diff|=0 tol=0.0 |
| E | R5 | cross-family ring gain paired t (corrected) | Paper E / MAINTENANCE | 2.45 | 2.44949 | rounding/0.05 | PASS | doc=2.45 computed=2.449489742783178 rel=2.083e-04 tol=0.05 |
| E | R1 | post-inversion flip acc uncoupled | abstract / Paper E | 0.898 | 0.898 | rounding/0.002 | PASS | doc=0.898 computed=0.898 rel=0.000e+00 tol=0.002 |
| E | R1 | post-inversion flip acc coupled | abstract / Paper E | 0.939 | 0.93875 | rounding/0.002 | PASS | doc=0.939 computed=0.93875 rel=2.663e-04 tol=0.002 |
| E | R1 | oracle post-inversion acc uncoupled | Paper E | 0.975 | 0.975 | rounding/0.002 | PASS | doc=0.975 computed=0.975 rel=0.000e+00 tol=0.002 |
| E | R1 | oracle post-inversion acc coupled | Paper E | 1 | 1 | rounding/0.002 | PASS | doc=1.0 computed=1.0 rel=0.000e+00 tol=0.002 |
| E | R1 | first-flip latency (blocks after inversion) | Paper E | 39 | 39 | identity/0 | PASS | doc=39.0 computed=39.0 |diff|=0 tol=0.0 |
| E | R1 | learned flip value v (post-flip contrast) | Paper E | 0.504 | 0.503762 | rounding/0.002 | PASS | doc=0.504 computed=0.5037621465429181 rel=4.722e-04 tol=0.002 |
| E | R1 | stable-stream whole-run acc uncoupled (bit-identical no-meta) | Paper E | 0.897925 | 0.897925 | precision/1e-05 | PASS | doc=0.897925 computed=0.8979251149468606 rel=1.280e-07 tol=1e-05 |
| E | R1 | stable-stream whole-run acc coupled (bit-identical no-meta) | Paper E | 0.938241 | 0.938241 | precision/1e-05 | PASS | doc=0.938241 computed=0.9382407854074017 rel=2.287e-07 tol=1e-05 |
| E | R1 | R1 zoo coupled full-swap rmhl post | Paper E / Table tab:zoo | 0.058 | 0.058 | rounding/0.01 | PASS | doc=0.058 computed=0.05800000000000001 rel=1.196e-16 tol=0.01 |
| E | R1 | R1 zoo coupled full-swap flip post | Paper E / Table tab:zoo | 0.942 | 0.942 | rounding/0.01 | PASS | doc=0.942 computed=0.942 rel=0.000e+00 tol=0.01 |
| E | R1 | R1 zoo coupled near-inversion rmhl post | Paper E / Table tab:zoo | 0.053 | 0.053 | rounding/0.01 | PASS | doc=0.053 computed=0.053000000000000005 rel=1.309e-16 tol=0.01 |
| E | R1 | R1 zoo coupled near-inversion flip post | Paper E / Table tab:zoo | 0.947 | 0.947 | rounding/0.01 | PASS | doc=0.947 computed=0.9469999999999998 rel=1.172e-16 tol=0.01 |
| E | R1 | R1 zoo partial-shift flip post-window | Paper E / Table tab:zoo | 0.485 | 0.48475 | rounding/0.01 | PASS | doc=0.485 computed=0.48475 rel=5.157e-04 tol=0.01 |
| E | R1 | R1 zoo partial-shift flip steady | Paper E / Table tab:zoo | 0.508 | 0.508 | rounding/0.01 | PASS | doc=0.508 computed=0.508 rel=0.000e+00 tol=0.01 |
| E | R1 | R1 zoo partial-shift flip mean flips | Paper E / Table tab:zoo | 2.7 | 2.7 | rounding/0.01 | PASS | doc=2.7 computed=2.7 rel=0.000e+00 tol=0.01 |
| E | R1 | corrupt-window acc uncoupled (honest negative) | Paper E | 0.106 | 0.106 | rounding/0.01 | PASS | doc=0.106 computed=0.10600000000000001 rel=1.309e-16 tol=0.01 |
| E | R1 | corrupt-window acc coupled (honest negative) | Paper E | 0.064 | 0.064 | rounding/0.01 | PASS | doc=0.064 computed=0.06400000000000002 rel=2.168e-16 tol=0.01 |
| E | R1 | rule-adjudication delta pre-swap uncoupled | Paper E | 0.99 | 0.99 | rounding/0.002 | PASS | doc=0.99 computed=0.99 rel=0.000e+00 tol=0.002 |
| E | R1 | rule-adjudication delta steady uncoupled | Paper E | 0.989 | 0.989 | rounding/0.002 | PASS | doc=0.989 computed=0.9889999999999999 rel=1.123e-16 tol=0.002 |
| E | R1 | rule-adjudication RMHL post-swap uncoupled | Paper E | 0.097 | 0.0965 | rounding/0.01 | PASS | doc=0.097 computed=0.0965 rel=5.181e-03 tol=0.01 |
| E | R2 | raw capacity probe coupled (circle) | Paper E | 0.84 | 0.841441 | rounding/0.01 | PASS | doc=0.84 computed=0.8414411764705884 rel=1.713e-03 tol=0.01 |
| E | R2 | raw capacity probe uncoupled (circle) | Paper E | 0.55 | 0.546843 | rounding/0.01 | PASS | doc=0.55 computed=0.546843137254902 rel=5.773e-03 tol=0.01 |
| E | R2 | RLS acc 4D shell2 (two-shell) | Paper E | 0.547 | 0.546798 | rounding/0.005 | PASS | doc=0.547 computed=0.5467976820568607 rel=3.700e-04 tol=0.005 |
| E | R2 | RLS acc 4D shell3 (three-shell) | Paper E | 0.559 | 0.559027 | rounding/0.005 | PASS | doc=0.559 computed=0.559026673082153 rel=4.771e-05 tol=0.005 |
| E | R2 | 4D-shell margin at N=64 | Paper E | 0.034 | 0.034 | rounding/0.02 | PASS | doc=0.034 computed=0.03400000000000003 rel=8.163e-16 tol=0.02 |
| E | R2 | 4D-shell margin at N=256 | Paper E | 0.1 | 0.0995 | rounding/0.02 | PASS | doc=0.1 computed=0.09949999999999992 rel=5.025e-03 tol=0.02 |
| E | R2 | 4D-shell margin at N=512 | Paper E | 0.107 | 0.107 | rounding/0.02 | PASS | doc=0.107 computed=0.10699999999999998 rel=1.297e-16 tol=0.02 |
| E | R2 | 4D-shell margin at kappa=20 (sampled peak) | Paper E | 0.1185 | 0.1185 | rounding/0.01 | PASS | doc=0.1185 computed=0.11850000000000005 rel=4.684e-16 tol=0.01 |
| E | R2 | 4D-shell margin at deployed kappa=25 | Paper E | 0.0995 | 0.0995 | rounding/0.01 | PASS | doc=0.0995 computed=0.09949999999999992 rel=8.369e-16 tol=0.01 |
| E | R2 | 2D-circle margin flat across N (N=64) | Paper E | 0.401 | 0.401 | rounding/0.01 | PASS | doc=0.401 computed=0.40100000000000013 rel=2.769e-16 tol=0.01 |
| E | R3 | per-segment memory gain seg1_acc (pp) | Paper E | 4.2 | 4.15 | rounding/0.02 | PASS | doc=4.2 computed=4.1499999999999995 rel=1.205e-02 tol=0.02 |
| E | R3 | per-segment memory gain paired t seg1_acc | Paper E | 2.72 | 2.72314 | rounding/0.02 | PASS | doc=2.72 computed=2.7231416669378654 rel=1.154e-03 tol=0.02 |
| E | R3 | per-segment memory gain seg2_acc (pp) | Paper E | 3.6 | 3.6 | rounding/0.02 | PASS | doc=3.6 computed=3.600000000000001 rel=2.467e-16 tol=0.02 |
| E | R3 | per-segment memory gain paired t seg2_acc | Paper E | 2.73 | 2.7309 | rounding/0.02 | PASS | doc=2.73 computed=2.7308973820905384 rel=3.286e-04 tol=0.02 |
| E | R3 | per-segment memory gain seg3_acc (pp) | Paper E | 2.6 | 2.55 | rounding/0.02 | PASS | doc=2.6 computed=2.5500000000000003 rel=1.961e-02 tol=0.02 |
| E | R3 | per-segment memory gain paired t seg3_acc | Paper E | 2.02 | 2.0214 | rounding/0.02 | PASS | doc=2.02 computed=2.021398875151293 rel=6.920e-04 tol=0.02 |
| E | R3 | per-segment memory gain seg4_acc (pp) | Paper E | 14.9 | 14.85 | rounding/0.02 | PASS | doc=14.9 computed=14.850000000000001 rel=3.367e-03 tol=0.02 |
| E | R3 | per-segment memory gain paired t seg4_acc | Paper E | 11.5 | 11.4637 | rounding/0.02 | PASS | doc=11.5 computed=11.46365971771185 rel=3.170e-03 tol=0.02 |
| E | R3 | population control (no frozen snapshot) mean acc | Paper E | 0.839 | 0.838739 | rounding/0.002 | PASS | doc=0.839 computed=0.8387389764076278 rel=3.112e-04 tol=0.002 |
| E | R4 | sense reliability ABAB SEG=500 | Paper E | 0.9 | 0.9 | rounding/0.01 | PASS | doc=0.9 computed=0.9 rel=0.000e+00 tol=0.01 |
| E | R4 | sense reliability ABAB SEG=250 | Paper E | 0.27 | 0.27 | rounding/0.01 | PASS | doc=0.27 computed=0.27 rel=0.000e+00 tol=0.01 |
| E | R4 | sense reliability ABAB SEG=125 (cliff floor) | Paper E | 0 | 0 | identity/0 | PASS | doc=0.0 computed=0.0 |diff|=0 tol=0.0 |
| E | R5 | learned snapshot gate v_snap final (dynamic) | Paper E | 0.252 | 0.252 | rounding/0.01 | PASS | doc=0.252 computed=0.25199999999999995 rel=2.203e-16 tol=0.01 |
| E | R5 | memory selection fraction (dynamic) | Paper E | 0.227 | 0.22695 | rounding/0.02 | PASS | doc=0.227 computed=0.22695000000000004 rel=2.203e-04 tol=0.02 |
| E | R6 | D2 post-swap at ratio 0.25 (fail band) | Paper E | 0.486 | 0.486161 | rounding/0.01 | PASS | doc=0.486 computed=0.48616124497991964 rel=3.317e-04 tol=0.01 |
| E | R6 | D2 post-swap at ratio 1.0 (intermediate) | Paper E | 0.561 | 0.561438 | rounding/0.01 | PASS | doc=0.561 computed=0.5614381282051282 rel=7.804e-04 tol=0.01 |
| E | R6 | D2 post-swap at ratio 1.25 (band interior) | Paper E | 0.791 | 0.791466 | rounding/0.01 | PASS | doc=0.791 computed=0.7914655046826222 rel=5.882e-04 tol=0.01 |
| E | R6 | D2 post-swap at ratio 1.5 (band interior) | Paper E | 0.85 | 0.849878 | rounding/0.01 | PASS | doc=0.85 computed=0.8498780578480224 rel=1.435e-04 tol=0.01 |
| E | R6 | D2 post-swap at ratio 1.75 (band interior) | Paper E | 0.909 | 0.909436 | rounding/0.01 | PASS | doc=0.909 computed=0.9094356885450701 rel=4.791e-04 tol=0.01 |
| E | R6 | D2 post-swap at ratio 2.0 (complete recovery) | Paper E | 0.938 | 0.937509 | rounding/0.005 | PASS | doc=0.938 computed=0.9375090789473683 rel=5.236e-04 tol=0.005 |
| E | R7 | single-speaker linear acc | Paper E | 0.938 | 0.937713 | rounding/0.002 | PASS | doc=0.938 computed=0.937713248410844 rel=3.058e-04 tol=0.002 |
| E | R7 | A-B-A-B linear acc (switch_every=200) | Paper E | 0.9 | 0.899586 | rounding/0.002 | PASS | doc=0.9 computed=0.8995859400517574 rel=4.603e-04 tol=0.002 |
| E | R7 | A-B-A-B memory-recovered acc | Paper E | 0.913 | 0.912895 | rounding/0.002 | PASS | doc=0.913 computed=0.912894776513153 rel=1.153e-04 tol=0.002 |
| E | R7 | capacity probe single-speaker | Paper E | 0.949 | 0.9493 | rounding/0.002 | PASS | doc=0.949 computed=0.9492999999999998 rel=3.160e-04 tol=0.002 |
| E | R7 | capacity probe A-B-A-B | Paper E | 0.9 | 0.9003 | rounding/0.002 | PASS | doc=0.9 computed=0.9003 rel=3.332e-04 tol=0.002 |
| E | R8 | minority-good majority-rule acc | Paper E / Table tab:multisource | 0.684 | 0.683686 | rounding/0.002 | PASS | doc=0.684 computed=0.6836862139142232 rel=4.590e-04 tol=0.002 |
| E | R8 | minority-good validation-selector acc | Paper E / Table tab:multisource | 1 | 1 | rounding/0.002 | PASS | doc=1.0 computed=1.0 rel=0.000e+00 tol=0.002 |
| E | R8 | drifting majority-rule acc | Paper E / Table tab:multisource | 0.681 | 0.680815 | rounding/0.002 | PASS | doc=0.681 computed=0.6808148036481495 rel=2.720e-04 tol=0.002 |
| E | R8 | drifting validation-selector acc | Paper E / Table tab:multisource | 0.991 | 0.991307 | rounding/0.002 | PASS | doc=0.991 computed=0.991306751086656 rel=3.094e-04 tol=0.002 |
| E | R8 | all-bad majority-rule acc (no source to trust) | Paper E | 0.494 | 0.493634 | rounding/0.01 | PASS | doc=0.494 computed=0.49363357704580296 rel=7.423e-04 tol=0.01 |
| F | C1 | B-softmax stream ppl | abstract / Paper F | 8.65 | 8.64834 | rounding/0.005 | PASS | doc=8.65 computed=8.648338148738018 rel=1.922e-04 tol=0.005 |
| F | C1 | stream CE improvement magnitude (nats/token) | abstract / Paper F | 0.307 | 0.306837 | rounding/0.05 | PASS | doc=0.307 computed=0.30683700086248233 rel=5.312e-04 tol=0.05 |
| F | C1 | clip-floor park rate low (~1.6%) | Paper F | 1.6 | 1.58175 | rounding/0.15 | PASS | doc=1.6 computed=1.5817545419189956 rel=1.153e-02 tol=0.15 |
| F | C1 | clip-floor park rate high (~10.2%) | Paper F | 10.2 | 10.2289 | rounding/0.15 | PASS | doc=10.2 computed=10.228901605644758 rel=2.825e-03 tol=0.15 |
| F | C1 | simplex does not repair (stream ppl delta vs lin-clip) | Paper F | 0.03 | 0.0283896 | rounding/0.1 | PASS | doc=0.03 computed=0.02838957053185709 rel=5.673e-02 tol=0.1 |
| F | C4 | Gate-C-topk stream ppl | abstract / Paper F | 7.03 | 7.03266 | rounding/0.01 | PASS | doc=7.03 computed=7.032655791397405 rel=3.776e-04 tol=0.01 |
| F | C5 | soft routing forgetting ppl | Paper F | 7.79 | 7.78883 | rounding/0.01 | PASS | doc=7.79 computed=7.788825326133671 rel=1.508e-04 tol=0.01 |
| F | C5 | hard routing forgetting ppl | Paper F | 6.94 | 6.93944 | rounding/0.01 | PASS | doc=6.94 computed=6.939440481059913 rel=8.063e-05 tol=0.01 |
| F | external | X-SelSSM-frozen stream ppl | Paper F / Discussion | 9.03 | 9.02967 | rounding/0.01 | PASS | doc=9.03 computed=9.029671573863947 rel=3.637e-05 tol=0.01 |
| F | external | F Gate-C-topk stream anchor (same host/stream) | Paper F | 7.03 | 7.03266 | rounding/0.01 | PASS | doc=7.03 computed=7.032655791397405 rel=3.776e-04 tol=0.01 |

failures: **0**
