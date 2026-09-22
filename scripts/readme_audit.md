# readme_audit

| id | pattern | claim | ok | note |
|---|---|---|---|---|
| S8-full | `0\.996\b` | full-system overall acc @ N=256 | PASS | ok |
| S8-base | `0\.973\b` | bare-baseline overall acc @ N=256 | PASS | ok |
| S30-full | `0\.9970\b` | full-system acc @ N=1024 | PASS | ok |
| S30-base | `0\.9753\b` | baseline acc @ N=1024 | PASS | ok |
| S11-reg | `8\.47\b` | regulated r3 MC after 3 disturbances | PASS | ok |
| S11-fix | `6\.41\b` | fixed-kappa r3 MC | PASS | ok |
| F-stream | `11\.75\b` | B-proj / B-lin-clip stream ppl (input path) | PASS | ok |
| F-softmax | `8\.65\b` | B-softmax stream ppl | PASS | ok |
| F-topk | `7\.03\b` | Gate-C-topk stream ppl | PASS | ok |
| E-mem | `\+8\.70\b` | frozen-snapshot retention gain (pp) | PASS | ok |
| E-ewc | `−0\.85|-0\.85\b` | EWC whole-run accuracy change (pp) | PASS | ok |
| A-kappa | `κ\*∈\(25,30\)|κ\\ast|25\.3` | kappa* lateral_ring (amax 0.1) | PASS | ok |
| A-r | `r=0\.97|r = 0\.97` | forgetting-kernel Pearson r | PASS | ok |
| C-s14 | `-0\.78|−0\.78` | s14 paired MC diff at r1 (dual-fast) | PASS | ok |
| B-nov | `14\.59` | novelty-guided rewiring MC_final | PASS | ok |

failures: **0**
