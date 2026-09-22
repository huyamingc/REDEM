# tex_number_audit

scanned: 6 manuscripts

## Forbidden wording

| pattern | reason | hits |
|---|---|---|
| `2\.45\s*<\s*2\.262` | false significance: cross-family t=2.45 is ABOVE 2.262 (see MAINTENANCE) | — |
| `requires a preheating simulation` | stale pending wording | — |
| `pending dark-matter normalization|pending preheating` | stale pending wording (physics-era; must not appear in REDEM papers) | — |
| `per-dimension correction required` | retracted structural correction claim | — |
| `1e-67\s*cm\^2` | stale physics magnitude from another repo | — |

## Headline magnitude presence (D/E/F)

| token | meaning | found in |
|---|---|---|
| 0.307 | F CE drop | PAPER_D.tex, PAPER_F.tex |
| 11.75 | F/D stream anchor | PAPER_D.tex, PAPER_F.tex |
| 8.65 | F softmax stream | PAPER_D.tex, PAPER_F.tex |
| 8.70 | E retention gain | PAPER_E.tex |
| 3.87 | EWC t | PAPER_E.tex |

## Light reverse scan (scientific-looking numbers)

- `paper_a/PAPER_A.tex`: 24 headline-like tokens
- `paper_b/PAPER_B.tex`: 17 headline-like tokens
- `paper_c/PAPER_C.tex`: 50 headline-like tokens
- `paper_d/PAPER_D.tex`: 149 headline-like tokens
- `paper_e/PAPER_E.tex`: 216 headline-like tokens
- `paper_f/PAPER_F.tex`: 17 headline-like tokens

total headline-like tokens: **473** (manual registry covers a subset; unregistered tokens are not auto-fail in MVP)

failures: **0**
