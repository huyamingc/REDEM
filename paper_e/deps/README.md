# Paper E — frozen core dependencies (`deps/`)

This folder is Paper E's **self-contained dependency set**. The self-evolution
derivation chain (scripts `s39`–`s64` in `../scripts/`) imports ONLY these
four modules, so Paper E is fully reproducible from this folder alone and
does **not** depend on the shared `../scripts/` core.

## Contents

| file | provenance |
|---|---|
| `shallow_trap_array_simulator.py` | Paper E's frozen copy; **one audited fix (2026-09-10)**: the noiseless branch now uses `np.broadcast_to(arr, shape).copy()` instead of `np.broadcast(arr, shape)` (which raised `ValueError` whenever `col_noise_rms == 0`), plus a `--self-test` noiseless regression check. The noisy main path is unchanged, so no committed number moves. The stale same-named copy in the shared `scripts/` is not used (every import resolves here). |
| `recurrent_substrate.py` | Paper E's frozen copy; **documentation only (2026-09-10)**: the docstring now records the `alpha_eff` clip-equivalence argument and the measured deployed saturation fraction (`CLIP_FRAC_DEPLOYED = 0.393`). No formula changed. |
| `online_readout.py` | **extended** (Optional `w_max` soft L2 cap on `ThreeFactorReadout`, additive — default `None` preserves original behaviour) |
| `streaming_tasks.py` | **extended** (+4 task generators: `gen_rotation2d`, `gen_nonlinear2d`, `gen_nonlinear3d`, `gen_nonlinear4d`; additive) |

## Why this separation exists (2026-09-09)

- Papers A–C use the original shared core for their committed data; Papers
  D–F host their own companions. Paper E freezes its dependency set here so
  this manuscript is self-contained.
- The 2026-09-08 extensions (w_max, the four 2-D/3-D/4-D input-encoding
  generators) were purely additive and are used **only** by the
  self-evolution chain (verified: every original function byte-identical;
  A–D scripts have zero dependency on the new symbols), so A–D's committed
  data is unaffected.
- Hygiene: the shared `scripts/` core keeps the original A–D API; Paper E
  carries its own frozen copies here — a clean architectural separation.

## Using this folder

The s39–s64 scripts resolve these modules here first
(`sys.path.insert(0, <this folder>)`), shadowing the shared `scripts/`. To
regenerate any Paper E result:

```bash
# from repository root
PYTHONUNBUFFERED=1 python scripts/s58b_relative_sense_stress.py
# or from paper_e/deps/
PYTHONUNBUFFERED=1 python ../scripts/s58b_relative_sense_stress.py
```

## Keeping in sync

`deps/` is Paper E's reproducibility anchor: it is FROZEN. Do not edit these
files for development; if the core evolves for a future line, copy the new
versions here and note the change in the Paper E README changelog.
