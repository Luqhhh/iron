# V24 results: the target-representation axis is closed

Pre-registration: [PREREGISTRATION.md](PREREGISTRATION.md),
`configs/round2_v24/SPEC.yaml`. Implementation:
`src/bf_tap_r2/v24_logtarget.py`. Private evidence:
`local/runs/round2-v24/dev-r1/screen.json` and the per-seed fit ledgers.

## What was tested

Every previously closed route changed members, weights, capacity or blends. V24
changed the **target representation** of the frozen winning time recipe
(`N-0048`, large `raw_tabm`, `target_transform: train_mean_std`), an axis never
applied to the winning neural families:

| probe | fit target | inverse | development fits |
|---|---|---|---:|
| `L_TIME` | `log(y)` | `exp` | 10 |
| `S_TIME` | `sqrt(y)` | square, clipped at 0 | 10 |

Every other parameter — architecture, loss, optimizer, schedule, seeds — was
held fixed. Rationale: `WMAPE` is a *relative*-error metric, so a log-target MSE
weights rows by relative error, unlike raw-unit MSE. `log1p` targets had only
ever been sampled for the old GBDT/expression families, which V22 screened to
weight `0`.

## Result: both probes select weight exactly zero

| probe | seed 42 gain | seed 3407 gain | mean | promoted |
|---|---:|---:|---:|:--:|
| `L_TIME` | 0.000000 | 0.000000 | **0.000000** | no |
| `S_TIME` | 0.000000 | 0.000000 | **0.000000** | no |

The prediction accuracy, however, shows the hypothesis was **partly right**:

| column | seed 42 WMAPE | seed 3407 WMAPE | ratio vs incumbent |
|---|---:|---:|---:|
| V21 incumbent | 0.037671 | 0.037573 | 1.000 |
| `N-0048` raw target (reference) | 0.040737 | 0.041091 | 1.081 / 1.094 |
| **`L_TIME`** `log` target | **0.040407** | **0.040976** | 1.073 / 1.091 |
| **`S_TIME`** `sqrt` target | 0.040540 | 0.040966 | 1.076 / 1.091 |

The log target is the **most accurate standalone member of this family measured
so far** — about 0.8 % / 0.3 % lower WMAPE than the raw-target parent — but the
accuracy ratio against the incumbent is still `1.07-1.09`, and V19/V23 established
that a member in that regime earns no blend weight. The improvement is real but
far too small to matter: it moves the member, not the column.

## Decision

Neither probe is promoted; no confirmation fit, no full-data fit, no package and
no upload follow. **The target-representation axis is closed.** The surviving
rule from V19 is reinforced: the binding requirement is *accuracy close to the
incumbent* (ratio near `1.0`), and no representation or capacity change tried so
far gets an existing family below `1.07`.

## Budget and a recorded incident

| item | value |
|---|---:|
| authoritative development outer fits | 20 |
| discarded fits from the first launch | 8 |
| confirmation fits | 0 |
| full-data fits | 0 |
| packages | 0 |
| agent uploads | 0 |

Two process facts are recorded:

1. the first launch fitted each fold and then failed during **inversion**: the
   spec named the inverse transforms `exp`/`square_clip` while the function
   recognised only `log`/`sqrt`. Eight fits were completed and discarded. The
   aliases were fixed and the fits re-launched (`ed0376f`). The failure events
   are preserved in `local/reports/v24-fit-seed{42,3407}.log`.
2. the recreated run directory was prepared with `rm -rf` over the first
   launch's `dev-r1`, which removed that run's append-only ledger. This is a
   deviation from the append-only rule; the eight failure events survive only in
   the stdout logs above. No authoritative evidence was affected.

## G0

Targeted tests: 4 in `tests/test_round2_v24_logtarget.py` (transform round-trips,
non-negative clipping, guard paths).

## Remaining routes

Every axis that changes members, weights, capacity, blends or target
representation of the existing families is now closed against the strongest
incumbent. Only two things remain: **platform feedback** on the pending
portfolio (which resolves the open N-weight question), and a **genuinely
different model class** — new inputs or a new paradigm, not a re-parameterisation
of the current ones. The `>96.4` objective is not achieved; the registered
platform best remains the user-reported V12 `96.3526`.
