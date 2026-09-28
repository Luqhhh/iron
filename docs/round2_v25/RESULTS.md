# V25 results: the time-family capacity axis is closed

Pre-registration: [PREREGISTRATION.md](PREREGISTRATION.md),
`configs/round2_v25/SPEC.yaml`. Implementation:
`src/bf_tap_r2/v25_capacity.py`. Private evidence:
`local/runs/round2-v23/dev-r1` (`screen-both-r1.json`, per-seed fit ledgers; the
directory keeps its historical V23 name and was not moved).

> **Numbering note (2026-09-28):** closed as **V23**, renumbered to **V25** to free
> the `V23` label for the teammate's histogram-target round. Public paths moved;
> private evidence paths are frozen. See
> [round2_round_numbering.md](../round2_round_numbering.md).

## What was tested

V22 left one untested axis: a capacity increase on the `raw_tabm` **time** family
whose `large` member (`N-0048`) produced the largest platform gain in the
project. Two frozen probes changed only capacity relative to the `N-0048`
recipe (`k`, `n_blocks`, `d_block`), with every other parameter — loss,
optimizer, learning rate, schedule, seeds — held fixed:

| probe | k | n_blocks | d_block | dropout | development fits |
|---|---:|---:|---:|---:|---:|
| `P_WIDE` | 32 | 3 | 768 | 0.15 | 10 |
| `P_DEEP` | 32 | 5 | 512 | 0.15 | 10 |

Both were fitted with the frozen `evaluate_v36_outer_folds` evaluator on all
five outer folds of split seeds 42 and 3407 (20 authoritative fits, zero
failures), then screened as an incremental nested blend on top of the V21 time
incumbent (`0.40 V36 + 0.25 V7m + 0.35 P-LL`), with the weight fitted on the
other split seed.

## Result: both probes select weight exactly zero

| probe | seed 42 gain | seed 3407 gain | mean | promoted |
|---|---:|---:|---:|:--:|
| `P_WIDE` | 0.000000 | 0.000000 | **0.000000** | no |
| `P_DEEP` | 0.000000 | 0.000000 | **0.000000** | no |

The nested weight is exactly `0.0` for both probes on both seeds. The reason is
visible in the standalone accuracy (pooled five-fold WMAPE on the time target):

| column | seed 42 | seed 3407 | ratio vs incumbent |
|---|---:|---:|---:|
| V21 incumbent | 0.037671 | 0.037573 | 1.000 |
| `N-0048` (`large`) | 0.040737 | 0.041091 | 1.081 / 1.094 |
| `P_DEEP` | 0.040972 | 0.041565 | 1.088 / 1.106 |
| `P_WIDE` | 0.041273 | 0.041683 | 1.096 / 1.110 |

Increasing capacity made the member **1.5-2 % less accurate** than its own
`large` parent while leaving it equally correlated, so it earns no weight. This
is the same regime V19 measured for the iron capacity family (accuracy ratios
`1.10-1.28`, every weight driven to zero) and it closes the last axis of the
time family.

## Decision

Neither probe is promoted; no confirmation fit, no full-data fit, no package and
no upload follow. **The time-family capacity axis is closed at complete
coverage — do not reopen it without new evidence.** Together with V19 (iron
capacity), V22 (whole reproducible library, including the completed N family)
and the V6 §7 stacking closure, every route that only re-weights, re-scales or
re-fits members of the existing families is now closed against the strongest
incumbent.

## Budget

| item | value |
|---|---:|
| development outer fits | 20 |
| confirmation fits | 0 |
| full-data fits | 0 |
| packages | 0 |
| agent uploads | 0 |

## G0

One slow-process episode is recorded: the first launch fitted both probes with
three workers and was killed when no fold completed inside the window; the
runner was made resumable and re-launched per probe, so the 20 authoritative
fits are complete and exactly the registered budget was spent. Targeted tests: 3
in `tests/test_round2_v25_capacity.py`.

## Remaining routes

Only two remain: **platform feedback** on the pending portfolio (which resolves
the N weight, the one open modelling question), and a **genuinely different model
class** — not another capacity, member or blend change. The `>96.4` objective is
not achieved; the registered platform best remains the user-reported V12
`96.3526`.
