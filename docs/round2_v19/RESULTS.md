# V19 results: the iron capacity lever is closed at complete coverage

Pre-registration: [PREREGISTRATION.md](PREREGISTRATION.md),
`configs/round2_v19/SPEC.yaml`. Screen implementation:
`src/bf_tap_r2/v19_screen.py`. Private evidence:
`local/runs/round2-v6-iron-capacity-networks/coverage-r2/seed-{42,3407}` and
`local/runs/round2-v19/screen-r1.json`.

## G0 engineering

The six frozen V6 iron medium/large trials were fitted with the frozen V3.6
evaluator (`evaluate_v36_outer_folds`) on **all five folds of both split seeds
42/3407**: 6 trials x 2 seeds x 5 folds = **60 authoritative outer fits**, all
successful, no epoch-cap or failure event. BLAS/OMP/MKL/NUMEXPR pinned to one
thread; the runner used 4 workers sequentially per seed because the large TabM
configuration (`k=32, n_blocks=3, d_block=512`, `max_epochs=120`, inner 5-fold
validation) is memory-heavy.

Two process facts are recorded without laundering:

* the **first** batch (`coverage-r1`) ran two seeds concurrently with 8 workers
  and was killed after 50 of 60 folds; those predictions are preserved but are
  **non-authoritative**, because the next commit changed `HEAD` and the frozen
  `_code_version` guard correctly invalidated the cache identity. The lever was
  refitted into `coverage-r2` under a stable HEAD.
* the authoritative screen therefore cost 60 fits; 50 aborted fits were spent
  and are kept as failure evidence. No threshold was changed.

## G1 result: every candidate selects weight zero

The binding statistic is the incremental nested blend gain on top of the
incumbent iron column (`V12 iron = 0.5*A35_iron + 0.5*V12_member_iron`, WMAPE
`0.037161` / `0.037198`), with the weight selected on the **other** split seed
over the frozen `0..1` grid.

| trial | family | accuracy ratio vs incumbent | residual `rho` (seed 42) | seed 42 delta | seed 3407 delta | mean gain | positive cells |
|---|---|---:|---:|---:|---:|---:|---:|
| `v6-s1-N-0024` | `raw_tabm\|large` | 1.104 | 0.9165 | 0.000000 | 0.000000 | **0.000000** | 0/10 |
| `v6-s1-N-0008` | `raw_tabm\|medium` | 1.137 | 0.8870 | 0.000000 | 0.000000 | 0.000000 | 0/10 |
| `v6-s1-N-0016` | `raw_mlp\|large` | 1.281 | 0.7984 | 0.000000 | 0.000000 | 0.000000 | 0/10 |
| `v6-s1-N-0000` | `raw_mlp\|medium` | 1.281 | 0.8014 | 0.000000 | 0.000000 | 0.000000 | 0/10 |
| `v6-s1-N-0020` | `ple_mlp\|large` | 1.112 | 0.9194 | 0.000000 | 0.000000 | 0.000000 | 0/10 |
| `v6-s1-N-0004` | `ple_mlp\|medium` | 1.112 | 0.9152 | 0.000000 | 0.000000 | 0.000000 | 0/10 |

Every nested weight is exactly `0.0` on both seeds: **the incumbent V12 iron
column already dominates every iron capacity member**, including the direct
analogue of the biggest platform winner. The gain is exactly zero, not merely
small.

### What this says about the folds-0/1 probe

The V6 probe read `-0.0014` for `N-0024` on folds 0/1. Complete coverage reads
`0.000000` with weight `0.0`. So the folds-0/1 screen did not merely add noise
around a true value — it **mis-signed and overstated** the candidate, exactly as
V6 section 2.2.2 warned. The V6 conclusion (no iron gain) nevertheless survives;
it now rests on complete-coverage evidence instead of the screen that could not
support it, which resolves the registered pre-registration defect.

### Why the analogue of the time winner does not transfer

`N-0048` (time) and `N-0024` (iron) share the winning structure, but the two
columns have opposite cost profiles for the metric. The time column is worth
~4.2x the iron column per absolute error unit, and the time incumbent is a
weak blend whose residual stream a decorrelated large member can complement; the
iron incumbent is already the strong `V12` joint column, and every capacity
member is 10-28% *less* accurate (ratio `1.10-1.28`). The recorded `JM1`
precedent (decorrelated at `rho = 0.777` but less accurate, weight optimised to
zero) is the same regime. **Independence alone does not earn blend weight; the
member must also be about as accurate as the incumbent.**

## Decision and budget

Per the pre-registered rule, no candidate is promoted (none has both
development seeds positive), so **no confirmation, no full-data fit, no package
and no upload follow.** The iron capacity lever is **closed at complete
coverage**; do not reopen it without new evidence. Post-hoc checks that combine
these members, or scan their other loss settings, would be the forbidden
"seed or loss diversity inside one structural family" and are not run.

| item | value |
|---|---:|
| development outer fits (authoritative) | 60 |
| aborted/invalidated fold fits (kept) | 50 |
| confirmation fits | 0 |
| full-data fits | 0 |
| packages | 0 |
| agent uploads | 0 |

## Where the surviving positive signal is

The V18 screen left four members with a positive incremental gain over `B0`, all
of them **accurate** members (accuracy ratio `1.00-1.04`), not capacity members:

| member | target | incremental gain over B0 | ratio | `rho` |
|---|---|---:|---:|---:|
| `v17/P_LL_T` | time | +0.004250 | 1.005 | 0.982 |
| `v15 joint-task_gated_experts` | iron | +0.002856 | 1.009 | 0.974 |
| `v11 tap_time_len-plr_quniform` | time | +0.002699 | 1.023 | 0.960 |
| `v9 tap_time_len-realmlp_td` | time | +0.002634 | 1.041 | 0.940 |

The next round should fit their full-data columns and package a `B0`-plus-expert
blend. Local gains are small (`+0.004`), the platform response to the time column
has historically been several times the local one, and the two-line ceiling from
V18 (`96.3992`) remains the bound to beat. **The `>96.4` objective is not
achieved; the registered platform best remains the user-reported V12 `96.3526`.**
