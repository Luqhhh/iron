# Round2 V3.2 Strong-Ensemble and Target-Expression Search

Status: **protocol patch completed; six-recipe seed ensemble and 256-item search not yet executed; no package and no upload.**

Base commit: `4b87834186a892a9299abf965973a12d74a5904a`.

## Current references

| reference | definition | local score |
|---|---|---:|
| P0 | AJ3 iron + full B3 time | 96.016655 under S3 protocol; 96.1259 user-reported platform |
| L0 | V3 frozen weighted fusion | 96.118109 under S3 protocol |
| L1-procedure | V3.1 S3 candidate member pool and inner LP procedure | 96.164036 under S3 protocol |

The V3.1 S3 candidate remains stronger than both P0 and L0 on the same
sample/duplicate-group-isolated outer protocol.  New V3.2 candidates must report
delta versus all three references; local gains are not platform predictions.

## Protocol patch

Implemented in `src/bf_tap_r2/v3_1_models.py` under protocol version
`v3.2-index-aware-v1`:

- CatBoost and XGBoost best iteration is treated as a zero-based index;
  a valid index `k` maps to `k+1` boosting rounds;
- index `0` is a valid best index and maps to one round, not to the configured
  maximum;
- LightGBM uses its retained iteration count directly;
- missing/negative best index falls back to the frozen configured round count;
- every early-stopped tree fit now records:
  - `best_iteration_index`
  - `selected_num_boost_round`
  - `actual_num_boost_round`
  - `configured_num_boost_round`.

Regression tests are in `tests/test_round2_v3_2_protocol.py` and cover index 0,
middle/last indices, no valid index, and LightGBM/XGBoost differences.

Synthetic boundary cases from the task book are recorded privately in:

`local/runs/round2-v3.2-ensemble-and-target-search/protocol-patch-r1/protocol_patch_report.json`

The historical V3/V3.1 scores are not retroactively invalidated by this patch.

## Next execution blocks

1. six base recipes x fixed training seeds `[42, 2026, 2027]`;
2. 256 new configurations by the frozen target allocation;
3. paired P0/L0/L1 replays under the corrected training protocol;
4. two-fold coarse screen, then seeds 42/3407 full OOF;
5. one final outer split (suggested seed 9091) for at most two frozen candidates.

No V3.2 package has been generated, no existing prepared ZIP has been modified,
and no platform upload has occurred.


## Six-recipe fixed three-seed ensemble

The six strong recipes were run with fixed training seeds `[42, 2026, 2027]`
under the corrected index-aware protocol:

- iron: `v31-s1-expr-iron-0018`, `v31-s1-expr-iron-0012`, V3 `0107`;
- time: `v31-s1-time-0021-0050`, `0031`, `0039`.

Each recipe's three-seed original-unit average improved over the mean of its
individual seeds:

| recipe | split 42 gain (WMAPE) | split 3407 gain (WMAPE) |
|---|---:|---:|
| iron 0018 | 0.0002522 | 0.0002298 |
| iron 0012 | 0.0003008 | 0.0003284 |
| iron 0107 | 0.0003996 | 0.0003475 |
| time 0050 | 0.0002621 | 0.0002790 |
| time 0031 | 0.0003942 | 0.0004099 |
| time 0039 | 0.0004798 | 0.0004780 |

A fixed-pool LP comparison was then made:

| pool | in-sample package score | nested cross-seed package score |
|---|---:|---:|
| corrected single-seed members | 96.153377 | 96.145620 |
| three-seed ensemble members | **96.157518** | **96.148790** |
| gain | +0.004141 | +0.003171 |

The three-seed ensemble is a real but small gain.  It is below the V3.2
`+0.02` promotion line, so this line does not justify expanding to more seeds.
It is retained as fusion material and for potential interaction with the
remaining V3.2 search lines.


## 256-configuration directed search

The 256-item batch completed:

| line | iron | time | total |
|---|---:|---:|---:|
| iron log target / feature expression | 96 | 0 | 96 |
| time new-center neighborhood | 0 | 80 | 80 |
| quantization and rsm | 16 | 16 | 32 |
| calibration and shallow residual | 16 | 16 | 32 |
| heterogeneous far family | 8 | 8 | 16 |
| **total** | **136** | **120** | **256** |

The first coarse pass completed 252/256 and preserved four kernel `random_state`
failures; after removing the invalid KernelRidge parameter and rerunning the
same trial identities, the final ledger has 256 complete trials and the failure
history retained.

Top refined single models on seeds 42/3407:

| target | candidate | mean pooled WMAPE |
|---|---|---:|
| iron | `v32-s1-iron_log_expression-0019` | 0.0388946 |
| time | `v32-s1-time_neighborhood-0126` | 0.0389088 |

The V3.2 fusion on seeds 42/3407 selected log-expression, quantization and
affine-calibration members:

- in-sample package: `96.165190`;
- nested cross-seed package: `96.160029`.

## Final S3-style outer validation

One fixed new outer split was used: outer seed `9091`, five
sample/duplicate-group-isolated folds, inner seed `5555`, three-fold inner OOF
for member/weight selection, then full outer-train refit.

Under that same protocol:

| candidate | package score | delta vs L0 | delta vs P0 |
|---|---:|---:|---:|
| L1 procedure | 96.136206 | +0.039349 | +0.122806 |
| V3.2 candidate | **96.140023** | **+0.043166** | **+0.126623** |

V3.2 improved over the L1 procedure under the same outer protocol by
**+0.003817**.  This is below the task-book `+0.01` independent-promotion line.

Decision:

- V3.2 is retained as fusion material;
- L1 remains the preferred procedure;
- no V3.2 package is promoted or generated;
- the five prepared V3.1 packages remain unchanged;
- no platform upload occurred.

The V3.2 target WMAPEs under outer seed 9091 were `0.03840082` for iron and
`0.03879872` for time, versus L1's `0.03833417` and `0.03894172`.  The package
gain therefore comes from the time side, while iron is slightly weaker.
