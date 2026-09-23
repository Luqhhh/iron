# Round2 V3 Local Search: First Batch Recon

Status: **first-batch reconnaissance complete; no package and no upload**.

## Executed budget

| family | planned | complete | blocked | note |
|---|---:|---:|---:|---|
| CatBoost | 240 | 240 | 0 | coarse seed 42 / folds 0–1 |
| LightGBM | 40 | 40 | 0 | coarse seed 42 / folds 0–1 |
| XGBoost | 40 | 0 | 40 | `xgboost` not installed in the locked environment |
| MLP | 20 | 20 | 0 | coarse seed 42 / folds 0–1 |
| kernel | 20 | 20 | 0 | coarse seed 42 / folds 0–1 |
| expression | 40 | 40 | 0 | CatBoost backbone, feature/target-expression axis |
| **total** | **400** | **360** | **40** | blocked XGBoost is not treated as a closed route |

Refinement used the top 8 overall trials plus the best non-CatBoost trial from
each available family, up to 12 per target.  Full 5-fold OOF was computed for
seeds 42 and 3407.  Thirty-two CatBoost prediction arrays were reused from the
CatBoost-only refine pass; 16 new `(trial, seed)` fits covered the selected
LightGBM/MLP/kernel/expression entries.

## Single-model findings

The new CatBoost search produced two useful local single models:

| target | candidate | seed 42 WMAPE | seed 3407 WMAPE | mean WMAPE | previous local strong single |
|---|---|---:|---:|---:|---:|
| iron | `v3-catboost-tap_iron-0075` | 0.03909385 | 0.03925472 | **0.03917428** | DJ 0.03920512 |
| time | `v3-catboost-tap_time_len-0021` | 0.03921803 | 0.03991548 | **0.03956676** | B3/full-time local reference 0.04048214 |

The iron improvement is only about `-0.0000308` in original-unit WMAPE, so by
itself it is not enough for a platform candidate.  The time improvement is
larger, about `-0.0009154`, and is the more important single-model signal.
LightGBM/MLP/kernel/expression entries did not beat CatBoost on the same full
OOF, but several are retained as potential complements.

## Nested cross-seed fusion findings

Two nested leave-one-seed-out runs were compared.

### CatBoost-refined candidate library

- iron held-out WMAPE: **0.03856469**
- time held-out WMAPE: **0.03942060**
- package local score: **96.100735**
- delta vs AJ3 local reference: **+0.088347**
- triage: `candidate_pool` (above 0.05, below 0.15–0.20 sprint band)

### All-family refined candidate library

- iron held-out WMAPE: **0.03857898**
- time held-out WMAPE: **0.03942060**
- package local score: **96.100021**
- delta vs AJ3 local reference: **+0.087633**
- triage: `candidate_pool`

The all-family additions did not improve the nested package score over the
CatBoost-refined-only pool, so the current leading local direction is the
CatBoost search plus fusion with the existing V2 OOF library.

## Selected fusion structure in the CatBoost-only run

- Iron, held-out seed 3407: new CatBoost `0075`, `AJM1`, CatBoost `0107`, `J1`,
  `AKW2`.
- Iron, held-out seed 42: `DJ`, CatBoost `0043`, `AKW2`, `J1`, expression `0002`.
- Time, held-out seed 3407: CatBoost `0021` with large weight, plus `T1`, CatBoost
  `0072`, `AFE4`, `ABAY`.
- Time, held-out seed 42: CatBoost `0021`, `AORD`, `T1`, and a second `T1`
  prediction.

The time side is more stable: CatBoost `0021` receives the largest weight in
both held-out directions.  The iron side is less stable and must not be frozen
from this single nested run.

## Interpretation and next steps

The first batch found a local complete-package gain of about `+0.088` against AJ3's
local reference under leave-one-seed-out fusion.  This clears the `0.05` local
candidate-pool gate and is worth continuing, but it does **not** yet justify a
package because:

1. the 400-item budget has 40 blocked XGBoost trials;
2. seed 2026 has not been used as a confirmation seed;
3. fusion member sets and weights differ between held-out directions on iron;
4. the final weights must be fitted inside an outer training split and then
   applied to an untouched outer evaluation, not reused from the same OOF
   selection pass;
5. local gain is not a platform forecast.

The next action is therefore confirmation, not submission: add seed 2026,
stabilize the time/iron fusion policy, and only then package if the confirmed
complete-package delta remains at least 0.05.
