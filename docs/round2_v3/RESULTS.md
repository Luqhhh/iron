# Round2 V3 Local Search: First Batch and Seed-2026 Confirmation

Status: **first-batch local search complete; seed-2026 outer confirmation completed; no package and no upload**.

## Executed budget

| family | planned | complete | note |
|---|---:|---:|---|
| CatBoost | 240 | 240 | coarse seed 42 / folds 0–1 |
| LightGBM | 40 | 40 | coarse seed 42 / folds 0–1 |
| XGBoost | 40 | 40 | completed after adding `xgboost==2.1.4` to the `round2` optional extra |
| MLP | 20 | 20 | coarse seed 42 / folds 0–1 |
| kernel | 20 | 20 | coarse seed 42 / folds 0–1 |
| expression | 40 | 40 | CatBoost backbone, feature/target-expression axis |
| **total** | **400** | **400** | — |

Refinement used the top 8 overall trials plus the best non-CatBoost trial from
each available family, up to 12 per target.  Full 5-fold OOF was computed for
seeds 42 and 3407.  Thirty-two CatBoost prediction arrays were reused from the
CatBoost-only refine pass; 16 new `(trial, seed)` fits covered the selected
LightGBM/MLP/kernel/expression entries.  The top three XGBoost trials per target
were also refined to full 5-fold OOF: their best mean WMAPE was `0.04203`
(iron) and `0.04659` (time), worse than the CatBoost candidates, and none
entered the selected fusion member set.

## Single-model findings

| target | candidate | seed 42 WMAPE | seed 3407 WMAPE | mean WMAPE | previous local strong single |
|---|---|---:|---:|---:|---:|
| iron | `v3-catboost-tap_iron-0075` | 0.03909385 | 0.03925472 | **0.03917428** | DJ 0.03920512 |
| time | `v3-catboost-tap_time_len-0021` | 0.03921803 | 0.03991548 | **0.03956676** | B3/full-time local reference 0.04048214 |

The iron single-model gain is marginal.  The time improvement is the more
important single-model signal.  XGBoost/LightGBM/MLP/kernel/expression did not
beat CatBoost on full OOF; they are retained as negative or complementary
evidence, not silently removed.


## Direct single-model pair confirmation

The borderline direct pair check was requested explicitly:

- iron: `v3-catboost-tap_iron-0075`
- time: `v3-catboost-tap_time_len-0021`
- weights: fixed 1.0 / 1.0, no learned fusion.

Reconstructed per-seed results:

| seed | direct package score | same-seed AJ3-like reference | delta |
|---|---:|---:|---:|
| 42 | 96.083953 | 96.001405 | +0.082548 |
| 3407 | 96.041314 | 96.023372 | +0.017942 |
| 2026 | 96.065114 | 96.027838 | +0.037276 |
| three-seed average | 96.063460 | 96.017538 | +0.045922 |

The earlier `+0.05056` used the two-seed AJ3 local reference as a single
reference point.  Propagating the same-seed reference and adding seed 2026 gives
a three-seed average delta of **+0.045922**, below the 0.05 admission gate, and
the untouched seed-2026 delta is **+0.037276**.  Therefore the direct single-model
pair **does not qualify** as a formal V3 pending strategy after confirmation.
Only the frozen weighted fusion strategy remains above the gate and stable on
seed 2026.

## Nested cross-seed fusion, seeds 42 and 3407

Two nested leave-one-seed-out runs were compared.

| candidate library | iron WMAPE | time WMAPE | local package score | delta vs AJ3 local reference | triage |
|---|---:|---:|---:|---:|---|
| CatBoost-refined only | **0.03856469** | **0.03942060** | **96.100735** | **+0.088347** | candidate_pool |
| all-family refined | 0.03857898 | 0.03942060 | 96.100021 | +0.087633 | candidate_pool |

Adding XGBoost/LightGBM/MLP/kernel/expression did not improve the nested package
score over the CatBoost-refined-only pool.  XGBoost in particular was not
selected by the outer weight fit.

## Frozen outer weights and seed-2026 confirmation

The selected outer procedure was:

1. fit the candidate pool and simplex weights on seeds **42 and 3407 only**;
2. freeze the selected members and weights;
3. reconstruct the required non-V3 members on an independently derived seed
   **2026** fold split without using 2026 labels;
4. evaluate the frozen complete package on seed 2026.

The frozen weights were:

- iron:
  - `AJM1` 0.338853,
  - V3 CatBoost `0107` 0.213013,
  - `J1` 0.232741,
  - V3 CatBoost `0043` 0.215392,
  - V3 CatBoost `0075` had a zero weight in the frozen fit and was omitted from
    the final prediction.
- time:
  - V3 CatBoost `0021` 0.668858,
  - `AORD` 0.174110,
  - `T1_B3_PREFIX1000` 0.157032.

Seed-2026 result:

| quantity | value |
|---|---:|
| seed-2026 AJ3-like local reference score | 96.027838 |
| frozen package local score on seed 2026 | **96.116174** |
| delta vs same-seed reference | **+0.088335** |
| delta vs original AJ3 local reference 96.012388 | +0.103786 |
| triage | candidate_pool |

The original `+0.088` gain was therefore retained in the frozen outer-weight
confirmation on seed 2026.  The result is still below the `0.15–0.20` sprint
band and is not a platform forecast.

## Limitations

- Local OOF is repeatedly used for development; seed 2026 is the first untouched
  fold seed for this fusion strategy, but the labels are not a new hidden
  competition set.
- Platform transfer is not guaranteed; the current platform reference remains
  AJ3 `96.1259`, and no V3 package has been submitted.
- XGBoost is now installed as an optional `round2` dependency; its 40 coarse
  trials are complete and its refined candidates are recorded, but it did not
  improve the selected fusion.
- No package, independent cold release, or platform upload was performed.

## Next action

The next step is package preparation only if the user explicitly requests it:
freeze the above members/weights, retrain the selected members on the full
training data with the same recipes, rebuild the time/iron columns, and run the
existing independent cold-release checks.  Until then the V3 result remains a
candidate-pool local evidence, not a submission.
