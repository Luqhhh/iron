# iron 96.4 — V20: P-LL time expert release on the combined B0 parent

Design date: 2026-09-28. Base commit `98a1d7d`. Status at design time: no V20
fit has run, no package has been written, no upload has been made.

## 1. Why this candidate

V18 established that the combined package
`B0 = (V12 iron, V7 time)` is the strongest untested platform candidate
(conditional `96.3679`) and that the existing endpoint library is exhausted on
top of it. The one surviving **accurate** expert is the V17 dual-periodic time
network `P_LL_T`:

* incremental nested gain over B0 `+0.00425` (V18 screen, complete coverage,
  weights fitted on the other split seed);
* accuracy ratio `1.005` against B0's time column and residual `rho = 0.982`, so
  it is in the regime that earns blend weight (V19 showed members with ratio
  `> 1.10` are driven to weight zero);
* **four-seed confirmation passed in V17**: gains `+0.002433 / +0.000706 /
  +0.006456 / +0.009330` vs B0, mean `+0.004731`, paired seed-level
  `LCB95 = +0.000143`, 4/4 positive
  (`local/runs/round2-v17/confirmation-r1/summary.json`).

V17 declined to release it only because the frozen local working gate `96.25`
was missed by `0.000691` **on the A35-parent slot design**
(`candidate_T = 0.5*A35_T + 0.5*P_LL_T`, dev package score `96.249309`).

On the **B0 parent** the same expert clears the frozen gate with no exception.
Complete-coverage development scores for
`time = (1-a)*B0_time + a*P_LL_T`, iron `= V12 iron`:

| `a` | seed 42 | seed 3407 | gate `96.25` |
|---:|---:|---:|:--:|
| 0.325 | **96.253035** | **96.250982** | both pass |
| 0.350 | 96.253082 | 96.250943 | both pass |
| 0.500 (V17 slot relative to B0) | 96.252170 | 96.249981 | seed 3407 fails |

## 2. Frozen design

* Parent: `V18_B0_V12IRON_V7TIME`, ZIP SHA-256
  `532118c9dd92467d16d5205073fee6b056ca7f60c0a59b053308e0ba9e073017`.
* Iron column: **byte-identical copy of the parent's iron strings**.
* Time column: `(1-a)*B0_time + a*P_LL_T_full`, with **`a = 0.325` frozen from
  the lower of the two held-out-seed nested optima (`0.325 / 0.35`)**. No alpha
  scan in V20.
* Expert: `V17TimeRegressor` with the frozen recipe
  `{target: tap_time_len, kind: single, selection: standardized_mae,
  projection: false, dual_scales: [0.01, 0.01]}` and the frozen V7 training
  settings (`configs/round2_v7/SPEC.yaml -> training`). No new hyperparameters,
  no external data, no pretrained weights.

## 3. Required steps and gates

1. **Replay check.** One V17 development fold is refitted and must reproduce the
   stored `P_LL_T` prediction bit-identically; otherwise stop.
2. **Four-seed check for the frozen `a = 0.325` design.** Reconstruct the B0
   reference columns at derived seeds `7777`/`12011` from the recorded caches
   (V5 replication `base`/`candidate` for the N line, V7 confirmation member,
   V12 confirmation joint member) and evaluate the `a = 0.325` design against
   B0 with the frozen promotion gate: four seeds, all positive, paired seed-level
   `LCB95 > 0`, fold level descriptive. If the reference columns cannot be
   reconstructed to the recorded `b0_score` values within `1e-9`, stop and record
   the limitation; do not substitute a different reference.
3. **Full-data fit.** One `V17TimeRegressor` fit on all 2754 training rows with
   the frozen inner-training-only preprocessing and epoch selection, then a
   fresh refit at the selected epoch.
4. **Cold audit.** Fresh-process inference, reverse/chunk/singleton invariance,
   no training-file reads.
5. **Package and verify.** 322 IDs in template order; unchanged iron column
   byte-identical (`0` mismatches); time blend arithmetic on read-back within
   `1e-12` relative; finite, non-negative; `x`-mode ZIP; final SHA-256 recorded.

## 4. Budget and boundaries

| item | value |
|---|---:|
| replay fits | 1 |
| four-seed reference fits | 0 (cached columns only; stop if not reproducible) |
| full-data fits | 1 |
| packages | 1 |
| desktop writes | 0 |
| agent uploads | 0 |

Private outputs under `local/runs/round2-v20`. The user uploads and returns the
score. This round does not reopen any closed route. It is a single-target
replacement on a verified parent, and it is the first candidate to clear the
frozen local working gate since the gate was recorded as too conservative.

## 5. What it does not claim

`a = 0.325` is the local optimum, and V18 showed that the local and platform
optima of a blend weight can differ in both location and sign. The expected
platform effect is small in absolute terms (local `+0.0035..+0.0051` over B0)
and is not a forecast of `96.4`. The conditional arithmetic of the combined
package (`96.3679`) is still `0.0321` below the target, and the concave
two-line ceiling recorded in V18 is `96.3992`.
