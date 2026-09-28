# V20 delivery: `V20_B0_PLLT_A325`

The user resumed optimization with the goal **platform 96.4**. V20 releases the
V17 dual-periodic time expert `P_LL_T` on top of the verified combined parent
`B0`. It is the first candidate since the local working gate was recorded as
over-conservative to **clear that gate with no exception**.

## Exact identity

* Candidate: **`V20_B0_PLLT_A325`**
* Recipe: parent `V18_B0_V12IRON_V7TIME` with only `pred_tap_time_len` replaced
  by `(1 - 0.325) * B0_time + 0.325 * P_LL_T_full`;
  `pred_tap_iron` is a **byte-identical copy** of the parent's field strings.
* Parent ZIP SHA-256:
  `532118c9dd92467d16d5205073fee6b056ca7f60c0a59b053308e0ba9e073017`
* Candidate ZIP SHA-256:
  **`f14f39df5474c8904639c0768bf8fa678f645f46fb0f987fbb84690a983eef57`**
* `result.csv` SHA-256:
  `661b3c9fb56b489f6b21628fa57222f27597f1db9042f506693953f8e2228055`
* Private run: `local/runs/round2-v20/release-r1`
* Frozen design: `configs/round2_v20/SPEC.yaml`,
  `docs/round2_v20/PREREGISTRATION.md`

## Why this candidate

`B0` combines the two platform-positive directions (`V12` iron, `V7` time) and
has conditional arithmetic `96.3679`. V18 showed the endpoint library is
exhausted on top of it; V19 closed the iron capacity lever at complete coverage.
The single surviving accurate expert was `P_LL_T`:

| property | value |
|---|---|
| incremental nested gain over B0 (V18, complete coverage) | `+0.004250` |
| accuracy ratio against B0 time / residual `rho` | `1.005` / `0.982` |
| V17 four-seed gains vs B0 (slot design) | `+0.002433 / +0.000706 / +0.006456 / +0.009330` |

## Four-seed check of the frozen design (the binding gate)

Re-evaluated for the released design on the B0 parent, at `alpha = 0.325`, from
the audited confirmation caches:

| seed | B0 score | candidate score | gain |
|---:|---:|---:|---:|
| 42 | 96.247974 | 96.253035 | **+0.005060** |
| 3407 | 96.247504 | 96.250982 | **+0.003478** |
| 7777 | 96.242873 | 96.244987 | **+0.002114** |
| 12011 | 96.233951 | 96.241727 | **+0.007776** |

Paired seed summary: mean `+0.004607`, sd `0.002431`, **paired LCB95
`+0.001746`**, 4/4 positive. Development mean candidate score **96.252008**,
above the frozen local working gate `96.25`. The gate is met **without any
candidate-specific exception**.

## G0 engineering

* **Replay.** One V17 development fold was refitted and reproduces the stored
  `P_LL_T` prediction **bit-identically** (`max |diff| = 0.0`, selected epoch
  `100`, 182 944 parameters). Nothing was substituted.
* **Full-data fit.** One `V17TimeRegressor` fit on all 2754 rows with
  inner-training-only preprocessing and epoch selection (selected epoch `75`,
  inner fit rows 2203), then a fresh refit at that epoch. Frozen V7 training
  settings; no new hyperparameters.
* **Cold audit.** Fresh-process inference through `v7_release infer`, with
  training-file reads prohibited: reverse/chunk/singleton invariance and
  bit-identical predictions (`max |diff| = 0.0`), cold CSV bytes equal to the
  packaged bytes.
* **Package readback** (independent, re-read from disk): 322 rows, 322 unique
  IDs, official template order; iron field strings **0 mismatches**; time blend
  recomputes to `0.0` maximum absolute difference; 0 negative rows
  (minimum `76.436`); archive holds exactly `result.csv` and equals the
  packaged payload.
* Targeted tests: 6 new in `tests/test_round2_v20_release.py`.

## Budget

| item | value |
|---|---:|
| replay fits | 1 |
| reference fits | 0 (audited caches only) |
| full-data fits | 1 |
| packages | 1 |
| desktop writes | 0 |
| agent uploads | 0 |

## Scope and limits

The expected platform effect is small in absolute terms (local
`+0.0035..+0.0051` over B0) and is **not** a forecast of `96.4`. The conditional
arithmetic of the combined package is still `0.0321` below the target, and the
concave two-line ceiling recorded in V18 is `96.3992`. `alpha = 0.325` is the
local optimum; V18 showed that the platform optimum of a blend weight can sit
elsewhere, so a platform-side line search on this direction is the natural next
step **after** this package reports. The user uploads and returns the score; the
assistant performs no upload.
