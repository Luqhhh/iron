# iron 96.4 — V21: accurate-expert time mixes on the B0 parent (zero fit)

Design date: 2026-09-28. Base commit `ce9a576`. Status at design time: no V21
fit has run, no package has been written, no upload has been made.

## 1. Why this round exists

`B0 = (V12 iron, V7 time)` has conditional platform arithmetic `96.3679`. V20
added the accurate P-LL time expert at the locally selected weight. V19 closed
the iron capacity lever. The remaining positive signal is the **accurate,
decorrelated experts** (V18 screen: `P_LL_T` ratio `1.005`, `rho 0.982`).

This round asks the next question up from V20: **what is the best time column
that can be built from members whose full-data predictions already exist?**
That set is `{V36, N, V7m}` (recovered from the A35/A60/V7 ZIPs) plus
`P_LL_T_full` (the V20 full-data model). No new model fit is required.

A nested convex-weight optimisation over those endpoints (weights fitted on the
other split seed, WMAPE objective) gives a clear answer: the optimum **drops N
entirely** and puts about `0.40 V36 + 0.25 V7m + 0.35 P_LL`. That is the
opposite of `B0`, which is `0.325 V36 + 0.175 N + 0.50 V7m`. The platform rated
N highly **in isolation** on its own line, so the disagreement is real and is
the object of this round: the three packages below span the N level from `0` to
the V20/B0 level, and the platform keeps the best score.

## 2. Four-seed evidence (frozen weights, complete coverage)

Gains are package-score points against `B0` on the frozen four split seeds; the
endpoints are the recorded complete-coverage OOF columns (development caches for
42/3407, confirmation caches for 7777/12011), and the weights are **frozen
before packaging**, not refitted per seed.

| package | time weights | 42 | 3407 | 7777 | 12011 | mean | paired LCB95 | dev mean score | gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|:--:|
| `V21_TIME_LOCAL` | `V36 .40 / V7m .25 / P-LL .35` | +0.010431 | +0.013954 | +0.009948 | +0.016577 | **+0.012728** | **+0.009048** | 96.259932 | pass |
| `V21_TIME_N10` | `V36 .30 / N .10 / V7m .20 / P-LL .40` | +0.007523 | +0.007363 | +0.007442 | +0.012446 | +0.008694 | +0.005749 | 96.255183 | pass |
| `V21_TIME_A35` | `A35 .45 / V7m .15 / P-LL .40` | +0.004490 | +0.003145 | +0.006008 | +0.009790 | +0.005858 | +0.002481 | 96.251557 | pass |

Every design is 4/4 positive with a positive paired seed-level LCB95 and a
development mean package score above the frozen `96.25` working gate, with no
candidate-specific exception. `V21_TIME_LOCAL` is the strongest local evidence
the project has recorded (mean `+0.012728`, LCB `+0.009048`), exceeding the V5
winner's four-seed record (`+0.00972` / `+0.00753`).

## 3. Frozen packages (0 fits, 3 packages)

* Parent: `V18_B0_V12IRON_V7TIME`, ZIP SHA-256
  `532118c9dd92467d16d5205073fee6b056ca7f60c0a59b053308e0ba9e073017`.
* Iron: **byte-identical copy of the parent's `pred_tap_iron` strings** in every
  package. No iron change is made in this round.
* Time: `sum(w_i * endpoint_i)` over the endpoints above, with `P_LL_T` taken
  from the V20 full-data model (`local/runs/round2-v20/release-r1/full/cold.npy`,
  SHA-256 pinned) and `V36/N/V7m` recovered field-exactly from the A35/A60/V7
  ZIPs as in V18.
* Weights are the frozen values in the table; **no alpha scan** is performed and
  no post-hoc weight is added.

## 4. Required verification

1. all source ZIP SHA-256 values match the V18 frozen constants;
2. the V20 full-data member file hash matches the V20 verification record;
3. endpoint recovery re-derives `A45/A72/A85/A100` to `<= 1e-9` (the V18 audit);
4. each package: 322 rows, 322 unique IDs, official template order, iron strings
   byte-identical to the parent, blend recomputes to `0.0` on read-back, all
   predictions finite and non-negative, `x`-mode ZIP, no overwrite;
5. an independent post-write audit re-reads every ZIP from disk and re-runs
   steps 1-4.

## 5. Budget and boundaries

| item | value |
|---|---:|
| new fits | **0** |
| new packages | 3 |
| desktop writes | 0 |
| agent uploads | 0 |

The user uploads and returns scores. This round does not reopen any closed route
and does not relax a gate. The objective is not claimed.

## 6. Limits

The `V21_TIME_LOCAL` design is a **gamble on the N question**: it removes the N
component that the platform rated highly on its own line, in exchange for a much
larger local gain. `V21_TIME_A35` is the conservative N-preserving hedge and
`V21_TIME_N10` sits between them. Local magnitude does not forecast the platform
(V5 delivered `4.19x` local; V7 and V12 delivered about `0.7-1.0x`), so no score
is forecast. The conditional `B0` arithmetic `96.3679` plus even the strongest
local gain is not guaranteed to reach `96.4`.
