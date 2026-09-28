# V22 results: the reproducible library is exhausted against the strongest incumbent

Purpose: the V18 member screen used `B0` as the incumbent. V21 then produced a
stronger incumbent — the time column `0.40 V36 + 0.25 V7m + 0.35 P-LL` on the
`B0` iron column. This round re-runs the incumbent-relative check against that
stronger package over the **whole reproducible library**, including the
CatBoost/XGBoost/LightGBM/kernel/expression and `v36_dev_experts` members that
the V18 screen never covered. Zero fits.

Spec `configs/round2_v22/SPEC.yaml`; implementation
`src/bf_tap_r2/v22_saturation.py`; private evidence
`local/runs/round2-v22/saturation-r1.json`.

## Method

* Library: `bf_tap_r2.v5_library.build_candidate_library` at the frozen V5 spec —
  **110 complete-coverage time columns and 114 complete-coverage iron columns**,
  all two-seed (`42`, `3407`), including the 83 `v2:*` columns per target.
* Incumbent: the V21 time column and the `B0` iron column (the V12 iron).
* Statistic: nested two-column blend, weight fitted on the *other* split seed
  over a 41-point `0..1` grid, scored as package-score points
  (`50 * (WMAPE_incumbent - WMAPE_blend)`), exactly the binding test recorded in
  the closure notes. Fold level is not used.

## Result

| target | members | best member | best gain | members above `+0.0005` |
|---|---:|---|---:|---:|
| `tap_time_len` | 110 | `v36_dev_experts:v36-s1-D-0048` | **+0.000158** | **0** |
| `tap_iron` | 114 | `v36_dev_experts:v36-s1-D-0029` | **+0.001491** | 5 |

* The time column is **saturated**: no member of the whole reproducible library
  adds even half a thousandth of a point on top of the V21 incumbent.
* The iron column has a small residual: `D-0029` at `+0.0015` (weights
  `0.15/0.125`, stable across the two held-out seeds), with four further
  `v36_dev_experts` D/O members between `+0.0006` and `+0.0009`. A nested convex
  optimum over `{V36, V12m, D-0029, D-0013, D-0025}` reaches only
  `+0.0018` (seed 42 `+0.0018`, seed 3407 `+0.0019`) and stays there under a
  `V12m >= 0.25..0.50` floor, i.e. it is real but an order of magnitude below the
  V21 time gain.

## Why the library cannot be squeezed further

A nested convex simplex over 14 endpoints (adding the best D/O and CatBoost
members to `{V36, N, V7m, P-LL, v9td, v11q, v15gt}`) gives **`+0.011983`** over
`B0`, *below* the 7-endpoint simplex's `+0.013048`. More library members do not
raise the attainable column; they only add selection variance, because the
`v36_dev_experts` are already inside the `V36` endpoint and the `v3_refine`
members are dominated. This is the member-level restatement of the V6 §7
stacking closure, now measured against the strongest incumbent instead of the
weak one.

## Decision

* Member search against the incumbent is **closed**: the time column gains
  nothing, the iron column at most `+0.0018`. Adding those members would require
  a full-data fit plus a ten-fold four-seed confirmation for a gain far inside
  the local resolution, so **no fits and no packages** are spent on them.
* The V21 portfolio already spans the only unresolved question — the platform's
  N weight — at N levels `0 / 0.10 / 0.1575` (plus the earlier `B0`/`A60V7`
  hedges at `0.175`/`0.30`).
* What remains is (a) platform feedback on the pending packages, which resolves
  the N weight, and (b) a genuinely new model family, which is a different
  problem class with an unknown prior.

## G0

No fit, no package, no desktop write, no upload. The screen is a pure read of
private OOF caches. Targeted tests: 5 in `tests/test_round2_v22_saturation.py`.

The `>96.4` objective is **not** achieved; the registered platform best remains
the user-reported V12 `96.3526`.
