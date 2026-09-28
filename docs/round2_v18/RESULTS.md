# V18 results: combined-direction platform candidates and the reachable ceiling

The user resumed optimization with the goal **platform 96.4**. The registered
platform best is the user-reported `V12_IRON_JOINT_PLR001_A50 = 96.3526` (not
independently verified), so the measured gap is `0.0474`. This round is a
zero-fit platform-side line search on the two directions the platform has
already scored positive, plus the exact arithmetic that bounds that family.

Pre-registration: [PREREGISTRATION.md](PREREGISTRATION.md),
`configs/round2_v18/SPEC.yaml`. Implementation:
`src/bf_tap_r2/v18_compose.py`, `src/bf_tap_r2/v18_surface.py`. Private run:
`local/runs/round2-v18/packages-r1`, analysis
`local/runs/round2-v18/surface-r1.json`.

## G0 engineering

Seven original ZIPs were read by SHA-256 and structurally audited:

| source | role | ZIP SHA-256 (prefix) |
|---|---|---|
| A35 | composition parent, `alpha = 0.35` | `b4e1fc2d1287a213` |
| A60 | composition parent, `alpha = 0.60` | `d5092d400fb5cc62` |
| V7 | composition parent (time) | `4382523c7bd68897` |
| V12 | composition parent (iron) | `a1c205a6722da397` |
| A45/A72/A85/A100 | N-line cross-check | `4c12bedae707c306`, `808f00a9a3385b62`, `2af4d68323ba2cf3`, `865f7290db001d41` |

All 322 IDs are unique, in official template order, with identical metadata
columns. The isolation checks hold: V12 time strings are byte-equal to A35 time
strings, V7 iron strings are byte-equal to A35 iron strings.

The member endpoints were recovered from the field strings and cross-checked
against every other point of the N line:

* `V36_time = (A35_time - 0.35 N)/0.65`, `N` solved from `(A35, A60)`;
* re-deriving `A45`, `A72`, `A85`, `A100` gives a worst absolute difference of
  **1.14e-13** (declared tolerance `1e-9`), so the recovered columns are the
  same affine family the platform scored;
* `V12_member_iron = 2 V12_iron - A35_iron` and
  `V7_member_time = 2 V7_time - A35_time`, both strictly positive
  (minimum 327.899 and 76.297, zero negative rows), so no clipping is applied.

Five packages were written and then independently re-read from disk
(`audit.json`):

| package | iron column | time column | ZIP SHA-256 |
|---|---|---|---|
| `V18_B0_V12IRON_V7TIME` | V12 copy | V7 copy | `532118c9dd92467d16d5205073fee6b056ca7f60c0a59b053308e0ba9e073017` |
| `V18_TIME_V75` | V12 copy | `0.25 A35_t + 0.75 V7m` | `4188acbed0b6d7affde7a6778e49a6aec9236a982d3c93537fff801712516df7` |
| `V18_TIME_V100` | V12 copy | `V7m` | `9845a604e697f6a51cfa8e8b502bf0393cdbc0857f884fbc6fe037daf4a763d1` |
| `V18_IRON_W75` | `0.25 V36_i + 0.75 V12m` | V7 copy | `dbd3b6b241190ec4bcc3143d30590dc134cdbe3acc76c3a9e0c1781874c585f2` |
| `V18_TIME_A60V7_50` | V12 copy | `0.5 A60_t + 0.5 V7m` | `54864561c057ae71f7779b15099750f7caf0141c149d609ae281dbaf9ef3aaa7` |

Audit result: 322 rows, 322 unique IDs, template order, copied columns with
**0 byte mismatches**, blend columns with relative read-back difference `0.0`,
all predictions finite and non-negative, `x`-mode ZIP creation, no overwrite.
No label was read, no model was fitted, no cold-inference step applies because
no member was refitted. Ten targeted tests pass
(`tests/test_round2_v18_compose.py`).

## G1 the reachable ceiling of the current two directions

The documented metric is `max(0, 100 - 50*WMAPE_iron - 50*WMAPE_time)`, so the
two columns contribute additively. For fixed endpoints the score is concave in
the blend weight, so extending a measured secant outside its interval is a
valid upper bound. Using the measured weights:

* iron line: `+0.0160` at weight `0.50` gives a slope of at most `0.0320`, so
  full V12m is worth at most `+0.0320` over A35;
* V7 time line: `+0.0153` at weight `0.50` gives at most `+0.0938` over V36.

The joint ceiling of the two lines is therefore **96.3992**, i.e. **0.0008
below 96.4** (`local/runs/round2-v18/surface-r1.json -> platform_bounds`). That
is an upper bound, not a forecast; the realised optimum is lower because the
local surface has an interior maximum. It also does not bound a new expert or a
different endpoint.

`B0` itself is the strongest untested package in this family. Its conditional
arithmetic is the recorded `96.3526 + 96.3519 - 96.3366 = 96.3679`, which is
`+0.0153` over the current best and `0.0321` short of 96.4.

## G1 local complete-coverage surface (descriptive only)

Computed on the existing complete five-fold caches at split seeds 42/3407, one
seed per candidate, no cross-seed OOF averaging (`surface-r1.json`):

| design | local mean delta vs A35 | seed 42 | seed 3407 |
|---|---:|---:|---:|
| `V18_B0_V12IRON_V7TIME` | **+0.038976** | +0.037122 | +0.040831 |
| `V18_TIME_V75` | +0.034683 | +0.032992 | +0.036374 |
| `V18_IRON_W75` | +0.031580 | +0.029221 | +0.033939 |
| `V18_TIME_A60V7_50` | +0.024957 | +0.024566 | +0.025348 |
| `V18_TIME_V100` | +0.020739 | +0.020325 | +0.021153 |

Local shape agrees that `B0` is the best of the five, and it says the released
0.50/0.50 weights are already the interior optimum. The platform N-line
contradicts the local N optimum in both location (`0.20 -> 0.60`) and slope
sign, so this table is **not** used to choose weights; it is recorded only as a
diagnostic.

**The existing library is exhausted on top of B0.** A nested two-column screen
of 31 reproducible members (grid `0..1` in `1/40` steps, weight fitted on the
other split seed) leaves only:

| member | target | incremental gain over B0 | weights |
|---|---|---:|---|
| `v17/P_LL_T` | time | +0.004250 | 0.325 / 0.350 |
| `v15/joint-task_gated_experts` | iron | +0.002856 | 0.175 / 0.450 |
| `v11 tap_time_len-plr_quniform` | time | +0.002699 | 0.200 / 0.175 |
| `v9 tap_time_len-realmlp_td` | time | +0.002634 | 0.175 / 0.125 |

22 of 31 members are `<= 0.000` (weight driven to zero). No existing member
leaves a measurable increment on `B0`, which upgrades the closure statement from
"no gain on the delivered incumbent" to "no gain on the stronger combined
incumbent". Reaching 96.4 therefore needs a **new model**, not another blend.

## Upload recommendation (user uploads; agent uploads 0)

1. **`V18_B0_V12IRON_V7TIME`** — the priority. It combines the two already
   platform-positive directions and is the only untested package with a
   conditional arithmetic above the current best (`96.3679`, `+0.0153`).
2. `V18_TIME_V75` — tests whether the platform wants more of the validated V7
   direction than the local 0.50 (the N line did want more).
3. `V18_TIME_A60V7_50` — tests N and V7m together.
4. `V18_IRON_W75` — tests a higher V12 iron weight.
5. `V18_TIME_V100` — the V7m endpoint.

The hedges are lower-mean variants; their positive expectation comes only from
the platform keeping the best score. Under the exact two-line ceiling none of
the five is expected to exceed 96.4, and `V18_TIME_V75`/`V18_IRON_W75` are
expected to be below `B0`. Recommend uploading `V18_B0_V12IRON_V7TIME` first;
the hedges are optional and should not be treated as promotions.

## Next round (registered, not run here)

Two levers survive this round's evidence, both requiring new fits because the
composable endpoints are exhausted:

1. **A better time expert.** `v17/P_LL_T` is the strongest incremental member
   (`+0.00425` over B0, four positive split seeds, local working gate missed by
   `0.000691`). A full-data fit plus a combination with `v11 plr_quniform`
   (`+0.00270`) and `v9 realmlp_td` (`+0.00263`) is the best available route to
   a larger time gain; their residual correlations with B0 are `0.982/0.960/`
   `0.940`, so a joint blend is the point.
2. **One local/platform conflict worth one test slot.** The local surface
   prefers removing the N component entirely from the V7 blend
   (`time = 0.5 V36_t + 0.5 V7m`, local `+0.046906` vs A35, i.e. `+0.0079` over
   B0), while the platform rates N highly on its own line. That disagreement is
   not resolvable offline and is not used to pick this round's designs.

## What this round does not do

No new fit, no desktop write, no upload, no gate relaxation, no reopening of a
closed route. The `96.4` objective is **not achieved** and the current best
remains the user-reported V12 `96.3526`. A larger gain requires a new model;
the surviving local candidates point at a better **time** expert
(`v17/P_LL_T` at `+0.0043` over B0) and, secondarily, the joint iron expert
(`v15` gated at `+0.0029`).
