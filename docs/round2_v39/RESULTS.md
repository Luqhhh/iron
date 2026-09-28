# V39 results: the first interior probes — a new best and a dead V7m endpoint

The user reported two of the five pending packages:

| package | time mixture | `(V36, N, V7m)` | platform score | vs B0 |
|---|---|---|---:|---:|
| `V32_TIME_A60V7_50` | `0.5*A60 + 0.5*V7m` | `(0.20, 0.30, 0.50)` | **96.3727** | **+0.0048** |
| `V32_TIME_A60V7_75` | `0.25*A60 + 0.75*V7m` | `(0.10, 0.15, 0.75)` | 96.3629 | −0.0050 |

**`V32_TIME_A60V7_50 = 96.3727` is the new platform best**, `+0.0048` over
`V18_B0_V12IRON_V7TIME = 96.3679`, and the target margin narrows to **−0.0273**.

## The interior hypothesis is confirmed

V32 warned that the `96.3992` ceiling covered only the *beyond-chord* region and
that the convex-hull interior was unmeasured and unconstrained. The first
interior probe is exactly such a point — the only measured point at `V7m = 0.5`
used to be `B0` — and it **beat** the previous best. So the boundary-probe
portfolio did what it was designed to do: the released weights were **not** the
platform optimum, and the optimum lies toward more `N` at the released `V7m`
weight.

The measured slope of that direction is **`+0.0384` per unit `N`** (from `B0` to
`I1`, trading `V36` for `N` at `V7m = 0.5`), while raising `V7m` from `0.50` to
`0.75` **cost** `0.0050`. Descent direction: more `N`, same `V7m`.

## The V7m endpoint is now provably dead

Adding `I1` and `I2` tightens the concave bound sharply:

| point | previous bound | **new bound** | source |
|---|---:|---:|---|
| pure `V7m` (`3_TIME_V100`) | 96.3832 | **96.3531** | `I1 -> I2` chord extended `lambda = 2` |
| `(0.30, 0.45, 0.25)` (`5_INTERIOR_A60V7_25`) | — | **96.3825** | `I1 -> I2` chord extended `lambda = -1` |
| `(0.00, 0.50, 0.50)` | — | 96.3804 | `B0 -> I1` chord extended |
| iron endpoint (`4_IRON_W100`) | 96.3846 | **96.3839** | measured iron line on the B0 time column |

`3_TIME_V100 <= 96.3531` is now **below the current best `96.3727`**: the `V7m`
direction has already turned before `v = 1.0`, so that package can no longer
improve the score. The bound moved from `96.3832` to `96.3531` because the
`I1 -> I2` chord is a much tighter constraint than the `A35 -> B0` chord.

The grid is now **33 bounded / 198 unbounded** points (was 22/209), and the
bounded-region maximum is **96.3825**, attained exactly at
`5_INTERIOR_A60V7_25 = (0.30, 0.45, 0.25)` — one of the still-pending packages.
With the iron endpoint the bounded ceiling is **96.3985**, still `0.0015` short of
`96.4`.

## What follows

* **No new package is built this round.** The highest-remaining bound is already
  covered by the pending `5_INTERIOR_A60V7_25`; the next probe should be chosen
  after its score is known, following the `N`-direction descent.
* The three remaining pending scores decide the rest: `5_INTERIOR_A60V7_25`
  (bound `96.3825`), `4_IRON_W100` (bound `96.3839`, and it composes additively
  with the time column), `3_TIME_V100` (now provably non-competitive).
* If the best of those lands below `96.4` — the bounded ceiling says at most
  `96.3985` — then the weight route is closed at the boundary of the measured
  family and the target requires a new model.

Budget: **0 fits, 0 packages, 0 uploads.** The two scores are user-reported, not
independently verified; the platform is deterministic so their differences are
exact. Implementation `src/bf_tap_r2/v39_interior_descent.py`, evidence
`local/runs/round2-v38/interior-descent-r1.json`.
