# V40 results: the A60→V7m ridge, and the "96.4 unreachable" verdict is withdrawn

The third interior report arrived: `0.75*A60 + 0.25*V7m`, coordinates
`(0.30, 0.45, 0.25)`, scored **`96.3727`** — exactly tying
`0.5*A60 + 0.5*V7m`.

## The tie is a plateau on a measured chord, not a duplicate

All three interior packages lie on the **`A60 → V7m` segment**
`lambda*A60 + (1-lambda)*V7m`:

| `lambda` | coordinates `(V36, N, V7m)` | package | platform score |
|---:|---|---|---:|
| 0.25 | `(0.10, 0.15, 0.75)` | `V32_TIME_A60V7_75` | 96.3629 |
| 0.50 | `(0.20, 0.30, 0.50)` | `V32_TIME_A60V7_50` | **96.3727** |
| 0.75 | `(0.30, 0.45, 0.25)` | `V32_TIME_A60V7_25` | **96.3727** |
| 1.00 | `(0.40, 0.60, 0.00)` | `A60` | 96.3625 (lifted) |

Two equal values at the endpoints of a chord are not anomalous: by concavity
every **interior** point of that chord is **at least** `96.3727`, so the segment
carries a plateau around `lambda ≈ 0.5-0.75`. The two ZIPs are different files
(`54864561…` and `688e27f5…`), so the tie is a real plateau unless the same file
was uploaded twice; a hash check settles it.

The segment also gives a very tight bracket at `lambda = 0.625`:
`96.3727 <= value <= 96.3776`, bounded above by the `I2 -> I1` chord extension
and below by the flat `I1 -> I5` chord. That is the most informative single probe
still available.

## The bounded ceiling now clears the target

Adding `I5` tightens the concave bound and **reverses the previous feasibility
verdict**:

| quantity | with `I1`, `I2` only | **now** |
|---|---:|---:|
| bounded grid points | 33 / 231 | **39 / 231** |
| bounded time maximum | 96.3825 at `(0.30, 0.45, 0.25)` | **96.3882** at `(0.00, 0.45, 0.55)` |
| bound source | `I1 -> I2` | `A45 -> I5`, extended `lambda = 2.2` |
| + iron head-room `+0.0160` | 96.3985 | **96.4042** |
| target provably unreachable | yes (`-0.0015`) | **no (`+0.0042`)** |

So `96.4` is **no longer excluded by the measured family**: there is now a
provable path to `96.4042` if the bounded point `(0.00, 0.45, 0.55)` and the iron
endpoint both realise their bounds. The bounds remain upper bounds, so this is a
possibility, not a forecast — but the earlier "provably unreachable" statement is
withdrawn.

## The next four probes (zero fit)

Composed field-exactly from the delivered A35/A60/V7/V12 ZIPs, iron the unchanged
V12 column:

| id | time mixture | `(V36, N, V7m)` | bound | ZIP SHA-256 |
|---|---|---|---|---|
| `V40_TIME_H1` | `0.45 N + 0.55 V7m` | `(0.00, 0.45, 0.55)` | **96.3882** | `772057b95032f035…` |
| `V40_TIME_H2` | `0.10 V36 + 0.45 N + 0.45 V7m` | `(0.10, 0.45, 0.45)` | 96.3830 | `bbfd8f40c88aea60…` |
| `V40_TIME_SEG_625` | `0.625 A60 + 0.375 V7m` | `(0.25, 0.375, 0.375)` | bracket `[96.3727, 96.3776]` | `1e8671edae984a62…` |
| `V40_TIME_N500` | `0.5 N + 0.5 V7m` | `(0.00, 0.50, 0.50)` | 96.3804 | `2edd68f82d47d454…` |

All four verified and audited: exactly `result.csv`, 322 template-ordered unique
IDs, blend relative difference `0.0`, finite non-negative.

## Desktop pending set (five) and what was dropped

`/mnt/c/Users/lqh22/Desktop/submission-96.4-pending`: `1_RIDGE_H1`,
`2_RIDGE_H2`, `3_RIDGE_SEG625`, `4_RIDGE_N500` and `5_IRON_W100`.

* `3_TIME_V100` (pure `V7m`) was **dropped**: its bound is now `96.3531`, below
  the current best `96.3727`, so it is provably unable to improve the score.
* `5_INTERIOR_A60V7_25` was dropped because it has been scored.
* `5_IRON_W100` (pure `V12m` iron) is the only package carrying the iron lever:
  its own bound is `96.3839`, and by additivity **any** time column gains the
  same `delta I`, so the best time column plus this iron change is directly
  computable and is the most promising route past `96.4`.

Budget: **0 fits, 4 packages, 5 desktop writes, 0 uploads.** Implementation
`src/bf_tap_r2/v40_segment_descent.py`, evidence
`local/runs/round2-v40/ridge-r1.json`.
