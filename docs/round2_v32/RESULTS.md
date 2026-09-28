# V32 results: the concave bound covers the beyond-chord region only

Two zero-fit deliverables:

1. a rigorous evaluation of the concave secant bound over the whole time simplex
   `{V36, N, V7m}` plus the iron-axis head-room
   (`src/bf_tap_r2/v32_family_ceiling.py`,
   `local/runs/round2-v32/family-ceiling-r1.json`);
2. three **interior** time-simplex probes composed from the delivered ZIPs
   (`src/bf_tap_r2/v32_interior_probes.py`,
   `local/runs/round2-v32/interior-probes-r1`).

## 1. What the bound is

The score is `100 - 50*WMAPE_iron - 50*WMAPE_time`, the two targets add, each
absolute residual is convex in the prediction, and the prediction is linear in
the mixture weights. The score is therefore **concave** in the weights, and a
secant extended *outside* its chord is an upper bound. This is what the earlier
"hedge / boundary probe" packages exploit.

Lifting every recorded score to the V12-iron reference with the measured
`+0.0160` iron effect gives the measured design set:

| point | `(V36, N, V7m)` | score with V12 iron |
|---|---|---:|
| V36 | `(1.00, 0.00, 0.00)` | 96.2894 |
| A35 | `(0.65, 0.35, 0.00)` | 96.3526 |
| A45 | `(0.55, 0.45, 0.00)` | 96.3598 |
| A60 | `(0.40, 0.60, 0.00)` | 96.3625 |
| A72 | `(0.28, 0.72, 0.00)` | 96.3585 |
| **B0** | `(0.325, 0.175, 0.50)` | **96.3679** |

## 2. The bound, and where it does **not** apply

Extending the `A35 -> B0` chord to `lambda = 2` reaches the pure `V7m` vertex
`(0, 0, 1)`, giving `96.3526 + 2*(96.3679 - 96.3526) = 96.3832`. Adding the
bounded iron head-room (`w=0.5 -> w=1` at most another `+0.0160`) gives a family
ceiling of **96.3992**, which is `0.0008` **below** the target.

**Critical scope correction.** Only points that lie *beyond a measured chord*
have a computable bound. On a `0.05` grid the simplex has 231 points and only
**22** are bounded (those with `V7m` weight above `0.5` along the `A35 -> B0`
direction, plus edge extensions); the other **209 interior points are
unbounded**. The bound therefore covers the beyond-chord region that contains
the pure-`V7m` and iron-endpoint corner — **it is not a proof over the whole
simplex**, and an interior mixture that combines a high N weight with a high
`V7m` weight is unmeasured and unconstrained. (An earlier statement in this
session that the bound closed the whole simplex is corrected here.)

## 3. The interior probes

Because the interior is the only unconstrained part of the measured endpoint
span, three interior packages were composed field-exactly from the delivered
A35/A60/V7/V12 ZIPs. `pred_tap_iron` is the V12 original string in all three, so
the iron axis stays at the measured `w = 0.5`.

| id | time mixture | `(V36, N, V7m)` | local dev mean | ZIP SHA-256 |
|---|---|---|---:|---|
| `V32_TIME_A60V7_50` | `0.5*A60 + 0.5*V7m` | `(0.20, 0.30, 0.50)` | 96.233720 | `54864561c057ae71f7779b15099750f7caf0141c149d609ae281dbaf9ef3aaa7` |
| `V32_TIME_A60V7_75` | `0.25*A60 + 0.75*V7m` | `(0.10, 0.15, 0.75)` | 96.237066 | `8eb5ab2c81032217a0ba19a8280b8f8fc35f39a59b02411671b47634b8a097dc` |
| `V32_TIME_A60V7_25` | `0.75*A60 + 0.25*V7m` | `(0.30, 0.45, 0.25)` | 96.219096 | `688e27f58cd5ee4a1e8ba01cf4fc05530236ab58001655a6071894b6ba36ff11` |

`V32_TIME_A60V7_50` is byte-identical to the earlier `V18_TIME_A60V7_50`
(`54864561…`), an independent consistency check of the composition.

All three score **below** `B0` locally (`-0.014 / -0.011 / -0.029`) — as expected,
because the local OOF optimum removes `N` — and local magnitude has failed twice
already. Their platform values are what the probes are for: the interior is
unbounded by any current measurement, so it is the only part of the measured
endpoint span that could exceed the `96.3992` beyond-chord ceiling.

## 4. Verification and budget

Independent read-back of every package: exactly `result.csv`; 322 unique IDs in
official template order; iron strings byte-identical to the V12 source; blend
recomputation relative difference `0.0`; finite and non-negative. The
`A45/A72` N-line cross-check still re-derives to `1.14e-13`.

| item | value |
|---|---:|
| new fits | **0** |
| packages | 3 (plus the analysis) |
| desktop writes | 3 (with the r10 README) |
| agent uploads | 0 |

## 5. Recommendation

The next five slots should mix the unbounded interior with the two most
informative bounded endpoints: `V32_TIME_A60V7_50`, `V32_TIME_A60V7_75`, the
r9 `2_TIME_V100` (pure `V7m`, bound `96.3832`) and `4_IRON_W100` (pure `V12m`,
bound `96.3846`), then `V32_TIME_A60V7_25`. Reading: any package above `96.3679`
opens its direction; all below `96.3679` means the released weights already sit
at the platform optimum and no further weight work is justified.

Targeted tests: 7 in `tests/test_round2_v32_family_ceiling.py`. **96.4 remains
unmet**; the registered best is `96.3679`.
