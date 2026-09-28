# iron 96.4 — V18 platform-side combined-direction line search

Design date: 2026-09-27. Base commit: `c16f831` (`round2-v6-iron-capacity-networks`).
Status at design time: no V18 fit has been run, no source ZIP has been modified,
no package has been written and no upload has been made.

## 1. Starting point

The registered platform best is the user-reported
`V12_IRON_JOINT_PLR001_A50 = 96.3526` (not independently verified). The target
is `> 96.4`, so the measured gap is `0.0474`. `V7_TIME_PLR001_A50 = 96.3519` is
the other scored isolated direction. Both are isolated changes against the same
A35 parent, so under the documented metric
`max(0, 100 - 50*WMAPE_iron - 50*WMAPE_time)` the two column effects add, and
the never-tested combined package has conditional arithmetic
`96.3526 + 96.3519 - 96.3366 = 96.3679` (±0.0003 for four-decimal receipts).

Measured platform points now available (all user-reported):

| package | iron | time | score | vs A35 |
|---|---|---:|---:|---:|
| V36 | V36 | V36 | 96.2734 | −0.0632 |
| A35 | V36 | 0.65 V36 + 0.35 N | 96.3366 | 0 |
| A45 | V36 | 0.55 V36 + 0.45 N | 96.3438 | +0.0072 |
| A60 | V36 | 0.40 V36 + 0.60 N | 96.3465 | +0.0099 |
| A72 | V36 | 0.28 V36 + 0.72 N | 96.3425 | +0.0059 |
| V7 | V36 | 0.50 A35 + 0.50 V7m | 96.3519 | +0.0153 |
| V12 | 0.50 A35 + 0.50 V12m | A35 | 96.3526 | +0.0160 |

`N` is the `v36-s1-N-0048` large raw-TabM time expert, `V7m` the V7-periodic
time expert and `V12m` the V12 joint iron expert. The N-line peak is at
`alpha ~ 0.60`; the V7 direction is stronger at its tested weight.

The **only untested high-prior package** is the combined `V12` iron + `V7` time
composition (`B0`). Two further questions are open and cheap to answer with the
same construction:

1. does the platform prefer **more V7m** than the locally selected 0.50 (as it
   preferred more N than the local optimum), and
2. does the platform prefer **more V12m** than the locally selected 0.50 iron
   weight?

## 2. Exact reachability bound (zero fit)

Every pure endpoint column is recoverable from the delivered ZIPs
(`V36_t = (A35_t - 0.35 N)/0.65`, `N` from `A35`/`A60`,
`V7m = 2 V7_t - A35_t`, `V12m = 2 V12_i - A35_i`). The metric is additive and,
for a fixed pair of endpoints, **concave** in the blend weight because each
absolute residual is convex. A secant extended outside its interval is
therefore an upper bound. With the measured iron weights `0` and `0.5`:

* iron: `(+0.0160)/0.5 = 0.0320` per unit weight, so at `w = 1` the iron column
  is worth at most `+0.0320` over A35 iron;
* time, on the A35 -> V7m line, weights `0` and `0.5`: at `v = 1` the time
  column is worth at most `+0.0153 + 2*0.0153 = +0.0938` over V36 time.

The joint ceiling of the two-line family is therefore
`96.2734 + 0.0320 + 0.0938 = 96.3992`, i.e. **0.0008 below 96.4**. This is an
upper bound, not a forecast: the local surface in section 3 says the realised
optima are interior, so the realised family maximum is lower. The bound is also
restricted to those two lines; a genuinely new expert or a different endpoint
can still exceed it.

## 3. Local complete-coverage surface (zero fit, descriptive only)

Recomputed on the existing complete five-fold caches for split seeds 42/3407,
single seed per candidate (no cross-seed OOF averaging), relative to A35:

| family | local optimum | mean gain |
|---|---|---:|
| iron line, time = A35 | `w = 0.50` | +0.018154 |
| V7 time line from A35 | `v = 0.50` | +0.020822 |
| N time line from V36 | `alpha = 0.20` | +0.005230 |
| 2-D time simplex (V36, N, V7m) with V12 iron | `b(N) = 0`, `c(V7m) = 0.50` | +0.046906 |

The local surface says the current released weights (0.50/0.50) are already the
interior optimum and that adding N on top of the V7 blend is locally harmful.
The platform N-line contradicts the local N optimum in both location
(`0.20 -> 0.60`) and sign of the slope, so local shape is **not** used here to
choose the submission weights; it is recorded as a descriptive diagnostic only.
The binding platform-side evidence is the measured line itself, which is why
this round tests weights rather than trusting the local optimum.

## 4. Frozen candidate set (5 packages, 0 new fits)

All packages are composed field-by-field from the seven hash-pinned original
ZIPs. `B0` needs no arithmetic at all: its iron column is V12's original CSV
field text and its time column is V7's original CSV field text.

| id | iron column | time column | question answered |
|---|---|---|---|
| `V18_B0_V12IRON_V7TIME` | V12 copy | V7 copy | the conditional `96.3679`; additivity test |
| `V18_TIME_V75` | V12 copy | `0.25*A35_t + 0.75*V7m` | does V7m want more weight? |
| `V18_TIME_V100` | V12 copy | `V7m` | the V7m endpoint |
| `V18_IRON_W75` | `0.25*V36_i + 0.75*V12m` | V7 copy | does V12m want more weight? |
| `V18_TIME_A60V7_50` | V12 copy | `0.5*A60_t + 0.5*V7m` | N and V7m together |

Weight choice is a bounded, pre-declared hedge around the released 0.50 slots,
not a dense scan. No weight outside `[0, 1]` and no nonlinear post-processing is
used. Every unchanged column keeps an original ZIP's exact field strings.

## 5. Required verification (G0)

1. all seven source ZIP SHA-256 values match the frozen constants;
2. exactly `result.csv` in each ZIP; 322 unique IDs, identical order, identical
   metadata fields across all sources;
3. `V12` time strings byte-equal `A35` time strings and `V7` iron strings
   byte-equal `A35` iron strings (isolation checks);
4. recovered `V36`/`N` time re-derive `A45`, `A72`, `A85`, `A100` row-wise to
   `<= 1e-9` absolute, cross-checking the affine N-line;
5. each written package re-reads to 322 rows, template order, byte-identical
   unchanged column, and blend arithmetic within `1e-12` relative;
6. finite, non-negative predictions; `x`-mode ZIP creation; no overwrite;
7. no labels, no model fit, no cold-inference step is required because no member
   is refitted and no new model parameter is introduced.

## 6. Interpretation and budget

`B0`'s expected platform score is the conditional `96.3679`, i.e. `+0.0153` over
the current best — still below 96.4. The hedges are lower-mean variants
(exploration, not promotion) whose only positive expectation comes from the
platform keeping the best score. Upload priority is `B0` first; the hedges are
optional and are ranked by the questions they answer.

Budget: **new fits 0, new packages 5, desktop writes 0, agent uploads 0.**
Platform scores are user-reported; a combined arithmetic is not a platform
observation and does not establish `> 96.4`. No gate from earlier rounds is
relaxed by this design. The user uploads and returns scores.

## 7. What this round does not do

It does not reopen any closed route (NODE, ODST, TabR fusion, residual
correctors, sample reweighting, dense alpha scans, learned combiners over the
library, sequential masks, joint PLE/gradient variants). It does not claim that
`96.4` is reachable from the current library: the screen of 44 reproducible
members against `B0` leaves at most `+0.00425` (`v17/P_LL_T`, time) and every
other member at `<= +0.0029` or exactly zero, so **a larger gain requires a new
model, not another blend of the existing pool.**
