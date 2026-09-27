# What the V12 result can support (2026-09-27)

V12_IRON_JOINT_PLR001_A50 = **96.3526** is the latest user-reported best.
It gains **0.0160** over its unchanged-time A35 parent, and **0.0061** over
A60. The target above96.4 remains unmet. V7 is the next already-delivered
isolated experiment; no V7 score has been reported.

## Documented formula and conditional scope

The official September competition-rules attachment was retrieved successfully
on2026-09-27. PDF page40 (printed page38), section6, defines
`max(0, 100 - 50*WMAPE_iron - 50*WMAPE_time)`; page41 reiterates equal weights.
[Official rules](https://www.aicomp.cn/wp-content/uploads/2026/09/%E9%99%84%E4%BB%B61%EF%BC%9A-AIC-%E4%BA%A7%E4%B8%9A%E5%91%BD%E9%A2%98%E8%B5%9B%EF%BC%88AI%EF%BC%8B%E9%92%A2%E9%93%81%EF%BC%89%E7%AB%9E%E8%B5%9B%E8%A7%84%E5%88%99.pdf)
This agrees with the archived task PDF, SHA-256
`18ee1e4c0217e8d6dab1225045418f8b79a93bd90cd4e958b6da23733dc57146`, pages4–5.
The new retrieval supplements the historical timeout record; frozen metric
contracts are not edited. The platform implementation and per-package receipts
are still not independently verified.

All arithmetic below assumes the documented formula, the same hidden rows and
labels, the exact original prediction columns, no nonlinear postprocessing, and
each reported four-decimal score within **±0.0001** of its actual value. This
last-digit allowance covers ordinary rounding or truncation; it does not cover
misattributed scores or a changed evaluation cohort. These are conditional
mathematical bounds, not platform forecasts or measured combined scores.

## Two columns: additive score arithmetic

Since each target contributes separately, copying V12 iron and A60 time would
give, under those assumptions:

`96.3526 + 96.3465 - 96.3366 = 96.3625`, with allowance ±0.0003.

That is below96.4. No combined prediction file or package has been generated.
For the original V7 time column, the analogous expression is
`V7_score + 0.0160`. Thus V7 would need to exceed96.3840 centrally, or strictly
exceed96.3843 allowing all three last-digit errors, for the **conditional**
V12/V7 combination to exceed96.4. This does not authorize packaging and does
not establish an actual combined platform result. Smaller V7 gains can still
be useful evidence.

## Within one column: concavity, not linear score extrapolation

For fixed endpoints, `p(a)=(1-a)*p0+a*p1`. Each absolute residual is convex
in `a`, so the un-floored score is concave. For `a < b < c`, its chord slopes
cannot increase. A chord is an upper bound only when extended **outside** its
interval; inside it is a lower bound. In particular, two endpoint scores alone
do not bound an unmeasured interior peak. The zero-score floor cannot turn any
of the positive upper bounds below into a higher score.

The measured time points are A35=96.3366, A45=96.3438, A60=96.3465,
A72=96.3425. Intersecting the valid secant extensions, with the ±0.0001
allowance applied before arithmetic, gives:

| Exact fixed family and weight range | Conditional score upper bound |
|---|---:|
| V36 iron; N-0048 time weight anywhere in[0,1] | 96.349298 |
| Released V12 iron A50; N-0048 time weight anywhere in[0,1] | 96.365498 |
| V12 iron weight in[0.5,1]; original A35 time | 96.368900 |
| V12 iron weight in[0.5,1]; A60 time | 96.379000 |
| V12 iron weight in[0.5,1]; N-0048 time weight anywhere in[0,1] | 96.381698 |

Bounds are rounded upward. The iron bound follows from the two observed
weights0 and0.5: beyond0.5, the slope cannot exceed their secant. It does
**not** bound iron weights below0.5, unrelated models, extrapolation outside
[0,1], or changed preprocessing. The location where an upper envelope is
largest is not an estimated optimum and is not a proposed submission weight.

Therefore neither finer tuning of the old time line nor increasing the V12
iron weight from50% to100%, even together, supplies a route above96.4 under
these assumptions. This narrows the next work to evidence from V7 and genuinely
different prediction directions; it does not declare the overall goal impossible.
The historical quadratic time-peak forecast is superseded for current planning.

## G0 verification and G1 limitation

The read-only audit checked the original A35/A45/A60/A72/A100 package hashes,
322 unique aligned IDs, unchanged iron field strings, and their common affine
time endpoint. Maximum affine discrepancy is5.7e-14. It also reconstructed V12's
iron A50 from its stored full-fit iron endpoint and checked unchanged A35 time
strings. The separate desktop audit covered V12, V7, A35 and A60.

Implementation: scripts/round2_platform_line_bounds.py, using exact rational
secant/intersection arithmetic. Ten targeted tests include sharp unobserved
peaks, the unbounded two-endpoint interior case, rounded observations, and forty
synthetic affine-prediction absolute-error curves. No official labels are read
by this calculation or these new tests. Private evidence:
local/reports/platform-line-bounds-20260927-r2.json. The first audit's incorrect
expectation of a two-output cold array is preserved in failed-r1.json; the
released IronView correctly stores one iron output per row. No model or package
was changed by that failed check.

G1: the V12 platform gain is user-reported. The new results are conditional
feasibility bounds, not model promotion. Existing four-seed gates, release
exceptions and original packages remain unchanged. Fits0, new packages0,
desktop writes0, agent uploads0.

Locked Python3.12 full suite: **1092 passed,23 warnings** (213.08s).
Log: local/reports/platform-line-bounds-python312-r1.log.
