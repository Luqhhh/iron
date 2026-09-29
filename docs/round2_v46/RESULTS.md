# V46: the old time mixture cannot reach even the first milestone

Current platform best remains user-reported **96.3727**. The 96.4 / 96.45 /
96.5 milestones remain unmet; this round produces no platform measurement.
Protocol 651f1da, implementation 7750ce6. All results below are conditional on
correct reported scores and package identities, the declared rounding intervals,
unchanged test rows and the documented additive equal-weight WMAPE formula.

## What is now excluded

| Fixed released prediction family | Certified score upper, rounded upward | Milestone conclusion |
|---|---:|---|
| V36/N0048/V7m time convex mixture; incumbent iron unchanged | **96.396835** | Below 96.4 |
| Same time family plus any V36/V12-joint iron convex mixture | **96.413535** | Below 96.45 and 96.5; does not exclude 96.4 |

The first statement covers the **entire continuous time simplex**, not only
the already measured lines or the four delivered ridge probes. Therefore
another time-only weight proposal from those same three fixed endpoints
cannot reach 96.4 under the assumptions. Smaller improvements remain possible.

The second statement allows both columns to vary independently, including
unmeasured iron weights between the reported values. It must not be read as
proof that 96.4 is achievable. The iron headroom is a mathematical upper,
not a measured gain or a release recommendation. Moving the iron weight from
the incumbent .5 to the measured endpoint 1 already lost on the platform.

Any route to 96.45 or 96.5 must leave this fixed-endpoint convex family.
This does not rule out a new model, a new fit/seed, extrapolation or nonlinear
transformations; those are outside the proved domain and receive no implied
authorization or quality claim. No package is ranked by its upper bound.

## New feedback and stronger continuous coverage

H1 is the user-reported **96.364** package at time weights (0,.45,.55), ZIP
SHA-256 `772057b95032f035ecdd1051133a7885925c51ee2aa611f55bc36744f19e44b8`.
The source is original-worktree commit
`ef10deba58ad7259d984d869b5e95914d9204296`. It has only three reported decimals,
so this calculation allows **+/-0.0005**. Old V41 report/lifting intervals
remain unchanged; no fourth H1 decimal is assumed.

| Calculation | Time upper, upward rounded | Both-column upper, upward rounded |
|---|---:|---:|
| V41 coarse domain, old observations | 96.445350 | 96.462050 |
| Same coarse method with H1 | 96.416784 | 96.433484 |
| V46 fixed cell refinement with H1 | **96.396835** | **96.413535** |

This separates the added-feedback effect on the coarse calculation from the
further tightening obtained by subdivision. The old reports, models, scores
and decisions remain preserved. No change in model quality is inferred.

Exact refined time upper: **62465149/648000**. Exact iron-domain upper:
**96.38455**; subtract the W05 lower score **96.36785** to give at most
**.01670** additional iron contribution. Exact combined upper:
**312379853/3240000 = 96.413534876543...**.

At this bound, 96.4 remains unexcluded by .013534876543...; 96.45 and 96.5
exceed the family upper by .036465123456... and .086465123456... respectively.
These are certificate margins, not attainable gains or statistical confidence
intervals. Incorrect score reports, misidentified packages or different metric
semantics are not covered by the result.

## G0 verification; G1 remains unchanged

The locked Python 3.12 suite passed **1264 tests**, 23 existing warnings,
including eight analytic/adversarial bound tests before the real-observation
calculation. The fixed grid contains **1600 time triangles and 40 iron
intervals**. It covers the full domains with exact rational coordinates.

Within each cell, one common observed anchor supplies every vertex's
supporting-gradient certificate. Convexity of that anchor's upper envelope
bounds the whole cell by its largest vertex certificate. Selecting the best
common anchor per cell and then maximizing over all cells is valid for the
whole domain. Taking a maximum of unrelated pointwise vertex bounds alone
would not provide that guarantee and is not the implemented calculation.

A separate auditor, without LP/optimizer or generator imports, passed
**4880 cell-vertex certificates plus 8 coarse certificates**, checked every
nonnegative rational dual equation, reconstructed the full canonical tiling,
and verified exact cell/global maxima, iron subtraction, outward presentation
and all milestone decisions. No certificate or coverage failures occurred.

G0: passed. G1: no new fitted candidate or measured platform improvement.
Training fits, label reads, test-prediction reads, packages, desktop writes
and uploads: **0**. The active staged optimization goal continues. Subsequent
running jobs retain the user's **ten-minute** monitoring preference.

Private evidence: `local/runs/round2-v46/bound-r1.json`, report SHA-256
`08c7c3b5408e2bddb31959fedcb8813431cad6dbdf074ebd276bc36aa56c15ab`;
independent `audit-r1.json`. Public source and summaries do not publish the
private certificate artifacts or any platform receipt.
