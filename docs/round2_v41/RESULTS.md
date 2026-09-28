# V41: a conditional continuous-family upper bound below 96.5

**The existing fixed endpoints cannot reach 96.5 by nonnegative mixing under
the recorded-score and documented-metric assumptions.** The conservative
continuous upper bound is **96.46205**, including score rounding and potential
iron improvement. This is an upper bound, not a feasible prediction or a claim
that a package attaining it exists. Current user-reported best remains
96.3727; no new platform score was obtained.

Frozen protocol: `6a7b6bf`, `configs/round2_v41/SPEC.yaml`. Source observations:
V40 branch commit `c5afb9c4cc903ef31f1ad55c825e8f9b263666a0`,
`docs/round2_v40/RESULTS.md`, including I5=96.3727 and iron w=1=96.3514.
These are user reports, not independently verified receipts. The calculation
assumes those reports identify the stated packages, unchanged fixed test rows,
and the documented additive equal-weight WMAPE formula. Unknown reporting or
package-identity errors are not covered. No new label access occurs.

## Continuous-domain certificate

The time domain is all nonnegative weights summing to one on **V36, N0048,
V7 periodic**. Iron ranges over **w in [0,1]** between V36 and the V12 joint
model, independently. Predictions themselves are fixed. The bound does not
cover extrapolation, nonlinear postprocessing, new training seeds or new models.

At an observed anchor, concavity restricts every supporting gradient. The
maximal supporting-plane value over those allowed gradients is a convex upper
envelope. Its maximum over a simplex is no greater than its largest value at
the vertices. Linear programming finds nonnegative dual coefficients; exact
rational arithmetic verifies each certificate independently of the solver.

| Quantity | Conservative upper |
|---|---:|
| Entire time simplex, fixed released iron w=0.5 | 96.445350 |
| Entire iron interval, fixed B0 time | 96.384550 |
| Iron improvement over w=0.5 | +0.016700 |
| Combined continuous family | **96.462050** |
| Margin below target 96.5 | **0.037950** |

For example, time anchor I5 has upper score 96.37275. Its three domain-vertex
certificates use respectively `3*A60`, `5*A72+6*I1`, and `A45+2*B0` as dual
coefficients and give 96.40395, 96.44535, 96.39565. Their maximum bounds the
whole simplex, including points not lying on any measured chord. This bound
is deliberately conservative and need not be the tightest available.

Four-place reports receive +/-0.00005 intervals. Older time observations
shifted to the released iron column by the measured B0-minus-V7 difference
receive +/-0.00015. Discarding correlations between these intervals enlarges
the uncertainty conservatively. Calculations concern the unclamped score;
the final bound is clamped at zero. The positive result above is unaffected.

## Corrections to the previous interpretation

- Equal I1/I5 scores establish a chord lower bound, not a flat plateau. An
  interior peak remains possible.
- Iron w=0.5 beating w=0 and w=1 does not prove the continuous maximum is at
  0.5. Positive iron headroom remains mathematically possible and is included.
- A point lacking a pairwise collinear bound is not necessarily unbounded
  when all observations are used. Multi-point certificates bound this whole
  continuous domain.
- Increasing the maximum over a changing subset of bounded grid points is
  not loosening a global bound. An upper bound above a target is not a
  proven attainable path to that target.

Historical files and package decisions are preserved. This analysis supersedes
those mathematical interpretations, not their measured scores or identities.
The distinct V39 hard-tree and concurrent V39 interior branches keep their
original identities; this work is V41 on `round2-v41-concavity-bound`.

## Existing probes: limited upside, no new scheduling authorization

With iron fixed at w=0.5, multi-point bounds (including rounding) are:

| Existing probe | Score upper bound |
|---|---:|
| H1 | 96.379826 |
| H2 | 96.379434 |
| SEG625 | 96.377700 |
| N500 | 96.380590 |

They may investigate a small remaining gain, but none can individually close
the 96.5 gap under the assumptions. Their upper bounds do not rank expected
outcomes. Do not fill the five-slot daily allowance merely to consume it;
prioritize qualified new predictions toward 96.5. Existing pending files stay
untouched and uploads remain the user's action.

## G0 and G1

G0: five analytic tests pass, including exact dual equations, concavity
inconsistency rejection, affine reproduction, equal-value interior peaks and
uncertainty monotonicity. An independent Fraction-only audit verifies all
continuous-domain rational certificates without invoking the solver.
Locked Python 3.12 full suite: **1233 passed**, 23 existing warnings.
Independent audit: **11 rational certificates passed**.

G1: no new model-quality or platform measurement. This is a conditional
mathematical restriction on the existing family. No promotion gates changed.
New fits, label reads, full-data fits, packages, desktop writes and uploads: **0**.
The 96.5 objective remains active; useful next work needs a prediction outside
the existing fixed-endpoint convex family.

Private evidence: `local/runs/round2-v41/bound-r1.json`, SHA-256
`b019ec4b84ec87ad76c39e6ac27fde4b28443ff4d7c55ae6baa234f16f240f36`;
independent audit `local/runs/round2-v41/audit-r1.json`. All rational dual
coefficients are preserved there; reports remain outside Git.
