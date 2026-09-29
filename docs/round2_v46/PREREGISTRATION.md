# V46: refine the fixed-family bound for three platform milestones

The active platform milestones are 96.4, 96.45, 96.5. Best remains
user-reported 96.3727. V45 completed without a finalist. This zero-fit
analysis determines which milestones the existing released-endpoint family
can still possibly support, not which package should be uploaded.

New evidence: V40_TIME_H1, time coordinates (V36,N0048,V7m)=(0,.45,.55),
unchanged released iron, reported score 96.364. Pin its identity and record
to original-worktree commit ef10deba58ad7259d984d869b5e95914d9204296. Use a
conservative +/-0.0005 because the report has only three decimal places;
do not assume the unreported fourth digit is zero. Retain V41's original
observations, uncertainties and metric assumptions unchanged.

## Fixed calculation

Use a fixed denominator 40. Partition the time simplex x>=0, y>=0, x+y<=1
into exactly 1600 elementary triangles with rational vertices (i/40,j/40).
Partition the iron interval [0,1] into exactly 40 adjacent rational intervals.
No adaptive choice of resolution after inspecting these results.

For each observed anchor a, maximize its supporting-plane value over all
gradients consistent with the other observed lower score bounds and this
anchor's upper score bound. This function U_a(x) is a supremum of affine
functions, hence convex, and bounds the actual unclamped score above.
V41's `upper_at` produces a rational dual certificate for its value at a
specified vertex. A certificate's validity, not floating LP agreement, is
binding. Retain an anchor for a cell only if every cell vertex is bounded.

For any fixed anchor, every point of a cell is a convex combination of its
vertices, so U_a inside the cell is no greater than the largest vertex value.
Choose the smallest of these certified maxima across anchors **within each
cell**. Then take the maximum across all cells to bound the full domain.
Do not take a maximum of pointwise minima at grid vertices: different vertex
anchors alone do not certify an entire triangle.

Keep domain coverage explicit: store integer grid-vertex indices, every cell,
its common selected anchor and its vertex certificates. Use exact rational
arithmetic for certificate equalities, nonnegative dual weights, cell/global
maxima, iron-reference subtraction and target comparisons. Report an outward
rounded floating approximation only for presentation. Iron improvement is its
domain upper minus the lower bound of W05; add that to the time upper, then
apply the documented score floor at zero.

## Verification and interpretation

Before executing the official observation calculation, test affine examples,
known concave interior peaks, cell coverage/area, uncertainty monotonicity,
refinement versus the coarse bound and inconsistent-input rejection. Run the
locked Python 3.12 suite. A separate verifier must use only Fraction arithmetic
to check dual equations and recompute the entire canonical grid, cell bounds
and global result, without calling LP or the generating bound routines.

Record the coarse whole-domain bound with and without H1 as controls. Any new
upper must not be weaker than the old bound; do not replace an old exclusion
with a weaker approximation. Report each milestone separately. A bound above
a milestone establishes only failure to exclude it, never a viable candidate
or attainable score. A bound below excludes only the declared fixed-endpoint
nonnegative convex family, conditional on correct reports/package identities,
the stated rounding intervals and fixed-row additive WMAPE. Extrapolation,
new models/seeds and nonlinear transformations are outside the domain.

No label reads, training, test-prediction reads, new packages, desktop writes
or uploads. Keep private certificates/reports append-only under
local/runs/round2-v46. Preserve all earlier reports and the H1 failed upload.
Public validated implementation/evidence must be committed and pushed. The
ten-minute monitoring preference applies to subsequent running experiments;
do not create an idle or rapid-polling training monitor for this calculation.
