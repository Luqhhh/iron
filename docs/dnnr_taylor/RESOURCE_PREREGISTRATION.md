# DNNR synthetic resource probe — pre-registration

This is the next G0 prerequisite. It does not read official targets, consume
an outer split seed, evaluate the incumbent, launch official fits or create a
package. Existing G1 gates remain unchanged. Frozen settings and rules are in
configs/dnnr_taylor/SPEC.yaml and dnnr_preflight.py.

The deterministic seed57331 generator produces2755 rows with21 numeric
features and four spout categories. The first2204 rows train and the remaining
551 are query-only. The target is
`30+2*sin(x0)+x1*x2+0.5*x3+0.1*epsilon`. This exercises26 encoded dimensions;
future official partitions must not exceed this dimension bound to use its
cost admission. No synthetic feature is a stand-in for official targets.

A complete matched unit fits and saves the three inner and three fresh outer
models with the formal neighbor counts and one-epoch metric ceiling. It then
independently audits all six states and endpoint selection. If calibration
selects epoch0, one extra whole-training learned model executes epoch1 and
undergoes the same saved-array cold audit. This covers the cost of a formal
unit selecting epoch1 without changing its calibration decision. That extra
model is solely a resource witness and has no candidate rank. Reservation
caps are one pair, seven estimators, three metric epochs and five derivative
banks; actual epoch reservations must be exactly two under either outcome.

Admission requires all three synthetic MAEs below the training-median
prediction's query MAE, all numerical/cold/order/chunk differences<=1e-8,
worker peak<=1536MiB, and available RAM>=4*measured_peak+1024MiB.
The projected20-unit development cost, including saved-model and cold-audit
cost, is
`20*(pair_execution_seconds+pair_audit_seconds+optional_extra_seconds)/4*1.5+300`,
and must not exceed7200 seconds. The optional model's entire measured cost is
included conservatively even though the ordinary pair already contains an
epoch0 learned refit. These rules are fixed before this full-size probe.

Freeze requires an externally anchored exact-source locked Python3.12
engineering receipt, unchanged source/config/test snapshot, identical checked
runtime and all four numeric thread variables at1. Output must be a new,
direct private directory inside this worktree's local tree, never shared
symlinked runs. The one-shot runner preserves failed/incomplete evidence and
reserves every fitting operation before computation. Completion reports bind
all artifacts, ledgers and measured costs. A separate verify_admission entry
reconstructs the numerical audit and resource arithmetic without new fits.

If any gate fails, do not shrink the fixed mechanism, loosen the ceiling or
retry the same frozen probe. Preserve the negative evidence. If all pass, the
formal controller still needs independent current-reference and frozen
data/fold/source preparation before official development. G0 admission alone
does not establish official model quality or authorize release.

Monitoring remains every600 seconds, with no between-check training polling.
An actual terminal event may trigger audit. Users upload themselves; this
probe produces no submission or desktop artifact.
