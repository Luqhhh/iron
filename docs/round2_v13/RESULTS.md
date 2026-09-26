# V13 results (2026-09-27)

No candidate qualified for confirmation. Restoring the PLE-B nonlinear branch
improved standalone prediction, but neither iron recipe added an increment over
V9 iron, and the repaired time recipe was slightly negative over V7 time.
The separately registered comparison against delivered V12 iron also returned
zero for both iron recipes. This closes the registered single-output PLE repair
experiment without a promoted candidate, confirmation fit or package.

## G0: mechanism and evidence verified

All 40 development fits and the separately budgeted exact V7 control completed
without failure. The control prediction difference was zero. The independent
audit verified all 40 prediction hashes, source/data/fold/reference identities,
training metadata, 20 distinct inner/full preprocessing reconstructions,
independent nested scoring, candidate tiers and the absence of a finalist.
There was one epoch-cap hit in the inactive time control; repaired recipes had
none. Failed or capped evidence was retained and no budget was extended.

The inactive branch retained zero nonlinear weights. Every repaired fit learned
all 5376 nonlinear weights. Both recipes used the same standardized inputs and
training-only bin construction. The locked Python 3.12 suite passed 1039 tests
at V13 launch; the subsequent V12 release suite, which also includes V13, passed
1052 tests (23 warnings). Neither audit consumed new model fits.

Private evidence: `local/runs/round2-v13-ple-repair/development-r1`, including
`manifest.json`, `summary.json`, `audit-r1.json`, and `v12-increment-r1.json`.
Models, predictions, ledgers and audit files remain outside Git.

## G1: incremental comparison

All gains below are local score points from same-seed nested blends. CURRENT
remains the preregistered V9 iron or V7 time isolated column, with the other
column held at A35. No reference was changed after observing the results.

| Target | Recipe | Mean vs A35 | Mean vs CURRENT | CURRENT seed 42 / 3407 |
|---|---|---:|---:|---:|
| tap_iron | ple_b_inactive_control | +0.001535 | +0.000000 | +0.000000 / +0.000000 |
| tap_iron | ple_b_repaired | +0.000337 | +0.000000 | +0.000000 / +0.000000 |
| tap_time_len | ple_b_inactive_control | +0.000000 | +0.000000 | +0.000000 / +0.000000 |
| tap_time_len | ple_b_repaired | +0.002137 | -0.000381 | +0.000000 / -0.000762 |

The repaired model's standalone WMAPE improved over the inactive control on
both development splits: iron 0.040256/0.040383 to 0.038826/0.039184;
time 0.040238/0.040643 to 0.039612/0.039461. This establishes a useful engineering
repair, not useful diversification of the existing blends. All four units are
`not_shortlisted`; no unit has two positive complete CURRENT-relative seeds.
Consequently confirmation fits, full-data fits, packages and uploads are all zero.

The supplemental comparison was declared before complete-pool readout and used
verified same-fold V12 predictions. Against fixed `0.5*A35 + 0.5*V12` iron,
both recipes select alpha zero on both splits and have exactly zero increment.
It does not alter the original V13 rankings or historical decisions.

V7 time and V12 iron remain separately delivered with platform feedback pending.
The current reported platform best is still A35=96.3366; the goal above 96.4 has
not been verified. V12's candidate-specific release exception changes no global
gate and does not authorize another package.
