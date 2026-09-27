# V14 results (2026-09-27)

No candidate qualified for confirmation. Joint supervision with repaired PLE-B
improves iron standalone WMAPE compared with the matched V13 single-output model,
but all four target/recipe units select zero weight over the delivered local
references at both development splits. The registered interaction experiment
therefore ends with no confirmation fits and no package.

## G0: complete and independently audited

Completed 20/20 joint development outer fits plus one separate exact periodic
joint control, with no failed fits. The control reproduced both V12 output
columns exactly (maximum absolute difference 0). One inactive-control fit hit
the frozen epoch cap; repaired fits had no cap hits. The budget was not extended.
Inactive nonlinear weights stayed zero; all 5376 nonlinear weights were nonzero
in each repaired fit.

Locked Python 3.12 tests: **1062 passed, 23 warnings**. Independent audit verified
all 20 joint prediction matrices and hashes, source/data/fold/reference identities,
training row identities, both target scalers, 20 inner/full bin reconstructions,
training budgets, nested score arithmetic, candidate tiers and finalist selection.
The prior four-seed reference preflight checked 188 files and establishes that
historical references are available; it consumed no fits or candidate scores and
does not authorize confirmation after this negative result.

Evidence lives under `local/runs/round2-v14-joint-ple/development-r1`, including
`manifest.json`, `fit_ledger.jsonl`, `summary.json` and `audit-r1.json`. All private
models, predictions, ledgers and audit files remain outside Git.

## G1: no increment beyond the delivered directions

Gains are local score points from the frozen same-seed nested blend procedure.
CURRENT is V12 isolated iron A50 or V7 isolated time A50; the other target remains
A35. These are local comparison references, not a two-column package or newly
reported platform best.

| Target | Recipe | Mean vs A35 | Mean vs CURRENT |
|---|---|---:|---:|
| tap_iron | joint_ple_inactive_control | +0.003317 | +0.000000 |
| tap_iron | joint_ple_repaired | +0.005416 | +0.000000 |
| tap_time_len | joint_ple_inactive_control | +0.000000 | +0.000000 |
| tap_time_len | joint_ple_repaired | +0.000976 | +0.000000 |

Every CURRENT comparison selected alpha 0 on both split seeds. All four units
are `not_shortlisted` and none has two positive complete CURRENT-relative seeds.
Confirmation fits, full-data fits, packages, desktop writes and uploads: **0**.
No fallback or replacement candidate was selected.

The mechanism comparison is asymmetric. Repaired joint iron WMAPE improves from
V13 single-output 0.038826/0.039184 to 0.038433/0.038216 on seeds 42/3407. Repaired
joint time gets worse, from 0.039612/0.039461 to 0.039895/0.039821. The inactive
control has the same pattern: joint training improves iron but worsens time.
This supports a directional sharing benefit within these fits, not an additional
blend contribution. The comparison also includes joint epoch selection, so it is
not an isolated measurement of gradient sharing alone.

The zero increment applies to this preregistered candidate pool and sparse blend
grid; no global impossibility claim follows. V12 and V13 historical results and
all thresholds remain unchanged. Delivered V7 and V12 remain pending platform
feedback. Concurrently registered alpha-line feedback (commit `c7ec56f`) makes
A60=96.3465 the current user-reported platform best, replacing A35=96.3366.
V14 was frozen earlier and retains its A35-derived comparisons; no post-result
reference substitution was made. A score above 96.4 is still unverified and the
user goal remains active.
