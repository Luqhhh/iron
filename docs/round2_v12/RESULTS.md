# V12 joint-output TabM: complete development

All20/20 joint candidate fits and the single-output control completed, with no
failed fits. The control reproduces V7 time seed42/fold0 exactly (difference0).
Each candidate jointly predicts both targets; this does not authorize a two-
column release. Private run: local/runs/round2-v12-joint-tabm/development-r1.

## G0

Locked Python3.12 tests before development:1034 passed,23 warnings. Independent
zero-fit audit `audit-r1.json` verifies20 prediction hashes, source/runtime/data/
fold/reference identity, both target scales in inner/full training partitions,
training row identities, preprocessing, complete coverage, sparse alpha choices,
per-fold and seed gains, confidence bounds, package scores and candidate tiers.
Six fits reached the240-epoch selection limit; no limit was changed. These fixed
budget results do not establish full convergence. All20 predictions contain two
outputs, so they represent40 target/fold evaluation cells per split pair overall.

## G1: periodic sharing supplies a larger iron increment

Full-package score-point gains; every development split contains all five folds.
CURRENT is V9's fixed iron10% candidate-pool blend or V7's fixed time50% delivered
blend. Other-column package scoring always retains A35. V9 is not a released
platform incumbent. Weight selection uses the other seed, without averaging OOF
vectors between split seeds.

| Target | Recipe | A35 mean | CURRENT seed42 | CURRENT seed3407 | CURRENT mean | Positive folds | Tier |
|---|---|---:|---:|---:|---:|---:|---|
| Iron | Joint raw | +0.002859 | -0.000602 | +0.000860 | +0.000129 | 5/10 | exploration |
| Iron | Joint periodic0.01 | +0.018159 | +0.011571 | +0.013317 | +0.012444 | 8/10 | formal |
| Time | Joint raw | 0 | 0 | 0 | 0 | 0/10 | not shortlisted |
| Time | Joint periodic0.01 | +0.009325 | +0.000859 | +0.000509 | +0.000684 | 5/10 | exploration |

The frozen ranking selects only **iron joint_plr001** for confirmation. Its
CURRENT-relative alpha is0.35 on both development splits; A35-relative alpha
is0.50. The increment is larger than the marginal V10/V11 gains, but it remains
selected development evidence. The two-seed isolated-column package score is
**96.226441 <96.25**, so even four-seed success would not itself authorize release.

Raw joint iron fails the two-positive-seed prerequisite. Periodic joint time has
only a small increment beyond V7. Neither is substituted or additionally confirmed.
The mechanism is direct supervised representation sharing; true target values
never become inference features. Earlier joint-tree and shared-low-dimensional
results remain historical, and this experiment does not reopen residual stacking.

Next action: freeze the selected iron recipe for ten joint fits on previously
used seeds7777/12011, evaluating only iron for promotion against A35 and V9.
All four gains must be positive with positive seed-level paired LCB95; fold counts
remain descriptive. No newly acquired labels are implied. The V7 delivery remains
unchanged and unreported on the platform. Current platform best A35=96.3366;
96.4 is unproven. Full-data fits0, packages0, desktop writes0 and uploads0.

## Confirmation allocation

`configs/round2_v12/CONFIRMATION.yaml` freezes iron joint_plr001 and ten candidate
fits on7777/12011, evaluating only iron for admission. A zero-fit preflight
verified97 reference files: the V9 candidate and its previously audited V8
same-fold V36 baseline. No new baseline fit is required. The joint prediction
files retain both outputs, but the nonselected time column cannot be promoted
from this confirmation.

The entrance rejects incomplete/duplicate pools and unelected or unearned
selection; the iron adapter rejects matrices without both expected output
columns. Eight targeted checks passed; the locked Python3.12 full suite passed
**1047 tests,23 warnings**. The independent confirmation audit is prepared in
`scripts/round2_v12_confirmation_audit.py`. All six development cap hits belonged
to joint_raw; the selected periodic recipe had zero cap hits.
