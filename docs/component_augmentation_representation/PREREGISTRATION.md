# Parallel local augmentation and coupled periodic representation

User explicitly requested parallel execution on 2026-09-29. Independent
worktree/branch codex/component-augmentation-representation; leave the running
BASE/EMA/SAM worktree and frozen source untouched. Share a four-worker queue
for the three new mechanisms; at most eight candidate workers across both
workflows. Pin BLAS/OMP/MKL/NUMEXPR and Torch to one thread per worker.

## Frozen pool and comparison

Current reference remains user-reported V32_TIME_A60V7_50=96.3727 (not verified
platform receipt). Same original-component .5 replacement: V12 joint_plr001
for iron, V7 tabm_plr001 for time. Other components and other target unchanged.
Original float32 preprocessing, loss, AdamW, architecture width/depth/dropout,
inner seed42/fold0 stopping and fresh refit remain fixed. No blend tuning.

Reuse the first batch's 20 BASE replay units only after verifying their complete
artifact hashes, manifest identity, original committed source bytes, target/
training/query rows, seeds, selected epochs, reference arithmetic and exact
native component predictions. A whole first-batch quality summary is not
required to reuse completed BASE units. Incomplete units are never accepted.
If controls are still running, wait on600-second observations; preserve failures.

Complete development seeds42/3407, five folds, both targets, three methods:
60 new outer units/120 optimizer runs;20 BASE units reused, no hidden new BASE
fits. No extra training-seed diagnostics in this batch. All three mechanisms
can execute in the shared queue; no cross-combinations or parameter grids.

## D-LMIX

Build neighbors from the current fitting partition only, using its standardized
21 numeric features and matching spout. Select nearest8 other rows; use fewer
if the group is smaller. Ties resolve by sample_id. A singleton has no synthetic
supervision. Never use calibration/query/test rows or target similarity.
Choose a partner uniformly among eligible neighbors and lambda~Beta(.2,.2)
with an independent NumPy RNG (training seed,54002). Interpolate standardized
numeric features and standardized targets; the same lambda applies to both
joint outputs. Categorical input stays unchanged. Original and synthetic
individual-member MSE receive .5/.5 weight. Normalized augmentation mean uses
only eligible synthetic rows. If none are eligible, use original loss alone.

Retain the original minibatch order/dropout stream for original samples;
synthetic dropout uses a separate Torch stream (training seed+54003), restored
after the synthetic forward. Both losses backpropagate in one optimizer update.
No augmentation at inference. No beta or neighbor-count scan.
Mechanism source: https://arxiv.org/abs/1710.09412; constrained pairing is a
task-specific hypothesis, not an established local result.

## R-FIXED and R-LEARNED

Keep the original per-feature periodic channel and add32 low-rank coordinates:
u=(z V^T) U^T, V shape4x21, U shape32x4; sin/cos gives64 new inputs to the
first TabM affine layer, before its activation/dropout. V~N(0,1/sqrt21),
U~N(0,1/sqrt4), initial scale1 radian on standardized inputs. Projection
initialization stream54001 is independent of the native model/dropout RNG.

Implement concatenation equivalently as an extra affine input term sharing
the original first layer's output scaling. New input fast scalings are standard
normal (sin/cos pair shares initial scale); new64xwidth weights start at zero.
All original parameter initializations remain exact, so the initial function
equals BASE. Extra affine weights and fast scalings train in both arms.
R-FIXED freezes U,V; R-LEARNED trains both. Both start with identical projections
and predictions. This is an input representation inside the jointly trained
backbone, not an output residual corrector. No width/depth/frequency sweep.
Source: https://arxiv.org/abs/2006.10739; this hypothesis changes the learning
inductive bias, not a claim of functions impossible for the original network.

## G0, G1, resource and release rules

Reuse the first batch's evaluator, saved-state audit and monitoring with isolated
configuration/model adapters; do not rebuild validation methodology. Tests must
cover neighbor exclusion/category/ties/singletons, shared target interpolation,
zero-augmentation native identity, coupled rank and projection freeze/gradients,
initial-function identity, unchanged native RNG, save/load/order/chunk, original
BASE-cache tamper rejection and full development/confirmation seed identities.
Full locked Python3.12 --no-sync suite before actual data; no new dependencies.

Six full-shape synthetic240-epoch fits use the first batch's frozen synthetic
data (seed52001, first2204 train/550 query). All must beat constant; fixed and
learned need not beat each other or BASE. Cold whole-batch difference0;
order/chunk tolerance5e-4. Peak worker<=1536MiB; available memory must exceed
four new peaks+1024MiB while existing work is running. Conservative projection
2*sum(six timings)*20/4/3600 <=7.61h, including twofold safety margin. Preserve
failed admission; do not tune using synthetic scores.

Same fixed promotion: positive complete development gains on both seeds,
mean>=.01 and improvement over BASE; rank by mean, ties D-LMIX,R-FIXED,
R-LEARNED. At most one finalist per target. Confirm on271828/314159 only after
audited development: four positive split gains, seed-LCB95>0, development
package score>=96.25; fold results descriptive. Additional splits are not new
independent data. Preserve tiers, training-defined tail diagnostics, row-bound
bootstrap and no cross-seed prediction averaging. Confirmation maximum20
candidate units/40 optimizer runs,20 BASE replay units/40 runs,10 matching
reference factory calls/320 pipelines. Those costs are separate from development.

600-second scheduled monitoring and actual terminal-event audits only. Private
root local/runs/component-augmentation-representation. No full-data fit,
package, desktop write or upload. No score forecast, no gate relaxation.
