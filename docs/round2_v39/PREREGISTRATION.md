# V39: user-authorized full hard-tree quality experiment

Goal is platform **96.5**. Latest registered user-reported best is
**V32_TIME_A60V7_50 = 96.3727**, not historical B0 96.3679. The target gap
is 0.1273. The external V38 interior-feedback task has its own historical
96.4 target; this thread's explicit 96.5 target controls this experiment.

On 2026-09-29 the user explicitly said **“7.61小时可接受”**. V37's
7.607345-hour conservative projection is therefore accepted for this full
development experiment; the original 6-hour refusal remains historical.
This is an explicit resource-budget authorization, not a quality-gate change,
a measured full-run duration or a guarantee of finishing in exactly 7.61 hours.
Memory and original equivalence checks still must pass. No resource probe is
repeated. The untested V38 custom-routing sketch was parked privately when the
user accepted V37; an unrelated concurrent task occupied V38. V39 uses the
already tested, unchanged V37 core and is isolated in a separate Git worktree.

Freeze both original arms and all model/training settings: 1024 trees,
depth five, selected-variable fraction .8, dropout .2, original initialization,
Adam parameter-group rates, MSE, max 250 epochs, patience 50. GLOBAL is a
mechanism control only; INSTANCE is the only eligible candidate. No control
is retroactively promoted. Both targets, complete seeds 42/3407, five folds:
40 outer units, each one calibration fit and one fresh refit, 80 optimizers.
Two synthetic 80-epoch learnability fits must precede official fits.

Train-only QuantileTransformer(normal) on the 21 numeric features, training
spout vocabulary one-hot with unknown categories all zero, training target
mean/population standard deviation. Save every transform and fitted model.
Inner split seed 27001, fold zero, is fixed to reuse the audited matching
component caches; early stopping and mixture weights use only calibration
rows. No outer-query labels enter fitting, selection or preprocessing. Blend
grid remains [0,.05,.1,.2,.35,.5,.75,1], smallest weight on ties. Refit from
fresh initialization at the selected epoch. No cross-seed OOF averaging.

The current reference is iron .5*V36+.5*V12_joint and time
.20*V36+.30*N0048+.50*V7_periodic. Reuse the 20 audited B0 component-cache
units after checking original manifests, source/data/row/fold identities and
old B0 arithmetic; then recompute the time mixture from those same component
columns. This is valid zero-fit reuse of components, not reuse of old B0
mixture predictions as the new reference. Confirmation refits the same
components with current weights. Preserve historical B0 time in reference
artifacts for diagnostic comparisons.

Eligibility requires INSTANCE's pooled gain positive on both complete
seeds, mean gain >=.01 score points, and mean greater than GLOBAL's. At most
one INSTANCE finalist per target; only audited eligibility can consume new
confirmation seeds 271828/314159. Existing four-seed gate: all seeds positive,
seed-level paired one-sided LCB95 >0, development package mean >=96.25;
fold profile is descriptive. Apply candidate tiers without overriding these
gates. Confirmation budget is conditional and separate from the quoted
40-unit development projection: at most 20 candidate units and 20 matching
reference factory calls. No automatic release follows promotion.

All fit starts and failures consume their frozen budget. Never overwrite run
directories or restart failed units. Independent cold audit checks persistence,
query order/chunks, train-only preprocessing, epochs, calibrated blend weights,
current-reference arithmetic, scores and eligibility. Reports distinguish G0
engineering from G1 quality. No full-data fits, packages, desktop writes or
agent uploads. Public implementation/evidence is committed and pushed after
checks; private artifacts stay under this worktree's local/runs/round2-v39.
