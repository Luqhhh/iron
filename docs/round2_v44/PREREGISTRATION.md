# V44: one separately frozen BART chain-length study

Goal remains platform **96.5**, current user-reported best **96.3727**.
V43's original 400-sweep run, failed eligibility and every model remain
unchanged. This is a new experiment under the user's continuing optimization
objective, not extra sweeps appended to that run or a reclassification of it.

## New evidence and narrow question

V43's saved **training-only** traces show BART RSS lag-one correlation
.5385–.9065 (median .7613); its latter retained half shifts by -.605 within-chain
standard deviations at the median. Its mean RSS change across the two retained
halves is -1.212%, with extremes -4.764%/+1.958%. These diagnostics use no
outer query labels, fits or new model choices. They show dependence and drift,
not a proof of non-convergence or that lengthening the chain will improve MAE.
The original [BART paper](https://www.rob-mcculloch.org/2025_gml/webpage/BART.pdf)
also makes run length application-dependent. Its benchmark outcomes do not
forecast this task.

One fixed change: multiply the sweep schedule by ten. **Burn 2000, retain
100 draws every 20 sweeps, total 4000** instead of burn 200 / 100 draws every
2 sweeps / total 400. Retained sample count and prediction rule stay fixed;
there is no search over chain lengths or choice of a favorable subwindow.
The same 200 trees, priors, move kernel, depth caps, target/input transforms,
cut grids, RNG seed, control/candidate roles and BLAS/runtime remain V43.
Single-chain convergence is still not guaranteed and will not be claimed.

STUMP control and BART candidate both receive the longer schedule on both
targets. Development remains 40 outer units / 80 fresh fits. No old process
is resumed; every long fit starts from the same frozen initialization. Capture
the old 202..400 retained sweeps as an audit-only prefix, separate from the
new 2020..4000 prediction draws. Capturing snapshots must not consume RNG or
change computations. Prefix draws never enter new predictions or selection.

## Required exact comparison and identity

Before official fitting, pin the V43 manifest/audit, original specification
and both full-shape synthetic model hashes. Verify all original sources,
units, fit/query IDs and artifacts. Match both synthetic prefixes and all
**80 development fitted prefixes** exactly: prior, transforms, target range,
cutpoints, first 400 scalar traces, 100 old sampled ensembles and old saved
predictions. Any mismatch fails engineering; preserve fitted evidence and
stop the frozen round without automatic repair or retraining. Prefix checks
consume no model fits and must be included in completed-unit artifact hashes.

Two full-shape synthetic long fits are budgeted, with the unchanged learning,
memory and cold-inference tests. Re-measure runtime rather than assume linear
scaling. The same twice-slower-fit projection over 80 fits / four workers must
be <=7.61 hours; each worker <=1536 MiB and enough current RAM for four workers.
No lower tree count, fewer draws or parameter fallback on resource failure.

## Evaluation and stopping

Keep the exact current-reference factory/cache, development seeds 42/3407,
five folds, inner calibration seed 27001/fold 0 and sparse weight grid.
Calibration chooses only blend weight; query targets are absent during fit.
No target reweighting, cross-seed OOF averaging, posterior-window choice or
post-hoc alpha tuning. Historical V43 gains remain descriptive comparisons.

Eligibility is unchanged: BART must improve both complete development seeds,
mean >=.01, mean above STUMP. At most one finalist per target may consume
confirmation seeds 271828/314159; retain four positive seeds, positive paired
LCB95 and local development package >=96.25. Folds and candidate-tier labels
cannot override these gates. A successful length comparison does not authorize
release. No further schedule extension in this round after any outcome.

Locked Python 3.12 tests, full saved-model and prefix audits, source/runtime/
data/fold identities and append-only evidence are required. Hourly-only health
checks, no intermediate polling or automatic restart. Public validated work is
committed and pushed; private models/predictions/reports remain out of Git.
Full-data fits, packages, desktop writes and uploads: **zero**.
