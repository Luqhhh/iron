# Fixed-weight replacement of effective components

Authorized 2026-09-29 after the user's review of SAM/EMA, local Mixup and
coupled periodic representations. This first batch tests BASE, EMA and SAM
only, using semantic identities rather than reserving a numbered round.
Other worktrees and historical experiments remain unchanged.

## Reference and estimand

Current platform reference: user-reported V32_TIME_A60V7_50, 96.3727,
ZIP SHA-256 54864561c057ae71f7779b15099750f7caf0141c149d609ae281dbaf9ef3aaa7.
Iron = .5 V36 + .5 V12 joint_plr001; time = .2 V36 + .3 N0048 + .5 V7
tabm_plr001. Recover the verified component cache through the existing V49
reference verifier (original V30, privately named round2-v27). Verify source,
data, row ordering, training partitions, folds and endpoint arithmetic.

Replace only the .5 V12 joint component for iron, or .5 V7 periodic component
for time. Keep the other target and all other components fixed. No weight
selection. Report both standalone improvement and package-score improvement,
against the original incumbent and against matched BASE. BASE must reproduce
each cached component exactly before candidate fits are admitted. A mismatch
is an engineering stop, not permission to relabel the baseline.

Native float32, train-only NumericPreprocessor, AdamW, individual-member MSE,
seed 42, group-safe inner seed 42/fold 0, native stopping and fresh refit are
preserved. Iron learns both standardized targets and selects by their mean
standardized MAE; only its iron output is replaced. Time is single-output.
Do not substitute V49's float64 single-output BASE or its calibration split.

## Frozen mechanisms

BASE follows the original V12/V7 training trajectory. EMA starts from initial
parameters and averages after every actual optimizer step with beta=.99;
its own validation trajectory selects its epoch. No extra clipping, loss,
capacity or learning-rate change. EMA floating buffers, if any, are copied
from the current network; integer buffers are copied. Save EMA inference
state. No BASE/EMA choice based on outer labels.

SAM uses rho=.05 and a global L2 gradient norm over all trainable parameters,
including periodic embeddings and TabM factors. Compute the first loss on a
minibatch, perturb by rho*g/(||g||+1e-12), compute the second gradient on the
same batch with the same dropout RNG state, restore parameters exactly, then
perform one AdamW update. Weight decay is applied once by AdamW. No gradient
clipping (native baseline has none). RNG advances as one ordinary forward;
zero norm gives zero perturbation. No SAM+EMA, parameter grid or rescue arm.
Same maximum epochs and optimizer updates, separately report SAM's doubled
gradient evaluations and wall time. Sources: [SAM](https://arxiv.org/abs/2010.01412).

## Coverage, gates and diagnostics

Complete development: seeds 42/3407, all five folds, two target components,
three arms = 60 outer units /120 optimizer runs (20 BASE units included).
Run all 20 exact BASE replays before the 40 candidate units. No two-fold
quality ranking. One predeclared initialization diagnostic per arm/target on
seed42/fold0 at training seed1042 adds 6 units/12 optimizer runs; these results
cannot select arms, epochs or weights. Report prediction sensitivity and gain
changes, not an independence claim.

At most one arm per target advances: EMA or SAM, both complete development
seeds positive, mean isolated package increment >=.01 and positive advantage
over matched BASE. Rank by mean gain, ties prefer EMA. Candidate-tier
classification remains descriptive alongside the controlling frozen gates.
Confirm eligible arms on existing declared derived splits 271828/314159,
five folds, using same-recipe reference refits. These are additional splits
of the same data, not fresh independent datasets. At most 20 candidate units
/40 optimizer runs; 10 reference factory calls /320 component pipelines shared
across targets (their internal neural selector/refits are separate work).
Require four positive split gains, paired seed LCB95>0, development package
score >=96.25. Fold counts descriptive. No gate relaxation or platform forecast.

Record train/inner validation errors in eval mode, selected/stopped epochs,
update/gradient counts, standalone WMAPE, fixed-replacement WMAPE, per-fold
and per-spout results, training-defined top-decile target errors, and fraction
of total improvement attributable to the largest row contributions. Paired
row bootstrap uses the same sampled row indices for every seed and recomputes
the WMAPE denominator each replicate; its interval is conditional/descriptive,
not a correction for adaptive selection or shared training sets. No cross-seed
prediction averaging before scoring. No outer labels enter model fitting.

## Engineering and cost admission

Locked Python3.12 --no-sync, existing CPU torch2.14.0+cpu; pinned BLAS/OMP/MKL/
NUMEXPR and torch threads=1. No dependency changes. Tests cover native replay,
SAM analytic update/restoration/zero gradients/RNG, EMA recursion and state,
train-only preprocessing, outer-label rejection, fixed replacement, selection,
tamper rejection, saved inference and paired bootstrap alignment.

Synthetic admission uses 2754 independent rows, 21 normal numeric features,
two spouts, target1=500+30*x0+20*sin(x1)+10*x2*x3+2*noise;
target2=100+6*x0+4*sin(x1)+2*x2*x3+.4*noise, RNG seed52001.
First2204 train, last550 query. Each target/arm trains a full 240-epoch
refit: 6 synthetic fits. All must beat the training-median predictor;
there is no required synthetic advantage over BASE and no toy tuning.
Fresh-process inference must match saved full-batch prediction exactly;
reversed/chunked raw-output maximum difference <=5e-4 (native float32).
Full-shape old/new BASE equivalence, selected epoch and saved trace checks
are mandatory. Project 2*sum(per-arm-target synthetic seconds)*22/4/3600
hours (120+12 optimizer runs with factor2 safety), <=7.61h; worker peak
<=1536MiB and available RAM >4*peak+1024MiB. Four training workers.

Append-only private run root local/runs/strong-component-regularization;
source/config/runtime/data/partition hashes bind all caches. Preserve failures.
Actual terminal events trigger fresh-process audit. Health observations only
every600s, no intermediate log/PID polling. No full-data fitting, packages,
desktop writes or uploads. Publish validated public changes promptly.

## V49 handoff

V49 local completion/audit ended 2026-09-29 06:06 UTC, no finalist. TANGOS
mean package increments iron -.002789242466, time +.000550881594 (time has
one negative seed). Saved audit passed 80 cold models, max diff1.14e-13;
summary SHA256 991f2b16712b9182983c72b90a964c64e2a8fa6ab2d746f840b2e9326e70a235.
This closes its frozen recipe, not SAM/EMA or all regularization.
