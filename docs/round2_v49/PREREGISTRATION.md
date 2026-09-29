# V49: latent-input-gradient regularization on periodic TabM

Prospective mechanism study, no V49 actual-data fits before this freeze.
V48's standalone improvements failed to increase incumbent blend quality, and
its zero cap hits close further epoch extensions of that recipe. Existing
library diversity searches changed member selection, not latent-gradient training.
Repository source/config/result searches found no TANGOS implementation.

## Mechanism and sources

[Jeffares et al., TANGOS, ICLR 2023](https://arxiv.org/abs/2303.05506) defines
specialization as the mean L1 norm of hidden-unit input attributions and
orthogonalization as mean absolute pairwise cosine similarity of those gradients.
We test this prior on a periodic TabM backbone while retaining its mean individual
MSE objective. This sampled, numeric-only TabM adaptation is not a reproduction
of the paper's architecture, complete Jacobian or benchmark configuration.

We decline a direct output-disagreement objective: the separate
[learner-collusion study](https://arxiv.org/abs/2301.11323) reports that joint
ensemble objectives can create non-generalizing diversity. That is a negative
prior, not a local impossibility proof. Hidden-gradient sparsity also does not
guarantee useful output diversity or a platform gain.

[TabM](https://arxiv.org/abs/2410.24210) motivates the strong ensemble backbone.
Both arms use the V7-like periodic recipe (two width-256 blocks, 16 members,
16 frequencies/embedding coordinates, initialization scale .01, dropout .1).
Both are fresh float64 fits in the current same-protocol evaluator; neither is
claimed to reproduce the historical float32 V7 prediction column. Neither uses
pretrained weights or external data. No target-derived extra input or residual
model. Independent individual losses and mean inference are retained.

## Frozen arms and auxiliary calculation

BASE: standard per-member standardized MSE. TANGOS: same initialization,
minibatch sequence, dropout, optimizer, stopping and refit, plus two fixed terms.
Every fourth zero-based training step use the first 16 shuffled minibatch rows,
put the network in eval mode, rotate the inspected member via (step//4)%16,
and sample 8 distinct final-backbone units using the independent NumPy seed
stream in SPEC.yaml. Compute gradients with respect to the 21 standardized
numeric inputs, holding spout category fixed. NumericPreprocessor's existing
float32 input representation is promoted to float64 for the network.

Specialization = mean over rows/units of sum over features of abs(gradient).
Orthogonalization = mean over rows and the 28 unordered unit pairs of absolute
cosine similarity; each L2 denominator is clamped at 1e-8 for zero gradients.
Weights .01/.01, no hyperparameter grid. On active steps multiply the penalty
by 4 to retain its average coefficient. Auxiliary eval-mode calls must restore
all module modes and must consume no main Torch RNG. No calibration/query rows
enter this term. Training uses differentiable second derivatives; detaching the
attributions would be an implementation failure. Prediction has no auxiliary
forward or test-time adjustment. All constants are frozen in SPEC.yaml.

## G0 and resource admission

Before actual data: full locked Python 3.12 tests including analytic attribution,
finite-difference parameter gradients, zero gradients, gradient descent on a
known penalty, RNG/mode isolation, zero-lambda control identity, row/label
protection and saved inference. Two full 240-epoch synthetic fits with seed
49001: 2754 independent standard-normal 21-feature rows, first 2204 train,
last 550 query; y=10+3*x0+2*sin(x1)+x2*x3+N(0,.2^2), independent spout 1/2/3.
Both arms must beat training-median constant MAE; candidate penalty must be
active/nonzero. No requirement that regularization beat control on this toy,
and no synthetic-based tuning. Report that contrast descriptively.

Fresh-process cold/reversed/chunked prediction difference <=1e-8. Verify
train-only normalization/vocabulary/target scaling and selected epoch traces.
No new package installation. Same runtime lock and reference-cache checks.
Conservative 2x four-worker development timing <=7.61 h; per-worker peak
<=1536 MiB and current available RAM >4 peaks+1024 MiB. Preserve any failure;
no silent reduction in network capacity or change in penalty after results.

## G1 and delivery limits

Current reference fixed at user-reported V32_TIME_A60V7_50=96.3727, with current
iron/time mixture refit per training partition. Verify/reuse the 20 old reference
units. Complete development seeds 42/3407, five folds, both targets and both
arms: 40 outer units, 80 candidate fits. Inner seed27001/fold0 selects raw
ensemble MAE epoch, restores selected state, selects frozen blend weight, then
freshly refits outer training for the chosen epoch count. No cross-seed OOF mix.

Only TANGOS may advance, with both development seed gains positive, mean >=.01,
and larger mean increment than BASE. At most one per target. Then derived
seeds271828/314159, all four gains positive, paired seed LCB95>0 and development
package score>=96.25; fold counts descriptive. BASE is not retroactively eligible.
Measure against the current incumbent, not just against the weaker control.
Audit complete coverage, saved states, epoch and weight selection, prediction
invariance, package scores and gates before any confirmation.

600-second monitoring, no between-check health/metric polling; actual terminal
events may trigger independent audit. No full-data fits, package, desktop writes
or uploads. Milestones 96.4/96.45/96.5 remain unmet. Preserve all earlier rounds.
