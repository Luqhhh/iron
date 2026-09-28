# V34: CRPS location/scale boosting against B0

Frozen before official candidate fitting, 2026-09-28. User goal remains platform
**96.5**, versus user-reported B0 **96.3679**. No platform forecast is made.

## Hypothesis and prior evidence

V33's three-component neural density failed the two-positive-seed prerequisite.
Its single-Gaussian time control had small positive development gains, but remains
ineligible in that frozen round. This is limited motivation to test a different
learner and distribution-training geometry, not permission to promote that control.
V4's low-capacity leaf-distribution/median forests were also negative; this round
uses sequential functional boosting of Gaussian location and log scale rather
than a forest's empirical leaf distribution. It does not reopen those recipes.

Use the probabilistic boosting idea in
[Duan et al., NGBoost (ICML 2020)](https://proceedings.mlr.press/v119/duan20a.html),
with a separately specified normal-CRPS implementation. This is not a benchmark
reproduction or an exact copy of the NGBoost package. Two fixed arms per target:

- **CRPS_FIXED**: update the normal location, with scale fixed at one in
  training-standardized target units.
- **CRPS_SCALE**: update both location and log scale with separate depth-three
  regression trees per boosting step.

Both arms are prospectively eligible candidates. Compare their gains to describe
whether adaptive scale helped; do not require SCALE to win before permitting a
useful FIXED model to advance. One selected recipe per target at most, with
frozen cost order FIXED then SCALE on equal gains.

## Exact numerical recipe

For z=(y-mu)/sigma, CRPS=sigma*[z*(2*Phi(z)-1)+2*phi(z)-1/sqrt(pi)].
Its derivatives in (mu, log sigma) are 1-2*Phi(z) and
sigma*[2*phi(z)-1/sqrt(pi)]. The metric is defined here as twice the integral
of outer products of CDF derivatives: diagonal entries
1/(sigma*sqrt(pi)) and sigma/(2*sqrt(pi)), cross term zero.
Use its inverse to form the natural gradient. This explicit metric is tested
against numerical quadrature; no equivalence to a library's normalization is
assumed. Point prediction is the normal location, also its median.

Train-only numeric mean/std and spout vocabulary; target mean/population std.
Initial standardized mu=0 and log scale=0. Trees use squared-error fitting of
the gradient targets, depth3, min leaf20, all rows/features, seed42. Up to500
iterations, learning rate0.05. Choose the first finite non-increasing training
CRPS step from the frozen descending backtracking grid, including a zero-step
fallback. SCALE log scale is constrained to [-3,2] at every step in training
and prediction. Record scale-boundary fractions and zero steps. This is a
model-parameter constraint, not prediction clipping or post-hoc calibration.
Stopping/selected iteration uses only inner calibration median MAE, patience60.
No hyperparameter, tree-count-cap or loss scan follows a negative result.

## Validation and immutable references

Reuse V30's 20 matching B0 reference units after the same original manifest,
audit, source, data, fold, fit/query and recipe checks as V33. B0 is the actual
deployment mixture, not a weakened comparison. Calibration seed27001/fold0 is
strictly inside each outer training partition. The selector chooses the epoch
and a blend weight from the existing frozen small grid using that calibration
partition only. Refit fresh on all outer training rows. Query frames contain
neither target; no cross-seed OOF averaging or outer-label weight selection.

Development: both targets, seeds42/3407, all five folds: **40 outer fits / 80
training runs**, plus two synthetic preflight runs (small unit-test fixtures
are separate). No new development baseline fits. At most four candidate workers,
each with one compute thread. Preserve CPU environment and locked Python3.12.

Only two positive complete development seed gains permit confirmation. Freeze
the highest-mean eligible recipe for each target before consuming predeclared
seeds271828/314159. Confirmation budget: at most20 candidate outer fits /40
training runs, plus20 B0 factory calls /640 component fits. Final admission:
four positive seed gains, positive seed-level paired LCB95, development package
mean >=96.25. Fold results and candidate tiers are descriptive and cannot
override these gates. Repeated split seeds are not new independent datasets.

## Preflight, audit and stopping

Before official fits: verify analytic gradients by finite differences, metric
by numerical CDF-derivative integration, fixed-scale invariance, training-score
monotonicity, synthetic learning, fit-only preprocessing, label rejection and
saved-model/order/chunk inference. Persist model state without training labels.
Frozen source/spec/runtime/data/reference hashes and append-only events live
under local/runs/round2-v34. Failures consume budget and are retained; no retries
or fallback configurations. Audit all fitted models and independently recompute
weights, scores, eligibility and any final gate decision.

No qualifying development candidate means stop this fixed round. Full-data
fits, new submission packages, desktop writes and agent uploads: **zero**.
Previously delivered packages and all older decisions are preserved.
