# V47: shared-factor spline interactions

Prospective protocol; no V47 real-data fits or label-based tuning precede this freeze.
Milestones 96.4, 96.45, 96.5 remain unmet. User-reported incumbent remains
V32_TIME_A60V7_50 = 96.3727. This round tests a new predictive mechanism;
V46 excluded time-only convex reweighting of the released endpoints from 96.4.

## Hypothesis and prior evidence

V4 explicit spline/tensor terms and selected EBM triples were weak; V42 adaptive
hinge products also failed. Those findings are negative priors, not evidence that
all smooth interactions fail. No existing model in this checkout uses shared
low-rank factors over all distinct-feature spline interactions. Shared parameters
may estimate such interactions more efficiently with 2754 rows. This is a
hypothesis, not an expected platform gain or an inference from marginal correlations.

Source: David Ruegamer, [Scalable Higher-Order Tensor Product Spline Models](https://proceedings.mlr.press/v238/ruegamer24a.html),
AISTATS 2024, equations 10-13 for distinct-feature products and power-sum recursion.
Our signed-rank, piecewise-linear, MAE-trained variant is not a reproduction of
paper benchmark settings, cubic splines or its automatic smoothness selection.

## Fixed paired arms

AFM2 control: additive numeric linear and centered spline effects, categorical
main effect and all distinct-group pair interactions through rank-16 factors.
AHOFM4 candidate: identical structure plus independent rank-16 factors for all
three- and four-group interactions. Groups are the 21 original numerics and one
spout category; never treat multiple basis columns of one feature as distinct
variables. Per-order parameters have independent deterministic random streams,
so pair parameters and minibatch orders start identically across arms.

Each numeric uses unique training-only quantiles at 16 equally spaced probabilities
as knots. Piecewise-linear hat functions interpolate and clamp outside the knot
range; constants give a single constant basis that vanishes after training centering.
Keep separate training-standardized numeric linear terms for tails. Each variable
has its own padded basis; spout is one centered one-hot group, with unknowns all
zero before centering. No targets, sample IDs, query distributions or test rows
enter the transformation. Fit normalization and target mean/std anew per fitting
partition and outer refit.

For each order, multiply basis by feature-specific factors, then compute the
appropriate elementary symmetric polynomial across feature groups using Newton
power sums. Sum rank terms with trainable signed weights (initial .25). AdamW,
standardized MAE, fixed .001 mean squared second coefficient-difference penalty
on valid numeric spline coefficients, no categorical smoothing. This coefficient
penalty is not claimed to equal integrated curvature on nonuniform knots.
Other exact constants are in SPEC.yaml. No tuning grid, residual learner or
pretrained/external data. Control cannot be promoted retrospectively.

## Evaluation and resources

Reuse only the 20 independently verified V30 reference units under their original
private V27 path, with the current frozen iron/time weights. Complete development
coverage: two seeds, five folds, two targets and two arms = 40 outer units,
80 fits (inner epoch/weight selection and fresh outer refit). Inner selection
seed 27001 fold 0, 400 epoch cap/patience 50; training sees no outer labels.
Select earliest strict improvement exceeding 1e-5 standardized MAE; restore that
state, select the frozen blend grid on inner calibration and refit from scratch
for precisely the selected epoch count. Never blend OOF across split seeds.

Only AHOFM4 can reach confirmation, requiring both development seeds positive,
mean package increment >= .01 and a larger mean increment than AFM2. At most one
candidate per target. Then use the two frozen derived seeds; all four gains must
be positive, paired seed LCB95 > 0 and development package score >= 96.25.
Fold sign counts descriptive. Existing gates and old decisions stay unchanged.

Before actual data fits: analytic enumeration/gradient and leakage tests, full
locked Python 3.12 suite, two full-size synthetic fits (400 epochs each), fresh
process inference against independent NumPy dynamic-programming predictions,
reversed/chunked rows, reference identity, memory and resource checks. Synthetic
response is a four-way product of three-level feature effects plus additive
signal and independent noise; both arms must beat a training-median constant,
AHOFM4 must improve over AFM2. No synthetic tuning after viewing its results.
Projected development <= 7.61 hours with 2x timing safety factor, four workers;
worker peak <=1536 MiB and available RAM >4 peaks+1024 MiB.

After a successful development run, independently audit all saved models,
training-only bases, row identities, epoch traces, calibration weights, complete
coverage, scores and decisions before any confirmation. Preserve any failure.
600-second scheduled monitoring only; actual terminal events may trigger audit.
No full-data fits, packages, desktop writes or agent uploads are authorized here.
G0 correctness and G1 quality will be reported separately.

Synthetic generator fixed before execution: seed 47001, 2754 independent
standard-normal 21-feature rows, first 2204 train / last 550 query. Response
10 + 3*x0 + 6*product(level(x1),...,level(x4)) + N(0,.1^2), where level is
-1 below -.5, +1 above .5 and 0 otherwise; independent uniform spout 1/2/3.
The synthetic evaluation labels never select parameters or epochs. Both arms
run all 400 epochs. This tests known four-way learnability, not real-data gain.
