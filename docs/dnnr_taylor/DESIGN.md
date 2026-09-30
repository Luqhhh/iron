# First-order DNNR matched mechanism — 2026-09-30

This implements the full first-order predictor and supervised diagonal metric
for a prospective independent family. It has no official-data phase runner yet.
The current best is user-reported DE3 iron, 96.3749; G1 is unmeasured. PTaRL
completed its synthetic probe and failed its frozen cost admission (4.43 hours
projected against 2 hours), with zero official candidate fits. DNNR is next in
the existing serial queue; DANet remains after it.

The mathematical source is the [ICML 2022 paper](https://proceedings.mlr.press/v162/nader22a.html)
and [author package](https://github.com/younader/dnnr), inspected at
`ca6070734a659a51cbbdfaf9b97e508e8843a1bd`. This is an independent,
code-inspired implementation, not exact paper/package reproduction. No package,
dependency version, baseline setting or historical candidate is changed.

## Frozen mechanism

KNN_FIXED averages three training labels. DNNR_FIXED averages three first-order
Taylor votes at training anchors, each with a uniform anchored minimum-norm
least-squares derivative. DNNR_LEARNED uses that same prediction rule after
learning a diagonal feature metric. Fixed Taylor can be evaluated against KNN;
learned Taylor can be evaluated against fixed Taylor. Both Taylor arms are
predeclared eligible methods; neither can be promoted on standalone accuracy.

Each actual fitting partition owns its float64 numeric z scores and sorted
spout one-hot vocabulary. The reserved unknown category has index zero; its
training column may be all zero, so rank-deficient least squares is expected.
Constant numeric columns have scale one. Only the exact unlabeled feature
schema is accepted; IDs never become model inputs. Distance ties use IDs and
an actual anchor identity determines inclusion/exclusion, including coincident
feature vectors. Prediction IDs may not overlap fitting IDs.

Derivatives use `min(n, 3*d)` training rows, including the anchor's zero
equation, in final scaled coordinates. Metric learning uses
`min(n-1, 8*d-1)` other training rows per anchor in unscaled encoder coordinates
for linearization errors. It maximizes centered cosine similarity between those
absolute errors and current scaled distances. One deterministic shuffled SGD
epoch uses learning rate .01 and distance floor 1e-6. Negative scale components
are permitted. Nonfinite or effectively zero scale is an engineering failure;
there is no silent fallback, target clipping or incumbent residual correction.

The metric graph is built once at initial unit scale and remains fixed through
the epoch. All its local linearization errors are also computed before SGD.
Thus scale updates only change distances within the same coherent graph. The
author constructs an index at the start of the epoch but queries it using an
updated scale; that source behavior is explicitly not reproduced. Other
declared deviations are exact neighbors, mathematically explicit centering,
derivative precomputation with a correct scalar anchor, and MAE calibration.
See SOURCE_AUDIT.md for reproduced source defects and numerical parity.

## Inner selection and fresh outer fitting

The unchanged group-safe splitter assigns five folds, seed42; fold0 is held
out for calibration. Duplicate numeric-feature groups stay together, and spout
stratification remains the established rule. All three inner models share this
partition and independently fit their own equivalent encoder. The learned
inner model always executes one epoch. Its initial endpoint is the exact fixed
Taylor control. Raw calibration MAE selects epoch0 or epoch1; exact ties choose
epoch0. Calibration targets do not enter the metric loss or derivatives.

All three outer models start from a newly fitted outer encoder and unit scale.
The learned outer model learns afresh for the selected zero/one epoch count;
no inner scale or coefficients are copied into its training. If epoch0 wins,
the entire learned outer bank must exactly reproduce fixed Taylor. Each pair
therefore consumes six estimators, four derivative banks and one/two metric
epochs. The reservation ledger records each actual fitting operation before
it begins; failed and crashed starts remain consumed. Local equation counts
and update counts are recorded separately from estimator counts.

## Cold evidence and limits

Models are exclusive NPZ writes with no pickle loading, float64 arrays and
external hashes. A pair receipt binds all six model identities, calibration
predictions, selected epoch and computational counts. The independent auditor
reconstructs fitting statistics, exact training arrays, initial graph, metric
errors and scale trajectory, final derivative graph/coefficients, calibration
selection and all outer states. It checks a separate NumPy predictor plus
cold/order/chunk outputs and reconciles the unit's append-only reservations.
Mathematical local-solution/gradient recomputations are reported explicitly;
the auditor creates no estimator fit, metric epoch or selection search.

Numerical tolerance is 1e-8. Saved-file hashes are necessary but insufficient:
tests deliberately rehash corrupted targets, preprocessing, derivatives,
gradients, selection and accounting to check independent rejection. Synthetic
tests do not establish official accuracy, blend gain or platform rank.

Before official execution, a complete-size synthetic probe, exact-source locked
Python3.12 test receipt, resource admission, current four-seed DE3 reference,
and frozen data/fold/source identities are required. The phase controller must
enforce full-coverage development and conditional confirmation budgets. The
prospective pool/gates are in configs/dnnr_taylor/SPEC.yaml; they preserve the
existing +.01 development mechanism and local96.25 gate, followed by four
positive split seeds and positive seed-level paired LCB95. Candidate versus
matched control is measured as incumbent-relative fixed-.2 blend gain; the
other target remains unchanged. Folds are descriptive, cross-split OOF
averaging is forbidden, and candidate tiers do not authorize release.

No official label read, official fit, full-data fit, package, desktop write or
upload occurred in this model preparation. The milestones96.4/96.45/96.5
remain active and unmet.
