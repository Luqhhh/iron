# DNNR numerical preparation — 2026-09-30

The next independent family after PTaRL is DNNR: local first-order target
extrapolation plus a supervised diagonal feature metric. Primary sources:
[ICML 2022 paper](https://proceedings.mlr.press/v162/nader22a.html),
[author package](https://github.com/younader/dnnr), and
[author API](https://younader.github.io/dnnr/api/).
The inspected package commit is `ca6070734a659a51cbbdfaf9b97e508e8843a1bd`.
The Git checkout and file hashes are private under
`local/research/dnnr-author-source-r1` and `author-dnnr-inspection-r1.json`.
No environment package or dependency lockfile was changed.

The existing V3 kernel family implements kernel ridge; the inspected repository
has no DNNR implementation. The new numerical module reads no data, schedules
no fit and computes no candidate score. Current platform reference remains
user-reported DE3 96.3749. It does not modify the running incumbent-reference
job, its frozen source or the PTaRL queue position.

## Mechanism and scope

For a training anchor `(x_i, y_i)`, prediction derivatives solve the uniformly
weighted local equation `delta_x @ gamma ≈ delta_y` without a free intercept.
A neighbor votes `y_i + gamma @ (query - x_i)`. The package normally averages
three votes and uses roughly three times the feature dimension for estimating
derivatives. This module exposes the first-order term and stable direct least
squares; higher-order terms and a predictor/neighbor index are not implemented.

The learned metric maximizes centered cosine similarity between local absolute
linearization errors and diagonal-scaled neighbor distances. For this term,
the supplied neighbor identities and linearization errors are fixed. It does
not differentiate discrete neighbor selection. Its local error calculation
preserves the author's pseudoinverse of the normal matrix for comparison;
prediction derivatives use direct least squares as the default prediction
solver does. No residual of the incumbent model enters these functions.

The scaling derivative explicitly subtracts the mean of the distance gradient.
It also sets the derivative of distances clipped below epsilon to zero.
Central finite differences independently verify that expression, including
negative scale components. Tests cover the zero-gradient conventions for
constant errors/distances and duplicate points, affine extrapolation beyond
the training neighborhood, rank-deficient minimum-norm solutions, scale
invariance and invalid inputs.

## Measured author-code findings

On synthetic data, the author's `precompute_derivatives=True` path passes a
matrix where its neighbor index expects one point and raises a dimension
error. The ordinary lazy first-order prediction path runs successfully.
This is a reproduced failure, not a forecast for competition data.

The author centering constructor produces a length-119 vector filled with119,
which broadcasts into a matrix, rather than the intended averaging projector.
Its cosine gradient is already nearly centered, so that discrepancy mostly
cancels in the inspected case. It did **not** produce a runtime failure in
the locked environment. The new term uses explicit mathematical centering
and no dense projector.

Separate source comparison on synthetic points measured:

| comparison | maximum difference |
|---|---:|
| local first-order predictions vs author's lazy KDTree path | 0 |
| centered-cosine cost | 0 |
| scale gradient vs author implementation | 1.10731e-11 |

These differences concern numerical terms only. No learned-metric model has
been trained, and this is not a full paper/package reproduction. Primary
source files, original failure, successful control and parity receipt are
preserved privately. The source's small denominator epsilon in its cosine
backward expression is not part of the exact analytical derivative used here.

## Required for a future candidate

The future matched set must distinguish ordinary KNN, fixed-metric Taylor
prediction, and learned-metric Taylor prediction. A full method requires a
training-only encoder, deterministic tied neighbors, explicit training-only
metric fitting/calibration, saved arrays and independent cold audit. The
author's epoch index is built at the beginning of an epoch while scales change
inside it; the handling of that graph/coordinate mismatch must be declared
before any evaluation, not silently repaired after seeing scores.

Keep the serial order: current reference completion, PTaRL, then DNNR and
DANet. No DNNR official fit, complete-size probe, finalist, package or upload
is authorized by numerical term preparation. Future formal execution needs a
complete implementation, frozen current reference, cost admission, exact-source
locked tests and append-only budget. Screening is at complete coverage, and
promotion remains the existing two-development-seed mechanism gate followed
by four positive split seeds and positive seed-level paired LCB95; folds are
descriptive and split-seed OOF averaging is forbidden. The saved numerical
terms do not relax any gate or reopen a measured negative route.

G0: numerical preparation and source comparisons only. G1: unmeasured.

## Locked engineering result

Locked Python3.12.12 full suite1321 passed,23 warnings,191.96s; targeted11
passed. Private receipt `local/research/dnnr-numerical-receipt-r1.json`, SHA-256
`325d88f08115316fbc75413762b0d37833c3a3ce7668fe174492343c4d23212a`. No official label read or estimator fit occurred in this numerical
preparation; no full-data fits, packages, desktop writes or uploads.
