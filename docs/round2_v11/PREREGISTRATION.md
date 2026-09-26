# V11: training-only marginal quantile representations

The user authorized continued optimization toward96.4 after the V7 delivery.
External pretrained weights and external data remain forbidden. This round
changes the input representation of from-scratch TabM; it does not modify V7's
release, reopen failed residual/stacking routes, or authorize more packages.

V7 showed that learned numeric representation mattered under complete coverage;
V10's small loss-recipe increment then failed both derived splits. The next
question is whether a **nonlinear marginal coordinate transform**, rather than
another loss/weight variant, improves the direct predictor or complements V7.
Repository searches found no previous QuantileTransformer/rank-Gaussian fit.
This is not a claim that the technique itself is new.

The [official TabM experiment preprocessing](https://github.com/yandex-research/tabm/blob/main/paper/lib/data.py)
includes a training-fitted noisy quantile-to-normal transform. We adopt its
training-size-dependent landmark count and fixed small fitting noise, with
explicit float64 fitting and no subsampling. This is an adapted experiment,
not reproduction of the paper's full training recipe. A uniform-output control
is centered/scaled by `sqrt(12)*(u-0.5)` to separate Gaussian tail geometry from
the empirical-rank mapping. Validation/test values outside the training range
map to the transform's bounds; this limitation must be disclosed.

## Frozen design

Four units per target: normal/uniform quantile mapping crossed with raw-input
TabM/periodic TabM (frequency0.01). Both targets receive all five folds of
seeds42/3407: **80 outer fits**, each with inner epoch selection and a fresh
outer-training-partition refit. All network dimensions, seeds, optimizer,
MSE loss, MAE epoch selection and training limits remain those of V7.
The existing standardized raw/periodic V7 columns are cached controls.

Only the21 original numerical features are transformed. Spout remains the
training-fitted categorical input; IDs, ordering and targets never enter the
transform. Use `n_quantiles=max(min(n_train//30,1000),10)` and fit noise with
standard deviation1e-5 in raw feature units, random seed42. Noise affects only
the fitted quantile landmarks; every actual training/validation/query row is
transformed without row-specific noise. Duplicate feature vectors therefore
remain indistinguishable at inference. Inner transforms see only inner training
rows; refit builds new landmarks using only the full outer-training partition.

Before the80 candidates, replay V7 time seed42/fold0 through the new adapter's
standard-scaling control; require exact prediction equality. This additional
one-fit control is budgeted separately. Any failed fit or incomplete coverage
prevents complete-pool classification; failed evidence is never replaced.

## Frozen comparisons and progression

Record A35 and Q20 comparisons, while judging **incremental** usefulness against
V7's fixed0.5 time blend and V9's fixed0.1 iron blend. V9 is still a local
candidate-pool reference, not a platform incumbent or authorized release.
Verify the same-row/same-fold prediction files and their historical hashes.
Do not compare only against an old weaker parent to count already captured gain.

Apply the existing candidate-tier classifier and frozen cost order (raw normal,
raw uniform, periodic normal, periodic uniform). Alpha selection uses only other
split predictions; OOF vectors from different split seeds are never averaged.
At most **one** target/recipe may enter confirmation, ranked by mean increment
over the appropriate current candidate, then cost order and target. It must
improve on both complete development splits against both A35 and that candidate.
If none qualify, stop with zero derived-seed fits.

Eligible confirmation uses ten fits at7777/12011 with the already verified
same-fold reference caches. These seeds have been used before, not new labels.
All four seeds must be positive and the seed-level paired LCB95 positive;
fold-level results are descriptive. The absolute96.25 local gate is unchanged.
V7's explicit release exception does not transfer to V11. Full-data fits0,
packages0, desktop writes0, uploads0 for this round.

## Engineering

Freeze runtime, source, input, row/fold and reference identities before fits.
Pin BLAS/OMP/MKL/NUMEXPR and torch threads. Tests must exercise training-only
landmarks/noise, monotonicity and bounded extrapolation, unknown spouts,
held-out label/feature isolation, standard-control identity, repeat fits and
cold/query-order inference. Preserve quantile metadata and cap-hit information.
Independent scoring arithmetic and prediction-hash audit precede any quality
claim. Private evidence stays under `local/runs/round2-v11-quantile-representation`.
