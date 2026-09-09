# OPT-20: rate/iron/time structural regression

The user explicitly authorized training. R2 remains the incumbent. v0.7 is retained
at db32d2f; old-model routing/blending and pseudo-history are closed. The only new
candidate is V1, specified before training in experiment.yaml. This is consumed
retrospective development, with the same strict gates as v0.7, not independent proof.

Train one new CatBoost MAE rate model at each April–November monthly cutoff.
Its features are the original E09 R2 schema, its parameters exactly the original
baseline parameters, its target r=iron/time. Zero-duration rows are excluded from
rate training only; positive durations weight the rate loss, normalized to mean 1.
For observed time t, t*abs(r-r_hat)=abs(iron-t*r_hat), motivating this weighting.
This does not claim equivalence to optimizing both official targets simultaneously.
No current iron/time labels or current observed rate are inference features.

Predict original frozen R2 iron I and time T with the existing complete 80/20 pair.
Predict nonnegative rate R. For R>1e-6, form dI=R*T-I and dT=I/R-T.
For unusable rates both directions are zero; record all such rows. V1 outputs
max(0,I+alpha_I*dI), max(0,T+alpha_T*dT). Both directions use the original I/T,
not one already corrected target. There is no iterative correction or pseudo-history.

For each outer origin, fit two scalar coefficients in [0,1] on only prior monthly
OOF rows whose true labels have become available by that origin. For each target,
minimize sum abs(y-base-alpha*d). This is exactly a weighted median of (y-base)/d
with weights abs(d), clipped to [0,1]; discard zero directions and choose the lower
minimizer on ties, or zero when all directions are zero. Since each target's WMAPE
normalizer is constant during coefficient fitting, this also minimizes that target's
OOF WMAPE. This is a structural correction using a newly trained rate model, with
no search among old endpoints, weights, horizons, spouts or intercepts.

The earliest outer origin June uses April and May OOF predictions. March is the
initial training warmup. Every OOF rate and R2 model is trained strictly before
its fold cutoff, with label availability <= cutoff and original per-sample as-of
history. April/May R2 pairs must be trained identically (8 target regressors total);
June–November R2 pairs are verified existing artifacts. Eight rate fits and twelve
scalar LAD fits complete the development budget. Missing later R2 artifacts fail
rather than silently launching additional training.

Training reads official labels only before November and only if available by
November 1. Every individual model and corrector further filters at its own cutoff.
November evaluation labels are read only from previously authorized score archives,
after all outer predictions finish. A frozen source/input/config manifest and new
append-only ledger precede any target read. No test inputs enter selection.

Evaluate U0/V1 on all 18 cells and DEV_LONG/SHORT, with one correction per outer
origin applied unchanged across its horizons. Reload saved models, verify feature
schemas, training/OOF identities and cutoffs, exact input reversal and serialized
coefficient reproduction. Reproduce archived R2 endpoints within 1e-8. Report model
fit counts separately from LAD coefficients, rate exclusion/fallback counts, rate
quantiles, coefficient values, signed prediction changes and target-specific errors.

Only a complete strict pass can authorize preparation of one final challenger;
final full-training model count is separate from this development budget. Preserve
R2 and require cold-process validation before release. No platform upload is done
by this experiment. A failure closes this specific V1 candidate without adaptive
variants or a second candidate after inspecting results.
