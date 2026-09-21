# Round2 v0.3: marginal and kernel check

Parent: 11db89e7. Reuse the original two five-fold assignments (42, 3407).
These are repeatedly used development validations, not independent holdouts.
Preserve M0 and all historical evidence. No platform uploads or new raw-data publication.

Frozen experiment.yaml specifies exactly one SVR configuration. Numeric scaling,
spout encoding and median target centering/scaling are training-fold-only. Predictions
are clipped at zero in both OOF and inference; no upper bound or calibration.
Convergence warnings or nonzero fit status fail engineering; no automatic retry.

M1 uses hdquantiles(prob=[0.5]) on each target's training-fold labels, without
trimming or grouping. Record ordinary/HD median differences and validation errors.
M1 requires 20 statistical estimates; K1 requires 20 CV fits and two small synthetic
engineering fits. At most two selected K1 full fits (total cap 24); at most two
selected M1 full statistical estimates. Old models and permutations are not rerun.

Target-wise gates: improve pooled WMAPE in both seeds, win at least 7/10 folds,
and degrade no spout by more than 0.001. If both pass, mean WMAPE wins, with
M1 preferred within 1e-6. Eligible candidates receive 2000 paired, spout-stratified
sample-ID bootstraps; identical indices across candidates/baseline and seeds.
This measures fixed-OOF composition sensitivity, not retraining uncertainty.
Intervals do not change frozen eligibility gates. Bootstrap seed: 20260921.

Only selected single-target challengers may be packaged; unchanged target strings
must come directly from M0. Duplicate serialized predictions produce no new package.
No combination without positive platform results for both single-target replacements.

References: [SciPy HD estimator](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mstats.hdquantiles.html),
[SVR API](https://scikit-learn.org/1.8/modules/generated/sklearn.svm.SVR.html).
