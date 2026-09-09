# optimization-v0.9: fixed dual-ratio structural regression

The user authorized OPT-21 then OPT-22 with exactly two development candidates,
eight new development inverse-rate fits and at most one conditional final fit.
The baseline is current V1 (user-reported test_a 83.0319); original V1 and R2 ZIPs,
models, coefficients, source implementations and release registrations are immutable.

OPT-21 reconstructs W0 using exact archived R2 iron and V1 time, aligned by sample
ID with identical metadata/labels. Evaluate all 18 cells and both DEV windows,
per-target/per-spout metrics, signed bias and shared-calendar-week paired bootstrap.
Bootstrap recomputes WMAPE denominators for every cell; the same sampled week
multiplicities apply to every origin. Average cells within horizons then horizons
for J. Also report per-target intervals and DEV intervals. This is a diagnostic
ablation with zero model or coefficient fits and no possible platform package.
It neither selects a new candidate nor changes the already registered OPT-22 gates.

For April–November, train q=time/iron only on positive-iron rows using iron divided
by its positive training mean as sample weight. Keep CatBoost MAE and all existing
parameters/schema fixed. Exclusion affects q only. At inference clip q at zero;
zero is a valid structural time prediction, not a reciprocal error or fallback.
No q floor or additional ratio definition is introduced. The weighting aligns
|time−true_iron*q|, but inference multiplies predicted R2 iron, so this is not an
identity with final time MAE.

Use V1's verified cached base/rate OOF predictions and per-cutoff saved history.
Reproduce R2/V1 outer endpoints from saved models with zero old-model fits. Do not
refit alpha_I: V2 copies exact verified V1 iron; V3 copies exact R2 iron. Both time
branches are max(0, T_R2 + beta*(q_hat*I_R2−T_R2)). All directions use R2 inputs,
never corrected iron. Fit one shared beta per outer origin with the unchanged
exact constrained LAD routine, using only earlier OOF references whose true labels
are available by that origin. Beta is constant across horizons and spouts.

Training feature construction uses the same original per-sample as-of history;
validation freezes history at the model cutoff. Check each OOF train reference
max < its model cutoff, train/history availability <= cutoff, disjoint fit/OOF IDs,
and coefficient target availability <= outer cutoff. No test data or score affects
training/selection. Access ledger and source/input/config manifest precede label
parsing. Official training targets need not be read again: verified saved history
provides each eligible training target and protects the existing model identity.

All gates and the exact V3 tie preference are frozen in experiment.yaml. The V2
iron regression cap is measured against V1, like the other new gates; V3's iron
condition is instead exact equality to R2. H1 non-worse includes equality, recent
improvement is strict, and the V3 tie preference is strict abs(J2−J3)<0.0002.
No candidate is eligible unless all its gates pass. Failure of both candidates
closes ratio-structure extension, with v0.10 causal OOF residual stacking next.

Only after complete evaluation and independent cold OOF/model verification may
one winning candidate get one final q fit. Reuse the final V1 rate, final R2 and
frozen final iron coefficient without retraining. Preserve both old archives;
produce at most one test_a challenger and never upload automatically. Final-target
access must use a separately frozen, manifest-bound final-training ledger if needed.
The desktop V1 file remains intact unless later delivery is requested.

All reported development/bootstrap results are consumed retrospective diagnostics,
not untouched confirmation and not estimates of platform score improvement.
