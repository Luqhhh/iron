# V7b: exact N-0048 architecture, expanded gradient-training coverage

Pre-registered before any V7b fit or outcome, 2026-09-26. This is independent
of the still-running V7 periodic search. User objective remains 96.4 and
external pretrained weights remain prohibited.

The new evidence is source-level: the released N-0048 recipe restores its
inner early-stopping checkpoint without refitting the full outer-training
portion. The full release likewise receives all 2754 rows but trains on its
inner-training subset. This suggests an unmeasured training-procedure change;
it does not reopen the closed in-pool blending or residual-correction routes.

Freeze one candidate: reinitialize the **same** raw-TabM large model, preserve
its MSE/Adam settings, numeric/category preprocessing, target scaling and
recorded best epoch, but train on all outer-training rows. Existing per-fold
epoch choices were made inside their own outer-training portions and can be
reused without consulting outer labels. Verify data/fold hashes, source-code
hashes, parameters, preprocessing statistics and epoch identities first.
No new epoch search is performed on development data. More rows per epoch also
means more optimizer updates; this tests that complete procedure, not a claim
that row coverage is its sole causal component.

One real control (seed 42, fold 2) replays the original inner-training subset
at its recorded epoch 55. It must match the recorded held-out predictions to
1e-9 maximum absolute difference before the ten candidate fits start. This
checks the optimizer loop and initialization, not just parameter names.
The smaller synthetic test also compares the control against the original
V36NetworkRegressor early-stopping implementation.

Primary comparison is fixed **0.65 V36 + 0.35 N-refit** against A35
**0.65 V36 + 0.35 N-original**. No blending weight is selected. Also report the
same endpoint replacement at Q20's fixed 0.20 weight and standalone N-refit
versus N-original. Iron is untouched. Both complete development split seeds
must improve before confirmation seeds may be consumed. Promotion retains
the four-positive-seed/LCB95 rule, descriptive fold counts, and unchanged
96.25 working gate. Candidate-tier classification is descriptive and gives
no release authorization. No new package or desktop write is authorized.

Confirmation can reuse the V5 same-fold time baseline only after checking its
identity and mask. Its N-0048 cache does not retain epoch metadata, so any
confirmation must rerun the original inner epoch selection within that fold,
verify its prediction against the cached N-0048 control, then refit the new
candidate on the full training fold. Up to ten such selection fits and ten
refits are budgeted. Previously used seeds 7777/12011 are excluded from this
round's selection, but are not described as globally untouched data.
