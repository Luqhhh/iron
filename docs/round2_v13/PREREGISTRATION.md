# V13: repaired PLE-B versus inactive-branch control

Frozen before any official-data fit. New evidence authorizing this separate
experiment is the [synthetic gradient diagnostic](MECHANISM_DIAGNOSTIC.md).
Historical V3.6 source and evidence remain immutable. The user goal is platform
score above96.4; reported best A35=96.3366 and delivered V7 feedback is pending.

## Frozen experiment

Both recipes use standardized numeric inputs and PLE-B embeddings with16
training-fitted quantile bins and embedding width16. Recipe1 retains
activation=True, which leaves the nonlinear branch inactive. Recipe2 uses
activation=False so that branch can learn. Both retain the V7 TabM dimensions,
MSE loss per ensemble head, AdamW settings, seeds, maximum240 epochs and
patience25. Epoch selection minimizes the inner validation standardized MAE
of the ensemble prediction. Fresh initialization and fresh feature/bin/target
statistics are fitted on the complete outer-training partition at that epoch.

The standardized coordinates used to compute bins are exactly those fed into
the embedding. Neither held-out features nor targets may enter fitted scalers
or bin boundaries. Spout uses the training-fitted categorical vocabulary, with
unknowns mapped to index0. Prediction uses original21 features and spout only.
No external data, pretrained weights, sample weighting or residual correction.

There are2 recipes x2 targets x2 split seeds (42/3407) x5 folds =40 outer fits.
Each includes inner selection plus a fresh refit. Before these, a separately
budgeted single-output periodic control through the adapter must reproduce
V7 time seed42/fold0 exactly. Synthetic tests must show zero versus nonzero
nonlinear gradients/weights, matching initial outputs between activation
variants, train-only bin construction, repeat fits and cold inference. Eight
workers and all numeric thread pools pinned to1. Do not launch simultaneously
with V12's eight-worker batch. Failed artifacts are preserved, never replaced.

## Comparisons and gates

All candidate outputs are evaluated as isolated columns. CURRENT is the fixed
V7 time50% blend or V9 iron10% candidate-pool blend; the other column stays A35.
Report A35/Q20 references too. V9 has no release authorization. Compare each
recipe's standalone quality and CURRENT-relative gain; also report the repaired
minus inactive control to identify the nonlinear branch contribution. The
inactive control can qualify: if it does, explicitly attribute gain to the
standardized linear embedding, not restored PLE.

Apply configs/candidate_tiers.yaml without threshold changes. The frozen cost
order is inactive control, repaired PLE, then target name. Select at most one
confirmation candidate by CURRENT-relative incremental mean then that order,
only after all40 fits complete and both development seeds are positive against
A35 and CURRENT. No across-seed OOF averaging. Keep the existing sparse alpha
grid; no additional weight scan. No fallback finalist if confirmation fails.

Eligible confirmation is at most10 candidate fits at7777/12011, with verified
existing same-fold references and no new baseline fits. These are previously
used splits, not newly acquired labels. Promotion requires four positive split
seeds and positive seed-level paired LCB95; fold counts remain descriptive.
The96.25 local working gate is unchanged and V7's release exception does not
apply. Full-data fits0, packages0, desktop writes0 and uploads0 for this round.
Freeze all source/runtime/spec/data/fold/reference identities before fitting.
Audit complete predictions and independent scoring arithmetic before any
quality claim. Official private run root: local/runs/round2-v13-ple-repair/.

## Supplemental reference, declared before complete-pool readout (2026-09-27)

V12 subsequently passed four-seed confirmation and the user explicitly authorized
its isolated50% iron release. V13's originally frozen V7/V9 ranking, candidate
pool, tier classification and confirmation criteria remain unchanged. To avoid
calling an already captured improvement a new one, also report both V13 iron
recipes against fixed `0.5*A35_iron+0.5*V12_joint_plr001_iron` on the complete
42/3407 development splits. This is additional evidence, not a replacement for
the frozen comparison or a new candidate/alpha search. It is declared while
V13 fitting is ongoing, before its complete-pool summary exists, and reads no
partial quality metrics. Use the same sparse alpha grid and verified same-fold
V12 prediction matrices. Run `scripts/round2_v13_v12_increment.py` only after
V13's full independent audit passes. Preserve the original V13 decisions and
report disagreement explicitly; no release or additional fits are authorized.
