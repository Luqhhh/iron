# V10: bounded loss control on the verified periodic representation

This is an offline follow-up to V7, not a new model family or a release.
The user continues to forbid external pretrained weights. No external data,
full-data fits, submission packages or desktop writes are authorized here.

V3.6 already tested MAE/Huber on raw and PLE networks. Those results remain
negative or weak priors; this round does not claim loss alignment was untried.
The new evidence is V7's complete-coverage periodic TabM: the raw control adds
little, while the learned representation has four positive split seeds and a
positive paired LCB. That representation has only been tested with per-head
MSE. The narrow question is whether an absolute-error-oriented objective adds
anything **beyond the already measured V7 blend**.

## Frozen pool and control

Only `tap_time_len` is evaluated. Two recipes share exactly V7 `tabm_plr001`'s
architecture, preprocessing, optimizer, learning rate, batch size, initial seed,
240-epoch cap, patience and fresh full-training-partition refit:

* per-head MAE;
* per-head SmoothL1, beta 0.1 in training-standardized target coordinates.

Head losses are averaged; inference averages head predictions. Inner selection
still uses MAE of that mean prediction. The beta is a design choice, not a tuned
optimum. A cap hit is reported as a budget limit, not a general family failure.
Identical optimizer settings do not equalize gradient magnitudes; this is a
fixed training-recipe comparison rather than an isolated statistical estimand.

Before the 20 candidate outer fits, the new training implementation must replay
V7's MSE time seed42/fold0 prediction with **zero maximum absolute difference**.
This one control fit is in addition to the candidate budget. An unsuccessful
control ends official-data execution with its evidence retained.

## Evaluation and decision

Both seeds42/3407 receive all five folds. No partial screen or interim selection.
Freeze A35 and Q20 comparisons, but rank and classify against the time candidate
pool reference `V7_TIME = 0.5*A35_time + 0.5*V7_tabm_plr001`.
This is a local candidate reference, not a newly submitted platform incumbent.
The raw MSE endpoint reconstructed from its verified cache is also reported as
a standalone control. Existing historical predictions are never overwritten.

Sparse blend weights use the other split's losses, never mixed-seed OOF vectors.
Apply the existing candidate-tier classifier with MAE before SmoothL1 in a cost
tie. At most one candidate may enter confirmation, and only if both complete
development seeds improve relative to **both A35 and V7_TIME**. Rank eligible
entries by mean V7-relative gain, then frozen cost order. If neither passes,
stop this bounded experiment without consuming derived seeds.

Confirmation, if earned, uses seeds7777/12011 and verified same-fold references.
These seeds have been used before and are not new labels. Promotion requires all
four seeds positive and positive seed-level paired LCB95; folds are descriptive.
The **96.25** working gate remains unchanged. Any release exception for V7/V9
would not imply permission to package V10 or change its gate.

## Engineering and evidence

Use inner-only feature/target preprocessing for epoch selection, then fresh
initialization and preprocessing on the complete outer-training partition.
Only the existing21 numeric features and spout enter the model. Outer labels
cannot enter fit/selection; ID fields only identify folds and ledger records.

Lock Python3.12, pin BLAS/OMP/MKL/NUMEXPR and torch threads, and bind the runtime,
source, spec, data, fold and reference hashes before any fit. Synthetic checks
must cover real gradient/loss behavior, MSE replay compatibility, inner-only
statistics, held-out label isolation and cold inference. Cache files and ledger
remain append-only under `local/runs/round2-v10-periodic-loss/`. Audit complete
coverage and direct scoring arithmetic independently before reporting a result.
