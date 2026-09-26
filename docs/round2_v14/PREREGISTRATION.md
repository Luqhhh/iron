# V14: joint supervision with repaired PLE-B

Frozen before official-data fitting. Platform objective remains above 96.4;
reported best is A35=96.3366. Delivered V7 time and V12 iron have no attributed
platform feedback. They are target-specific local references, not platform bests.

## Rationale and bounded question

V12 established a four-seed iron increment from joint supervision. V13 established
that repairing PLE-B restores nonlinear gradients and improves standalone models,
but its single-output models add no gain over existing candidates. This new round
asks whether joint supervision changes that negative incremental result. It does
not reopen the V13 decision, assume that two improvements add, or retune its bins.

Compare two joint TabM recipes: PLE-B activation=True (inactive nonlinear branch)
and activation=False (repaired branch). Both use the exact V13 standardized
coordinates, 16 training-only quantile bins and embedding width 16. Both use V12's
joint target order iron/time, training-only separate target means/stds, equal MSE
per row/head/target, inner mean standardized MAE selection, and fresh full outer
training at the selected epoch. Width 256, two blocks, 16 heads, dropout 0.1,
AdamW lr 0.001/wd 0.0001, batch 256, maximum 240 epochs and patience 25 are frozen.

Budget is 2 recipes x 2 complete split seeds (42/3407) x 5 folds = 20 joint fits.
A separate one-fit periodic joint control must reproduce BOTH columns of V12
seed42/fold0 exactly before the batch starts. Each outer fit includes inner epoch
selection and a fresh fit. All fits are from scratch; no external data/weights.
Eight workers; pin BLAS/OMP/MKL/NUMEXPR and torch threads to one. Preserve failed
runs and cap hits; do not extend budgets in response to measured quality.

## Frozen comparisons and decisions

Primary CURRENT references are now delivered V12 isolated iron
`0.5*A35_iron + 0.5*V12_joint_plr001_iron`, and delivered V7 isolated time
`0.5*A35_time + 0.5*V7_tabm_plr001_time`. Each target comparison keeps the other
column at A35. This is not an evaluated or authorized two-column combination.
Report A35 and Q20 comparisons too, with the same sparse nested blend grid
[0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1]. Weight selection excludes the held split
seed; never average OOF predictions across seeds. Report standalone WMAPE against
the corresponding verified V13 single-output control as a mechanism comparison,
not an independent experiment: epoch selection also changes under joint training.

Freeze cost order inactive control, repaired, then target name. Apply the
unchanged candidate-tier policy. Select at most one confirmation finalist by
CURRENT-relative mean increment and frozen cost order, only after all 20 fits
complete and both development splits are positive versus both A35 and CURRENT.
The inactive control may qualify; such an outcome is evidence for joint training
with a standardized linear embedding, not for the repaired nonlinear branch.

Confirmation, if eligible, is at most 10 joint fits on previously used split
seeds 7777/12011, with existing verified same-fold references and zero new baseline
fits. No fallback finalist if confirmation fails. Promotion still needs all four
seeds positive and positive seed-level paired LCB95; folds are descriptive only.
The global local working gate 96.25 is unchanged. V12's one-candidate exception
does not apply here. Full-data fits, packages, desktop writes and uploads: zero.

## Engineering checks

Before fits, verify synthetic train-only bins/scalers for both targets, independent
output gradients through the shared repaired embedding, exact joint control,
repeat and cold inference, absence of outer-label influence, and cached-reference
hash/row/fold guards. Run the locked Python 3.12 suite. Freeze source, runtime,
specification, data, folds and every reference-file identity in the new manifest.
An independent complete-pool audit must reconstruct preprocessing, validate all
joint prediction matrices and metadata, and recompute blend arithmetic, tiers and
finalist selection without the training scorer. No partial quality selection.

Official run root: `local/runs/round2-v14-joint-ple/`. The public specification is
[configs/round2_v14/SPEC.yaml](../../configs/round2_v14/SPEC.yaml). Historical source
modules and experiment specifications remain unchanged.
