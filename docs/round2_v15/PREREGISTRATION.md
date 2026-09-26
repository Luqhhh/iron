# V15: task-specific expert routing

Frozen before official-data fits. Goal: platform score above 96.4. Latest
user-reported platform best is A60=96.3465; V12 iron and V7 time are separately
delivered local references with scores pending. No package is authorized here.

## New evidence and question

V12 and V14 both show asymmetric sharing: joint models improve standalone iron
but worsen time relative to their matched single-output models. This is evidence
consistent with negative transfer, not proof of a gradient-conflict mechanism.
V14's repaired PLE adds zero incremental gain over the delivered local references.
The new question is whether task-specific expert routing can preserve useful
sharing while reducing that asymmetry. Repository search found no prior MMoE or
cross-stitch experiment; closed residual-correction and stacking routes stay closed.

The architectural source is [Ma et al., KDD 2018, Multi-gate Mixture-of-Experts](https://research.google/pubs/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-of-experts/):
shared experts with a learned gate for each task. This round adapts that mechanism
to from-scratch TabM experts. It is not a reproduction of the paper's benchmarks,
and its results are not evidence that this adaptation will improve this dataset.

## Frozen two-recipe experiment

Two dense experts each have two 128-wide TabM blocks, 16 ensemble heads,
independent periodic embeddings (frequency 0.01, embedding/frequency width 16),
dropout 0.1 and 32 latent outputs. Each target has its own linear latent-to-scalar
head. Each row's two latent expert vectors are mixed independently for iron/time,
then passed through the corresponding head. All experts run on every row; no
sparse dispatch, residual prediction or cached OOF vector is an inference input.

The learned gate is a linear map from the 21 training-standardized numeric
features plus training-vocabulary spout one-hot to 2 tasks x 2 expert logits,
followed by softmax over experts. Zero-initialized logits give 0.5/0.5 weights.
The uniform control has exactly the same modules and initial predictions, but
its gate parameters remain frozen at zero. Experts and task heads have identical
capacity across the two recipes. Both can qualify; if only the uniform control
helps, attribute that to the expert/head structure, not learned routing. The
128-wide experts bound compute; no claim of parameter matching to V12 is made.

Target order is iron/time. Retain V12's equal standardized MSE across both
outputs and ensemble members, AdamW lr 0.001/wd 0.0001, batch 256, inner mean
standardized MAE epoch selection, maximum 240 epochs/patience 25, and a fresh
full outer-training refit at that epoch. Each inner/full training partition fits
its own feature and target scalers. No auxiliary gate loss, balancing penalty,
pretrained weights, external data or target-at-inference features.

Budget: 2 recipes x 2 complete seeds (42/3407) x 5 folds = 20 joint outer fits,
plus one separate periodic joint control reproducing both V12 seed42/fold0 outputs
exactly. Eight workers, numeric thread pools and torch pinned to one. Do not
extend the epoch budget after observing cap hits or quality. Preserve all failures.

## References and admission

New platform comparison is **A60**, reconstructed from the verified frozen V36
and N-0048 caches with time alpha 0.60. Also report historical A35 and Q20.
CURRENT local reference remains fixed V12 iron A50 or V7 time A50, each using
its original A35 parent. No reparenting of delivered recipes is implied. For
isolated-column package scores the other target stays A60. Historical V12-V14
specifications and decisions remain unchanged.

Use the frozen sparse grid [0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1]. Weights
for the held seed are selected only on the other seeds; no cross-seed OOF-vector
averaging. Rank incremental gain over CURRENT, with uniform then learned gate
as cost tie order, then target name. Apply unchanged candidate tiers. At most
one finalist may enter confirmation, after all 20 development fits and the
independent audit, only with positive complete gains on both development seeds
against **both A60 and CURRENT**. No partial-pool selection or fallback finalist.

Conditional confirmation: at most ten candidate joint fits at previously used
seeds 7777/12011, zero new baseline fits. Four-seed promotion still requires every
seed positive and positive seed-level paired LCB95; fold counts descriptive only.
The global local working gate 96.25 stays unchanged. V7/V12's release exceptions
do not apply. Full-data fits, new packages, desktop writes and uploads: zero.

## Validation

Before official fits: synthetic checks for exact periodic control; equal initial
uniform/learned outputs; simplex gates and manual mixture arithmetic; gradients
from each task into gates and experts; uniform gates remaining frozen versus
learned task/row-dependent weights; train-only target/feature scalers; outer-label
exclusion; repeat and cold feature-only inference; complete-pool and A60-specific
selection guards. Run locked Python 3.12 tests and reference preflight.

Freeze source/runtime/spec/data/fold/reference hashes before fitting. The complete
independent audit must validate all 20 two-column matrices, metadata and gate
activity, then recompute nested gains, package scores (A60 other column), tiers
and selection without the training scorer. Gate summaries are descriptive, not
extra selection criteria. Official root: `local/runs/round2-v15-task-experts/`.
