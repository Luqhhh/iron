# V15 development results (2026-09-27)

All 20 joint development fits and the exact two-output periodic control completed
without failures or epoch-cap hits. The independent audit passed all prediction,
identity, training-scale, gate-metadata, nested arithmetic, tier and selection
checks. Locked Python 3.12 tests: 1073 passed, 23 warnings. Private evidence:
`local/runs/round2-v15-task-experts/development-r1`.

## G0: task gates are active

The fixed control retained exactly equal gate weights and zero gate parameter
norm. Learned gates have norms 0.550–0.649 and mean absolute task-weight
differences 0.128–0.173. Thus learned routing is active. Both recipe modules
contain 284870 parameters; learning the gate adds 100 trainable parameters.
These diagnostics are not extra selection criteria. No historical source or
specification changed, and all private evidence stays outside Git.

## G1: only gated time qualifies for confirmation

Gains below are local package score points under the frozen sparse nested blend.
CURRENT is delivered V12 iron A50 or V7 time A50. A60 is the latest user-reported
platform best but its local OOF time optimum differs from its platform optimum;
a large A60-relative local delta is not a platform forecast. Both delivered local
references retain their original A35 parent. Other-column package scores use A60.

| Target | Recipe | Mean vs A60 | Mean vs A35 | CURRENT seed 42 | CURRENT seed 3407 | Tier |
|---|---|---:|---:|---:|---:|---|
| tap_iron | uniform_experts | +0.014270 | +0.014270 | +0.000000 | -0.002092 | not_shortlisted |
| tap_iron | task_gated_experts | +0.023538 | +0.023538 | +0.007200 | -0.002079 | exploration |
| tap_time_len | uniform_experts | +0.028104 | +0.010638 | +0.000778 | -0.000169 | exploration |
| tap_time_len | task_gated_experts | +0.028536 | +0.012050 | +0.001062 | +0.001413 | formal |

Only `tap_time_len / task_gated_experts` passes the two-positive-complete-seed
prerequisite against both A60 and CURRENT. Its increment over V7 is +0.001237
(8/10 positive folds), with nested weights 0.20/0.10. Its isolated-column
development package score is 96.230821, below the unchanged 96.25 working gate.
The effect is small and remains selection evidence, not a reliable magnitude
prediction. Uniform time and both iron units fail a required CURRENT split;
none is substituted or additionally confirmed.

The learned-gate time model is not uniformly better alone than the fixed expert
control: its standalone WMAPE is 0.038856/0.038837 versus 0.038606/0.038831.
Its selected value is the incremental blend, not standalone ranking. Nor does
this prove a specific gradient-conflict mechanism or generic MMoE superiority.

Next: freeze only gated time for ten joint confirmation fits at previously used
split seeds 7777/12011. Keep four positive seeds and positive seed-level paired
LCB95 as admission conditions; folds remain descriptive. Full-data fits, packages,
desktop writes and uploads remain zero. A60=96.3465 is the registered reported
platform best; the above-96.4 goal is not achieved. V12 then V7 remain the pending
platform priorities; no original package is changed.

## Confirmation allocation

`configs/round2_v15/CONFIRMATION.yaml` freezes only gated time for ten joint
candidate fits at 7777/12011. A zero-fit preflight verified 105 reference files
for A60/A35/Q20 and the unchanged V7 time blend on all four split seeds. A60
is reconstructed directly from the frozen V36 and N-0048 fold cache; a separate
affine identity using A35 and Q20 checks the new alpha on every seed. No new
baseline fit is needed. Previously used split seeds are not newly acquired data.

Nine targeted tests passed, covering complete-pool selection, the A60 prerequisite,
rejection of unearned selections and correct time-column extraction from the
joint matrix. The other output is retained for artifact integrity, not for an
additional iron promotion. The independent confirmation audit reconstructs both
target scalers, checks learned-gate metadata and independently recomputes the
four-seed nested blend and admission decisions. No fallback finalist is allowed.

Before confirmation, the locked Python 3.12 full suite passed **1082 tests,
23 warnings**. The original development suite and audited decisions remain
recorded above. The four-seed check is mandatory before any promotion claim.

## Four-seed confirmation: failed

All ten confirmation joint fits completed with no failure and no epoch-cap hit;
selected epochs span 69–137. The independent audit passed 20 candidate prediction
matrices (ten development plus ten confirmation), source/data/reference/fold
identities, training rows and both target scalers, gate metadata, blend arithmetic,
seed-level confidence bounds and admission decisions. No additional fit was used
by the audit. Total round budget consumed: 20 development + 1 exact control +
10 confirmation outer fits; full-data fits and packages remain zero.

| Split seed | Gain vs A60 | A60 weight | Gain vs V7 CURRENT | CURRENT weight |
|---|---:|---:|---:|---:|
| 42 | +0.027189 | 0.50 | +0.001164 | 0.10 |
| 3407 | +0.029884 | 0.50 | +0.001413 | 0.10 |
| 7777 | -0.000307 | 0.50 | -0.001665 | 0.20 |
| 12011 | +0.029504 | 0.50 | +0.001997 | 0.10 |

Against A60: mean **+0.021567**, seed-level LCB95 **+0.004351**, but only **3/4**
positive seeds (7777 is **−0.000307**). A positive lower bound alone is not
sufficient under the frozen all-seeds-positive rule. Against V7 CURRENT: mean
**+0.000727**, LCB95 **−0.001194**, **3/4** positive seeds; 7777 is **−0.001665**.
Fold counts (15/20 and 14/20) remain descriptive, not the reason for refusal.

Weights are reselected on the other three seeds in four-seed nested evaluation;
this explains the small change in the seed42 CURRENT gain from development.
No development result or frozen decision is replaced. The development package
score remains 96.230821 <96.25. The confirmed disposition is **not promoted**.
No fallback finalist, full-data model, package, desktop write or upload follows.

Private confirmation evidence is under
`local/runs/round2-v15-task-experts/confirmation-r1`, including the manifest,
append-only fit ledger, summary and independent audit. The failed evidence is
retained. This registered expert-routing round is complete; the global objective
above 96.4 remains open. Current user-reported best is A60=96.3465. V12 then V7
remain the pending platform priorities, with original A35-parent bytes unchanged.
