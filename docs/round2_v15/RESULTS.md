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
