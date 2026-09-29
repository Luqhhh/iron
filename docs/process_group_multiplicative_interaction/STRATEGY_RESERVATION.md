# PROCESS_GROUP_MULTIPLICATIVE_INTERACTION — latest strategy reservation

Date: 2026-09-29. User requested analysis only, then requested submission of
this direction as the latest attempted strategy. This document reserves the
first-ranked direction from that analysis: a process-group multiplicative
interaction network. It authorizes this public strategy reservation only.
Status: **candidate reserved; implementation and formal runs not started**.
No numbered V round is claimed, avoiding collisions with teammate strategies.

## Objective and source of the hypothesis

Explore a materially different representation for the current static round-two
prediction task. Current platform reference is user-reported
V32_TIME_A60V7_50=96.3727; objective >96.4. Neither this reservation nor external
results establish an improvement or forecast its magnitude.

The same-contest baseline uses static process interactions, while related
RIST and slag-viscosity projects motivate explicit process representations:

- [Same-contest baseline](https://github.com/bigbigstrange/Blast-Furnace-Iron-Prediction/tree/c8f38ab3c6e32e9e73a7fdb0c221296b7711098e).
- [RIST ironmaking model](https://github.com/rahulgondhali/RIST-Modelling-for-Blast-Furnace-Ironmaking).
- [Slag-viscosity representation](https://github.com/nav0225/Adv-BF-eta-pred).

Evidence and static source limitations are recorded in
[the repository research](../research/SAME_CONTEST_GITHUB_20260929.md).
These sources inspire a task-specific inductive bias, not a physical law,
validated contest recipe, or permission to reuse external datasets/weights.

## Proposed representation

Use the 21 available instantaneous numeric features. Encode small semantic
feature groups separately, then combine selected pairs of group embeddings
through low-rank bilinear/elementwise multiplicative channels. Retain a raw
feature channel and concatenate the interaction channels before prediction.
A spout category, if supplied by the approved input adapter, is encoded
separately; sample_id is an identity field only.

A preliminary grouping for dictionary review is:

| Group | Features |
|---|---|
| Delivery | air_volume, oxygen, hot_air_temp, humidity, air_speed |
| Pressure | cold_air_press, hot_air_press, furnace_top_press, upper_press_diff, lower_press_diff, total_press_diff, air_press_ratio |
| Thermal | furnace_top_temp_avg, furnace_throat_temp |
| Material/process | coal_rate, gas_rate, pig, all_quality, consumption, fuel_rate, coke_rate |

This grouping covers each numeric input exactly once, but is **not frozen**.
Field meanings, units, existing derived relationships and gas_rate placement
must be verified before choosing the final grouping and interaction pairs.
Multiplication operates on learned representations; no raw feature product,
ratio, monotonic constraint or conservation equation is accepted solely from
its name. Do not invent unavailable liquid levels, geometry or viscosity.

## What the eventual comparison must resolve

The intended matched comparisons are: raw representation; unrestricted
low-rank interaction representation; semantic-group interaction representation;
and an otherwise identical shuffled-group control. Before implementation,
freeze which controls are necessary, their exact parameter counts, input
paths, initialization, backbone, objective and training schedule. Capacity
changes must be disclosed; a larger candidate versus a smaller control cannot
establish a grouping effect.

The mechanism question is whether semantic grouping adds incremental value
beyond generic feature interactions. It is different from teammate R-FIXED /
R-LEARNED periodic projections: those learn cross-feature sinusoidal inputs,
whereas this candidate constrains interactions through process groups. No
SAM/EMA/Mixup/periodic-projection combinations are part of this reservation.
Existing negative results for residual correction, plain rate targets,
ordinary PySR and old-library stacking remain intact. Graph models and target
cascades remain separate research alternatives, not additional candidate arms.

## Evaluation and admission requirements before any execution

Freeze exact target scope, candidate/control pool, fixed blend or component
replacement, matching current-reference caches, all fit/optimizer budgets,
inner epoch selection, fresh refit, source/runtime hashes and measured resource
admission before a synthetic probe or formal training. No numerical budget or
runtime estimate has been approved or measured for this candidate.

Use complete development coverage and same-protocol incremental comparisons
against the current reference; no two-fold candidate ranking. Preserve the
controlling candidate-tier and promotion rules, including four positive split
seed gains and positive paired seed-level LCB95 for promotion. Exact additional
split seeds and any binding round-specific thresholds must be registered in
advance; do not silently copy a historical round's budget or alter its gates.
Repeated splits are not new independent observations. Report historical B0
separately. No cross-seed OOF mixing or labels in model input.

Fit preprocessing only within the relevant training partition. Require cold
saved-model inference, order/chunk checks, leakage checks, mechanism controls
and independent decision verification. Report G0 engineering and G1 quality
separately. Preserve failed evidence and append-only fit identities.

## Current execution and publication state

New probes=0; formal fits=0; full-data refits=0; packages=0; uploads=0.
No training controller or monitoring automation has been started. The next
stage is a concrete frozen design and implementation plan, not a formal run.
Public strategy documentation uses ordinary commit/push on the current branch.
Models, predictions, local reports, ledgers, receipts and ZIPs remain private;
platform submission packages are a separate user-controlled handoff.

## Execution design follow-up

The user requested starting training on 2026-09-29. A concrete
[execution design](DESIGN.md) now specifies the time-only first batch,
controls and cost admission. Written design review is pending; no fits started.
