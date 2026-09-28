# V33: continuous conditional mixtures against B0

Frozen before official candidate fits, 2026-09-28. User instruction: correct
the interpretation and continue optimization toward platform **96.5**. Current
user-reported best B0 is **96.3679**. No score forecast is made.

## Hypothesis and prior evidence

A three-component continuous Gaussian mixture can represent input-dependent
asymmetry and regimes. Its predicted conditional median minimizes absolute
error under the fitted distribution. Whether likelihood learning improves the
actual held-out absolute error is the experiment, not an assumption.
The network emits component means, scales and weights directly. It is distinct
from V23's fixed 64-bin histogram, whose negative result remains a negative
prior for this route. No histogram bandwidth, kernel, residual corrector, old
capacity recipe or existing blend grid is reopened.

Method source: [Bishop, Mixture Density Networks (1994)](https://www.microsoft.com/en-us/research/publication/mixture-density-networks/).
This is a small implementation of the conditional-mixture idea, not a claimed
reproduction of a published benchmark or evidence of competition performance.

Two fixed arms per target: **GAUSS1**, a single heteroscedastic Gaussian control;
**MDN3**, a three-component mixture. Both use training-standardized numeric
inputs, training-only spout vocabulary, learned periodic encoding and a
128/128 SiLU trunk. The Gaussian control shares the trunk design but has fewer
output parameters; it does not isolate parameter count perfectly. All settings,
including the scale floor preventing density collapse, are in SPEC.yaml.
Only MDN3 may advance, and its mean incremental gain must exceed the control's.

## Validation and identity

Use the actual B0 deployment recipe, refitted on matching training partitions.
The V30 development run already contains the needed 20 B0 reference units:
seeds 42/3407, all five outer folds, inner calibration seed 27001 held fold 0.
Reuse only these references after verifying the pinned original manifest/audit,
original source Git objects or unchanged private files, data/fold hashes,
fit/query IDs, reference weights and component arithmetic. Do not rewrite or
reinterpret V30's failed candidate models. If identity verification fails, stop.

For each outer fold fit a selector on the inner fitting subset, choose its
epoch by calibration median MAE, then its weight by calibration blend MAE on
the frozen V30 grid (smallest weight on ties). Refit fresh on all outer training
rows at that epoch. Outer query frames exclude both targets. No cross-seed OOF
averaging or weight selection using outer labels is permitted.

Development covers **40 candidate outer fits / 80 optimizer runs**, two arms,
two targets, two complete five-fold seeds. No new development baseline fits.
Synthetic preflight permits two optimizer runs and no official labels.
All failures consume budget, are retained and stop completion; no parameter
fallback, additional arm or automatic retry.

Only audited MDN3 with both complete development seed gains positive and a
positive mean advantage over GAUSS1 may reach confirmation. At most one per
target. Freeze eligibility before using previously unconsumed split seeds
271828/314159. Confirmation costs at most 20 candidate outer fits / 40 optimizer
runs and 20 matching B0 factory calls (640 component pipeline fits).
Final admission retains four positive seed gains, positive paired seed LCB95,
and development package score >=96.25. Folds are descriptive. Emit existing
candidate-tier classification as a diagnostic, never as a gate override.
Repeated splits reuse labels; this is stability evidence, not independent data.

## Engineering, stopping and delivery

Pin the existing CPU environment and all compute threads to one per worker;
use locked Python 3.12 with --no-sync. Before official fits verify density against
an independent normal-mixture expression, the median against an independent
scalar CDF root, single-Gaussian behavior, gradients, synthetic learnability,
train-only preprocessing, absent query labels and saved-model inference.
Audit saved models in a fresh process and independently recompute gains,
eligibility and any final decisions. New evidence is append-only under
local/runs/round2-v33; freeze transitive source, runtime, data and reference
identities. Full-data fits, submission packages and agent uploads: **zero**.

Stop this frozen round if no arm qualifies. Do not change parameters to rescue
negative results. The five existing desktop ZIPs remain byte-identical. Only
their explanatory README is corrected under the user's current instruction.
