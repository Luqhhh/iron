# Four local/platform screening diagnostics

The user requested four desktop packages after selecting EMA time, D-LMIX
iron, SAM iron and SAM time to test whether a local zero/negative result can
hide a positive platform effect. This explicit authorization applies to these
four exploration releases only. It does not revise historical no-finalist
decisions, the four-seed promotion rule, candidate tiers or global thresholds.

All four use the exact V32_TIME_A60V7_50 parent (user-reported 96.3727,
SHA-256 `54864561c057ae71f7779b15099750f7caf0141c149d609ae281dbaf9ef3aaa7`).
The current reported best at freeze is DE3 iron, 96.3749. Method effects are
judged against V32; beating the incumbent is a separate comparison.

| Candidate | Isolated target | Development seed42 | Development seed3407 | Mean |
|---|---|---:|---:|---:|
| LOC_DIAG_EMA_TIME | time | +0.001992716 | +0.002344908 | +0.002168812 |
| LOC_DIAG_LMIX_IRON | iron | +0.009755154 | -0.012630200 | -0.001437523 |
| LOC_DIAG_SAM_IRON | iron | -0.006021182 | -0.000520690 | -0.003270936 |
| LOC_DIAG_SAM_TIME | time | -0.012875987 | -0.019983331 | -0.016429659 |

Freeze: configs/local_platform_diagnostic_release/RELEASE.json. Each package
uses `parent + 0.5*(new_component - original_component)`, exactly the original
complete-development fixed replacement. Iron replaces the V12 joint component;
time replaces V7 periodic. Preserve the other target's original CSV field
strings. Do not search weights, training seeds, epochs or hyperparameters using
test data or platform scores. The original train-only inner MAE protocol selects
the epoch, then a fresh estimator refits the full training partition.

Execute each method with its unchanged audited historical trainer and runtime:
regularization in the current worktree; D-LMIX in the augmentation worktree.
No mixing of different component trainer versions. Pin development manifests,
summaries, independent audits, all saved development artifacts, source files,
data, full native models and parent before fitting. Reuse the original native
full models; their predictions must replay exactly. There are four new full-data
estimators/eight optimizer runs, zero new CV fits or confirmation seeds. There
is no wall-clock budget. Keep outputs append-only under local/runs; preserve any
failed attempt. Refuse a negative final replacement rather than silently adding
postprocessing absent from the local protocol.

Before desktop delivery, independently audit train-only preprocessing, epoch
selection and full-data refit states. Fresh-process inference prohibits training
reads, requires exact warm/cold predictions and retains the original 5e-4 raw
order/chunk tolerance. Separate processes using the respective historical
sources reload saved states and original native models, then recompute package
arithmetic; CSV numeric tolerance 1e-10, unchanged-field mismatches zero.
Require 322 unique IDs in template order, finite nonnegative predictions, one
result.csv per ZIP, valid CRC, and hash-verified desktop readback. Run the locked
Python3.12 --no-sync tests and private-artifact guard before publication.

Interpretation is declared before feedback: EMA tests whether the +0.01 floor
misses small consistent improvements; D-LMIX tests near-zero, split-disagreeing
methods; SAM iron tests mild consistent local losses; SAM time tests larger
consistent local losses. One platform measurement only establishes its effect
on the fixed test set, not a population false-negative rate or permission to
reopen all negative families. Platform scores are pending, not forecasts.

Destination: C:\Users\lqh22\Desktop\submission-local-platform-diagnostic-20260930.
Users upload and return candidate-specific scores; agent uploads remain zero.
Existing score-seeking chord packages and all old evidence remain preserved.
