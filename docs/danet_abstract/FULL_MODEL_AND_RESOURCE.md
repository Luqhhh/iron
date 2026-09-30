# DANet complete matched model and synthetic resource admission

The next measured question is whether learned abstract-layer feature masks
add useful incremental blend gain against the current DE3 incumbent.
DE3=96.3749 is user-reported, not independently verified platform receipt.
The milestones remain96.4/96.45/96.5. Numerical parity and synthetic
learnability are G0 evidence; neither is a G1 score or platform forecast.

## Model and adaptation

`configs/danet_abstract/SPEC.yaml` fixes a20-main-layer network with ten
blocks, width64 and five groups. Each block has two main abstract layers and
a separate raw-feature abstract shortcut; there are30 abstract layers total.
The regression head is256/512/1. With26 encoded inputs it has766487
parameters. The two arms share exactly the same initialization; DANET_FIXED
freezes only mask logits, while DANET_LEARNED learns them. Only the latter
is an eligible optimization candidate.

The original sources are pinned at DANet
`b007c57121ec9082f6ef19ec7465d9df70767c26` and QHoptim
`e81dea3f2765780cf4fbb90b87b22ba7604b8625`. Both MIT notices are retained
under `licenses/`; no package was installed or upgraded in the shared locked
environment. The existing abstract-layer numerical results remain historical
evidence in `SOURCE_AUDIT.md`.

This is a CPU regression adaptation, **not reproduction of paper training**.
The authors' regression wrapper uses MSE and much larger batch/epoch settings.
This recipe fixes normalized-target MAE, batch256, ghost256 and240 maximum
epochs, QHAdam(.9/.999,.8/1), learning rate.008, coupled L2 decay1e-5,
gradient clipping2, and learning-rate factor.95 every20 epochs. Numerical
features use train-partition population z-scores; spout uses train-partition
sorted one-hot categories and a reserved unknown category. All inputs are
cast tofloat32; target center/scale are computed infloat64 on the actual
fitting partition. No clipping or other output postprocessing is applied.

Initial full-network forward parity against the author implementation has
maximum difference0 in the saved synthetic witness. Twenty synthetic QHAdam
equation steps also have maximum difference0 against its author source.
These are equation checks, not trained-network reproduction or official fits.

## Calibration and independent saved audit

Each outer fold trains a matched pair through a group-safe five-fold inner
partition, seed42/held fold0. Both arms run all240 inner epochs. Every epoch's
raw calibration prediction is retained; first minimum raw MAE selects each
arm's own epoch. A fresh network, encoder, target normalization and optimizer
then fit every outer-training row for that selected epoch. Outer query labels
are absent. Each pair consumes four estimator and four optimizer reservations.

Models are exclusive, non-pickle NPZ archives with external SHA-256 anchors,
source hashes, actual partitions, normalization parameters, state tensors,
epoch predictions, permutations, mask movement and learning-rate trajectories.
The independent NumPy path evaluates bisection entmax, affine layers, stored
normalization, gates, group sums, raw shortcuts and head without Torch forward.
It checks all four saved states, train-only data/statistics, matched initial
states, fixed masks, selected-state mask movement, normalization batch counts,
epoch choice and fresh outer refits. Exact saved cold replay is required;
independent, reversed and chunked inference use the prospectivefloat32 bound
`1e-4 + 1e-6*abs(expected)`. That bound was declared before this probe and
does not alter DNNR's preserved, failed1e-8 contract.

Fitting reservations are append-only, hash-bound and single-use; crashes and
failed starts remain consumed. A zero-fit unit audit checks every scoped
reservation, row count, arm, role, epoch count and terminal artifact identity.

## One-shot full-size resource decision

The probe reads no official table. Its fixed synthetic function is
`30+2*sin(x0)+x1*x2+0.5*x3+0.1*epsilon`, seed57331,2204 training rows,
551 queries and26 encoded dimensions. Both inner and outer models per arm
run the full240 epochs, **960 optimizer epochs total**, regardless of the
inner selected epoch. This resource-only mode cannot masquerade as a formal
selected-epoch unit.

Admission requires all of the following:

- All four saved states pass the independent audit, complete reservations and
  closed artifact checks; both arms beat the synthetic training-median control.
- Learned masks move; fixed masks remain exactly unchanged.
- Measured worker peak is at most1536MiB; available memory covers
  `4*peak+1024MiB` both at admission and later verification.
- Conservative20-pair, four-worker development projection
  `20*(fit_and_save_seconds+independent_audit_seconds)/4*1.5+300` is at most7200s.

The exact implementation, licenses, spec, lockedPython3.12 runtime and a passed
full-suite receipt must be frozen before execution. A durable one-shot
supervisor and read-only600-second timer monitor the probe; actual terminal
file events trigger the final audit. No between-check training polling,
restart, failed-probe retry, smaller architecture or relaxed budget is allowed.
All artifacts remain directly under this worktree's private `local/` directory.

## Formal transition remains separate

Successful synthetic admission does not launch official training. A separately
checked, frozen controller must bind the unchanged admitted core and current
DE3 four-seed reference before20 complete development pairs on42/3407
(80 estimators and80 optimizer runs). The isolated endpoint is
`.8*current+.2*member`; the other target stays current. Both seeds must be
positive, mean gain at least.01, mean package score at least96.25, and mean
candidate-minus-matched-control gain positive.

Only earned candidates and necessary controls may proceed to7777/12011.
Promotion requires four positive complete split seeds, positive paired
seed-levelLCB95 using t=2.3533634348018264 and ddof1, and positive four-seed
control contrast. Folds remain descriptive; OOF vectors never mix split seeds.
Existing candidate-tier and evidence-verification policies remain binding.
There is no authorization for automatic full-data fits, packages, desktop
writes or upload. No DANet G1 result is available yet.


LockedPython3.12 final-source checks: **1454 passed,23 existing warnings**;
56 focused checks passed. Exact source and license inventory:841 files.
Private full-suite receipt SHA-256:
`d41139cf367d0f3147e19d105b485be5805c836d8b33386dc983ef9559a9505b`.
The full-size synthetic admission is pending execution; official model fits0.


DANet one-shot full-size synthetic resource probe launched2026-09-30T14:11:49.455483+08:00.
Implementationd4c791a committed/pushed; manifestSHA
60a1c14b8c0d663110bb5c122912af67ed3869003fcb08ef31aae07e3d6940a2.
Private rootlocal/danet-full-resource-r1; serviceiron-danet-resource-r1.service
initialMainPID450875 active/running. Read-only600s timer
iron-danet-resource-r1-monitor.timer; actual terminal supervisor-terminal.json.
No between-check training polling. Planned4estimators/4optimizers/960epochs;
resource resultpending, G1 unmeasured, official fits0. No restart/shrink/gate
relaxation, automatic formal training, full-data fit, package or upload.
