# V28_EDGE_KAN results: no development finalist

User approved the fixed EDGE/SHARED design. Reservation92b0d71 and immutable SPEC SHA-256 `df80064923350b9499352b3d7868d607abf43a78e09410203a79d11339cc947d` remain the scientific identity. This KAN round on `codex/round2-v28-edge-spline-network` is distinct from the later other-branch V28 axis-endpoint packages. No second recipe was tested.

## G0 engineering

Pinned MIT efficient-kan core initialization, zero-coefficient and deterministic nonzero-coefficient forward/gradient comparisons all had maximum difference0. Full training-only quantile transformation, categorical vocabulary, target scaling, inner epoch selection and fresh outer refit are implemented. EDGE learns output-specific edge curves; SHARED ties coefficient shape across outputs before multiplying per-edge scalers. Both arms retain the same arrays, author RNG order and zero initial coefficients.

Exactly4 G0 optimizers were started and completed with durable started markers and an append-only ledger:2 resource probes and2 synthetic learnability checks. Full-path peak RSS was423.347656MiB SHARED and423.019531MiB EDGE. Train-step p95 was0.008175163s/0.008072775s; evaluation p95 was0.002750592s/0.002825121s. Frozen conservative development projection491.230265s (8.19minutes) passed the7200s cap,1024MiB worker cap and available-memory check. The1024-row synthetic validation MAE was0.379469444/0.338556036 vs training-median constant1.977750944. All spline layers learned nonzero coefficients.

Fresh read-only review identified a nested-directory budget bypass and an original-unit chunk precision gap. Both received failing regression tests before fixes: only canonical phase paths are admitted, and inference pads each model call to256rows (row-independent operations) to preserve batch-shape arithmetic. No model parameters, precision, budget or scientific gate changed. Original-unit numerical fixture at standard deviation300 now passes the1e-6 tolerance. All40 persisted official models independently cold-load and predict with maximum chunk difference `0.0`.

Locked Python3.12 full tests:1152passed,23warnings,117.07s. Clean export without torch or private caches:963passed,67skipped,3warnings,48.57s.12 new tests cover model/control behavior, train-only transforms, refit/cold inference, resource refusal, phase budget enforcement, ledger coverage, independent score arithmetic and hash-bound confirmation. Warnings are existing sklearn/InterpretML/PyTorch-Lightning notices. Private source/runtime/data/fold/reference/G0 identities and run evidence remain under `local/`.

## G1 model quality

Complete development:40/40 outer fits,0failed,80official optimizers,4workers, BLAS/OMP/MKL/NUMEXPR and torch threads1. Actual batch wall time `167.141s` (2.79minutes). Independent cold/model/ledger/transform audits and separate pooled arithmetic/eligibility audit passed.

Primary R28 is the frozen V12 iron plus V21 local time composition `.40*V36 + .25*V7_member + .35*P_LL_T`. Each member replaces only its target column by `.80*R28 + .20*member`; the other column is unchanged. Historical B0 is V12 iron plus V7 time. No OOF vectors are averaged across split seeds.

| Target | Arm | Seed42 gain vs R28 | Seed3407 gain vs R28 | Mean gain vs R28 | Mean same-column gain vs B0 |
|---|---|---:|---:|---:|---:|
| tap_iron | SHARED | +0.000774262 | -0.004309877 | -0.001767808 | -0.001767808 |
| tap_iron | EDGE | -0.008620841 | -0.007363135 | -0.007991988 | -0.007991988 |
| tap_time_len | SHARED | -0.001293642 | -0.003495392 | -0.002394517 | +0.009797812 |
| tap_time_len | EDGE | -0.006378756 | -0.005645385 | -0.006012070 | +0.006180259 |

EDGE minus SHARED mean gain: iron `-0.006224180`, time `-0.003617553`. EDGE is negative on both complete development seeds for both targets and worse than its paired control. No target passes the positive-seed/.01increment/mechanism-contrast gates; selected_for_confirmation is null. Historical positive time gains are effects already contained in the frozen V21 anchor, not a new incumbent-relative improvement. This is a negative result for one frozen recipe, not a general rejection of KAN.

Confirmation0fits, full-data0fits, packages0, desktop writes0 and uploads0. No gates were relaxed and no automatic retry occurred. Summary SHA `907858b099be8a5768fc7ca2e08b89ab3fdabaa866be815a064f0d050ace7055`, cold audit SHA `5dcc148796b87f7b432be6852c6b9203f29ad41495b772a46f0faee444b84590`, independent arithmetic audit SHA `0253f8f10386a2de4027d012a11970edfb46bdee25fbfce3cbfce358979e8cba`.

## Platform context and two-slot request

The reservation's platform reference96.3526 is historical. Other-branch commit1f1b426 later records user-reported B0=96.3679, V20_B0_PLLT_A325=96.3676 and V21_TIME_LOCAL=96.3567; these are not independently verified receipts. B0 is the latest registered platform best. R28 remained frozen despite this feedback. Target>96.4 remains unmet; local changes do not forecast platform magnitude.

The user reports two available submissions today for this strategy line only; teammates own their branches and submissions. This line currently has no newly qualified submission from V17, V20 masked reconstruction, V23_HIST, V27_HARD_TREE or V28_EDGE_KAN. Recommendation: preserve both opportunities rather than fill them with failed candidates or repeat already scored V12/V7/B0. No platform action or package creation follows from this recommendation.

## Subsequent user-authorized SHARED exploration release and feedback (2026-09-28)

The user later approved two exploratory packages from today's own-chat strategy
pool, superseding the initial preserve-slots recommendation above. This is a
separate release exception; it does not revise the frozen EDGE/SHARED development
specification or promote the SHARED control. The mistaken public release
reservation was reverted by a721483; fitted artifacts and delivery records
remain private.

The V28 package is **TODAY_V28_SHARED_TIME_A20**, preserving original V12 iron
field strings and replacing only time by
`max(0,0.80*original V7 time+0.20*new full-fit SHARED member)`.
Its parent is B0, rather than the original development's R28/V21 time anchor.
The fixed-B0/A20 two-seed development mean is -0.000026889501 score points;
the historical same-column B0 gains in the earlier table use a different
resulting column and must not be substituted for this new composition.

The user reports **96.3671**, **-0.0008** against registered B0 **96.3679**.
Evidence is USER_REPORTED_NOT_INDEPENDENTLY_VERIFIED; the platform best stays
B0 and the >96.4 target remains unmet. ZIP SHA256:
`1020e9dd36b09d7edc50180fb502e037a8d24cfc167d5c01a12a06825401a06a`.

G0 release verification passed:2754 full training rows after inner epoch
selection (selected73), zero iron field mismatches, exact blend recomputation
and warm/cold identity,322 ordered test IDs and single-result.csv ZIP.
This separate two-package release used exactly2 full wrapper fits /4 optimizer
runs total (one V28 SHARED and one V23_HIST Gaussian); no CV or confirmation
was repeated. Feedback registration adds no fits, packages or agent uploads.
V23_HIST's separate exploratory package has no reported score yet.

The earlier reported “missing sample_id” failure is resolved by the user's
explicit clarification: **the user uploaded the wrong file**. It is not evidence
that the generated V28 CSV lacked sample_id or that ZIP naming caused that
failure. The formal-name handoff correction remains recorded separately.
Original failed-incident records, ZIPs and model evidence are preserved.
