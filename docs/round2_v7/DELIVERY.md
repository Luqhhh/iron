# V7_TIME_PLR001_A50 delivery (2026-09-26)

The user explicitly requested the V7 package in the desktop submission folder.
It has been built, independently audited and copied to:

`C:\Users\lqh22\Desktop\submission\V7_TIME_PLR001_A50\Luqhhh_bf_tap_predict_round2.zip`

ZIP SHA-256:
`4382523c7bd688974f87eab2f42502bf8f54b2b330008b36e797ae7672490299`

The ZIP contains only `result.csv`. A separate `README.txt` beside it identifies
the package. The user uploads; agent uploads0. Platform score is not yet known.

## Exact identity

Only time changes: **0.5 × A35_time + 0.5 × V7 tabm_plr001**. Iron retains A35's
original CSV field strings, with0 mismatches. The parent ZIP SHA-256 is
`b4e1fc2d1287a213e9d88d1c30a7420ecc222235fcc6b7191e341dfd16924b06`.
Its old desktop location was absent, so the exact-byte private archived copy
supplied the parent. No new parent model or two-target combination was fitted.

Frozen execution: `configs/round2_v7/RELEASE_R3.yaml`; private run:
`local/runs/round2-v7-periodic-networks/release-r3`.
The original search/release plans and failed preflight directories are retained.
R1 failed before fitting because the desktop parent path was missing. R2 failed
before fitting because a YAML date needed JSON conversion. Both consumed0 fits;
their failure ledgers and R2's partial manifest remain. The date boundary now
has a regression test.

## G0 engineering

The locked Python3.12 suite passed **1011 tests, 23 warnings** before execution.
One original seed42/fold0 model replay reproduces its recorded predictions
exactly. Its saved model also predicts identically in a new process.
One full-data fit then supplies the2754 rows to the original V7 fit procedure:
inner-training-only preprocessing and group-safe MAE epoch selection, followed
by a fresh initialization/refit on all2754 rows. The selected epoch is **75**;
this was selected internally, not changed from a test result.

The full saved model's fresh-process predictions and final CSV bytes are exact
matches; training-file reads are prohibited. Reverse/chunk/singleton checks pass
the frozen1e-6 relative tolerance. All322 IDs are unique and in template order.
The unchanged iron strings have0 mismatches, and direct blended time read-back
has maximum absolute difference0. Source/input hashes remained unchanged.

An independent audit rechecks parent/copy identities, row order, arithmetic,
prediction files, training/inner-row metadata, preprocessing means, exact replay
and the fit ledger. It confirms **one replay fit + one full-data fit total**,
including failed attempts, with no audit fits. Desktop ZIP bytes match the local
verified package. Evidence: `verification.json`, `delivery.json` and
`independent-audit-r1.json` in the private run.

## G1 quality and remaining objective

Four split seeds were positive relative to A35; mean local gain+0.015270 and
seed-level paired LCB95+0.007168. Only14/20 folds were positive, disclosed as
descriptive. The fixed development package score96.229585 is below96.25, so
this delivery uses the user's **candidate-specific exception**; the global gate
remains unchanged. No V9/V10 package is implied by this authorization.

The current registered platform best remains user-reported A35=96.3366. There
is no V7 platform result and no guarantee of96.4. Further offline optimization
is separately preregistered as V11, with no pretrained weights, new submission
packages or automatic uploads authorized for that search.
