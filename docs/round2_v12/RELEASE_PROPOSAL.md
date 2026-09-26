# Proposed isolated V12 iron release — authorization pending

Concrete candidate: **V12_IRON_JOINT_PLR001_A50**.
Parent: existing A35 ZIP, SHA-256
`b4e1fc2d1287a213e9d88d1c30a7420ecc222235fcc6b7191e341dfd16924b06`.
Only iron changes: `0.5*A35_iron + 0.5*V12_joint_plr001_iron`.
The time CSV field strings are copied from A35 exactly. Joint training predicts
both outputs, but the time output is not used in this package. No V7-time or
V9-iron full model is needed and no two-column combination is implied.

The0.50 weight is the choice from both original complete development splits
relative to A35. It is not chosen by a new alpha scan after confirmation.
The unchanged promotion evidence is nested four-seed A35 gain+0.016088,
LCB95+0.010939,4/4 positive seeds, and V9-relative gain+0.011627,
LCB95+0.008416,4/4 positive seeds. A separate fixed0.50 read-back yields
seed gains+0.016899/+0.019418/+0.020213/+0.009709, mean+0.016560,
LCB95+0.010934,17/20 positive folds. This is descriptive release-recipe
arithmetic, not a new gate, selection or platform forecast.

The frozen development package score96.226441 remains below96.25. The repository
contract prohibits automatic release, and V7's explicit exception was limited
to V7. Therefore this proposal requires **explicit authorization for this one
candidate's gate exception, one full-data joint fit, one independent replay fit,
and delivery to desktop submission/V12_IRON_JOINT_PLR001_A50**. The user uploads.
`configs/round2_v12/RELEASE_PROPOSAL.yaml` records pending authorization; the
release command refuses it before creating any output directory or fitting.
No global threshold change is proposed.

The prepared runner first verifies original specification, source, data, fold,
reference and prediction identities. It replays joint seed42/fold0 and requires
both output columns to be bit-identical, then cold-checks that model. It uses
the same inner-only scaling and epoch selection followed by a fresh full2754-row
refit for the one joint model. All weights are trained from scratch; no external
data or pretrained weights. Saving an iron-only view retains the fitted joint
model and exposes its iron component at inference.

A fresh process reads only the saved model and label-free query data; full,
reversed, chunked and singleton predictions must meet the frozen tolerance, and
full prediction arrays and CSV bytes must be identical to warm inference.
Validate the322 unique template-ordered IDs, frozen nonnegative clipping, exact
blend arithmetic, zero time-string mismatches and ZIP content/hash before copying
to a new desktop directory. No overwrite, no agent upload. Every failure retains
its run directory and evidence. Historical A35, V7 and alpha packages are preserved.

The zero-fit parent/fixed-weight check is recorded privately in
local/reports/v12-release-proposal-preflight-r1.json. Synthetic release tests cover
time-string preservation, exact arithmetic, both-output and iron-view cold
inference, protected-training-read denial, and pending-authorization rejection.
No official release fit or submission artifact has been produced by this proposal.

## Prepared implementation checks

The adapter is implemented in `src/bf_tap_r2/v12_release.py`; the five synthetic
release tests passed, followed by the locked Python3.12 suite **1052 passed,
23 warnings**. The first synthetic attempt exposed that an iron-only view does
not itself own the joint estimator's optimizer. Saving now removes the optional
optimizer from the underlying estimator, idempotently; the original failed log
is retained as v12-release-targeted-tests-r1.log, and r2 passed. No official-data
release fit was attempted. Parent identity and fixed-weight preflight passed
without model fitting. The pending proposal is rejected before output creation.

## Explicit authorization received (2026-09-26)

The user answered **“授权这一个 V12 包”** to the concrete question covering the
single-candidate gate exception, one replay fit, one full-data joint fit, and
submission-folder delivery at the frozen50% iron recipe. The authorized execution
specification is `configs/round2_v12/RELEASE.yaml`; the original pending proposal
is retained. The scope is this candidate only, with unchanged A35 time strings.
Agent uploads remain0. No second permission is needed for the authorized work.
The private authorization receipt is under local/authorizations/.

The authorized execution completed successfully on2026-09-27: one replay, one
full-data joint model, one independently verified isolated iron ZIP and desktop
copy. See [delivery](DELIVERY.md). The pending proposal YAML is retained as the
original proposal; RELEASE.yaml records the authorization actually executed.
