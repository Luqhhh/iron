# DE3 iron exploration delivered (2026-09-30)

> 状态范围更新（2026-10-01）：DE3 铁量已回传 96.3749，相对 V32 +0.0022；它是历史最佳，不是当前参照。当前同分最佳 Q75/Q100 为 96.3920；DE3＋Q75 已准备但仅作替补、暂不平台测试。下方交付/回传时的“current best”和剩余差值只限当时；最新状态见 [文档索引](../INDEX.md)。

The user requested the highest recent isolated local-gain candidate among EMA
and independent three-model averaging, with a submission package on the desktop.
DE3 iron is the largest of these completed candidates: +0.004381694674 on
seed42 and +0.002578715152 on seed3407, mean **+0.003480204913**. EMA iron is
+0.002709168619 and EMA time +0.002168812097. These are complete five-fold
development comparisons against the current reference, not platform forecasts.

The explicit release is **DE3_IRON_USER_REQUESTED**. Parent is
V32_TIME_A60V7_50, user-reported platform 96.3727. Only the iron component
changes: `parent_iron + .5*(mean(J42,J104729,J130363)-J42)`. This is equivalent
to `.5*V36_iron + .5*mean(J42,J104729,J130363)` up to floating arithmetic.
The unchanged time fields remain byte-for-byte strings from the parent.
There is no EMA stacking, new weight search or two-target combination.

## G0: release and desktop validation passed

- Reused the hash-pinned original full-data J42 estimator and exact saved
  prediction; no J42 retraining. Its original selected epoch is85.
- Added only J104729 and J130363, with selected epochs75 and88: two full-data
  estimators, four optimizer runs, zero new CV or confirmation fits.
- Four new selector/refit saved models pass train-only preprocessing,
  target-scaling, epoch-selection and checkpoint checks.
- Fresh-process inference reads no training inputs; full-batch warm/cold
  prediction difference0. Order/chunk checks pass the existing1e-6 relative
  tolerance.
- Independent fresh-process audit separately reconstructs all three members,
  ensemble arithmetic and CSV bytes, with zero additional fits.
- ZIP contains exactly result.csv;322 unique template-ordered IDs, finite
  nonnegative predictions; time-string mismatches0; component-replacement
  readback difference0. Desktop ZIP is an exact copy by SHA-256.
- Locked Python3.12 full suite1310 passed,23 existing warnings. After the
  fold-array interface correction, focused release/component suite17 passed.
  No broader retest was needed for that array-conversion fix.

The first release attempt failed before any new fit because the native fold
array was mistakenly treated as a pandas Series. Its manifest and failure
remain underrelease-r1. The correction handles the existing ndarray return;
native model/training code was unchanged. Successful fitting usedrelease-r2.

## G1 at delivery: user-requested exploration; platform feedback pending

This candidate still fails the frozen +0.01 mechanism gate and has no derived
four-split-seed confirmation. Preserve its historical no-finalist decision and
its separate legacy candidate-tier classification. Explicit user release
authorization does not establish promotion or expected platform improvement.
Current best96.3727 remains unchanged. No platform upload was performed.

Desktop directory:
`/mnt/c/Users/lqh22/Desktop/submission-DE3-IRON-20260930`.
Upload file: `Luqhhh_bf_tap_predict_round2.zip`;README.txt records recipe,
two-seed evidence and exploration status. The user uploads and returns scores.

ZIP SHA-256:
`81d9d1b12c3b4fa9f40770b2ac2fef0caef7b500c5ff0ab0d7cc2948aa679ee0`.

Private artifacts stay under
`local/runs/de3-user-requested-release-20260930/release-r2`:

- verification.json: `628e4606eb1d778ba5fa80bac0ca5d52d75dfe55c7852ca13d439d1cf662f60e`.
- independent-audit.json: `5139c2ebd130136167273fb5374e290a6c9801e5587d9fe34bd9e939175cb5f4`.
- desktop-delivery.json: `aa67e358991e830e38e4f6abdd7e80c54d7976e5a49b8d29a44b52dbb36211da`.

Implementation8d6c7c2; bounded interface recovery5e0608e. All original
development/cache/package bytes and negative decisions remain preserved.

## Platform feedback received (2026-09-30)

The user reported **DE3_IRON_USER_REQUESTED = 96.3749**, a **+0.0022** gain
over its V32 parent at 96.3727. This is now the registered current reported
platform best. The feedback is user-reported, not an independently verified
platform receipt; the candidate association follows the immediately preceding
DE3 package handoff. Original ZIP bytes, release evidence and desktop README
remain unchanged.

The local gain was +0.003480204913 and the reported platform gain is +0.0022:
both have the same sign for this package. One result does not establish a
general transfer ratio or validate EMA or other unsubmitted candidates.
The original +0.01 mechanism failure, lack of four-seed confirmation and
exploration release identity remain unchanged. G0 release validation still
passes; G1 now has a small positive reported platform result. No additional
fit, package, desktop write or agent upload was performed for the feedback.

Remaining gaps are **0.0251 to 96.4**, **0.0751 to 96.45** and **0.1251 to
96.5**. None of the three milestones has been reached. New comparisons use
this latest registered best; historical frozen experiments retain their own
original references.

Private feedback:
`local/runs/de3-user-requested-release-20260930/platform-feedback-r1/feedback.json`.
