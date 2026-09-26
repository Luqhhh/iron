# V10 periodic loss control (2026-09-26)

## G0

All **20/20** candidate outer fits and the one predeclared MSE control completed
without failure. None reached the240-epoch cap. The MSE time seed42/fold0 replay
is **bit-identical** to V7 (maximum absolute prediction difference0).

Locked Python3.12 full test path: **1007 passed, 23 warnings**. The new tests
verify non-cancelling per-head losses and exact gradients, full MSE training
compatibility, train-only preprocessing, outer-label isolation, deterministic
refits, row-independent inference and exact cold-process predictions.

Private run: `local/runs/round2-v10-periodic-loss/development-r1`.
The independent `audit-r1.json` checks20 prediction hashes, source/runtime/data
and fold identity, training and inner-row IDs, preprocessing means, selected
epochs, loss metadata, historical references and direct scoring arithmetic to
1e-12. It also independently reproduces the selected candidate. Audit fits:0.

## G1: small increment beyond V7; confirmation pending

All values below are full-package score-point gains; each development split
contains all five folds. The main reference is the existing **local candidate**
`V7_TIME = 0.5*A35_time + 0.5*V7_tabm_plr001`, not a new platform incumbent.
Cross-seed weight selection never averages OOF vectors between split seeds.

| Loss | A35 mean | V7 seed42 | V7 seed3407 | V7 mean | V7 positive folds | V7 blend weights | Development tier |
|---|---:|---:|---:|---:|---:|---|---|
| MAE | +0.016313 | +0.001001 | +0.000934 | +0.000967 | 7/10 | 0.20 /0.20 | formal |
| SmoothL1 beta0.1 | +0.020849 | +0.002689 | +0.003449 | +0.003069 | 8/10 | 0.35 /0.20 | formal |

Q20-relative means are +0.016426 (MAE) and +0.021295 (SmoothL1). Comparing only
to A35 would mostly re-count V7's captured direction. The smaller V7-relative
increment is the relevant result. Both recipes pass the two-complete-positive-
split prerequisite; the frozen rank selects **SmoothL1 only** for confirmation.
The MAE direction is preserved without a second confirmation allocation.

The selected direction's development package score, retaining A35 iron, is
**96.232644 <96.25**. Its per-seed scores are96.233749/96.231539. Even if the
four-seed check passes, it remains in the candidate pool under the unchanged
absolute gate. Formal development classification is not a release permission.

The confirmation implementation additionally refuses missing/duplicate candidate
pools, foreign or incomplete seed/fold coverage, and selection that loses its
V7-relative incremental gate. A zero-fit preflight verifies32 reference files
and exact agreement with the frozen development reference hashes. Two targeted
selection guards also pass after the final identity-check additions.

The next ten fits use seeds7777/12011 and the existing verified same-fold V7
and V5 reference caches, with no new baseline fits. These are previously used
split seeds, not independent newly collected labels. Positive seed-level LCB
and all four seeds positive remain required; folds are descriptive only.

No external weights/data, full-data model, package, desktop write or upload.
Platform best remains user-reported **A35=96.3366**. No result here establishes
96.4, and no fixed local-to-platform multiplier is used.
