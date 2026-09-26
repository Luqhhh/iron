# V7 complete development evidence (2026-09-26)

The platform objective is now **96.4**. Current best remains user-reported
**A35 = 96.3366**, leaving 0.0634. No V7 platform result or release exists.
The user reconfirmed the ban on external pretrained weights.

**Development finished: 120/120 outer fits, zero failures.** The strongest
candidate is time `tabm_plr001`: +0.020125 / +0.021452 package-score points on
the two complete development splits, 8/10 folds positive. Its mean gain against
Q20 is also +0.020736. It is frozen for four-split-seed confirmation, now running.
Mean development package score is **96.229585**, below the unchanged 96.25
working gate. No release is authorized and no platform gain is forecast.

## G0 engineering

Latest locked Python 3.12.12 test path: **987 passed, 3 warnings**, including
V7, coverage-control and confirmation tests. The previous root lockfile omitted already-declared V4.2 optional
dependencies; reconciliation removed/upgraded zero previously locked versions.
Runtime versions for neural experiments are recorded and enforced separately.
Private-artifact guard passed before publication. Predictions, fit ledger,
runtime manifest and test logs remain under `local/`.

Run: `local/runs/round2-v7-periodic-networks/development-r1`.
Frozen budget: 12 target/recipe units, two complete five-fold split seeds,
120 outer fits (each with an epoch-selection run and a fresh refit).
Eight workers completed the batch with BLAS/OMP/MKL/NUMEXPR and torch pinned
to one thread each. All 120 prediction hashes were independently checked
against the append-only ledger. Direct arithmetic reconstructed the selected
blend's pooled scores independently of the selection summary (`audit-r1.json`).

All five original desktop alpha-search ZIP hashes match the recorded delivery:
A45 `4c12bedae707c306…`, A60 `d5092d400fb5cc62…`, A72 `808f00a9a3385b62…`,
A85 `2af4d68323ba2cf3…`, A100 `865f7290db001d41…`.
No desktop writes, packages, full-data fits or uploads were performed by V7.

## G1 initial interim record (retained)

Only units complete on **both** seeds are shown here. These three iron units
finished before the TabM and time units; no final selection has been made.
Gains are package-score points against A35's unchanged V36 iron column.

| Iron recipe | seed 42 gain | seed 3407 gain | selected blend weights |
|---|---:|---:|---|
| raw MLP control | 0 | 0 | 0 / 0 |
| periodic MLP, initial scale 0.01 | +0.002894 | +0.001480 | 0.10 / 0.10 |
| periodic MLP, initial scale 0.1 | +0.005235 | +0.003347 | 0.10 / 0.20 |

Blend weights use the other development seed's OOF losses. These are repeated
splits of the same labelled rows, not independently collected data. Positive
development gains do not constitute four-seed confirmation, a local working
gate pass, or a forecast of 96.4. The complete classification follows below.

## G1 complete frozen-pool results

All rows use both complete five-fold development splits. Delta units are full
package score points; iron's Q20 and A35 reference columns are identical.

| Target / recipe | mean delta vs A35 | mean delta vs Q20 | positive folds | development tier |
|---|---:|---:|---:|---|
| time / TabM PLR 0.01 | +0.020788 | +0.020736 | 8/10 | formal |
| time / TabM PLR 0.1 | +0.011477 | +0.011943 | 8/10 | formal |
| time / MLP PLR 0.01 | +0.008353 | +0.009620 | 8/10 | formal |
| iron / TabM PLR 0.01 | +0.005274 | +0.005274 | 8/10 | formal |
| iron / MLP PLR 0.1 | +0.004291 | +0.004291 | 8/10 | formal |
| iron / TabM PLR 0.1 | +0.002319 | +0.002319 | 7/10 | formal |
| iron / MLP PLR 0.01 | +0.002187 | +0.002187 | 7/10 | formal |
| time / MLP PLR 0.1 | +0.001329 | +0.001401 | 5/10 | exploration |
| iron / raw TabM | +0.000893 | +0.000893 | 6/10 | exploration |
| iron / raw MLP | 0 | 0 | 0/10 | not shortlisted |
| time / raw MLP | 0 | 0 | 0/10 | not shortlisted |
| time / raw TabM | 0 | 0 | 0/10 | not shortlisted |

The legacy candidate-tier policy is applied unchanged. Its `formal` label is
only the two-seed development classification; it does not pass the later
four-seed promotion rule. At most one exploration entry is selected by that
policy (time MLP PLR 0.1), and no exploration package is recommended here.

The global rank-one candidate is time TabM PLR 0.01; both cross-seed blend
choices are 0.50. Its standalone time WMAPEs are 0.038216 / 0.038245 against
A35's 0.038284 / 0.038281. The mixed column reaches 0.037880 / 0.037852.
Residual correlations with A35 are 0.9617 / 0.9556; these are descriptive,
not a reopened correlation gate. The raw-input counterpart has zero incremental
blend gain, supporting the learned representation as the measured difference
under this common training protocol.

Confirmation uses seeds 7777/12011, held out of this round's recipe selection
but used in earlier V5 work. Verified same-fold V36 and N-0048 cache columns
reconstruct A35. Each candidate is freshly fitted on the matching outer
training portion. The four-seed blend is evaluated without averaging prediction
vectors across split seeds. Identity, mask, complete-pool and runtime checks
pass before those fits are scheduled.

## Additional training-coverage finding

Source audit of `v5_candidate_release.build_member_package` -> `V36Regressor`
-> `V36NetworkRegressor.fit` shows that the historical N-0048 release passes
the full training frame to `fit`, but its optimizer sees only the inner
training portion (about 80%). The remaining portion selects an early-stopping
checkpoint; the method restores that checkpoint and returns without retraining
on the complete frame. Thus historical wording "refit on all training rows"
describes the frame supplied to `fit`, not full gradient-training coverage.
The original bytes, recorded decisions and submitted scores remain unchanged.

V7 uses inner-only preprocessing for epoch selection, followed by fresh
preprocessing and model initialization on the full outer-training portion.
Its raw controls share this procedure. They differ from N-0048 in architecture
and settings, so this is **not yet a controlled N-0048 coverage experiment**.
An exact-architecture coverage control is a separate potential next strategy;
the source finding itself does not establish a gain.
