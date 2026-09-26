# V7 running evidence (2026-09-26)

The platform objective is now **96.4**. Current best remains user-reported
**A35 = 96.3366**, leaving 0.0634. No V7 platform result or release exists.
The user reconfirmed the ban on external pretrained weights.

## G0 engineering

Locked Python 3.12.12 test path: **984 passed, 3 warnings**, including eight new
V7 tests. The previous root lockfile omitted already-declared V4.2 optional
dependencies; reconciliation removed/upgraded zero previously locked versions.
Runtime versions for neural experiments are recorded and enforced separately.
Private-artifact guard passed before publication. Predictions, fit ledger,
runtime manifest and test logs remain under `local/`.

Run: `local/runs/round2-v7-periodic-networks/development-r1`.
Frozen budget: 12 target/recipe units, two complete five-fold split seeds,
120 outer fits (each with an epoch-selection run and a fresh refit).
Eight workers are running with BLAS/OMP/MKL/NUMEXPR and torch pinned to one
thread each. Failed evidence will be retained. Partial completion is **not**
the final candidate ranking.

All five original desktop alpha-search ZIP hashes match the recorded delivery:
A45 `4c12bedae707c306…`, A60 `d5092d400fb5cc62…`, A72 `808f00a9a3385b62…`,
A85 `2af4d68323ba2cf3…`, A100 `865f7290db001d41…`.
No desktop writes, packages, full-data fits or uploads were performed by V7.

## G1 interim complete-coverage results

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
gate pass, or a forecast of 96.4. Final classification awaits the entire pool.

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
