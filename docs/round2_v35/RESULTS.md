# V35: extended training removes the cap limitation, not the negative decision

Goal: platform **96.5**. Current user-reported best B0 remains **96.3679**,
gap **0.1321**. No independently verified platform receipt is claimed.
Frozen experiment commit: `b2d50ec`.

## G1: complete development results

Only the iteration ceiling changed from 500 to 3000. Both original arms, both
targets and both complete split seeds were refitted; no subset was selected
from favorable V34 cells. All 40 outer fits / 80 training runs completed without
failure. Table gains are whole-package score points over the matching B0
reference, with the other target unchanged and calibration-only blend weights.

| Target | Recipe | Seed42 gain | Seed3407 gain | Mean gain |
|---|---|---:|---:|---:|
| tap_iron | CRPS_FIXED | +0.000000 | +0.000000 | +0.000000 |
| tap_iron | CRPS_SCALE | +0.000000 | -0.002073 | -0.001036 |
| tap_time_len | CRPS_FIXED | -0.001700 | +0.000000 | -0.000850 |
| tap_time_len | CRPS_SCALE | -0.003333 | +0.000000 | -0.001666 |

No recipe has two positive complete development seeds. Of 40 calibration
blend selections, 36 choose zero weight. The four nonzero choices do not yield
a positive complete-seed increment. Candidate-tier classification remains a
descriptive output and cannot override the frozen stability gates.

Longer training improves the time models' standalone WMAPE: FIXED by
0.000392/0.000385 and SCALE by 0.001250/0.000917 on seeds 42/3407. Those
improvements do not translate into an incremental blend gain on B0. Iron
standalone changes are very small and mixed for SCALE. No local-to-platform
magnitude forecast is inferred.

## The convergence question is now better resolved

All 40 selectors stopped through the unchanged early-stopping rule, **zero**
hit the new 3000-iteration cap. Selected iterations span **256–1084**; the
latest stop was iteration **1144**. This removes the old 500-iteration ceiling
as the explanation for these recipes' current lack of incremental gain under
the fixed stopping rule. It is not a proof of mathematical/global convergence
or of impossibility for every CRPS model. No further cap, patience, learning
rate or tree-capacity change is made within V35. V34's original decision and
artifacts are preserved.

## G0: independent identity and reproduction

- Locked Python 3.12 full suite: **1204 passed, 23 warnings**. V35 targeted tests 3.
- The training configuration differs from V34 in exactly the declared cap;
  original preflight, source, data, references and gates passed identity checks.
- **40/40 original calibration histories reproduced exactly as prefixes**.
- **80/80 original selector/refit prediction vectors reproduced exactly** by
  truncating the new models at the old selected iterations; maximum difference 0.
- Fresh-process audit: 60 units, 80 saved models; cold/order/chunk inference
  difference 0 and independent pooled score difference <=1.26e-16.
- Actual tree fits **75694** across 80 training runs; retained trees **72094**.
  Twenty audited matching B0 units reused; new development reference fits 0.
- Models, predictions, full traces, ledgers and private reports remain outside Git.

## Decision and remaining goal

No confirmation finalist; the conditional confirmation runner returned
`no_development_finalist`. Confirmation fits 0, full-data fits 0, packages 0,
desktop writes 0, agent uploads 0. Seeds 271828/314159 remain unconsumed.
This experiment supplies no new package for the five available user slots.
The platform 96.5 goal remains active and unverified.

## Evidence hashes

- `manifest.json`: `8d429b9d060bd7769e44dcdfe075aeb9422ef6d205350d564f941b9499e9d3fc`
- `summary.json`: `0d8414be08578b3efa67984c3ce8230bbf6e08100c61703b078ecc4a0c310d45`
- `audit.json`: `62ff7ac071e30e19a2fe066fb63250d0d5bad0f607d8dd0200ac20a9b54426be`

Private run: `local/runs/round2-v35/development-r1`. Logs under
`local/reports/v35-{python312-tests,development,audit,confirmation-decision}-r1.log`.
