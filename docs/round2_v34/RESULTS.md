# V34 CRPS trees: complete development, no qualifying candidate

Current goal remains platform **96.5**, current user-reported B0 best
**96.3679**, gap **0.1321**. No independently verified receipt is claimed.
Implementation/specification frozen at `27010e6`.

## G1: incremental quality

All 40 outer development fits / 80 training runs completed with zero failures,
both targets, both arms, seeds 42/3407, all five folds. Each blend weight was
selected inside its own outer training partition using the matching B0
calibration reference. No cross-seed OOF averaging or outer-label selection.
The other target remains B0. Gains below are whole-package score points.

| Target | Arm | Seed42 gain | Seed3407 gain | Mean gain |
|---|---|---:|---:|---:|
| tap_iron | CRPS_FIXED | +0.000000 | +0.000000 | +0.000000 |
| tap_iron | CRPS_SCALE | +0.000000 | -0.001042 | -0.000521 |
| tap_time_len | CRPS_FIXED | -0.001701 | +0.000000 | -0.000850 |
| tap_time_len | CRPS_SCALE | +0.000000 | +0.000000 | +0.000000 |

Neither arm has two positive complete development splits. Thirty-seven of
40 calibration blend selections chose zero member weight. The three nonzero
weights were all 0.05: two iron SCALE folds and one time FIXED fold; their
resulting complete-seed effects were negative. Both arms were prospectively
eligible, so no favorable control was suppressed or substituted after scoring.
The existing candidate-tier classifier was emitted as a diagnostic; it does
not override the two-seed/four-seed requirements. No confirmation finalist.

## Interpretation: training cap matters

This is a negative result for the **frozen 500-iteration recipe**, not proof
that the model family cannot work after adequate convergence.

| Target / arm | Calibration runs hitting cap |
|---|---:|
| iron FIXED | 1/10 |
| iron SCALE | 6/10 |
| time FIXED | 8/10 |
| time SCALE | 10/10 |

In total 25/40 hit the cap; 22/40 chose an iteration >=475. Of the 25 capped
runs, 23 had a lower mean calibration MAE in the last 25 iterations than in
the preceding 25. That is evidence of incomplete convergence for many fits,
not proof that extra iterations would yield a positive outer increment.
No iteration cap or parameter was changed after seeing these outcomes.
The frozen round stops here, preserving both the negative gains and the
convergence limitation. Any later convergence study would require a separate
pre-registered budget and must preserve this decision, rather than relabel
these fits as a successful model or silently extend their run directories.

## G0: engineering and audit

- Locked Python 3.12 full suite: **1201 passed, 23 warnings**; targeted V34 tests 7.
- CRPS gradients match finite differences; the declared metric matches
  numerical integration of CDF-derivative products. Synthetic learning and
  fixed-scale/control invariance passed before official candidate fitting.
- 20 matching B0 reference units reused with original provenance checks;
  new development baseline fits 0. No new dependency or environment upgrade.
- Fresh-process audit: 60 units, 80 saved models; maximum cold/order/chunk
  inference difference **0**, pooled score difference **6.12e-17**.
- Independent supplemental checks covered 80 payloads and 53388 retained trees:
  frozen tree parameters, finite leaf values, step grid membership, calibration
  epoch selection and fresh-refit iteration counts. All passed.
- 54926 actual tree fits across the 80 candidate training runs, including
  early-stopping iterations later discarded. Zero accepted zero-step updates.
- Full models, predictions, detailed logs, ledgers and reports remain private.

## Budget and next action

Conditional confirmation returned `no_development_finalist`. Confirmation
fits 0; seeds 271828/314159 unconsumed; full-data fits 0; packages 0; desktop writes 0;
agent uploads 0. This round supplies no replacement for tomorrow's pending
probes. Platform 96.5 remains unverified; the thread goal remains active.

## Evidence identities

- `manifest.json`: `d93193067764bf32f980af1c407b7ebcdbb84fabd749af204eec8cd0aa35c863`
- `summary.json`: `8a65eb288ba587eb2cc00868763575df00ab3021b921f0a5158b95ea0b552b28`
- `audit.json`: `e0c2cff6a94f91a7a425fe033a1097ae504688084def048187634247e11874cd`
- `supplemental-audit.json`: `b1a838c84dce913c7345dcdbfc2891a2cd4fb5597f4b259522c860b2780bffea`

Private run: `local/runs/round2-v34/development-r1`. Logs under
`local/reports/v34-{python312-tests,development,audit,confirmation-decision}-r1.log`.
