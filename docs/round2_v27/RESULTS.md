# V27 deep kernels: complete development, no finalist

2026-09-28, branch `round2-v27-deep-kernel`. The platform target is **96.5**.
The user-reported incumbent remains **B0 = 96.3679**, a gap of **0.1321**.
No platform result was produced by this round.

## Decision

**None of the six frozen recipe/target combinations improved a complete
development split.** Every seed-level gain is zero or negative. No candidate
qualifies for confirmation, the historical candidate-tier diagnostic shortlists
neither a formal nor an exploration candidate, and this round stops here.

Do not spend a platform submission on these candidates or rescue them with a
post-hoc weight, epoch or inducing-rank scan. This is a negative result for the
three specified sparse-GP recipes and selection procedure; it does not establish
that every possible GP or deep-kernel design is ineffective.

## G1: model quality

All deltas below are local package-score points against the matching B0 refit,
with only the named target replaced by its calibration-selected blend. They are
not forecasts of platform deltas. Complete coverage: 2754 rows per split seed,
five outer folds, seeds 42 and 3407.

| Target | Recipe | Seed 42 delta | Seed 3407 delta | Mean delta | Positive seeds |
|---|---|---:|---:|---:|---:|
| iron | GP_ARD | -0.000392 | -0.004092 | -0.002242 | 0/2 |
| iron | DKL_RAW | -0.004925 | -0.003138 | -0.004032 | 0/2 |
| iron | DKL_PLR | -0.002746 | -0.002398 | -0.002572 | 0/2 |
| time | GP_ARD | 0.000000 | 0.000000 | 0.000000 | 0/2 |
| time | DKL_RAW | 0.000000 | -0.003461 | -0.001731 | 0/2 |
| time | DKL_PLR | -0.002363 | 0.000000 | -0.001181 | 0/2 |

The same-protocol B0 package scores are **96.2487508473** and
**96.2465893506**, mean **96.2476700990**. The older cached B0 development mean
was 96.247739, a difference of approximately -0.000069. Their numerical proximity
is descriptive; the matching refit remains the reference for every V27 delta.

### What the mechanism did

| Target | Recipe | Mean standalone WMAPE / B0 WMAPE | Nonzero calibration weights |
|---|---|---:|---:|
| iron | GP_ARD | 1.3243 | 8/10 |
| iron | DKL_RAW | 1.2588 | 6/10 |
| iron | DKL_PLR | 1.1352 | 6/10 |
| time | GP_ARD | 1.6267 | 0/10 |
| time | DKL_RAW | 1.4409 | 1/10 |
| time | DKL_PLR | 1.3670 | 1/10 |

The learned representations improved standalone accuracy relative to the GP_ARD
control in both targets. Even the best of these three standalone recipes,
DKL_PLR, remained about 13.5% worse than B0 on iron and 36.7% worse on time.
Calibration sometimes selected small positive weights, but those selections
yielded no positive pooled outer-seed gain. Nonzero inner weights therefore
provided no evidence of a useful increment on the incumbent.

The failure occurs before the two-positive-development-seed gate. The unused
confirmation seeds 271828 and 314159 were not evaluated. The four-seed rule and
96.25 working gate were preserved; neither needs to be relaxed to interpret
this result. The four-seed rule remains a split-stability check on reused data,
not an independent-data or platform guarantee.

## G0: engineering and evidence

- Locked Python 3.12 suite: **1169 passed, 23 warnings**. The nine new focused
  tests include independent sparse/dense GP algebra, exact-GP inducing limit,
  encoder gradients, training-only preprocessing, persistence and immutable units.
- Fresh-process real-data audit: **passed**, 80 completed units and **120 saved
  models** (60 calibration plus 60 outer-refit models).
- Maximum cold/reversed-order/chunked prediction difference:
  **3.0127012e-11**, below the frozen 1e-9 tolerance.
- Maximum independently recomputed score-delta difference:
  **2.4069288e-16**. Calibration-only weight choices also matched independent
  NumPy arithmetic.
- Reference audit checks fit/query row identities and saved endpoint arithmetic.
  It does not claim an additional cold reconstruction of all B0 components.
- Source, runtime, data, fold and artifact identities were verified. Query frames
  supplied to fit/predict interfaces omitted both target columns. No failed units
  or ownership collisions occurred.

## Cost and execution

| Item | Actual count |
|---|---:|
| Candidate outer fits | 60 |
| Candidate optimizer runs, calibration plus fresh refit | 120 |
| B0 factory calls, shared across recipes and targets | 20 |
| B0 component pipeline fits | 640 |
| Confirmation fits | 0 |
| Full-data fits | 0 |
| Packages / desktop writes / agent uploads | 0 / 0 / 0 |

A B0 pipeline can itself contain bags, inner selection or a fresh refit; 640 is
a pipeline-call count, not an optimizer-run count. The development fit interval
was about **70.44 minutes**, excluding implementation, tests and final audit.

The execution-only amendment overlapped references with a maximum of six B0
factory calls and overlapped the first 24 candidate units at four workers.
The tail and bridge precomputed 10 of the same 20 reference units; the primary
controller verified and reused all 10. No scientific configuration or fit budget
changed. See [execution amendment](EXECUTION_AMENDMENT.md).

## Evidence identities

Private directory: `local/runs/round2-v27/development-r1`.

| Artifact | SHA-256 |
|---|---|
| manifest.json | `d84817b95671bfa415269fdefb94eb4edd0f8a358e351b301885ee596c1adadc` |
| summary.json | `0c882d9e0cb1570ecf0ff8094305ff2a587ad098283c5997994356e90773f5d9` |
| audit.json | `8e62fc1f374f23d719a67287fcc714d3b6e19d78741511f64c461c5a88ccb3fc` |

Private logs: `local/reports/v27-python312-tests-r1.log`,
`v27-development-r1.log`, `v27-audit-dev-r1.log` and
`v27-confirmation-decision-r1.log`. Models, predictions, detailed metadata and
ledgers remain outside Git. Public summaries do not replace those artifacts.

## Next action

Keep the original B0 package and its reported 96.3679 result. This V27 batch
provides no reason to use a platform slot. The gap to 96.5 remains 0.1321.
Any later experiment needs a separately specified hypothesis; these results do
not authorize a broader GP search or another scan of existing blend weights.
