# optimization-v0.10 / OPT-23 results

The registered V4_R2_CAUSAL_RESIDUAL fails all nine V1-relative quality gates. G0 development execution and independent cold audit both pass; all six outer prediction maximum differences are zero. Close this fixed V4 without hyperparameter, feature, shrinkage or target-ablation search. No final fit, challenger or platform upload. Current release remains V1, user-reported test_a 83.0319, with R2 83.0207 retained for rollback.

## Frozen implementation

Two low-capacity CatBoost MAE regressors predict iron/time residuals relative to R2. Each uses depth 2, 200 rounds, learning rate .03 and L2 20; all other parameters are frozen in experiment.yaml. Inputs are R2 iron/time, rate, inverse-rate, the two symmetric structural directions, spout, 6-hour mean air volume and 6-hour mean total pressure difference. No new base/rate/q training. No V1 coefficients are applied in-sample to construct meta labels. Prediction is max(0, R2 plus learned residual), without a post-score scale search.

All meta training rows use certified monthly OOF forecasts. Labels come from the verified history available at each outer cutoff; all outer predictions complete before scoring-label access. No official test labels or test inputs enter selection. The original as-of builder supplies the two process means. IDs, timestamps, horizons and history target values are excluded from meta features.

## Quality against V1

Deltas are V4 minus V1; positive is regression. J averages cells within each horizon and equally averages the four horizons.

| Metric | V4 | Requirement |
| --- | ---: | --- |
| J | 0.171045320752 | V1 = 0.165732536588 |
| J delta | +0.0053127842 | ≤ −0.0005 |
| H1 E delta | +0.0015334708 | ≤ 0.0002 |
| H1 iron WMAPE delta | +0.0027288677 | ≤ 0.0003 |
| H1 time WMAPE delta | +0.0003380738 | ≤ −0.0005 |
| H2 E delta | +0.0048049822 | ≤ 0.0005 |
| H2 iron WMAPE delta | +0.0053260152 | ≤ 0.0003 |
| H2 time WMAPE delta | +0.0042839492 | ≤ 0.0005 |
| H3 E delta | +0.0074079494 | ≤ 0.0005 |
| H3 iron WMAPE delta | +0.0061956889 | ≤ 0.0003 |
| H3 time WMAPE delta | +0.0086202099 | ≤ 0.0005 |
| H4 E delta | +0.0075047343 | ≤ 0.0005 |
| H4 iron WMAPE delta | +0.0072405006 | ≤ 0.0003 |
| H4 time WMAPE delta | +0.0077689681 | ≤ 0.0005 |
| H1 non-worse origins | 2/6 | ≥ 4/6 |
| Sep/Oct/Nov H1 improved | 1/3 | ≥ 2/3 |
| DEV_LONG E delta | +0.0037369914 | ≤ 0.0007 |
| DEV_SHORT E delta | +0.0036161763 | ≤ 0.0007 |

Every gate fails: J, H1 E, H1 time, H1 origin count, recent H1, H2/H3/H4 E, H2/H3/H4 time, iron and DEV. No candidate is selected.

| H1 origin | E delta vs V1 |
| --- | ---: |
| O202406_H1 | +0.0026650519 |
| O202407_H1 | -0.0009563379 |
| O202408_H1 | +0.0044199456 |
| O202409_H1 | +0.0010251708 |
| O202410_H1 | +0.0023478686 |
| O202411_H1 | -0.0003008744 |

## Diagnostics

Both targets regress at every horizon on average. Iron has a larger positive signed bias: the mean of per-origin H1 bias increases from V1 +3.8529 to V4 +17.0900, and H3 from +25.8866 to +43.6025 (prediction minus actual, original target units). This establishes an overprediction symptom; it does not identify a unique cause or justify another coefficient/parameter scan. The observed E regression is larger at H3/H4 than H1.

Shared calendar-week paired bootstrap uses 1,000 draws, seed 2026, with 975 valid draws having positive target denominators in all units. The V4−V1 J 95% percentile interval is [+0.0037556485, +0.0068453015]; both endpoint signs indicate regression in this retrospective sample. H1 E interval is [−0.0002587264, +0.0035962098]. These consumed retrospective results are not selection-adjusted or independent platform evidence. No platform-score change is inferred.

Per-target/per-spout WMAPE, signed bias and all 18 cells plus DEV_LONG/SHORT are saved. Prediction-change p05/median/p95 relative to both R2 and V1 are recorded by origin, diagnostic only.

## Budget and engineering

Twelve new development residual fits completed: two each at June–November, using 591, 883, 1193, 1506, 1794 and 2127 eligible causal OOF rows respectively. Zero new R2/rate/q fits, zero LAD fits, zero final fits and zero challengers. The conditional final budget (one q plus two residual models) was not consumed. Synthetic persistence tests are separate from competition-data model fits.

All six inference-order reversal checks and residual serialization checks pass, with zero inference fit attempts. The locked Python 3.12.12 test suite passes 221 tests. Independent cold audit reloads eight q models and twelve residual models, reconstructs all OOF process features and label provenance, reproduces all six outer forecasts and repeats reversal checks without fitting. The cold audit passes all eight OOF folds and all six outer origins with maximum forecast difference 0 and zero fit attempts; hashes are recorded in the completion receipt.

## Evidence and reproduction

- Frozen registration and access scope: `configs/optimization_v0_10/experiment.yaml`, `access_scope.yaml`.
- Pre-fit source/registration commit: `bc50eb0`; plan: `docs/optimization_v0_10/PLAN.md`.
- Development: `local/runs/optimization-v0.10-opt23-r1/`.
- Independent cold: `local/runs/optimization-v0.10-opt23-cold-r1/validation.json`.
- Diagnostics: `local/reports/optimization-v0.10-diagnostics-r1.json`.
- Completion: `local/reports/optimization-v0.10-completion-r1.json`.
- Locked tests: `local/reports/pytest-optimization-v0.10-r1.xml`.
- Access ledger: `local/ledgers/optimization-v0.10-development.jsonl`.

```bash
.venv/bin/python -m bf_tap.optimization.residual_run --output local/runs/optimization-v0.10-opt23-NEW
.venv/bin/python scripts/optimization_v10_cold_check.py --source local/runs/optimization-v0.10-opt23-r1 --output local/runs/optimization-v0.10-opt23-cold-NEW
```

Training command is historical reproduction guidance, not authorization to consume additional fits after closure. Preserve existing run directories, models, failures and source manifests. Original V1/R2 ZIPs, release registrations, prior ledgers and desktop V1 are unchanged. Any further residual-model design needs a separate registration; this result does not establish that every possible residual method must fail.
