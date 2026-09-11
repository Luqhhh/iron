# optimization-v0.11 / OPT-24 results

Completed locally on 2026-09-11. **G0 PASS; G1 FAIL_CLOSE_REGISTERED_V5.** The sole registered V5_TIME_TRAJECTORY fails four quality gates: J, H1 time, H1 improved-origin count and H4. Close this fixed candidate and retain V1_RATE_STRUCTURAL. No final fit, final LAD, challenger ZIP, platform upload or desktop overwrite. V1's user-reported test_a 83.0319 and R2 fallback 83.0207 remain unverified platform reports.

## Frozen implementation and scope

Started from optimization-v0.10@a7bf539c07dc98d59b5c2432c1a60694c52f1fd1 on optimization-v0.11. Registration commit 2722d8f; pre-training legacy metadata compatibility fix 1a5558a. Exactly 24 appended columns: four registered operation variables × 6h/24h × real-time OLS slope, mean absolute adjacent step rate and valid-pair count. Original feature columns/dtypes/values, operation duplicate semantics and E09/R2 training IDs/order/history remain unchanged. Parameters inherit the complete frozen baseline, including MAE/800/depth 5/learning rate .03/L2 5/seed 2026. No windows, variables, seeds, weights, clipping or coefficient searches.

The new time base mixes nonnegative new E09 time with original nonnegative E04 time using the old .8/(1-.8) arithmetic. Direction uses original R2 iron/original rate minus new base time; the existing unusable-rate rule is retained. The six coefficients use only earlier eligible new causal OOF. V5 iron directly copies original V1, without feeding new time back into its iron correction. Old iron, E04, rate, q and residual models receive no new fits.

November is consumed. All results are retrospective, on the six H1 origins, 18-cell grid and DEV_LONG/SHORT. E is the equal-target WMAPE mean; J first averages origins within each horizon, then equally averages four horizons. All deltas below are V5 minus V1. This is not independent holdout or platform confirmation. The timestamp contract remains ASSUMED.

## Frozen acceptance

V1 J = 0.165732536588; V5 J = 0.165870895655. H1 E delta is +0.0001240518. Iron WMAPE deltas are exactly zero in every horizon and iron arrays are exactly equal in all evaluation units.

| Gate | Observed | Requirement | Result |
| --- | ---: | --- | --- |
| Iron exact equality | PASS | all samples exact | PASS |
| delta J | +0.0001383591 | <= -0.0005 | FAIL |
| H1 mean time WMAPE delta | +0.0002481035 | <= -0.0010 | FAIL |
| Strictly improved H1 origins | 2/6 | >= 4/6 | FAIL |
| Improved Sep/Oct/Nov H1 origins | 2/3 | >= 2/3 | PASS |
| H2 mean E delta | +0.0002847835 | <= +0.0005 | PASS |
| H3 mean E delta | -0.0004600572 | <= +0.0005 | PASS |
| H4 mean E delta | +0.0006046582 | <= +0.0005 | FAIL |
| DEV_LONG E delta | +0.0004147516 | <= +0.0007 | PASS |
| DEV_SHORT E delta | -0.0004891348 | <= +0.0007 | PASS |
| Engineering and causal checks | PASS | all required checks | PASS |

| H1 origin | E delta vs V1 |
| --- | ---: |
| O202406_H1 | +0.0004302349 |
| O202407_H1 | +0.0000926157 |
| O202408_H1 | +0.0000793812 |
| O202409_H1 | -0.0004121672 |
| O202410_H1 | +0.0006443187 |
| O202411_H1 | -0.0000900728 |

H2/H3/H4 time WMAPE deltas are +0.0005695671, -0.0009201145, and +0.0012093164. No favorable month, spout, raw-base diagnostic or partial component is promoted.

## P0, budget and provenance

Both old 335-row V1/R2 releases reproduce exactly through their original loaders. All eight original monthly OOF folds and six original outer forecasts reproduce with maximum difference 0. Reused bundles, model hashes, feature schemas, training identities and availability are verified. No identical pre-existing trajectory family was found. All 16 signals are nonconstant in each training cutoff, with finite coverage 99.6633%–99.9587%; the earliest sample lacks enough history. Counts and real observation spans are in feature_audit.json. No external evaluation labels or test distribution were used for feature selection.

Exactly eight competition-data single-target time CatBoost fits completed:

| Cutoff month | Original E09/R2 training rows |
| --- | ---: |
| 04 | 297 |
| 05 | 588 |
| 06 | 888 |
| 07 | 1180 |
| 08 | 1490 |
| 09 | 1803 |
| 10 | 2091 |
| 11 | 2424 |

Exactly six scalar time LAD fits completed:

| Outer month | Earlier eligible OOF rows | alpha_T |
| --- | ---: | ---: |
| 06 | 591 | 0.2706019012 |
| 07 | 883 | 0.3851702136 |
| 08 | 1193 | 0.4040600388 |
| 09 | 1506 | 0.6241796415 |
| 10 | 1794 | 0.4719052195 |
| 11 | 2127 | 0.4394167545 |

Zero new iron/E04/rate/q/residual fits; zero final fits/LAD/challengers. Synthetic test fits are separate. Original OOF provenance and target availability are reconstructed from saved bundle histories; no official target-column reads. All outer predictions and their hashes were frozen before scoring-label access.

## Engineering and preserved repairs

The locked Python 3.12.12 / uv.lock suite passes **235 tests**. Feature tests cover real irregular times, future events/availability, shuffled inputs, duplicates/conflicts, missing/constant/long-gap/boundary cases, unchanged old aggregates under different internal ordering, and exact preservation of old columns. Candidate checks include fixed iron, fallback rules, sample-ID alignment, LAD timing/ties, single-target persistence/budget and strict acceptance.

Two engineering failures are preserved, not deleted or presented as clean first-pass execution:

1. P0 initially stopped before any fit because legacy June+ bundles store component identity in the certified registry rather than training.component. The fix uses the original loader's registry fallback and also verifies role/component/variant. Original manifest, zero-fit counts and source are retained at the run root; verified execution is its execution/ child.
2. After all 8+6 fits and frozen predictions, the first cold audit rejected otherwise identical LAD rows because pandas distinguishes Asia/Shanghai from CSV UTC+08:00 dtype. All 591 June rows had equal values; canonicalizing to the original bundle timezone restored exact whole-frame equality. Commit e7952d5 adds this type repair and a zero-fit completion entry. cold_repair_manifest.json binds the original script snapshot, repaired source, failure, counts and unchanged prediction receipt. No model, feature, label, coefficient, threshold, prediction or numeric prediction tolerance changed.

The repaired independent cold audit passes all eight folds and six outer origins, with **maximum prediction difference 0**, exact frozen iron, input reversal, coefficient optimality certificates and zero CatBoost/LAD fits. Ordinary source verification remains strict for every training, feature and candidate source; only the explicitly archived cold audit script has a recorded replacement. All original V1/R2 bundles and ZIPs and prior ledger hashes verify unchanged.

## Diagnostics

The shared calendar-week paired bootstrap uses 1,000 draws, seed 2026, with 975 valid draws whose target denominators are positive in all units. V5−V1 J 95% percentile interval is [-0.0001560153, +0.0004680808]; H1 E interval is [-0.0002360350, +0.0004527368]. These intervals cross zero and are consumed-data stability diagnostics, not independent confirmation or a basis to relax gates. No expected platform-score gain is inferred.

Per-target, per-spout and per-month WMAPE, signed bias, absolute-error sums and target sums are saved. Per-origin prediction-change quantiles and unusable-rate counts are saved. Paired new base-time versus corrected time error sums/denominators are also saved; base time is diagnostic only and is never a second candidate. The structural correction reduces H1 time error versus its new base in June/July/August/November and increases it in September/October; it still fails the V1-relative gate.

## Evidence and operational boundary

All generated evidence is under `local/runs/optimization-v0.11-opt24-r1/` (ignored by Git). Main completed execution: `execution/`.

- Registration: configs/optimization_v0_11/experiment.yaml, features.yaml, access_scope.yaml; PLAN.md.
- Original P0 failure: manifest.json, failure.json, fit_counts.json, p0_repair.json, trajectory_run_at_p0.py at the run root.
- Execution: manifest.json, p0.json, feature_audit.json, training_records.json, OOF_provenance.json, models/, oof/, corrections/, predictions_complete.json.
- Audit repair: preserved failure.json, source_before_cold_repair/, cold_repair_manifest.json, cold_validation.json, completion.json.
- Decision: acceptance.json, summary.json, metrics.json, fit_counts.json, final_status.json.
- Diagnostics: week_bootstrap.json, paired_changes.json, base_time_diagnostic.json, diagnostics_by_dimension.json, target_month_spout_diagnostics.json, spout_metrics.json and all_units.csv.
- Locked tests: local/reports/pytest-optimization-v0.11-locked-r3.xml; environment-optimization-v0.11-r1.json.
- Append-only access ledger: local/ledgers/optimization-v0.11-development.jsonl.
- Consolidated receipt: local/reports/optimization-v0.11-completion-r1.json.

Training and completion budgets are exhausted/closed for this candidate. Do not rerun the training command under another directory, search additional seeds/windows/weights, or use platform feedback to reopen it. The completion entry only consumes saved predictions and does not authorize additional fits. A final 2,754-row model, final coefficient and new 335-row challenger test_a package are **not executed because quality failed**. P0's test_a checks concern the unchanged old releases only. No upload, release pointer change or desktop replacement occurs. V2/V3/V4 remain closed.
