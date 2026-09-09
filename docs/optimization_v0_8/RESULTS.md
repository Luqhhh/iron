# optimization-v0.8 / OPT-20 results

**Development G0 PASS; V1 passes every frozen strict gate. R2 remains incumbent.**
The new candidate uses a duration-weighted MAE rate regressor and two constrained
LAD structural coefficients learned exclusively from earlier temporal OOF rows.
It does not re-open old-model routing/blending searches or pseudo-history.

The experiment was registered in local commit 8f9df0b before target reads or fits.
Six outer origins, 18 cells and DEV_LONG/SHORT were evaluated. All historical months
are consumed retrospective development; these results do not establish platform
improvement. In particular the H1 mean gate passes by only **0.0000116387** beyond
the required 0.0010 improvement. No further candidate was tried after seeing this.

Delta is V1 minus frozen R2; negative is better.

| Check | Delta / result | Gate |
|---|---:|---|
| H1 mean E | −0.0010116387 | PASS |
| H1 improved origins | 5/6 | PASS |
| Sep / Oct / Nov H1 | all improve | PASS |
| H1 iron WMAPE | −0.0001631315 | PASS |
| H1 time WMAPE | −0.0018601458 | PASS |
| H2 mean E | −0.0013970920 | PASS |
| H3 mean E | −0.0018679823 | PASS |
| H4 mean E | −0.0007549185 | PASS |
| extended J | −0.0012579079 | PASS |
| DEV_LONG E | −0.0010661214 | PASS |
| DEV_SHORT E | −0.0017485091 | PASS |

J: **0.16699044445671873 → 0.1657325365884109**.
H1 mean E: **0.15765228628387426 → 0.15664064762790242**.
The H1 improvement is predominantly in time; iron improves only slightly.

| H1 origin | E delta | OOF rows for coefficient fit | alpha iron | alpha time |
|---|---:|---:|---:|---:|
| June | −0.0011946576 | 591 | 0.728344 | 0.326735 |
| July | −0.0002102547 | 883 | 0.791007 | 0.335668 |
| August | +0.0011904659 | 1193 | 0.777700 | 0.316446 |
| September | −0.0008014278 | 1506 | 0.327189 | 0.536127 |
| October | −0.0001003610 | 1794 | 0.404738 | 0.488714 |
| November | −0.0049535967 | 2127 | 0.502430 | 0.428275 |

Every origin uses one pair of coefficients across all its horizons. Coefficients
change only because the training cutoff and eligible prior OOF data change; there
is no horizon/spout routing or held-origin label fitting. April/May seed R2 pairs
were trained with the exact existing algorithm. Later R2 pairs were verified and
reused. Each rate model uses only eligible labels and original as-of features.

Development model fits: **8 rate + 8 early R2 target regressors = 16**.
Separately, **12 scalar LAD coefficients** were fitted. All eight rate training
sets had zero excluded zero-duration rows. No unusable predicted rate occurred
in evaluation. Maximum absolute changes versus R2 were 74.8974 iron and 11.6223
minutes. Per-unit rate quantiles, rate MAE and prediction shifts are retained.

## Engineering evidence

All 5,651 origin-sample outer predictions reproduce exactly in an independent
cold process; input reversal is exact, and inference attempted fits remain zero.
The verifier reconstructs each correction's OOF inputs from its earlier saved
fold, checks disjoint train/evaluation IDs and actual history availability bounds,
and matches fitting labels to the current origin's eligible stored history.
It then independently recomputes coefficients and reloads saved rate/base models.
Archived R2 endpoints reproduce within the frozen 1e-8 tolerance.

Locked Python 3.12 tests: **196 passed**, final report
`local/reports/pytest-optimization-v0.8-r2.xml` (41.67 seconds).
The 14 added tests cover simultaneous structural correction, exact LAD optimization,
clipping/ties, zero-rate fallback, target-column rejection, late-label exclusion,
temporal leakage, insufficient/duplicate OOF and rate model persistence/schema.

The initial cold verifier treated JSON origin keys as integers; normalizing keys
fixed it. The original failed check directory is retained. This changed no model,
coefficient or score. The final-release launcher also had a missing quote in its
OOF CSV path, caught at Python parse time before any execution; it was repaired and
all release scripts compiled before the first final training. No extra model fit
or alternate candidate resulted from either engineering repair.

Development evidence: `local/runs/optimization-v0.8-opt20-r1/`.
Independent audit: `local/runs/optimization-v0.8-opt20-cold-r2/validation.json`.
The shared gate/error artifacts call V1 `U1`; this is explicitly tagged in the
acceptance JSON and does not refer to the closed v0.7 pseudo-history candidate.

## Final candidate: READY_CHALLENGER

One additional rate model was trained on **2,754 rows**, exactly the incumbent R2
training identities and cutoff **2024-12-01 01:44:00+08:00**. The latest observed
label availability is 01:32 that morning, before the cutoff. Final coefficients
use **2,457** April–November OOF rows: alpha iron **0.41483851005524874**, alpha time
**0.6224873204982649**. Final OOF provenance and coefficient reproduction pass an
additional independent audit, including the November fold. No test target was read.

Total model fits this phase: **17 CatBoost target regressors** (16 development +
1 final rate), plus **14 scalar LAD coefficients** (12 development + 2 final).
Original final R2 models were copied unchanged, with **0** new final R2 fits.

The first final cold launch found a missing `schema_version: 1` in its YAML config.
The trained model was retained. A narrowly scoped packaging recovery verifies the
archived original launcher from commit 451c042, proves the fix only adds that schema
field, and verifies all other source/input identities. It uses the same bundle and
coefficients with **zero additional fits**. The failed config and failure report
remain alongside `packaging_repair.json`; `release_validation.json` is the final
successful status. No prediction or model was altered to repair packaging.

The 335-row test_a predictions exactly match a fresh process, input reversal is
exact, and the recovered R2 endpoint exactly matches the unchanged original v0.4
loader: every maximum absolute difference is **0**. Submission IDs, columns,
nonnegative finite values, six-decimal CSV and ZIP readback all pass.

Unique challenger:
`local/runs/optimization-v0.8-v1-challenger-r1/Luqhhh_bf_tap_predict_prelim.zip`

SHA-256:
`fdcbe03e8ea31577bfeed0c45cd0a0013bda42ab88d29db703fad9c2f7e557aa`

Independent final OOF audit: `local/reports/optimization-v0.8-final-oof-r1.json`.
Cold inference config: `local/runs/optimization-v0.8-v1-challenger-r1/cold_data_repaired.yaml`.

**Not uploaded; no platform score exists for V1. R2 remains active at the
user-reported 83.0207.** The desktop still contains R2, and its hash and the original
repository ZIP both remain
`e42602d3045e43b4b49dd1e1c104aa8e5c1f29c639ff5f892b3b07434ed9bbdf`.
Only upload after current platform last-vs-best submission behavior is confirmed;
this execution does not assume failed submissions preserve the current score.
