# Round2 V3.1 Directed Search: S0 Fusion Progress

Status: **S0 complete; S1 320-item directed search complete; S1 fusion is provisional pending S3 outer validation; no package and no upload.**

Base commit named by the task book: `5bf681b7cd1680e06aad452a8239decaf7666acb`.
Implementation branch: `round2-v3.1-directed-search`.

## What was implemented

- `src/bf_tap_r2/v3_1_fusion.py`
  - pooled WMAPE over one complete aligned coverage set;
  - monotonic V3.1 triage bands, including `>= 0.15` staying primary;
  - exact fixed-member nonnegative/sum-to-one WMAPE LP using
    `scipy.optimize.linprog(method="highs")`;
  - greedy member selection with LP refitting for a fixed member count.
- `src/bf_tap_r2/v3_1_search.py`
  - S1 budget contract: 320 items, 128 iron / 192 time;
  - canonical trial hash and batch identity requirements;
  - exact identity comparison for cache reuse.
- `configs/round2_v3_1/search.yaml`
  - P0 and L0 references, full-precision L0 weights, monotonic thresholds,
    S1 allocation, pooled-metric rules, batch identity rules, no-upload rule.
- `tests/test_round2_v3_1_search.py`
  - 8 lightweight tests covering LP objective/constraints, cancelling experts,
    monotonic triage, simple mixes, budget contract, and identity mismatch.

No new base model was fitted for S0.

## S0: existing-OOF fusion

S0 used the already available seeds 42/3407 OOF predictions and did not read
labels for any new fit.

Reference values on those two development seeds:

| quantity | value |
|---|---:|
| P0 package score | 96.012388 |
| L0 package score with both-seed in-sample weights | 96.114307 |
| L0 nested cross-seed estimate from V3 | +0.088347 vs P0 |

The `96.114307` value uses weights fitted on the same two seeds and is
optimistic; it is not exchangeable with the V3 cross-seed estimate, and it is
not a platform forecast.

Exact LP behaviour:

- For the frozen L0 member set, the LP reproduced L0 weights and L0 WMAPE to
  numerical tolerance.
- Adding L0 itself as a candidate to the existing OOF library produced only a
  marginal iron target gain:
  - iron target score `96.162254` vs L0 iron target score `96.159157`;
  - time LP selected L0 with weight 1.0.
- The resulting package-level gain is about `+0.0015`, far below any useful
  submission threshold.
- No existing-OOF fusion beyond L0 is therefore promoted.

## S0-directed candidates for S1

The S0 screen selected up to 12 items per target for the next directional batch:
six single-model strong performers, four complementary candidates with the best
simple L0 mixes at alphas `{0.10, 0.25, 0.50}`, and two structurally different
candidates.  Representative selected complements include:

- iron: `KWT`, `KW2`, `QRC`, `C2`, `B3_C2_SEED_ENSEMBLE`;
- time: `KW2`, `JM1`, `AJM1`, `AKW2`, `C2`, `B3_C2_SEED_ENSEMBLE`.

The full machine-readable S0 selection is in the private run:

`local/runs/round2-v3.1-directed-search/s0-fusion-r1/s0_summary.json`.

## Current conclusion

S0 found no existing-OOF combination that materially improves L0.  The useful
result of S0 is not a new package candidate: it is the selection of directions
and complementary members for the S1 320-item neighborhood/expression search.

## Next step

Execute S1 in a separate batch directory with:

- batch ID and canonical trial hashes;
- seed-42 / folds 0–1 pooled-WMAPE coarse screening;
- identity-checked resume only;
- no per-trial release audit;
- exact LP refit for fixed member sets when fusion is evaluated.


## S1: 320 directed configurations

S1 completed with the frozen allocation:

| line | items |
|---|---:|
| time CatBoost neighborhood | 128 |
| iron CatBoost neighborhood | 64 |
| feature/expression | 48 |
| smooth/residual | 48 |
| far family | 32 |

The first coarse pass completed 304/320 and preserved 16 residual-family
failures caused by an index-alignment bug.  The bug was fixed, the failed trial
IDs were rerun in the same identity-checked directory, and the final coarse
ledger has **320 complete trials** with the original failure evidence retained.

Per-target best refined single models on seeds 42/3407 pooled WMAPE:

| target | candidate | pooled WMAPE |
|---|---|---:|
| iron | `v31-s1-expr-iron-0018` | **0.03900850** |
| time | `v31-s1-time-0021-0050` | **0.03898744** |

The iron single model is better than P0 iron but still behind L0 iron.  The time
single model is better than L0 time, but the useful signal appears only in the
fusion stage.

## S1 provisional fusion

Using the existing V2/V3 OOF library plus the S1 refined OOF, with exact LP
refitting on seeds 42/3407, the provisional fusion is:

| target | members | weights |
|---|---|---|
| iron | L0, `v31-s1-expr-iron-0018`, `AJM1`, `v31-s1-expr-iron-0012`, `J1` | 0.304732, 0.287595, 0.161639, 0.123473, 0.122561 |
| time | `v31-s1-time-0021-0050`, `v31-s1-time-0021-0031`, `v31-s1-time-0021-0039`, V3 CatBoost `0021`, `v31-s1-time-0021-0002` | 0.299188, 0.155347, 0.219946, 0.166819, 0.158701 |

Provisional package comparison on the same two development seeds:

| package | local score |
|---|---:|
| P0 | 96.012388 |
| L0 | 96.114307 |
| S1 provisional fusion | **96.157620** |
| S1 delta vs P0 | **+0.145232** |
| S1 delta vs L0 | **+0.043313** |

These weights and member choices were fitted on the same seeds used for the
comparison, so this is **not** an outer or sample-isolated estimate.  It is a
promising provisional direction, not a promoted package.  S3 must be run before
any platform queue decision.


## S1 nested cross-seed estimate

A leave-one-seed-out LP check was run on the same S1 refined library:
for each held-out seed, member selection and weights were fit on the other seed
only.  This is a cross-seed stability estimate, not sample-level isolation.

| package | nested cross-seed local score |
|---|---:|
| P0 | 96.012388 |
| L0 | 96.114307 |
| S1 provisional fusion | **96.139015** |
| S1 delta vs P0 | **+0.126626** |
| S1 delta vs L0 | **+0.024707** |

The S1 fusion therefore remains a possible B-role candidate, but its
cross-seed gain over L0 is in the 0.02–0.03 gray band, below the 0.03
prioritization line.  The in-sample `+0.043313` over L0 should not be used as
the promotion number.  S3 sample/group-isolated outer validation is still
required before any queue decision.


## S3 sample/duplicate-group-isolated outer validation

S3 was run with:

- outer split seed `7777`, 5 sample/duplicate-group-isolated folds;
- inner split seed `3333 + outer_fold`, 3-fold inner OOF within each outer
  training part;
- member selection and LP weight fitting only on the inner OOF of the outer
  training part;
- base models retrained on the full outer training part after weights were
  frozen, then used to predict the untouched outer validation fold;
- P0 and L0 replayed under the same protocol.

The candidate member pool was frozen from the S1 fusion:

- iron: L0, `v31-s1-expr-iron-0018`, `AJM1`, `v31-s1-expr-iron-0012`, `J1`;
- time: `v31-s1-time-0021-0050`, `0031`, `0039`, V3 CatBoost `0021`,
  `v31-s1-time-0021-0002`.

The LP weights were refit inside each outer training part; per-fold package
scores were:

| outer fold | candidate | L0 | P0 |
|---|---:|---:|---:|
| 0 | 96.122888 | 96.042605 | 95.985653 |
| 1 | 96.138136 | 96.074684 | 95.975282 |
| 2 | 96.223183 | 96.196907 | 96.058543 |
| 3 | 96.181852 | 96.155395 | 96.065231 |
| 4 | 96.154120 | 96.120953 | 95.998567 |

Aggregate results:

| package | local score |
|---|---:|
| P0 | 96.016655 |
| L0 | 96.118109 |
| S3 candidate | **96.164036** |
| S3 candidate vs P0 | **+0.147381** |
| S3 candidate vs L0 | **+0.045927** |

The candidate target WMAPE over outer validation was:

- iron: `0.03819727`
- time: `0.03852201`

Under the task-book queue rules, the confirmed candidate relative to L0 is
above the `+0.03` prioritization line and is therefore preferred over L0 for
the next queue slot.  This remains conditional local validation: the member pool
was selected during prior development, so S3 reduces but does not erase all
selection bias.  It is not a platform score forecast.

No package was generated in this step.


## Prepared package set (not a delivery decision)

Five V3.1-based ZIPs were generated locally for preparation only under:

`local/runs/round2-v3.1-directed-search/prepared-packages-r1/`

They are **not** the next delivery object, have **no independent cold-release audit**,
and have `platform_score = null`.

| package | role | result SHA-256 prefix | ZIP SHA-256 prefix |
|---|---|---|---|
| `01_v31_s3_s1_full` | primary S3-confirmed S1 fusion | `4f57fa9fa224` | `1a26439a557d` |
| `02_l0_v3_frozen` | V3 frozen L0 control/fallback | `9d74136a7866` | `f11ce68ab4d7` |
| `03_s1_iron_l0_time` | S1 iron upgrade with L0 time | `10c3f9ad3524` | `bec5c0393a80` |
| `04_l0_iron_s1_time` | L0 iron with S1 time upgrade | `527f5b0e774e` | `560d3c1da9ff` |
| `05_aj3_iron_s1_time` | AJ3/P0 iron with S1 time upgrade | `f433b40551db` | `654afd218df9` |

For the S1 portion, the package weights are the arithmetic mean of the five S3
outer-fold LP weights, renormalized.  L0 and AJ3/P0 portions use their fixed
frozen formulas.  All five result files passed V2 ID/order/nonnegativity checks
and ZIP round-trip checks.  No desktop write and no platform upload occurred.
