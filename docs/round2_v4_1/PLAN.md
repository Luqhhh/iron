# Round2 V4.1 strong-base increment implementation plan

Date: 2026-09-25  
Branch: `round2-v4.1-strong-increment`  
Status: **implemented; first-round and iron full follow-up executed descriptively; no promotion, packaging, or upload**

> Execution update: see `docs/round2_v4_1/STRONG_INCREMENT_RESULTS.md`.  The
> original V3.6 package remains untouched and no submission was produced.

## 1. Scope and guardrails

This plan implements the 2026-09-25 V4.1 task book for the first-round
C2/C3/C4 screen only.  It does not rerun C0/C1, does not open bins/learning-rate
/regularisation/cap grids, and does not overwrite the original V3.6 package.

The original user target remains **93.5**.  The user-returned
`V36_USER_REQUESTED_OUTER_FAILED = 96.2734` is recorded but not independently
verified.  V3.6 development scores remain development replays, not platform
scores.

## 2. Reference restoration

`bf_tap_r2.v4_1_reference.V36DevelopmentReference` reconstructs the exact
development replay for seeds 42 and 3407 from:

- the released `v36-summary.json` composition (exact unrounded weights and
  selected experts);
- the frozen `fixed-r2-final/fit_ledger.jsonl` trial specifications;
- the already-recorded V34_A development OOF vector;
- the four recorded V3.6 expert OOF `.npy` vectors in `complete-dev-r2-final`.

The reconstructed per-seed package score must equal the manifest score before
any candidate run is allowed.  Reconstructed B36 vectors are labelled
`DESCRIPTIVE_FROZEN_V36_DEVELOPMENT_REPLAY_NOT_NEW_OUTER`.

For an unseen outer split, `V36FixedRecipeFactory` is available as a fallback.
It refits A-frozen and the selected V3.6 experts on the supplied training part;
it reuses the released top-level weights and never reads query labels.

## 3. Fixed-slot replacement contract

For target `t` and its fixed slot `j`:

```text
candidate_t = B36_t + w_j * (replacement_t - parent_j_t)
```

All three inputs are in original target units.  The weights are read from
`v36-summary.json`.

| target | slot | weight source |
|---|---|---|
| `tap_iron` | `v36-s1-D-0029` | exact manifest weight |
| `tap_time_len` | `v36-s1-O-0057` | exact manifest weight |

Only the selected slot is replaced.  The other target and all other members
and weights remain B36.

## 4. Model implementation

`src/bf_tap_r2/v4_1_models.py` introduces an independent
`V41EBMRegressor`.  It reuses the frozen target transform, feature-frame
constructor, group-safe bag protocol, and V3.6 effective-parameter audit but
accepts explicit pair *and* triple terms.  V3.5/V3.6 classes are not modified.

`src/bf_tap_r2/v4_1_c234.py` implements:

- C2: exact V36 parent fit on each outer training part, checked against the
  frozen parent replay;
- C3: explicit replay of the pair set fitted by the current training-part C2;
- C4: group-safe internal-training selection of `k in {0,1,2}` triples, with
  pair terms unchanged and triples generated only from shared-feature unions of
  parent pair edges ranked by parent pair importance.

The outer validation fold is never passed to pair extraction, triple
generation, or `k` selection.

## 5. Budget

First round:

- 2 targets × C2/C3/C4 = 6 target recipes;
- seeds 42 and 3407;
- outer folds 0 and 1;
- 24 outer evaluations.

Internal C4 parent/candidate fits are recorded separately.  Completed controls
may be reused, but no synthetic fit count is reported.  Full five-fold
development is only started if a C3/C4 fixed-slot package delta is positive in
both seeds.

## 6. Reporting

The runner writes only under `local/runs/`.  It records:

- fold-level and seed-pooled metrics using `sum(abs(y-p)) / sum(abs(y))`;
- single-target replacement WMAPE;
- C3-C2 and C4-C3 deltas;
- full-package fixed-slot replacement delta;
- actual outer and internal fit counts;
- source hashes and the frozen-reference identity.

Package score is `S = 100 - 50 * (W_iron + W_time)`.  Seeds are averaged only
after each seed's covered rows are pooled.

## 7. Deliverables

Implemented files:

- `src/bf_tap_r2/v4_1_models.py`
- `src/bf_tap_r2/v4_1_c234.py`
- `src/bf_tap_r2/v4_1_reference.py`
- `src/bf_tap_r2/v4_1_run.py`
- `tests/test_round2_v4_1.py`
- `configs/round2_v4_1.yaml`
- `docs/round2_v4_1/PLAN.md`

No candidate package, submission ZIP, or platform upload is produced by this
plan.  Real training results remain explicitly reported in the delivery summary.
