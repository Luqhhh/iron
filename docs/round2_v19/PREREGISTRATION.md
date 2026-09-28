# iron 96.4 — V19: complete-coverage iron capacity screen (V6 stage B)

Design date: 2026-09-28. Base commit `2296523`
(`round2-v6-iron-capacity-networks`). Status at design time: no V19 fit has run,
no package has been written, no upload has been made.

## 1. Why this round exists

The single largest platform gain in the project is the **time-column large
raw-TabM** member `v36-s1-N-0048` (`+0.0409` over V36 at weight 0.20). Its
`tap_iron` structural analogue — `raw_tabm|large|mse_adam`, trial
`v6-s1-N-0024` — **was never evaluated on complete coverage.** It was rejected
by the V6 iron-capacity probe, which fitted only **folds 0/1 of seed 42**
(1102 rows) and read a mean nested gain of `-0.0014`.

Two facts recorded in the same V6 round make that rejection unsafe:

1. **V6 §2.2.2, written after the probe:** the folds-0/1 screen is too noisy to
   rank candidates. The validated winner `N-0048` itself has single-fold alpha
   estimates of `0.38 / 0.16 / 0.29 / 0.12 / 0.28` and only converges with
   sample size (10 cells: `0.20/0.21`). "**Any subsequent search round must move
   screening to complete coverage**, not collapse it to two folds to save
   compute."
2. **V6 §3, registered as a pre-registration defect:** the probe was the only
   object the round existed to test, and the gate that stopped it was written
   from the screen that cannot support the conclusion. `stage_a2_unlocked` was
   left `false` as a user decision.

The two candidates have nearly identical marginal signatures on folds 0/1
(`N-0048` time: `rho 0.888`, accuracy ratio `1.060`; `N-0024` iron: `rho 0.896`,
ratio `1.095`), and the one that *was* measured on complete coverage won
`+0.0409`. The iron capacity space therefore deserves the screen the V6 spec
already budgeted and never ran.

## 2. Frozen scope amendment (the only change)

**No threshold, gate or budget in `configs/round2_v6/SPEC.yaml` is changed.**
The amendment is scope only: the V6 `stage_a2` probe gate ("if neither probe
trial is admissible, the new iron capacity space is not screened further") is
**superseded**, on the two recorded grounds above, by the already-frozen
`stage_b` budget. This is recorded here before any fit.

## 3. Frozen candidate set (stage B)

The V6 `non_goals` forbid "seed or loss diversity inside one structural family",
so the six slots are six **structurally distinct** families, each with the
frozen winning training setting `mse_adam`:

| trial | family | capacity |
|---|---|---|
| `v6-s1-N-0024` | `raw_tabm` | large (primary analogue) |
| `v6-s1-N-0008` | `raw_tabm` | medium |
| `v6-s1-N-0016` | `raw_mlp` | large |
| `v6-s1-N-0000` | `raw_mlp` | medium |
| `v6-s1-N-0020` | `ple_mlp` | large |
| `v6-s1-N-0004` | `ple_mlp` | medium |

`ple_tabm` is excluded because the recorded probe `v6-s1-N-0028` has an accuracy
ratio of `3.795` (`WMAPE 0.1385` against a `0.0365` iron baseline) — a broken
configuration, not an untested one.

Budget: `6 trials x 2 split seeds (42, 3407) x 5 folds = 60` outer fits, the
frozen `stage_b.max_fit_slots_per_target`. Fits use the frozen V3.6 evaluator
(`evaluate_v36_outer_folds`); it trains on the outer-training part only and
never reads validation labels. Trials are fitted with `mse_adam` only, so no
loss/seed scan is performed.

## 4. Decision rule

The binding statistic is the **incremental nested blend gain on top of the
incumbent iron column**, measured on complete coverage (all 5 folds of both
split seeds), with the blend weight selected on the *other* split seed:

* `incumbent_iron = V12 iron = 0.5*A35_iron + 0.5*V12_member_iron`;
* `candidate = (1-a)*incumbent_iron + a*member`, `a` from a frozen `0..1` grid
  in `1/40` steps fitted on the other seed;
* the unit is package-score points (`50 * (WMAPE_incumbent - WMAPE_candidate)`),
  because the metric is additive across the two targets;
* fold-level counts are **descriptive only** (per the V5/V6 correction).

A candidate is promoted to four-seed confirmation only if **both development
seeds are positive**. Confirmation reuses the V6 `stage_c` budget (derived seeds
`7777`/`12011`, cached baselines) and must then satisfy the frozen promotion
gate: `min_seeds = 4`, paired seed-level `LCB95 > 0`, all contributing seeds
positive, fold level descriptive. If nothing is promoted, the iron capacity
lever is closed with complete-coverage evidence and no full-data fit, package or
upload follows.

## 5. Budget and boundaries

| item | value |
|---|---:|
| development outer fits | 60 |
| confirmation fits (only if a candidate is promoted) | <= 10 |
| full-data fits | 0 unless confirmation passes (then <= 1) |
| packages | 0 unless confirmation passes (then <= 1) |
| agent uploads | 0 |

Private outputs only (`local/runs/round2-v6-iron-capacity-networks/coverage-r1`
and, if reached, `local/runs/round2-v19`). Labels stay private; the user uploads
and returns scores. This round does not reopen NODE, ODST, TabR, residual
correctors, sample reweighting, dense alpha scans, learned combiners, or the
existing OOF pool; it tests exactly one previously mis-gated structural family.
