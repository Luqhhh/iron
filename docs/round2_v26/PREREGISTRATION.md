# iron 96.4 — V26: target-representation probe on the winning time recipe

Design date: 2026-09-28. Base commit `b2ec1bb`. Status at design time: no V26 fit
has run, no package has been written, no upload has been made.

> **Numbering note (2026-09-28):** this round was pre-registered and closed as
> **V24**; it is renumbered to **V26** together with the V23 -> V25 move so the
> local sequence stays monotonic. Only public labels moved
> (`configs/round2_v26`, `docs/round2_v26`, `src/bf_tap_r2/v26_logtarget.py`); the
> frozen private run directory keeps its historical name
> `local/runs/round2-v24`. See
> [round2_round_numbering.md](../round2_round_numbering.md).

## 1. Why this probe

Every closed route so far changed *members, weights, capacity or blends*. This
probe changes the **target representation** of the winning time recipe, an axis
never applied to the winning neural families:

* the frozen V3 search did sample `log1p` targets, but only for the
  GBDT/expression families (which are in the reproducible library and were
  screened to weight `0` in V22);
* the winning neural families — the V3.6 `raw_tabm` time line (`N-0048`), the V7
  periodic line and the V12 joint line — all use `target_transform:
  train_mean_std`, i.e. raw-unit targets.

The metric is `100 - 50*WMAPE_iron - 50*WMAPE_time`, a **relative** error
metric. Raw-unit MSE weights every row by its absolute error, so it over-weights
high-output taps; a `log`-target MSE weights every row by its *relative* error,
which is the metric's own weighting. The targets are mildly right-skewed
(`tap_time_len` skew `0.408`, range `62-223`), so the two objectives are not
equivalent.

Hypothesis: a log-target fit of the winning time recipe changes the error
structure enough to add a measurable increment over the V21 incumbent.

## 2. Frozen probes

Both probes reuse the frozen `N-0048` recipe (large `raw_tabm` time, `k=32,
n_blocks=3, d_block=512`, mse/adam, lr `1e-3`, `batch_size 256`, `max_epochs
120`, `patience 15`, `n_bins 16`, `d_embedding 8`, seed 42, inner 5-fold) and
change **only** the target passed to the fitter:

| probe | fit target | inverse |
|---|---|---|
| `L_TIME` | `log(y)` | `exp` |
| `S_TIME` | `sqrt(y)` | square, clipped at 0 |

The model's internal `train_mean_std` standardisation is applied to the
transformed target, so this is a representation change, not a loss or
preprocessing change. No capacity, loss, optimizer or schedule is scanned.

## 3. Decision rule

Incremental nested blend gain on top of the V21 time incumbent, complete
coverage, weight fitted on the other split seed over the frozen `0..1` grid in
`1/40` steps:

* promote only if mean gain over the two development seeds `>= +0.003` **and
  both development seeds positive**;
* a promoted probe is confirmed on `7777`/`12011` (complete five-fold) and must
  satisfy the frozen four-seed gate (all four positive, paired seed-level
  `LCB95 > 0`, development package score `>= 96.25`) before a full-data fit,
  package or cold audit.

Fold-level counts are descriptive only. Incumbent weights are frozen
(`v36_time 0.40`, `v7_member_time 0.25`, `pll_time 0.35`).

## 4. Budget

| item | value |
|---|---:|
| development outer fits | 20 (2 probes x 2 seeds x 5 folds) |
| confirmation outer fits | <= 10, only if promoted |
| full-data fits | <= 1, only after confirmation |
| packages | <= 1 |
| desktop writes | 0 |
| agent uploads | 0 |

Three workers, BLAS/OMP/MKL/NUMEXPR pinned to one thread. Private outputs under
`local/runs/round2-v24` (historical name, kept frozen across the V24 -> V26
renumbering). If both probes fail, the target-representation axis is closed and
nothing is packaged.

## 5. Limits

The prior is weak-positive: raw-unit MAE was already tested (V10) and did not
beat raw-unit MSE, so metric alignment alone is not sufficient. This probe is
justified by the representation change, not by the alignment argument alone. A
positive result is not a forecast of `96.4`; local magnitude does not forecast
the platform. The conditional `B0` arithmetic is `96.3679`.
