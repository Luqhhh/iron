# iron 96.4 — V23: capacity probe on the winning time family

Design date: 2026-09-28. Base commit `a566261`. Status at design time: no V23 fit
has run, no package has been written, no upload has been made.

## 1. Why this probe

V22 closed member search against the strongest incumbent (the V21 time column
`0.40 V36 + 0.25 V7m + 0.35 P-LL` on the V12 iron column): over 110 complete
time columns the best incremental gain is `+0.000158`, and a supplementary
screen of the **whole completed N family** (`time-n-family-r1`, `N-0040..N-0051`)
also gives exactly `0.000000` with weight `0` for every member.

The remaining lever is a **new model**, and the single family with the largest
historical payoff is the `raw_tabm` time line: its `large` member (`N-0048`)
produced the biggest platform gain in the project (`+0.0409` over V36 at
`alpha = 0.20`). The one axis of that family never explored is a capacity
increase beyond `large` (`k=32, n_blocks=3, d_block=512`): the frozen sampler
only ever emitted `small/medium/large`, and V6's capacity work was iron-only.

Hypothesis: a wider/deeper `raw_tabm` time network is less correlated with the
current incumbent and adds a measurable increment. This is a real model change,
not another member of the existing library.

## 2. Frozen probes

Both probes keep the `N-0048` recipe exactly (`target tap_time_len`,
`target_transform train_mean_std`, `loss mse`, `optimizer adam`,
`learning_rate 0.001`, `weight_decay 0`, `batch_size 256`, `min_delta 1e-5`,
`inner_validation_folds 5`, `inner_validation_seed 42`, `n_bins 16`,
`d_embedding 8`, `random_seed 42`) and change only capacity:

| probe | k | n_blocks | d_block | dropout | max_epochs | patience |
|---|---:|---:|---:|---:|---:|---:|
| `P_WIDE` | 32 | 3 | 768 | 0.15 | 120 | 15 |
| `P_DEEP` | 32 | 5 | 512 | 0.15 | 120 | 15 |

They are fitted with the frozen `evaluate_v36_outer_folds` evaluator (outer
training part only; validation labels are never seen). No other capacity, loss,
optimizer or schedule is scanned; this is a two-point probe, not a search.

## 3. Decision rule

The binding statistic is the **incremental nested blend gain on top of the V21
time incumbent**, complete coverage, weight fitted on the other split seed over
the frozen `0..1` grid in `1/40` steps:

* promote only if the mean gain over the two development seeds is
  `>= +0.003` **and both development seeds are positive**;
* `+0.003` is ten times the V22 saturation floor (`+0.000158`) and is the
  smallest increment that could matter next to the V21 gain (`+0.0127`);
* a promoted probe is then confirmed on the derived seeds `7777`/`12011`
  (complete five-fold each) and must satisfy the frozen four-seed gate
  (all four positive, paired seed-level `LCB95 > 0`, development package score
  `>= 96.25`). Only then are a full-data fit, a package and a cold audit made.

Fold-level counts remain descriptive. The V21 time weights used as the incumbent
are frozen (`v36_time 0.40`, `v7_member_time 0.25`, `pll_time 0.35`).

## 4. Budget

| item | value |
|---|---:|
| development outer fits | 20 (2 probes x 2 seeds x 5 folds) |
| confirmation outer fits | <= 10, only if promoted |
| full-data fits | <= 1, only after confirmation |
| packages | <= 1 |
| desktop writes | 0 |
| agent uploads | 0 |

Workers are capped so the widened TabM (`~57M` parameters for `P_WIDE`) fits the
15 GiB host: at most 3 workers, with BLAS/OMP/MKL/NUMEXPR pinned to one thread.
Private outputs under `local/runs/round2-v23`. If both probes fail, the
capacity axis of the time family is closed with complete-coverage evidence and
nothing is packaged.

## 5. Limits

This probe tests capacity, not a new paradigm; the prior is weak but the axis is
genuinely untested for the time target. A positive result is not a forecast of
`96.4`; local magnitude does not forecast the platform. The conditional `B0`
arithmetic is `96.3679` and the V18 two-line ceiling is `96.3992`.
