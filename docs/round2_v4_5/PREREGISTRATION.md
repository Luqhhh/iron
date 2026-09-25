# Round2 V4.5 pre-registration: does the strong base's residual carry direction?

Date: 2026-09-25
Branch: `round2-v4.2-n2-seed-repair` (base `d774195`)
Status: **PRE-REGISTERED — written before any evaluation was run**

This is the third item of the user's 2026-09-25 plan. Priorities 1 and 2 (NODE
per-depth selection, TabR full fusion) have been executed and closed negative
(`docs/round2_v4_4/RESULTS.md`). The user's section 5 stated that this diagnostic
decides "whether the next phase continues to develop models, or changes the
training and composition approach". It is therefore run before any further
residual-corrector work.

## 1. The question, stated so it cannot be answered by error magnitude

With OOF residual `r = y - y_hat`, WMAPE's denominator is fixed on a given
evaluation set, so each target is an absolute-error problem. For a correction
`delta`,

    L(delta | x) = E[ |r - delta| | x ]  and  dL/ddelta = 2 F_{r|x}(delta) - 1,

so the L1-optimal correction is the **conditional median** of `r` given `x`.
Two questions, deliberately separated:

* **Q1 (direction)** — can `P(r > 0 | x)` be predicted better than a calibrated
  constant? Knowing that a region has large errors is *not* the same as knowing
  which way to move them.
* **Q2 (utility)** — does a fitted `delta(x)` reduce absolute error on rows it
  has never seen?

A negative Q1 closes the residual-correction route regardless of how predictable
the error *magnitude* turns out to be.

## 2. Data, and why no new base fits are needed

Strong base: the frozen V3.6 composition refit per outer fold (`B_fit`), i.e. the
same formal reference the V4.1–V4.4 screens used. Its out-of-fold predictions
already exist for both targets and all ten cells:

* source: `local/runs/round2-v4.4-mechanism-completion/N2-tap_iron/baselines.npz`
  (two independent runs, `.../n2-seed-init-v2/baselines.npz` and
  `.../full-r1/baselines.npz`, hold the same vectors);
* layout: `s{seed}-f{fold}-pred` is a `(2754, 2)` float64 array, columns ordered
  `("tap_iron", "tap_time_len")`, `NaN` outside that cell's validation rows;
* verified before pre-registration: for each of the 2 seeds and each of the 2
  targets, **all 2754 rows are non-NaN in exactly one fold**, with fold sizes
  `[551, 551, 551, 551, 550]`.

So every row has an honest OOF prediction from a base that never saw it. No
baseline is refit and no candidate model is trained for this diagnostic.

## 3. Protocol (frozen)

* **Outer loop**: for each split seed in `{42, 3407}` and each held-out fold
  `f` in `{0,1,2,3,4}`, train the corrector on the other four folds of the *same*
  seed and evaluate on fold `f` of that seed. Folds partition the rows, so no row
  is ever in both the corrector's training and evaluation sets.
* **Group safety**: the fold vectors come from the repository's own splitter,
  which is group-aware over `exclusion_group_keys`. The corrector's train and
  evaluation sets are unions of whole folds, so group separation is inherited.
  Asserted, not assumed: no group key may appear on both sides.
* **Feature sets** (both pre-registered, primary reported first):
  * `own`: the 21 numeric features + `spout_no` indicator + base prediction state
    `y_hat` and `|y_hat|`;
  * `plus_other`: `own` plus the *other* target's base prediction (legal at
    prediction time, since the package always contains both columns).
* **Scaling**: `StandardScaler` fitted on the corrector's training folds only.

## 4. Candidates compared on every held-out fold

| id | correction | role |
|---|---|---|
| `delta0` | `0` | incumbent, no correction |
| `delta_const` | median of the training folds' residuals | calibrated constant baseline |
| `delta_sign` | `sign(p_hat - 0.5) * median(|r|)` from the logistic model | direction only, no magnitude model |
| `delta_median` | fitted conditional median (quantile regression, `alpha=0.5`) | the L1-optimal target of Q2 |

Sign model for Q1: L2 logistic regression, `max_iter=2000`, `C=1.0`.
Median model for Q2: gradient-boosted quantile regression, `loss="quantile"`,
`alpha=0.5`, `n_estimators=200`, `max_depth=3`, `learning_rate=0.05`,
`random_state=0`. These are fixed here; no hyper-parameter is chosen after
seeing a held-out score.

## 5. Metrics

Per cell (seed, fold, target):

* Q1: sign AUC, accuracy, and the **Brier score against the training-fold base
  rate** (a calibrated constant), plus observed positive rate vs predicted;
* Q2: MAE for each candidate and `delta_MAE` against `delta0`;
* distribution: mean, median, std, sign balance of `r`.

Reported per target and pooled, with the per-cell spread, never only a mean.

## 6. Decision rule (frozen before evaluation)

Let `cells` be the 10 (seed, fold) cells per target and `c(t)` the number of
cells where `delta_median` reduces MAE against `delta0` for target `t`.

* **CLOSE the residual-correction route** if `mean AUC <= 0.52` for both
  targets, or if `c(t) < 8` for either target.
* **ESCALATE** to the training-scale diagnostic (section 7) only if, for at
  least one target: `mean AUC >= 0.55`, `c(t) >= 8`, and the sign of the mean
  `delta_MAE` is the same under both split seeds.
* **WEAK / INCONCLUSIVE** otherwise; recorded as such, not escalated, and no
  residual-corrector budget is spent on it.

No threshold is moved after seeing the result. Magnitude predictability (error
size) is explicitly *not* evidence for the direction route.

## 7. Conditional stage 2 (only if escalated)

Fix the recipe and the outer validation rows, vary only the base training
fraction of `T` (e.g. 25% / 50% / 75% / 100%), and compare the residual
distribution and the sign-predictability across fractions. This tests the
user's concern that a correction fitted on a small-training base may learn
errors that the full-training base no longer makes. It is declared here so it
cannot be presented later as an unplanned rescue.

## 8. Known limitations, stated up front

* The OOF residuals come from bases trained on 4/5 of `T`; the deployed base is
  trained on all of `T`. Section 7 exists precisely to bound this gap.
* Rows within one fold share a base model, so residuals are not independent;
  the per-cell spread is reported instead of a naive standard error.
* The two split seeds share rows, so the two per-seed results are not
  independent replications; agreement between them is a consistency check, not
  a second experiment.
* This diagnostic measures the strong base only. It says nothing about whether a
  *different* model class could have a more predictable residual.
