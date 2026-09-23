# Round2 V3 Local Search Plan

## Starting point correction

The latest repository record, superseding the older 96.0982 C2+B3 snapshot, is:

- current platform preferred package: `AJ3_IRON`;
- user-reported platform score: **96.1259**;
- corresponding local reference score: **96.0123883047359**;
- gap to the 96.2 sprint target: **+0.0741** on the platform scale.

The V3 phase therefore uses AJ3 iron + unchanged B3/full-time package behavior as
the current platform reference, and the AJ3 local complete-package score as the
local reference.

## Search shape

The first batch is a 400-item `(configuration, target)` schedule:

| family | items | intent |
|---|---:|---|
| CatBoost | 240 | main family; depth/loss/regularization/tree/bootstrap joint search |
| LightGBM | 40 | tuned alternative and fusion complement |
| XGBoost | 40 | independent tree family; blocked locally when the dependency is absent |
| MLP | 20 | low-budget structural alternative |
| kernel | 20 | low-budget structural alternative |
| expression | 40 | feature-set and target-transform variants on a tree backbone |

The sampler is deterministic and label-free.  It draws only legal conditional
parameter combinations, for example:

- `Ordered` boosting is not paired with non-symmetric trees;
- `MVS` and `Bernoulli` carry `subsample`;
- `Bayesian` carries `bagging_temperature`;
- `No` bootstrap does not carry sampling parameters;
- depthwise/lossguide trials carry `min_data_in_leaf`, while symmetric trees do
  not.

## Three-stage execution

1. **Coarse**: seed 42, frozen folds 0–1, inner-split early stopping only.
2. **Refine**: seeds 42 and 3407, full 5-fold OOF, original-unit WMAPE.
3. **Confirm**: add seed 2026 and outer validation before any package.

Fusion is a separate first-class stage.  The allowed form is:

$$
\hat y = \sum_{m=1}^{M} w_m \hat y_m, \quad w_m \ge 0, \quad \sum_m w_m = 1,
$$

with 2–5 members.  The nested leave-one-seed-out scaffold ranks candidates and
fits weights on the fit seed, then scores the held-out seed.  Existing V2 OOF
predictions are reused as a fast candidate library; new CatBoost/refined
predictions are appended.  The initial thresholds remain:

- `< 0.02`: no independent submission;
- `0.02–0.05`: backup/fallback;
- `>= 0.05` and confirm-stable: candidate pool;
- `0.15–0.20`: primary sprint candidate.

No V3 step uploads a package.  Platform uploads remain user-only.
