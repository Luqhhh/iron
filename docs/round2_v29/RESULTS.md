# V29 results: the last derivable-input avenues are closed

Two zero-fit diagnostics against the strongest incumbent (the V21 time column on
the V12 iron column, platform `96.3679`). Spec `configs/round2_v29/SPEC.yaml`,
implementation `src/bf_tap_r2/v29_information_screen.py`, private evidence
`local/runs/round2-v29/information-screen-r1.json`.

## 1. Derived-feature screen: nothing left in the residual

Every pairwise **product and ratio** of the 21 frozen features was built
(630 candidates) and correlated with the incumbent residual (mean over the two
split seeds, `n = 2754`). The null scale for the maximum absolute correlation is
`3/sqrt(n) = 0.0572`.

| target | max absolute correlation | candidates above the null scale |
|---|---:|---:|
| `tap_time_len` | **0.0286** (`furnace_throat_temp/humidity`) | **0** |
| `tap_iron` | **0.0333** (`air_volume*pig`) | **0** |

Not one of the 630 derived features reaches even the null scale, let alone a
predictive level. So the incumbent's residual contains **no pairwise interaction
or ratio structure** that a new model could pick up from these inputs. This is
consistent with, and sharper than, the V4.5 residual-direction result (mean sign
AUC `0.5092`/`0.5070`) and with the V3 expression-family screen.

## 2. Covariate-shift screen: nothing to correct

The 322-row test set and the 2754-row training frame agree on every feature:

* maximum absolute standardised mean difference **0.0915**
  (`upper_press_diff`: train `51.906`, test `51.575`, pooled sd `3.611`);
* the two distributions overlap so closely that transductive normalisation,
  pooled standardisation or test-time feature calibration have **nothing to
  correct**;
* the spout composition matches (train `1386/1368`, test `164/158`), and no
  feature shows a level shift that would justify re-centring.

## 3. What this closes, and what it implies

With V29, every axis that is **derivable from the given data and the current
model family** is now closed:

| axis | round | evidence |
|---|---|---|
| members / blends / stacking | V22 (+V6 §7) | whole reproducible library, best `+0.000158` |
| capacity (iron) | V19 | six families, all weights `0.0` |
| capacity (time) | V25 | two probes, all weights `0.0` |
| target representation | V26 | log/sqrt, all weights `0.0` |
| iterated-model features | V27 (teammate) | six sparse-GP recipes, all non-positive |
| **derived features (interactions/ratios)** | **V29** | **0 of 630 above null** |
| **transductive/covariate correction** | **V29** | **no shift (max `0.09` sd)** |
| weight re-selection on measured lines | V18/V31 | concave ceiling **96.3992** |

**The consequence is sharp.** The measured weight family `{V36, N, V7m}` time ×
`{V36, V12m}` iron has a concavity **ceiling of 96.3992**, so no choice of the
released weights — including the pending V31 axis endpoints — can reach `96.4`.
A new measured *direction* is mandatory, and V29 shows that direction cannot come
from pairwise feature engineering or from correcting a covariate shift.

## 4. Budget

| item | value |
|---|---:|
| new fits | 0 |
| packages | 0 |
| desktop writes | 0 |
| agent uploads | 0 |

Targeted tests: 3 in `tests/test_round2_v29_information_screen.py`.

**96.4 remains unmet** (registered best `96.3679`). The remaining candidates are
(a) platform feedback on the pending V31 axis endpoints — useful to price the
curves, but bounded by `96.3992` — and (b) a genuinely different model class
whose inputs or inductive bias are **not** a function of the 21 frozen features
and their pairwise combinations.

## Interpretation correction (2026-09-28)

The measured low marginal correlations are a negative linear screening result,
not a proof that every nonlinear combination carries no conditional information.
Small marginal standardized mean differences and similar spout counts likewise
do not exclude joint covariate shift. The measured numbers and stopped recipes
remain unchanged. The later V32 bound covers only beyond-chord regions, not all
mixtures. These limitations do not themselves establish a useful new candidate.
