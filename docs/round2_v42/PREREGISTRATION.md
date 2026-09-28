# V42: adaptive hinge-product splines

Goal remains platform 96.5; current user-reported best 96.3727. V41 restricts
only mixtures of fixed endpoints. This experiment creates new predictions
without reopening the failed neural capacity, robust-loss or residual routes.

## Mechanism and frozen implementation choices

Adaptive regression splines fit products of hinge functions with selected
knots and remove unnecessary terms. Sources:
[Friedman 1991](https://doi.org/10.1214/aos/1176347963) and the
[SAS ADAPTIVEREG description](https://support.sas.com/documentation/onlinedoc/stat/151/adaptivereg.pdf).
We implement a limited deterministic hinge-search variant, not a full MARS
or SAS reproduction: a finite quantile knot grid and inner held-out MAE choose
complexity instead of GCV. The scientific question is whether selected local
linear interactions supply useful incremental predictions over the incumbent.

**ADDITIVE** is the degree-one mechanism control. **PAIR** permits degree-two
products and is the only eligible candidate. Both start with an intercept,
add mirrored hinge pairs by training least-squares reduction, then remove one
nonconstant term at a time by smallest training RSS increase. Retaining a
product does not require retaining its parent term; its factors remain saved.
No feature group repeats within a product; all spout one-hot columns share a
group. Quantile knots are computed only on active training rows of the parent;
each side must contain at least 10 such rows. Categorical knots are 0.5.
Forward search ends at 63 total terms, exhaustion, or relative RSS reduction
<=1e-8. Rank handling uses SVD tolerance 1e-10 and minimum-norm coefficients.
The paired-gain computation is checked against independent least-squares fits.

All numeric inputs use training mean/std; spout uses a training-only one-hot
vocabulary, unseen categories all zero. Targets use training mean/std.
No clipping, IDs as features, external data, pretrained models or temporal
inputs. Select among backward-path counts 1/5/9/17/33/63 that exist, plus the
maximum available count, by calibration MAE then smaller count. Rebuild the
forward/backward path fresh on outer training rows and use the selected count;
if forward exhaustion leaves fewer terms, use its available maximum and record
that fact. This deterministic rule is not a failed-fit fallback.

## Protocol, budget and protection

Keep the V39 current reference, all component recipes, seeds 42/3407, five
folds, inner calibration seed 27001/fold 0 and eight blend weights unchanged.
Reuse the 20 audited V30 component references only after the same original
source/data/fold/fit-ID/query-ID/hash and reweighting checks. Query frames omit
both targets. Each unit performs one selector fit and one fresh refit. Inner
calibration selects complexity and blend weight; outer labels enter only
evaluation. No cross-seed OOF averaging.

Two arms x two targets x two seeds x five folds = **40 outer units / 80 solver
fits**. Synthetic preflight permits two full-shape solver fits. Four workers,
one BLAS thread each, existing locked CPU environment. Project the complete
80-fit cost conservatively using twice the slower synthetic fit / four workers;
admit only <=7.61 hours and <=1536 MiB per worker, with actual available RAM
checked for all workers. No extra resource retry or smaller model on failure.

Eligible PAIR needs both complete development gains positive, mean >=0.01,
and mean above ADDITIVE. At most one finalist per target. Preserve four-seed
positive gains, paired seed LCB95>0, development local score >=96.25 and
descriptive-only fold criteria. Confirmation seeds 271828/314159 remain
unconsumed until audited eligibility, with at most 20 candidate units and 20
matching reference factory calls. Repeated splits reuse labels, not new data.
No thresholds are relaxed. Candidate-tier diagnostics do not override gates.

All failed evidence remains append-only, no automatic retries. Freeze runtime,
source, data, folds, saved basis definitions, preprocessing, selected counts,
weights and predictions. Independent cold audit must check all models, training
identities, count-selection traces and metric/gate arithmetic. Separate G0 from
G1. Commit/push validated public work; keep private evidence out of Git.

New full-data fits, packages, desktop writes, uploads: **zero**. User performs
uploads. Training monitoring is once per hour, not polling. Stop the frozen
round without a finalist if the specified development gate fails; do not scan
more knots, degrees, losses or term counts to rescue it.
