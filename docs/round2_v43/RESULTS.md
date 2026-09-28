# V43: fixed-chain BART does not pass development eligibility

Goal **platform 96.5** remains unmet. The latest registered user-reported
platform best is still **V32_TIME_A60V7_50 = 96.3727**, gap **0.1273**.
No new platform measurement was obtained. Protocol `4321331`, implementation
`5bd460f`, G0 admission `1f36931`, official start `d50c132`.

## G1: one small, unstable time increment; no finalist

All **40 candidate outer units / 80 fixed-chain fits** completed without a
training failure. Twenty matching reference units were reused. Gains are
whole-package score points versus the frozen current reference, with the
other target unchanged and weights selected only on inner calibration rows.

| Target | Arm | Seed 42 gain | Seed 3407 gain | Mean |
|---|---|---:|---:|---:|
| Iron | STUMP control | -0.000415 | -0.000190 | -0.000303 |
| Iron | BART candidate | -0.002145 | -0.000298 | **-0.001222** |
| Time | STUMP control | 0 | 0 | 0 |
| Time | BART candidate | -0.000383 | +0.005173 | **+0.002395** |

BART time improves the mean but has only one positive complete seed and is
below the frozen +0.01 prerequisite. Its six positive folds out of ten are
descriptive, not an override. `candidate_tiers` retains it as **exploration**
with `both_splits_improve` and `minimum_improved_folds` failures; that diagnostic
label does not authorize confirmation, release or platform testing. Iron loses
on both seeds and is below its control mean. Neither target has a finalist.

BART standalone WMAPE is .04276–.04285 (iron) and .04671–.04686 (time), versus
reference .03715–.03720 and .03813–.03817. It is better than the additive control
but still materially less accurate than the incumbent. Twenty-two of the forty
calibration blend weights are zero; the remaining weights are .05/.10/.20.
No post-hoc weight or prior scan was performed.

## Fixed-chain limitation

Every model retains the original 200 trees, 400 total sweeps and 100 saved
draws. The scalar traces and all saved tree samples remain available. BART
grow acceptance ranges .2181–.2621 and prune acceptance .2403–.2981 across
the 40 selector/refit models; final ensembles have 444–491 leaves in total.
STUMP has 377–393 total leaves. These are working transition diagnostics,
not proof that the posterior has mixed or converged.

A post-fit descriptive trace calculation compares mean training RSS in sweeps
201–300 versus 301–400. BART changes range **-4.764% to +1.958%**, with mean
**-1.212%** across its selector/refit fits. This records finite-chain drift;
it neither establishes a stationary posterior nor permits extra sweeps in
this frozen round. No chain extension, restart, prior change or new fit was
performed. The result applies to this declared predictor, not all Bayesian
tree models.

## G0 and the auditor repair

The locked Python 3.12 suite passed **1246 tests**, 23 existing warnings,
before official execution. Full-shape synthetic learning and resource checks
passed. Actual candidate fit span **153.619 seconds / 2.56 minutes**, peak
recorded worker RSS **356.34 MiB**.

The first auditor had an implementation defect: its per-tree split mask
overwrote the outer fold mask, causing an IndexError (1762 versus 2754 rows).
This affected audit execution, not model training or saved predictions.
`audit-failure-r1.json` and the original auditor are preserved. The separate
`v43_audit_r2.py` changes that local variable name only, plus report provenance;
no model, prediction, score, threshold or original frozen source is changed.

The final **r2 audit passed all 60 units / 80 saved models**. Maximum
cold/order/chunk inference difference is **0**; independent pooled score
discrepancy is **1.15e-16**, with the no-finalist selection verified. It checks
training-only preprocessing and cuts, target ranges and prior calibration,
all chain lengths and retained indices, every saved tree's topology/depth/
training support, sampled-ensemble training RSS, saved noise scales and
calibration-only blend weights. This complete readback also exercises the
repaired mask scope across every seed, fold and target. No audit refits.

## Disposition and evidence

Hourly-only checks observed normal terminal completion and each stopped its
timer. No intermediate progress polling, automatic restarts or extra fits.
The conditional confirmation command returns `no_development_finalist`, with
**0 confirmation fits**; seeds 271828/314159 remain unconsumed. No full-data
fit, package, desktop write or upload. The broader 96.5 goal remains active;
the current fixed-chain candidate does not qualify.

Private root: `local/runs/round2-v43/development-r1`.

- Manifest SHA-256: `8d757b9b608e5b9d72498a245e027d82d7e5313914188edaec80bc885683d475`.
- Summary SHA-256: `456653eac71f4e9559c44fc98787514e0eec906ecdba4b9a6bdefc57564e3a02`.
- `chain-diagnostics-r1.json`: post-fit descriptive traces, not model selection.
- `audit-failure-r1.json`: original checker failure, retained unchanged.
- `audit.json`: complete passing r2 audit; auditor SHA-256
  `e0a2f95d9f2a5fd8bfaee8b8e012fa180fbf75208c9e40ff3baae93371d8446d`.

Models, predictions, local reports and monitoring logs remain outside Git.

## Subsequent training-trace diagnostic (before V44)

A separate zero-fit read of the existing training traces, without outer query
labels, finds BART RSS lag-one correlation .5385–.9065 (median .7613), versus
STUMP .2320–.6382 (median .4247). BART's retained-half RSS mean shift has median
-.6051 within-chain standard deviations. Private append-only evidence:
`trace-dependence-r1.json`. These are descriptive dependence/drift measurements,
not a convergence test, a quality gate change or a V43 continuation. They
motivate the separately frozen V44 length study; V43's decision stays failed.
