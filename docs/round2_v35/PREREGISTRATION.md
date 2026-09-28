# V35: independent convergence check for the V34 tree recipes

Frozen before V35 official fits, 2026-09-28. Platform goal **96.5**, current
user-reported best B0 **96.3679**. V34 remains a failed, immutable round.

## Why this new experiment is justified

V34's independent supplemental audit measured 25/40 calibration runs hitting
500 iterations, 22/40 selecting iteration >=475, and 23/25 capped runs with
lower mean validation MAE over the last25 iterations than over the previous25.
This is new evidence of a convergence limitation. It does not prove that a
longer fit will help B0 or reach the platform target.

This study asks one question: do the same algorithms add incremental accuracy
after a larger, fixed convergence budget? **Only max_epochs changes from500 to
3000**. Keep learning rate0.05, depth3, leaf minimum20, patience60, stopping
metric/tolerance, initialization, both target/recipe pools, reference, inner
calibration splits, weight grid and every promotion gate unchanged. No capacity,
learning-rate, feature, score-rule or weight search. There is one new cap, not a
cap grid. No further extension or rescue is authorized within V35.

## Execution and prefix requirement

Use the exact committed `bf_tap_r2.v34_crps` implementation. Its prior synthetic
preflight is reused only after verifying its pinned report, source/specification
hashes and the precise one-field training-configuration difference. No new
synthetic fits are needed. Freshly fit all40 outer units /80 training runs on
development seeds42/3407 with all five folds. Do not select only the favorable
V34 target/fold cells. Reuse only the20 verified matching B0 references; no new
development baseline fits. Maximum four candidate workers, one compute thread
each, unchanged locked CPU/Python3.12 environment.

All40 calibration histories must reproduce their entire original V34 histories
exactly as prefixes. The new saved selector and refit models, truncated at the
old selected iterations, must reproduce all80 original prediction vectors
exactly. Prefix failure stops interpretation; never patch old V34 evidence or
silently restart. Ordinary cold/source/data/fit-query/weight/metric audits also
remain mandatory. Count actual tree fits, retained trees, cap hits and failed
units. Store all results append-only under local/runs/round2-v35.

## Candidate decision and confirmation

CRPS_FIXED and CRPS_SCALE are both eligible as in V34. At most one recipe per
target, chosen by mean B0-relative incremental gain with FIXED first on ties.
Require two positive complete development seeds before consuming new seeds.
Confirmation seeds271828/314159 remain unconsumed by prior rounds and are
frozen here. Conditional maximum20 outer candidate fits /40 training runs plus
20 matching B0 refits /640 component fits. Confirmation does not use any outer
query labels for preprocessing, iteration or blend-weight choice.

Final admission retains all four seed gains positive, paired seed LCB95>0,
and development package mean>=96.25. Fold results are descriptive and tier
classification cannot override these gates. Repeated splits reuse the same
labels and are not independent datasets. Local improvement is not a platform
forecast. New fits provide convergence evidence, not a new architecture claim.

No development finalist means stop V35. Full-data fits0, packages0, desktop
writes0, uploads0. The five existing ZIPs and their historical release decisions
are preserved; this round does not spend user platform quota.
