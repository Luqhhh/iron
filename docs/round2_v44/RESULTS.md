# V44: longer BART chains do not improve the incumbent blend

> Historical execution contract (reading update 2026-10-01): keep all measured failures, counts, frozen gates and source identities below. Subsequent user instructions resumed optimization, set milestones 96.4/96.45/96.5, changed monitoring to 600 seconds and removed future wall-clock budgets. Earlier pause/hourly/time-admission statements are not current instructions, and old failures are not relabeled as passes. See [current rules and status](../INDEX.md).

The platform **96.5** goal remains unmet. The latest registered user-reported
best is **V32_TIME_A60V7_50 = 96.3727**, a gap of **0.1273**. No platform score
was obtained in this experiment. Protocol `d2728e8`, implementation `e590cfc`,
G0 admission `03eeb1d`, official start `43cfe08`.

## G1: no candidate qualifies for confirmation

All 40 candidate outer units / 80 fresh fits completed, with 20 matching
reference units reused. Each model used 200 trees and the frozen 4000-sweep
schedule: burn 2000, retain 100 draws every 20 sweeps. Calibration alone
selected the sparse blend weight; the outer query did not select models,
sampling windows or weights. Deltas below are local whole-package score
points against the matching current reference, with the other target unchanged.

| Target | Arm | Seed 42 | Seed 3407 | Mean | Mean change vs V43 |
|---|---|---:|---:|---:|---:|
| Iron | STUMP control | -0.000380 | -0.001905 | -0.001143 | -0.000840 |
| Iron | BART candidate | -0.002365 | -0.002292 | **-0.002329** | -0.001107 |
| Time | STUMP control | 0 | 0 | 0 | 0 |
| Time | BART candidate | -0.001953 | +0.003390 | **+0.000719** | -0.001676 |

Neither BART target improves both complete development seeds or reaches the
frozen +0.01 mean prerequisite. Iron also loses to its STUMP control. Time
has five positive folds out of ten, descriptive only. The candidate-tier
diagnostic retains time as exploration with `both_splits_improve` and
`minimum_improved_folds` failures; it does not override the frozen round gates
or authorize a platform test. Neither target has a finalist.

Standalone BART WMAPE improves from V43 on all four target/seed combinations:
iron by .000738/.000669 and time by .000465/.000333. Its new standalone WMAPE
is .042113/.042094 for iron and .046244/.046526 for time, still worse than
the current reference (.037152/.037204 and .038127/.038169). Those standalone
improvements do not produce a larger incremental blend gain. Twenty of the
40 calibration weights are zero; the remainder are .05 (12), .10 (4), .20 (4).

## What the chain comparison establishes

This is the one separately frozen length study, not a continuation of V43.
The old-prefix draws are audit-only and never enter V44 predictions. Both
rounds and their original decisions remain unchanged. No further schedule,
prior, kernel or post-hoc weight scan is authorized by these results.

Training-only diagnostics over sweeps 2001–4000 show BART RSS lag-one
correlation .7643–.9442 (median .8568). Mean RSS shifts between the latter
period's two halves range -3.838% to +2.753%, median +.210%; the standardized
half-mean shifts range -1.344 to +1.006 within-chain standard deviations,
median +.096. These are descriptive single-chain dependence and drift
statistics, not proof of convergence. The measured schedule change does not
rescue quality, and the result does not exclude every Bayesian-tree design.

## G0 and execution

The locked Python 3.12 suite passed **1248 tests**, with 23 existing warnings,
before official execution. Both full-shape synthetic fits passed exact old
prefix, cold prediction and resource checks; see [G0 results](G0_RESULTS.md).
The development candidate event span was **1409.95 seconds / 23.50 minutes**,
with peak recorded worker RSS **404.93 MiB**. Training exited successfully.

The final independent audit **passed all 60 completed units, 80 saved models
and 80 original prefixes**. Cold/order/chunk inference and original-prefix
prediction differences are **0**; the maximum independent score discrepancy
is **1.77e-16**. It verifies training-only preprocessing/cuts/priors, sampled
tree support, retained indices, calibration-only weights and independent
score/selection arithmetic. No audit refits were performed. Reference audit
scope is saved fit/query identities and endpoint arithmetic, not an additional
cold reconstruction of every baseline component.

An hourly-only timer was configured. The first manual terminal observation
occurred at 09:39:52 Asia/Shanghai, about four minutes before the advertised
09:43:37 checkpoint. This was a monitoring-cadence deviation, not an hourly
timer event. Future manual checks must follow actual timer records. No
automatic restart or retraining occurred.

The scheduled checker subsequently recorded successful terminal completion
at 09:41:36 Asia/Shanghai (the actual timer time, distinct from the advertised
wall-clock estimate), 40/40 candidates, and stopped its timer. The timer is
inactive; the training service is retained as `active/exited` with PID 0 and
exit code 0, not as a running training process.

Private evidence: `local/runs/round2-v44/development-r1`, including the
append-only `length-diagnostics-r1.json`. The conditional confirmation command
returned `no_development_finalist`, **0 confirmation fits**; seeds 271828 and
314159 remain unconsumed. No full-data fits, packages, desktop writes or uploads.
Private artifacts remain outside Git.

| Artifact | SHA-256 |
|---|---|
| manifest.json | `3a5c1110aff9816f82aa837721e8037bdf3f9a7a2e7b157e60b305dcceda5f3f` |
| summary.json | `468565b59d73f3db9e3c7ea00407eb06dd13df18ea822b2dd1e04d86c93cb13c` |
| audit.json | `d8ebf2bf400d478e5cb4991791e3627dd5469df76e1cf6a5354ccb16176a15dd` |
| length-diagnostics-r1.json | `3418ee293136b71cdc3f0e58a546852d72c09e56b4d885fbaad5ef0284e42533` |

The user requested **“完成本轮后暂停”** during final verification. Finish this
round's audit and public evidence publication, then pause the 96.5 objective.
Do not start another optimization round. Preliminary literature lookup did
not create a new experiment, fit, package or scheduled job.
