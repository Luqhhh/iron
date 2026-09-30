# V39: complete hard-tree validation, no eligible finalist

> Historical execution contract (reading update 2026-10-01): keep all measured failures, counts, frozen gates and source identities below. Subsequent user instructions resumed optimization, set milestones 96.4/96.45/96.5, changed monitoring to 600 seconds and removed future wall-clock budgets. Earlier pause/hourly/time-admission statements are not current instructions, and old failures are not relabeled as passes. See [current rules and status](../INDEX.md).

Goal **platform 96.5** remains unmet. Current user-reported best remains
**V32_TIME_A60V7_50 = 96.3727**, gap **0.1273**. No independently verified
platform receipt is claimed. Frozen protocol `657a72d`, implementation
`0a96868`, G0 evidence `e2aa609`, official start `4f5175c`.

## G1: full-coverage quality result

All **40 outer units / 80 candidate optimization runs** completed with no
failures. Both targets, both original arms, both complete development seeds
and five folds were retained. Gains below are isolated-column whole-package
score points against the frozen current reference, with blend weights
selected only inside each outer training partition.

| Target | Arm | Seed 42 gain | Seed 3407 gain | Mean gain |
|---|---|---:|---:|---:|
| tap_iron | GLOBAL | -0.001620 | -0.000759 | -0.001190 |
| tap_iron | INSTANCE | -0.001683 | -0.001834 | -0.001758 |
| tap_time_len | GLOBAL | +0.000792 | -0.001322 | -0.000265 |
| tap_time_len | INSTANCE | -0.003643 | -0.004931 | -0.004287 |

INSTANCE, the only eligible candidate, is negative on both seeds for both
targets. Its mean is also below GLOBAL's on both targets. GLOBAL is a
mechanism control, not retrospectively promotable; its time-column result
has only one positive seed and a negative mean. Thus no arm has two positive
complete seeds, irrespective of the additional +0.01 mean threshold. The
independent selector agrees with the frozen runner: **no finalist**.

Calibration selects zero weight in 16/40 units; the other 24 select weights
from .05 to .50. Nonzero calibration preference does not establish held-out
increment. Candidate-tier labels cannot override the frozen eligibility rules.

Selected epochs span **14–43**; selectors stopped at **64–93**, all through
the fixed patience rule. **Zero** hit the 250-epoch limit. There is no measured
cap-truncation explanation for this result, and no cap extension, arm removal,
retrospective role change, weight scan or gate relaxation is made here.
The conclusion is specific to this frozen hard-tree recipe and protocol,
not impossibility for every tree model or for platform 96.5.

## Current platform reference is not a fixed local-score ordering

Reference component reuse was audited before recomputing the new time weights.
The unchanged iron column plus .20*V36+.30*N0048+.50*V7_periodic gives:

| Split seed | Historical B0 local score | Current reference local score | Current minus B0 |
|---|---:|---:|---:|
| 42 | 96.248751 | 96.236050 | -0.012701 |
| 3407 | 96.246589 | 96.231327 | -0.015263 |

The same released weight change has user-reported **+0.0048** on the platform,
while both local seeds are negative. This is another concrete reason not to
claim a constant local-to-platform offset or universal rank/sign preservation.
These are two-seed reference diagnostics, not a four-seed promotion experiment;
they neither prove nor refute a universal guarantee for the four-seed rule.
The current reference mean is below the unchanged 96.25 local working gate.
No gate is changed here; V39 already fails at the earlier negative-increment
condition. Historical B0 comparisons were saved after evaluation, marked
`descriptive_only`, and did not change selection or consume any new fits.

## G0: reproduction, runtime and hourly monitoring

- Locked Python 3.12 full suite: **1228 passed**, 23 existing warnings.
- Independent audit: **60 units, 80 saved models**, including every calibration
  selector and fresh refit; maximum cold/order/chunk inference difference **0**.
- Independent pooled score discrepancy <= **2.33e-16**; selection verified.
- Train-only quantiles/categories/target scaling, fit/query IDs, contiguous
  epoch histories, selected/refit epoch equality, calibration-only blend
  arithmetic and current reference component arithmetic all passed.
- Twenty audited reference units reused/reweighted; new reference fits **0**.
- Peak recorded worker RSS **1152.19 MiB**, below the original 1536 MiB cap.
- Actual fit phase **1.3371 hours**, below the user-accepted 7.61-hour
  conservative projection because early stopping selected short fits.

The user explicitly accepted “7.61小时可接受”; the historical V37 6-hour
resource refusal remains intact. Monitoring followed the later instruction
“每小时监控一次是否正常运行，不要轮询”: one scheduled observation found
31/40 units complete with four live workers and no failures; the next found
40/40 complete, the supervisor terminated and no failures. The timer stopped
automatically. No intermediate training polling occurred after that instruction.

## Decision and preserved budget

The conditional confirmation command returned `no_development_finalist`.
Confirmation fits **0**; seeds 271828/314159 remain unconsumed. Full-data fits,
packages, desktop writes and uploads **0**. No new platform candidate is
recommended from V39; the 96.5 objective remains active and unverified.

All private checkpoints, predictions, traces, ledgers and monitoring records
remain under the isolated worktree's `local/runs/round2-v39/development-r1`.

Evidence SHA-256:

- Manifest: `ff8920256d3b60876bffb0093eeb4bb17b040e5d85ccca10204b27b83d86b575`.
- Summary: `64c8077209fa15b8f91bb6d47d54e69831c6e6dca8e370b97de0867511e43fe2`.
- Independent audit: `ad3daf91cced60ea6c6e9db382517b9fdbf718716f11e5a6b62bcd4ed4231823`.
- Descriptive historical comparison: `a36e6931a0abc27412ac24594a50762865be4d796560d01963e0c6a119a70086`.
