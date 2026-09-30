# V37: faster contraction, time admission still fails

> Historical execution contract (reading update 2026-10-01): keep all measured failures, counts, frozen gates and source identities below. Subsequent user instructions resumed optimization, set milestones 96.4/96.45/96.5, changed monitoring to 600 seconds and removed future wall-clock budgets. Earlier pause/hourly/time-admission statements are not current instructions, and old failures are not relabeled as passes. See [current rules and status](../INDEX.md).

Platform goal **96.5** remains unachieved. Current user-reported best is
**B0 96.3679**, gap **0.1321**. Frozen implementation: `28ba272`.

## G0: checked equivalence and resource result

Both arms retain the full 1024-tree/depth-five model, 1,271,808 parameters
at the 24-column probe shape, all original initialization and optimizer
settings. Leaf outputs and INSTANCE gate logits are contracted bottom-up
before tree-weight normalization; no model reduction was made.

All eight full-shape train/eval, zero/nonzero-gate comparisons pass:
maximum output difference **5.82e-9**, all input/parameter gradient differences
**<=1.31e-10**, under atol 1e-6 / rtol 1e-5. Floating-point accumulation order
changes, so this is numerical equivalence within tolerance, not bit identity.
Additional tests cover noninitial parameters and arbitrary output cotangents.
Every tree still routes to one leaf.

| Arm | Training-step p95 seconds | Evaluation p95 seconds | Peak RSS MiB |
|---|---:|---:|---:|
| GLOBAL | 0.294442 | 0.095461 | 897.738 |
| INSTANCE | 0.439754 | 0.133498 | 935.250 |

Same one-thread, batch-256 synthetic probes: three warm-up and 20 measured
optimizer steps, 20 evaluation forwards. Two exclusive start records and two
successful results; no retries. Same conservative projection:
`10*1.5*250*(16*max_training_p95+2*max_evaluation_p95)`
= **27386.443 seconds / 7.6073 hours**, versus the unchanged **6-hour** limit.
The prior V36 projection was 9.9316 hours; V27 was 50.4701 hours. These are
recorded-runtime estimates, not controlled hardware speedup measurements or
elapsed official-training time. RSS and available-RAM checks pass.

INSTANCE performs two separate bottom-up contractions and is now the slower
arm. Sharing routing for both outputs is a concrete implementation hypothesis
for a future separately frozen test, not a verified optimization. V37 does not
drop INSTANCE, alter the model, loosen a threshold or repeat resource probes.

The independent audit verified source/spec/license/start identities, every
p95 timing, all eight equivalence records and independently recomputed refusal.
Locked Python 3.12 full suite: **1224 passed**, 23 existing warnings. The audit
also executed successfully against the actual private records. Resource
coverage is core/optimizer/synthetic tensors only; preprocessing, persistence
and full official runner overhead remain unmeasured.

## G1 and budget

**G1 not evaluated.** Resource optimizer runs 2; synthetic learnability,
development, confirmation and full-data fits 0; packages, desktop writes and
uploads 0. Existing packages and current best are unchanged. This result
supplies no new platform candidate and does not reject the model family's
quality. No platform-score forecast is inferred.

Private artifacts: `local/runs/round2-v37/preflight-r1`, outside Git. Audit SHA-256:
`770f7f365bea4d5035fefcf980f78891d8081779e501a121aed40668afc00c4f`. Its evidence map binds every original probe/start record. The next
unreserved local round is V38, subject to a new all-branch reservation scan.
