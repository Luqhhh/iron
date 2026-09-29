# V45: rotation forests add no increment to the incumbent

The 96.4 / 96.45 / 96.5 platform milestones remain unmet. Current best is
user-reported **V32_TIME_A60V7_50 = 96.3727**; gaps .0273/.0773/.1273.
No platform result or submission was produced. Protocol e1de45f, implementation
a2106ec, G0 admission 1965937, official-start record 3d805f3.

## G1: every calibration blend rejects the new member

All **40 candidate units / 80 forest fits / 20480 component CART fits**
completed. Both targets, matched AXIS and ROTATE arms, complete seeds 42/3407
and all five folds were retained. Twenty original matching reference units
were hash-verified and reused; no new baseline fit.

Every one of the **40 calibration-selected blend weights is 0**. Thus every
outer candidate blend equals the incumbent, with exactly zero score gain.
This is rejection of the new member, not equality of its standalone accuracy.

| Target | Arm | Standalone WMAPE, seed 42 | Seed 3407 | Mean package gain |
|---|---|---:|---:|---:|
| Iron | AXIS control | .057693 | .058031 | 0 |
| Iron | ROTATE candidate | .056732 | .056517 | 0 |
| Time | AXIS control | .068311 | .068077 | 0 |
| Time | ROTATE candidate | .072444 | .072667 | 0 |

Current-reference WMAPE is .037152/.037204 for iron and .038127/.038169 for
time. Rotation modestly improves standalone iron over AXIS, but worsens time;
both arms remain far less accurate than the incumbent. The mechanism's
synthetic advantage does not establish a useful increment on the real task.

No positive complete development seed, no formal/exploration shortlist, and
no finalist. The conditional confirmation command returns
`no_development_finalist`, **0 confirmation fits**. Derived seeds 271828 and
314159 remain unconsumed. Do not rescue this frozen round with a different
group size, leaf estimator, tree count, hyperparameter grid or weight scan.
This result concerns the declared bagged rotation variant and does not prove
that every possible oblique-tree design fails.

## G0: complete saved-model audit passes

The locked Python 3.12 full suite passed **1256 tests**, with 23 existing
warnings, before fitting. Full-shape synthetic and resource admission passed
as recorded in [G0 results](G0_RESULTS.md).

The independently invoked auditor passes **60 units / 80 forests** and checks
every saved component tree. It verifies train-only preprocessing and PCA
centers, bootstrap/group/sample identities, complete orthogonal PCA bases and
training-covariance diagonalization, categorical exclusion from rotation,
tree topology, bootstrap node counts and response means, calibration-only
weights, reference endpoint arithmetic, complete-fold scores and selection.
Cold/order/chunk prediction and independent score differences are **0**.
No audit refits. Baseline audit scope is saved identities and endpoint
arithmetic, not an additional cold fit of all baseline components.

Candidate event span: **120.12 seconds**; peak recorded worker RSS:
**359.48 MiB**. Development returned success at 10:31:39 Asia/Shanghai and
the sequential independent audit at 10:33:57. These were completion-triggered
steps, not repeated health checks. All failed-evidence preservation rules
remained in force; no failed fit or restart occurred.

## Monitoring update and disposition

The user interrupted the hourly wait with **“每十分钟检查一次”**. The ensuing
authorized check confirmed both stages had already completed (PID 0,
`active/exited`, success) and stopped the old hourly timer. The original
checker log uses the generic string `completed_awaiting_audit`; its configured
completion report is already the passing `audit.json`. The independent audit
and process exits above are authoritative; no audit remains pending.

Subsequent running jobs use **600-second checks**, with no between-check
polling. There is no active V45 job or idle timer to monitor. The three-stage
optimization goal remains active. Full-data fits, packages, desktop writes,
platform uploads: **0**. No V45 platform package is recommended.

Private root: `local/runs/round2-v45/development-r1`.

| Artifact | SHA-256 |
|---|---|
| manifest.json | `0cef4b2fa7620a44bdfbdedbf0180990e81bbb1363a492e343829a0b1a9a6eab` |
| summary.json | `f45eefd734630c2f397658cad04384598320c8b3fe2c61449557eb0f3a98d393` |
| audit.json | `978c07eeb8f02d08faaa3b8976709a4f36e26f1299f9c0ab55fcbba8d8cc7cd9` |

Private models, predictions, execution summary, disposition and monitoring
logs remain out of Git. Public evidence is committed and pushed separately.
