# Reference completion launched — 2026-09-30

The frozen reference completion started **10:03:25 CST**, as user service
`iron-incumbent-de3-reference-r2.service`. The launch check confirmed
ActiveState=active, SubState=running, MainPID417199. This is a launch
observation, not a terminal result or a current fit count. Maximum budget:
20 new native estimators /40 optimizer starts, at most4 workers.

Implementation commit: `6a3bbc27c5d66c55bdec62688957be2d951a0739`; locked Python3.12 full suite
1358 passed. All727 source/configuration/test hashes remain unchanged at
launch. Frozen manifest SHA-256:
`fba318a03e76c7e2adc31bebaf376ab7b2550178b916cb3ba72a3c8924ca8a8c`.

Private output: `local/incumbent-de3-reference-four-seed-r2`.
Private launch receipt: `local/research/incumbent-reference-phase-r1/launch-receipt.json`
(SHA-256 `48b4c965ae57c2656696ad144df66bc662966eaec6b2e17fc47d045af14dfeed`). All models, vectors, ledgers and detailed receipts
remain outside Git.

Formal resource admission passed with8598.68MiB available, against7168MiB
required; projected4388.31s and1095.01MiB/worker satisfy unchanged bounds.
Before launch, all60 reused development states passed cold audit, with10
exact native J42 replays and no fit call. The native four-split reference
loader verifies436 original files. Its correct reference root is the main
repository `/home/lux1/iron`, as recorded in the original RFM manifest.

The first freeze used the RFM *execution* worktree as its reference root.
That worktree carries a partial private cache and lacked the primary reference
audit, so freezing failed before any estimator was started. Its directory
`local/incumbent-de3-reference-four-seed-r1` and failed record are preserved.
The successful r2 uses the original recorded reference root; no check or cap
was relaxed, no evidence copied over a prior file, and no existing model refit.
The earlier7075.55MiB failed cost/memory observation is also retained.

Monitor every600s, with no between-check polling. First observation at
02:03:25UTC; next observation no earlier than02:13:25UTC. The one-shot
controller runs the complete zero-fit audit in a fresh Python process after
execution, and produces the PTaRL-compatible overlay only if all checks pass.

G0 execution is running; G1 unmeasured. This is incumbent-reference completion
only, not retrospective DE3 promotion. Current platform best remains
user-reported96.3749, with the three milestones unmet. No full-data fits,
packages, desktop writes or uploads are part of this job.
