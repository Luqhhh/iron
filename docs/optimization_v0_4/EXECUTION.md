# optimization-v0.4 execution

Base: b57d2ef21b35f5c6d57be3a8a05efcff9f834f01. Local branch only.
User authorized execution on 2026-09-09. Old evidence and E16 are retained.

Registration: configs/optimization_v0_4/experiment.yaml and access_scope.yaml.
All November evaluation is POST_HOLDOUT_CONSUMPTION retrospective development.
Global holdout_consumed remains true. The old lifecycle and ledger are unchanged.
The new access ledger binds each read to a frozen manifest and historical ledger digest.

The isolated evaluator rejects future snapshots, evaluation IDs in history and rows
that differ from the official source. Training uses original sample-as-of features.
Crossed inference cannot call fit. R1/R2 are trained from scratch with fixed parameters.
R3 combines complete previous/current raw anchors at 0.5/0.5.
The 18-cell and legacy 14-cell summaries remain separate. No platform score selects arms.

Initial run r1 failed before label access: configuration schema needed integer 1.
Its failure record remains intact. r2 is the first actual execution attempt.

Engineering checks and actual results will be recorded after execution; this file
is not evidence of successful training, acceptance, or release.

Actual early engineering verification: locked Python 3.12 full suite passed 127
checks in 98.20 seconds. Expanded v4 boundary suite subsequently passed 15 checks.
Full-suite rerun after final changes remains required.

r2 was interrupted after 4 completed single-target fits (June E09/E04) when the
OPT-10 saved anchor inventory was established. Its artifacts are retained. r3
validates and reuses June plus OPT-10 August–November/final anchors, and fits July.
Cache validation covers model hashes, parameters, feature schema, semantic contracts,
ordered training sample IDs, full history values, and original feature/training code.
The supplemental cache_process_contract_audit.json verifies operation/burden hashes.
E16 rollback ZIP hash was verified against the registered value.

r3 completed the July anchor (4 target fits) and was interrupted before scoring to
avoid repeating public feature construction at each historical cutoff. r4 reuses
both new anchors and the validated older bundles. A new test verifies that public
feature reuse plus full history-column regeneration is identical to direct history
construction under input reordering. No extra model fit is introduced by this change.
The sensitivity path now explicitly refuses to fit missing release anchors.

Execution commands:

```bash
UV_CACHE_DIR=/tmp/iron-uv-cache uv run --locked --python 3.12 python -m bf_tap.optimization.refresh_factorial --output local/runs/optimization-v0.4-r4
UV_CACHE_DIR=/tmp/iron-uv-cache uv run --locked --python 3.12 pytest -q --junitxml=local/reports/pytest-optimization-v0.4-r2.xml
```

The optional challenger packaging command requires complete development execution
and every registered risk check. It chooses minimum J among fully passing candidates,
uses a label-free cold-process configuration, and does not activate or upload the ZIP.

Final locked Python 3.12 suite at current implementation: **132 passed**, 62.64 seconds.
Report: local/reports/pytest-optimization-v0.4-r2.xml. This is G0 engineering
evidence, not a model-quality acceptance claim.

Completed outcome: see RESULTS.md. OPT-11/12/13 development, sensitivity, week
diagnostics and the sole accepted R2 candidate package are complete. E16 is unchanged.
All original official input/source digests and old ledger were reverified after execution.
