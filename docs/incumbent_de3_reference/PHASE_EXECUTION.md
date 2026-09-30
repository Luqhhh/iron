# Frozen incumbent-reference completion

The phase completes only the missing current-incumbent iron references. Its
fixed task pool is `(split, fold, training seed)` with splits 7777/12011,
folds 0–4 and training seeds 104729/130363. There are 20 new native joint
estimators / 40 optimizer starts. No factory reconstruction, development
retraining, epoch sweep, quality selection, full-data fit or package is allowed.

`incumbent_native.py` and `incumbent_reference_identity.py` preserve the RFM
zero-fit four-seed native loader from the PTaRL branch at ba5ce299. It verifies
original development **and confirmation** audits, frozen data/folds, fit/query
identities, source bytes, runtime and component arithmetic. Native J42 is
reused only through its original complete four-seed prediction hashes. No
strong-control substitution or cross-split OOF averaging is permitted.

`incumbent_reference_cache.py` binds the original DE3 manifest, summary and
audit digests. It hashes the complete ten iron unit trees (263 files including
top-level anchors), validates original unit/nested identities and native
settings, and cold-audits all 60 reused development states. The arithmetic mean
must reproduce the recorded DE3 predictions, and reused J42 must reproduce
the native vector exactly. No fit occurs in inspection or audit.

Three historical orchestration/test sources changed after DE3 finished for
the separate E-COMPOSE recovery:
`independent_checkpoints_run.py`, `independent_checkpoints_preflight.py` and
`test_independent_checkpoints.py`. The phase verifies their exact original
Git bytes at `906d31b281e2a208d43f188e63816f1f89b221fb`, records their present
hashes separately, and does not execute those files. Every actual native model
module remains source-identical between the current, original-DE3 and native
reference worktrees. An arbitrary model change cannot use this bridge.

Historical `local/runs` is shared through a symlink. Read-only verification
permits that exact declared private root and prevents nested symlink escapes.
The two original shared `configs/data.local.yaml` / `predict.local.yaml` are
hashed as historical bytes only, with resolved paths recorded; neither is
parsed or executed. Official Round2 inputs retain their strict direct-path
schema. November 2024 baseline targets are not part of this phase.

The source manifest freezes all current Python source/tests, public YAML
configuration, lockfile, package definition and controller. It also binds a
passed full locked Python 3.12 receipt for those exact source bytes/runtime,
native audit, original DE3 evidence, data/row/fold/parent identities, task pool,
budget and resource receipt. Implementation must be committed before freezing.
Every CLI execution/audit rechecks those bindings.

## Resource admission fixed before execution

Admission uses ten recorded complete iron unit traces for this unchanged
native recipe. The largest seconds/update ratio is applied to the worst-case
240 selector + 240 refit epochs, conservatively using nine updates/epoch in
both phases. Twenty estimators on four workers receive a 1.5 time multiplier
and 300 seconds fixed overhead. Projected time must be at most 7200 seconds.
Recorded maximum RSS also receives a 1.5 multiplier and must be at most
1536 MiB per worker. Current available memory must be at least
`4 × 1536 + 1024 = 7168 MiB`, and this is checked again before execution.

Metadata inspection projects 4388.3064 seconds and 1095.0117 MiB/worker.
The first available-memory observation was 7075.55 MiB and failed admission;
that private negative record is retained. These are estimates/observations,
not a successful formal admission or measured new-job runtime. No new probe fit
is needed because the exact native architecture and optimizer already supplied
both the cost witnesses and a bit-identical wrapper test. Memory/time caps are
not relaxed after an admission failure.

## One-shot lifecycle

`prepare` creates a new private directory, records admission, cold-verifies
reuse, then writes the frozen manifest. Failure leaves a failed freeze record.
`scripts/run_incumbent_reference.py` is a one-shot controller: it runs the
execute phase, closes its log and launches a new Python process for audit.
Starts, failures and completions are exclusive writes. No checkpoint resume,
automatic retry or directory overwrite is supported.

Execution uses spawn processes and at most four in-flight estimators. Query
labels are absent from worker arguments. The ledger reserves each estimator
and optimizer before creation and must end with exactly 20/40 completed starts,
zero failures and zero incomplete events. Auditing checks all 40 new saved
states, held-out identities, selected/refit epochs, complete budget receipts,
observed worker RSS and full-batch/order/chunk cold predictions.

The overlay writer assembles all 60 member units on their own splits, independently
recomputes the delivered DE3 endpoint and verifies unchanged parent time. It
writes four private iron vectors, a zero-fit audit and `complete.json` in the
schema required by `ptarl_reference.load_current_reference`. The audit hashes
only completed artifacts; its currently open stdout log is deliberately
excluded. No score is computed on confirmation labels or used for selection.

Monitoring remains every 600 seconds with no between-check polling. A genuine
completion event can trigger the independent completion audit. The output is
an incumbent reference, not a promoted DE3 candidate or a submission.
