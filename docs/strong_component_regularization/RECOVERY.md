# Explicit serial recovery, 2026-09-29

The user requested serial execution, with BASE/EMA/SAM completed first. The
original supervisors and workers disappeared without terminal records. The
cause is unconfirmed. Preserve both interrupted queues and all partial fits.

`scripts/resume_component_regularization.py` creates development-r2 under the
existing private run root. It verifies the original manifest, entire frozen
source inventory, data, runtime, successful synthetic admission and reference
cache. It copies all 37 completed development units and ten reference units
verbatim with their original identities, and records provenance separately.
The four interrupted units remain untouched in development-r1. Only the 23
missing development units and six predeclared initialization diagnostics are
fitted in fresh directories, with the unchanged four-worker recipe. Completed
units are not refitted; partial selector checkpoints are not reused.

The recovered run retains the exact original manifest and frozen training
sources. A fresh-process original audit follows completion. If that audit
fails, its log is retained and a separate corrective audit reconstructs the
selector target array as the native `y[inner != 0]` slice. This repairs the
known NumPy reduction-layout mismatch documented by the parallel worktree;
every other audit check is unchanged. The effective auditor, recovery script
and original auditor hashes are recorded. No tolerance or promotion change,
and no new training, is permitted during the corrective audit.

Only audited development finalists can consume the frozen confirmation seeds.
Confirmation uses confirmation-r2, retaining the original four-seed gates.
The supervisor records observations every 600 seconds and actual terminal
events; it never retries automatically. A user-level systemd service keeps
this queue independent of the chat process. The augmentation/representation
queue is deferred. No full-data fits, package, desktop write or upload.
