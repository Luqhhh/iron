# Serial recovery after regularization completion

On 2026-09-29 the user requested the other queue after BASE/EMA/SAM completed.
That workflow exited successfully with no development finalist. Only the
augmentation/representation queue now runs, with its original four workers.

The interrupted Budget8 development-r1 contains 20 complete native BASE units
and ten complete references, but none of the 60 new D-LMIX/R-FIXED/R-LEARNED
units completed. Four partial directories contain start events only. Preserve
all original directories, the original runtime-only failed admission, and the
explicit eight-hour readmission.

scripts/resume_augmentation_queue.py verifies both admission reports, the six
existing synthetic artifacts, the original 425-file source inventory, runtime,
data, reference and cache identities. It creates development-r2 exclusively,
copies the original manifest and complete units verbatim, and fits only the
60 missing units (120 optimizer runs). It never imports partial evidence,
overwrites a directory, retries automatically or repeats synthetic admission.
The frozen source/configuration/test files and gates remain unchanged.

The shared virtual environment normally imports the root checkout. Execution
therefore explicitly sets PYTHONPATH to this worktree's src directory; the
supervisor records the actual imported frozen module. All four numerical
thread variables remain 1. The supervisor is an independent user systemd
service with no restart, append-only private logs, ten-minute observations,
and audit triggered by actual stage completion. Conditional confirmation uses
the original runner and derived seeds only after the unchanged development
gate and independent audit pass. No full-data fits, packages or uploads.

Recovery tests exercise byte preservation, corruption and identity rejection,
partial evidence preservation and exclusive destination creation. Their file
name is outside the frozen test_component_*.py inventory, as are the recovery
script and this document; no previously registered source hash is changed.

Validation before launch: the locked Python 3.12 full suite passed 1318 tests
with 23 existing warnings. The original 425 source hashes, data hashes, runtime,
readmission artifacts and 80-unit frozen inventory were independently verified
with zero fits. The 60 new units exclude the 20 reused BASE controls.

Serial recovery launched 2026-09-29T11:18:13.244791+00:00; implementation b065c41 committed/pushed.
User service iron-component-augmentation-recovery-r2.service, supervisor PID
257154. Frozen worktree module identity confirmed; original manifest
and 20 BASE controls copied verbatim, 60 new units admitted. G1 in progress.
