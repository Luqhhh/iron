# Parallel implementation and launch

User authorized parallel experiments on 2026-09-29. The existing regularization
workflow remains in its original worktree with frozen source. This branch adds
D-LMIX, R-FIXED and R-LEARNED through an adapter to the same evaluator and audit.
Four new workers plus four existing workers, one CPU thread each; no new GPU or
dependency installation. Private data, checkpoints and logs remain ignored.

## Engineering validation before admission

Locked Python 3.12 full suite: 1312 passed, one workspace-layout failure, 23
existing warnings. The old V4.2 private-package test correctly rejected a
symlinked local/tmp outside this worktree; local/tmp is now a real private
directory. Focused rerun of both component suites plus that failure:25 passed.
No package containment policy or test assertion was loosened.

Original reference cache and original BASE source identities independently
verified. Cold audit of the completed seed42/fold0 BASE units for both targets
verifies saved selection, refit and exact predictions with zero new fits.
The audit now reconstructs inner targets using the native outer-array Boolean
slice. Rebuilding a DataFrame array changed floating-point reduction order by
2.27e-13 on real joint targets. Exact comparison remains required; no tolerance
was introduced and training/predictions did not change. This correction is in
this isolated auditor; the running original workflow's frozen files remain
unchanged. If its older auditor encounters this difference, preserve its failure
and perform a separately recorded corrective audit rather than rerunning fits.

A setup-only source check initially found missing ignored *.local.yaml files;
verified local configuration copies now permit reference verification. No real
or synthetic fits preceded that repair. No private configuration is published.

## Execution

The supervisor first runs six full-shape synthetic240-epoch fits, cold inference
and resource admission. Only successful G0 admits real-data development. Twenty
native BASE units are reused after the source workflow completes them and all
identities/hashes/native predictions match; no partial unit is accepted.
Then60 new outer units/120 optimizer runs cover three methods, two targets,
two development seeds and five folds. Confirmation is conditional on the frozen
gates. Scheduled observations are600seconds apart; actual terminal events trigger
stage transition and audit. No source edits during this workflow.

G1 remains unmeasured. Current platform reference remains user-reported
96.3727; neither unit tests nor synthetic fit quality predict platform gain.
No full-data fits, packages, desktop writes or uploads.

Supervisor launched 2026-09-29T07:15:51.391090+00:00, implementation47ea6e4. Initial terminal
stream confirms preflight start. Six synthetic fits admitted to the four-worker
queue; their final resource/cold result is pending. Session62002 retains the
supervisor. Sources/config remain frozen.
