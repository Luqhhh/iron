# RFM Metric Learning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement, audit and launch the approved bounded RFM experiment, stopping automatically when resource, engineering or quality gates fail.

**Architecture:** Pure float64 numeric kernel primitives feed a partition-aware regressor. A budgeted runner consumes verified read-only reference caches; an independent cold audit controls a one-shot development/conditional-confirmation controller. Private artifacts live in the execution checkout, never in Git.

**Tech Stack:** Existing NumPy/SciPy/Pandas/PyYAML, Python3.12 and pytest; existing neural runtime only for reference verification. No new product dependencies or external weights.

**Spec:** [DESIGN.md](DESIGN.md), explicitly approved by the user on2026-09-29. This implementation plan awaits review and execution-method selection; implementation has not started.

## Global Constraints

- Arms FIXED_KRR and FULL_RFM; only FULL_RFM eligible; both targets independent.
- Laplace bandwidth=training positive-pair median at identity; K+0.01*I; no tuning grid, adaptive jitter or tree partition.
- FULL states0/1/2/3; float64;21 numeric coordinates; categorical metric fixed; AGOP query block128.
- Development40 outer fits/80 procedures/max200 solves/120 updates; confirmation at most the same; failed starts count.
- G0 four procedures/max10 solves/six updates, full-size synthetic seed56001; no retries or synthetic-based tuning.
- Workers<=4; numeric threads1; worker<=1536MiB; RAM>=4 peaks+1024MiB; development projection<=2h. Cap is not ETA.
- Complete seeds42/3407; eligible target confirmation7777/12011. Fixed isolated0.2 blend, native current-reference/B0 identity, no reference refits.
- All source/spec/runtime/data/reference identities frozen before G0; no modification after measured admission.
- Every validated public milestone: scoped private guard, ordinary commit/push to execution branch upstream. No ZIP, desktop write or upload.

## Review Focus

1. A coincident query/center must contribute zero kernel derivative, not NaN (Task1).
2. Unknown spouts must retain the native reserved category, without learning a query vocabulary (Task2).
3. Concurrent duplicate fit/solver starts must fail before consuming a second fit; failed evidence remains present (Task3).
4. A cold audit must reject altered centers, row order or a plausible but wrong score/reference (Task4).
5. Existing output directories or a stale successful audit must not permit phase re-entry (Tasks5/6).

## Execution workspace and publication

At execution time read using-git-worktrees, inspect attached artifacts again and
reuse a suitable isolated checkout or create a managed worktree from the exact
committed plan revision. Use a distinct codex/rfm-metric-learning branch if no
suitable branch exists; configure its matching upstream with an ordinary first
push. Do not change teammate branches or published history.

Set RFM_WORKTREE to that returned checkout and RFM_REFERENCE_ROOT to
/home/clairvoyant/code/iron. Import source from RFM_WORKTREE/src explicitly when
using the original .venv/bin/python, whose editable install otherwise points to
the original checkout. Assert module.__file__ belongs to the execution worktree.
Use the original root only for read-only native caches and their original files;
do not create symlinks that route run writes into historical directories.

All commands below run from RFM_WORKTREE. PY denotes the verified Python3.12
runtime with the execution source import path; UV_PROJECT_ENVIRONMENT for locked
checks is an isolated ignored local/locked-python312 environment. Preserve the
original neural .venv and locks. Python test fixtures are tiny synthetic numerical
examples; every full-size G0 or official model procedure is separately reserved.

Execution order: Tasks1–4; Task5 implementation/tests; Task6 implementation/tests;
Task5 full verification/admission; Task6 launch/monitor; Task7 terminal audit.
This ensures the complete controller is frozen before any measured G0 work.

## Task 1: Kernel mathematics and frozen settings

**Files:** create src/bf_tap_r2/rfm_kernel.py, configs/rfm_metric_learning/SPEC.yaml, tests/test_rfm_kernel.py.

**Interfaces:** laplace_kernel(z,c,centers,center_c,M,h)->ndarray; numeric_gradient(z,c,centers,center_c,M,h,alpha)->ndarray; update_metric(gradients)->ndarray; training_bandwidth(z,c)->float; solve_kernel(K,y,regularizer=0.01)->ndarray. All arrays float64; target coefficients shape(n,).

- [ ] Write test_kernel_and_gradient_oracles: K(x,x)=1; symmetric K; finite-difference numeric gradients agree at r>1e-12 with rtol1e-5/atol1e-6; coincident gradient contribution=0; category changes affect kernel but do not enlarge M.
- [ ] Run `PY -m pytest tests/test_rfm_kernel.py -q`; confirm failure is missing implementation.
- [ ] Implement primitives,128-query blocking, PSD threshold-1e-10 and trace21 shrinkage from DESIGN. Use scipy.linalg.cho_factor/cho_solve without fallback; reject nonfinite inputs and invalid shapes. SPEC encodes all design constants and exact runtime versions read from the execution environment, not guessed versions.
- [ ] Add test_cholesky_residual and test_metric_trace_psd: max abs((K+.01I)@alpha-y)<1e-8; metric trace=21 within1e-10, symmetric, eigenvalues>=.01 within roundoff; zero trace/materially negative eigenvalues rejected. Run the complete kernel file, expect PASS.
- [ ] Guard, commit and push this independently tested milestone.

## Task 2: Partition-aware fitting and model persistence

**Files:** create src/bf_tap_r2/rfm_model.py, tests/test_rfm_model.py.

**Interfaces:** RFMRegressor(arm:str).fit_path(frame:DataFrame,y:ndarray,updates:int,reserve:Callable)->list[SavedRFM]; SavedRFM.predict(frame)->ndarray, save(directory:Path)->dict, load(directory:Path)->SavedRFM; fit_partition(train:DataFrame,target:str,arm:str,reserve:Callable)->tuple[SavedRFM,dict].

- [ ] Write test_state0_matches_fixed, test_inner_isolation_unknown_spout and test_fresh_refit: FULL state0 and FIXED exactly match; inner scaler/vocabulary never see inner validation; unknown spout uses index0; fresh refit starts M=I and recalculates outer bandwidth rather than keeping the selected inner M.
- [ ] Run `PY -m pytest tests/test_rfm_model.py -q`, verify red assertions/import failure.
- [ ] Implement NumericPreprocessor(structure='raw_tabm') reuse, transform_mlp numeric/categorical output promoted to float64, training target mean/std, inner seed42/fold0 selection with earliest exact MAE tie. Only explicit input feature columns are read. Reject invalid y, nonfinite predictions and constant-column errors; preserve native behavior instead of adding imputation.
- [ ] Serialize numeric model arrays as non-pickle NPZ plus JSON preprocessing/identity metadata and hashes. Implement test_saved_cold_roundtrip and test_target_id_not_features: altered query labels/IDs do not change predictions; reordered/chunked results agree<=1e-8. Rerun Task1+2 files, expect PASS.
- [ ] Guard, commit and push model milestone.

## Task 3: Reservations, references and complete-coverage runner

**Files:** create src/bf_tap_r2/rfm_protocol.py, src/bf_tap_r2/rfm_run.py, tests/test_rfm_protocol.py, tests/test_rfm_run.py.

**Interfaces:** reserve_event(root:Path,key:tuple,payload:dict)->Path; phase_tasks(phase:str,eligible_targets:list[str])->list[dict]; verify_reference(reference_root:Path)->tuple[DataFrame,dict,dict,dict,dict]; run_phase(workspace:Path,reference_root:Path,output:Path,preflight:Path,phase:str,development:Path|None)->dict.

- [ ] Write test_budget_and_duplicate_reservations: development tasks=40, procedure cap80, solver cap200, updates cap120; two-target confirmation matches, one-target confirmation halves; duplicate/failed reservation cannot be retried. Concurrent creation of one key yields one success only.
- [ ] Run the two Task3 test files; confirm failure before implementation.
- [ ] Implement atomic exclusive reservation files for outer fit, procedure, solve and update starts; parent enforces aggregate caps and each worker enforces its fixed local schedule. Event keys include phase/target/arm/seed/fold/stage/state. Append completion records; retain started-without-completion failures. Per-unit files avoid shared multiprocess append corruption.
- [ ] Reuse v46_cache.load_reference_cache(reference_root) without editing it. Verify/import identical helper source and all four native cache identities. Freeze actual input IDs/fold hashes; outer query supplied to workers excludes target columns. Every fitted model predicts only its registered query rows. Limit worker pool to4 and cap threads before importing numeric libraries.
- [ ] Test tampered/missing caches, wrong-target cache, missing fold, extra task, output-directory collision and worker failure. Stop scheduling new units after failure; record already-started work without retry. `PY -m pytest tests/test_rfm_protocol.py tests/test_rfm_run.py -q` must PASS; then guard/commit/push.

## Task 4: Independent cold audit and decision calculation

**Files:** create src/bf_tap_r2/rfm_audit.py, tests/test_rfm_audit.py.

**Interfaces:** audit_phase(workspace:Path,reference_root:Path,output:Path,development:Path|None)->dict; score_isolated(y:dict,current:dict,historical:dict,predictions:dict)->dict; decide(records:dict,phase:str)->dict. Runner may call the scorer for progress, but audit reconstructs records from original rows and cold-loaded model outputs, never trusts runner summaries.

- [ ] Write test_independent_endpoint_and_gate_boundaries: fixed0.2 replaces one target only; mean gain exactly.01 is admitted if other criteria pass; one nonpositive seed rejects; wrong raw-vs-blended member rejects; LCB uses sample sd/ddof1 and2.3533634348018264 at four seeds.
- [ ] Verify Task4 tests fail before implementing the audit.
- [ ] Implement fresh-process loading and query-order/chunk checks<=1e-8; reconcile all reservations, optimizer starts=0, solve/update counts, selected state and fresh-refit identity. Check source/runtime/data/reference hashes, entire row coverage and per-target arm pairing.
- [ ] Add tests for altered NPZ centers, plausible wrong predictions, wrong row IDs, altered source, missing completion and modified previous audit. Audit writes a new exclusive report; no report overwrite. Complete model/audit tests must PASS; guard/commit/push.
- [ ] Provide a separate private arithmetic check from saved per-seed predictions using direct WMAPE and t-bound expressions, without importing decide(). Its output must match the public audit before confirmation admission.

## Task 5: Full verification and frozen synthetic admission

**Files:** create src/bf_tap_r2/rfm_preflight.py, tests/test_rfm_preflight.py; private local/rfm-metric-learning/control/ and local/runs/rfm-metric-learning/preflight-r1/.

**Interfaces:** prepare_manifest(workspace:Path,reference_root:Path,output:Path)->dict; run_synthetic_admission(manifest:Path)->dict. Freeze all runner/model/audit/controller files before measured runs; Task6 implementation must therefore be complete before executing the full-size probe.

- [ ] Test resource formula, exact G0 limits4/10/6, forced-three-update synthetic refit, shape-bound rejection, hash drift, and rejection of an existing preflight directory. Run red tests.
- [ ] Implement seed56001 synthetic generation, full training/query bounds2204/551, train-median constant comparison, numerical updates and cold checks. Resource calculation=20*(FULL+FIXED path seconds)/4*1.5+300; worker RSS<=1536MiB; RAM reserve4 peaks+1024MiB. Record total measured path timing including preprocessing/save/cold audit. Strictly stop on failure.
- [ ] After Task6 code/tests exist, run all focused tests and `PY -m pytest -q`. Run locked authoritative path in isolated local environment: `uv sync --locked --extra dev --extra round2 --python 3.12`, then `uv run --no-sync python scripts/check_no_private_artifacts.py` and `uv run --no-sync pytest -q`. Record interpreter/runtime and skip reasons; do not call the neural environment alone authoritative lock reproduction.
- [ ] Commit/push fully tested implementation, pin its revision/runtime/config/controller hashes, reverify references, then execute the **four** approved synthetic procedures only (two per arm). Count max10 solves/six updates. No optimizer starts, no official fits. Confirm manifest uses the approved design's exact4 procedures; this written count controls.
- [ ] Independently review the admission record; if any bound fails, retain evidence and stop without starting development or revising the frozen source.

## Task 6: One-shot controller and quiet monitoring

**Files:** create scripts/run_rfm_frozen.py, scripts/monitor_rfm_once.py, tests/test_rfm_controller.py; private control logs and independent arithmetic script under local/rfm-metric-learning/control/.

**Interfaces:** controller CLI accepts --workspace, --reference-root and --preflight; monitor CLI accepts --control-root and returns structured phase/completion/failure counts without fitting or modifying predictions.

- [ ] Write test_controller_state_machine with mocked subprocess outcomes: development -> cold audit -> independent arithmetic -> eligible target confirmation -> final audit; empty finalist set completes with zero confirmation; each nonzero exit stops. Re-entry is rejected even if an older success file exists.
- [ ] Run red controller tests; implement exclusive orchestration-started and phase-started reservations, log files opened with mode x, source-hash verification and exact argument lists. Controller training calls resolve execution-worktree source. Monitor is read-only.
- [ ] Verify no poll loop, model download, package generation or platform call exists; test terminal/unchanged monitor states. Run `PY -m pytest tests/test_rfm_controller.py -q`, expect PASS. Finish before Task5 measured probe so controller bytes can be frozen.
- [ ] After passed Task5 admission, publish that evidence and launch exactly one detached controller with hidden Windows wrapper if needed. Record PID, command/import path, phase reservation and first genuine started event; only then report training started.
- [ ] Create a thread heartbeat through automation_update every30minutes, preserving quiet normal-state behavior. Pull current execution branch --ff-only at wake-up, run monitor once, notify only failure/stop/completion/action. No additional continuous polling.

## Task 7: Terminal evidence and next-family handoff

**Files:** create docs/rfm_metric_learning/G0_RESULTS.md, RESULTS.md; update EVIDENCE_STATUS.json in its existing style only after source audit and exact-scope review.

- [ ] On terminal event read full manifests/ledgers/cold audits and independent decisions. Reconcile G0, development and conditional confirmation budgets independently; never rerun an incomplete fit.
- [ ] Write G0 engineering separately from G1 quality, both target-specific current/B0 deltas, mechanism contrast, failures and eligibility. No model package or forecasted platform score.
- [ ] Run private-artifact guard, scoped diff check and precise staged-path check, then commit/push and verify remote HEAD. Do not rerun full fits/tests for documentation-only results.
- [ ] Pause the heartbeat after final audit/publication. Proceed to the next research family's concrete specification in the user-approved priority order; this plan does not allocate any fits to PTaRL, DNNR or DANet.

## Self-review and execution handoff

Coverage: kernel/math(Task1), preprocessing/selection/refit(Task2), complete
coverage/cache/budget(Task3), independent decisions/cold audit(Task4), resource
and locked tests(Task5), no-retry controller/monitor(Task6), publication(Task7).
Five Review Focus cases each have named tests. No training has been started.

Recommended execution: native implementation in this conversation, with an
independent whole-branch review before source freeze. Closely coupled numeric,
ledger and audit interfaces favor one implementer. Alternative: subagent-driven
task implementation/review. The user reviews this plan and selects the method
before implementation; preserve the approved algorithm and budget either way.
