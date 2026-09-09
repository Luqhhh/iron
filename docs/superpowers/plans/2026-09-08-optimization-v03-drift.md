# optimization-v0.3-drift Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现并真实运行 C1/C2 漂移候选和预注册条件式 C3，同时保持 baseline-v0.1 与 v0.2 证据不变。

**Architecture:** 新建 `bf_tap.optimization_v03` 薄编排层，调用 v0.2 五起点引擎生成 E00/E09 核心证据，再按 origin 因果派生 M1/C1/C2/C3。v0.3 负责诊断、bootstrap、Stage 2 门控和顶层 G0/G1。

**Tech Stack:** Python 3.12.13、pandas 2.3.3、NumPy 2.2.6、CatBoost 1.2.8、PyYAML、pytest 9.0.2。

**Spec:** `docs/superpowers/specs/2026-09-08-optimization-v03-drift-design.md`

## Global Constraints

- 不改 baseline-v0.1-reproducible 或 `src/bf_tap/optimization/` 的 v0.2 行为。
- 不读取 reference_time >= 2024-11-01 的目标，不运行保护 holdout。
- C1 固定 30 天且标签在 cutoff 可用；C2 逐行等于核心 E09。
- 不改模型参数、目标、损失、基础窗口、裁剪或 acceptance thresholds。
- C3 仅在 C1/C2 各自通过时生成，固定为 C2 iron + C1 time。
- 输出目录不可覆盖；失败证据保留；G0/G1 分离。
- 使用锁定 Python 3.12 路径作权威验证。
- 直接使用当前 checkout；不建 worktree；不 commit、不 push。

---

### Task 1: 冻结配置与候选派生

**Files:**
- Create: `configs/optimization_v0_3/experiment.yaml`
- Create: `src/bf_tap/optimization_v03/__init__.py`
- Create: `src/bf_tap/optimization_v03/config.py`
- Create: `src/bf_tap/optimization_v03/candidates.py`
- Create: `tests/optimization_v03/test_config.py`
- Create: `tests/optimization_v03/test_candidates.py`

**Interfaces:**
- `load_drift_experiment(path) -> DriftExperiment`
- `recent_target_medians(samples, fit_cutoff, window_days) -> (medians, audit)`
- `derive_stage1_predictions(eval_samples, e00, b1, e09, recent, experiment) -> dict[str, DataFrame]`
- `derive_stage2_prediction(c1, c2, experiment) -> DataFrame`

- [ ] Write config tests that load canonical values and mutate every frozen field to require `ContractError`.
- [ ] Run locked pytest on `test_config.py`; verify RED because the package is absent.
- [ ] Implement a frozen `DriftExperiment` and exact-key validator. Canonical values are ID `optimization-v0.3-drift`, baseline `baseline-v0.1`, core IDs `E00/E09_PROCESS_CHANGE_E02`, 30 days, weights iron=0.5/time=0.0, 2000 bootstrap repetitions, seed 20260908, and the four canonical candidate IDs.
- [ ] Re-run config tests; verify GREEN.
- [ ] Write candidate tests with literal cutoff rows. Require the three temporal predicates, exact medians, exact M1/C1/C2/C3 formulas, one-to-one IDs, duplicate rejection and empty-window rejection.
- [ ] Run candidate tests; verify RED on missing symbols.
- [ ] Implement minimal candidates. Align via one-to-one sample_id merge, output only sample_id/pred_tap_iron/pred_tap_time_len, and preserve the existing lower clip at 0.
- [ ] Run both files; record exact pass count in `progress.md`.

Locked command:

```bash
env UV_PROJECT_ENVIRONMENT=/tmp/iron-optimization-v02-py312 uv run --frozen --extra dev --python /home/clairvoyant/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12 pytest -q tests/optimization_v03
```

---

### Task 2: 切片诊断与日块 bootstrap

**Files:**
- Create: `src/bf_tap/optimization_v03/diagnostics.py`
- Create: `tests/optimization_v03/test_diagnostics.py`

**Interfaces:**
- `diagnostic_metrics(actual, predicted) -> dict`
- `scenario_summary(rows) -> dict`
- `paired_daily_block_bootstrap(rows, candidate_id, control_id, repetitions, seed) -> dict`

- [ ] Write failing tests with hand-derived rows covering two months/spouts, four hour buckets and four target ranks. Assert overall E, target WMAPE, signed bias, group keys and literal values.
- [ ] Run the test; verify RED.
- [ ] Implement fixed hour labels and target-specific quartiles using `rank(method="first")` then four-way `qcut`. Every group must call repository `score_predictions`; error is prediction minus actual.
- [ ] Add failing bootstrap tests: deterministic same-seed output, ordered interval, negative candidate-control delta and reversed-sign swapped delta.
- [ ] Aggregate `unit_id + local_date` blocks into target actual/absolute-error sums; sample blocks using `np.random.default_rng(seed)`; return point_delta, ci_low/high, repetitions, seed and block count.
- [ ] Run `test_diagnostics.py`; record pass count.

---

### Task 3: 全流程编排与 Stage 2 门控

**Files:**
- Create: `src/bf_tap/optimization_v03/run.py`
- Create: `tests/optimization_v03/test_run.py`

**Interface:** `run_drift_validation(*, data_config_path, baseline_config_path, feature_config_path, semantic_contract_path, protection_policy_path, protection_ledger_path, drift_experiment_config_path, core_experiment_config_path, optimization_feature_config_path, optimization_validation_config_path, optimization_acceptance_config_path, optimization_model_config_path, output) -> Path`.

- [ ] Write failing helper tests: `stage2_decision` runs only if canonical C1/C2 pass; `quality_status` ignores controls/M1; `assert_c2_equivalence` detects any key/value delta; an existing output fails before core invocation.
- [ ] Run `test_run.py`; verify RED.
- [ ] Create destination with `exist_ok=False`, RUNNING state and code/input/ledger identities. Validate config paths. Call `protection.validate_development_read` before `read_development_labels`; merge only label availability metadata required for causal splits.
- [ ] Invoke `run_optimization_validation(... suite="all", output=destination/"core_v02", candidate_ids=["E00","E09_PROCESS_CHANGE_E02"])`. Require core execution PASS and protected labels false.
- [ ] Per origin, use `select_partitions` and `MedianControls`; load E00/E09; derive B0/B1/M1/C1/C2; audit recent rows; assert C2/E09 exactness.
- [ ] Per unit, persist predictions, row errors and diagnostics. Build `metrics.json`, `grid_summary.json`, `scenario_summary.json`, `candidate_comparison.csv` and `predictions_with_actual.csv`.
- [ ] Evaluate Stage 1 with unchanged `aggregate_grid/evaluate_acceptance`. If and only if C1/C2 both pass, materialize C3 for every unit and recompute; otherwise write SKIPPED without any C3 prediction.
- [ ] Bootstrap C1/C2/conditional C3 against E00/B0/B1 with canonical repetitions/seed.
- [ ] Reverify inputs, source snapshot and ledger; write run_manifest then final_status. On exception write FAILED and preserve evidence.
- [ ] Run all `tests/optimization_v03/test_*.py`; record pass count.

---

### Task 4: CLI 接口

**Files:**
- Modify: `src/bf_tap/cli.py`
- Modify: `src/bf_tap/optimization_v03/__init__.py`
- Create: `tests/optimization_v03/test_cli.py`

- [ ] Write failing parser/dispatch tests for `optimize-v0.3 --data-config ... --output ...`. Assert all default config paths and forwarded arguments exactly.
- [ ] Run the CLI test; verify argparse RED.
- [ ] Add one parser and dispatch branch. Expose no suite/candidate-search option: v0.3 always runs the frozen all-suite set. Leave v0.2 arguments unchanged.
- [ ] Run CLI test plus all v0.2/v0.3 optimization tests; record pass counts.

---

### Task 5: 锁定回归与真实运行

**Runtime:** `local/optimization_v03_runs/v03-drift-real-20260908-01/`

- [ ] Run locked `pytest -q tests/optimization_v03`; require zero failures.
- [ ] Run locked `pytest -q`; require zero failures and no warnings.
- [ ] Run locked Python on `scripts/check_no_private_artifacts.py`; require PASS.
- [ ] Start the authoritative command below and do not edit source/config while it runs.

```bash
env UV_PROJECT_ENVIRONMENT=/tmp/iron-optimization-v02-py312 uv run --frozen --extra dev --python /home/clairvoyant/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12 -m bf_tap optimize-v0.3 --data-config configs/m234.local.yaml --output local/optimization_v03_runs/v03-drift-real-20260908-01
```

- [ ] Read top/core final status, manifests, acceptance, Stage 2, grid and bootstrap. Recompute metrics from `predictions_with_actual.csv`; require serialization-level equality, C2/E09 max delta 0, identities stable, protected labels false and ledger unchanged.
- [ ] Append exact command, Python, counts, path, wall time, G0/G1, gates, metrics and intervals to `progress.md`/`findings.md`.

---

### Task 6: 报告与最终复核

**Files:**
- Create: `docs/optimization_v03/RESULTS_SUMMARY.md`
- Modify: `task_plan.md`, `progress.md`, `findings.md`

- [ ] Report G0/G1 separately; C1/C2 gates; C3 trigger; J/H1-H4; pooled/monthly/target/spout/hour/quartile/bias; bootstrap; seen-development caveat; no protected labels.
- [ ] Programmatically compare every headline number with artifacts. Scan report/plan for `TBD`, `TODO`, stale paths and independent-validation claims.
- [ ] Run fresh full locked pytest, private-artifact guard, `git diff --check` and `git status --short --branch`.
- [ ] Use requesting-code-review on all v0.3 code/config/tests. Any Critical/Important fix starts with a failing test and repeats focused/full verification.
- [ ] Mark Phases 10-14 complete only after code, real run, recomputation and report checks pass. Leave changes unstaged/uncommitted.
