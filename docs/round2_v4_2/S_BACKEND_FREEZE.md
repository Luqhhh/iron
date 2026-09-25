# V4.2 S-line backend: budget-control verification and freeze

Date: 2026-09-25
Status: **verified and frozen before the first real training run**

## 1. The question the task book requires answering first

> 每训练子集表达式评价上限拟为 200000；实际后端能否精确控制此预算须先验证，不支持就明确
> 修改并在首次真实训练前冻结，而不是静默忽略。

So the S line may not run a single real fit before it is established whether the
chosen backend can enforce an exact per-training-subset expression-evaluation
budget, and the answer must be frozen on the record.

## 2. Method

The frozen V3.6 environment was **not** modified. PySR was installed into an
independent interpreter with its own Julia depot:

```bash
uv venv --python 3.12 local/envs/v4_2_pysr
UV_CACHE_DIR=local/tmp/uvcache \
  uv pip install --python local/envs/v4_2_pysr/bin/python pysr
JULIA_DEPOT_PATH=$PWD/local/envs/julia-depot \
  local/envs/v4_2_pysr/bin/python src/bf_tap_r2/v4_2_pysr_worker.py <job.json>
```

Three probes were run and their raw output is preserved under
`local/runs/round2-v4.2-structure-search/backend-probe/`:

1. API surface inspection (`PySRRegressor.__init__` signature).
2. A budget-control and throughput probe on a `1800 x 25` design matrix with
   `maxsize=24`, `maxdepth=5`, 8 populations of 40, `deterministic=True`,
   `parallelism='serial'`.
3. An end-to-end worker run through the adapter used by the runner.

No example dataset was downloaded: PySR was only ever handed a caller-supplied
design matrix and target, and `from_pmlb`-style helpers are never called.

## 3. Findings

| property | result |
|---|---|
| PySR version | `2.5.0` |
| exact budget parameter | `max_evals` (default `None`) |
| Julia runtime | usable; the depot precompiles successfully |
| run with `max_evals = 2000` | ok, 15.35 s (includes cold Julia start-up), 4 hall-of-fame equations |
| run with `max_evals = 20000` | ok, 1.31 s (warm), 8 hall-of-fame equations |
| measured cost | ~6.5e-05 s per evaluation at `n = 1800`, `p = 25` |
| projected 200000-evaluation search | ~13 s |
| peak RSS of the worker | ~1.0 GB |

Because `niterations` is set to `10**7`, `max_evals` is the binding stopping
condition, so the pre-registered cap is the quantity actually enforced.

Additional adapter-level findings that are recorded because they changed the
implementation rather than being silently ignored:

* `temp_equation_file=True` is mutually exclusive with `output_directory`; the
  temporary-file mode is used so nothing is written next to the sources.
* `parallelism='serial'` must not be combined with `procs`/`multithreading`;
  the modern switch alone is used, which is also what deterministic mode
  requires.
* custom operators require matching `extra_sympy_mappings` entries.
* the no-self-nesting rules are expressed through `nested_constraints`, not
  `constraints`, and only declared operators may appear in them.
* PySR emits fitted constants as SymPy `Float` nodes, so equations convert back
  into the frozen AST language exactly; integer exponents `x**2` are converted
  to a repeated `mul`, and anything outside the frozen operator set is rejected
  rather than approximated.

## 4. Decision (frozen)

1. **PySR is the primary S backend**, because the exact `max_evals` budget is
   both accepted and enforced and the cost is affordable for this round's slot
   count.
2. The per-training-subset budget remains **200000 expression evaluations**, as
   pre-registered, and `maxsize=24` / `maxdepth=5` are passed through to the
   backend.
3. The **in-repo bounded AST search is retained** as a deterministic fallback
   and as an independent cross-check of candidate scoring. A backend failure or
   an unconvertible hall of fame falls back explicitly and records
   `fallback_used: true` plus the failure reason in the fit ledger; it is never
   silently substituted and no result is reported as PySR-derived when it is not.
4. Every PySR equation is converted to the frozen AST language before it is
   scored, selected, constant-refitted and exported, so backend inference and
   exported NumPy inference share one implementation and are checked against each
   other on real and boundary inputs by the test suite.
5. The independent environment stays independent: `local/envs/v4_2_pysr` and
   `local/envs/julia-depot` are the only places PySR and Julia live, and both are
   under the git-ignored private `local/` tree.

## 5. Reproduction

```bash
# 1. recreate the independent backend environment
uv venv --python 3.12 local/envs/v4_2_pysr
UV_CACHE_DIR=local/tmp/uvcache uv pip install --python local/envs/v4_2_pysr/bin/python pysr

# 2. re-run the capability probe (writes into the private run directory)
JULIA_DEPOT_PATH=$PWD/local/envs/julia-depot \
  local/envs/v4_2_pysr/bin/python local/tmp/pysr_probe3.py

# 3. run the runner's S line (uses the adapter and the frozen budget)
.venv/bin/python -m bf_tap_r2.v4_2_screen --lines S --jobs 2
```

`src/bf_tap_r2/v4_2_s_symbolic.pysr_backend_report()` returns the same
capability summary at run time and is stored in every S fit record, so a future
environment change that removes `max_evals` shows up in the evidence instead of
silently altering the search budget.
