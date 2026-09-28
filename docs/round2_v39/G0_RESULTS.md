# V39 G0 passed; official quality results are pending

Implementation `0a96868`; frozen protocol `657a72d`. User accepted V37's
7.607-hour development projection with “7.61小时可接受”. Original V37
6-hour refusal remains unchanged. No model, training or quality gate was
relaxed by that resource authorization.

## Engineering checks

- Locked Python 3.12 full suite: **1228 passed**, 23 existing warnings.
- First isolated-worktree run had one temporary-directory confinement failure;
  the task-created `local/tmp` symlink pointed outside the worktree. Replacing
  that link with a real private directory fixed the environment; original
  files were preserved. The related 37 tests and then the entire suite passed.
- Original V37 full-model equivalence/source and explicit time authorization
  verified. No new resource optimizer probes.
- Two full-size synthetic learnability optimizers, exactly 80 epochs each;
  exclusive start/completion ledger, no retries.

| Arm | Synthetic MAE | Constant MAE | Peak process RSS MiB |
|---|---:|---:|---:|
| GLOBAL | 0.231549956 | 1.507812500 | 1140.430 |
| INSTANCE | 0.227399080 | 1.507812500 | 1140.430 |

The synthetic target is an independent three-feature threshold function;
these numbers are functionality checks, not platform forecasts. Both models
passed separate-process persistence, reverse order and 37-row chunk checks
with **zero prediction difference**. The complete wrapper's observed peak
RSS passes 1536 MiB; four workers plus 1024 MiB require
5585.719 MiB against
6763.270 MiB available at admission.

Twenty matching reference component-cache units passed original source,
manifest/audit/data/row/fold identity checks. New time-reference arithmetic is
.20*V36+.30*N0048+.50*V7_periodic, with unchanged iron; historical B0 remains
an explicit separate column. Current reference ZIP SHA-256 verified locally:
`54864561c057ae71f7779b15099750f7caf0141c149d609ae281dbaf9ef3aaa7`.
Its platform score **96.3727** remains user-reported, not independently
verified; the active goal is **96.5**, gap **0.1273**.

## Next frozen execution

Run all 40 development units (80 candidate optimization runs) with four
workers. GLOBAL remains control-only; INSTANCE must beat it and the current
reference on the frozen criteria before any confirmation seed is consumed.
No official candidate has been evaluated at this G0 snapshot. No full-data
fits, package, desktop write or upload is authorized by G0 admission.

Private evidence: `local/runs/round2-v39/preflight-r1` in the isolated V39 worktree.
Preflight SHA-256 `5309f50f8f77049800f41d2837aa883cd38185099de5f6a262da75e882729a44`;
independent memory/start-ledger admission SHA-256 `e3a550b0cb2e9eaa74a9e1835cc2c278e4bc1ac254949b7dbe10514d08427f07`.
