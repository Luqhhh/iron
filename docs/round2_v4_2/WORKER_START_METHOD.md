# V4.2 screen workers: the fork-with-torch deadlock and its fix

Date: 2026-09-25

## Symptom

The first full coarse-screen launch computed the four shared `B_fit` baselines
correctly and then stopped: no outer fit completed, no slot was written to the
ledger, and the machine load fell back to idle while the process stayed alive.

## Diagnosis

The baseline stage is executed **in the parent process**: `V36FixedRecipeFactory`
fits the frozen V3.6 recipe, which includes a torch network expert, so the parent
ends up with an initialised torch/OpenMP runtime. Stage B then created a
`ProcessPoolExecutor` with the default `fork` start method, handing the children
a copy of that already-initialised runtime. The children never made progress.

A controlled reproduction isolates the cause:

| parent state before the pool | `fork` | `spawn` |
|---|---|---|
| bare `import torch` | works | works |
| a real `B_fit` (torch network fitted in the parent) | **hangs** | works |

The `spawn` run of the same reproduction completed: `B_fit done in 110.8 s`,
then `R0` and `R1` outer fits finished in 17.0 s and 51.1 s.

## Fix

`bf_tap_r2.v4_2_screen.run_screen` now takes `start_method` (default `spawn`,
also exposed as `--start-method`) and passes it as the pool's `mp_context`.
`fork` and `forkserver` remain selectable for debugging but are not the default.

Two further robustness changes were made at the same time:

* each outer fit is persisted (ledger entry plus its `V` prediction vector) as
  soon as it completes, instead of only after the whole stage finishes, so an
  interrupted run keeps every completed slot and resumes from it;
* a failed slot is recorded with `event: failed` rather than `complete`, so it
  stays retryable instead of being silently treated as done on the next run.

## Consequence for the evidence

Slots completed before the fix were not persisted, because the pre-fix runner
only wrote at the end of the stage; they were recomputed after the fix. No
partial or failed result was carried into the reported screen.
