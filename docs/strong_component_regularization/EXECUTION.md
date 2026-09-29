# Execution record

Implementation b0d372a; preregistration 48b77c2. Locked Python 3.12.12
`uv run --locked --no-sync --python 3.12 pytest -q`: 1301 passed, 23 existing
warnings. The subsequently completed summary/confirmation-seed checks passed
in the 14-test focused suite. Runtime torch 2.14.0+cpu and the original
reference-cache manifests, sources, data and 20 cache units verified.

Started 2026-09-29 06:33:55 UTC (14:33:55 Asia/Shanghai), private root
`local/runs/strong-component-regularization`. Process supervisor records
stage starts/exits and blocks on child completion. Its 600-second wait timeout
records scheduled observations; no intervening training log/PID polling.
Preflight first, then exact BASE replay, candidate development, fresh-process
audit and strictly gated confirmation. It stops on any failed stage and does
not restart, change hyperparameters or overwrite evidence.

G0 implementation/tests/reference verification passed; full-shape synthetic
learnability, cold inference and resource admission pending at launch.
G1 is unmeasured. No full-data fit, package, desktop write or upload.

The seed1042 diagnostic changes the native training seed (initialization,
dropout and minibatch order together). It measures training-seed sensitivity,
not the isolated causal effect of initialization. It never selects candidates.

## Admission and development start

Preflight exited0 at06:49:25 UTC; G0_RESULTS.md records the successful six-arm
synthetic audit and two native controls. The supervisor immediately launched
the frozen development stage (20 exact BASE replay units before 40 EMA/SAM
units, then six training-seed diagnostics). G1 results remain pending.

An initial wall-clock-based observation just after600s preceded the supervisor's
actual scheduled record by about13s. No intermediate training metrics were
available in that observation; the subsequent actual scheduled record is the
authoritative check (5/6 complete, zero failures). Subsequent checks should use
scheduled events or actual stage-completion notifications, not an estimated due time.
