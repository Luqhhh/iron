# Synthetic admission passed

Completed 2026-09-29 06:49:25 UTC. Six full-shape 240-epoch fits plus two
original-code BASE controls, with no official-data model fitted in preflight.
All six beat their training-median constant. Both native BASE parameter states
and predictions match exactly. These synthetic scores do not select a mechanism.

| Component / arm | Query MAE (primary output) | Seconds | Gradient evaluations |
|---|---:|---:|---:|
| Joint iron BASE | 1.803220 | 277.53 | 2160 |
| Joint iron EMA | 1.755465 | 274.05 | 2160 |
| Joint iron SAM | 1.971069 | 500.26 | 4320 |
| Periodic time BASE | .366476 | 275.33 | 2160 |
| Periodic time EMA | .348853 | 262.52 | 2160 |
| Periodic time SAM | .408774 | 416.23 | 4320 |

The native-code controls additionally cost 206.98/206.33 seconds. Synthetic
constant MAEs are 26.792506 /5.358501. Fresh-process saved-state, train-only
preprocessing and epoch/update trace checks pass for all six models. Full-batch
cold difference is exactly zero; maximum order/chunk difference 1.210067e-5
is below the predeclared native-float32 raw-output tolerance5e-4.

Peak worker RSS593.78MiB <1536MiB, four-worker available RAM check passed.
Conservative development-plus-diagnostic projection6.12918h <7.61h, including
the prescribed factor2 safety margin. This is resource admission, not an ETA
or a platform forecast. Frozen sources/config/runtime/reference hashes pass.

The supervisor observed 5/6 completed synthetic units with no failures at its
first scheduled checkpoint and advanced on the actual successful preflight
exit. Official development started at06:49:25 UTC, native BASE replays first.
G0 passed; G1 unmeasured. No package, desktop write or upload.

Private report: local/runs/strong-component-regularization/preflight-r1/report.json.
