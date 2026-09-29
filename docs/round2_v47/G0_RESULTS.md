# V47 G0 admission

Implementation a93da5c, protocol acb581c. Full locked Python 3.12 suite:
1272 passed, 23 existing warnings. Fixed synthetic generator and full 400-epoch
schedule were declared before the two fits; no post-result tuning occurred.

| Synthetic arm | MAE | Constant MAE | Seconds | Peak MiB |
|---|---:|---:|---:|---:|
| AFM2 | 2.686038667 | 3.002375755 | 13.5364 | 497.184 |
| AHOFM4 | 1.918426723 | 3.002375755 | 28.4630 | 507.215 |

Both beat the constant and AHOFM4 improves over AFM2 on the predeclared
four-way synthetic signal. This is learnability evidence, not an expected
real-data or platform gain. No synthetic rerun or settings change.

Fresh-process predictions verified against independent NumPy interpolation and
ANOVA dynamic programming; maximum cold/order/chunk/independent differences:
AFM2 1.24345e-14, AHOFM4 9.76996e-15, below the frozen 1e-8 limit. All training
basis identities, moments, target scales, coefficient shapes and epoch traces
verified. Source/runtime and all 20 original reference units reverified.

Conservative four-worker development projection: .316255 hours (18.98 minutes),
including the frozen 2x timing factor, below 7.61 hours. Worker peak 507.215 MiB
below 1536; current available 12675.05 MiB > required 3052.86 MiB.

Private report: local/runs/round2-v47/preflight-r1/report.json,
SHA-256 b8fb1185dd2140a4ca1f46f63f4a17d31818ec198262837298d9e0d23e9f2f39.
Spec SHA-256 5d2aa4bd0c7a166c6176313af106e61cc924117adb6658e70773b7c2f06416bd.

Completion event records exit 0 at 2026-09-29 03:26:06.668055 UTC. An event-file
notification was consumed before the first 600-second monitoring checkpoint;
terminal service verified MainPID 0, active/exited, status 0. The unused timer
was stopped. This is completion-triggered audit, not an early periodic health
check. No intermediate training metrics were polled.

Admit the frozen 40 real-data outer units / 80 fits. The 20 baseline units are
reused only after original identity verification and frozen endpoint reweighting.
No additional real-data reference fits are needed for development. G1 remains
unmeasured until complete coverage and independent audit. No derived seeds,
full-data fits, packages, desktop writes or uploads at this stage.
