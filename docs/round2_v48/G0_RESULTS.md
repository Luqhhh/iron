# V48 G0 admission

Protocol 91c9eba; implementation 8e15245. The locked Python 3.12 suite passed
1279 tests, with 23 existing warnings. Seven new tests cover exact refit and
selector checkpoints, an intentionally earlier best epoch, state/history/settings
tampering, early-stopped control, fresh-seed behavior and full partition replay.

Both preregistered synthetic models ran all 2000 epochs from scratch. Captured
actual new-run checkpoints at epoch 400 exactly reproduce V47 parameters and
traces; prefix predictions are bit-identical (difference 0). No old parameter
state was loaded into training and no extra prefix-only fit was performed.

| Arm | V47 MAE (400) | V48 MAE (2000) | Constant | Seconds | Peak MiB |
|---|---:|---:|---:|---:|---:|
| AFM2 | 2.686039 | 2.479139 | 3.002376 | 75.4941 | 498.363 |
| AHOFM4 | 1.918427 | 1.607712 | 3.002376 | 188.6768 | 510.688 |

Both beat the constant and AHOFM4 improves over AFM2. This is a fixed synthetic
learnability/resource check, not a prediction of real-data or platform gain.
No post-result tuning or synthetic rerun.

Fresh-process saved inference against independent NumPy interpolation/ANOVA DP,
reverse and chunk queries: max difference 1.24345e-14, below 1e-8. Old prefix
parameters/traces and original predictions exact; independent prefix-DP difference
<=1.24345e-14. Train-only basis, scaling, row digests and model metadata verified.

Conservative four-worker development projection 2.096409 hours, including the
frozen 2x timing margin, below 7.61 h. Peak worker 510.688 MiB <1536; available
12577.09 MiB > required 3066.75 MiB. Source/runtime, original V47 source/data/audit
and all 60 old units reverified, as well as the 20 reference-cache units.

Private report local/runs/round2-v48/preflight-r1/report.json:
ae6f6b2f45838abc83a4b4058e7ba0df1b6dcf28ffc6d3d2c565596d49c7e245.
Spec: e8a9dc078b75971f12844ed2acb1b616def6277e8b27d73662d552e5b7c5f276.

An inotify subscription received the real terminal completion event at
2026-09-29 03:55:05.969804 UTC, exit 0. Service PID 0, active/exited, successful
exit verified; the 600-second timer stopped before its first checkpoint. No
intermediate health or training-metric polling.

Admit the frozen 40 real-data units /80 fits and require every one of their
80 old model prefixes to reproduce exactly. G1 remains unmeasured. No full-data
fit, package, desktop write or upload. All gates and staged targets unchanged.
