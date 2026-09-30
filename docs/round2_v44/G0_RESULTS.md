# V44 long-chain engineering admission

> Historical execution contract (reading update 2026-10-01): keep all measured failures, counts, frozen gates and source identities below. Subsequent user instructions resumed optimization, set milestones 96.4/96.45/96.5, changed monitoring to 600 seconds and removed future wall-clock budgets. Earlier pause/hourly/time-admission statements are not current instructions, and old failures are not relabeled as passes. See [current rules and status](../INDEX.md).

Locked Python 3.12 suite: **1248 passed**, 23 existing warnings.
Implementation `e590cfc`, initial execution record `45581b3`.

Both full-shape long fits completed on the same 2204 synthetic training / 550
query rows as V43, with 200 trees, 4000 sweeps and 100 retained new draws.

| Recipe | Long MAE | V43 short MAE | Constant MAE | Seconds | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|
| STUMP | 1.116597 | 1.116267 | 1.875230 | 37.2349 | 450.64 |
| BART | 0.562954 | 0.485595 | 1.875230 | 95.6851 | 500.52 |

Both learn and BART beats the additive control, satisfying the frozen G0
criterion. **The longer BART chain is worse than the short one on this
synthetic query set.** Thus this check is not evidence that lengthening will
improve official-data quality, and no such gain is presumed.

For both models, the first 400 scalar traces, prior, transforms, target range,
cutpoints and all 100 old sampled ensembles exactly reproduce V43. The old
saved predictions also reproduce with maximum difference **0**. The new
retained sweep indices are exactly 2020,2040,...,4000. Old-prefix samples are
audit-only. Separate-process, reversed-row and chunked long predictions have
maximum difference **0**.

The conservative 80-fit/four-worker projection is **1.06317 hours**, below the
accepted 7.61-hour budget. Peak RSS is below 1536 MiB; four peaks plus a
1024 MiB reserve fit available RAM, rechecked at admission. Projection is not
an actual-runtime guarantee. Exact runtime versions and source hashes, all
60 original V43 units, reference identity and new specification hash pass.

Specification SHA-256:
`0e321074387275fb62f39d34d8e2fbbfbd73e590cb3c81179f684a6243bc7c0e`.
Private evidence: `local/runs/round2-v44/preflight-r1/report.json`,
`admission.json`, both prefix-verification records and saved models/traces.

The hourly checker observed successful terminal state and stopped its timer.
No intermediate progress polling or automatic restart occurred. G1 remains
unmeasured; proceed with the separately frozen 40-unit development. Preserve
V43's failure. No full-data fits, packages, desktop writes or uploads.
