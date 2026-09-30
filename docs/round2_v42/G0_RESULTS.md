# V42 engineering admission

> Historical execution contract (reading update 2026-10-01): keep all measured failures, counts, frozen gates and source identities below. Subsequent user instructions resumed optimization, set milestones 96.4/96.45/96.5, changed monitoring to 600 seconds and removed future wall-clock budgets. Earlier pause/hourly/time-admission statements are not current instructions, and old failures are not relabeled as passes. See [current rules and status](../INDEX.md).

Locked Python 3.12 suite: **1240 passed**, 23 existing warnings.
Implementation `1ff53a1`; initial execution record `02e01d0`.

Two full-shape synthetic fits completed successfully, using 2204 training
rows, 550 query rows and the frozen 63-term search/pruning procedure.

| Recipe | Query MAE | Constant MAE | Selected terms | Seconds | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|
| ADDITIVE | 0.985312 | 1.818132 | 9 | 0.5445 | 411.98 |
| PAIR | 0.086315 | 1.818132 | 17 | 3.6992 | 413.77 |

The planted hinge interaction is learned, with PAIR better than the additive
control. Both searches built all 63 terms. Separate-process persistence,
query order and chunk-size comparisons have maximum difference **0**.
These checks establish engineering capability, not official-data quality.

Twice the slower measured fit, scaled to 80 solver fits on four workers,
projects **0.04110 hours / 2.47 minutes**, below the accepted 7.61-hour budget.
Peak worker memory is below 1536 MiB; four peaks plus a 1024 MiB reserve fit
within available RAM, rechecked immediately before official admission. The
runtime projection is conditional on this workload and is not a guarantee.

Reference-cache provenance, frozen source hashes, runtime versions and
specification digest all pass. The specification SHA-256 is
`2c139a294ce85c6fcaa470ab11b284f02f94cb9603e96a15c3b23d77e21cca71`.
Full private evidence: `local/runs/round2-v42/preflight-r1/report.json`,
`admission.json`, both saved models, preprocessing and search metadata.

Hourly-only monitoring observed successful process exit and stopped its timer;
no intermediate training polling or automatic restart occurred. Official G1
quality remains unmeasured at admission. Proceed with the frozen 40 development
units, not full-data fits or packages. Goal remains platform 96.5.
