# V43 engineering admission

Locked Python 3.12 suite: **1246 passed**, 23 existing warnings.
Implementation `5bd460f`, initial execution record `8a0568a`.

Both full-shape synthetic fits completed using 2204 training rows and 550
query rows, 200 trees, 400 sweeps and 100 retained draws per arm.

| Recipe | Query MAE | Constant MAE | Seconds | Peak RSS MiB |
|---|---:|---:|---:|---:|
| STUMP | 1.116267 | 1.875230 | 4.1995 | 420.88 |
| BART | 0.485595 | 1.875230 | 9.9290 | 463.77 |

BART learns the planted interaction better than the additive control. Both
saved models reproduce separate-process, reversed-row and chunked predictions
with maximum difference **0**. Sampling algebra, state-dependent proposal
ratios and independent mixture-median checks passed in the test suite. These
are engineering checks, not proof of posterior convergence or official quality.

The frozen conservative 80-fit/four-worker projection is **0.11032 hours /
6.62 minutes**, below 7.61 hours. Peak RSS is below 1536 MiB. Four measured
peaks plus the 1024 MiB reserve fit available RAM, rechecked at admission.
Actual development runtime is not guaranteed by a synthetic projection.

Original reference-cache provenance, source/specification digests, exact
runtime versions, saved chain lengths, tree counts and retained sweep indices
pass verification. Specification SHA-256:
`ca27471b7ad7b3bc646cbb3f675c767688366f5807f2a5d863752615776680f4`.

Private report and admission: `local/runs/round2-v43/preflight-r1/report.json`
and `admission.json`; models, traces, preprocessing and predictions remain
private. The hourly checker observed normal terminal exit and stopped itself.
No intermediate progress polling or automatic restart occurred.

G1 remains unmeasured. Admit the frozen 40 development units; no full-data
fit, submission package, desktop write or upload. Goal remains platform 96.5.
