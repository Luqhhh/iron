# TABM_BOOTSTRAP_EMA_V1 results

G0 passed: 2 saved states/2 optimizer runs, 4 candidate/control cold models. Independent float64 recurrence replayed initialization and every raw post-update state; Poisson weight/order certificates, selected epochs, training partitions, preprocessing and cold sequential/chunk predictions passed. Maximum G0 prediction difference 7.516235626781054e-6. No new control/reference fits.

G1 completed with actual exit0: 10 development outer fits, 20 saved states/20 optimizer runs, 2754 unique IDs and five-fold coverage per seed. Independent cold audit passed all 40 new/old checkpoints, with maximum prediction difference 5.631260250993364e-5 and peak RSS439.60546875MiB. No science retries. Neural full tests1494passed23warnings; locked Python3.12 full tests1064passed107skipped3warnings.

| Time A20 relative to Q75 | Seed42 | Seed3407 | Mean |
|---|---:|---:|---:|
| Bootstrap+EMA | +0.002750073362 | +0.006373575902 | +0.004561824632 |
| Original Bootstrap | +0.002350669349 | +0.006934220048 | +0.004642444699 |
| Paired mechanism difference | +0.000399404012 | -0.000560644146 | -0.000080620067 |

Both candidate seeds are positive relative to Q75. Mean paired mechanism difference is slightly negative, so the frozen automatic confirmation gate was not met: confirmation0 outer fits/0 states/0 optimizers; no formal four-seed promotion. Retain BOOTSTRAP_EMA_TIME_A20 as the sole positive two-seed exploration candidate. This small local deficit is not a platform rejection; no platform superiority or score is inferred. Current Q75=96.3920 is user reported, not independently verified.

Iron mean +0.000653333518 and mechanism +0.000266017466 are descriptive only. Time was the sole preregistered selectable target; per-target descriptive eligibility in the immutable original evaluation does not authorize iron confirmation or release.

The original startup stopped after full tests when inherited SPEC prose was clarified before any scientific freeze. Original sources/failure receipts were retained; recovery verified unchanged numerical recipe and scientific runner AST (except dependency collector), reused the completed tests and consumed G0 only once. After science completed, the original final verifier incorrectly included descriptive iron in target selection; a separate time-only verifier read the unchanged frozen SPEC, independently recomputed complete OOF/endpoints and confirmed the actual decision. That original verifier and failure were retained; zero fits and an unchanged fit ledger in scope recovery. Neither event is a model training failure.

Private evidence: local/runs/tabm-bootstrap-ema-v1 and local/bootstrap-ema-20261003, including preflight, raw EMA updates, original failure/source bridges, audit/evaluation/actual exit and final-verification-scope-recovery-r1.json. No fullfit, package, upload or desktop write. Complete results only; model/prediction/labels/ledgers remain local/.
