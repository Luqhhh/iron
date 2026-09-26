# V7b execution evidence (2026-09-26)

**G0: real-data control passed, maximum absolute prediction difference 0.**
The original inner-training procedure at seed 42 / fold 2 / epoch 55 reproduces
the frozen N-0048 held-out predictions bit for bit. The independently written
fixed-epoch optimizer loop therefore passed the predeclared 1e-9 control gate
before any candidate refit started. Control metadata and predictions remain in
`local/runs/round2-v7-coverage/development-r1/`.

The ten full-training-fold refits completed with two single-thread workers,
zero failures, and verified full gradient-training row coverage.
They retain the original architecture, optimizer, preprocessing and cached
inner-selected epochs. Their additional training rows also increase optimizer
steps; this is the declared procedure, not an isolated causal estimate of row
coverage alone. Original models, ZIPs and past decisions are unchanged.

**G1: failed the predeclared two-positive-development-seed condition.**

| Reference | seed 42 score gain | seed 3407 score gain | mean fold-aggregated gain |
|---|---:|---:|---:|
| A35, endpoint weight 0.35 | -0.001901 | +0.005020 | +0.001559 |
| Q20, endpoint weight 0.20 | -0.000915 | +0.001944 | +0.000515 |

A35 pooled gains are -0.001888 / +0.005027, with the same mixed signs; 7/10
folds improve. The frozen tier policy labels this `exploration` solely because
the mean improves; its explicit failed condition is `both_splits_improve`.
That classification does not authorize confirmation or a platform experiment.
No further seeds are consumed and no package is recommended for this small,
unstable gain. The tested fixed-epoch coverage procedure stops here.

Both standalone N refits have slightly lower WMAPE than their originals
(0.040737 -> 0.040683 and 0.041091 -> 0.040569), yet one blend split loses.
Residual correlations with V36 rise from 0.8868/0.8885 to 0.8950/0.8976.
These are descriptive observations, not a causal proof or a new correlation
gate; incremental blend gain remains the binding comparison.

Independent direct arithmetic reproduces both pooled A35 gains to 1e-12.
All ten prediction files are hashed in private `audit-r1.json`, and every
fit's metadata confirms `gradient_rows == provided_rows`. Actual experiment
cost: one exact real control plus ten candidate refits. No four-seed
confirmation, full-data fit, package or upload occurred. The platform target
remains 96.4; current user-reported best remains A35 = 96.3366.

Latest locked Python 3.12 test path: **987 passed, 3 warnings**. This includes
the new coverage-loop control and V7 confirmation identity/coverage guards.
The V7 confirmation runner waits for the complete frozen candidate pool,
selects its highest-ranked two-development-seed-positive candidate, and checks
data, code, runtime, cached-reference identity and held-out row masks. Existing
V5 confirmation caches contain time only; an iron winner requires fresh
same-fold baseline fits and is explicitly refused by that cache-only runner.
No time cache can be silently used for an iron candidate.
