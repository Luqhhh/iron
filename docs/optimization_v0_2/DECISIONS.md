# optimization-v0.2 decisions

## D001 — Isolate optimization from the frozen baseline

The baseline parser, feature configuration, CatBoost parameter guard and bundle v3
loader are unchanged. Optimization code lives under `bf_tap.optimization` and must
explicitly load the frozen contracts before fitting any candidate.

## D002 — Build baseline features once, then apply auditable column masks

OPT-02 changes only source/component inclusion. The same as-of feature builder is
used for E00 and every ablation. Unknown feature columns and history columns that do
not map to exactly one of age/count/target fail closed.

## D003 — Treat E06 as a diagnostic family, not 25 pseudo-candidates

The registered E06 prediction uses separate fixed weights for iron and time. Each
fold additionally reports the five pre-registered weights per target. This answers
whether shrinkage helps without presenting every weight pair as an independent
model candidate.

## D004 — Do not pool origin scenarios as independent OOF rows

Prediction identity is `(candidate_id, origin_id, sample_id)`. J is computed by
averaging origin losses within each horizon and then assigning equal weight to the
four horizon means.

## D005 — Stop adaptive platform iteration before OPT-03

The user-reported platform sequence improved monotonically, but it is a sequence of
related, adaptively selected candidates. It is retained as exploratory evidence,
not an independent validation set. After the time-calibrated 75/25 blend, candidate
selection returns to development-only evidence. The next platform check requires a
pre-registered and frozen OPT-03 candidate.
