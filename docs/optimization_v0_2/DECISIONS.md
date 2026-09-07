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

## D006 — Pre-register two frozen-history adaptations

Before reading any new development result, OPT-03 registers two candidates with
synthetic history ages fixed at `{0, 7, 30, 60, 90}` days:

- `E07_FROZEN_E02` keeps the E02 feature route (sample, process, burden and
  history count/target values, without raw history-age columns);
- `E08_FROZEN_E04` keeps the E04 feature route (sample plus all history
  components).

Every view of an original sample remains in the same outer and audit-only inner
partition. Each view receives weight `1/5`, so every original sample contributes
total weight one. A history row is visible only when its complete result is
available by the synthetic history origin. E00, E01, E02 and E04 remain direct
original-history, no-history and unadapted-route comparisons. Promotion is based
on the unchanged development acceptance policy; platform scores do not select the
candidate or alter its weights.

## D007 — Preserve the failed gate while honoring an explicit probe request

OPT-03 passed G0 but neither registered candidate passed G1. E07 was the stronger
candidate: it improved J and every horizon versus E00, but failed the required
DEV_LONG comparison with the better median control. The user explicitly requested
that a submission package nevertheless replace the desktop package. E07 is
therefore frozen and labeled `FROZEN_EXPLORATORY_PLATFORM_PROBE_NOT_G1_ACCEPTED`.
The acceptance threshold and failed result are unchanged. The subsequently
reported `82.3610` score must not be used for further adaptive tuning.

## D008 — Pre-register signed process changes for OPT-04

Before reading OPT-04 development results, F1 is fixed as three signed differences
for every one of the 16 frozen hourly operation values:

- latest value minus the 6-hour as-of mean;
- latest value minus the 24-hour as-of mean;
- 6-hour as-of mean minus the 24-hour as-of mean.

No new lookback window or raw source is introduced. The features are derived from
the same baseline as-of aggregates and retain missing values rather than imputing
them. Three candidates isolate the increment: `E09_PROCESS_CHANGE_E02` adds F1 to
E02, `E10_PROCESS_CHANGE_E01` adds F1 to the no-history E01 route, and
`E11_PROCESS_CHANGE_E04` adds only the F1 process source to E04. E00, E01, E02 and
E04 remain controls. Model parameters, post-processing and acceptance are
unchanged; no platform result participates in feature or candidate selection.
