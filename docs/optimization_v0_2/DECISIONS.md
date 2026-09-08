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

## D009 — Promote E09 as the platform incumbent without retuning F1

E09 passed every development gate before upload. The user subsequently reported a
test_a score of `82.8174`, improving the previous incumbent by `0.1128`. E09 is the
new incumbent, but the external total score does not authorize changing the fixed
F1 windows, signed-difference definitions, model parameters or acceptance policy.

## D010 — Pre-register a three-candidate OPT-05 derived batch

Before reading OPT-05 results, the derived batch is fixed without a weight grid:

- `E12_BLEND_E09_E04_80_20` uses uniform per-target weights `0.8/0.2`; E09 is
  dominant and E04 supplies the documented H3/H4 complement;
- `E13_BLEND_E09_E10_50_50` equally averages the two promoted F1 routes;
- `E14_TIMECAL_E09` subtracts `1.68610975` minutes from E09 time predictions and
  leaves iron unchanged. This residual was frozen from early development OOF before
  its earlier platform use and is not refitted from OPT-04 platform feedback.

All components remain separately registered model candidates and are fitted once
per origin. Derived predictions are evaluated by the same full acceptance policy.
The platform score `82.8174` selects no weights or calibration values, and no grid
or post-result substitution is permitted in this batch.

## D011 — Select E12 by the registered aggregate objective

E12 and E14 both passed all six development gates. E12 is selected because its
registered equal-horizon objective J is lower (`0.171713` versus `0.171946`), not
because of any platform observation. E14's slightly lower H1 is retained as a
diagnostic and does not replace the declared aggregate selection rule. E13 is
rejected because it did not beat the better DEV_LONG median control.

The E12 platform artifact must be the exact `0.8×E09 + 0.2×E04` derivation from
separately manifested component predictions. Its package is frozen before upload;
the returned score may update external evidence and the incumbent, but it cannot
retroactively tune OPT-05 weights or substitute E14.

## D012 — Record E12 as incumbent without reopening OPT-05

The user reported an E12 test_a score of `82.9543`, a `0.1369` improvement over
E09. This confirms E12 as the platform incumbent but remains external evidence.
OPT-05 stays frozen: the result does not authorize changing the `0.8/0.2` weights,
applying E14 calibration, or evaluating another OPT-05 blend. OPT-06 must begin
with a separate development-only pre-registration.

## D013 — Pre-register target-level composition for OPT-06

Before reading any OPT-06 run result, two deterministic candidates are fixed from
the OPT-05 per-target evidence. `E15_TARGETWISE_E12_E14` takes `tap_iron` from E12
and `tap_time_len` from E14. `E16_TIMECAL_E12` subtracts the already frozen
`1.68610975` residual from E12 time and leaves E12 iron unchanged. E15 answers
whether the documented target complement transfers across horizons; E16 isolates
whether calibration should occur after rather than before the E09/E04 blend.

Nested derivations must reference earlier registered candidates, carry component
manifest and result hashes, and preserve sample order. No new residual, blend weight,
CatBoost parameter, feature window or acceptance threshold is introduced. Promotion
is based on G1 and J only; the OPT-05 platform score is not a selection input.

## D014 — Select E16 and preserve E15 as the H1 diagnostic

E15 and E16 both passed every G1 gate. E16 is selected because its registered J is
lower (`0.169902` versus `0.171057`) and it wins H2, H3 and H4. E15's slightly lower
H1 (`0.156197` versus `0.156308`) is retained as a diagnostic, not used to override
the aggregate rule. The frozen E16 package is the exact E12 test_a result with only
`1.68610975` subtracted from time predictions and the nonnegative floor preserved.

The pending platform score may update the incumbent but cannot refit the residual,
switch to E15, or create another target combination within OPT-06.
