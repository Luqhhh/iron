# V47 results: no incremental gain; high-order training hit the cap

V47 finished the frozen 40 candidate outer units / 80 fits and reused 20
verified reference units. Independent audit passed all 60 units and 80 saved
models. No finalist, confirmation fit, full-data fit, package, desktop write
or upload. Platform best remains user-reported V32_TIME_A60V7_50 = 96.3727;
all three milestones 96.4 / 96.45 / 96.5 remain unmet.

## G1: incremental quality under the frozen procedure

Numbers below are package score-point gains from replacing one incumbent
column with the inner-calibration-selected blend; the other column is unchanged.
They are local measurements, never platform forecasts.

| Target | Arm | Seed 42 | Seed 3407 | Mean |
|---|---|---:|---:|---:|
| Iron | AFM2 control | -.002163824 | -.002152473 | -.002158148 |
| Iron | AHOFM4 candidate | -.000448959 | .000000000 | -.000224480 |
| Time | AFM2 control | .000000000 | .000000000 | .000000000 |
| Time | AHOFM4 candidate | .000000000 | .000000000 | .000000000 |

AHOFM4 standalone WMAPE improves over AFM2 on both targets and both seeds:
iron .05418-.05467 versus .05928-.06006; time .06977-.07059 versus
.07984-.08183. Both remain substantially worse than the current reference
(iron .03715-.03720; time .03813-.03817). Thus the standalone improvement
has not delivered an incremental blend gain. Nineteen of twenty AHOFM4
calibration weights are zero; one iron unit chooses .05 and loses on its outer
fold. AFM2 has three iron weights .05 and all other weights zero. No positive
blended outer fold among either arm's 20 cells. The conditional confirmation
command returned no_development_finalist, confirmation_fits=0; derived seeds
271828 and 314159 remain unconsumed.

## Training-cap evidence and its limit

All 20 AHOFM4 selectors hit 400 epochs. Selected epochs: iron 385-399,
time 396-400. Across every candidate unit, the best calibration MAE continues
to improve between the first 300 and full 400 epochs: mean relative reduction
3.5811% for iron and 4.2642% for time. AFM2 hits the cap in only 2/20 units
(both time); iron control selects epochs 17-21. This post-hoc diagnostic uses
saved inner-calibration traces only and adds no fits or new promotion decision.
Private trace arithmetic is saved as training-cap-diagnostic.json.

The frozen 400-epoch candidate fails. It is not demonstrated to have converged,
and these measurements justify evaluating a separately preregistered longer
schedule, with the same data isolation and unchanged gates. They do not prove
that longer training will approach the incumbent, transfer to the platform,
or reach any milestone. Do not extend or overwrite V47, retrospectively promote
AFM2, lower the increment gate, or manufacture a package from this result.

## G0: execution and independent audit

Locked Python 3.12 suite: 1272 passed, 23 existing warnings, before execution.
Two preregistered synthetic fits passed (G0_RESULTS.md). Actual candidate event
span 398.11974 seconds; peak worker RSS 427.20 MiB. Development exited 0 at
03:36:30 UTC, independent audit exited 0 at 03:36:37 UTC on 2026-09-29.
An inotify completion-file subscription received the real terminal event;
no intermediate health/metric polling. Terminal service MainPID 0,
active/exited, exit 0 verified; the 600-second timer was stopped before its
first due checkpoint. No idle timer or training remains.

The independent audit verified saved parameter shapes/order, training-only
knots, centering and target scales, fresh-refit epoch counts and calibration
trace selection, fit/query row identities, calibration blend weights, all
complete OOF columns and resulting gates. Fresh-process predictions matched
an independent NumPy interpolation and distinct-feature ANOVA dynamic program,
including reversed and chunked queries. Max inference difference 5.68434e-13
(<1e-8); max package-gain recomputation difference 1.81279e-16. Reference scope
is verified source/cache/data/fold/row identity and endpoint arithmetic, without
additional baseline refits. G0 success does not override the G1 failure.

Private evidence root: local/runs/round2-v47/development-r1.

| Artifact | SHA-256 |
|---|---|
| manifest.json | 8e4b6ce49a5373d616364b8c7b8c042b3e8dbb337d65b171cd88353fdc9055db |
| summary.json | 772da240a5d03108c273150430a2f51ef032a6a4c2cdf8c6189bd4c60b7b86ad |
| audit.json | 3f37e371f1daa0de25af5eb763729eee728ce371f6ffeedf3f9b3e211f909137 |
| auditor source | 3c69f97f17d0a3eb1d5c6df0ca7cbf61e81d7b0b7329687c096fb8a9c9d415f6 |
