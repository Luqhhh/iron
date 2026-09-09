# optimization-v0.5 results

Current phase status: **V05_CLOSED_KEEP_R2**. The authorized continuation completed
OPT-15 with 24 target fits; T1/T2 also failed both gates. OPT-16 was not triggered.
See [OPT15_EXECUTION.md](OPT15_EXECUTION.md) for the completed continuation and
validation. The following OPT-14 first-delivery record is preserved as historical
evidence; its pending statements describe that earlier delivery.

## OPT-14 first-delivery record

First delivery complete; OPT-15 and OPT-16 are not executed. No model was trained,
no challenger was packaged, and the R2 active release remains unchanged.
No Git push or platform upload was performed. November is POST_HOLDOUT_CONSUMPTION.

## Real endpoint reproduction

24 complete saved components (six cutoffs × four roles), each containing both
targets, were loaded in a fresh Python process. All 20 units (18 grid cells plus
DEV_LONG/DEV_SHORT) exported O0/H0/OR/HR and unrounded clipped predictions.
The fit guards recorded zero attempted and zero completed fits. Public feature
construction received only sample_id/spout_no/reference_time. No official target
columns or test_a inputs were read for selection. Input-order reversal passed.

40 A00/A11 endpoint comparisons passed; maximum sample-level difference:
**1.7053025658242404e-13**, below 1e-8. Recomputed E12-raw/R2/E09/B0/B1
reference grid metrics match v0.4 within 5.551115123125783e-17.

## Fixed candidate outcomes

Lower E/J is better; positive deltas are regressions against R2.

|Pipeline|H1 E|H1 delta vs R2|18-cell J|J delta vs R2|STAGE_A|GENERAL|
|---|---:|---:|---:|---:|---|---|
|E12-raw|0.1561869850|-0.0014653012|0.1683691454|+0.0013787009|reference|reference|
|R2|0.1576522863|+0.0000000000|0.1669904445|+0.0000000000|reference|reference|
|P1|0.1579920495|+0.0003397632|0.1689005866|+0.0019101421|FAIL|FAIL|
|P2|0.1561766835|-0.0014756028|0.1668583460|-0.0001320985|FAIL|FAIL|
|P3|0.1584617697|+0.0008094835|0.1707686871|+0.0037782427|FAIL|FAIL|

P2 (original main + robust auxiliary) improves H1 by 0.0014756028082509032
and improves five of six H1 origins. It also improves J by 0.00013209847228135008.
However H4 regresses by **0.002751537083795913**, beyond the pre-registered
0.0015 STAGE_A ceiling (and 0.002 GENERAL ceiling). It is rejected without exception.
Compared with E12-raw, its H1 change is only -0.00001030156788239811: this is
primarily recovery of the existing near-term loss, not an across-the-board breakthrough.
P1 and P3 worsen both H1 and J. All three candidates fail both gates.

## Actual auxiliary contribution

D_help = WMAPE(R2) - WMAPE(OR), averaged equally across origins within each horizon.
Negative values mean adding HR at the fixed 20% weight helped in the measured units.

|Horizon|Target|Mean D_help|Mean D_O|Mean D_H|Mean interaction|
|---|---|---:|---:|---:|---:|
|H1|tap_iron|-0.0015934742|0.0017966265|-0.0001941780|-0.0007205841|
|H1|tap_time_len|-0.0000254928|0.0014840407|-0.0001558867|0.0000616608|
|H2|tap_iron|-0.0055705952|0.0016213148|-0.0020297659|-0.0012367136|
|H2|tap_time_len|-0.0045055802|0.0003684499|-0.0026543013|-0.0004275469|
|H3|tap_iron|-0.0049130518|0.0014924231|-0.0025906567|-0.0000399988|
|H3|tap_time_len|-0.0062842409|0.0012349358|-0.0019293619|-0.0005126754|
|H4|tap_iron|-0.0034891155|-0.0011725347|-0.0021305366|-0.0001374295|
|H4|tap_time_len|-0.0038443909|-0.0041710974|-0.0019990789|-0.0001814547|

HR helps both targets on every horizon mean, including H1; individual months/spouts
can differ. Removing the auxiliary is therefore not supported by these results.
The symmetric representation effects suggest robust-main near-term regressions
and robust-auxiliary benefits on average, with robust-main H4 benefits; these
are predictive component comparisons, not physical causal effects.
All month/spout effects, opposite-sign errors, cancellation amounts and target
numerators/denominators/signed bias remain in the local reports.

## Paired week stability

Intervals are retrospective diagnostics, not selection-adjusted confirmation.
All repeated samples/origins share calendar-week weights; denominators are recomputed.

|Candidate|Valid / requested draws|H1 delta 2.5% / median / 97.5%|J delta 2.5% / median / 97.5%|
|---|---|---|---|
|P1|982 / 1000|-0.0004039 / 0.0003358 / 0.0011875|0.0013538 / 0.0019124 / 0.0024340|
|P2|982 / 1000|-0.0027222 / -0.0014553 / 0.0002106|-0.0010679 / -0.0001450 / 0.0006999|
|P3|982 / 1000|-0.0001258 / 0.0008871 / 0.0019333|0.0027876 / 0.0037908 / 0.0048029|

## Evidence and next boundary

- Export: `local/runs/optimization-v0.5-opt14-export-r1/` (manifest, component identities, predictions, fit guards).
- Scores: `local/runs/optimization-v0.5-opt14-score-r2/` (40 endpoint comparisons, all units, six H1 origins, recent three months, complementary errors, separate gates and intervals).
- Access: `local/ledgers/optimization-v0.5-development.jsonl`; old protected ledger verified unchanged.
- Failed scoring r1 is retained: JSON serialized integer origin keys as strings. The scorer now normalizes those keys, and its original source is retained with the failure. No model inference was repeated to repair scoring.

The full v0.5 phase is **not yet complete**. The first-delivery OPT-14 scope is complete.
No STAGE_A candidate passed, so the pre-registered OPT-15 condition is satisfied.
T1/T2 (at most 24 development target fits) and conditional OPT-16 remain pending.
There is no new submission package, threshold exception, platform claim, or B/C result.

## Locked engineering validation

After the JSON-origin repair: **150 passed** in 57.10 seconds under locked Python 3.12.
Report: `local/reports/pytest-optimization-v0.5-opt14-r2.xml`.
Tests include fixed component alignment, missing/duplicate/nonfinite/negative data,
missing models, wrong source/cutoff/component identity, fit rejection, metadata-only
feature input, future/current-history rejection, decomposition identity, zero target
denominators, and distinct STAGE_A/GENERAL gate behavior. Real all-unit endpoint
reproduction and source/snapshot checks complement these synthetic tests.
