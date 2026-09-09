# optimization-v0.4 results

All five months are retrospective development; November is POST_HOLDOUT_CONSUMPTION.
Negative changes mean lower loss. No platform outcome is inferred.

## OPT-11

|Month|E12 model effect|E12 history effect|Joint change|Interaction|
|---|---:|---:|---:|---:|
|2024-07|-0.0006067|-0.0013079|-0.0019146|-0.0016665|
|2024-08|-0.0035039|-0.0032534|-0.0067573|0.0010784|
|2024-09|-0.0045078|-0.0069736|-0.0114814|-0.0034553|
|2024-10|-0.0024288|0.0005028|-0.0019260|-0.0036112|
|2024-11|-0.0038284|-0.0010242|-0.0048525|-0.0077457|

All 24 legacy endpoint comparisons (four months × two endpoints × three components)
passed, maximum numeric difference 1.1368683772161603e-13.
The E09 model effect improves in every month; E04 model effect worsens in every
month. E12 historical refresh improves in four months and worsens slightly in October.
These data do not establish historical refresh as the cause of the December platform decline.
No algorithm weights were changed in response to this observation.

Full component/target metrics, numerators, denominators, signed errors and spout splits:
local/runs/optimization-v0.4-r4/factorial_scores.json and factorial_by_spout.json.

## Engineering

Locked Python 3.12 suite: 132 passed (62.64 seconds).
The actual two-anchor R3 research composite restored the old bundles and predicted
with a cold subprocess configuration containing no official label/history paths.
Maximum difference from the original predictors: 2.2737367544323206e-13.
No fit, submission package or activation occurred in that engineering check.

## Candidate validation

All 18 cells and DEV_LONG/DEV_SHORT completed with common controls.

|Candidate|Extended 18-cell J|Improvement vs E12-raw|Gate|
|---|---:|---:|---|
|E12-raw|0.16836914536496014|—|Reference|
|R1|0.16906557410954984|-0.0006964287445896944|FAIL|
|R2|0.16699044445671873|0.0013787009082414092|PASS|
|R3|0.16798146208439044|0.00038768328056970613|FAIL|

R2 improves H2/H3/H4; H1 worsens by 0.0014653012403684773, below the
registered 0.002 regression ceiling. Its target equal-horizon WMAPE changes are
iron -0.0008018268454387037 and time -0.001955574971044101. DEV_LONG beats the
better simple control; DEV_SHORT improves by 0.00006271453245978997.
R1 improves only one horizon; R3 fails minimum J gain, improved-horizon count,
and DEV_SHORT regression. No exception is used.

The legacy 14-cell E12-raw J remains 0.17171342055719652. It is kept separate
from the expanded score and is not overwritten in old evidence.
Both November combined endpoints additionally align with old prepared predictions
(329 rows each, maximum difference 1.1368683772161603e-13).

R2 is the sole development-accepted challenger. Final training, package verification and bootstrap diagnostics are complete. E16 remains incumbent.

## Unlabeled sensitivity and week diagnostics

Both raw publication anchors (2024-11-01 00:00:00+08:00 and
2024-12-01 01:44:00+08:00) completed all last-one/last-three/group and individual
masks. All furnace/spout history columns were recomputed together; no fit or scoring
was performed by sensitivity. Full target/component median/p95/max values are in
local/runs/optimization-v0.4-r4/sensitivity.json.

|Cutoff|Mask|Target|Median absolute change|p95|Maximum|
|---|---|---|---:|---:|---:|
|2024-11-01 00:00:00+08:00|last1|tap_iron|12.98440|26.35626|43.18229|
|2024-11-01 00:00:00+08:00|last1|tap_time_len|2.99199|8.37477|10.79432|
|2024-11-01 00:00:00+08:00|last3|tap_iron|8.54593|18.04911|22.42099|
|2024-11-01 00:00:00+08:00|last3|tap_time_len|2.63495|8.13754|11.72309|
|2024-12-01 01:44:00+08:00|last1|tap_iron|19.51712|30.45017|38.93287|
|2024-12-01 01:44:00+08:00|last1|tap_time_len|1.89600|5.81613|7.93779|
|2024-12-01 01:44:00+08:00|last3|tap_iron|5.46232|14.81695|29.44535|
|2024-12-01 01:44:00+08:00|last3|tap_time_len|2.94956|4.94334|6.39788|

Paired calendar-week bootstrap: 1,000 requested draws, 982 valid draws.
R2-minus-reference J interval: [-0.0023916579379916277, -0.00032730153143576204],
median -0.0013710680340023249. Repeated origin samples share week weights.
This is not selection-adjusted inference or an untouched confirmation.

R2 November H1 E is worse by about 0.00685. The pre-registered gate is based on
equal-horizon means and screening folds, not a new November-only veto or preference.
All six H1 origins and every September–November cell are listed in
local/runs/optimization-v0.4-diagnostics-r1/h1_six_origins.csv and
recent_three_months.csv.

## Final candidate delivery

R2 final fit used 2,754 eligible rows at 2024-12-01 01:44:00+08:00.
E09/E04 remain 800 rounds, 80/20, nonnegative clipping, zero calibration.
The registered last30/last100 medians and true counts are the only feature strategy change.
The 335-row test_a CSV and ZIP passed exact sample/order checks, finite/nonnegative
checks and archive readback. Cold-process maximum raw difference: 5.684341886080802e-14.
Its configuration omits official label/history paths; stored training snapshots are included.

Candidate ZIP:
`local/runs/optimization-v0.4-r2-challenger-r1/Luqhhh_bf_tap_predict_prelim.zip`

SHA-256: `e42602d3045e43b4b49dd1e1c104aa8e5c1f29c639ff5f892b3b07434ed9bbdf`.

E16 remains active and its original ZIP hash was verified unchanged. The challenger
has not been uploaded or activated. There is no new platform result or B/C assessment.

Completed target-fit accounting across preserved attempts: 8 for OPT-11 June/July,
48 for the two fixed OPT-12 candidates, 4 for the May R3 anchor, 4 for R2 final fit;
total 64. Sensitivity and cold-process checks perform zero fits.

Initial failure/interrupted evidence and all previous-version results remain present.
No remote write was performed in this phase.
