# optimization-v0.6 results

OPT-17 real zero-fit replay passed every frozen deployment gate. S1 routes H1–H3
to P2 and H4 to R2. These are retrospective development results, not platform gains.
All old v0.5 failures remain failures under their original definitions.

|Horizon|S1 E delta vs R2|Iron WMAPE delta|Time WMAPE delta|
|---|---:|---:|---:|
|H1|-0.0014756028|-0.0014363345|-0.0015148711|
|H2|-0.0005788172|-0.0010029580|-0.0001546765|
|H3|-0.0012255109|-0.0014724237|-0.0009785981|
|H4|0|0|0|

Equal-horizon J = **0.1661704617134884**, improvement over R2 =
**0.0008199827432303353**. This is recomputed from individual sample predictions,
not simply copied from the earlier horizon-delta arithmetic. H1 improves at five
of six origins. DEV_LONG improves 0.0032766040 and DEV_SHORT improves 0.0004699869;
DEV_LONG uses sample-level routing, including its H4 rows.

|H1 origin|E delta vs R2|
|---|---:|
|June|-0.0024854830|
|July|-0.0006243406|
|August|-0.0007287207|
|September|-0.0005162311|
|October|+0.0007539001|
|November|-0.0052527417|

G0: 18 existing complete dual-target components reloaded at six cutoffs;
20 evaluation units and 40 P2/R2 endpoint checks; maximum absolute difference
**1.7053025658242404e-13**, below 1e-8. H4 predictions exactly equal the freshly
recovered R2 endpoint. Fit guards recorded zero attempted/completed fits.
Feature builders receive only sample metadata, public as-of process features and
verified bundled history. The inference configuration contains no training label
paths. Row reversal is checked during prediction; IDs and cross-origin labels
are checked separately during scoring.

Per-sample errors, raw error sums, denominators, signed bias, target/month/spout
scores, complete endpoint checks and acceptance are retained in
`local/runs/optimization-v0.6-opt17-r1/`. Calendar-week bootstrap uses shared week
weights across origins and recalculates denominators: 982/1000 valid draws.
The J 95% percentile interval is [-0.0016223355, -0.0000006797]; H1 is
[-0.0027221562, +0.0002105883]. These intervals do not cover selection bias and do
not establish independent confirmation.

Locked Python 3.12 suite: **173 passed**, 54.45 seconds.
Report: `local/reports/pytest-optimization-v0.6-r1.xml`.
Initial new-test fixture timestamps mixed date-only and minute formats; normalizing
the fixture fixed its parsing failure. No model or routing definition was changed.

## OPT-18 and final decision

LOO target/horizon routing fails promotion: J = **0.16703904097505887**,
regression against S1 = **0.0008685792615704679**. H1 is unchanged, but H2 iron
regresses 0.0016426290 and H3 iron/time regress 0.0023343872/0.0021202629,
exceeding the 0.001 per-target/horizon cap. DEV_LONG/SHORT diagnostic deltas are
+0.0030319561/+0.0006618475. No final S2 mapping is frozen and no S2 package is made.
Detailed held-origin exclusions, other-origin means and choices are in
`local/runs/optimization-v0.6-opt18-r1/decisions.json`; routed errors and both gate
checks are retained alongside it. No new training occurred.

The completed phase selects **S1_HORIZON_ROUTER**, with a **READY_CHALLENGER**
test_a package. This is a deployment-gate pass, not a new platform result. Current
incumbent R2 remains unchanged (user-reported 83.0207); neither desktop copy nor
platform upload was performed.

Final existing components all use cutoff **2024-12-01 01:44:00+08:00** and the
same 2754 training sample identities. Three complete dual-target components were
copied and verified; new training calls remain zero. Independent cold processes
and the original v0.4 R2 loader agree to maximum absolute difference
**5.684341886080802e-14**. The bundle records each component's own variant, schema,
model/history/source digest and common exact cutoff. Submission ID/column/value
checks and ZIP readback pass.

|Stage|Rows|Routing|Deliverable|
|---|---:|---|---|
|test_a|335|H1 P2|challenger ZIP|
|test_b|322|H2 P2|unscored dry-run CSV|
|test_c|548|303 H3 P2; 245 H4 R2|unscored dry-run CSV|

Package: `local/runs/optimization-v0.6-s1-challenger-r1/Luqhhh_bf_tap_predict_prelim.zip`

SHA256: `5d74272c9abc6d12924f2c1ad96f534a0f19042210bdf7746a27e125a7498267`

Retained R2 SHA256:
`e42602d3045e43b4b49dd1e1c104aa8e5c1f29c639ff5f892b3b07434ed9bbdf`.
Old v0.4 source, old v0.5 results, active release pointer and protected ledger
remain unchanged. A separate append-only v0.6 access ledger records new reads.

Current platform last-submission vs best-submission behavior has not been verified
in this execution; verify it before any later authorized platform upload.

## Authorized desktop delivery

After the completed experiment, the user explicitly requested desktop replacement
and Git commit/push. The S1 ZIP was copied to
`/mnt/c/Users/lqh22/Desktop/Luqhhh_bf_tap_predict_prelim.zip`; its SHA256 was
verified as `5d74272c9abc6d12924f2c1ad96f534a0f19042210bdf7746a27e125a7498267`.
The earlier no-desktop-overwrite statement describes the initial experiment delivery.
The archived R2 ZIP and active release config remain unchanged. No platform upload
or new platform score is recorded by this desktop-copy action.

## User-reported test_a feedback

After verified S1 desktop delivery, the user reported **82.7707**. This is associated
with the delivered S1 ZIP from conversation context; no independent platform receipt
or cross-submission evaluation-version verification was obtained. Compared with the
user-reported R2 83.0207, the score change is **-0.2500** (vs E16 82.9918: -0.2211).

S1 is not promoted. R2 remains the active repository release. S1's earlier deployment
PASS and READY_CHALLENGER checks remain valid historical engineering/development
results; this external result does not rewrite them. The development H1 improvement
did not carry through to the reported test_a score. A total score cannot identify
a target-specific problem or justify changing sample predictions, routing weights,
windows or calibration. No further v0.6 candidate is introduced.

The desktop still contains the explicitly requested S1 package; this score-recording
action does not overwrite it or upload anything. The original R2 archive remains
available for rollback. Official last-vs-best submission behavior remains unverified.
