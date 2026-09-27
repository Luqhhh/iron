# V17 results: no promoted candidate

The user explicitly resumed work with `D:\Edge\iron_964_v17_plan_and_tools.zip`
after the stop-after-V16 instruction. The ZIP supplied the V17 plan and a
standard-library source-ZIP composition auditor; it did not contain model
training code or competition data. This round implemented its six frozen units
and ran the full registered development and conditional confirmation budget.

## G0 engineering and source audit

The exact original V12, V7 and A35 ZIP SHA-256 values matched the plan:
`a1c205a6722da3976a12e258458b649967c7c25130a2c55d840c5ecb2a1dc669`,
`4382523c7bd688974f87eab2f42502bf8f54b2b330008b36e797ae7672490299`,
and `b4e1fc2d1287a213e9d88d1c30a7420ecc222235fcc6b7191e341dfd16924b06`.
The auditor verified 322 unique ordered IDs, identical CSV columns, finite
nonnegative predictions, V12 time strings byte-for-byte equal to A35, and V7
iron strings byte-for-byte equal to A35. It produced only a private report,
`local/runs/round2-v17/b0-audit-r1.json` (SHA-256
`fe56e0def388e076b60b9747ea042f5a1460a870d781460a436c993dd04e5507`).
No combined ZIP was created. `96.3679` remains conditional score arithmetic,
not a platform observation.

The local B0 was independently rebuilt within each seed/fold: V12 iron is
`0.5*A35_I + 0.5*V12_member_I` and V7 time is
`0.5*A35_T + 0.5*V7_member_T`. Its complete pooled package scores are
`96.247974/96.247504` on development seeds 42/3407 and
`96.242873/96.233951` on confirmation seeds 7777/12011. No OOF vectors were
averaged between split seeds.

Development completed **60/60 outer fits**, all successful, with two optimizer
runs per fit (inner epoch selection and fresh outer-training refit).
Confirmation completed **20/20 outer fits**, all successful, for the frozen J2
iron and P-LL time finalists. An interrupted J2 implementation smoke run did
not complete an outer fit or enter any score; it is not counted in either
registered budget. BLAS/OMP/MKL/NUMEXPR were pinned to one thread for both
parallel batches. No selected model hit the 240-epoch cap.

The independent read-only audits passed all 60 development and 20
confirmation prediction matrices, the append-only fit ledgers, training-row
exclusions, data/fold/source/reference hashes, and recomputed fixed-slot
scores and gates. Private evidence directories:
`local/runs/round2-v17/development-r1` and
`local/runs/round2-v17/confirmation-r1`. Their summary SHA-256 values are
`5bfc50e4faa7009360cd728a35748bad22bc0aade53c86f53c9bfe7e20bef1c2`
and `5bc0ab77c69f32ff36c4a23c6af4b34ba8c3bee29bd625aac567e0c65b8918b4`.
Audit status files are `audit-r1.json` in each directory. The candidate tier
record is private `development-r1/tiers-r1.json`.
The final locked Python 3.12 test path passed **1120 tests, 23 warnings**.

J2's shared-gradient conflict frequency averaged **0.1807** across its ten
development fits; mean cosine was **0.2230**. The asymmetric projection was
active but did not produce a stable improvement. Each dual embedding preserves
a 16-dimensional TabM input. P-LL and P-LH have the same parameter count:
**187056** for joint iron, **182944** for single-output time. Historical
single-branch counterparts have **187312/183200** parameters, so the V17
dual modules use 256 fewer parameters. Mean final frequency-weight standard
deviations were P-LL iron `0.06795/0.06681`, P-LH iron
`0.06083/0.10539`, P-LL time `0.05588/0.05950`, and P-LH time
`0.05261/0.10649`. Both branches were active and their initialization-scale
difference remained visible; this is not evidence of a quality gain.

## G1 model quality

The full candidate register is [CANDIDATES.md](CANDIDATES.md). Each entry
replaces only one original 0.5 expert slot against B0, retaining the other B0
column exactly. The two development seeds selected J2 iron and P-LL time.
The existing `candidate_tiers` policy classified J2 as formal; P-LL iron was
the one exploration shortlist but failed the separate two-positive-seed
confirmation prerequisite. Tier classification did not authorize release.

| Confirmed unit | Seed 42 | Seed 3407 | Seed 7777 | Seed 12011 | Mean gain | Seed LCB95 | Disposition |
|---|---:|---:|---:|---:|---:|---:|---|
| J2 iron | +0.001613 | +0.003280 | −0.001212 | −0.000269 | +0.000853 | −0.001499 | Fails all-four-positive and LCB gates |
| P-LL time | +0.002433 | +0.000706 | +0.006456 | +0.009330 | +0.004731 | +0.000143 | Fails frozen local 96.25 gate |

J2 had 13/20 descriptive positive folds. P-LL time had 15/20; its four split
seeds are positive and the paired seed-level lower bound is barely above zero.
Its development mean package score is **96.249309**, which is **0.000691 below**
the preregistered 96.25 gate. J2's development mean is `96.250185`, but its
two confirmation seeds are negative. The frozen gates were applied without
exceptions. Split seeds reuse the same labeled rows, so the LCB is a split
stability diagnostic, not a guarantee from four independent samples.

No candidate was promoted. There were **0 full-data fits, 0 new model or
submission packages, 0 desktop writes and 0 agent uploads**. A cold release
check is inapplicable because no model reached release eligibility. Existing
V12/V7/A35 package bytes and historical decisions remain unchanged. The
current best platform score remains user-reported V12 `96.3526`, not an
independently verified receipt; the requested `>96.4` target is unmet. The
conditional B0 platform arithmetic `96.3679` has not been tested as a combined
submission. No account quota is inferred.
