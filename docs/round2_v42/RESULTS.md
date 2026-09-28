# V42: negative spline quality and a preserved numerical audit failure

Goal **platform 96.5** remains unmet. Current user-reported best remains
**V32_TIME_A60V7_50 = 96.3727**, gap **0.1273**. No new platform measurement.
Protocol `f17e3ce`, implementation `1ff53a1`, engineering admission `81c4900`,
official start `cb2b037`. All **40 candidate outer units / 80 solver fits**
completed without a training failure; 20 reference units were reused after
the frozen identity checks. Actual candidate fit span: **67.99 seconds**.

## G1: the frozen candidate does not qualify

Values are whole-package score-point increments over the current reference,
with the other target unchanged. Each seed has complete five-fold coverage;
complexity and blend weights were selected inside each outer training split.

| Target | Arm | Seed 42 | Seed 3407 | Mean |
|---|---|---:|---:|---:|
| Iron | ADDITIVE control | +0.000097 | -0.002558 | -0.001231 |
| Iron | PAIR candidate | +0.002125 | -0.008024 | **-0.002950** |
| Time | ADDITIVE control | -0.004144 | -0.004682 | -0.004413 |
| Time | PAIR candidate | -0.065441 | -0.043239 | **-0.054340** |

No arm has two positive complete development seeds. PAIR also falls below
the control mean on both targets and below the frozen +0.01 gain requirement.
The independent diagnostic audit recomputed every pooled score and verified
that neither target has a finalist. Maximum score-arithmetic discrepancy:
**2.71e-16**. The numerical G0 failure below is separate; it does not transform
these measured negative increments into eligible candidates.

PAIR improves standalone WMAPE over ADDITIVE, but remains worse than the
incumbent: iron approximately .03983–.04000 versus .03715–.03720; time
.06060–.06148 versus .03813–.03817. Better performance than a weak mechanism
control is insufficient when its incremental blend loses on the held-out rows.

All 40 selectors built the 63-term forward maximum. Calibration selected
17 terms in 8 units, 33 in 18 and 63 in 14. All ten iron PAIR cells
selected 33/63 (specifically six 33 and four 63); time PAIR selected 63 in
five cells. The boundary selections limit any claim of family-wide capacity
adequacy. They do not authorize extending the frozen term, knot or degree grid.
Half of the 40 blend weights were zero; no post-hoc weight repair was made.

## G0: tests and fit identities pass; full inference audit fails

The locked Python 3.12 suite passed **1240 tests**, with 23 existing warnings,
before the frozen run. Synthetic full-shape admission passed as documented in
`G0_RESULTS.md`; official workers recorded peak RSS **302.57 MiB**. Original
source/spec/runtime/data/fold and every completed unit artifact hash verify.

The original cold audit stopped on
`tap_iron-ADDITIVE-s3407-f4/model.json`: reversing rows or predicting in chunks
of 37 changes the output by **1.37198981065e-8**, exceeding the predeclared
absolute tolerance **1e-8**. **The original audit is failed, not passed.**
No tolerance was widened, no prediction overwritten, and no model refitted.

A separate diagnostic auditor retains all original checks but collects cold
failures so the rest of the evidence can be inspected. It writes only
`audit-diagnostic-r1.json`, never the runner's required passing `audit.json`.
Its status is **failed_cold_invariance**, `release_authorized=false`.
Across all **80 models / 60 units** it finds exactly the one failure above.
Both PAIR candidates' saved models satisfy the original cold tolerance.
It independently verifies preprocessing and training IDs, knot origin and
support counts, factor definitions, coefficient predictions, calibration-only
complexity/weight selection and the no-finalist decision.

A zero-fit numerical diagnostic finds direct cold prediction in the original
row order exactly equal to the saved predictions for all 80 models. The
offending model has coefficients up to 164180.45 and absolute dot-product
terms summing to about 2.094e6; subtractive cancellation makes BLAS summation
order visible. A per-row `math.fsum` diagnostic differs from the saved values
by at most **1.97397e-8**. That is an observation, not a replacement inference
contract or a retrospective pass. Original predictions and algorithms remain
unchanged. Any future reuse needs a separately validated numerical repair.

## Monitoring, decision and preserved evidence

The preflight and official development each received one scheduled hourly
observation, both reporting normal terminal state. Each timer stopped itself.
No intermediate progress polling, retries or automatic restarts occurred.
Systemd `active/exited` reflects retained terminal status, not a running fit.

Confirmation fits **0**; split seeds 271828/314159 remain unconsumed. The
confirmation runner is not invoked because the required passing audit is
absent and the development eligibility is empty. Full-data fits, packages,
desktop writes and uploads **0**. This frozen round is closed negative with
the explicit G0 failure preserved. No family-wide impossibility is claimed.
The broader 96.5 goal stays active.

Private evidence root: `local/runs/round2-v42/development-r1`.

- Manifest SHA-256: `954bb5d37b82afacda8845a7d64b4c18db6fa629c9a141f86e651157bb00c5dc`.
- Summary SHA-256: `a95e0b847954ddc332ef3cec93d073db0a071eb7fe368399dd3abb9079c43dea`.
- Original failure: `audit-failure-r1.json`.
- Complete diagnostic audit: `audit-diagnostic-r1.json`.
- All-model numerical diagnosis: `numerical-diagnostic-r1.json`.

The diagnostic auditor itself is pinned inside its report with SHA-256
`0eaedde572678fc58ff1b5bd2dd27bb32cddef92a628a6e9fa2153892983b927`.
Models, predictions, reports and monitoring logs stay outside Git.
