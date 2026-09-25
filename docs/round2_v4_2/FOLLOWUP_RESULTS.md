# Round2 V4.2-r2 follow-up: N2 seed-initialisation repair and full coverage

Date: 2026-09-25
Branch: `round2-v4.2-n2-seed-repair` (based on the reviewed commit
`63d0adc3e11fc6e22e08eef62fb68963ea0959ae` of `round2-v4.2-structure-search`)
Repair version: **`SEED_INIT_V2`**
Private evidence: `local/runs/round2-v4.2-structure-search/n2-seed-init-v2`
Status: **repaired coarse gate PASSED; repaired full-coverage gate FAILED on the
mean threshold; no fusion, no replication, no package, no upload**

> Workspace note: a second session (`Luqhhh`) was active in this same checkout
> during this round. It committed the core r2 repair files verbatim as `96d1f85`
> on `round2-v4.2-structure-search` and then moved on to a V4.4 round. The
> collision is recorded here because it overwrote one private file
> (`candidate_identity.json`) inside the run directory; see section 8.

## 1. What this round repaired

| item | before | after |
|---|---|---|
| initialisation seed timing | network built with the ambient torch RNG, seed set only afterwards inside `_run_epochs` | each stage builds its network inside an explicit `torch_seed_context`, before the first `torch.randn` |
| batch-order randomness | reused the training seed implicitly | a separate, explicitly registered numpy stream (`training_seed + 1_000_003`, `+1_000_004`) |
| per-fit reproducibility record | `"initialisation": "torch_manual_seed"` | stage-1/stage-2 initialisation hashes, final state hash, source digest, dependency identity and an explicit scope |
| full-coverage gate state | the `+0.02` gate was reported as failed before folds 2/3/4 existed | `NOT_EVALUATED_INCOMPLETE_COVERAGE` on partial coverage; `PASSED`/`FAILED` only on the complete ten cells |
| N2 encoding in `RESULTS.md` | described as PLE | corrected to `raw` (N3 is the PLE two-layer recipe) |
| cache identity | `line/recipe/target/seed/fold` only | a full candidate identity digest plus a separately keyed `B_fit` identity |

The repair keeps the loss, learning rate, tree count, depth, routing
temperature, encoding and fusion weight unchanged. It does not upgrade the
frozen V3.6 environment and does not change `B_fit`.

Scope note: the R line has the same *structural* seeding ordering as the old N
line. It is deliberately left unchanged here — path A is a repair and full
coverage of N2 only, the R evidence was not rerun, and silently changing R would
desynchronise already-recorded R results from the code.

## 2. Repaired protocol

* candidate trial id `N-N2-tap_iron-SEED_INIT_V2`, numeric encoding `raw`;
* split seeds 42 and 3407; training seed 42; inner U/H split seed 20260925;
* coarse folds 0/1, full coverage folds 0/1/2/3/4;
* fixed path `fixed_quarter` at weight 0.25; the `direct` path is reported but
  never combined with it;
* identity digest
  `1356558248df257433d2b4b337467caa0faf7211b46d8eb68a2c05a8f1cc3693`;
* model/training source digest
  `c78134ea7167b835646ae67b61fe095caae030de1d6f1e09c5288e61256f868f`;
* `torch.set_num_threads(4)`, recorded because the thread count is part of the
  numeric environment;
* dependencies: `torch 2.14.0+cpu`, `numpy 2.2.6`, `pandas 2.3.3`,
  `catboost 1.2.8`, `lightgbm 4.6.0`, `scikit-learn 1.8.0`, `sympy 1.14.0`,
  `PyYAML 6.0.2`, Python 3.12.12.

Both stages intentionally use the one registered initialisation seed, so the
stage-1 and stage-2 initialisation hashes are equal by construction
(`stage_inits_share_registered_seed: true`). Stage 2 is a refit from the same
deterministic starting point with the epoch count selected on `H`.

The repaired initialisation hash
`7e27ed9933201404ea36501d89b0216293a9ae89d48073cf9c8cfc755572fc30`
is identical for **all ten real fits**, which ran in ten separate `spawn`
workers. That is cross-process initialisation equality on the real data, not
only on a synthetic test frame.

### Numeric scope (reported, not overclaimed)

A fixed fitted network returns bitwise-identical values for a fixed batch shape;
`predict()` called twice is exactly equal. Changing the batch shape (chunked,
reversed, subset, single row) changes the float32 GEMM reduction order. The
observed N-line difference is `~3.6e-6` absolute on predictions of order 24
(`~1.5e-7` relative). Agreement across batch shapes is therefore asserted at
`atol=1e-4, rtol=1e-5` (`v4_2_train.NEURAL_INFERENCE_ATOL/RTOL`) and the
observed difference is recorded. No bitwise claim is made across thread counts
or batch shapes.

## 3. Q0 engineering evidence

All synthetic and engineering checks were run before the real fits:

* `tests/test_round2_v4_2_followup.py` (16 tests) covers the RNG plan, the
  in-process/after-unrelated-task/new-process initialisation equality, the
  stage machine, the incomplete-coverage state, recipe filtering, identity
  isolation and `B_fit` key independence;
* `tests/test_round2_v4_2.py` (33 tests) keeps the R/S/N mechanism, isolation,
  screen arithmetic and packaging contracts green;
* the coverage gate returns `NOT_EVALUATED_INCOMPLETE_COVERAGE` with
  `passes: null` on 4/10 cells, and `PASSED`/`FAILED` only on 10/10;
* an unknown recipe is an error, and unit selection is exact
  (`select_units(spec, lines=["N"], recipes=["N2"], targets=["tap_iron"])`);
* the `B_fit` cache was keyed independently of the N repair: **0** refusal to
  verify, **10/10** vectors reused from the pre-repair caches once their stored
  identity reproduced exactly, **0** refits, and **no** `B_replay` substitution.

Repository-level verification of this revision:

* `scripts/check_no_private_artifacts.py`: **PASS** (819 tracked/untracked files,
  no private artifact in Git);
* full test suite: **921 passed, 1 failed**. The single failure is
  `tests/test_round2_next_phase_v31_ref.py::test_default_path_still_requires_the_ledger`,
  which expects the V3.1 centres to be absent from
  `local/runs/round2-v3-local-search/coarse-catboost-r1/fit_ledger.jsonl`
  (dated 2026-09-23). That ledger now resolves six centres, so the test's
  precondition no longer holds in this checkout. It touches no V4.2 module and is
  unrelated to this round; it is reported rather than hidden.

> Concurrent-session caveat: the repository-level suite ran while the other
> session was editing shared V4.2/V4.4 files. The result above is therefore a
> snapshot, not a frozen revision hash; the V4.2-scoped suites (49 tests) are the
> ones this round is accountable for.

`B_fit` package scores over the five folds: `96.20378939769073` (seed 42) and
`96.20474104232599` (seed 3407).

## 4. Path A results

### A1 — four repaired coarse units (folds 0/1)

| split seed | `fixed_quarter` gain | `direct` gain |
|---|---|---|
| 42 | **+0.010583** | −0.124810 |
| 3407 | **+0.009031** | −0.154154 |
| mean | **+0.009807** | −0.139482 |

One and the same pre-declared path (`fixed_quarter`) is positive in both split
seeds with a mean above `+0.005`, so the **coarse continuation gate PASSED**.
Fold-level positives were 2/4 for `fixed_quarter` and 0/4 for `direct`, which is
recorded separately and is not used as the pooled score.

### A2 — six fill-in units (folds 2/3/4)

| split seed | `fixed_quarter` gain | `direct` gain |
|---|---|---|
| 42 | **+0.014528** | −0.110459 |
| 3407 | **+0.014221** | −0.130700 |
| mean | **+0.014374** | −0.120580 |

Per-fold `fixed_quarter` gains:

| fold | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| seed 42 | +0.031219 | **−0.009845** | +0.017397 | +0.005787 | +0.027915 |
| seed 3407 | +0.021424 | **−0.003404** | +0.013302 | +0.020940 | +0.018718 |

So the complete ten cells have both split seeds positive and **8/10 positive
folds**, but the mean is `+0.014374`, below the pre-registered `+0.02`.

**Full-coverage fusion gate: FAILED (mean threshold only).** No alpha rescue scan
was run; A3 fusion and A4 replication are `NOT_EVALUATED`.

### Stage table

| stage | status | detail |
|---|---|---|
| coarse | `PASSED` | `fixed_quarter` +0.009807, both seeds positive (folds 0/1) |
| full_coverage | `FAILED` | +0.014374 mean, both seeds positive, 8/10 folds positive, needs ≥0.02 |
| fusion | `NOT_EVALUATED` | gate failed; the line stops with no alpha rescue |
| replication | `NOT_EVALUATED` | requires a completed fusion selection |
| submission | `NOT_EVALUATED` | this runner never packages or uploads |

### Per-slot fit record

All ten fits used the same initialisation hash
(`7e27ed9933201404ea36501d89b0216293a9ae89d48073cf9c8cfc755572fc30`) because both
stages deliberately share the registered initialisation seed; the fitted state
and the prediction differ per cell, so no cell is a copy of another.

| seed | fold | selected epoch | stage-1 epochs | final model state SHA-256 | prediction SHA-256 |
|---|---|---|---|---|---|
| 42 | 0 | 486 | 536 | `5fb0efee471ed3b7d3cbc71d05a2c4317bd09808afb1442933cfd422b3456f9d` | `fe0de6d72a127c49b0f7052dae0adb5ac4c108f9d42cdbe8b391ba4bf89686ab` |
| 42 | 1 | 560 | 610 | `452f346153e626c0f1d20fd940ab5848d61839b41e23625f6ea36a203baa4a34` | `4615fa562ab0f2d224379af827dc173e42983194ae8e6c4c164b94742270c6eb` |
| 42 | 2 | 407 | 457 | `2e48c817a1971f7e0ed4d3e12437eda69d3196469fb524071c4cbf3374e552b6` | `cfcfa943c7375614a21707f94a14a539fe0876160e9904a523fff298c75f8e75` |
| 42 | 3 | 572 | 622 | `bef06c83a626410af7590e14af39dfaba4873d4c89d846ad8c667448cb1f71c9` | `4f2eafa70aa4a575af930e928186ba43adab6c5c7bca535fb267560e39371d7d` |
| 42 | 4 | 453 | 503 | `42b21444153f9589486dbaebd59b06302f3e1038667961234e7efc2608b130f9` | `c2e7820a4909a8e8c99b258942c4e858d45b22c7c8936a14d5cec652e3ada275` |
| 3407 | 0 | 491 | 541 | `8ce85bb0f40a308c0720ad650d59e978c70ca899fb046bb1ec5f1953f8405783` | `acf4484c511db86b70adf8f091ea8b007d2d21ed3371b4bbdf65d752dd91770c` |
| 3407 | 1 | 501 | 551 | `133f0c9081b34fd199f789a69d92a70b277b400b075ce1231b06593bce8025b9` | `67f8e4ae70a08bc0814fef6d96ccf50ccb3dc3b7ba6e19e2d2335cf910a334e3` |
| 3407 | 2 | 389 | 439 | `f1ac52a9be1c896ded087836c6a8bece9ae398a4e27e8f6a4176aec77fbedc4c` | `f60a13cf3d457ed02149f85b5177a6728cea3bc89f044834c4240664671c5c03` |
| 3407 | 3 | 439 | 489 | `266b9a8427ef877c3fae80a33cf096b6956afd8694fe5e03a55bee21d63f0778` | `8d4f5af668c08e330cddf73d16cf22c5a40b9271d4bfe2102421ac5d70adc03f` |
| 3407 | 4 | 453 | 503 | `0226efecf75dd2b2a76e67ec04eb702776a7bbd28cf29d2c97c637c0e19b4b6e` | `f9971c1c8c1a0a23b5c674b6b507ec2d3beeca5cf8be38c9065c4d1d04096d93` |

Every slot stopped by early stopping; none was `budget_limited`. The complete
per-slot record, including requested/effective parameters, stopping reason,
routing diagnostics and memory, is in `fit_ledger.jsonl`.

### Fit bookkeeping

| item | count |
|---|---|
| outer recipe slots (A1 + A2) | 4 + 6 = 10 |
| N2 `fit` calls | 10 (all stopped by early stopping) |
| underlying neural trainings | 20 (stage-1 selection + stage-2 refit) |
| `B_fit` factory calls | 0 (10/10 verified cache hits) |
| models fitted on `V` | 0 |

Selected epochs ranged 389–572; every slot recorded its requested and effective
parameters, stopping reason, initialisation hashes, final state hash and
prediction SHA-256. Wall clock for the whole stage machine: 1550.8 s.

## 5. Preserved pre-repair diagnostics

The un-repaired evidence is kept and never merged with the repaired identity:

| run | candidate | folds | mean `fixed_quarter` gain | note |
|---|---|---|---|---|
| `coarse-r1` | `N-N2-tap_iron` | 0/1 | +0.009316 | historical coarse report |
| `full-r1` | `N-N2-tap_iron` | 0–4 | +0.014214 (8/10 folds) | un-repaired full coverage |
| `repl-t3407-r1` | `N-N2-tap_iron` | 0/1 | training-seed 3407 | un-repaired replication |

The repaired full-coverage mean (`+0.014374`) is close to the un-repaired one
(`+0.014214`): controlling the initialisation seed moves the per-seed values
noticeably (coarse seed 42 `+0.0069 → +0.0106`) but does not change the
conclusion. The repaired evidence is a separate identity in a separate
directory; no five-fold vector mixes old and new predictions.

## 6. What this does and does not show

Does show: with the initialisation seed genuinely applied before the first
random draw, the N2 quarter-blend contribution is reproducible and the
pre-registered full-coverage gate is **not** met — the shortfall is in the mean
size (`+0.0144` vs `+0.02`), not in sign stability or fold coverage.

Does **not** show: any platform gain, any promotion, any packaging. This remains
two-seed development-split evidence; the historical selection bias of the whole
V4 search is not removed by a new directory or a new seed. No test label was
read, no ID/row-order feature was built, and no unrelated task was inserted into
any fit.

## 7. Path B design (declared, not executed)

`configs/round2_v4_2/FOLLOWUP_SPEC.yaml` declares the next structural hypothesis
with `path_b.enabled: false`, and `load_followup_declarations` rejects a spec that
turns it on, so it cannot start implicitly because path A missed a gate.

The hypothesis: a tree currently reuses one feature projection at every depth and
only the threshold changes. `N2_DEPTHSELECT` would give each depth its own
selector:

```text
feature_logits : (trees, depth, features)
selectors      : entmax15(feature_logits, dim=-1)
z              : einsum('bf,tdf->btd', x, selectors)
route          : sigmoid(z - threshold[trees, depth])
```

It keeps the raw encoding, two layers, 64 trees per layer, depth 4, the linear
head, threshold initialisation, fixed temperature, MAE, AdamW and the early
stopping protocol; learnable temperature, data-aware initialisation, PLE and
more trees are explicitly forbidden. It adds parameters, so the true parameter
count is reported and no strict equal-capacity causal claim is made. The control
is the **repaired** shared-selector N2, never the pre-repair one. Acceptance
requires that binding every depth's selector to one vector reproduces the shared
selector forward output, that perturbing one depth does not rewrite another, that
each depth receives gradient, and that serialisation round-trips the
`(trees, depth, features)` shape. Budget if separately enabled: 4 coarse slots,
then 6 extension slots only if the same path is positive in both seeds with a
mean ≥0.005 and a positive increment against the control. It does not mix with
path A and does not redefine the original full gate.

## 8. Workspace collision record

During this round a second session was active in the same checkout:

* it committed the core repair files (`v4_2_rng.py`, the N/train/screen/package
  changes and the test updates) as `96d1f85` on `round2-v4.2-structure-search`,
  byte-identical to the files produced here, and then began a V4.4 round;
* it created `local/runs/round2-v4.2-structure-search/pinned-r2/` and
  `full-r2/` and overwrote `n2-seed-init-v2/candidate_identity.json` with an
  identity that records `torch_threads: null` instead of the run's actual `4`;
* the run ledger, the ten prediction files, the stage summaries and
  `followup_manifest.json` in `n2-seed-init-v2/` remain the repaired run's own
  artifacts, and the manifest's `identity_digest` matches the ledger rows.

Consequence: the on-disk `candidate_identity.json` in that directory must not be
hashed as the run identity; the authoritative identity is the manifest digest
plus the ledger rows. The evidence was copied to a collision-safe sibling
directory, and nothing was deleted or overwritten.

## 9. Guardrails held

* Only the round-two V2 snapshot was read; no external data, no pretrained
  weights, no test labels, no ID/row-order features, no per-row editing.
* Candidate and `B_fit` were always fitted on the same `T` and scored on the
  same `V`; `B_replay` was never used as a substitute.
* Predictions, models, ledgers and reports stay under the git-ignored `local/`
  tree; no private artifact entered the repository.
* The frozen V3.6 environment was not upgraded.
* No package was produced, no upload was performed, and no platform quota was
  assumed.

## 10. Appendix: unchanged-column preservation fix in `v4_2_package.py`

The iron-only packaging helper read the parent package with `pd.read_csv` and
then rebuilt the unchanged time column with `str(v)`. Measured on the real
parent (`V36_USER_REQUESTED_OUTER_FAILED/result.csv`): **136/322 field strings
differ** from the parent text, 74 of them real float64 differences with a maximum
of `2.84e-14` (about 2 ULP), while the manifest claimed
`time_column_preserved_byte_for_byte: true`.

Fix: `_read_parent_csv` takes the field text straight from the CSV with the
`csv` module (no float round-trip) and `_align_parent_columns` reorders onto the
test IDs without retouching it. Re-measured with the reader: **0/322 strings
differ**. The fix is demonstrated against the real parent and covered by
`tests/test_round2_v4_2_packaging_fix.py` (4 tests, synthetic parents only).

Scope: this corrects the *attribution/contract* defect for future builds. It does
not rewrite the already-delivered `V42_IRON_N2_Q25` ZIP
(`7fdfec5b…`) or its platform result `96.2533`, and it changes no model, blend or
score. The platform conclusion recorded in `FEEDBACK.md` — that the local
`+0.0144` N2 gain did not transfer (`−0.0201` vs the current V36 best) — is
unaffected.

## 11. User-requested corrected exploratory package

At the user's explicit request (2026-09-25) one exploratory package was generated
outside the pre-registered A3/A4 chain, which the failed full-coverage gate would
have stopped:

| item | value |
|---|---|
| candidate | `V42R2_IRON_N2_Q25` |
| recipe | parent V36 `tap_iron` + `0.25 ×` full-data repaired `N-N2-tap_iron-SEED_INIT_V2`; `tap_time_len` untouched |
| alpha basis | pre-declared screening weight; A3 inner-OOF selection never ran |
| status | `USER_REQUESTED_EXPLORATORY_PACKAGE_FUSION_GATE_NOT_MET_REPAIRED_R2` |
| private path | `local/runs/round2-v4.2-structure-search/release-r1/V42R2_IRON_N2_Q25` |
| `result.csv` SHA-256 | `2c7092fbe02f428c9f135174b05817a8e69beeb3f44597c07276a0a1249d0adc` |
| ZIP SHA-256 | `c6e09a4aaf64f66e1d7b6af10045c976b79fec74e4b86e900a72d54213e0208c` |
| init hash | `7e27ed9933201404ea36501d89b0216293a9ae89d48073cf9c8cfc755572fc30` |
| B36 reproduction | passed, max rel `1.39e-16` |
| cold consistency | passed at `atol=1e-4`, chunked max diff `6.41e-05` |
| parent time column | **0/322** strings differ (the prior package had 136/322) |

Honest limits: the iron column is numerically the same blend as the
already-tested `V42_IRON_N2_Q25` (max absolute difference `2.3e-13`), so the
expected platform outcome is essentially the reported `96.2533`, which is
`−0.0201` against the current best `96.2734`. Generating this artifact corrects
the unchanged-column contract and binds the repaired identity and gate record; it
is **not** a promotion, does not reclassify the negative route evidence, and was
not uploaded or copied to the desktop.
