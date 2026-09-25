# Round2 V4.4 mechanism-completion results — N line

Date: 2026-09-25
Branch: `round2-v4.2-n2-seed-repair` (HEAD `96d1f85`)
Private evidence: `local/runs/round2-v4.4-mechanism-completion/`
Status: **N-line and R-line coarse screens complete. Only the repaired N2
control passes the coarse gate, and only on `tap_iron`; it then fails the
full-coverage fusion gate on complete coverage. Per-depth selection (N4), the
ODST core (N5) and every R unit (R2/R4/R5, both targets) fail the coarse gate.
No package, no upload.**

> **Concurrency caveat.** A second agent session was active in this same
> checkout throughout this round. It committed `96d1f85` at 21:29:13, which
> swept up this session's then-uncommitted `v4_2_n_node.py` mechanism edits
> (so `HEAD` already contains the N4/N5 code), and it completed its own V4.2-r2
> N2 repair run in `local/runs/round2-v4.2-structure-search/n2-seed-init-v2/`
> (documented in `docs/round2_v4_2/FOLLOWUP_RESULTS.md`). It was still running
> at the end of this round: it wrote replication output into `repl-t3407-r2/`
> and a desktop delivery receipt for `V42_IRON_N2_Q25` at 22:08:03, so the two
> sessions were sharing the checkout while both were training. This session
> overwrote one file in that directory (`candidate_identity.json`) before the
> collision was known; see section 7. Nothing was deleted, and no run directory
> belonging to the other session was modified except that one file. No commit or
> push was made by this session, because committing contested files while
> another writer is active would capture a mixed state.

## 1. Source verification of the round's premises

Checked against the authors' primary sources, not against the task book alone:

| claim | verdict | evidence |
|---|---|---|
| the in-repo tree reuses one projection at every depth | **confirmed** | `v4_2_n_node._ObliviousLayer.feature_logits` was `(trees, features)`; `routing` broadcast one scalar per tree across all depths |
| the authors' ODST carries a depth axis | **confirmed** | `Qwicen/node lib/odst.py`: `feature_selection_logits` is `(in_features, num_trees, depth)`, plus `log_temperatures` `(num_trees, depth)` and a data-aware `initialize()` |
| the in-repo model adds neighbour-label means to the final scalar | **confirmed** | `v4_2_r_tabr` returned `head(cat[q, agg]) + weights @ labels` |
| the authors encode labels and transform key differences | **confirmed** | `tabular-dl-tabr bin/tabr.py`: `k = K(x)`; `values = E_y(y_j) + T(k(x) - k(x_j))`; `x = x + Σ probs·values`; then `blocks1` + head |
| the `+0.02` gate was applied before full coverage existed | **confirmed and already corrected** | `RESULTS.md` §6; `SEARCH_SPEC.yaml` has `continuation_gate(0.005)` → `full_coverage` → `fusion_eligibility_gate(0.02)` |
| N2 is `raw`, not PLE | **confirmed** | `N_RECIPES`, the fit ledger, all record `numeric_encoding: raw` |

**One correction to the round's arithmetic.** The frozen target is `> 96.30`
(`EVIDENCE_STATUS.json`, `docs/round2_v3_4/`), not 96.35. Against 96.2734 the
required reduction is `0.0266/50 = 0.000532` in `W_I + W_T`, i.e. mean-WMAPE
`3.7266% → 3.6734%`, a **~0.71%** relative reduction. The 2.06% figure follows
from a 96.35 anchor, which the repository does not record. The frozen anchor was
used for this round.

**One stale premise.** N2 full coverage had already been run before the repair:
`local/runs/round2-v4.2-structure-search/full-r1` reaches `+0.014213` over ten
cells with both split seeds positive. It still fails the `+0.02` fusion gate, so
the line legitimately stops there — at the correct gate. Because its
initialisation seed was not controlled, it is not the V4.4 control and its four
pre-repair coarse cells were not merged with repaired cells.

## 2. What was implemented

| item | change |
|---|---|
| N4 (attribution control) | selector restored to `(trees, depth, features)`; every other N2 property unchanged (raw encoding, 2×64 trees, depth 4, fixed temperature, threshold init, MAE, AdamW, early stopping) |
| N5 (ODST core) | N4 plus a learnable per-tree/per-depth log-temperature and the authors' data-aware threshold/temperature initialisation on the layer's own input |
| R repair | the R line had the *same* init-before-seed defect as the old N line; both stages now build inside `torch_seed_context` with a separate registered batch-order stream |
| R4 (authors' fusion) | separate key projection, label encoder, query-minus-neighbour difference transform, fusion into the internal representation, then predictor blocks and head |
| R5 (architecture control) | byte-identical architecture and parameter count to R4, retrieval channel off |
| V4.4 pre-registration | `configs/round2_v4_4/SEARCH_SPEC.yaml` (12 units, 48 slots) and `FOLLOWUP_SPEC.yaml`; frozen gates copied verbatim |
| spec loader | schema-aware (`v4.2`, `v4.4`); unit/slot counts derived from the declared recipe table, so v4.2 still derives 24/96 |

True parameter counts: **N2 9,881, N4 31,385, N5 31,897**; **R2 52,738, R4
136,578, R5 136,578** (R4 and R5 are exactly equal, as the control requires).

Acceptance tests: `tests/test_round2_v4_4_n_depth.py` (12) and
`tests/test_round2_v4_4_r_fusion.py` (9), all checking the properties
numerically. The shared-selector equivalence, per-depth independence, non-zero
per-depth gradient, serialisation round-trip, the equal-parameter R control, the
manual recomputation of the fusion formula, and an excluded-neighbour leak test
are all present.

## 3. N-line coarse screen (folds 0/1, split seeds 42/3407, `fixed_quarter`)

| unit | seed 42 | seed 3407 | mean | both seeds positive | coarse gate |
|---|---|---|---|---|---|
| N2 control / `tap_iron` | +0.007660 | +0.010348 | **+0.009006** | yes | **PASS** |
| N2 control / `tap_time_len` | −0.008060 | −0.017640 | −0.012850 | no | FAIL |
| N4 depth-only / `tap_iron` | +0.002451 | −0.000572 | +0.000939 | no | FAIL |
| N4 depth-only / `tap_time_len` | +0.004670 | −0.006540 | −0.000930 | no | FAIL |
| N5 ODST core / `tap_iron` | −0.006000 | −0.000220 | −0.003110 | no | FAIL |
| N5 ODST core / `tap_time_len` | −0.003960 | +0.013130 | +0.004590 | no | FAIL |

All 24 coarse cells completed. Every unit's `direct` path is strongly negative
(−0.13 to −0.32), so the whole signal lives in the fixed 25% blend, exactly as in
the V4.2 screen. No unit's `direct` path passes on either target.

**Reading.** Restoring the depth axis did not produce a gate-passing gain on
either target. N4 — the clean attribution test, with no learnable scale and no
data-aware init — moved `tap_iron` by `+0.00094` with the seeds disagreeing in
sign, and `tap_time_len` by `−0.00093`. Adding the ODST core on top (N5) was
worse on `tap_iron` (`−0.00311`) and did not stabilise `tap_time_len`. The
capacity increase (`3.2×` the selector parameters) did not buy a gain, so this is
not a capacity story.

**Fold structure.** The fixed-quarter signal is concentrated in fold 0 and is
negative in fold 1 for almost every unit, including the control (`+0.0249 /
+0.0214` on fold 0 versus `−0.0094 / −0.0008` on fold 1). The coarse gate's
"both seeds positive" rule is satisfied by the control only because fold 0
dominates the two-fold mean. That is a property of the gate, not of the
mechanism, and it should temper any reading of the coarse screen.

## 4. N-control full coverage (ten cells)

The repaired N2 control passed the coarse gate on `tap_iron`, so folds 2/3/4 were
added:

| path | seed 42 | seed 3407 | mean | positive folds |
|---|---|---|---|---|
| `fixed_quarter` | +0.013874 | +0.015993 | **+0.014934** | 8/10 |
| `direct` | — | — | negative | 0/10 |

Per-fold `fixed_quarter`: seed 42 `+0.02486, −0.00937, +0.01832, +0.00741,
+0.02791`; seed 3407 `+0.02142, −0.00076, +0.01848, +0.02188, +0.01888`.

Stage states: coarse **PASSED** → full coverage **FAILED** (`+0.014934 < +0.02`,
judged only now that all ten cells exist) → fusion `NOT_EVALUATED` → no alpha
scan, no replication, no package. This is the honest sequence the round asked
for: the candidate reached the stage it had earned and was eliminated there.

For comparison, the pre-repair extension scored `+0.014213` over the same ten
cells with the same sign pattern. The repair moved the number by `+0.0007` and
did not change the verdict, which is a useful reproducibility signal rather than
a new result.

## 4b. Platform cross-check of the N2 control, and what it implies for N4/N5

While this round was running, the concurrent session's exploratory package
`V42_IRON_N2_Q25` (parent V36 `tap_iron` + 0.25 × the V4.2 `N-N2` model) returned
a user-reported platform score of **96.2533**, i.e. **−0.0201 against the
incumbent V36 `96.2734`** (`docs/round2_v4_2/FEEDBACK.md`, committed as
`2dd5772`). V36 remains the platform best.

| evidence for the same N2 quarter-blend | protocol | mean gain | positive folds |
|---|---|---:|---|
| `n2-seed-init-v2` (concurrent session, `torch_threads=4`) | 2 split seeds × 5 folds | +0.014374 | 8/10 |
| this round under the V4.4 spec (`torch_threads=2`) | 2 split seeds × 5 folds | +0.014934 | 8/10 |
| **platform, relative to V36** | 322 test rows | **−0.0201** | — |

Three readings matter for V4.4:

1. **The two independent local estimates agree** (`+0.0144` vs `+0.0149`, same
   8/10 sign pattern) under different thread counts. The N-Control measurement is
   solid; it is the transfer that fails.
2. **The local gain did not reach the platform.** A `+0.014` local development
   gain became a `−0.020` platform loss against the same incumbent, a gap of
   roughly `0.034`. That is consistent with the pre-registered `+0.02` fusion
   gate having failed: the gate was doing real work, not being conservative.
3. **This makes the N4/N5 negative result robust.** N4 and N5 did not clear the
   coarse `+0.005` continuation gate on either target, so they never even
   produced a `+0.01`-class local gain. Given the observed transfer gap, a
   `+0.01` local gain would in any case be far too small to move the platform
   score. The conclusion does not depend on resolving the transfer question.

The V4.2 feedback document recommends, as the next structural hypothesis, the
`N2_DEPTHSELECT` design that its `FOLLOWUP_SPEC.yaml` declared but left
`enabled: false` — "rather than continuing small weight tweaks on the existing
package". **That design is exactly N4 in this round, and it has now been tested
for the first time: it fails the coarse continuation gate on both targets
(`+0.00094` on `tap_iron`, `−0.00093` on `tap_time_len`, seeds disagreeing in
sign).** The recommended next step has therefore been answered, and it does not
rescue the oblivious-tree direction.

## 5. Two defects found and fixed

**(a) R retrieval distance was batch-shape dependent.** The expanded squared
distance `|q|² + |k|² − 2q·k` computes an exact-duplicate or self match as a
cancellation residual whose sign follows the BLAS tiling. `clamp_min(1e-8)`
therefore floored the same pair to different distances in a batch than alone, and
because the top-1 self weight is ≈1 while labels are of order 60, the prediction
moved by ~`2e-3`. `tests/test_round2_v4_2.py::test_r_chunked_and_single_row_inference_agree`
failed under the repaired initialisation because of this. The numerically-zero
band is now collapsed to exactly zero, making the distance identical for every
batch composition, with a regression test.

**(b) The routing-entropy diagnostic produced `NaN` at saturation.**
`1 - 1e-9` is exactly `1.0` in float32, so the upper clamp left `p == 1` and
`(1 - p) * log(1 - p)` evaluated to `0 * -inf = NaN`. A learnable routing scale
saturates far more often, so six of the eight N5 fits recorded a `NaN`
layer-2 entropy. This is **diagnostics only** — predictions, scores and gates are
unaffected — but it silently disables the collapse warning (`nan < 0.05` is
`False`). `binary_routing_entropy` now clamps to a float32-representable margin
and evaluates in float64.

> Reproducibility note: fix (b) landed in `v4_2_n_node.py` *after* the N-line runs
> finished, so the `source_digest` recorded in their ledgers predates it. Model
> predictions and all reported scores would reproduce; only the
> `routing_diagnostics` field would differ (finite instead of `NaN`). Any
> promotion-tracked candidate must be re-run from the current source under a
> fresh identity.

## 6. R-line coarse screen (folds 0/1, split seeds 42/3407, `fixed_quarter`)

All 24 R cells completed. Private evidence:
`local/runs/round2-v4.4-mechanism-completion/r-coarse-r2/`.

| unit | seed 42 | seed 3407 | mean | direct mean | coarse gate |
|---|---|---|---|---|---|
| R2 control / `tap_iron` | −0.03954 | −0.06842 | −0.05398 | −0.60442 | FAIL |
| R2 control / `tap_time_len` | −0.05628 | −0.09445 | −0.07537 | −0.85134 | FAIL |
| R4 authors' fusion / `tap_iron` | −0.02855 | −0.05484 | −0.04169 | −0.59552 | FAIL |
| R4 authors' fusion / `tap_time_len` | −0.07693 | −0.09733 | −0.08713 | −0.95393 | FAIL |
| R5 no-retrieval control / `tap_iron` | −0.06046 | −0.05819 | −0.05933 | −0.75361 | FAIL |
| R5 no-retrieval control / `tap_time_len` | −0.09362 | −0.08808 | −0.09085 | −1.10679 | FAIL |

**Reading.** The complete authors' mechanism does *something*, and the control
separates what:

* **retrieval channel**: R4 versus R5 (identical architecture and parameter
  count, retrieval switched off) is `+0.01763` on `tap_iron` and `+0.00372` on
  `tap_time_len`. So the retrieval channel is not inert — the earlier R3
  negative result is not explained by "retrieval did nothing".
* **fusion mechanism**: R4 versus the adapted control R2 is `+0.01229` on
  `tap_iron` but `−0.01176` on `tap_time_len`. The improvement does not
  reproduce across targets, so it does not support a general claim that the
  authors' fusion is better on this data.
* **magnitude**: every unit is negative by 0.04–0.09 on the fixed-quarter path
  and 0.60–1.11 on the direct path. The gate threshold is `±0.005`, so no unit
  is close. Restoring the missing mechanism moves the R line's position without
  making it competitive, which is the direct answer to the question the round
  asked.

No R unit reached full coverage, so no R fusion, replication or packaging was
performed.

## 6b. Aborted R attempt and the cost fix it exposed

The first R attempt (`r-coarse/`, 8 of 24 cells, preserved) did not finish in
~20 minutes. The cause was in this session's first version of the fusion: it
built the `(B, S, W)` query-minus-neighbour tensor over **every** support row
(≈290 MB per intermediate at `S = 2203`). The authors compute the neighbourhood
values only over the retrieved context. Because `attention` gives exactly zero
weight outside the kept set, gathering the non-zero columns is exact, not an
approximation: the existing hand-computed fusion test still passes unchanged and
the fit cost drops by roughly the support-to-context ratio. The aborted cells
were not merged; the screen was re-run from scratch so all 24 cells share one
source identity.

## 7. What was not done

* No R unit passed the coarse gate, so no R full coverage, fusion alpha
  selection, replication seed, package or cold audit was performed.
* No package was produced from any V4.4 mechanism, no upload was performed, and
  no platform quota was assumed. The N4/N5/R4/R5 results do not meet the local
  working gate and are not platform candidates.

## 8. Collision record (this session's side)

* This session launched a repaired N2 run into the other session's output
  directory `local/runs/round2-v4.2-structure-search/n2-seed-init-v2/` and killed
  it after its `source_digest` became unreproducible. That is what overwrote
  `candidate_identity.json` and appended four coarse rows; the other session
  detected this and recorded it in `FOLLOWUP_RESULTS.md` §8. The directory now
  holds that session's own ten-fit run.
* `96d1f85` (authored by the other session) contains this session's N4/N5
  `v4_2_n_node.py` edits, because they were uncommitted in the shared working
  tree when that commit was made.
* This session's own evidence lives only in
  `local/runs/round2-v4.4-mechanism-completion/`, one directory per unit, with
  `-V4_4` trial ids, and was not written into any of the other session's paths.
* The other session has since pushed its work (`origin/round2-v4.2-structure-search`
  is at `2dd5772`) and stopped writing. It left the follow-up runner it authored
  (`src/bf_tap_r2/v4_2_followup.py`) plus `configs/round2_v4_2/FOLLOWUP_SPEC.yaml`,
  `docs/round2_v4_2/FOLLOWUP_RESULTS.md` and `tests/test_round2_v4_2_followup.py`
  uncommitted. **No commit or push was made by this session**, on the user's
  instruction to run only the R-line fits without committing while the checkout
  was contested; the commit decision is still open.

### Reproduction prerequisites (open item)

The N-line full-coverage stage was executed through the V4.2-r2 follow-up runner
`bf_tap_r2.v4_2_followup`, which the concurrent session authored and left
untracked. This round's edits to it are the schema-aware trial-id suffix and the
`v4.4-r1` declarations loader. Until that file (and, for the full test set,
`configs/round2_v4_2/FOLLOWUP_SPEC.yaml` and its test) is committed, the N-line
evidence in section 3–4 can be inspected but not re-executed from the public
repository. The R-line evidence has no such dependency: it used
`bf_tap_r2.v4_2_screen`, which is committed.

## 9. Guardrails held

* Only the round-two V2 snapshot was read; no external data, no pretrained
  weights, no test labels, no ID/row-order features, no per-row prediction
  editing.
* Candidate and `B_fit` were fitted on the same `T` and scored on the same `V`;
  `B_replay` was never substituted.
* All models, predictions, ledgers and reports stay under the git-ignored
  `local/` tree.
* The frozen V3.6 environment was not upgraded.
* No package was produced, no upload was performed, and no platform quota was
  assumed.
