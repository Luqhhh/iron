# Round2 V4.2 coarse-screen results

Date: 2026-09-25
Branch: `round2-v4.1-strong-increment`
Private evidence: `local/runs/round2-v4.2-structure-search/coarse-r1`
Status: **coarse screen executed; two units pass the continuation gate on the
fixed-quarter path; no promotion, no package, no upload**

## 1. What ran

The 96 pre-registered outer recipe slots — 24 `(line, recipe, target)` units ×
split seeds 42/3407 × outer folds 0/1 — all completed:

| item | value |
|---|---|
| outer recipe slots scheduled / completed | 96 / 96 |
| failed slots | 0 |
| wall clock | 2299.8 s (~38 min) |
| worker start method | `spawn` (see `WORKER_START_METHOD.md`) |
| B_fit baselines reused from the identity-checked cache | 4 |
| platform uploads / packages | 0 / 0 |

Median outer-fit wall time: R 36.2 s, S 35.8 s, N 217.8 s. Every slot is in the
private fit ledger with its requested and effective parameters, stopping reason,
fit-row ID hash, group hash and transformation hash.

## 2. Reference separation

`B_replay` (the frozen `V36DevelopmentReference`) reconstructs to a mean package
score of `96.20376256899249` against the recorded `96.20376256899247`, a replay
gap of `1.4e-14`. It is a diagnostic only.

`B_fit` (the same fixed V3.6 recipe refit inside each outer training part) is the
formal reference. Its coarse-fold package score is **96.287377** for split seed
42 and **96.154992** for split seed 3407. These are deliberately not equal to the
historical replay value and are reported as-is; no score was patched.

## 3. Screen outcome

Score is `S = max(0, 100 - 50 * (W_I + W_T))`, recomputed on the combined
prediction; per split seed the WMAPE numerators and denominators are pooled over
the covered samples before the two seed scores are averaged.

Only two of the 24 units pass the pre-registered continuation gate, and both do
so **only** through the `fixed_quarter` path:

| trial | path | seed 42 gain | seed 3407 gain | mean gain | direct-path mean gain |
|---|---|---|---|---|---|
| `N-N2-tap_iron` | fixed_quarter | +0.006867 | +0.011764 | **+0.009316** | −0.126807 |
| `N-N0-tap_iron` | fixed_quarter | +0.006177 | +0.009965 | **+0.008071** | −0.137873 |

Both satisfy the rule as written: one and the same pre-declared path is positive
in both split seeds, and its mean full-package gain is above `+0.005`. Every
other unit fails; the closest miss is `N-N1-tap_iron` at a fixed-quarter mean gain
of `−0.002834` (positive in seed 3407, negative in seed 42).

Because at most one recipe per family and target may advance, the single
finalist is **`N-N2-tap_iron`** (2-layer raw-standardised differentiable oblivious
tree ensemble, only the `tap_iron` column replaced at weight 0.25).

> **Correction (2026-09-25, V4.2-r2 review).** Earlier revisions of this file
> described `N-N2-tap_iron` as PLE-encoded. That was a documentation error:
> `SEARCH_SPEC.yaml`, `v4_2_n_node.N_RECIPES` and the private fit ledger all
> record `numeric_encoding: raw` for N2 (N3 is the PLE two-layer recipe), with
> `n_features_numeric = 21` and `n_features_categorical = 3` and transform hash
> `35c31efc563928175f33f31e45a6e8959176832fc014be017a8b7c9c4fa38bca`. Only the
> wording is corrected here; no score was recomputed or altered.

The full ranking is in `coarse_summary.csv`; the complete per-unit record,
including both paths' per-seed gains, is in `coarse_summary.json`.

## 4. Mechanism evidence actually recorded

### R line (retrieval)

All 32 R slots stopped by early stopping. The audits are uniformly clean:

* `self_retrieval_count = 0` for every slot;
* `max_illegal_attention_weight = 0.0` for every slot, so no support row sharing
  the query's exact-feature group was ever attended to;
* retrieval gradients are non-zero for R1/R2/R3 and zero for the R0 control, as
  designed;
* parameter counts: R0 52737, R1/R2 52738 (the single learnable retrieval
  temperature), R3 71554.

The R line's own predictions are far behind `B_fit` on the direct path
(R3 iron mean gain `−0.348`), so the retrieval mechanism produced no standalone
win, and its quarter-blend contribution (R3 iron `−0.0197`) is negative in both
seeds.

### S line (symbolic regression)

All 32 S slots ran on the **PySR** backend, `fallback_used = false` everywhere,
with the exact `max_evals = 200000` budget saturated in every slot. Selected
expressions have 9–23 nodes (limit 24) and every slot re-estimated its constants
on the complete training part. No expression was rejected as non-finite at
prediction time.

The S line is the weakest of the three: its best unit, `S-S1-tap_iron`, has a
fixed-quarter mean gain of `−0.2165`. A single short expression does not reach
the level of the frozen V3.6 ensemble on this data.

### N line (differentiable oblivious trees)

All 32 N slots stopped by early stopping (none was `budget_limited`). The
diagnostics show the ensembles genuinely route:

* routing entropy 0.9387–0.9571 (normalised to [0, 1]);
* used-leaf fraction 0.8359–1.0;
* zero `collapse_warning` across all 32 slots.

True parameter counts differ across the four recipes, as the task book requires
them to be reported rather than matched: N0 5785, N2 9881, N1 24748, N3 28844
(per-tree 45.2 / 77.2 / 193.3 / 225.3). Peak Python-allocator memory during
training was 0.94–0.99 MB for N; the worker process high-water mark was
0.85–0.86 GB.

## 5. What this does and does not show

Does show: with 2 layers and raw-standardised numerics, a differentiable
oblivious tree ensemble that is clearly worse than `B_fit` on its own can still
carry a small, consistent positive contribution when given a fixed 0.25 weight on
the iron target. The effect is small (`+0.0093` mean package gain on folds 0/1).

Does **not** show: any platform gain, and — as originally reported — not a
full-coverage or fusion result. At the time of the coarse screen only folds 0/1
existed, so the full-coverage gate was not evaluated at all; the inner fusion
selection and the training-randomness replication had not been run, and nothing
was packaged or uploaded.

## 6. Pre-registered stopping point

The pre-registration separates two different gates:

* the **coarse continuation gate** — one and the same pre-declared path positive
  in both split seeds with a mean full-package gain of at least `+0.005` on folds
  0/1. `N-N2-tap_iron` met this at `+0.0093` on `fixed_quarter`;
* the **full-coverage fusion gate** — after folds 2/3/4 are added, a mean gain of
  at least `+0.02`, both split seeds positive and at least 8/10 positive folds.
  This gate may only be judged on the complete ten cells.

> **Correction (2026-09-25, V4.2-r2 review).** The original text of this section
> applied the `+0.02` full-coverage gate to stop *before* folds 2/3/4 were run,
> and recorded that as a gate failure. That is inconsistent with
> `configs/round2_v4_2/SEARCH_SPEC.yaml`: with only folds 0/1 the full-coverage
> gate state is `NOT_EVALUATED_INCOMPLETE_COVERAGE`, never a failure and never a
> pass. The original wording is preserved below as the historical record; the
> 0.005 coarse gate was met, so full coverage was the permitted next step.
> `bf_tap_r2.v4_2_screen._coverage_gate` now returns exactly those three states.

<details>
<summary>Original section 6 text (historical record, superseded)</summary>

The next stage — folds 2/3/4 coverage and inner fusion selection — is gated on a
unit reaching a mean gain of `+0.02` with both seeds positive and at least 8/10
positive folds on complete coverage. The best coarse unit is at `+0.0093`, less
than half of that. Per the frozen stop rule, **the line stops here and no alpha
rescue scan is performed**; folds 2/3/4, the inner nested fusion and the
seed-3407 replication were deliberately not started.

</details>

The subsequent repair round re-ran the candidate under an explicit
initialisation-seed protocol and completed folds 2/3/4; the outcomes, the
repaired-coverage gate state and the preserved pre-repair diagnostics are in
`docs/round2_v4_2/FOLLOWUP_RESULTS.md`.

## 7. Guardrails held

* Only the round-two V2 snapshot was read; no external data, no pretrained
  weights, no test labels, no ID- or row-order-derived features, no per-row
  prediction editing.
* The `B_fit` and candidate models were always fitted on the same `T` and scored
  on the same `V`; the replay gap is reported, never patched.
* R support sets and every model, prediction, OOF vector and ledger stay under
  the git-ignored `local/` tree; nothing private entered the repository.
* The frozen V3.6 environment was not upgraded; PySR lives in its own interpreter
  and Julia depot.
* No submission package was produced, no upload was performed, and no remaining
  platform quota was assumed.
