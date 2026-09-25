# V4.2 implementation plan

Date: 2026-09-25
Branch: `round2-v4.1-strong-increment`
Base commit: `4821d2eb15e7bfd3fe2f3c3728a4c83487d1185f`
Status: **implemented; coarse screen execution recorded separately; no promotion, no
packaging, no upload**

Source contract: `docs/round2_v4_2/V4_2_TASKBOOK.md`
Machine-readable form: `configs/round2_v4_2/SEARCH_SPEC.yaml`

## 1. What this round changes

V4.1 asked whether a small correction appended to an already trained B36 helps.
V4.2 asks a different question: can changing sample relations (R), tree routing
(N) and functional form (S) end-to-end produce a new strong predictor.

All three lines learn the target `y` directly. None of them is trained on the B36
residual. No third target, no output ratio, no multi-target chain and no
pseudo-labelling is introduced in this first round.

## 2. Delivered files

| file | role |
|---|---|
| `configs/round2_v4_2/SEARCH_SPEC.yaml` | declarative pre-registration read by the runner |
| `src/bf_tap_r2/v4_2_spec.py` | loads and hard-validates that specification |
| `src/bf_tap_r2/v4_2_prep.py` | target scaling, train-internal encoders, split and hash identity |
| `src/bf_tap_r2/v4_2_train.py` | shared training convention, WMAPE early stopping, bookkeeping |
| `src/bf_tap_r2/v4_2_r_tabr.py` | R line: TabR-style retrieval networks R0-R3 |
| `src/bf_tap_r2/v4_2_s_symbolic.py` | S line: constrained symbolic regression S0-S3 |
| `src/bf_tap_r2/v4_2_s_pysr.py` | S line: PySR backend adapter and equation-to-AST conversion |
| `src/bf_tap_r2/v4_2_pysr_worker.py` | standalone PySR worker run by the independent interpreter |
| `src/bf_tap_r2/v4_2_n_node.py` | N line: differentiable oblivious tree ensembles N0-N3 |
| `src/bf_tap_r2/v4_2_reference.py` | B_replay versus B_fit separation |
| `src/bf_tap_r2/v4_2_screen.py` | 96-slot coarse screen: fits, scoring, gating, ranking |
| `tests/test_round2_v4_2.py` | mechanism, isolation and screen-arithmetic tests |
| `docs/round2_v4_2/S_BACKEND_FREEZE.md` | S backend budget-control verification and freeze |

## 3. Common training convention

Every fit uses `m = mean(abs(y))` of the current training subset as the target
scale, and every metric is the original-unit WMAPE. R and N share one fixed MAE
scheme; S optimises absolute error.

The neural starting point is `AdamW`, `lr=0.001`, `weight_decay=0.0001`,
`batch_size=256`, `max_epochs=1000`, patience 50, initialisation seed 42. These
are this round's design values, not an external paper's optimum.

Every fit is two-stage and group-safe:

1. split the outer training part `T` into an inner `U` (fit) and `H` (early
   stop / formula selection);
2. determine the epoch count (R/N) or the formula (S) using `H` only;
3. refit on the complete `T` with the determined structure — the outer
   evaluation part `V` is never consulted.

A trial that reaches `max_epochs` without the stopping criterion firing is
labelled `budget_limited`, not treated as a permanent family failure.

## 4. R line

`TabRRetrievalRegressor` learns the representation, the retrieval metric and the
support aggregation jointly. It is labelled
`ADAPTED_TABR_STYLE_IMPLEMENTATION_NOT_PAPER_REPRODUCTION`: exact Euclidean
distances in the learned representation replace FAISS, and the neighbour set is
the full legal support with symmetric tie handling.

Enforced isolation:

* the support set is built from the fit part only, so `V` is never in it, and the
  inner model that early-stops on `H` uses `U` as its support;
* exclusion keys are derived from the 21 numerical values plus `spout_no`
  byte-for-byte, which is stricter than the frozen fold grouping, so a training
  query can never retrieve itself or an identical-feature duplicate;
* no legal support row raises instead of silently admitting an excluded sample;
* the support set is serialised with the model together with the source-ID and
  fit-row hashes, and may only be written under `local/`.

Audit counters (`self_retrieval_count`, `max_illegal_attention_weight`,
`retrieval_gradient_nonzero`) are recorded in the fit ledger; the self and
duplicate retrieval count must be zero and the retrieval parameters must receive
a non-zero gradient.

## 5. S line

`BoundedExpressionRegressor` searches short expressions over the standardised 21
numerical inputs and the `spout_no` indicator, with at most 24 nodes, depth 5,
no self-nesting of the nonlinear unary operators, no self-nesting of
`safe_ratio`, and no bare division, power, trigonometry or conditionals.

The two-stage protocol is exactly the task book's: search structure and initial
constants on `U`, keep at most 12 candidates of different complexity, select on
`H` by original-unit WMAPE with shorter-formula tie-breaking, then re-estimate
the selected structure's constants on the complete `T` only.

The frozen backend is PySR, verified in
`docs/round2_v4_2/S_BACKEND_FREEZE.md`. Every returned equation is converted back
into the frozen AST language so that constant re-estimation, non-finite rejection
and NumPy export share one code path. The in-repo bounded AST search remains as a
recorded fallback and cross-check; a fallback is labelled in the ledger, never
silently substituted.

Non-finite expression output is a hard failure. Only the globally frozen
non-negative output handling is inherited.

## 6. N line

`NodeEnsembleRegressor` builds oblivious trees: every node at one level shares
that level's selected feature and threshold. The differentiable feature
selection uses an exact-gradient `entmax15`, soft routing uses a sigmoid (no hard
argmax anywhere), and the leaf outputs are optimised jointly. One and two layer
variants concatenate each layer's output into the next layer's input.

The task book's tree count and depth are fixed (128 trees, depth 4). Parameter
counts are *not* equal across N0-N3; the true parameter count, per-tree count,
training volume and peak memory are reported per fit rather than claimed to
match.

Routing entropy, leaf usage and parameter updates are recorded, and a
`collapse_warning` is raised rather than hidden when the ensemble stops routing.

## 7. Reference separation

`B_replay` is the frozen `V36DevelopmentReference`. Its reconstructed mean
development score is verified against the recorded value
`96.20376256899247` and is used for regression tests and diagnostics only.

`B_fit` is `V36FixedRecipeFactory` refitted inside the current training subset.
It is this round's formal relative-increment reference. Both the new model and
`B_fit` are fitted on the same `T` and scored on the same `V`; the replay gap is
recorded side by side and never patched.

## 8. Coarse screen

24 `(line, recipe, target)` units × split seeds 42/3407 × outer folds 0/1 = 96
outer recipe slots. Each target is scored on two pre-declared paths:

```text
direct        : the new model replaces target t; the other target stays B_fit
fixed_quarter : 0.75 * B_fit_t + 0.25 * f_t; the other target stays B_fit
```

`S = max(0, 100 - 50 * (W_I + W_T))` is recomputed on the combined prediction.
Per split seed the WMAPE numerators and denominators are pooled over the covered
samples before the two seed scores are averaged. The blend weight is fixed at
0.25 and is never scanned.

Continuation gate: **one and the same** path must have a positive gain in both
split seeds and average at least `+0.005` on the full package. A `direct` gain in
one seed may not be combined with a `fixed_quarter` gain in the other. The R0
capacity control never counts as retrieval success even if it passes numerically.

At most one recipe per family and target, at most six finalists, ordered by the
better mean gain across the two paths, then the worse seed, then the fixed trial
ID. This is development screening, not independent validation.

## 9. Explicitly not done in this round

* No folds 2/3/4 full coverage, no inner fusion selection and no training-seed
  3407 replication: those stages are gated on the coarse screen producing a
  finalist, and the pre-registered gate stopped the round before them.
* No blend-weight or alpha scan.
* No candidate package, no preferred submission, no platform upload, no
  submission-quota assumption.
* No change to the frozen V3.6 environment: PySR runs in its own interpreter and
  Julia depot under `local/envs/`.

## 10. Reporting

The runner writes only under `local/runs/`. It records, per outer fit, the
requested and effective parameters, the stopping reason, the randomness control,
the fit-row ID hash, the group hash, the input/target transformation hash, the
peak memory, and the line-specific audit (retrieval audit for R, candidate
selection and budget for S, routing diagnostics for N). Aggregated results and
the ranking live in `coarse_summary.csv`, `coarse_summary.json` and
`coarse_selection.json`.
