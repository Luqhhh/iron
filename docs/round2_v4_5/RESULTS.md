# Round2 V4.5 results: the strong base's residual has magnitude, not direction

Date: 2026-09-25
Branch: `round2-v4.2-n2-seed-repair`
Pre-registration: `docs/round2_v4_5/PREREGISTRATION.md` (written before evaluation)
Private evidence: `local/runs/round2-v4.5-residual-direction/diagnostic-r1/diagnostic.json`
Status: **pre-registered rule returns `CLOSE_RESIDUAL_ROUTE` for both targets.
The conditional scale diagnostic was not escalated. No package, no upload.**

## 1. Verdict

The user's section 5 asked whether the remaining error is *directionally*
predictable or only predictable in *size*. Using the strong V3.6 `B_fit`
out-of-fold residuals (2 split seeds × 5 folds × 2 targets, verified one-fold
coverage of all 2754 rows), the answer is unambiguous:

| target | mean sign AUC | sign accuracy | Brier model | Brier constant | cells where the fitted median correction reduces MAE |
|---|---:|---:|---:|---:|---:|
| `tap_iron` | **0.5092** | 0.5038 | 0.25261 | **0.25036** | **0 / 10** |
| `tap_time_len` | **0.5070** | 0.5038 | 0.25317 | **0.25028** | **0 / 10** |

The pre-registered rule closes the route if `mean AUC <= 0.52` or fewer than 8
of 10 cells improve. Both conditions fire on both targets. Adding the other
target's base prediction as a feature does not change it (`tap_iron` AUC
0.5143, still 0/10 cells improved), so the negative does not come from a thin
feature set.

**Direction is not predictable. Only magnitude is.**

## 2. Why the negative is not a capacity artefact

Two independent readings agree, and the second needs no model capacity at all:

* the fitted logistic sign model is **worse than a calibrated constant** by
  Brier score on both targets (0.2526/0.2532 versus 0.2504/0.2503), so it is
  adding noise rather than signal;
* the residual is essentially symmetric and centred: `P(r > 0)` is 0.4938
  (`tap_iron`) and 0.4996 (`tap_time_len`), and the median residual is −0.33 and
  +0.03 against standard deviations of **26.07** and **6.23**. A near-zero
  median with a large spread is the signature of magnitude-dominated error.

The per-cell AUC range (0.481–0.538 for `tap_iron`, 0.465–0.540 for
`tap_time_len`) straddles chance with no cell far from it, so this is not one
unlucky fold.

## 3. Every pre-registered correction made held-out error worse

Mean MAE over the ten cells per target (delta = 0 is the incumbent):

| correction | `tap_iron` | vs base | `tap_time_len` | vs base |
|---|---:|---:|---:|---:|
| `delta0` (none) | **20.2145** | — | **4.9047** | — |
| `delta_const` (training-fold residual median) | 20.2371 | −0.0226 | 4.9089 | −0.0042 |
| `delta_sign` (direction only) | 24.6203 | **−4.4058** | 5.9225 | **−1.0178** |
| `delta_median` (fitted conditional median) | 20.5157 | −0.3012 | 5.0129 | −0.1082 |

The `delta_sign` row is the sharpest illustration of the whole point: acting on
the *decided sign* alone, with the training-fold median magnitude, destroys
`4.41` MAE on `tap_iron`. When direction is at chance, a direction-based
correction is not neutral — it actively randomises the prediction.

Even the calibrated constant is not stable across folds (better in only 2/10
cells on `tap_iron`, 0/10 on `tap_time_len`), which is consistent with a
conditional median that is already ≈0.

## 4. What this decides

This is the diagnostic the user said would determine "whether the next phase
continues model development, or changes training and composition". It decides in
favour of the second:

* **Do not spend further budget on residual correctors** (this also retroactively
  explains the V4.3 result: a residual corrector fitted on inner-fold residuals
  had nothing to learn, because the sign carries no information and the median
  is already ≈0);
* **Do not delete, down-weight or hand-edit samples on the basis of their error
  size.** The magnitude is predictable only in the sense that some rows are
  intrinsically noisier; that is not a licence to edit them, and the user's plan
  already forbade it;
* **the remaining headroom must come from lowering `|r|` itself** — a stronger
  point predictor, or a training/composition change — not from shifting
  predictions after the fact.

The observed local-to-platform non-transfer for the V4.2 N2 blend
(`docs/round2_v4_2/FEEDBACK.md`: local `+0.0144`, platform `−0.0201`) points the
same way: at this error level the gains being chased are smaller than the noise
in the evaluation itself.

## 5. Stage 2 was not run, by the frozen rule

Section 7 of the pre-registration escalates the training-scale diagnostic only
if `mean AUC >= 0.55` **and** at least 8/10 cells improve **and** both seeds
agree. None of those holds, so stage 2 was not started. It is not silently
skipped: there is no direction signal whose scale-sensitivity could be
meaningful.

## 6. Guardrails

* Only the round-two V2 snapshot was read; no external data, no pretrained
  weights, no test labels, no ID/row-order features, no per-row prediction
  editing.
* No base model was refitted for this diagnostic: it reuses the frozen `B_fit`
  OOF predictions, and the loader verifies one-fold coverage, the fold/NaN
  pattern and duplicate-group separation before use.
* The corrector's train and evaluation folds are disjoint; the scaler is fitted
  on training folds only; nothing is tuned on a held-out cell.
* All artifacts stay under the git-ignored `local/` tree. No package was
  produced, no upload was performed, and no platform quota was assumed.

## 7. Limitations carried forward

* The OOF residuals come from bases trained on 4/5 of `T`, while the deployed
  base sees all of `T`; the un-escalated stage 2 is exactly the test that would
  bound this, and it remains open if a future round produces a direction signal
  worth checking.
* Rows within a fold share a base model, so residuals are not independent; the
  per-cell range is reported instead of a naive standard error.
* This measures the strong V3.6 base only. It does not rule out that a different
  model class could have a more predictable residual.
