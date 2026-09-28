# V27: deep kernels against the measured B0 platform incumbent

Frozen before V27 model evaluation, 2026-09-28; base `1f1b426`. The user
corrected the target to **96.5**, then explicitly instructed the new experiment
to begin. B0 (`V18_B0_V12IRON_V7TIME`) is the user-reported platform best at
96.3679, leaving 0.1321 points. This round does not promise that gap is attainable.

## Hypothesis and scope

Jointly learning an input representation and a GP covariance may improve the
accuracy/diversity tradeoff over the recorded fixed or importance-weighted
RBF/polynomial kernels. Three frozen recipes, separately on both targets:

1. `GP_ARD`: standardized numerical inputs and one-hot spout, ARD Matern-5/2 GP.
2. `DKL_RAW`: small 64/64 SiLU network to eight coordinates, the same GP.
3. `DKL_PLR`: learned periodic numerical embeddings before the same network/GP.

All fit from scratch using only the authorized V2 training snapshot. No temporal
features, IDs as predictors, external data, pretrained weights, residual
corrector or old capacity scan is introduced. The GP control is a mechanism
control; changing a kernel-regression solver alone is not claimed as novelty.

The implementation uses the collapsed sparse variational GP bound (128 inducing
training rows; fixed input coordinates, shared learned encoder), rather than
claiming exact full GP inference or reproducing every detail of the original DKL
paper. Kernel/noise/encoder parameters are learned. All numerical choices and
cost tie ordering are frozen in `configs/round2_v27/SPEC.yaml`.

Method sources: [Wilson et al., Deep Kernel Learning](https://proceedings.mlr.press/v51/wilson16.html),
[Titsias, Variational Learning of Inducing Variables](https://proceedings.mlr.press/v5/titsias09a.html).
Their results are not evidence of a gain on this competition.

## Reference and validation

The actual frozen deployment composition is refitted, with unchanged component
recipes and weights:

`B0_I = .5 V36_I + .5 V12_joint_I`

`B0_T = .325 V36_T + .175 N0048_T + .5 V7_periodic_T`.

This same factory is used inside calibration and for the outer reference.
Historical development caches mix development/release composition conventions;
they remain descriptive comparisons, not replacements for a matching refit.
The factory entails 21 recovered A base pipeline fits, four A experts, four V36
experts and three new endpoints, i.e. 32 pipeline fits per call. Some pipelines
include internal early stopping, bags or fresh refits; 32 is not an optimizer-run
count. Query labels are removed before any predictor receives a query frame.

For every outer split/fold:

1. Split the outer training rows into an 80% fitting part and a 20% calibration
   part using the existing group-safe five-fold helper, seed 27001, held fold 0.
2. Fit B0 on the fitting part only and predict calibration rows. This reference
   is shared across all three candidate recipes and both targets.
3. Fit each candidate's preprocessors/inducing inputs/target scaler on the
   fitting part only. Select its epoch by calibration MAE; select its blend
   weight from the frozen eight-point grid on these calibration predictions.
   Reusing calibration for these choices is permitted; outer labels remain
   untouched. This is an inner holdout, not full inner cross-validation.
4. Freshly initialize and fit the candidate on all outer training rows for the
   selected epoch count; refit B0 on exactly those rows. Apply the already
   chosen weight to outer predictions, then score pooled complete OOF columns.

Changing outer labels must leave epoch/weight/predictions unchanged. A new split
seed alone does not provide new labels. In particular, another seed's whole-data
OOF predictions may not select weights for the held-out rows in this protocol.

## Budget, admission and stopping

- Development: seeds 42/3407, five folds, three recipes and two targets: **60
  candidate outer fits / 120 candidate optimizer runs**.
- Reference development: **20 B0 factory calls / 640 component pipeline fits**
  (one calibration plus one outer refit per cell); separately recorded and shared.
- After two positive complete development seeds, at most one recipe per target
  proceeds, selected by mean incremental gain and the frozen cost order.
- Confirmation uses predeclared split seeds **271828/314159**, not the repeatedly
  used 7777/12011. It costs at most **20 candidate outer fits / 40 optimizer runs**
  plus **20 B0 factory calls / 640 pipeline fits** shared across finalists.
  Seeds are only consumed after a finalist is frozen. Same labels are reused, so
  confirmation still does not create independent datasets.
- Preserve the four-seed positive-gain, positive paired seed LCB and 96.25
  development working gate. Fold results are descriptive. Existing
  `candidate_tiers` classification is also emitted as a historical development
  diagnostic; it cannot override the later four-seed rule. At most one exploration
  recommendation across targets, and no automatic release.
- Gains around +0.01 or larger receive strategic priority; this is not an added
  acceptance threshold or platform forecast. Failed recipes are not rescued by
  an unregistered grid, extra epochs, training seeds or increased inducing rank.
- No full-data fit, package, desktop write or upload is budgeted in this round.

## Engineering and evidence

Pin BLAS/OMP/MKL/NUMEXPR to one thread, plus torch to one thread. Use the locked
Python 3.12 environment. Validate the sparse algebra against an independently
formed dense covariance on synthetic data, gradient flow, training-only
preprocessing and selection, model persistence and batch/order-invariant
predictions. Audit real prediction files and recompute metrics independently.

Private append-only evidence lives under `local/runs/round2-v27/`. Freeze input,
configuration, transitive source, reference manifest and environment digests.
Identity uses those files rather than a changing Git HEAD. Never overwrite a
run, model, completed unit or failure; resume only verified completed units.
Concurrent workers own distinct unit directories. Public code/configuration and
result summaries are committed and pushed after their checks; private evidence
stays outside Git.

## Corrections carried into this round

WMAPE is sum absolute error divided by sum absolute target; for fixed rows it
has the same ordering as MAE. Log-MSE does not inherently optimize WMAPE. V26's
negative outcome is preserved. The V18 nominal 96.3992 bound concerns the
quadrant with both weights in [.5,1], not the entire convex family; including
the declared score error yields 96.3999 for that quadrant. B0's observed additive
identity supports the metric assumption, not a second independent rectangle.
The V21 packages changed several weights at once, so their results do not
identify the causal effect of N alone. Historical decisions/packages are intact.

## Execution

The pre-existing CPU torch/tabm/embedding environment is version-checked against
the spec. Use `uv run --locked --python 3.12 --no-sync`: synchronizing the general
`round2_v4_2` extra selects a CUDA torch distribution and would change this frozen
CPU environment. The first such sync attempt failed before downloading; no
real-data fit was involved. Do not upgrade the environment to rescue a result.

With `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1 MPLCONFIGDIR=/tmp/iron-v27-mpl
UV_CACHE_DIR=/tmp/iron-uv-cache` set for each command:

```bash
uv run --locked --python 3.12 --no-sync pytest -q
uv run --locked --python 3.12 --no-sync python -u -m bf_tap_r2.v27_run --output local/runs/round2-v27/development-r1
uv run --locked --python 3.12 --no-sync python -m bf_tap_r2.v27_audit --directory local/runs/round2-v27/development-r1
```

Only an audited positive development finalist permits:

```bash
uv run --locked --python 3.12 --no-sync python -u -m bf_tap_r2.v27_run --development local/runs/round2-v27/development-r1 --output local/runs/round2-v27/confirmation-r1
uv run --locked --python 3.12 --no-sync python -m bf_tap_r2.v27_audit --directory local/runs/round2-v27/confirmation-r1
```
