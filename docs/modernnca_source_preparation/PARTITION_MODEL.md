# ModernNCA complete default encoder and partition model

The current user-reported platform best remains DE3=96.3749. Goals96.4,
96.45 and96.5 remain unmet. This preparation adds no official fit or G1 score.
The running PTaRL formal job retains priority; ModernNCA has no formal queue.

## Model and matched control

[ModernNCA implementation](../../src/bf_tap_r2/modernnca_model.py) preserves
TALENT's default zero-post-block model: lite PLR embeddings (77frequencies,
34channels, frequency scale.04431360576139521),128-dimensional linear encoder,
unsquared Euclidean distances, temperature1 and softmax-weighted neighbor
responses. The source is pinned to author commit
`1b973adffa4203c3f62c4949957b1ba3efbe60d3`; license is
[MIT](../../licenses/TALENT-MIT.txt). Computation is float64, preserving the
author's float32 initialization followed by conversion. There is no clipping
or target post-processing, and responses are interpolated within the bank's
range. Other author architectures are not implemented here.

LEARNED_ENCODER optimizes the full representation/frequencies with normalized
MSE and AdamW (lr.01, weight decay.0002). FIXED_ENCODER has the exact same
initial parameters, preprocessing and label-bank construction, with all
representation parameters frozen and zero optimizer steps. It selects epoch1
and is a control, not a second optimization candidate. This directly tests
learned distance geometry against fixed random geometry; it does not reuse
an already fitted incumbent's representation.

The original author model, embedding implementation and our complete default
network have **identical initialization, inference, training prediction,
gradients and one AdamW update**, differences0. Independently reconstructed
NumPy embedding/distance/response arithmetic differs by2.220446049250313e-16.
The artificial witness has98919 parameters with21numerical/4category inputs.
The source-equivalence witness performs2artificial optimizer steps, not an
actual-data training run. Receipt SHA-256:
`22f149a0a857a83d2c036ffa83a6f4327c8b90537b803f887f43267facccb17d`.

## Train-only protocol and independent audit

Both arms fit native NumericPreprocessor on their own fitting partition;
float32 transformed inputs are cast to float64, and spout vocabularies reserve
unknown index0. Target mean/std are fitting-only. Calibration rows never
enter the selector's neighbor bank. Inner MAE chooses the first strict-best
epoch, with max240/patience25/min_delta0, then outer models initialize fresh,
refit preprocessing/response scale on the full outer train, and train at the
selected epoch. This is a project adaptation, not a full paper benchmark.

The author adapter excludes batch indices before candidate sampling and the
core masks each own-label diagonal. Our adaptation additionally excludes all
native21-feature groups of the batch from the external pool and masks all
same-group batch-prefix responses. This intentional extension prevents
borrowed labels from duplicate feature rows. The group rule matches native
outer splitting, independent of spout; training with an empty sampled pool
refuses rather than falling back. Complete inner/outer row unions and feature/
response contents are checked before initialization.

States and private label banks are caller-SHA-anchored nonpickle NPZ files.
[Independent auditor](../../src/bf_tap_r2/modernnca_audit.py) reconstructs raw
training statistics, category codes, exact bank features/targets, default
PLR/linear geometry, response weighting and epoch selection from saved
calibration predictions. It checks fixed-control immutability, optimizer-step
counts, selected-checkpoint predictions, and query reverse/singleton behavior.
It fits or initializes no model. No cross-split-seed OOF averaging is present.

A fixed small artificial coupled partition witness saves4inner/outer states,
with2learned optimizer runs and0fixed-control optimizer runs. A fresh process
intercepts model training/initialization, preprocessor fit, linear parameter
initialization and optimizer steps, then independently audits all4models.
Maximum difference from stored-model prediction is **0**; unknown query
spout is included. This is a small witness using reduced unit-test dimensions/
epochs, explicitly not the full-size resource admission or a capacity choice.
Receipt SHA-256:
`97ff3789ce3bfa254f119570ba0d6ba936ac31c1ed424024fbde9afa818951f4`.

## Checked scope and remaining work

The locked Python3.12.12 full suite passed **1326 tests**, no skips and23existing
warnings.16new focused cases cover four-model cold inference, independent
training statistics/fresh refit, query identities/categories, duplicate-group
response gradients, invalid boundaries, corrupted/reanchored states/banks/
scales/epochs, externally held state hashes and empty-pool/one-shot refusal.
The checked803-source/config/script/test/license snapshot did not change.
Full-suite receipt SHA-256:
`5cae6cde97de7f70cd2b2afb6082a075f84cdb11d48e99c002348ef673232717`.

G0 is **partition model and independent auditor checked**, not full formal
admission. Remaining mandatory work is complete phase execution, immutable
fit ledger, audited current reference binding, full-size non-time resource
admission and exact-source candidate-pool freeze. No official reads/fits,
resource probe, full-data fit, package, desktop write or upload has occurred.
Future formal training remains serial after PTaRL's actual terminal event.

[Preparation SPEC](../../configs/modernnca/SPEC.yaml) keeps complete42/3407
five-fold development, fixed isolated20% blend, both positive seeds, mean
score gain>=.01, positive learned-minus-fixed contrast and local package>=96.25.
Only earned targets proceed7777/12011, with all4positive split gains and
positive seed-level pairedLCB95; folds are descriptive. Quality gates are
unchanged. The user instruction “不要再设置时间预算” remains controlling:
no elapsed-time cap or projected-time rejection; timings are descriptive.

## Failed first-epoch guard correction

A direct artificial failure witness found that a first-epoch empty-pool error
left zero completed epochs, allowing a second training attempt. The partition
now records attempted/completed states: a failed attempt cannot train again or
save a checkpoint. This changes no architecture, loss, hyperparameter or
numerical tolerance. The original witness and r1 receipts remain preserved.
The extended refusal test and full locked suite again pass1326 tests,0skips,
23existing warnings, with the803-file source snapshot unchanged during testing.
Current r2 receipt SHA-256:
`95fb52911c62b8b74d243fef7b633609605e299f0fbf20e0bd3e53e4d23f2367`.
The earlier fresh-process partition witness covers the pre-guard source; its
scientific core is unchanged, and current full-suite tests audit four states.
