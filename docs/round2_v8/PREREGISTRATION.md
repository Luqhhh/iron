# V8: feature-token attention from random initialization

Pre-registered before fitting, 2026-09-26. Platform objective **96.4**;
current best remains user-reported A35 **96.3366**. External pretrained weights
remain explicitly forbidden. The five desktop alpha packages are untouched.

V7's periodic TabM supplies a four-seed-qualified new direction, but does not
reach the unchanged local working gate. This round tests a different mechanism:
learned attention **between features of the same row**, not retrieval between
training samples, a residual corrector, or another raw/PLE MLP configuration.

Use the authors' [FT-Transformer implementation](https://github.com/yandex-research/rtdl-revisiting-models/blob/main/package/README.md),
[`rtdl-revisiting-models==0.0.2`](https://pypi.org/project/rtdl_revisiting_models/).
Only this 12.5 kB source-code wheel is installed with `--no-deps`; no pretrained
model or example data is downloaded and existing environment versions remain
unchanged. Record runtime versions and installed implementation hash.

Freeze two mechanisms at one compact size (96 channels, two blocks, four
heads), each evaluated on both targets. This is a compact configuration, not
a claim to reproduce the paper's default benchmark configuration.

* `ft_learned`: original learned Q/K softmax attention.
* `ft_uniform`: same initial model, tokenizer, V/output projections, FFNs,
  residuals, normalization and dropout; replace attention probabilities with
  a uniform distribution over the row's tokens. Q/K parameters remain present
  but inactive. This removes input-dependent attention weights while preserving
  the rest of the architecture. It is an attribution control, not a fully
  parameter-count-matched competing architecture.

The uniform module must equal ordinary attention with all Q/K weights and
biases zero when dropout is disabled. Learned Q/K must receive nonzero gradients;
uniform Q/K must not. Test deterministic training and row/chunk/single-row
prediction independence. No token ever represents another sample's label.

Use V7's verified inner-only preprocessing, MAE epoch selection, and fresh
full-outer-training refit; AdamW uses the author's zero-weight-decay groups
for tokenizer, normalizations and biases. Both complete five-fold seeds
42/3407 are mandatory. All 40 outer fits (80 optimizer runs) are declared
before results. No folds-0/1 screen, adaptive grid or early pool expansion.
If best epochs reach the cap, disclose budget limitation.

Primary comparison is the actual current A35 recipe. Report Q20 and the frozen
V7 candidate-pool time column (`0.5*A35 + 0.5*V7_tabm_plr001`) separately. The
latter is not registered as platform best. A time candidate must improve on
both full development seeds against both A35 **and V7** before further seed
confirmation; this avoids spending confirmation on an increment already
available from V7. Iron uses unchanged A35 iron. Apply the frozen candidate-tier
policy descriptively; promotion still requires four positive split seeds and
positive seed-level LCB95, with fold counts descriptive only.

At most one candidate can enter confirmation, ranked by its mean increment
over V7 (time) or A35 (iron), with the frozen cheaper uniform control first
on exact ties. Seeds 7777/12011 have previously been used in V5/V7 and are
not described as new labels. Same-fold reference identity is mandatory.
The local working gate remains 96.25. No full-data model, new submission package,
desktop write or upload is authorized by this experiment.
