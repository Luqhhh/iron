# V9: from-scratch RealMLP training recipes (2026-09-26)

Platform objective **96.4**; current best remains user-reported A35 **96.3366**.
No external training data, pretrained weights, submission package or upload.
The existing five desktop alpha ZIPs are unchanged.

V7 established a useful periodic-network direction. V8 showed that learned
feature attention improves its uniform control but offers only small additional
blend gains. V9 tests the authors' complete RealMLP training recipes against
these stronger local references. This is a comparison of complete neural
training recipes, not a claim to introduce a new model family or identify the
causal effect of a single component.

Use the authors' [pytabkit implementation](https://github.com/dholzmueller/pytabkit),
version **1.7.3**, with `RealMLP_TD_Regressor` and `RealMLP_TD_S_Regressor`.
The [documented manual refit procedure](https://pytabkit.readthedocs.io/en/latest/models/examples.html)
passes the selected stop epoch to a fresh estimator with no validation split.
The exact resolved configurations are frozen in `configs/round2_v9/SPEC.yaml`.
Both use width256 x three layers, train from random initialization, and retain
the authors' 256-epoch schedule. Full TD includes its periodic representation,
parametric Mish, dropout and regularization; simplified TD-S uses its own
published defaults. No hyperparameter search or adaptive expansion is allowed.
The shared override selects epochs by **MAE**, matching the scoring objective;
the training loss remains the author's regression MSE. Inner training runs its
full schedule, then the fresh refit stops at the selected epoch while keeping
the original schedule horizon. There is one inner model and one refit per outer
fold, with no hidden ensemble, automatic split or HPO.

## Input and validation isolation

Use only the 21 instantaneous numeric features and `spout_no` from the approved
round-two training snapshot. Never pass `sample_id`, labels, ordering, lag data,
or target predictions as model features. IDs are used only for split identity.

The public sklearn interface ordinarily fits a category converter on the
provided train-plus-validation feature table. To keep this project's stricter
inner-only preprocessing contract, first fit numeric missing-value medians and
a one-hot spout vocabulary on the inner training portion alone. Map unseen
spouts to all-zero indicators. Pass a purely numeric matrix into RealMLP, so its
outer interface has no category vocabulary to learn from validation rows.
The native median-centering and robust-scaling fitters must be observed in a
synthetic test to receive only inner training rows; on fresh refit they receive
all outer-training rows. This input adapter is an explicit protocol adaptation,
not an exact reproduction of the authors' native categorical entry path.

Use existing group-safe inner splits and complete five-fold outer seeds42/3407.
Freeze all four target/recipe units before fitting: **40 outer fits / 80 optimizer
runs**, four workers, all BLAS/OMP/MKL/NUMEXPR and torch threads pinned to one.
Test repeatability, cold-process inference, query order/chunk independence,
unknown-spout handling and omission of labels/IDs before official-data fitting.
Synthetic feasibility probes have already passed fit/refit and cold prediction
with maximum absolute difference0; they are not model-quality evidence.

## Comparisons and continuation

Primary reference: A35. Report Q20 separately, plus the frozen V7 candidate-pool
time column `0.5*A35+0.5*V7_tabm_plr001`. Time must improve on both complete
development seeds against **both A35 and V7** before consuming more split seeds.
Iron must improve on both seeds against A35. Also report an iron diagnostic
against `0.9*A35+0.1*V8_ft_learned`; V8 confirmation is still in progress and that
column is not the registered platform best or an extra promotion gate.

Use the frozen sparse blend grid, fit weights on other split seeds only, and
never average prediction vectors across split seeds. Apply the existing
candidate-tier classifier without changing historical decisions. At most one
candidate proceeds, ranked by mean increment over V7 for time or A35 for iron;
simplified TD-S wins exact ties. Additional seeds7777/12011 are disclosed as
previously used split seeds, not untouched labels. Same-fold references and
source identities are mandatory; no time cache may stand in for an iron cache.

Promotion still requires four positive split seeds and a positive seed-level
paired LCB95. Fold results are descriptive. The **96.25** local working gate is
unchanged. No full-data model, new ZIP, desktop write or platform action is
part of this experiment.

## Environment

The 364005-byte official source wheel was checked against its PyPI SHA256.
Installation used exact constraints for every existing distribution: all **106**
old versions remained unchanged, with 12 new code/runtime dependencies listed
in `runtime-additions.txt`. Private installation and synthetic-probe evidence
lives in `local/research/realmlp-v9-feasibility/`. No model checkpoint or external
dataset was downloaded. The author trainer uses a dummy logger; its optional
cloud-logging installation hint does not enable an external logger.
