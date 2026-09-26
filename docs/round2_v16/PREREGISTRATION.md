# V16: sequential sparse feature selection

Frozen before any V16 official-data fit,2026-09-27. Current platform best is
user-reported V12_IRON_JOINT_PLR001_A50=96.3526; the target remains >96.4.
V7's original isolated package is pending feedback. No new delivery is authorized.

The current V12 iron direction has a measured platform gain, while V13–V15
have not supplied another promoted direction. The fixed V12/N-0048 weight
families have limited conditional upside (PLATFORM_BOUNDARIES.md in round2_v12).
V8 tested row-wise feature-token attention; no previous round tested TabNet's
successive sparse feature masks. This is a distinct, finite mechanism experiment,
not a claim that a new architecture will beat the incumbent.

Use the sequential feature-selection mechanism from
[TabNet](https://arxiv.org/abs/1908.07442), through the maintained
[DreamQuark PyTorch implementation](https://github.com/dreamquark-ai/tabnet).
That implementation is not the original authors' TensorFlow implementation.
Install only `pytorch-tabnet==4.1.0` with `--no-deps`, preserving existing runtime
versions. Record both installed tab_network.py and sparsemax.py hashes. No
pretrained weights, pretraining, external dataset or test-set training is used.

Freeze two jointly trained two-output models at one compact size:

- `uniform_masks`: constant uniform simplex masks at all three steps. Retain
  attention projections and normalizations but replace their selector; those
  projections receive no task gradient. All feature-transformer GLUs remain.
- `sparse_masks`: the implementation's learned sparsemax masks and its
  multiplicative feature-reuse prior. Negative mask-entropy output is subtracted
  with coefficient0.001 from equal standardized target MSE.

Both use n_d=n_a=32, three steps, two shared and two independent GLUs,
gamma1.3, virtual batch64, physical batch256, Adam0.01, zero weight decay,
StepLR0.9 every50 epochs, maximum400 epochs and patience50. The uniform
entropy term is constant and contributes no gradient. Initial parameters and
total parameter counts match; effective gradient paths do not. This is an
attribution control, not a claim of equal effective capacity or the paper's
benchmark recipe. Preserve finite losses and disclose cap hits.

Numeric preprocessing and each target's mean/std are fitted only within each
training partition. Spout is one-hot with the training vocabulary plus unknown.
Select epochs on the fixed group-safe inner split using equally weighted
standardized MAE, then initialize afresh and refit the whole outer training
partition. Batch normalization uses training rows only; inference is eval mode
and must be independent of query order, chunking and singleton batching. Merge
a singleton final training batch into its predecessor rather than discard it.

Development is exactly20 joint outer fits: both recipes, seeds42/3407, all five
folds. Also replay exactly one frozen V12 joint seed42/fold0 model with its
original training settings before candidate fits; both outputs must match.
Eight workers; BLAS/OMP/MKL/NUMEXPR and torch each one thread. No partial-fold
screen, adaptive pool expansion or changing the specification after evaluation.

Compare each isolated output to the actual **V12 platform recipe**: V12 iron
A50 and original A35 time. Separately report A60, A35, Q20, and CURRENT
(V12 iron A50 / delivered V7 time A50). Other-column package scores retain
the V12 platform recipe, never a simultaneous two-output replacement.
The sparse frozen alpha grid is inherited without expansion. Select alpha on
the other split seed, and retain seed separation throughout.

At most one target/recipe enters confirmation: both complete development seeds
must have positive incremental blend gain against V12_PLATFORM and CURRENT.
Rank by CURRENT mean, then uniform-before-sparse cost order, then target name.
Confirmation uses ten joint fits at7777/12011, but only the selected target may
be promoted. These seeds were previously used; they are not new independent
labels. Promotion requires four positive seeds and positive seed-level paired
LCB95; folds descriptive. The96.25 local gate remains unchanged. Candidate
tiers do not authorize release. Full-data fits, packages, desktop writes and
uploads are all0.

User steering during implementation: **v16完成后停止后续工作**. Finish this
frozen round including confirmation only if earned, audit and publish evidence,
then stop further optimization. Do not start V17.

Before official-data fitting:17 targeted tests passed; locked Python3.12
full suite **1109 passed,23 warnings**. The reference preflight verified83
files and both installed implementation hashes, with0 model fits.
