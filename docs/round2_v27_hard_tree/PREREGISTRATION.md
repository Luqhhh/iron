# V27_HARD_TREE: hard trees with input-dependent ensemble weights

Status: strategy reservation, awaiting design approval. No V25 implementation, dependency installation or candidate fitting has begun.
Branch: codex/round2-v27-hard-tree-weights. Machine-readable freeze: configs/round2_v27_hard_tree/SPEC.yaml.

## Purpose and current evidence

Continue the user-authorized search for a feasible new learner that can add an isolated-column increment over the latest local reference. The platform ambition remains >96.4; the latest registered best is V12_IRON_JOINT_PLR001_A50=96.3526, user reported and not independently verified. No local result forecasts a platform score.

V23_HIST completed 40 fits and was negative against the current reference: Gaussian mean increments were -0.009783 on iron and -0.007036 on time. Its time gain +0.005156 against historical B0 was already captured by the newer V21 anchor. Other-remote capacity and target-transform rounds were formerly V23/V24 and are now V25/V26 (1b00f30); this experiment reserves the next free V27. The V24 standalone accuracy improvements with zero blend weights do not establish a universal accuracy-ratio gate. WMAPE uses a pooled target denominator, not rowwise relative-error weights.

## Hypothesis and alternatives

Selected: adapt the pinned PyTorch GRANDE core. Hard axis-aligned, non-oblivious trees use straight-through gradients; the hypothesis is that leaf-dependent ensemble weights add a useful member to R27. GLOBAL tree weights form the paired control. This differs from existing CatBoost boosting and NODE/ODST soft routing; that difference is a reason to test, not evidence of improvement.

Alternatives considered: ModernNCA changes learned neighbor aggregation but overlaps the already tested TabR retrieval route; KAN learns edge-wise splines but overlaps the existing periodic/piecewise/PLS-projection basis explorations. DGT adds porting cost from its archived older runtime. These alternatives are not part of this round, and no fallback search is authorized by this reservation.

Primary sources: [author GRANDE repository](https://github.com/s-marton/GRANDE), [original GRANDE paper](https://arxiv.org/abs/2309.17130), [ModernNCA paper](https://arxiv.org/abs/2407.03257), [KAN paper](https://arxiv.org/abs/2404.19756), [DGT author repository](https://github.com/microsoft/DGT). The original GRANDE paper evaluated classification; the current author repository includes regression. Neither establishes performance on this dataset.

## Source adaptation and fixed recipe

Pin author commit 07f7278b30ab9ebbbdc544e5f9a91f1b5df7fecb. GRANDE/GRANDE.py SHA-256 is 53f58a980b751aca88ef61615d9e4b1904525952afcbed2678396c877f998064; MIT license SHA-256 is c433b2b805878f09d67ab901d67ee26168779f39ec192dabc414b5ccfc7ab99a. Preserve attribution when adapting the core. The author package bounds conflict with the current torch/sklearn versions and brings unnecessary dependencies. Do not install it, execute its full wrapper, or use its optional supervised embedding preprocessing. Use only the fixed core and current runtime, without external data or pretrained weights. This is a local adapted-core experiment, not author benchmark reproduction.

Freeze 1024 trees, depth 5, feature fraction 0.8 (author min/max feature-count rule), normal(0,0.05) initialization, dropout 0.2, no bootstrap/subsampling. Both arms draw the same author initialization and then zero all weight logits. Their initial parameters and evaluation predictions must match. INSTANCE uses the reached leaf's weight per tree and row; GLOBAL uses each tree's mean leaf-weight logit, constant across rows. Parameter arrays and shape are identical; GLOBAL's leaf-weight gradients are tied, so active degrees of freedom differ intentionally. Both arms retain the same routing computation. GLOBAL is a mechanism control and never a fallback promotion candidate.

Use the declared 21 numeric features, train-only normal quantile transformation and spout_no one-hot training vocabulary. Reject nonfinite numeric inputs. No IDs as features, label encoding, numeric embeddings, clipping or output calibration. Inner and outer transforms are fitted separately. Targets use training mean/population standard deviation, floored at 1e-8; MSE training, original-unit decoding. Adam betas=(0.9,0.95), eps=1e-8, zero weight decay, constant group learning rates: weights .001, feature indices .01, thresholds .05, leaves .05. Batch/chunk 256; max 250 epochs, patience 50, min_delta 1e-5 on inner MAE/training target std. Select epoch on group-safe inner fold 0 at seed 42, then initialize a fresh model and refit the outer training subset for that epoch count, using training seed 42. No scheduler, SWA or gradient clipping.

## G0 feasibility before official fits

First run meaningful synthetic checks: INSTANCE forward/gradient agreement with the pinned author core under identical zero-logit initialization (atol 1e-6, rtol 1e-5), hard single-leaf routing, finite gradients, paired initialization, query-invariant GLOBAL weights and query-dependent INSTANCE toy weights, train-only transforms, cold/chunk inference (atol 1e-6). Extract only the necessary author definitions for the golden check; do not execute the full author package.

Synthetic learnability: RNG seed 25001; 1024 rows of 21 independent normal numeric features plus uniform spout 1/2/3, first 768 train and last 256 validate; target 10+3*I(x0>0)+2*I(x1>0)-I(x2>0). Train each full-size arm once for 80 epochs. Both validation MAEs must beat the training-median constant. This is engineering evidence only.

Resource probe each full-size arm with batch 256 and 24 synthetic input columns, 3 warm-up plus 20 timed optimizer steps; separately measure evaluation forward time. Use max-arm p95 timings in the SPEC projection, conservative upper training-row counts, 250 inner plus 250 outer epochs and a 1.5 safety factor. Require projected development wall time <=6 hours at four workers, peak worker RSS <=1536 MiB, and 4*peak_RSS+1024 MiB <= currently available RAM. All BLAS/OMP/MKL/NUMEXPR and torch intra/inter threads=1. Probe RSS includes preprocessing/model/optimizer/intermediates; timing projection is an estimate, not a promised duration.

G0 consumes at most four synthetic optimizer runs (two learnability, two resource probes), separate from official fits. If infeasible, stop and register G0 failure. Do not shrink the tree count/depth or start a second specification. Official data fits remain zero until G0 passes and all source/config/runtime/data/fold/reference hashes are frozen.

## G1 paired evaluation and earned confirmation

R27 equals the audited R23 reference: V12/B0 iron; time .40*V36+.25*V7_member+.35*P_LL_T (V21 local, not platform verified). Reference cache identities must match before fits. Report historical B0 increments descriptively as well.

Exactly two targets x GLOBAL/INSTANCE x split seeds 42/3407 x all five outer folds =40 outer fits, each with inner selection and fresh outer refit (80 optimizer runs). Use existing duplicate-group-safe/spout-stratified outer assignments. No partial-fold screening. For each target replace only its column with .8*R27+.2*member; other column stays equal to R27. No alpha scan and no OOF averaging across seeds.

WMAPE=sum(abs(y-p))/sum(abs(y)); package score=100-50*(WMAPE_iron+WMAPE_time). Per-seed gain compares the same-seed complete package to R27. To earn confirmation, INSTANCE must be positive on both development seeds, have mean gain >=.01, and its mean gain must exceed GLOBAL's. Among eligible targets select at most one with highest INSTANCE mean; exact ties favor iron. No eligible target ends the round.

Only after complete development, cold inference and independent score/eligibility audits run both arms on the selected target at seeds 7777/12011, all five folds: at most20 further fits/40 optimizer runs. Promotion is INSTANCE only: all four seed gains positive, positive one-sided seed-level paired LCB95 (df=3), and development mean candidate package score >=96.25. Fold profiles are descriptive. No full-data fit, package or upload is part of this round.

## Evidence integrity and publication

New fit directories only; no overwrite, deletion of failure evidence, implicit retries, or repetition of completed fits. Failed fits consume the frozen budget. Any incomplete/failed development closes the phase without confirmation; preserve diagnostic evidence. Models, OOF, ledgers, audit records and experiment reports stay under local/.

Persist every model, OOF column, epoch trace, selected epoch, transform parameters, target scaling, train/validation ID/group hashes and full source/runtime identity, plus routing occupancy, gating entropy, effective tree count, gradient norms, wall time and RSS. Bind development audits to the exact summary hash and independently recompute eligibility before confirmation; independently recompute all four-seed decisions afterward. Do not reuse the unused V23 confirmation trust path. Hash all transitive model/preprocessing/scoring helpers and configuration files; abort formal execution if identities change.

Public summaries distinguish G0 engineering feasibility from G1 model quality, current-reference increments from historical-reference increments, and local evidence from platform reports. User cadence: publish this reservation promptly; publish validated public implementation/tests/results together at closure. No intermediate pushes during training. Check private-artifact exclusion before each commit/push. Future approved training uses a 30-minute monitor, quiet unless failed, stopped, completed or user action is needed.

## Approval boundary

The user authorized further strategy research and early reservation publication. This reservation fixes the proposed scientific recipe but does not claim approval for new implementation/fits. The bounded brainstorming design is: add this learner and its paired control to the existing offline fit/predict/audit flow; pass G0, evaluate40 fits, earn at most20 confirmation fits, preserve frozen references/gates and all evidence. Await explicit design approval before code, dependency changes or fitting. No implementation plan document is created for this bounded change.
