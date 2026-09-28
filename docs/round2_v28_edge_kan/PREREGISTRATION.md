# V28_EDGE_KAN: compositional learned spline edges

Status: scientific strategy reservation, awaiting approval of the new bounded design; implementation, dependencies and fits0.
Branch codex/round2-v28-edge-spline-network. Controlling freeze: configs/round2_v28_edge_kan/SPEC.yaml.

## Purpose and teammate evidence

The user requested another strategy and explicitly invited improvements combining teammate ideas. Aim: a CPU-feasible new member with measurable isolated-column increments over the latest V21 local time/V12 iron reference; platform ambition >96.4 remains unmet. Current platform best V12=96.3526 is user reported, not independently verified. Latest teammate commit e0d38a9 adds a three-package delivery record but no new platform score.

Borrow the useful part of V7/V17/V21: learn a nonlinear numerical representation, retain complete-fold validation, inner epoch selection/fresh outer refit, and compare increment over the strongest current reference. V21's .40*V36+.25*V7_member+.35*P_LL_T time composition is an anchor, not a new fit or platform guarantee.

Do not repeat teammate V19 iron capacity, V25 time width/depth, V26 log/sqrt targets or V22 old-pool reweighting. The current library is saturated against V21. Existing V4 F1-F5 already tested random forests, leaf-distribution medians, median votes and honest partitions; presenting QRF as an untried family would be misleading. Existing V4 S/P tested input additive/tensor/PLS-projection splines, not independently learned functions on every hidden-layer edge. Those negative recipes remain closed.

V27_HARD_TREE stopped at resource G0: no quality evidence; full-size forward/gradient equivalence passed, but projected50.47h and peak1871MiB exceeded caps. V28 replaces the learner with compact dense matrix evaluation, not a smaller V27 recipe. Actual CPU throughput is unmeasured until approved G0.

## Choice, controls and limitations

Recommended: KAN learned edge functions on every layer, candidate EDGE, paired with SHARED input-channel spline shape. Alternative1 ExtraTrees/QRF overlaps previously tested forests; alternative2 ModernNCA overlaps the closed TabR retrieval question; neither is included. This remains a bounded addition to the existing fit/predict/audit flow. Scientific preregistration is written for the user's early strategy-occupancy publication rule; no separate implementation-plan document.

Primary sources: [KAN original paper](https://arxiv.org/abs/2404.19756), [efficient-kan source](https://github.com/Blealtan/efficient-kan), [fairer KAN/MLP comparison](https://arxiv.org/abs/2407.16674). KAN uses trainable spline edge functions. The comparison study reports MLP generally stronger outside symbolic tasks; no general KAN superiority or local gain is assumed. Prior periodic numerical embeddings motivate testing another representation; they do not prove edge functions will add an increment.

Pin efficient-kan commit7b6ce1c87f18c8bc90c208f6b494042344216b11. Source src/efficient_kan/kan.py SHA-256 c2b984359fe483640fcc2313570bbbd2128eede99bdbf0fa586ec8e0e1a1ecb7, MIT license SHA-25609c34a5fa51d62153d52c2870d0a4491a1d84b980000863a8b9d055adf7e45fc. Only torch and math are imported by that core. Adapt it in the current frozen runtime; do not install the whole package, torchvision or new dependencies. Preserve complete attribution. No external training data or pretrained weights. This is an adapted local recipe, not author benchmark reproduction.

## Frozen learner and data path

Layers [D,64,64,1], D=21 numeric plus training spout vocabulary. Three KANLinear layers, SiLU base branch, cubic B-splines with grid_size5 and fixed range[-1,1], author initialization/scales/noise, output-specific spline scaler enabled. No grid updates, spline regularization, scheduler, SWA, dropout, extra embeddings or postprocessing. In particular do not call the author's entropy regularizer at zero spline weights (its normalization is undefined there); regularization is disabled.

Execute the same author initialization/RNG order for both arms, then zero all spline coefficient arrays after whole-network initialization. Base weights, spline scalers, grids and all parameter shapes are identical; initial outputs must match. EDGE uses independent coefficients for every input-output edge. SHARED averages the full coefficient tensor over the output dimension before broadcasting and multiplying by output-specific scalers, at every layer. It retains the same arrays but ties effective curve shapes; active freedom intentionally differs. The single-output layer is identical across arms. SHARED is a diagnostic control, never a fallback promotion candidate.

Parameters =10*(64*D+4096+64); at D24 this is56,960, not the V27 tree scale. Forward evaluates basis tensors per input and dense linear maps, avoiding a batch-by-input-by-output activation tensor. This operation-count distinction is structural only; G0 must measure speed and full-path RSS.

Numeric QuantileTransformer is fitted only on the current training subset, uniform distribution, then2*u-1. n_quantiles=min(1000,training_rows), subsample=training_rows, random_state42, declared feature order. Reject nonfinite inputs. spout_no is training-vocabulary one-hot in ascending numeric order; unknown categories map to all zeros. No sample ID input or target encoding. Targets use training-only mean/population std (floor1e-8), MSE in standardized units, decode to original units. Do not clip hidden values or predictions; record hidden values outside the central grid range diagnostically.

AdamW lr.001, weight_decay.0001, betas(.9,.999), eps1e-8; batch/chunk256, max240epochs, patience25, min_delta1e-5 on standardized inner MAE. Training/inner seed42; group-safe inner fold0 validation, fresh outer-training transform and fresh initialization refit at selected epoch. These training choices follow V7's validated orchestration, not its model family.

## G0 admission and cost stop

Before official fits: author EDGE forward/gradient/initialization agreement, both zero-initialized and deterministic nonzero-coefficient fixtures (atol1e-6,rtol1e-5); meaningful shared/independent curve tests; finite basis/outputs/gradients; leakage-safe transformations/scaling; nonzero spline learning; cold persistence and chunk invariance (atol1e-6).

Resource probe first for cheap refusal: full initialization/preprocessing/model/optimizer path on2204 synthetic rows,21 uniform numeric features plus spout1/2/3, batch256; each arm one fresh optimizer,3 warm-up+20timed training steps,20evaluation forwards. All BLAS/OMP/MKL/NUMEXPR and torch intra/inter threads1. Measure worker peak RSS including preprocessing and retained data.

Require maximum worker RSS<=1024MiB; four times that measured peak plus1024MiB <= currently available RAM; projected40-fit development<=7200seconds (2h). Frozen projection: (40/4)*1.5*240*(16*max_arm_training_p95 +2*max_arm_eval_p95), based on upper outer2204/inner1764/inner-validation441 row counts already evidenced by prior same-fold records. Abort if actual development subset sizes exceed these bounds. The estimate uses maximum inner+outer epochs and is not promised actual duration. Failed resource admission closes this recipe, with no shrink or second specification.

Only if resource feasible run two full-size synthetic learnability optimizers: seed28001,1024 rows,first768train/last256validate;21 uniform[-1,1] inputs and spout1/2/3; y=10+3*sin(pi*x0)+2*x1*x2. Each arm80epochs without early stopping must beat a training-median constant on held-out MAE and have nonzero learned spline coefficients. Max G0 optimizers4, separate from official fits. Append started event before every optimizer; preserve completed and failed events, no retry, no evidence overwrite.

## G1 fixed budget and promotion

Current primary R28 equals R23/R27: V12/B0 iron, V21 local time(.40V36+.25V7_member+.35P_LL_T). Historical B0 is descriptive. Freeze exact data/feature/fold/reference/source/config/runtime identities only after G0 passes; source changes afterward abort execution. Use existing same-seed duplicate-group-safe/spout-stratified assignments, never average OOF across split seeds.

Development: two targets x SHARED/EDGE x42/3407 xall5folds =40outerfits/80optimizer runs. Isolated-column blend .8*R28+.2*member, other column unchanged; no alpha scan. WMAPE pooled denominator; package score100-50*(WMAPE_iron+WMAPE_time). Compare complete coverage against same-seed R28.

EDGE earns confirmation only if both development seed gains>0, mean>=.01 and EDGE mean gain exceeds SHARED mean. Select at most1target by highest eligible EDGE mean, tiesiron. No eligible target closes the round. Complete cold/model/OOF audit and independent eligibility arithmetic must pass, with audit bound to exact summary/config/source hashes, before consuming confirmation.

Earned confirmation: both arms, selected target,7777/12011,all5folds =>max20additionalfits/40optimizers. EDGE promotion requires4positive seed gains, one-sided seed-level paired LCB95>0(df3), and development mean candidate score>=96.25. Fold profile descriptive. Independently recompute eligibility and final4seed decision. Never reuse unused V23 confirmation trust logic.

## Persistence and publication

All model, prediction, trace, diagnostic, ledger and audit files stay local. New run directories only. Started failures consume frozen budget, no implicit retries or deletion. Save model/OOF, preprocessing/scaling, inner/outer IDs/groups/epochs, per-layer coefficient norms/support diagnostics, source/settings/reference/fold identities, memory and timing. No full-data fit, package, desktop write or upload.

Two public pushes: this scientific reservation, then validated implementation/tests/results at closure. User uploads packages; no release is part of this round. If training starts after approval, use a30-minute heartbeat, quiet while normal and actionable notification only on failure, stop, completion or user action. No background training or monitor is started during reservation.

Approval boundary: user authorized new strategy research and reservation publication. The new bounded design above still requires explicit design confirmation before code/dependencies/fits under brainstorming; earlier V27 confirmation does not cover a different model. Preserve this freeze if approved, and close rather than search a second recipe on failure.
