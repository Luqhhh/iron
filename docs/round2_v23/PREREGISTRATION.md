# V23: histogram target prediction after masked reconstruction

Date: 2026-09-28. Status: strategy reservation only; no V23 production code,
synthetic fits, official-data fits, packages or uploads have run. The user
requested a new score-improving strategy after reading the negative V20 handoff
and prompt strategy reservation push. This document makes the next experiment
reviewable; it does not claim that this mechanism will improve the platform.

## Context and identity

The starting checkout is `3b31f2b` on `codex/round2-v20-masked-recon`.
Its V20 is **masked reconstruction**, not the other branch's V20 P-LL release.
Preserve both histories. At conversation start `git pull --ff-only` completed.
The other live remote `round2-v6-iron-capacity-networks` was inspected through
`a566261` (V22). V21/V22 already occupy their numbers; this round uses V23.

Current highest platform score is **V12_IRON_JOINT_PLR001_A50 = 96.3526**,
user-reported, not independently verified. Goal remains **>96.4**. No V21
platform feedback is inferred. B0 (V12 iron plus V7 time) arithmetic 96.3679
remains conditional and is not a measured combination result.

V20's fixed p=.15/lambda=.2 masked auxiliary objective had mean -0.012990
against B0 and -0.008283 against its paired control. It offers no measured
benefit at that specification. This does not close the entire mechanism family.
The V22 remote evidence closes current-member search against its stronger V21
local reference. Do not rescan the pool or count captured gains again.

## Three candidate directions considered

1. **Histogram target prediction (selected).** Predict an entire conditional
   target histogram, with absolute-error-oriented median decoding. This changes
   the prediction problem and training gradients, rather than reconstructing
   inputs or stacking existing predictions. A same-shape hard-label arm checks
   whether Gaussian soft labels improve that new predictor.
2. Teacher/tree-leaf distillation. A different supervised representation could
   complement the neural model, but a safe teacher requires nested training
   inside inner and outer partitions. Its added fit cost and failure modes are
   high, while the recorded teacher library is already dominated. Defer.
3. More reconstruction weights/feature masks/frequencies. These reuse weak or
   closed priors and invite selected-configuration gains. Do not reopen them.

The first direction is motivated by [Imani and White, ICML 2018](https://proceedings.mlr.press/v80/imani18a.html)
and [Imani et al., Investigating the Histogram Loss in Regression](https://arxiv.org/abs/2402.13425).
These papers motivate an optimization hypothesis; they do not establish a gain
on this task. This adapted median-decoded TabM is not a paper reproduction.

## Frozen predictor and paired control

Use single-target periodic TabM (V7 `tabm_plr001`) independently for each target.
Preserve the 21 numeric features, categorical spout, train-only numeric scaling,
256 width, 2 blocks, k=16, dropout=.1, frequency=.01 and 16 embedding/frequency
dimensions. IDs are fold/ledger keys only. No external data or weights.

Replace the scalar output with **64 logits per ensemble member**. A softmax
produces bin masses. Every head receives target cross-entropy separately; mean
loss across rows/heads, never loss of an averaged head prediction. Two arms:

* `hard`: one-hot at the bin containing the observed target; upper endpoint
  belongs to final bin. Out-of-support training/encoding queries map to the
  boundary bin only. Count these events, never expand support using validation.
* `gaussian`: probabilities from integrating N(y, sigma^2) over each bin,
  truncated and renormalized to support, sigma **0.75 bin widths**. Compute
  targets in float64, then cast to float32. Identical model/seed/optimizer and
  batching as `hard`; do not scan sigma, bin count, decoder or architecture.

Fit support from the current training partition only. If min=m and max=M,
let R=M-m>0, bounds [m-.05R, M+.05R], and 64 equal-width bins. Constant,
nonfinite, empty or invalid targets fail before optimization. Inner selection
fits its own support from inner training; fresh full-outer-training refit fits
new support. No support or target scaling from outer held-out data.

For inference, **average head probabilities first**, then take the median of
the mixture histogram by linear interpolation within its first bin whose CDF
reaches .5. Do not average per-head medians. No scalar residual head, auxiliary
objective, calibration, clipping, target-combination or mean/median selection.
Record the histogram mean as a diagnostic only; it is not another candidate.
Support-limited extrapolation and smoothing bias are explicit limitations.

Optimizer/schedule stays at V7: AdamW lr=.001, decay=.0001, batch=256,
max_epochs=240, patience=25, min_delta=1e-5, training/inner seed42. Select epoch
by median prediction MAE divided by inner-training target standard deviation;
then reset initialization and train fresh on all outer-training rows for the
selected epoch. Equal optimizer settings do not equalize gradient magnitude.
This is a frozen recipe comparison, not isolation of every optimizer effect.

## References and scoring (freeze before candidate fits)

All references use the identical split seed, row IDs and exclusion folds.
Never average OOF predictions across split seeds. Check prediction hashes and
ledger identities before candidate execution; missing or failed identities
block execution and are reported rather than silently replaced.

* B0 iron = .5 V36_iron + .5 V12 joint member iron.
* B0 time = .5 A35_time + .5 V7 member time;
  A35_time = .65 V36_time + .35 N0048_time.
* **Primary stronger local reference R23**: B0 iron and V21_TIME_LOCAL time =
  .40 V36_time + .25 V7_member_time + .35 P_LL_T time.
  The V21 recipe comes from remote `a566261` and its frozen prior V21 SPEC,
  not a claim that V21 is a platform incumbent. Development P_LL_T caches are
  under `local/runs/round2-v17/development-r1/`; confirmation under
  `local/runs/round2-v17/confirmation-r1/`.

For each target, evaluate **.8 R23_target + .2 new_member**, keeping the other
R23 column unchanged. This weight is fixed before training; no nested or dense
alpha search. Also report the *same resulting candidate column* against B0,
so already captured V21 gains are separated. Raw member metrics and gaussian
minus hard differences are descriptive. Exact pool metric is
max(0,100-50 WMAPE_iron-50 WMAPE_time), computed on complete row coverage;
fold averages are not substitutes for pooled WMAPE.

## Gates and budget

Development: both targets, both arms, split seeds42/3407, all5 folds:
**40 outer fits**, each inner selection plus fresh outer refit (80 optimizer
runs). Start no official-data fits until probability/decoder isolation tests,
synthetic learnability, same-shape paired initialization, cold inference and
reference preflight pass. Identity gate never compares to another machine's
bitwise neural-cache outputs (the V20 trap).

Only Gaussian-arm units are main candidates. At most **one** target enters
confirmation: it must have positive R23-relative gain on **both** complete
seeds, mean R23-relative gain >= **.01**, and positive Gaussian-minus-hard
mean gain. Rank by mean R23 gain, then iron before time in an exact tie.
This extra .01 screen focuses resources; it does not relax historical gates.
A hard arm is a mechanism control, not a fallback release candidate.

Earned confirmation: that target's **both arms** on7777/12011, all5 folds,
20 additional outer fits maximum. Reuse frozen references and .2 weight.
Four Gaussian candidate gains positive, seed-level paired LCB95>0 (same V5
implementation), mean development candidate score >=96.25. The four splits
reuse labels, so LCB measures split stability, not four independent samples
or a 95% platform generalization guarantee. Fold level remains descriptive.

Absolute cap:60 official-data outer fits (120 optimizer runs), zero full-data
fits, zero packages, zero desktop writes, zero uploads. No second specification
or budget extension after failure/cap hits. Eight workers maximum; all four
BLAS/OMP/MKL/NUMEXPR environment variables set to1 before launch, torch intra-
/inter-op threads1. Never overwrite a run directory or discard failed evidence.

## Required engineering evidence

Meaningful tests: probability mass normalization, Gaussian tails, hard boundary
handling, float64 target construction, median interpolation, mixture-before-
median, finite gradients, constant/invalid inputs, fit-only support/feature
statistics, deterministic query reordering and save/load cold equivalence.
Synthetic predictive feature/target relation must beat the training-target
constant median. Do not describe a failed synthetic test as a negative G1 result.

Freeze source/spec/runtime/data/fold/reference digests and append-only fit ledger
before real fits. Record every epoch's train CE and inner median MAE, selected
and executed epochs, cap hits, support bounds, sigma/bin widths, outside-support
row counts, predicted entropy and both decoded mean/median MAE diagnostics.
Outer labels enter scoring only, never fit/checkpoint/target support.

Independent post-run audit reassembles OOF by sample_id, checks coverage,
prediction hashes and training metadata, and recomputes scoring and decisions
with separate arithmetic. Report G0 engineering separately from G1 quality.
Run locked Python3.12 tests and private-artifact guard before public commits.
All models, predictions, runtime reports, ledgers and plans remain in local/.

## Publication cadence and review boundary

User cadence from V20 handoff: push strategy reservation, then final completed
results; no intermediate pushes. The inherited four optional-torch test guards
are the handoff's CI prerequisite for safe reservation publication, not a model
change. Validate against a genuinely torch-free locked environment.

This reservation is a written design for review. Per the brainstorming skill's
bounded-design gate, implementation/real training follows design confirmation;
the reservation contains no production predictor or measured candidate result.
No package or upload is authorized by this design or by later classification.

## Reservation verification and design confirmation

The user confirmed **按冻结规格实施（推荐）** on2026-09-28, then explicitly
prioritized **先push策略占位**. This first push is the strategy reservation,
not completed implementation or a positive G1 result. V23 real fits remain0.

A genuinely torch-free, uv.lock-based Python3.12.13 environment ran the full
suite: **952 passed,46 skipped,3 warnings**. The initial full run exposed
13 failures and11 fixture errors after the four collection guards; their
missing torch/sympy dependencies were verified from the traces and protected
only at the consuming tests/fixtures in three additional files. Pure gating,
identity and score tests still run. All seven modified files change tests only.
The original four guarded test modules also ran with torch: **26 passed**.
A full Python3.12.3/torch2.14.0+cpu suite is running independently; its result
will be recorded with final round evidence, not presumed in this reservation.
Private failed/successful logs are retained underlocal/v23-strategy-20260928/.
