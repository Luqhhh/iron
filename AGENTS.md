# Repository execution contract

This repository implements the frozen `baseline-v0.1`. Unless the user explicitly opens a separate optimization phase:

- do not change CatBoost parameters, iteration count, targets, loss, feature windows, post-processing, or acceptance thresholds;
- do not read November 2024 targets from `train_samples.csv` or `tap_history_train.csv` in development workflows;
- protected-label access requires `configs/protection.yaml`, a frozen-manifest digest, and an append-only local access ledger;
- use the same as-of feature builder for train, validation, and prediction;
- never overwrite run directories or remove failed evidence;
- keep official CSV/XLSX data and dictionaries out of Git unless the user explicitly authorizes publication; publication is authorized for `初赛数据集/` as of 2026-09-07 and V2 `复赛_train/`, `复赛_test/` as of 2026-09-22; always keep models, predictions, reports, ledgers, and submissions out of Git;
- run the locked Python 3.12 test path before claiming authoritative reproduction;
- report G0 engineering status separately from G1 model quality.

## Commit and publication cadence

- After an authorized implementation or evidence update is complete and its required checks pass, promptly commit all in-scope public repository changes and push the current working branch to its configured upstream. Do not leave validated public changes uncommitted or unpushed while continuing the same task.
- Use ordinary, non-force pushes. Never rewrite published history, push directly to another branch, or include unrelated user changes in the commit.
- Before every commit and push, verify that private local artifacts remain outside Git. Models, predictions, reports under `local/`, access ledgers, platform receipts, and submission packages must never be committed or pushed.
- This standing Git publication rule does not authorize platform uploads, desktop writes, deletion of evidence, or publication of otherwise private data. If commit or push is blocked, preserve the work and report the exact blocker promptly.

## Platform handoff preference

- As explicitly instructed on 2026-09-22, users always upload packages themselves and return scores; the assistant does not upload.
- The user states a daily cap of 5 submissions. Do not repeatedly ask about upload mode or remaining quota. Use the latest explicit quota and recorded feedback for planning, retaining uncertainty about unrecorded account activity.
- At the start of V2.2 the user reported 1 remaining submission. Apply the frozen single-slot priority rule and deliver the preferred isolated package without spending the slot on an unverified two-target combination.

The `md/` tree is the archived original implementation package. Active configuration and status live at repository root under `configs/`, `docs/`, and `EVIDENCE_STATUS.json`.

The `baseline-v0.1-reproducible` tag is the immutable engineering baseline. Do not change its model, feature, semantic, source, or acceptance contracts in place. Start model-quality work as optimization-v0.2 on a separate branch and preserve baseline comparisons.

## Future V2 candidate triage

- For subsequent V2 optimization rounds, apply `configs/candidate_tiers.yaml` through `bf_tap_r2.candidate_tiers` and follow `docs/candidate_tiers.md`.
- Freeze the candidate pool, target-specific current references, and cost-based tie ordering before evaluation. Preserve formal promotion gates; retain mean-improving candidates that fail stability gates as exploration candidates with explicit failure reasons.
- Recommend at most one exploration candidate per round across targets, after formal candidates. Keep evidence verification and isolated release checks mandatory; classification never authorizes automatic release or platform upload.
- Do not retroactively reclassify or replace frozen V2.1/V2.2 decisions or packages.

## Latest V2 handoff (2026-09-22)

- The latest user-reported remaining submission quota is 0; the earlier V2.2 value of 1 is a historical snapshot only.
- On 2026-09-23 prioritize the already delivered V22_I_ONLY ZIP, SHA-256 `46c16936b13e18f3271f58f4de4aa18a87595f6a6612031710360c11f362b8fb`. Preserve its C2/D4 equal iron blend and full B3 time strings. Never overwrite it with new experiments or substitute T1.
- V2.3 is offline normalized RMSE/Huber/MultiRMSE exploration. Its frozen AH/AJ anchors remain C2 iron and full B3 time even if a later platform receipt favors V22. Report both current-anchor and V22-relative differences; do not move a new local winner ahead of the scheduled V22 first submission.

- Subsequent explicit user authorization generated V22_T_ONLY and V23_AJ_I_ONLY as isolated alternatives to the same B parent. Three ZIPs are now pending platform feedback; V22_I_ONLY remains first. Details and exact hashes: `docs/round2_v2_3/CANDIDATE_RELEASE.md`. Additional full J1 fits: 1; no additional CV fits. Score-transfer arithmetic is not a verified platform forecast.
- Latest queue correction: the user explicitly removed V22_T_ONLY from pending platform tests. Preserve its ZIP and evidence but do not recommend or schedule its upload. Current pending priority is V22_I_ONLY, then V23_AJ_I_ONLY; the earlier three-package delivery remains historical evidence.

## Optimization delivery after V2.4–V2.6

- Three new isolated packages are ready: V24_FORMAL_IRON_DJ, V25_FORMAL_TIME_AORD, and low-priority V24_EXPLORATION_IRON_BAY. Keep the existing V22_I_ONLY and V23_AJ_I_ONLY first, then those three in that order. Do not imply all three have stable evidence; BAY is exploration only.
- V26 J3/AJ3/DJ3 passed the C2-relative formal gate but did not satisfy the predeclared stricter delivery gate against DJ, so no V26 package was generated. Preserve the negative evidence and do not silently relax that gate.
- V22_T_ONLY remains removed from pending tests. All packages use the unchanged B parent for isolated-column replacement. Details: `docs/round2_v2_4/DELIVERY.md`.

## Final Top 5 handoff (2026-09-23)

- The user explicitly requested the final Top 5 from all verified offline candidates, including not-yet-packaged V2.6 candidates. The resulting final handoff order is DJ, J3, DJ3, AJ3, AJ, under desktop submission/final-top5. All retain the original B package full-B3 time strings.
- Treat the older queue above as historical; preserve its files and decisions. This separately requested ranked release does not retroactively change V2.6's failure to beat DJ. DJ and AJ ZIPs are exact copies of original deliveries; three new packages share two newly fitted full joint models.
- Exact identities and validation: docs/round2_final_top5/DELIVERY.md. Local OOF rank does not establish platform rank. Users upload and return scores; do not ask again about mode or daily quota.

## Final Top 5 scores received (2026-09-23)

- All final Top5 scores were explicitly returned by candidate: DJ=96.1079, J3=96.1131, DJ3=96.1191, AJ3=96.1259, AJ=96.1035. These are user reports, not independently verified platform receipts.
- Current preferred package is final-top5/04_AJ3_IRON, SHA-256 a9e1a57ab6bba020ab4729bdac4f7a657f7f40d249504e0c35f5b0d9c48d0354: AJ3 iron with unchanged full B3 time. Preserve original package bytes and old decisions.
- The final Top5 batch is no longer pending. Future optimization compares against the latest registered current platform best in `EVIDENCE_STATUS.json`; as of 2026-09-24 that is V34_A at `96.2684`, not AJ3 `96.1259`. Preserve AJ3 and the other final Top5 records as historical evidence. Do not infer account quota or automatically generate a two-target combination from unreported time results.

## Platform test priority preference (2026-09-23)

- The user prioritizes packages with larger offline gains against the current target reference or a clearly justified, important exploration question. Give small-gain, closely related variants lower platform-test priority; do not fill slots merely to use the daily allowance.
- The observed final Top5 platform/local gap is about 0.09–0.11 score points: modest in absolute size but enough to reorder closely scored candidates. Do not generalize this to a fixed score correction or claim only low-scoring models can change order.
- This manual release/scheduling preference supersedes automatic formal-first upload ordering, not candidate-tier classification, frozen gates, exploration caps, or historical evidence. Preserve small-gain candidates without promising they will be tested. See docs/candidate_tiers.md for the controlling explanation.

## Round2 V3 local search (2026-09-23)

- Branch `round2-v3-local-search` contains the unified V3 local-search sampler,
  runner, fusion utilities, tests, and first-batch evidence.  Public heads:
  `configs/round2_v3/experiment.yaml`, `src/bf_tap_r2/v3_local_search.py`,
  `src/bf_tap_r2/v3_run.py`, `tests/test_round2_v3_local_search.py`.
- At the time of the V3 local-search round, the platform reference was AJ3 iron + full B3 time, user-reported
  `96.1259`; the corresponding local reference was `96.0123883047359`. For the current V3.4 reference see the 2026-09-24 section below.
- The first V3 batch is complete: 400/400 coarse `(configuration, target)` items
  (CatBoost 240, LightGBM 40, XGBoost 40, MLP 20, kernel 20, expression 40).
  `xgboost==2.1.4` is installed through the `round2` optional dependency.  The
  top three XGBoost candidates per target were also refined; they did not enter
  the selected fusion.
- The CatBoost-refined nested fusion reached local package score `96.100735`,
  delta `+0.088347` vs the AJ3 local reference.  The all-family refined library
  was slightly lower at `96.100021`, delta `+0.087633`.
- The selected outer procedure fit members/weights on seeds 42 and 3407 only,
  then reconstructed the required members on independently derived seed 2026
  without using 2026 labels.  Seed 2026 package score: `96.116174`; delta vs the
  same-seed AJ3 local reference: `+0.088335`.  The gain was retained, but this
  remains local `candidate_pool` evidence, not a platform forecast.
- No V3 package, independent cold release, or upload has been authorized.  Do
  not generate or upload a V3 package until the user explicitly requests it.
  The existing frozen release and platform-upload rules still apply.

## Round2 V3.4 next phase (2026-09-24, historical context)

- User-reported platform scores: `V34_A = 96.2684`, `V34_B = 96.2660`. These are user reports, not independently verified platform receipts.
- At that time the current platform best was `V34_A`. Historical AJ3 `96.1259` remains evidence but is no longer the current reference.
- Next-phase platform target: **> 96.3**. This target is still in force.
- Pre-registered local working gate: **>= 96.25**; local stretch target: **>= 96.30**. The gate is derived from observed local-to-platform gaps of roughly `+0.069` to `+0.114`, but local scores remain non-guarantees.
- **Do not consume platform test quota unless a candidate first reaches the local working gate and passes same-protocol outer validation.** Local score or local delta alone is not acceptance.
- Continue **large-scale optimization search**: broad local exploration, not small variants of already-tested packages. Preserve data protection, dedup, append-only cache identity, candidate freeze and cold-audit rules.
- New candidates must have at least two positive complete development splits before any new outer seed is consumed; only clearly justified exceptional cases may bypass, with explicit pre-registration.
- Users upload packages themselves and return scores. Do not auto-upload, auto-package for platform, or spend quota manually.
- Latest score-transfer and gate analysis: `docs/round2_v3_4/SCORE_TRANSFER_AND_NEXT_TARGET.md`.

## Round2 V4.2 through V4.5: mechanisms tested, residual route closed (2026-09-25, historical context)

Current platform best is still **`V36_USER_REQUESTED_OUTER_FAILED` = 96.2734**
(user-reported, not independently verified); next-phase target remains **> 96.3**.
The V4.2–V4.5 rounds produced **no promoted candidate, no package and no upload**.

Two independent facts now constrain the next round:

1. **Local gains are not transferring.** `V42_IRON_N2_Q25` (parent V36 iron +
   0.25 × V4.2 `N-N2`) scored **96.2533** on the platform, **−0.0201** against
   V36, while its local same-protocol gain was **+0.0144**. The local-to-platform
   gap across verified pairs is not constant (`+0.0687` for V34_A, `+0.0696` for
   V36, `+0.0347` for V42_IRON_N2_Q25), so the gap's own spread is larger than
   the effects being selected on.
2. **The local ranking signal is weaker than the effects.** The N2 blend's
   per-fold gain spans `−0.009` to `+0.028`; V34_A's final outer five folds span
   `+0.0078` to `+0.0641` around a `+0.0394` mean. The pre-registered coarse
   threshold of `+0.005` sits inside that noise.

**Closed by direct measurement — do not reopen without new evidence:**

- **NODE per-depth selection (N4) and the ODST core (N5):** both fail the coarse
  gate on both targets (`N4 +0.00094 / −0.00093`; `N5 −0.00311 / +0.00459`). The
  repaired shared-selector control N2 passes on `tap_iron` (`+0.00901`) and then
  fails the full-coverage fusion gate on complete coverage (`+0.014934`, 8/10
  folds). `docs/round2_v4_4/RESULTS.md`.
- **TabR full retrieval fusion (R4) and its no-retrieval control (R5):** all six
  units fail the coarse gate. R4 − R5 is `+0.0176` / `+0.0037`, so the retrieval
  channel is active, but R4 − R2 is `+0.0123` / `−0.0118`, so the fusion gain
  does not reproduce across targets and the family stays far below `B_fit`.
- **Residual correctors:** the strong base's out-of-fold residual has no
  predictable direction (mean sign AUC `0.5092` / `0.5070`; the fitted sign model
  is worse than a calibrated constant by Brier score; `P(r>0)` is 0.4938/0.4996
  against residual standard deviations of 26.07/6.23). Every pre-registered
  correction raised held-out MAE. `docs/round2_v4_5/RESULTS.md`.
- **Temporal/lag features are unavailable, not merely untried:**
  `复赛_train/train_features.csv` carries only `sample_id` plus the 21
  instantaneous features — there is no timestamp or ordering column.

**Where the remaining leverage is, given the above:** the V36 composition is
weight-concentrated (`tap_iron` 0.604 A + 0.360 D + 0.036 N; `tap_time_len`
0.781 A + 0.090 O + 0.129 D), `tap_time_len` is worth ~4.2× per absolute-error
unit because its WMAPE denominator is ~4.2× smaller, and the gains actually being
harvested are diversification gains (the N2 candidate is *worse* alone, MAE
21.4 vs 20.2, but only 0.887-correlated, and a 0.25 blend wins 0.16 MAE).
A 482-file / 444-trial OOF prediction pool already exists under
`local/runs/round2-v3-local-search/*/pred-*.npy`, so an explicit
error-covariance-based member search needs no new model fits. The V3.4
"diversity" work was *structural* de-duplication, not error-correlation
selection.

**Process constraints for the next round:** raise the local evaluation
resolution (more split seeds, paired standard errors, admit on a lower confidence
bound rather than a mean) before trusting any `+0.01`-scale candidate; and treat
any further submission as an experiment worth declaring, not a routine slot.

- Implementation commits on `round2-v4.2-n2-seed-repair`: `2208bd1` (V4.4
  mechanisms), `d774195` (bit-identical N-line reproduction through the
  committed screen), `91dded2` (V4.5 residual diagnostic).
- A second agent session was active in this checkout during these rounds; it
  authored `src/bf_tap_r2/v4_2_followup.py` and the V4.2-r2 repair. Its
  collision record is in `docs/round2_v4_2/FOLLOWUP_RESULTS.md` section 8.


## Round2 V5: error covariance, resolution, and the platform noise floor (2026-09-25, historical context)

Branch `round2-v5-error-covariance-resolution`; pre-registration `configs/round2_v5/SPEC.yaml`
+ `docs/round2_v5/PREREGISTRATION.md` (commit `f0d7900`); implementation commit `06f956c`;
results `docs/round2_v5/RESULTS.md`. **The round exceeded the 96.3 target: `V5_TIME_N0048_Q20`
scored 96.3143, +0.0409 over the V36 parent.** Two packages were delivered for the user to upload —
a declared noise-floor control and the promoted candidate — and nothing was uploaded by the agent.

**Premise correction that must not be repeated.** The earlier statement that
`local/runs/round2-v3-local-search/*/pred-*.npy` holds "482 files / 444 trials with two split
seeds" is wrong. Those 482 files are **400 distinct trial ids**; only **38 trials** have complete
five-fold coverage at both seeds 42 and 3407. The other 400 entries are the coarse batch at
**seed 42, folds 0/1 only** (1102/2754 rows). The same `trial_id` is a *different fit* in
`coarse-*` (early stopping on) and `refine-*` (early stopping off), so any library key must be
`(source directory, seed, trial id)`. The real zero-fit library is: 114 (iron) / 110 (time)
two-seed complete columns assembled from the V3.6 development cache, the V3 refined trials and
the recorded `round2-v2*` OOF columns.

**Measured results.**

- *Existing pool exhausted.* With the pre-registered rule (`rho <= 0.95`, single-model WMAPE
  `<= 1.25x` base, interior alpha), 16 of 60 screened candidates have a positive nested mean and
  **zero** pass the fold gate. Best: `v3-mlp-tap_time_len-0000` at `+0.0055` score points,
  7/10 positive cells. Best composition *replacement* (drop the near-zero `O-0057`, add that
  model) reaches `+0.0057`. Leave-one-out: `N-0005`, `O-0057`, `D-0048` together carry 0.255
  weight and contribute about 0.002 points.
- *The time column's N family was the real gap, and completing it nearly pays.* The frozen V3.6
  round stopped the N line for `tap_time_len` on "single models are weak", although the iron side
  was equally weak singly and still supplied the released composition's entire gain. Refining
  the selected N candidates to complete coverage gives `v36-s1-N-0048` (large raw-TabM)
  `+0.00978` points, **8/10 folds positive**, both split seeds positive (`+0.0121`/`+0.0075`),
  `rho = 0.888`, accuracy ratio 1.07, fold-level LCB `-0.00015`. Lightweight N candidates only
  reach `+0.002..0.003`. The family's value is correlation, not single-model score.
- *Local resolution is the binding constraint.* Shifting **only the training seeds** (recipe,
  weights, experts and split seeds unchanged) moves the local score by `-0.00496` (iron) and
  `+0.00035` (time), and the `shift = 0` rebuild reproduces the parent columns to `2.6e-16`
  relative. Candidate effects are therefore of the same order as a pure seed change, so any
  further promotion needs the four-split-seed rule, not a mean threshold.
- *Fold-level lower bounds are not protective.* The recorded N2 cells give a positive fold-level
  95% lower bound (`+0.0067`, 8/10) even though the candidate lost `0.0201` on the platform. The
  V5 rule refuses it only because it requires at least four split seeds. That arithmetic is pinned
  by `tests/test_round2_v5.py`.
- Backtest: 4/4 known-bad candidates (V42 N2, N4, N5, R4) rejected; the old mean-`+0.005` gate
  would have admitted two of them.

**Next action (platform).** `V5_SEED_SWAP_S1000` is prepared and verified
(ZIP SHA-256 `4ff97c033f05b7e686453e913460b548cd7231d3736b3f431bb3882e1a82460f`). It is the frozen
V34_A endpoint plus the four frozen experts with every training seed shifted by +1000. Upload is
performed by the user. `|delta| ~ 0.02` against 96.2734 ends thousandth-chasing; `|delta| ~ 0.005`
means seed sensitivity dominates and the error-covariance route continues under the four-seed
rule.

**Stage 2b passed.** `v36-s1-N-0048` was replicated on the derived split seeds `7777` and `12011`
(`V36FixedRecipeFactory` refit per fold; the released column is the blend endpoint, with the weight
selected on the other three seeds). Per-seed gains are `+0.01233 / +0.00808 / +0.00970 / +0.00876`,
mean `+0.00972`, sd `0.00186`, **seed-level paired LCB95 `+0.00753` with 4/4 seeds positive**, and
the derived-seed baselines (`0.03825`/`0.03842`) match the recorded ones (`0.03829`/`0.03848`).
This is the first candidate in the project to pass an independent-split-seed gate. Its fold profile
is the reservation: 14/20 cells positive (70% against the 80% reference), reported under
`decision.fold_criteria` and `descriptive_only` per the specification. **It still does not reach the
frozen local working gate** (`96.2038 + 0.0097 ≈ 96.2135 < 96.25`), so it enters `candidate_pool`
and no candidate platform package was generated.

Three implementation defects were found and fixed during replication, all now pinned by tests in
`tests/test_round2_v5.py`: the fold criterion was an absolute count (8) instead of the
specification's fraction (so 20 cells only needed 40%); `admit` treated the `descriptive_only` fold
level as binding; and the replication initially compared the raw candidate rather than the weighted
blend, which produced a spurious `-0.11..-0.14` before being caught. Stage 1 was re-run after the
fixes: the backtest still rejects 4/4 known-bad candidates, and the full-coverage N2 record is still
admitted by the fold-level rule alone and refused only by `insufficient_split_seeds`.

**Noise-floor experiment returned (user-reported, 2026-09-26).** `V5_SEED_SWAP_S1000 = 96.2739`,
**+0.0005** against V36, while its local seed sensitivity is `-0.0046`. Two consequences, and the
earlier reading must be replaced:

* **the platform resolution for this perturbation class is about 0.0005** — the platform does not
  jitter at the 0.01–0.02 level, so "local gains do not transfer because the platform is noisy" is
  refuted;
* the earlier claim that "the transfer error of a clean control is about 0.005" is **REVISED and must
  not be reused**: the candidate that followed showed a transfer error of `+0.031`, so the transfer
  error is candidate-dependent and can be several times the local effect in either direction. Local
  evidence is reliable for *sign and ranking within four-seed-rule candidates*, never for magnitude.
  The `V42` inversion (`+0.0144` local, `-0.0201` platform) is still better explained as local
  selection overfitting (it had only two split seeds) than as platform noise.

`V5_SEED_SWAP_S1000` is now the highest user-reported score and is registered in
`EVIDENCE_STATUS.json -> round2_current_platform_best` as a **noise-floor control, not an
optimization candidate**; every candidate still uses the frozen `V36` as its parent.
`v36-s1-N-0048` (four-seed replication, LCB `+0.0075`) was predicted at `+0.005..+0.015` from its
local evidence; the actual platform effect was `+0.0409`. That single comparison is the reason the
magnitude reading above is retracted, and it is why the frozen local working gate `96.25` — which
would have blocked this candidate — is recorded as too conservative. Re-deriving the gate is a user
decision and has not been made.

**TARGET EXCEEDED (user-reported, 2026-09-26).** `V5_TIME_N0048_Q20` scored **96.3143**,
**+0.0409 over V36** and **+0.0143 over the 96.3 target**. It is now the registered current platform
best in `EVIDENCE_STATUS.json`. Recipe: the frozen V36 parent with only `tap_time_len` replaced by
`0.8 x V36 + 0.2 x v36-s1-N-0048` (large raw-TabM refit on all training rows); `tap_iron` byte-identical
(0 mismatches), blend recomputes to 0 difference on read-back, cold inference diff 0. ZIP SHA-256
`5ed99b8014fb1650b88b6ae57cc3ed4dac378a20831eab5efc800de91ab2c982`.

Two claims are now measured, and one earlier claim is retracted:

* **the four-seed rule passed its first live test**: the candidate was positive on the two derived
  split seeds that never entered selection, and the platform delivered a gain 4.19x its local
  prediction (`+0.00975` local, `+0.0409` platform; the implied test-set time-column reduction was
  2.13% against 0.51% out of fold);
* **local magnitude is compressed**, so local screening ranks and orients but does not forecast;
* **the frozen local working gate (96.25) is proven too conservative**: it would have blocked this
  candidate (expected local package 96.2135). The local-to-platform offset for this candidate was
  `+0.1008` against a historical range of `+0.0347..+0.1008`. Re-deriving the gate from the measured
  offset is a user decision; no threshold was changed.

**The time-column route is exhausted with the current library.** The four large N members are
near-duplicates of each other (pairwise residual correlation 0.979-0.998) so stacking them adds
nothing, and adding the decorrelated but less accurate `JM1` (`rho = 0.777`) to `N-0048` optimises
to weight 0 and does not improve the held-out seeds. Going further needs new model families, not
more member search in the same pool.

**Closed routes are unchanged** (NODE per-depth, ODST core, TabR full fusion, residual
correctors, sample reweighting, dense alpha scans). Temporal/lag features remain *unavailable*
(`复赛_train/train_features.csv` has no timestamp or ordering column), not merely untried.

## Round2 closure (2026-09-26, current instruction): target exceeded, round 2 closed

**Final platform best: `V5_TIME_N0048_Q20 = 96.3143`** (user-reported, not independently verified),
**+0.0409 over the previous best `V36 = 96.2734`** and **+0.0143 over the 96.3 target**. Recipe: the
frozen V36 parent with only `tap_time_len` replaced by `0.8 x V36 + 0.2 x v36-s1-N-0048` (large
raw-TabM refit on all training rows); `tap_iron` is byte-identical to the parent. ZIP SHA-256
`5ed99b8014fb1650b88b6ae57cc3ed4dac378a20831eab5efc800de91ab2c982`, on the desktop under
`round2-V5-time-N0048-Q20-20260926`. The round-2 search itself is closed with no promoted candidate
pending, **but a five-package exploration portfolio is now pending user upload** — see the
"Portfolio delivery" section below and `EVIDENCE_STATUS.json -> round2_v6_portfolio_2026_09_26`.

**Transferable conclusions** (full evidence in `docs/round2_v5/RESULTS.md`,
`docs/round2_v6/RESULTS.md`, `EVIDENCE_STATUS.json -> round2_closure`):

1. **The four-split-seed rule works.** A member that was positive on the two derived split seeds
   that never entered selection delivered `+0.0409`; the old "two seeds with the same sign and a
   mean of at least +0.005" gate would have admitted the historically failing N2. Promotion is
   therefore: at least four split seeds, a positive seed-level paired LCB95, every contributing
   seed positive — with the fold level **descriptive only**.
2. **Local magnitude is compressed and does not forecast the platform.** Local `+0.00975` became
   platform `+0.0409` (4.19x), while the seed-swap control moved `-0.0046` locally and `+0.0005` on
   the platform. Local evidence may be used for **sign and ranking** among four-seed-rule
   candidates only. The earlier claim of a constant `0.005` transfer error is **retracted**.
3. **Marginal statistics do not identify a useful member.** Over fourteen witnesses, residual
   correlation, accuracy ratio, row-win rate, mean absolute error and blend stability all fail to
   order success from failure; only the **mean nested gain** does. The binding test is the
   **incremental blend gain** on top of the incumbent, not any single-model quality statistic.
4. **The folds 0/1 screen (1102 rows, one split seed) is too noisy to rank candidates.** The
   validated winner's single-fold alpha estimates span `0.12-0.38` and only converge with sample
   size (2 folds `0.31`, 5 folds `0.24`, 10 cells `0.20/0.21`) — yet every V5 and V6 selection
   decision was made on that screen. Any future search must screen at complete coverage.

**Closed by direct measurement — do not reopen without new evidence:** the blend-stability
signature; a second recorded member on top of the winner (exactly zero incremental gain); the iron
capacity lever (its most favourable single test passed rho and accuracy ratio but produced no blend
gain); a **learned/regularized combiner over the whole reproducible OOF library** (a nested-`alpha`
ridge over 27-28 reproducible members adds only `+0.00425` over the delivered incumbent column,
6/10 cells, one split seed negative, below the `0.0232` MDE, and `-0.00139` without N-0048; the
apparent `+0.017..+0.023` against V36 is mostly restatement of what N-0048 already captured, and
LightGBM stackers are negative on both targets — `docs/round2_v6/RESULTS.md` §7); and, from the
earlier rounds, NODE per-depth, the ODST core, TabR full fusion, residual correctors, sample
reweighting and dense alpha scans. Temporal/lag features remain *unavailable*
(`复赛_train/train_features.csv` has no timestamp or ordering column).

**Incumbent-relative member check (2026-09-26, `docs/round2_v6/RESULTS.md` §8).** Under the correct
reference — the delivered incumbent column `0.8*V36 + 0.2*N-0048`, not V36 — a nested 2-column blend
of every one of the **27 non-v2 complete members** leaves **only one positive**: `v3-mlp-tap_time_len-0000`
at `+0.00065` (6/10 cells), i.e. 3% of the `0.0232` MDE; the other 26 are exactly `0.00000` (weight
driven to zero) or negative. This upgrades the V5 statement from "no gain relative to V36" to "no
member of the existing library leaves a measurable increment on top of the delivered incumbent".
The complete-coverage `alpha` surface is a clean interior optimum at `alpha=0.20` (`0.25`: `-0.0005`,
`0.30`: `-0.0021`, `0.40`: `-0.0095`, `0.60`: `-0.0370`), so the recorded `alpha=0.25` delta is now
10-cell evidence. **D-prime-2 (completing `raw_mlp|large`, `ple_mlp|large`, `ple_tabm|large`,
`raw_tabm|medium`, ~40 fold fits) is demoted to a negative prior**: it would add new configurations
of the same families whose existing complete members are all exactly zero, and the recorded `JM1`
precedent (`rho` 0.777 but less accurate, weight optimised to zero) is the situation it faces;
`raw_tabm|large` is N-0048 itself.

**Platform scoring note (user-reported 2026-09-26): the platform keeps the best score.** A new
submission therefore carries **no downside beyond quota** — it is a free option — so a portfolio of
**genuinely independent** candidates would be rational. Be precise about the limit: no such candidate
exists in this repository without new fits, and near-duplicates (shared base, members at `rho` 0.98)
have correlated realizations, so they do not diversify. The `alpha` probe is safe but is a diagnostic
with no expected gain, not an optimization.

**Leakage rule added by the stacking diagnostic (`docs/round2_v6/RESULTS.md` §7):** averaging OOF
vectors **across split seeds** is a leak, not a variance reduction — a row held out under one split
seed can carry another seed's component produced by a model trained on that row, so the held-out
feature itself encodes the label. That defect manufactured spurious `+0.0125/+0.0135` incumbent
gains before it was caught. Any design matrix that mixes split seeds must be closed **within a single
seed**. This is the same error family as the V5 replication defect (comparing the raw candidate
instead of the blended column, which produced a spurious `-0.11..-0.14`).

**If the search is ever resumed**, the controlling design is: screen at complete coverage, judge by
incremental blend gain, promote under the four-seed rule, and treat any submission as a declared
experiment. The user has confirmed the platform **keeps the best score**, so the earlier condition
("only worth a slot if the platform keeps the best score") is satisfied: the `alpha = 0.25` probe is
now safe to spend, but it is a **diagnostic with no expected gain** (locally `-0.0005` at complete
coverage), not an optimization. The stacking closure and the incumbent-relative member check remove
the last in-pool hopes: a better combiner is not a new problem class, and every existing member is
already exactly zero on top of the incumbent.

### Resolution correction (2026-09-26, after the closure section)

The closure section's claim that "the platform resolution for this perturbation class is about 0.0005" was
**wrong and is retracted**. That number was the *observed change* produced by one particular perturbation (a
training-seed shift), not the platform's resolving power. A paired row bootstrap of the delivered blend
against the released column (2000 replicates, seeds 42/3407, scaled to the 322-row test set) gives:

| quantity | value |
|---|---:|
| sd of the paired delta at 2754 rows | 0.00397 |
| **sd of the paired delta at 322 rows** | **0.0116** |
| 95% minimum detectable delta on 322 rows | **0.0228** |
| observed platform delta (+0.0409) | **3.5 sigma** |
| sd of the score *level* at 322 rows | 0.119 (irrelevant for A/B: the platform scores the same fixed rows, so the level noise cancels) |

The strategic consequence is sharper than the earlier picture: a single-column improvement must be worth
**more than about 0.023** to be distinguish-able from row-sampling noise at all. That is why `V42`'s `+0.0144`
local gain could not transfer and why every `+0.005..0.015` candidate was unreadable, and it means the only
strategy with a positive prior is a **large structural gain**, not member tweaking.

### Platform determinism and leaderboard context (2026-09-26, user-reported)

Two facts from the user change the framing and must be carried forward:

* **The platform is deterministic.** Re-submitting the same package returned the same score, so an observed
  difference between two packages is **exact**, with no platform-side jitter to average out. The paired
  `0.0116` sd above is therefore **not** the platform resolution; it is the **generalization uncertainty** —
  how far a local gain can differ from its realization on the fixed 322-row test set. A package with a local
  gain of `+0.01` lands around `+0.01 ± 0.012`; a `+0.005` local gain is a coin flip. The delivered `+0.0409`
  is an exact leaderboard improvement, 3.5 sd above zero in generalization terms.
* **Leaderboard position: first place is `96.5088`, and the project sits around the median.** The gap to first
  is `0.1945` — about **five times** the largest gain this project has ever produced (`+0.0409`). Since the
  empirical ceiling of member blending on the time column is about `+0.010` and it has already been captured,
  that gap **cannot** be closed by further member search. Climbing the field requires a fundamentally better
  model (different features or paradigm) — a different problem class with an unknown prior.

Because the score is deterministic, submitting several **genuinely independent** candidates is a portfolio
play (each has a fixed realization and the best is kept); near-duplicate candidates (shared base, members at
`rho 0.98`) have correlated realizations, so the portfolio only diversifies with independent models.

### Portfolio delivery (2026-09-26, current pending handoff)

`EVIDENCE_STATUS.json -> round2_v6_portfolio_2026_09_26`; evidence `docs/round2_v6/RESULTS.md` §9;
desktop `round2-V6-portfolio-20260926/`. Five packages are **pending user upload** (the agent does not
upload). Full analysis: `EVIDENCE_STATUS.json -> round2_current_candidate_queue`.

Premise: the incumbent `V5_TIME_N0048_Q20 = 96.3143` is already on the board, so all slots go to new
candidates; and with the platform keeping the best score a submission is a free option, so the rational
play is a portfolio of partially independent realizations keeping the max. Selection used a paired
322-row row-bootstrap over 48 directions, greedy on `E[max]`, **half-sample validated** (select on one
half +0.0072, honest on the other +0.0071) — not a bootstrap overfit. The most valuable entry is the
**iron** direction, because the incumbent changed only `tap_time_len` and `tap_iron` is still
byte-identical to V36: `v3-kernel-tap_iron-0005` has residual correlation `rho = 0.767` (marginal
+0.00403).

| tier | slots | packages | E[max] if best-kept | if latest-kept instead |
|---|---|---:|---:|---:|
| A | 2 | `V6_PORT_IRON_K0005_W10`, `V6_PORT_TIME_MLP0000_W10` | +0.0057 | about -0.001 |
| B | 3 | A + `V6_PORT_TIME_A05` | +0.0065 | about -0.006 |
| all | 5 | B + `V6_PORT_TIME_A35`, `V6_PORT_IRON_K0005_W20` | +0.0075 | about -0.015 |

**Every entry has a NEGATIVE local mean** (-0.0007 to -0.0153): these are portfolio lottery tickets,
not better models, and the positive expectation is entirely `E[max]` under the best-kept rule. Deliver
**tier A only** unless the user is certain about that rule. Upload weakest-first so the last slot lands
on the best-mean entry (`V6_PORT_IRON_K0005_W10`). Verified: both new members' full-data fits reproduce
their fold OOF bit-identically (`v3-mlp` max abs diff `0.000e+00`, `v3-kernel` `1.4e-11`), cold checks
1e-11..1e-14, and all five packages read back with template order, blend arithmetic and a byte-identical
unchanged column. New fits 2, new packages 5, agent uploads 0.

### Portfolio delivery, second batch (2026-09-26, pending)

`docs/round2_v6/RESULTS.md` §10; desktop `round2-V6-portfolio-r2-20260926/`. Two further iron
packages from the newly completed **N line** (`v36-s1-N-0008`, `v36-s1-N-0006`; raw-TabM small):

- `V6_PORT_IRON_N0008_W20` — `0.8*V36 iron + 0.2*N-0008`, local mean `-0.00955`, marginal E[max]
  `+0.00065`, rank 4; ZIP `0ec711819f6dc5482819b1fd9b93dbe1d5cce1117ea7fd2ea643dac0d2274bb8`
- `V6_PORT_IRON_N0006_W20` — `0.8*V36 iron + 0.2*N-0006`, local mean `-0.00869`, marginal `+0.00039`;
  ZIP `186302ffbcc548c2f6e798f374c71d3df0e9b356ba893a95cf197ee0f1b49720`

**The portfolio play is near saturation** (`EVIDENCE_STATUS.json ->
round2_v6_portfolio_2026_09_26.saturation_2026_09_26`). Completing 41 fresh N-line directions (410 fold
fits; including the most decorrelated time family found, `raw_mlp` at `rho` 0.69-0.72) raised `E[max]`
only from `+0.00748` to `+0.00843`, i.e. `+0.00095`, and the newly selected rank-4 entry has a *worse*
local mean (`-0.00955`). Marginal contributions of the top five are now `+0.0040/+0.0017/+0.0008/+0.0007/+0.0005`.
`E[max]` grows logarithmically in the number of directions, so multi-day continuation is worth roughly
`+0.008..0.01` in total and **cannot** approach the `0.1945` gap to first place. `ple_tabm` is excluded
as broken (`N-0060` fold-0 WMAPE 0.1712 against a 0.0384 baseline).

**Engineering rule (G0): any parallel fitting batch must pin BLAS/OMP/MKL/NUMEXPR thread counts before
launch.** With `workers=16` and only `OPENBLAS_NUM_THREADS=1`, each forked worker inherited 16 BLAS/torch
threads (16 threads, 432% CPU each), giving 128-256 threads on 32 cores, load 106 and *zero* completions
in five minutes, while single-fold timing said 6-31 s. Adding `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1` made 41 trials finish in ~110 s per seed.
