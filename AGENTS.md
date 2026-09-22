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
- The final Top5 batch is no longer pending. Future optimization, if requested, compares against AJ3 iron/full B3 time and 96.1259; do not keep using C2 iron as the current preferred reference. Details: docs/round2_final_top5/FEEDBACK.md. Do not infer account quota or automatically generate a two-target combination from unreported time results.

## Platform test priority preference (2026-09-23)

- The user prioritizes packages with larger offline gains against the current target reference or a clearly justified, important exploration question. Give small-gain, closely related variants lower platform-test priority; do not fill slots merely to use the daily allowance.
- The observed final Top5 platform/local gap is about 0.09–0.11 score points: modest in absolute size but enough to reorder closely scored candidates. Do not generalize this to a fixed score correction or claim only low-scoring models can change order.
- This manual release/scheduling preference supersedes automatic formal-first upload ordering, not candidate-tier classification, frozen gates, exploration caps, or historical evidence. Preserve small-gain candidates without promising they will be tested. See docs/candidate_tiers.md for the controlling explanation.
