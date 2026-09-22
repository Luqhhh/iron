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
