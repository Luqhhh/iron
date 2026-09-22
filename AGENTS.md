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

The `md/` tree is the archived original implementation package. Active configuration and status live at repository root under `configs/`, `docs/`, and `EVIDENCE_STATUS.json`.

The `baseline-v0.1-reproducible` tag is the immutable engineering baseline. Do not change its model, feature, semantic, source, or acceptance contracts in place. Start model-quality work as optimization-v0.2 on a separate branch and preserve baseline comparisons.
