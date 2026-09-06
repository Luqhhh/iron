# Repository execution contract

This repository implements the frozen `baseline-v0.1`. Unless the user explicitly opens a separate optimization phase:

- do not change CatBoost parameters, iteration count, targets, loss, feature windows, post-processing, or acceptance thresholds;
- do not read November 2024 targets from `train_samples.csv` or `tap_history_train.csv` in development workflows;
- protected-label access requires `configs/protection.yaml`, a frozen-manifest digest, and an append-only local access ledger;
- use the same as-of feature builder for train, validation, and prediction;
- never overwrite run directories or remove failed evidence;
- keep official CSV/XLSX data, dictionaries, models, predictions, reports, ledgers, and submissions out of Git;
- run the locked Python 3.12 test path before claiming authoritative reproduction;
- report G0 engineering status separately from G1 model quality.

The `md/` tree is the archived original implementation package. Active configuration and status live at repository root under `configs/`, `docs/`, and `EVIDENCE_STATUS.json`.

The `baseline-v0.1-reproducible` tag is the immutable engineering baseline. Do not change its model, feature, semantic, source, or acceptance contracts in place. Start model-quality work as optimization-v0.2 on a separate branch and preserve baseline comparisons.
