# Synthetic core status — 2026-09-30

G0: 1327 tests passed with the locked Python 3.12.12 path (23 warnings,
141.90 seconds). The targeted suite passed 17 checks. A separately started
Python process audited both saved synthetic model states with full-batch,
reverse-order and chunk differences exactly zero and no new optimizer call.
The actual synthetic ledger contains one estimator and two optimizer starts,
all complete and unchanged by that audit.

Private engineering receipt: `local/research/incumbent-reference-core-r1/receipt.json`
(SHA-256 `cc7b33dbc66bf3028e0a6a880cee29af491530c0daacbba31683df65a69858c2`). Logs, model states, predictions and ledger stay outside Git.

G1: unmeasured. Current best remains user-reported DE3 96.3749.
Official reference estimators/optimizer runs, full-data fits, new packages,
desktop writes and uploads in this stage are all zero.

The proposed next stage completes only the missing 20 estimators / 40
optimizers for the current reference. Its phase runner, original-cache and
source freeze, resource admission, complete audit and overlay writer remain
pending. This result neither permits bypassing those checks nor revises DE3's
historical failure to reach the 0.01 mechanism threshold.

## Complete phase engineering

The bounded phase, historical cache/source verification, fixed resource
admission, zero-fit complete audit and PTaRL-compatible overlay writer are
implemented. Final locked Python3.12 checks:1358 passed,23 warnings,191.91s.
Focused phase/core/native tests35 passed; separate freeze guard tests13 passed.
The complete synthetic executor/auditor exercises20 estimators,40 optimizer
starts and40 saved-state audits per run; its budget is unchanged by audit.
These are synthetic checks, not official model-quality evidence.

Exact-source/runtime receipt: `local/research/incumbent-reference-phase-r1/receipt.json`,
SHA-256 `f415901c178b180d92b08595cfcdae15cff80b9d4349c7b213ee48ecda43e886`. The first cost/memory observation failed at7075.55MiB available
against7168MiB required and is preserved; projected cost4388.3064s and worker
RSS1095.0117MiB met their bounds. Formal freeze/execution is still pending,
with0 official reference estimators/optimizer starts and0 packages/uploads.
Caps, DE3's historical decision and quality promotion gates are unchanged.

## Frozen execution launched

Service iron-incumbent-de3-reference-r2.service started10:03:25 CST with
resource admission passed at8598.68MiB available. All60 reused development
states passed pre-fit cold verification;10 native J42 replays are exact.
The fixed20-estimator/40-optimizer execution is running, not a G1 result.
The failed wrong-root r1 freeze is preserved with0 estimator starts.
Use the original RFM manifest's reference_root=/home/lux1/iron.
Full identity and monitoring cadence: [LAUNCH.md](LAUNCH.md).

## First scheduled monitor

At `2026-09-30T02:14:14.950113+00:00`, service iron-incumbent-de3-reference-r2 remained
active/running, MainPID417199. Estimators:12 started,8 completed,4 active;
optimizers:24 started,20 completed,4 active. No failed estimator/optimizer
receipt or terminal failure artifact. Budget remains20 estimators/40
optimizers. No between-check polling; next observation no earlier than
`2026-09-30T02:24:14.950113+00:00`. G1 remains unmeasured.
