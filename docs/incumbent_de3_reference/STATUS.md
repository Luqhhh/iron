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
