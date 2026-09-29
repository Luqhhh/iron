# E-COMPOSE engineering recovery

The original E-COMPOSE synthetic admission stopped22:30:04 CST on2026-09-29.
Its normally distributed air_volume/total_press_diff inputs violated the
already frozen positive ratio-denominator contract. It finished24 observed
leaf solver calls before failing; no official E-COMPOSE candidate fit occurred.
All original files, partial solver events and terminal failure are preserved.

The recovery shifts only those two synthetic inputs by+10, preserving the
synthetic latent signal, targets, IDs, split, seed and other inputs. The actual
reference feature builder now verifies that its ratio expressions are finite.
Both selector/refit workers and the fresh cold-audit subprocess use this same
fixture. DE3's original normal fixture is unchanged. This adjustment never
applies to official train/test data or model preprocessing.

Explicit --output and --preflight paths let the unchanged E-COMPOSE queue use
fresh private directories under
local/runs/independent-ensemble-checkpoints-20260929/recovery-r2. Admission must
match the queue, frozen spec and current source hashes. Each new run manifest
binds its successful admission report path and SHA256. No existing directory
or report may be overwritten. Candidate algorithms, checkpoint observers,
native early stopping, partitioning, RNG, float32, refit, fixed package weights,
cost budgets, promotion gates and the independent G1 auditor are unchanged.

Before fitting, the new E-only supervisor verifies the original DE3 manifest,
summary/audit hashes,90 complete artifact identities, nested hashes, original
data and source hashes. Only the preflight/routing files and regression test
may differ from that original source freeze; their original bytes are verified
against DE3's recorded starting commit906d31b and both old/new hashes are bound
in engineering-provenance.json. All model source and spec hashes must remain
unchanged. This preserves DE3's original audit at its own source version.
It does not relabel the old result as an audit using later recovery code.

The E-only recovery performs synthetic/resource admission, complete development,
independent audit and earned confirmation serially. It never repeats DE3 fits,
adds a new recipe, selects seed/epoch subsets or adjusts a threshold. It records
actual600-second observations and terminal events; no intervening health/metric
polling or automatic restart. No full-data fits, package, desktop write or upload.

Validation:1344 full locked Python3.12 tests passed,23 existing warnings.
Regression coverage verifies legal ratio inputs, unchanged synthetic signal
and targets, deterministic fixture reconstruction, queue/scope mismatches and
frozen-source tampering rejection. Zero-fit original-evidence verification
passes90 completed units. Private test log and initial corrected exception-text
test failure are retained underengineering-r2/.
