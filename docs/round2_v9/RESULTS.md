# V9 execution status (2026-09-26)

The complete-coverage development batch is running: four frozen target/recipe
units, 40 outer fits, two optimizer runs per fit. Four single-threaded workers
use only the approved round-two training data. Private evidence is in
`local/runs/round2-v9-realmlp/development-r1/`.

## G0

Locked Python 3.12 test path: **997 passed, 23 warnings**, including 20
PyTorch Lightning deprecation warnings from the new synthetic tests.
Both author recipes pass repeat-fit determinism, reverse/chunk/single-query
consistency, and exact cold-process prediction equality. The input adapter
excludes sample IDs and targets, imputes using training-only medians, and maps
unknown spouts to all-zero indicators.

An observer on the author's numeric median fitter verifies the exact training
row multisets: the inner training rows during selection and all outer-training
rows during the fresh refit. The initial guard incorrectly expected a single
fitter call and preserved row order. Native TD additionally visits an empty
feature branch; no-validation refit permutes rows. The corrected guard checks
all nonempty fitted inputs by exact row multiset. Both initial failed test logs
are retained under `local/reports/v9-targeted-tests-r1.log` and `-r2.log`;
the corrected run is `-r3.log`. No official-data fit started before it passed.

Native shuffled `drop_last=True` minibatches are retained. A full-training-fold
refit means the full training partition is supplied; it does not assert that
every row contributes a gradient in every epoch. The original 256-epoch schedule
horizon remains fixed even when the refit stops at an earlier selected epoch.

All 106 existing distribution versions were preserved; 12 runtime/code packages
were added under exact constraints. No pretrained weights or external dataset
were downloaded. Before fitting, the runner verified the historical primary
reference hashes and the V7/V8 diagnostic data, fold and prediction identities.
The manifest also freezes all author source-file hashes and resolved recipes.

## G1

Pending complete development results and independent arithmetic. Current
platform best remains user-reported A35 = **96.3366**; the **96.4** objective
is unproven. V7 is a qualified local candidate-pool direction, and V8 iron is
undergoing additional split validation. Neither becomes a new platform best
without feedback. No full-data model, package, desktop write or upload occurred.
