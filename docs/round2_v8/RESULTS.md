# V8 execution status (2026-09-26)

Status: complete-coverage development batch running; no quality result yet.
Four frozen target/mechanism units, 40 outer fits, two optimizer runs per fit.
Eight workers run with BLAS/OMP/MKL/NUMEXPR and torch pinned to one thread each.
Private evidence: `local/runs/round2-v8-feature-attention/development-r1/`.

**G0:** locked Python 3.12 test path **990 passed, 3 warnings**. New tests
verify that uniform attention equals zero-logit attention, learned Q/K receive
gradients while uniform Q/K do not, repeated fits are deterministic, and a
fresh process reproduces predictions without training data. Reverse, chunked
and single-row queries are also checked. Only the fixed code package was added;
no existing dependency version was changed and no pretrained weight was used.

The V7 secondary reference was checked against its original data/fold identities
and per-fold prediction hashes before the batch launched. Comparisons will
report A35, Q20, and the already-qualified V7 time column separately.
The N-0048 time reference and all four V36 experts were additionally checked
against their historical ledgers: ten member/seed records reproduce the original
data identity, five-fold assignment, pooled WMAPE and every fold WMAPE to 1e-12.
Private `reference-audit-r2.json` records the checks and current hashes of 15
source files. These historical ledgers did not store prediction-file hashes;
this is identity/metric verification, not a claim of historical byte identity.
The first audit parser expected `fold_seed` in the older V36 ledger, which uses
`batch_id`; that parser failure is preserved separately. No source prediction
was changed and no additional fit was needed.

An increment over A35 that does not survive the V7 comparison cannot trigger
time-side confirmation. Existing four-seed, local working-gate and release
constraints remain unchanged. No models were fitted on all 2754 training rows,
no packages were generated, and no desktop or platform writes occurred.

**G1:** pending. Current platform best remains user-reported A35 = 96.3366;
the 96.4 objective has not been established. V7 periodic time remains a
four-seed-qualified candidate-pool entry, not a released platform best.
