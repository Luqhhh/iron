# V9 execution status (2026-09-26)

All 40 predeclared outer fits completed without failure: four target/recipe
units, seeds42/3407, five folds each, with inner epoch selection and fresh
full-training-partition refitting. Private evidence is in
`local/runs/round2-v9-realmlp/development-r1/`.

## G0

Locked Python 3.12 test path: **999 passed, 23 warnings**, including 20
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

## G1: a stronger iron direction, still development evidence

Score-point changes use fold means within each complete split. Sparse-grid
weights are selected on the other split's predictions; no OOF vectors are
averaged across split seeds.

| Target | Recipe | A35 seed42 | A35 seed3407 | A35 mean | Increment over V7 time | Development tier |
|---|---|---:|---:|---:|---:|---|
| Iron | TD | +0.007482 | +0.001497 | +0.004490 | same iron reference | formal |
| Iron | TD-S | +0.005500 | +0.004964 | +0.005232 | same iron reference | formal |
| Time | TD | +0.010876 | +0.013264 | +0.012070 | +0.002320 | formal |
| Time | TD-S | 0.000000 | -0.000601 | -0.000301 | 0.000000 | not shortlisted |

The predeclared incremental ranking selects **iron `realmlp_td_s`** as the sole
confirmation candidate. Its alpha is 0.10 on both splits, with 10/10 positive
folds. The V8-iron diagnostic also improves on both splits, +0.003224/+0.003739,
mean +0.003482, 8/10 folds. Its standalone WMAPE is worse than full TD
(0.041956/0.042371 versus 0.039539/0.041321), yet its blend gain is larger and
more consistent: incremental usefulness remains the selection criterion.

Time full TD has a larger A35-relative mean, but most of that direction is
already covered by V7. Its additional V7-relative gains are only
+0.001830/+0.002810, with 7/10 positive folds, so it is not the extra candidate.
Time TD-S improves relative to historical Q20 (+0.001352 mean) but loses against
the actual A35 reference; it fails continuation. No old reference is substituted
to admit it. These tiers are descriptive development classifications and do not
establish four-seed promotion or platform rank.

The selected iron direction's two-seed pooled package score is **96.214003**,
below the unchanged 96.25 working gate. Even a successful four-seed check would
leave it in `candidate_pool` under the frozen release rule. The 10/10 folds are
descriptive, not ten independent datasets or a replacement for seed-level evidence.

Private `audit-r1.json` independently verifies all 40 prediction hashes, full
row coverage, inner/outer training identities, median/vocabulary metadata,
model/source/runtime identity and all reference hashes. Direct arithmetic
reproduces each of the four reference comparisons, alpha choices, fold and seed
gains, paired lower bounds and eligibility to 1e-12. No audit fit was needed.

## Four-seed confirmation: qualified local candidate pool

All ten additional candidate fits completed. No new baseline fit was needed:
V8's ten same-fold baseline files were reused only after their complete ledger
entries, hashes, row positions, data identity and fixed-recipe identity passed
checks. A live incomplete final ledger line is ignored until terminated, and
partial, duplicate or foreign fold coverage cannot trigger evaluation. V8's
attention candidate failed its quality gate; that does not invalidate its
separately verified reconstruction of the frozen baseline.

| Split seed | A35-relative iron score gain | Held-seed alpha |
|---|---:|---:|
| 42 | +0.005500 | 0.10 |
| 3407 | +0.004964 | 0.10 |
| 7777 | +0.004779 | 0.10 |
| 12011 | +0.002313 | 0.10 |

Mean **+0.004389**, seed standard deviation0.001418, seed-level paired LCB95
**+0.002721**, **4/4** positive seeds: the frozen four-seed rule passes.
The fold profile is **15/20 positive (75%)**, below the descriptive80% reference;
per the frozen rule it is disclosed rather than converted into a new veto.
These are previously used split seeds of the same labelled dataset, not newly
independent test datasets.

Private `confirmation-r1/audit-r1.json` independently reconstructs all 20
candidate prediction files, reference identities, sparse-grid choices, fold
and seed gains and paired lower bound to 1e-12. No audit fit was needed.
The fixed two-development-seed package score remains **96.214003 <96.25**.
Disposition: **four-seed-qualified candidate_pool only**. The local working gate
was not changed and no release exception is inferred from a positive LCB.

This supplies a second measured direction alongside V7 time. It remains a small
local increment and is not a platform forecast or permission for a two-target
package. Current platform best remains user-reported **A35 =96.3366**, and the
**96.4** objective is unproven. No full-data model, package, desktop write or
upload occurred.
