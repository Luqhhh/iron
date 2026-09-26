# V8 feature attention: complete development evidence (2026-09-26)

All 40 predeclared outer fits completed without failure: four target/mechanism
units, seeds 42/3407, five folds each, with inner epoch selection and a fresh
full-training-fold refit. No external data or pretrained weight was used.
Private evidence: `local/runs/round2-v8-feature-attention/development-r1/`.

## G1: attention helps, but the incremental effects are small

Values below are full-package score-point changes, averaged over folds within
each complete split. Blend weights are selected on the other split's OOF
predictions; vectors from different split seeds are never averaged.

| Target | Mechanism | A35 gain, seed 42 | A35 gain, seed 3407 | A35 mean | Additional mean over V7 | Development tier |
|---|---|---:|---:|---:|---:|---|
| Iron | Uniform | +0.002554 | -0.000260 | +0.001147 | same iron reference | exploration |
| Iron | Learned | +0.003866 | +0.001845 | +0.002856 | same iron reference | formal |
| Time | Uniform | 0.000000 | -0.002487 | -0.001244 | 0.000000 | not shortlisted |
| Time | Learned | +0.003006 | +0.004145 | +0.003575 | +0.000718 | formal |

The time learned-attention model adds only +0.000184 / +0.001252 over the frozen
V7 candidate-pool time column, with weights 0.05/0.05 and 5/10 positive folds.
Its Q20-relative mean is +0.004794. The iron learned model uses weights 0.10/0.10
and improves 8/10 folds. Iron uniform fails both-split and improved-fold gates;
time uniform has a negative mean. The frozen `candidate_tiers` classification
is descriptive development evidence, not four-seed promotion or release.

Learned attention improves standalone WMAPE over its uniform control on both
targets and both split seeds, but standalone quality is not the admission test.
The strongest remaining increment is iron learned attention at +0.002856.
The predeclared ranking therefore selects **iron `ft_learned`** as the sole
confirmation candidate, ahead of time's V7-relative +0.000718. Selection does
not imply a useful platform effect. The gains remain small, and neither
candidate is scheduled for platform testing.

Two uniform-control fits reached the 240-epoch selection budget: iron seed42
fold4 (best epoch225) and time seed42 fold3 (best epoch232). All learned-attention
fits stopped before that cap. This is evidence about the frozen compact
configuration, not proof that every attention architecture is exhausted.

## G0: execution and independent arithmetic passed

Locked Python 3.12 test path: **993 passed, 3 warnings**. Tests cover uniform
attention versus zero logits, Q/K gradient attribution, deterministic fits,
row/chunk/single-query consistency, fresh-process synthetic-model inference,
complete-pool selection against the correct incremental reference, and removal
of both target columns from every baseline/candidate query frame.
Only the fixed source-code package was added; existing dependency versions
were preserved. The cold-process test is synthetic, not a full-data release audit.

Private `audit-r1.json` verifies all 40 prediction hashes, full OOF coverage,
outer/inner training-ID digests, selected epochs, implementation hashes, runtime
versions and V7 reference hashes. Independent arithmetic reconstructs all three
reference comparisons, sparse-grid alpha choices, per-fold gains, paired seed
statistics and confirmation eligibility to 1e-12. No audit refit was needed.

The N-0048 time reference and four V36 experts were additionally checked against
their historical ledgers: ten member/seed records reproduce data identity,
five-fold assignment, pooled WMAPE and each fold WMAPE to 1e-12.
`reference-audit-r2.json` records these checks and current hashes of 15 source
files. Historical ledgers did not store prediction-file hashes, so historical
byte identity is not claimed. The first parser's missing-`fold_seed` failure
is retained; V36 uses `batch_id`. A premature descriptive iron summary was also
refused while one fit was pending; the later complete iron summary is retained.
No original prediction or failed evidence was overwritten.

## Next offline check and release boundary

The selected iron candidate needs same-fold iron baselines at seeds 7777/12011.
The V5 replication caches contain time only; they cannot be substituted for iron.
Fresh `V36FixedRecipeFactory` fold fits are therefore required. Before trusting
new baseline outputs, the first fit's time column will be compared with the
existing same-fold V5 cache as an engineering control. Both reconstructed
baseline columns will be preserved, although only the selected iron candidate
enters confirmation. Execution details are frozen in `CONFIRMATION.yaml` before
those fits; the V8 model recipe and development decision stay unchanged.

Four positive split seeds and a positive seed-level LCB95 remain mandatory;
fold counts are descriptive only. The local working gate remains 96.25.
Current platform best remains user-reported **A35 = 96.3366**. The **96.4**
objective is unproven. No full-data fit, package, desktop write or upload occurred.

Confirmation implementation is running its first same-fold reference control
under the frozen execution specification. No candidate confirmation result is
yet available. The baseline manifest binds recovered source, source catalogues,
weights, configuration files, runtime versions, fold identity and data identity.
Both baseline columns will be retained for every successful fold.
