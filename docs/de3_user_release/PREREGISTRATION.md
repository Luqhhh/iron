# User-requested isolated DE3 iron release (2026-09-30)

The user explicitly requested a submission package and desktop delivery for
the highest recent local gain among EMA and independent three-model averaging.
The selected single-column candidate is DE3 iron: development gains
+0.004381694674 / +0.002578715152, mean +0.003480204913, both complete seeds
positive. EMA iron mean is +0.002709168619 and EMA time +0.002168812097.
E-COMPOSE iron is +0.001488613937; augmentation and RFM have no finalist.
This comparison covers the recent completed mechanisms, not all historical
gains or untested combinations.

This is an explicitly requested platform exploration, not automatic promotion.
DE3 remains below its frozen +0.01 mechanism gate, with no derived-seed
confirmation. Preserve its original `no finalist` and candidate-tier decisions.
The release does not relax any threshold or assert a platform forecast.

Use V32_TIME_A60V7_50, user-reported 96.3727, as the hash-pinned parent. Replace
only its half-weight V12 joint iron component with the equal average of native
joint models trained with seeds 42, 104729, 130363. Keep the native inner split
seed 42, preprocessing, architecture, optimizer, stopping and full-data fresh
refit protocol unchanged. Keep every time field string verbatim. Do not add EMA,
change weights, mix outer split seeds, or generate a two-target package.

Reuse the audited full-data seed42 V12 model and its exact saved test prediction.
Fit only the two missing full-data members, each with one native inner selector
and one fresh full-data refit. No new CV fits or derived split labels are needed.
The budget is two full-data estimators / four optimizer runs, one private ZIP,
and one new desktop directory after verification. Preserve all historical files.

Before fitting, bind hashes of original DE3 summary/audit/manifest, data,
runtime, native model/predictions, parent ZIP and current source. Verify native
recipe, training IDs, preprocessing and exact saved prediction replay. Audit
the new checkpoints for train-only scaling and native epoch selection. Verify
fresh-process label-free inference, exact full-batch predictions, row/chunk
invariance at the existing 1e-6 relative tolerance, exact component-replacement
readback, 322 template-ordered unique IDs, finite nonnegative outputs, zero
time-string mismatches, and source/artifact identities unchanged on completion.

Destination: desktop `submission-DE3-IRON-20260930`, never overwrite it.
Models, predictions, machine reports and ZIPs stay private under `local/`.
The user uploads and returns the score; agent uploads remain zero.
