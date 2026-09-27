# V16 complete: no incremental candidate; work stops here

All20 development joint fits and the one frozen V12 control replay completed.
The independent audit passed. Both recipes have **zero incremental blend gain**
against the V12 platform recipe and target-specific CURRENT references on both
complete development splits. No candidate qualifies for confirmation. The user
instructed `v16完成后停止后续工作`; this round is complete and further optimization
stops after evidence publication. No V17 is started.

## G0 engineering

Locked Python3.12: **1109 tests passed,23 warnings**, including17 new targeted
tests. Control seed42/fold0 reproduces both V12 outputs with maximum difference0.
The eight workers each used one thread. Official-data outer fits21 in total:
20 candidate fits plus1 control, with2 optimizer runs per fit for inner epoch
selection and fresh full-outer-training refitting. Failures0. Confirmation fits0,
full-data fits0, packages0, desktop writes0 and agent uploads0.

The independent audit verified20 complete prediction matrices, source/spec/data/
fold hashes, both installed implementation hashes, training-row identities,
inner and outer target scales, numeric preprocessing means, mask metadata,
the nested weights and gains, candidate tiers and the empty finalist selection.
83 development reference files and145 four-seed reference files were checked
without model fitting. Those reference checks do not constitute V16 confirmation.

Both recipe modules contain83504 parameters. Uniform masks remain exactly
constant across rows, with zero attention-projection gradient and all features
active. Learned masks are sparse and input-dependent: nonzero feature fractions
range0.08235–0.23992 across fitted steps, maximum per-feature row standard
deviation ranges0.20765–0.32101, and attention gradient norms0.22911–0.68176.
The mechanism is active; the absence of blend gain is not a disabled-selector bug.

Uniform selected epochs range102–231 with0/10 cap hits. Sparse selected epochs
range324–398, with **9/10 reaching the400-epoch selection limit**. No training
budget is extended after seeing the result. This limits conclusions to the
frozen compact recipe and budget; it does not prove the whole TabNet family
cannot improve with other settings.

## G1 quality

V12_PLATFORM means the actual released V12 iron A50 with original A35 time.
CURRENT means V12 iron A50 / delivered V7 time A50; V7 remains unreported on
the platform. Every cell below uses complete five-fold predictions, never a
folds0/1 screen or cross-seed OOF averaging.

| Target | Recipe | Standalone WMAPE seed42 /3407 | Gain vs V12_PLATFORM | Gain vs CURRENT |
|---|---|---:|---:|---:|
| Iron | Uniform masks | 0.056616 /0.055906 | 0 /0 | 0 /0 |
| Iron | Sparse masks | 0.052850 /0.051474 | 0 /0 | 0 /0 |
| Time | Uniform masks | 0.063494 /0.063389 | 0 /0 | 0 /0 |
| Time | Sparse masks | 0.059657 /0.059916 | 0 /0 | 0 /0 |

All CURRENT and V12_PLATFORM nested weights are0. All four target/recipe units
are `not_shortlisted`. Sparse masks improve standalone errors over the uniform
control, but that is insufficient: the admission criterion is incremental blend
gain. Against the older A35/A60/Q20 iron reference, sparse masks give mean
−0.000617 (weights0.05/0); other secondary comparisons are0. No positive
two-seed finalist exists, so the preregistered confirmation branch is not run.

The current highest platform score remains **user-reported V12=96.3526**,
not an independently verified receipt. The above96.4 goal is **not achieved**.
Existing V12 and V7 package bytes, release decisions and global gates remain
unchanged. No pending feedback is fabricated and no account quota is inferred.

Private evidence: local/runs/round2-v16-sequential-masks/development-r1 contains
manifest.json, control.json, the append-only fit_ledger.jsonl,20 prediction
matrices, summary.json and audit-r1.json. Execution handle81665 and audit
handle81340 returned exit0; neither process remains running. Models, predictions,
private audit files and reports remain outside Git.

On publication, pause the active optimization goal pursuant to the user's
instruction. Resume only on a subsequent explicit user request.
