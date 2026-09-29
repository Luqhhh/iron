# V48: separately frozen factorized-spline epoch extension

V47 has completed and failed its gates. Its audit passed 60 units / 80 models,
and all 20 AHOFM4 inner selectors reached 400 epochs. Between the first 300 and
full 400 epochs, best calibration MAE improves in every candidate unit, averaging
3.5811% iron /4.2642% time. Selected epochs are 385-400. This is evidence of
truncation while calibration errors were still falling, not proof of convergence
failure, expected platform gain or permission to change the V47 decision.

V48 changes exactly one model/training field: max_epochs 400 -> 2000 for both
AFM2 control and AHOFM4 candidate. Keep rank, basis, factors, optimizer, learning
rate, RNG, minibatches, penalty, patience 50, selection rule and all other settings
unchanged. There is one longer schedule, no duration grid or adaptive extension.
Only AHOFM4 is eligible for confirmation. No retrospective promotion of AFM2.

## Matched prefix and evidence requirements

Run each fit from its original initialization. During each new fit, capture the
actual parameter state at the old selected epoch and its training history through
the old stopping epoch. These must exactly equal the frozen V47 parameters and
history, apart from explicitly different metadata such as peak RSS. Reconstruct
the prefix model using newly computed training-only basis and normalization,
then independently verify its predictions against V47 saved predictions. Do not
load old model parameters into new training, copy an old checkpoint as evidence,
or add extra fits solely to reproduce prefixes. Persist prefix snapshots separately
and hash-bind them to the new append-only unit completion record.

For the selector, the 400-epoch prefix snapshot is the best selected state within
the old 400-epoch run, not necessarily the raw epoch-400 state. For the fresh
outer refit, capture its actual state after the old selected number of epochs.
If AFM2 stops early at the same old epoch, its entire fit should reproduce V47.
If any prefix differs, stop admission/audit; never accept an approximate prefix,
relax cold tolerance, overwrite old artifacts or restart a failed evidence unit.
Confirmation on new split seeds has no old prefix and must state that explicitly.

## Evaluation and admission

Preserve the V47/current-reference protocol: seeds 42/3407, complete five-fold
coverage, both targets and both arms; inner seed 27001 fold 0 selects epochs and
blend weight, followed by a fresh outer refit. 40 candidate units /80 fits, 20
verified reference units reused. Verify V47 manifest/audit/source/data/row/fold
identities and every old unit before execution. Preprocessing learns only from
its training partition; query targets are removed; no cross-seed OOF averaging.

G0: existing analytic/gradient/isolation tests plus prefix tamper, checkpoint
selection, deterministic trace and settings tests; full locked Python 3.12 suite.
Run both full 2000-epoch synthetic fits on the identical V47 generator, capturing
and verifying their exact old 400-epoch prefixes. Both must beat the same median
constant and AHOFM4 must beat AFM2; no synthetic tuning. Independently audit
saved inference through the NumPy ANOVA DP, reversed/chunked rows and old prefix
states. Resource criteria unchanged: conservative 2x development timing <=7.61h,
worker peak <=1536 MiB, four-worker available RAM >4 peaks+1024 MiB.

G1: both development seed gains >0, mean package increment >=.01 and larger than
AFM2; then at most one finalist per target uses derived seeds 271828/314159.
All four seed gains positive, seed-level paired LCB95>0, development package
score >=96.25; fold counts descriptive. No gate, reference or target changes.
Report V47-relative and incumbent-relative results, training cap hits and failures.
No claim that old weights, longer epochs or offline scores predict platform gain.

Monitoring every 600 seconds, no between-check health/metric polling. Actual
completion events may trigger independent audit. Failed evidence retained.
No external data/pretrained weights, full-data fits, automatic package, desktop
write or upload. User-reported best remains 96.3727; milestones 96.4 /96.45 /96.5
are not achieved. The optimization goal remains active.
