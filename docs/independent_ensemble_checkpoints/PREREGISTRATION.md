# Independent retraining and deployment-aware checkpoints

The user authorized full implementation of `iron_next_phase_20260929.zip`
on 2026-09-29. Its SHA-256 is
`c3cfdd16f1476f04b7fa843e191e377102913fcf757b5c341430d0df32c39da6`.
SOURCE_PLAN.md preserves the supplied plan. This is a new bounded phase on
`codex/independent-ensemble-checkpoints`, based on `8567911`, in an isolated
worktree. The existing two serial queues are terminal; local code/configuration
search found no equivalent DE3 or E-COMPOSE experiment. No active iron user
service existed at the initial execution inventory. These observations are not
an exhaustive audit of an unconnected remote machine.
After fetching origin, all95 remote refs were searched in their active source
and configuration trees for the two fixed new seed identities and E-COMPOSE;
there were zero matching files. The latest score-bearing handoff branch still
records96.3727. No same-protocol formal negative evidence was found. A remote
training host outside these repository/service inventories was not observed.

The fixed platform reference remains user-reported
V32_TIME_A60V7_50=96.3727, not independently verified. Milestones are
96.4, 96.45, 96.5; no constant local/platform offset is assumed.

## Frozen protocol

DE3 runs first, followed by its independent audit and earned confirmation.
Only then does E-COMPOSE run its synthetic admission, development, audit and
earned confirmation. Four candidate workers or one reference factory with
four internal workers is the execution limit. Queues and factory calls never
run simultaneously. Every source, runtime, data file, row order, outer fold,
native inner partition and prediction artifact is hash-bound. A failed stage
stops the supervisor without automatic retry or overwriting any run.

Iron stays `.5 V36 + .5 V12_joint`; time stays
`.2 V36 + .3 N0048 + .5 V7_periodic`. Only the native V12 iron or V7 time
component is replaced. Native float32, preprocessing, 16 TabM members,
joint/single-output mean MSE, AdamW, schedule, patience and 240-epoch cap
remain frozen. Joint fitting always uses both outputs, but only iron is replaced.

DE3 freezes training RNG seeds 42, 104729, 130363, and arithmetic weights
1/3 each. Outer development seeds remain 42/3407, all five folds. Native
inner partition seed remains 42 for every training seed. The independently
audited original twenty seed42 BASE units are reused only after exact
source/runtime/data/partition/preprocessing/target-scale/cold-prediction checks.
There are forty new outer members, eighty selector/refit trajectories. Seed1042
single-fold diagnostics are not members, candidates or confirmation data.
No seed subset, fitted seed weight, or K=8/16 continuation is permitted.

E-NATIVE, E-TARGET and E-COMPOSE observe the same native training trajectory
and every epoch it actually visits. Native stopping remains unchanged.
E-TARGET and E-COMPOSE are prospectively eligible on both targets; at most
one earns confirmation per target. The latter selects raw target MAE of
`r + .5 f_epoch`. The former selects raw component target MAE.
Both use exact argmin and earliest epoch on exact ties. E-NATIVE preserves
the original min_delta=1e-5 selection logic. Positive scaling equates the
time objectives mathematically; this does not make a tolerance-based native
selector identical to an exact argmin under float32 arithmetic. Epoch equality
and the distinction are measured and reported without changing native logic.

For each outer training partition, native inner fold0 is C and its complement
is A. Candidate selectors and the V36/N0048 background train on A only,
including their own internal partitions. C labels are used only for selecting
checkpoints, never fitting the background. Outer query Q labels never enter
training or selection. The existing V30 calibration cache uses a different
inner split, so it is not reused. A shared legal background is refitted for
each of ten A/C pairs: thirty component pipelines per factory, three hundred
in development. Leaf solver calls have an additional append-only ledger;
pipeline counts are never presented as counts of internal optimizer fits.

After selection, each distinct selected epoch is freshly initialized and
refitted on A+C. Identical selected epochs share the identical refit within
that unit. Original native refits may be reused at the exact same epoch and
training/runtime identity. Twenty selectors and at most forty new development
refits are budgeted. All three selected selector states and all per-epoch
calibration predictions are saved for independent arithmetic/cold audit.

## Decisions and costs

Screening requires complete five-fold coverage at both development split seeds.
An eligible candidate must have two positive full-split gains, mean isolated
package-score gain >=.01 and strictly beat native BASE (whose gain is exactly
zero). Positive means below .01 remain small-gain evidence. E-COMPOSE is also
compared to E-TARGET; eligibility and evidence of a composition mechanism
are distinct fields. Tie preference is E-TARGET before E-COMPOSE.
The repository candidate-tier classification remains descriptive and cannot
override this mechanism gate.

Only audited finalists consume the existing designated confirmation split
seeds 271828/314159. Only eligible target components are fitted there.
Reference factories fit the same frozen recipe on the supplied outer partition.
Verified matching DE3 confirmation references may be reused by E-COMPOSE.
Four positive complete split seeds, positive seed-level paired LCB95 and the
unchanged development package working gate 96.25 control promotion. Fold
results are descriptive. Reused data/splits and adaptive prior experiments mean
these rules do not prove four independent samples or selection-error control.
Maximum conditional DE3 member fitting is60 complete members/120 candidate
optimizer trajectories across two eligible targets; native controls are included
in that count. E-COMPOSE confirmation needs at most20 selectors and60 distinct
refits when no identical verified refit already exists, plus separately counted
outer and inner reference factories. These are conditional costs, not fits
authorized for nonfinalists.

Synthetic resource/learnability checks precede each queue's real fitting.
The conservative per-queue resource limit is the existing 7.61 hours and
1536 MiB per candidate worker, with four-worker RAM admission. E-COMPOSE
explicitly includes measured background factory time in its projection.
BLAS/OMP/MKL/NUMEXPR threads are pinned to one before Python import.
All candidate selector/refit, reused native units, reference factory and
reference pipeline counts are separately recorded; raw reference solver
start/completion/failure events are retained too.

The standalone supplied DE3 arithmetic auditor keeps
`provenance_verified=false`. The repository auditor verifies partition,
source, model and cold-prediction identities separately. It never fits.
No prediction vectors are averaged across outer split seeds. Full-split gains
are averaged only after scoring each split independently.

The durable supervisor records observations at actual 600-second child-wait
timeouts and actual completion/failure events. No intervening training polling.
The pipeline stops after these two bounded queues. No mechanism combination,
additional training-seed/epoch/old-weight scan, full-data fit, package,
desktop write or agent upload is authorized. Current best package bytes and
all historical negative evidence remain intact. The user performs platform
submissions and returns scores.

## Execution

Use the existing locked Python3.12 CPU environment with `--no-sync` and explicit
runtime checks. From this worktree, with PYTHONPATH=src and all four thread
variables set to1:

```text
uv run --locked --no-sync --python 3.12 pytest -q
.venv/bin/python -m bf_tap_r2.independent_checkpoints_workflow
```

Private run root: `local/runs/independent-ensemble-checkpoints-20260929`.
Each new stage directory refuses overwriting and implicit resumption.
Implementation/tests/G0 and real model-quality/G1 results must be reported
separately. A successful synthetic/test run is not evidence of platform gain.
