# V30 execution-only concurrency amendment

Naming update (2026-09-28): local V27 is now **V30**. This is a public-name
change only; original run paths, hashes and delivery IDs remain frozen. See
[the migration record](../round2_round_numbering.md#7-exact-public-migration-and-frozen-identities).

Recorded during the first two reference units, before any candidate evaluation.
The first two B0 refits take more than ten minutes and each active parent uses
one CPU core after its initial component pool. Read-only process inspection
confirmed continuing computation, not a deadlock.

Raise **reference concurrency only**, from two to at most six calls. Precompute
the eight far-tail units (seed 3407, folds 1..4, calibration and outer) with four
additional workers. The unchanged primary controller handles the other twelve
reference units and verifies/reuses the tail when its queue reaches them. The
candidate worker count stays four. Initial nested component pools can use at
most 24 workers, below the machine's 32-core capacity; numerical thread counts
remain one.

This supersedes the execution-only `concurrent_reference_calls: 2` ceiling for
the reference phase. It changes **zero** model parameters, epochs, folds,
training rows, calibration rules, thresholds or fit budgets. There are still
20 development B0 refits and 60 candidate outer fits. The original scientific
specification/source hashes and every existing run directory remain unchanged.
No in-progress unit is replaced, retried or overwritten. An ownership collision
fails visibly at atomic directory creation instead of allowing a duplicate fit.

Implementation: `scripts/run_v30_reference_tail.py`, whose hash and schedule are
appended to the private run before execution. A dry run verifies source/data
hashes, runtime versions and all eight unclaimed units before launch. Existing
unit tests cover immutable/failed-unit rejection; the underlying fit function is
the same implementation that passed 1169 tests. This is a scheduling amendment,
not a new candidate or evidence obtained by changing the frozen recipe.

The candidate stage may overlap reference work: `scripts/run_v30_candidate_head.py`
runs only the first four seed-42 cells (24 of the same 60 candidate units), at
the already declared four-worker limit, after each cell's verified calibration
reference becomes available. These cells precede six further reference cells,
leaving a scheduling buffer before the primary controller reaches candidates.
The outer reference is required for scoring but not for candidate fitting.
All other candidate units remain with the primary controller. Maximum combined
model workers is 28 (24 reference constituent workers plus four candidates),
with numerical thread counts still one. The auxiliary scheduler and its exact
unit list are hashed in an append-only execution record; atomic unit claiming
and the original scientific identity checks are unchanged.

An optional bridge scheduler (`scripts/run_v30_reference_bridge.py`) may fill
two reference slots after at least six of the eight tail units have completed.
It precomputes only seed-3407 fold 0, calibration and outer, while the primary
queue still has at least four seed-42 reference units unfinished. If that queue
buffer disappears before launch, it skips and leaves both units to the primary
controller. At most two tail, two primary and two bridge calls then coexist:
the six-call ceiling and all fit budgets remain unchanged. Its source hash and
launch/skip outcome are recorded separately in the private run. This amendment
uses only completion status and timing; it does not inspect validation scores.
