# OPT-19: causal pseudo-history, zero fit

U0 is the unchanged frozen R2. U1 uses the same saved OR/HR models,
E09/E04 80/20 clipped predictions, schemas, last30/last100 windows and parameters.
Only the inference history state changes. Base histories and active R2 are immutable.
Each origin runs all months consecutively before any scoring labels are loaded.
DEV_LONG/SHORT are slices of the corresponding complete origin trajectories.
No historical validation label is appended. A pseudo row is made only from an
already generated U1 prediction, and becomes visible when predicted completion
is <= the current reference time. Equal timestamps share one snapshot and are
registered together after prediction. Sorting by timestamp and sample ID fixes ties.
All surviving history features (including availability age) use this same state.

The exact gates are frozen in experiment.yaml. Nonfinite/negative output or
invalid completion aborts. Per-sample target distributions, monthly distributions,
pending completion overlaps, conservative ancestry depth, deltas and state hashes
are retained for explicit recursive drift inspection in addition to numeric gates.
Depth is 1 + maximum depth of visible pseudo ancestors (a conservative upper bound,
not an assertion that every visible row changed a model split).

prediction_before_update denotes the current U1 output before registering its row;
prediction_after_history_state denotes its U1 output using the visible history
snapshot, compared separately with prediction_frozen_U0. Registration does not
re-predict the current sample. State hashes cover all visible pseudo row fields.

This is consumed retrospective development. Official train metadata is read with
explicit usecols; scoring uses previously authorized archived errors only after
all trajectories finish. The manifest binds configs/protection.yaml and a new
append-only access ledger records the manifest before scoring reads.
No platform action is authorized by this evaluator. Only a full pass plus drift
review permits one test_a challenger, with cold verification and original ZIP
retained. Failure closes pseudo-history; v0.8 structural regression is next,
without inventing training/OOF specifications in this phase.
