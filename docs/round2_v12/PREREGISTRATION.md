# V12: joint-output TabM, frozen before fitting

The continuing user goal is a platform score above 96.4. The current reported
best remains A35=96.3366; V7 is delivered with no platform feedback recorded.
V10/V11 failed their incremental confirmation. This experiment tests supervised
sharing between the two regression outputs in the stronger TabM architecture.
Earlier V2 joint trees and V4 low-dimensional shared functions are historical
controls, not this architecture. No claim of a new general method is made.

Two recipes are frozen: standardized raw-input TabM and periodic TabM at
frequency 0.01, with the V7 dimensions, optimizer and training budget. Each has
two output channels, in order iron/time. Each target is standardized separately
using only the current training partition. The loss averages squared residuals
over rows, ensemble heads and both targets; it never averages predictions before
computing training loss. Inner epoch selection uses the equally weighted mean
of the two standardized MAEs. A fresh network and fresh feature/target scalers
are refitted on the full outer-training partition for the selected epoch count.
Only features enter prediction; neither true target becomes an input feature.

Development is 2 recipes x 2 split seeds (42/3407) x 5 folds = 20 outer fits,
producing both columns per fit. No folds-0/1 screening. One additional single-
output time control must reproduce V7 seed42/fold0 exactly through the new
trainer before any candidate starts. Eight workers; BLAS/OMP/MKL/NUMEXPR and
torch threads pinned to one. Runtime, source, data/fold and reference identities
are frozen in the manifest. Failed evidence is retained, outputs never replaced.

CURRENT means the fixed V7 0.50 blend for time or the V9 0.10 candidate-pool blend
for iron. Report A35 and Q20 too. Each target is assessed as an isolated column
with the other column still A35. Joint training does not authorize joint release.
Use the existing candidate-tier policy and sparse weight grid. Never average
OOF vectors across split seeds. Rank eligible candidates by CURRENT-relative
incremental mean, then raw before periodic, then target name. At most one
target/recipe proceeds, only if both complete development seeds are positive
against A35 and CURRENT. No incomplete-pool selection or fallback finalist.

If eligible, allocate at most ten joint candidate fits at previously used seeds
7777/12011; evaluate only the selected target for promotion. These are reused
splits of the same dataset, not new independent labels. Promotion requires all
four seed gains positive and positive seed-level paired LCB95. Fold results are
descriptive. The 96.25 local gate and explicit release authorization remain
unchanged; V7's exception does not extend here. Full-data fits, packages, desktop
writes and uploads are all zero. External data/pretrained weights forbidden.

Tests cover both output gradients, singleton equivalence, per-partition target
scales and features, exact repeat fits, outer-label isolation, cold prediction
and row-order invariance. A separate zero-fit audit verifies all prediction
hashes, training metadata and independently recomputes weights and scores.
Private evidence: local/runs/round2-v12-joint-tabm/.
