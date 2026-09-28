# V43: finite-chain Bayesian additive trees

Platform goal remains **96.5**, current user-reported best **96.3727**. V42's
negative quality and original cold-audit failure remain frozen. No old loss,
capacity, residual-corrector or in-library mixture scan is reopened.

Method motivation: [Chipman, George and McCulloch, BART (2010)](https://arxiv.org/abs/0806.3286).
BART regularizes a sum of trees with priors and fits it using Bayesian
backfitting. Our restricted grow/prune sampler, cutpoint grid and depth cap
are an explicit implementation variant, not a benchmark reproduction. Its
competition gain and finite-chain adequacy are unproven. No external data,
pretrained weights or environment upgrade is used.

## Frozen predictor

Two arms, each with 200 trees: **STUMP** (maximum depth one, additive control)
and **BART** (maximum depth four, only eligible candidate). Use one chain,
200 burn sweeps, then retain 100 draws every two sweeps: 400 sweeps total.
Both predict the **median of the equally weighted posterior-predictive Normal
mixture**, with each draw's sampled noise scale. This rule is fixed before
evaluation; no choice among posterior means/medians is made using labels.
No uncertainty interval or claim of posterior convergence will be released.
Record the complete scalar noise/RSS trace, acceptance counts and tree sizes.

Training targets are mapped by training midrange and full range to [-.5,.5].
Leaf prior mean is zero and sd `.5/(2*sqrt(200))`. Noise prior is scaled inverse
chi-square with df=3, placing probability .9 below the scaled training sample
standard deviation (ddof=1). Initialize all trees at a zero-valued root and
noise variance at that sample variance. Each sweep updates trees in fixed
order: remove its old prediction from the sum, propose a structure update
using collapsed Gaussian leaf likelihoods, draw every leaf from its exact
Normal conditional, then restore it. Draw common noise variance from its
inverse-Gamma conditional after the tree sweep. Fixed RNG seed 42; no retries.

Numeric inputs use training mean/std, spout a training-only one-hot vocabulary,
unknown spout all zero. Numeric cuts are training marginal deciles .1 through
.9; one-hot cuts .5. A split is admissible only if both children have at least
10 training rows and the parent is below the arm's maximum depth. The split
probability is `.95/(1+depth)^2` when any split is admissible, otherwise zero.
Given a split, the prior is uniform over all admissible (feature,cut) pairs.
This differs from uniform-feature-then-uniform-cut when feature cut counts
differ; the same declared prior is used in every acceptance ratio.

Propose grow or prune with equal probability over the move types currently
available. Grow selects a growable leaf uniformly, then one admissible split
uniformly; prune selects uniformly from nodes whose children are both leaves.
The Metropolis-Hastings ratio includes **both state-dependent proposal
probabilities**, all structure-prior factors and the integrated likelihood.
Both move types can reach the entire declared finite tree space via the root.
An LRU node cache capped at 4096 entries changes only repeated arithmetic,
never RNG calls, priors or proposals. Tree sums use fixed sequential addition;
mixture CDF reductions run within each row to avoid batch-dependent BLAS dots.

## Validation and budget

Keep the current-reference deployment recipe and matching cached components,
development seeds 42/3407, all five folds and inner calibration seed 27001/fold
0. The old calibration `epoch_selection` metadata is retained solely for
reference-cache identity; this sampler has no epoch/complexity selection.
Its fixed chain is fitted on the inner training subset, the unchanged sparse
grid selects its blend weight on calibration rows, and a fresh fixed chain is
fitted on all outer training rows. Both query targets are removed. No
cross-seed OOF averaging or outer-label access inside fitting.

40 outer units / 80 fixed-chain fits, four workers, one numerical thread per
worker. Reuse the 20 verified development reference units without new fits.
At most two full-shape synthetic preflight fits; project 80-fit development
using twice the slower synthetic time divided by four workers. Admit only
at <=7.61 hours, <=1536 MiB per worker, and sufficient current RAM. A failed
resource check does not authorize fewer trees/draws or a different chain.

G0 must validate collapsed likelihood against an independent dense Normal
density, leaf conditional algebra, grow/prune detailed balance (including
forced one-way proposal choices), mixture median against an independent root,
synthetic learning of interactions, train-only transforms, query-label
rejection, persistence and cold/order/chunk equality. Preserve all traces,
sampled trees and noise scales. Locked Python 3.12 tests and original
reference-cache/source/runtime/fold/row checks are mandatory.

BART can advance only if both complete development gains are positive, mean
gain >=.01, and mean exceeds STUMP. At most one finalist per target, then
confirmation seeds 271828/314159, four positive seed gains, positive paired
seed LCB95 and local development package score >=96.25. Folds are descriptive.
No gate changes, candidate-tier overrides or automatic release. Repeated
split seeds are stability evidence on reused labels, not independent data.

Hour-only monitoring, no polling or automatic restarts. Append-only private
runs, preserve failed evidence. Full-data fits, packages, desktop writes and
uploads: **zero**. If no candidate qualifies, close this frozen round; do not
extend the chain or change priors to rescue the observed result.
