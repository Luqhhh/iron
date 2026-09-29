# RFM_METRIC_LEARNING — proposed first execution design

2026-09-29. User requested proceeding by priority: RFM/xRFM, PTaRL, DNNR,
DANet. This first written design is awaiting review. No model fits or resource
probes have started. The previously reserved process-group strategy remains
on hold. No numbered V identity is allocated to avoid teammate collisions.

## Question, scope and sources

Test whether a learned full numeric distance matrix adds useful prediction
beyond a matched fixed-distance kernel. Start with a single global RFM;
no xRFM tree partitions, ensemble-tree count search or additional kernel family.
Both tap_iron and tap_time_len are evaluated independently. Only FULL_RFM is
eligible; FIXED_KRR is an ineligible mechanism control.

Sources: [author xRFM](https://github.com/dmbeaglehole/xRFM/tree/0cea9ba107c0a26dc1376a4c61c0d0bfa3e0ee1e)
and its ALGORITHM.md / recursive_feature_machine.py. Implement the explicitly
specified small CPU kernel below using existing numeric libraries; do not
install the upstream package or invoke its adaptive bandwidth, temperature,
tree/validation-refill defaults. This is an RFM mechanism adaptation, not exact
reproduction of the authors' complete xRFM recipe or benchmark performance.

## Reference and estimand

Current platform reference is user-reported V32_TIME_A60V7_50=96.3727;
objective >96.4. Use verified same-fold current and historical B0 caches.
The existing v46_cache verifier was run read-only in the current repository:
2754 rows and all seeds42/3407/7777/12011 passed on2026-09-29; new reference
fits=0. Reverify hashes in the execution workspace before fitting.

For each evaluated target, fixed candidate endpoint is0.8*current+0.2*new
member; the other target remains current. No alpha search or simultaneous
two-target package. Score improvements are complete-package points, evaluated
separately per target. Report standalone member WMAPE, incumbent-relative
increment, control contrast and historical B0 separately. A good standalone
score or low residual correlation is not a promotion rule.

## Exact model

Use the existing train-only NumericPreprocessor for the21 numeric features and
spout vocabulary; convert its numeric output to float64. One-hot encode the
training vocabulary and use the native unknown-category mapping. sample_id is
only an identity/split field. Standardize y using current training mean/std;
constant/nonfinite targets cause an engineering stop.

Let z be numeric inputs and c the categorical vector. The squared distance is
q=(z-z')^T M (z-z')+||c-c'||^2, with M initially I_21. The Laplace kernel is
exp(-sqrt(max(q,0))/h). h is the median positive distance of all distinct
training row pairs at M=I, computed only from the fitting partition. If no
positive distance exists, stop. Keep h fixed throughout that fitting path;
recompute from fresh outer-training data during refit. No bandwidth grid.

Solve (K+0.01*I)*alpha=y_standardized by float64 Cholesky. The literal diagonal
regularizer is0.01, not0.01*n. No automatic jitter escalation, approximate
solver, weight clipping or prediction clipping. Singular/nonfinite solve stops
and retains evidence. This single regularizer is a deliberately bounded first
probe; a failure does not prove all kernel regularization settings ineffective.

FIXED_KRR fits only M=I. FULL_RFM has states0/1/2/3, starting from the identical
fixed model; each transition uses numeric input gradients of the preceding
kernel predictor evaluated at training rows only. For positive distance r,
the per-center derivative is -K*M*(z-z')/(h*r); for r<=1e-12 set this gradient
term to zero. Compute G=mean_i(g_i g_i^T), symmetrize it and reject materially
negative eigenvalues. Roundoff eigenvalues in[-1e-10,0] may be set to zero;
lower values stop G0. Normalize G to trace21, then set
M_next=0.99*(21*G/trace(G))+0.01*I_21. A nonpositive/nonfinite trace stops.
Categorical coordinates keep their fixed Euclidean distance, no learned
numeric/categorical cross terms. No centered gradients or diagonal-only arm.

Gradient accumulation uses a fixed128-row query block and all training
centers; store only the accumulated21x21 matrix rather than an n*n*21 tensor.
Save centers, preprocessing, target scaling, M, h, alpha, selected update
count and source identities. These saved models contain private training
information and stay outside Git. Query order/chunking must not affect output.

## Selection and exact budgets

Use existing group-safe inner seed42/fold0. FULL_RFM solves all four states on
inner training and selects the lowest inner MAE, ties choosing fewer updates.
FIXED_KRR has one state. Then initialize a fresh model on all outer-training
rows and run the selected number of updates, with freshly fitted preprocessing
and bandwidth. Never reuse the inner fitted metric in outer refit. An outer
query label cannot enter preprocessing, bandwidth, AGOP or state selection.

Development: two arms *two targets *two split seeds42/3407 *five folds =40
outer fits,80 fit procedures (inner+outer), at most200 kernel solver starts
and120 AGOP updates. Arithmetic per target/seed/fold: FIXED has1 inner+1 outer
solve; FULL has4 inner+at most4 outer solves,3 inner+at most3 outer updates.
No gradient-descent optimizer starts occur in this direct-solve model.

Confirmation is conditional per target and includes its paired FIXED control.
Seeds7777/12011, five folds, maximum40 outer fits/80 procedures/200 solves/
120 updates when both targets qualify; half those counts if only one qualifies.
These are reused split seeds, not new independent observations. Maximum
formal total80 outer fits/160 procedures/400 solves/240 AGOP updates.
Failed solver/procedure starts consume budget. No retry, hidden extra fits,
manual second batch or adaptive parameter replacement.

## Engineering admission and resource limit

Before formal data fits: analytic-vs-finite-difference kernel gradients, matrix
PSD/trace and categorical isolation, identical FULL state0/FIXED predictions,
train-only transformations, split identity, budget reservations, saved cold
inference, query order/chunk invariance and independent score/decision tests.
Run the locked Python3.12 full suite; no new dependencies are proposed.

G0 synthetic budget is one complete worst-case inner/refit path per arm:
4 procedures, at most10 kernel solves and6 AGOP updates. Use2755 independent
standard-normal21-feature rows from NumPy default_rng56001, alternating
spouts1/2; y=10+2*sin(z0)+z1*z2+0.5*z3+0.1*epsilon, with independent noise
drawn after features. First2204 train/last551 query. Both synthetic selector
and fresh refit use this full training bound; FULL refit deliberately performs
all3 updates for timing regardless of selector choice. All official fit/query
partitions must fit the measured row bounds. This is a synthetic resource
probe, not the formal inner-validation split procedure.

Both saved models must produce finite held-out MAE below a training-median
constant, and cold/reversed/chunked maximum prediction difference<=1e-8.
No requirement that FULL beat FIXED on synthetic data, no synthetic tuning.
Include preprocessing, gradient updates, solver, serialization and cold audit
in timing. Pin OPENBLAS/OMP/MKL/NUMEXPR and torch threads to1. Workers<=4;
peak per worker<=1536MiB, available RAM>=4*maximum measured peak+1024MiB.
Projected development=20*(FULL path seconds+FIXED path seconds)/4*1.5+300s
must be<=2h. This is a stopping cap, not an ETA. Failure stops this design;
do not shrink data, remove controls, retry solvers or alter thresholds.

## Quality gate and sequence

Development admission per target requires two positive complete seed gains,
mean gain>=0.01 points, positive mean advantage over FIXED, mean local package
score>=96.25 and complete independent artifact audit. Only then consume its
confirmation budget. Final promotion requires all four seed gains positive,
one-sided95% seed t LCB=mean-2.3533634348018264*sample_sd/2>0, plus positive
four-seed mean FULL-vs-FIXED advantage. Disclose mechanism intervals and fold
signs as descriptive. No cross-seed OOF prediction averaging. Preserve current
candidate-tier classification and frozen historical decisions.

Finish or stop this first family before starting the next priority. PTaRL,
DNNR and DANet require their own concrete designs and budgets; this document
allocates no fits to those families and no permission to rerun a failed RFM
specification under a different name. Preserve all negative evidence.

## Execution boundary

Written design review precedes implementation planning under the active
brainstorming workflow. Implementation uses an isolated checkout, transitive
source/config/runtime/data/reference hashes, fresh private phase directories
and append-only reservations. One controller may advance only the frozen
eligible phase. Quiet30-minute checks notify failure/stop/completion/action.
Public validated changes are committed/pushed promptly; models, predictions,
ledgers and local evidence remain private. Full-data fits, ZIPs, desktop writes
and platform uploads=0. No estimate or claim of G1 benefit is made here.
