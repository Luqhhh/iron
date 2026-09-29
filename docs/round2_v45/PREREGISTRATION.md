# V45: a matched forest partition-geometry experiment

The user resumed optimization with platform milestones **96.4, 96.45, 96.5**.
The current user-reported best is 96.3727; gaps are .0273/.0773/.1273. Reaching
an intermediate milestone does not complete the ultimate goal. Existing local
promotion gates and historical decisions remain unchanged.

## Distinct question and limits of the evidence

V4 compared forest leaf estimators, loss-based partitions and honesty. V3.3 F7
replaced two fixed feature groups with a reduced global PCA representation;
V39 learned hard axis selections. None tests a separate, full-rank random
block-PCA coordinate system per tree. This round asks whether that geometry
adds a useful increment to the incumbent. It does not reopen a leaf-estimator
scan or claim that existing forest failures have been reversed.

A label-free read of the 2754-row, 21-feature V2 training matrix finds 18 of
210 feature pairs with absolute correlation >.5, two >.8 (maximum .866083).
Correlation-matrix eigenvalues range approximately 3e-15 to 5.069627. This
motivates testing correlated coordinates while retaining every component;
it neither establishes target signal nor predicts improvement. No test features
or labels enter this diagnostic. The rank deficiency requires explicit tests.

Primary background: [Rotation Forest, 2006](https://www.lucykuncheva.co.uk/papers/jrlkcatpami06.pdf)
and [Rotation Forests for regression, 2013](https://doi.org/10.1016/j.amc.2013.03.139).
Those works motivate coordinate rotation, not these settings or a platform
forecast. Our paired bagged regression variant is not a paper reproduction.

## Frozen mechanism and predictor

Two arms: AXIS control and ROTATE candidate, both targets. Each forest has
256 squared-error CART trees, min leaf 5, min split 2, no depth cap/pruning,
all features available. The same n-row bootstrap indices and per-tree seeds
are used across arms. All tree predictions are averaged in fixed tree order.
No target transform, leaf-median substitution or prediction postprocessing.

Fit the existing train-only numeric mean/std and spout one-hot transformer.
For each tree, randomly permute the 21 numeric coordinates into seven groups
of three. For each group sample ceil(.75*n) rows with replacement from that
tree's bootstrap bag. Center these samples and compute full SVD; retain all
three right singular vectors without whitening or eigenvalue division.
Make the largest-absolute loading in each component positive (first index
breaks ties). Apply the resulting centered blocks to all rows; categorical
coordinates remain untouched. AXIS uses the identity basis; bootstrap and
tree-seed streams are independent of rotation work. No PCA sees query rows.
Training-scale constants below 1e-12 use scale 1, as in the frozen preprocessor.

Independent RNG streams per tree use SeedSequence([42, tree_index, stream]):
stream 0 for bootstrap, 1 for groups, 2 for group-PCA samples, 3 for the CART
seed. Save transforms, sample-index digests, tree arrays and fit-ID digests.
Prediction accumulates each projection in a fixed input-column order with
float64 scalar-column operations, then casts to CART's float32 input dtype.
Cold inference uses saved arrays and cannot reopen training data.

## Selection and budget

Full coverage from the start: split seeds 42/3407, five folds, two targets,
two arms: 40 outer units, each a calibration forest and a fresh outer forest,
80 forests / 20480 CART fits. Inner split seed 27001/fold 0 and blend grid
[0,.05,.1,.2,.35,.5,.75,1] remain fixed. No forest hyperparameter selection;
calibration chooses only blend weight, smallest on ties. Query targets never
enter fitting. Never average OOF inputs across split seeds.

Reference iron=.5*V36+.5*V12_joint; time=.2*V36+.3*N0048+.5*V7_periodic.
Reuse only the 20 hash-verified matching component-cache units from the
original V30 run, including original source/row/fold and endpoint arithmetic.
Confirmation requires new matching reference refits if a finalist exists.

Only ROTATE is eligible: both complete development seed gains >0, mean >=.01
score points, mean greater than AXIS. At most one per target may consume
confirmation seeds 271828/314159. All four seeds positive, paired seed LCB95
>0, and development package mean >=96.25 remain required. Fold-level counts
are descriptive, and candidate-tier labels cannot override the gates.

## Engineering admission, execution and stopping

Before official data fits: locked Python 3.12 tests, matched sampling, full-rank
orthogonality, train-only PCA provenance, constant/rank-deficient inputs,
identity-rotation equivalence, exported tree/sklearn equality, label guards,
and cold/order/chunk prediction tests. Do not upgrade dependencies.

Two full-shape synthetic forests (2204 train/550 query rows, 21 numeric inputs
plus spout) test a declared correlated oblique response. Both must beat a
training-median constant and ROTATE must beat AXIS. Do not change the synthetic
generator or forest after a failed admission. Twice the slower fit projected
over 80 forests/four workers must be <=7.61 hours, each worker <=1536 MiB,
and current available RAM must exceed four peaks plus 1024 MiB.

Synthetic generator is frozen at NumPy default_rng(45001): draw 2754 standard
normal latent values and then a 2754x21 independent standard normal matrix;
x=.8*latent[:,None]+.6*noise. Let z be each row's 21-feature mean and draw
response noise next: y=10+4*I(z>0)+3*z+.1*N(0,1). Finally draw spout values
uniformly from {1,2,3}; IDs are synthetic-v45-0..2753. First 2204 rows train,
remaining 550 query. This is a mechanism test, not resemblance to real labels.

Independent final audit checks all saved forests, train-only transforms and
sampling identities, bootstrap leaf counts/means, tree topology/support,
calibration weights, current-reference arithmetic, score and selection.
Cold/order/chunk tolerance is 1e-8. Runs are append-only, failures retained;
no auto-restart, hyperparameter rescue or tree-count extension.

Use hourly-only monitoring and actual timer events, not wall-clock estimates,
to trigger manual checks. A real completion/failure event also permits audit.
Publish validated public implementation and evidence; private artifacts stay
out of Git. Full-data fits, packages, desktop writes, agent uploads: zero.
