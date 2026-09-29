# PROCESS_GROUP_MULTIPLICATIVE_INTERACTION: proposed execution design

2026-09-29. Status: written design awaiting review; no implementation, probe
or formal fit has started. The user requested starting the reserved strategy;
this document makes its architecture, scope and budget reviewable. Semantic
identity is retained instead of allocating a possibly conflicting V number.

## Scope and reference

First batch is time-only, to limit cost and test the representation on the
more score-sensitive target. Iron experiments are not an automatic extension.
Only SEMANTIC can advance; BASE, SHUFFLED and DENSE are mechanism controls.
Use current user-reported V32_TIME_A60V7_50=96.3727 (not an independently
verified receipt), objective >96.4. Fixed time-column replacement is
0.2*V36 + 0.3*N0048 + 0.5*candidate, compared with the incumbent's same
formula using V7 tabm_plr001. Iron remains the verified incumbent column.
Report historical B0 separately. No blend or target-scope search.

Use the existing four-seed native cache verification from v46_cache, including
original V7 audits, fit/query identity and restored original configuration
hashes. Verify all four seeds before official fitting. Missing/incompatible
cache stops G0; new reference factory fits=0. Do not confuse a released blend
with its raw V7 component. Cache verification is still pending for this design.

## Architecture and four arms

Retain the native V7 tabm_plr001 numeric preprocessing, periodic embedding,
spout encoding, two-block width256 TabM with16 members and dropout0.1.
BASE is the original network with no new parameters and must exactly replay
its native reference predictions and selected epochs before candidate fits.

For each augmented arm, keep the original raw/periodic input path. Build four
additional group embeddings h_g=tanh(B_g z_g+b_g), each width16. All six distinct
unordered group pairs produce h_g*h_j; concatenation gives96 coordinates.
A shared bias-free Linear(96,256) with all-zero initial weights adds its output
to every member's first affine preactivation, before the native activation and
dropout. Existing affine outputs and member scalings remain intact. Subsequent
layers follow the original network. This is a jointly trained input/hidden
representation; no incumbent output or residual label is an input.

The three augmented arms each add exactly24976 trainable parameters:
21*16 numeric projection weights +4*16 biases +96*256 injection weights.
Group biases start zero; projection weights are normal with standard deviation
1/sqrt(group size). Independent initialization stream55001 is isolated from
native initialization/dropout. No extra dropout or additional loss. Initial
augmented functions must exactly equal BASE. Native initialization, minibatch
order and dropout RNG are preserved; absence of exact initial equivalence is
an engineering failure. No width/rank/frequency or activation sweep.

Dictionary-reviewed SEMANTIC grouping, each numeric feature exactly once:

| Group | Features |
|---|---|
| Delivery (5) | air_volume, oxygen, hot_air_temp, humidity, air_speed |
| Pressure (7) | cold_air_press, hot_air_press, furnace_top_press, upper_press_diff, lower_press_diff, total_press_diff, air_press_ratio |
| Furnace state (3) | furnace_top_temp_avg, furnace_throat_temp, gas_rate |
| Material (6) | coal_rate, pig, all_quality, consumption, fuel_rate, coke_rate |

The official dictionary identifies gas_rate as gas utilization percentage.
The grouping is a learning hypothesis for synthetic data, not a conservation
law or proof of causality. No invented units or raw dimensional products.

SHUFFLED uses a fixed permutation of FEATURES from NumPy default_rng(55002),
partitioned into sizes5/7/3/6. DENSE uses all-feature mixed coordinates zQ,
where Q is obtained by QR of a21x21 normal matrix from default_rng(55003),
with column signs chosen to make diag(R) positive; partition by the same sizes.
Both use the same embeddings/injection and identical parameter counts as
SEMANTIC. Q is fixed, label-independent and serialized with its hash; it is
not trainable. These controls distinguish semantic grouping from arbitrary
grouping and generic dense low-rank interactions. BASE has lower capacity;
SEMANTIC-vs-BASE alone does not establish a semantic grouping mechanism.

## Training and budget

Native float32, standardized single-output MSE, AdamW lr0.001/decay0.0001,
batch256, maximum240 epochs, patience25, min_delta1e-5, training seed42.
Group-safe inner seed42/fold0; select by standardized MAE of member-mean
prediction, then fresh outer-training refit at the selected epoch. Inner and
outer preprocessing are separately fitted on their own training rows.
No external data/weights, sample_id input, temporal ordering or target hints.

Development: four arms, seeds42/3407, all five folds =40 outer fits,
80 optimizer starts. Run all10 BASE fits before the30 augmented fits.
Only complete successful development and independent audit admit confirmation.
Confirmation: all four arms, seeds7777/12011, all five folds =maximum40
outer fits/80 optimizer starts; run the10 BASE replays first. These are already
used split seeds, not untouched labels or independent new datasets.
Maximum official total80 outer fits/160 optimizer starts, including20 BASE
fits. Failed starts consume budget; no retries, hidden diagnostics or rescue arms.

G0 permits exactly8 synthetic optimizer starts: one full-shape inner selector
and fresh maximum240-epoch outer refit for each of four arms. Synthetic data
contains2754 independent standard-normal rows,21 numeric inputs in FEATURES
order, alternating spouts1/2, NumPy default_rng seed55004. Let
y=5+sin(z0)+0.5*z1*z6+0.25*z15*z18+0.1*epsilon, with independent standard-normal
epsilon drawn after the feature matrix. First2204 rows form outer training;
its first1763 form inner training and remaining441 inner validation; last550
rows form outer query. Verify these bounds dominate actual fitting partitions.
For conservative timing only, disable synthetic early stopping and execute
all240 selector epochs, retaining its best validation checkpoint; fresh refit
also executes240 epochs. This does not change formal early stopping. Use the
identical synthetic data for every arm. Check finite gradients, actual
parameter updates, saved inference and better-than-constant held-out MAE for
all arms; do not rank arms or change design based on synthetic scores.

Measure both full selector and refit paths, with validation, preprocessing,
saving and cold loading included. Workers<=4, all numerical/torch thread caps1;
peak worker<=1536MiB and available memory>=4*measured maximum peak+1024MiB.
Projected development=(10*sum(four measured selector+refit times)/4)*1.5
+300s must be<=4h. This is an admission cap, not a measured ETA. Failure stops
this specification without shortening epochs or removing controls.

## Quality decision and audit

Development requires SEMANTIC incumbent-relative gain positive on both complete
seeds, mean>=0.01 score points, mean local package score>=96.25, and positive
mean package advantage over both SHUFFLED and DENSE. Only SEMANTIC can advance.
Confirmation requires all four SEMANTIC seed gains positive, one-sided95%
paired seed-level t LCB=mean-2.3533634348018264*sample_sd/sqrt(4)>0, and positive four-seed mean advantage over each
mechanism control. Disclose mechanism intervals; fold signs are descriptive.
Candidate-tier classification remains descriptive beside these frozen gates.
No averaging OOF vectors across seeds, gate relaxation or platform forecast.

Freeze transitive source/config/runtime/data/reference/fold hashes before G0
and official phases. Independently audit saved models and cold order/chunk
invariance at absolute tolerance5e-4, prediction identity, epoch/refit logs,
all budgets, fixed endpoint arithmetic and decisions. Strict BASE replay
requires exact native predictions/epochs, not the cold chunk tolerance.
Report G0 engineering and G1 model quality separately; retain all failed evidence.

## Execution and publication boundary

Implementation requires the written design review, then a reviewed execution
plan; this document does not claim those stages complete. No new dependencies
are proposed. Run focused mechanism/audit tests and the locked Python3.12 full
test path before official training. An isolated managed worktree is required
for implementation, preserving other branch workflows and frozen source.

After admission, use a one-shot budgeted controller and quiet30-minute checks;
notify only failure, stopping, completion or required user action. New private
runs use local/runs/process-group-multiplicative-interaction with fresh phase
directories and append-only reservation ledgers. Never overwrite evidence.
Full-data fits, ZIPs, desktop writes and uploads=0. Public validated code and
results are committed/pushed normally; private artifacts remain outside Git.
