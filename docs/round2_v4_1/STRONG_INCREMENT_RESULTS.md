# Round2 V4.1 strong-base increment: C2/C3/C4 execution result

Date: 2026-09-25  
Branch: `round2-v4.1-strong-increment`  
Status: **EXECUTED_DEVELOPMENT_SCREEN_NO_PROMOTION**

## 0. What was executed

The new `c234` path was executed against the frozen V3.6 development replay on
the task-book first-round budget:

- targets: `tap_iron`, `tap_time_len`;
- variants: C2, C3, C4;
- seeds: 42, 3407;
- outer folds: 0, 1;
- 24 recipe units, of which 20 were available outer model evaluations;
- four `tap_time_len / C4` units are recorded as `blocked` because the exact
  parent uses `max_interaction_bins=64` and native EBM multidimensional
  partitioning with an added triple was not operable (a timed native fit did
  not finish in 120 seconds).  No coarse-bin substitution was made.

The `tap_iron / C4` path had a positive fixed-slot package delta in both seeds
on the first round, so it was taken to complete development coverage on folds
0-4.  C2/C3 controls were kept.  The complete iron run finished in 763.6 s
(30 outer recipe units) with real fit counts recorded below.

This is a frozen V3.6 development replay, not a new independent outer split.
No package was created and no upload was performed.

## 1. First-round fixed-slot screen (seeds 42/3407, folds 0/1)

Pooled `sum(abs)/sum(abs)` WMAPE per seed follows the project rule.  Package
delta is `100 - 50*(W_iron + W_time)` under the one-slot replacement formula.

| target | variant | seed42 delta | seed3407 delta | mean delta | both seeds positive |
|---|---|---:|---:|---:|---:|
| `tap_iron` | C2 | 0.000000 | 0.000000 | 0.000000 | no |
| `tap_iron` | C3 | 0.000000 | 0.000000 | 0.000000 | no |
| `tap_iron` | C4 | +0.000507315 | +0.000333510 | **+0.000420412** | **yes** |
| `tap_time_len` | C2 | 0.000000 | 0.000000 | 0.000000 | no |
| `tap_time_len` | C3 | +0.000513897 | -0.000402607 | +0.000055648 | no |
| `tap_time_len` | C4 | blocked | blocked | blocked | not evaluated |

Per-target pooled rows:

| seed | target | variant | rows | replacement WMAPE | baseline target WMAPE | candidate target WMAPE | package delta |
|---:|---|---|---:|---:|---:|---:|---:|
| 42 | `tap_iron` | C2 | 1102 | 0.037051 | 0.036498 | 0.036498 | 0.000000 |
| 42 | `tap_iron` | C3 | 1102 | 0.037051 | 0.036498 | 0.036498 | 0.000000 |
| 42 | `tap_iron` | C4 | 1102 | 0.037000 | 0.036498 | 0.036488 | +0.000507 |
| 3407 | `tap_iron` | C2 | 1102 | 0.039002 | 0.038664 | 0.038664 | 0.000000 |
| 3407 | `tap_iron` | C3 | 1102 | 0.039002 | 0.038664 | 0.038664 | 0.000000 |
| 3407 | `tap_iron` | C4 | 1102 | 0.038981 | 0.038664 | 0.038657 | +0.000334 |
| 42 | `tap_time_len` | C3 | 1102 | 0.039532 | 0.037760 | 0.037750 | +0.000514 |
| 3407 | `tap_time_len` | C3 | 1102 | 0.040094 | 0.038180 | 0.038188 | -0.000403 |

Real first-round fit counts:

- C2 outer fits: 8
- C3 outer fits: 8
- C4 outer fits: 2 (only where internal selection chose k>0)
- C4 inner parent fits: 12
- C4 inner candidate fits: 24
- blocked `tap_time_len / C4` units: 4

## 2. Full iron follow-up (seeds 42/3407, folds 0-4)

`tap_iron / C4` passed only the first-round directional rule.  Complete
five-fold development coverage produced:

| target | variant | seed42 delta | seed3407 delta | mean delta | positive folds |
|---|---|---:|---:|---:|---:|
| `tap_iron` | C2 | 0.000000 | 0.000000 | 0.000000 | 0/10 |
| `tap_iron` | C3 | 0.000000 | 0.000000 | 0.000000 | 0/10 |
| `tap_iron` | C4 | +0.00000435 | +0.00079691 | **+0.00040063** | **5/10** |

Pooled package scores:

| seed | B36 package | C4 candidate package | delta |
|---:|---:|---:|---:|
| 42 | 96.20130931837855 | 96.20131366831262 | +0.00000435 |
| 3407 | 96.20621581960641 | 96.20701273078774 | +0.00079691 |
| seed mean | 96.20376256899247 | 96.20416319951393 | +0.00040063 |

Per-fold C4 deltas:

| seed | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.000000 | +0.00100954 | -0.00124365 | -0.00224720 | +0.00245478 |
| 3407 | +0.00066586 | 0.000000 | +0.00115665 | -0.00011213 | +0.00226952 |

Full-run real fit counts:

- C2 outer fits: 10
- C3 outer fits: 10
- C4 outer fits: 8
- C4 inner parent fits: 30
- C4 inner candidate fits: 60

## 3. Decision

Candidate-pool gate:

- mean complete-development delta vs B36: `+0.00040063` (requires `>= +0.02`);
- both seeds positive: yes;
- positive development folds: `5/10` (requires at least `8/10`);
- workflow S_dev after C4: approx `96.204163` (submission working gate `96.25`);
- C4 is better than C3, but C3 has no standalone increment over C2 in this run.

Result: **no promotion, no preferred package, zero submissions, zero uploads**.
The original V3.6 package remains untouched.  `outer_seed 23003` was consumed
earlier and is not reused as an unseen split.

## 4. Artifacts

Private evidence, not committed to Git:

```text
local/runs/round2-v4.1-strong-increment/strong-increment-c234-r1/
local/runs/round2-v4.1-strong-increment/c234-iron-full-r1/   # interrupted shell run, optional
local/runs/round2-v4.1-strong-increment/c234-iron-full-r2/   # complete iron follow-up
```

The C4 blocked units are recorded explicitly in
`strong-increment-c234-r1/manifest.json` with candidate triples and the reason
`exact_parent_max_interaction_bins_64_triple_partition_not_operable`.
