# Four free-option candidates for the 2026-09-30 submission slots

Current user-reported platform best is **DE3_IRON_USER_REQUESTED = 96.3749**
(parent `V32_TIME_A60V7_50` 96.3727, +0.0022). Target **96.4**; gap **0.0251**.
The user stated four submissions remain before midnight. This round performs
**zero new fits** and delivers four composition packages to the desktop; the
user uploads them and returns scores.

## What is delivered

Desktop: `/mnt/c/Users/lqh22/Desktop/submission-96.4-slots-20260930`.

| # | package | iron column | time column | ZIP SHA-256 |
|---|---|---|---|---|
| 1 | `DE3_TIME_L0575` | shipped DE3 iron (byte-identical) | A60–V7m chord `lambda=0.575` | `5239cf54aaad52c8…` |
| 2 | `DE3_TIME_L0625` | shipped DE3 iron (byte-identical) | A60–V7m chord `lambda=0.625` | `03a1d2f0568ff628…` |
| 3 | `DE3_TIME_L0675` | shipped DE3 iron (byte-identical) | A60–V7m chord `lambda=0.675` | `ece8def7cecfa2bb…` |
| 4 | `DE3W100_IRON_TIME_A60V7_50` | `parent_iron + 1.0*(mean3-native)` | unchanged I1 (`0.2/0.3/0.5`) | `cde783badb7cb6e7…` |

The A60–V7m chord is written in the measured time simplex coordinates
`(V36, N, V7m)`: I1 is `(0.2,0.3,0.5)` at `lambda=0.5`, I5 is
`(0.3,0.45,0.25)` at `lambda=0.75`.

With the platform keeping the best score, a submission is a free option: the
four packages are the highest-expected-value zero-fit directions available
tonight, not a guarantee of 96.4.

## Why these four

1. **Concavity of the measured time chord.** The score is
   `100 - 50*(WMAPE_iron + WMAPE_time)`. Mixing two fixed prediction columns is
   convex inside the absolute loss, so the score is concave in the mixture
   weight. The two measured platform points I1 (`lambda=0.5`) and I5
   (`lambda=0.75`) are **both 96.3727** with the V12 iron. Therefore every
   interior point of that chord scores at least 96.3727, and under the
   previously confirmed additive model the shipped DE3 iron keeps its +0.0022.
   Packages 1–3 sample that chord.
2. **Iron ensemble weight.** The audited `authorized-training-oof.csv` from the
   DE3 development shows the iron component replacement is monotone in the
   ensemble weight: `w=1.0` beats the delivered `w=0.5` on **both** development
   split seeds (+0.004382 / +0.002579 versus +0.003076 / +0.002003). Package 4
   pushes the same audited recipe to full weight. The time arm is **not**
   ensembled: its `w=1.0` is negative on split seed 42.

The shipped-iron arithmetic was re-verified zero-fit: the delivered DE3 zip's
iron column equals `parent_iron + 0.5*(mean3 - native)` to `2.3e-13` over the
322 rows. Packages 1–3 preserve the DE3 iron **field text byte-for-byte**.

## Why 96.4 is still not guaranteed

The V46 certificate bounds the old fixed-endpoint time simplex (V36 / N0048 /
V7m) at **96.396835** with the incumbent iron. Adding the measured +0.0022 DE3
iron effect gives about **96.399** — still below 96.4. Therefore tonight's
candidates are expected around **96.375–96.380**, and reaching 96.4 needs a
genuinely new column direction rather than another mixture of the measured
endpoints. Offline work continues (PTaRL formal execution running under the
no-time-budget authority; DNNR formal development stopped on a cold-audit
tolerance failure).

## Verification

- Every ZIP contains exactly `result.csv`; 322 unique template-ordered IDs;
  finite non-negative values.
- Packages 1–3: iron field byte-identical to the DE3 package.
- Package 4: iron recomputed from the same formula at `w=1.0`; the `w=0.5`
  variant reproduces the delivered DE3 package.
- Desktop copies were read back independently by SHA-256 and structure.
- New fits, new full-data fits, new CV fits and agent uploads: **0**.

Private artifacts: `local/runs/round2-slots-20260930/packages-r2/`
(`build-summary.json`, `verification.json`, `desktop-delivery.json`) and the
inference/aggregation scripts under `local/tmp/slots-20260930/`.
Evidence: `EVIDENCE_STATUS.json -> round2_slots_20260930`.
