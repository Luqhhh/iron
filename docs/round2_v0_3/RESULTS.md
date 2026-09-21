# Round2 v0.3 results

Conclusion: retain M0 for both targets. Neither M1 nor K1 passes the frozen
target-wise promotion gates. End blind model expansion on this data version.
No new release, full-data estimator or platform upload was produced.

## Identity and engineering (G0)

Run: `local/runs/round2-v0.3/validation-r1`; experiment code `71707bd`.
All nine round2 input digests match the v0.2 manifest. Original frozen fold identity
matches; independent recomputation exactly reproduces the old M0 OOF predictions.
Dependencies and raw data were not changed. No old models or permutation audit were rerun.

M0 ZIP remains `5adc32eb9918c2e0235b193cbc00c20220b11f5263225589c02d7c7fea5f4b61`;
it was hash-checked, not regenerated. Existing M0 manifest has no platform submission
ID or score; no round2 platform receipt was found in the v0.2 run tree.

G0 PASS: locked Python 3.12 full suite 606 passed before CV; eight targeted non-fit
checks passed after adding convergence/clipping tests (two fitting tests excluded
to avoid spending the budget again). The initial convergence mock attempted to set
a read-only sklearn property; corrected to a fake estimator, without any real fit
or change to K1. Independent verification loaded all 20 K1 models and reproduced
OOF predictions, metrics and decisions, with zero additional fits.
K1 fit statuses all zero, no convergence warnings; iterations 1922–2403,
support vectors 2203–2204, validation predictions clipped: zero.

## Model quality (G1)

WMAPE and J below are percentages; lower is better. These are reused development
folds, not independent holdouts and not platform scores.

| Route | Seed | Iron WMAPE | Time WMAPE | J |
| --- | ---: | ---: | ---: | ---: |
| M0 | 42 | 15.571905 | 15.271460 | 15.421683 |
| M0 | 3407 | 15.572952 | 15.278193 | 15.425573 |
| M1 | 42 | 15.570869 | 15.271573 | 15.421221 |
| M1 | 3407 | 15.573016 | 15.279995 | 15.426506 |
| K1 | 42 | 15.977141 | 15.528014 | 15.752577 |
| K1 | 3407 | 16.031068 | 15.630978 | 15.831023 |

| Candidate / target | Seeds improved | Fold wins / 10 | Worst spout WMAPE delta (fraction) | Decision |
| --- | ---: | ---: | ---: | --- |
| M1 iron | 1 / 2 | 6 | 0.0000066063 | Reject |
| M1 time | 0 / 2 | 1 | 0.0000192855 | Reject |
| K1 iron | 0 / 2 | 0 | 0.0046487490 | Reject |
| K1 time | 0 / 2 | 0 | 0.0042565461 | Reject |

HD and ordinary median predictions differ little: fold differences range from
-0.542773 to -0.006858 for iron, and -0.051478 to +0.077217 for time.
The small seed-42 combined gain does not survive seed 3407 or target-wise gates.
K1 does not improve any validation fold for either target. These results do not
prove feature-label independence or a faulty official dataset.

All full OOF, per-fold ordinary/HD medians, training/validation errors, per-spout
metrics, model diagnostics and selection evidence remain under the private run.
No candidates are eligible, so the conditional 2000-resample sensitivity analysis
and release-stage inference tests are not triggered. No interval claim is made.

## Budget and handoff

Two synthetic engineering fits + 20 K1 CV fits = 22 / 24 regression fits.
M1: 20 fold statistical estimates; zero full estimates. Full regression fits: zero.
The unchanged M0 is still the only intended submission package. Its approximately
84.5783 / 84.5744 reference scores are local formula conversions, not platform scores.
No login, upload, desktop write, new data publication or history rewrite occurred.

Summary SHA-256: `41eeb861c0fd12e0e769b0dee7941ec7591bafcddc0994de79ef518333477c65`.
Independent verification: `local/runs/round2-v0.3/validation-r1/independent_verification.json`.
Next evidence should be an authorized M0 platform receipt or a new official data
version; neither licenses further parameter scans in this closed experiment.
