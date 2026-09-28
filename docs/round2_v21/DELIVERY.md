# V21 delivery: three accurate-expert time mixes on the B0 parent

The user resumed optimization with the goal **platform 96.4**. V21 asks the next
question after V20: what is the best time column that can be built from experts
whose full-data predictions **already exist**? The answer is a large,
four-seed-validated local gain — with **zero new model fits**.

Pre-registration: [PREREGISTRATION.md](PREREGISTRATION.md),
`configs/round2_v21/SPEC.yaml`. Implementation:
`src/bf_tap_r2/v21_time_mix.py`. Private run:
`local/runs/round2-v21/packages-r1` (`verification.json`, `audit.json`).

## Packages

All three share the verified parent `V18_B0_V12IRON_V7TIME`
(`532118c9dd92467d16d5205073fee6b056ca7f60c0a59b053308e0ba9e073017`) and keep
`pred_tap_iron` **byte-identical**. Only the time column changes.

| id | time column | local mean gain vs B0 | paired LCB95 | dev mean score | ZIP SHA-256 |
|---|---|---:|---:|---:|---|
| `V21_TIME_LOCAL` | `0.40 V36 + 0.25 V7m + 0.35 P-LL` | **+0.012728** | **+0.009048** | 96.259932 | `32a74052899bd0b02d841d2e07e87645fdaf4b8030fcc69c7d7b12a654fc2016` |
| `V21_TIME_N10` | `0.30 V36 + 0.10 N + 0.20 V7m + 0.40 P-LL` | +0.008694 | +0.005749 | 96.255183 | `a93a972f301d09b684153a24798c4739001267c5751e7b5c22082338b552cccc` |
| `V21_TIME_A35` | `0.45 A35 + 0.15 V7m + 0.40 P-LL` | +0.005858 | +0.002481 | 96.251557 | `3d2bdcc19267ef1312e59c47a389abbd5d6b56d51c46d0727c1ca43b0ba40ca0` |

## Four-seed check (weights frozen, not refitted per seed)

| seed | `V21_TIME_LOCAL` | `V21_TIME_N10` | `V21_TIME_A35` |
|---:|---:|---:|---:|
| 42 | +0.010431 | +0.007523 | +0.004490 |
| 3407 | +0.013954 | +0.007363 | +0.003145 |
| 7777 | +0.009948 | +0.007442 | +0.006008 |
| 12011 | +0.016577 | +0.012446 | +0.009790 |

All three are **4/4 positive with a positive paired seed-level LCB95** and a
development mean package score above the frozen `96.25` working gate, with no
candidate-specific exception. `V21_TIME_LOCAL` is the strongest local evidence
the project has recorded — mean `+0.012728`, LCB `+0.009048` — above the V5
winner's four-seed record (`+0.00972` / `+0.00753`).

## Why the three exist: the N question

`B0` is `0.325 V36 + 0.175 N + 0.50 V7m` in the time column. The nested convex
optimum over the available endpoints **drops N entirely** and shifts toward
V36. The platform, however, rated `N` highly *in isolation on its own line*
(`V36 -> A35` worth `+0.0632`, `A35 -> A60` a further `+0.0099`). The three
packages span the N level:

* `V21_TIME_LOCAL` — N `0` (highest local gain, the gamble);
* `V21_TIME_N10` — N `0.10`;
* `V21_TIME_A35` — N `0.1575` (the conservative, near-`B0` level).

The platform keeps the best score, so submitting the set is a portfolio over the
one unresolved question rather than three independent models.

## G0 engineering

* Zero new model fits. The full-data endpoints are the V20 P-LL member
  (`local/runs/round2-v20/release-r1/full/cold.npy`, SHA-256
  `e5a1c36543b120fc9290371874a5b7467015292dec173ec89a0c76fc59ef30ee`,
  322 rows, minimum `75.993`) and the V36/N/V7m columns recovered
  field-exactly from the A35/A60/V7 ZIPs.
* Endpoint recovery re-derives `A45/A72/A85/A100` with a worst absolute
  difference of **1.14e-13** (declared `1e-9`).
* Independent post-write audit (`audit.json`): official template order; 322
  rows / 322 unique IDs; iron field strings **0 mismatches**; time blend
  recomputes to relative `0.0`; all predictions finite and non-negative
  (minima `76.363 / 76.426 / 76.493`); each archive holds exactly `result.csv`
  and matches the packaged payload.
* Targeted tests: 6 new in `tests/test_round2_v21_time_mix.py`.

## Budget

| item | value |
|---|---:|
| new fits | **0** |
| packages | 3 |
| desktop writes | 0 |
| agent uploads | 0 |

## Upload priority and limits

Priority: **`V21_TIME_LOCAL`**, then `V20_B0_PLLT_A325`, then `V21_TIME_N10`,
`V21_TIME_A35`, then `V18_B0_V12IRON_V7TIME`. If the platform confirms that N
can be dropped, `V21_TIME_LOCAL` is the strongest candidate the project has; if
the N-heavy hedge wins, the N question is answered the other way.

No score is forecast. Local magnitude does not forecast the platform (V5
delivered `4.19x` its local gain; V7 and V12 about `0.7-1.0x`). The conditional
`B0` arithmetic is `96.3679`, so even the strongest local gain is not guaranteed
to reach `96.4`, and the V18 concave two-line ceiling (`96.3992`) still applies
to families bounded by those endpoints. The user uploads; the assistant performs
no upload.
