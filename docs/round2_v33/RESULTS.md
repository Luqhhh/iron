# V33 continuous mixtures: complete development, no qualifying candidate

User objective: platform **96.5**. Current user-reported best B0 remains
**96.3679**, gap **0.1321**; no independently verified platform receipt is claimed.
Frozen implementation/specification commit: `c23a008`.

## G1: incremental model quality

All 40 development outer fits completed, with 80 optimizer runs (inner selection
and fresh outer refit), zero failures and complete five-fold coverage at seeds
42/3407. The reference is the actual B0 deployment recipe on matching training
partitions, not a weaker historical endpoint. Table gains are whole-package
score points from an isolated replacement of the target column. The other
column remains B0. No cross-seed OOF averaging is used.

| Target | Arm | Seed 42 gain | Seed 3407 gain | Mean gain | Positive folds (descriptive) |
|---|---|---:|---:|---:|---:|
| tap_iron | GAUSS1 | -0.013900 | -0.002307 | -0.008104 | 4/10 |
| tap_iron | MDN3 | +0.009044 | -0.000739 | +0.004153 | 5/10 |
| tap_time_len | GAUSS1 | +0.001803 | +0.004840 | +0.003321 | 7/10 |
| tap_time_len | MDN3 | -0.006211 | -0.001693 | -0.003952 | 6/10 |

**MDN3 iron** has a positive mean and beats GAUSS1 on average, but seed3407
is negative. It fails the frozen two-positive-development-seed prerequisite.
Its development package mean 96.251823 clears the 96.25 working gate; that
does not rescue the failed stability requirement. The two-seed LCB95 is
-0.026732, not evidence of a robust positive increment.

**MDN3 time** is negative on both complete seeds. **GAUSS1 time**, the control,
is positive on both seeds (mean +0.003321, local package mean 96.250992), but
its two-seed LCB95 is -0.006268 and it was explicitly ineligible for promotion.
It is retained as an observation, not substituted for the failed candidate.
No post-hoc weight scan, model-capacity change or additional mixture arm is run.

The candidate-tier classifier was applied as the predeclared descriptive
diagnostic. Its output cannot override MDN3-only eligibility or the later
four-seed gate. Both targets have **no confirmation finalist**. The conditional
confirmation command returned `no_development_finalist`; split seeds
271828/314159 remain unconsumed. No candidate is recommended for platform upload.
This result rejects the frozen three-component recipe, not all continuous
distribution models or all possible nonlinear information.

## G0: engineering and evidence

- Locked Python 3.12 full suite: **1194 passed, 23 warnings**.
- V33 targeted tests: **8 passed**. Synthetic preflight: both arms learned
  the synthetic relation; median MAE 0.92863 / 1.03429 against constant 8.44174.
- 20 existing B0 reference units reused after the original manifest/audit,
  original committed source, data, fold and fit/query identity checks. The
  copied prediction arrays are bit-identical to the original references.
- New development reference fits: **zero**. Both inner selectors and outer
  refits use training-only numeric statistics and categorical vocabulary.
- Fresh-process audit: **60 units, 80 saved models**, maximum cold/order/chunk
  prediction difference **4.55e-13**, score recomputation difference **1.85e-16**.
- Supplemental audit independently checked all 80 saved parameter payloads,
  numeric means/stds, categorical vocabularies, settings, selected epochs from
  calibration traces and fresh-refit epoch counts. All passed. No epoch cap hit.
- Manifest freezes 1470 source/evidence files. Private models, predictions,
  logs, fit ledger and all audit files stay outside Git.

## Budget and tomorrow's slots

Confirmation fits **0**, full-data fits **0**, new packages **0**, agent uploads
**0**. The five original desktop ZIPs retain their registered SHA-256 values.
Their README now distinguishes failed measured points from a global optimum,
restricts the concavity bound to its proper region, corrects unchanged-column
descriptions, and states the current 96.5 target.

The provisional plan remains: first interior 50%, then 75%; retain three slots
for separately qualified candidates. This V33 round supplies none. If no new
qualified candidate exists before the daily deadline, remaining old probes
are optional information purchases, with no promise of 96.5.

## Evidence hashes

- `manifest.json`: `b56b810e90008b10fea1c4b9e561a2229611eb45f9fd74af003aa1d611a24b96`
- `summary.json`: `0f7d79462d4cb8ee87e4b68eb478cf0e5811c9643e2a7075dd9310167a07a2d8`
- `audit.json`: `b15201bf2701c75bcc1503689814855af7e9e175843e3a7bfb2a434f44756c82`
- `supplemental-audit.json`: `e7f4c8b303a2dfbd5283adf178baa4fb73b4a22d05e2da7bc6fb5422d894e684`

Private run: `local/runs/round2-v33/development-r1`. Logs:
`local/reports/v33-python312-tests-r1.log`, `v33-development-r1.log`,
`v33-audit-r1.log`, `v33-confirmation-decision-r1.log`.
