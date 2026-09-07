# optimization-v0.2 results summary

Review date: 2026-09-07. Scope: authorized local development labels strictly before
2024-11-01; no protected-label access. Detailed predictions, sample-level errors,
source identities and run artifacts remain local and Git-ignored.

## OPT-01/02 screening

E00 exactly reproduced the frozen run in both DEV_LONG and DEV_SHORT: raw maximum
absolute difference `0.0`, and final prediction CSVs were byte-identical.

| Candidate | DEV_LONG E | DEV_SHORT E | Screening observation |
| --- | ---: | ---: | --- |
| B0 | 0.176148 | 0.178157 | Global median control |
| B1 | 0.176101 | 0.178897 | Per-spout median control |
| E00 | 0.185646 | 0.171456 | Exact frozen baseline reproduction |
| E01 | 0.181472 | 0.171330 | No-history improves long; short essentially flat |
| E02 | 0.178950 | 0.163320 | Removing history age improves both folds |
| E03 | 0.186757 | 0.186624 | Removing historical target values harms both folds |
| E04 | 0.173937 | 0.173907 | Best long-fold result; beats both controls there |
| E05_OPERATION | 0.181621 | 0.170517 | Operation-only helps short, not enough on long |
| E05_BURDEN | 0.190771 | 0.178576 | Weak on both folds |
| E06 | 0.182079 | 0.172013 | Fixed 0.75 shrinkage helps long but hurts short slightly |

The DEV_LONG E00 gap versus B1 is concentrated in August and September and is much
larger on spout 2. Both targets contribute. History ages of 30–90 days show the
largest positive absolute-error deltas versus B1, while the greater-than-90-day
group does not continue that monotonic pattern. This supports testing the age-column
ablation and frozen-history adaptation, but does not prove a single causal mechanism.

E00's per-target shrinkage grid selects B1 (`w=0`) on DEV_LONG and the learned model
(`w=1`) on DEV_SHORT. No single E00-to-B1 weight is uniformly best across these two
folds.

These are screening results, not full-grid acceptance and not platform-score
promises. E02 and E04 are promoted for the origin×horizon grid; E06 remains as the
low-complexity risk-control comparison.

## Promoted origin×horizon grid

| Candidate | J | H1 | H2 | H3 | H4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0 | 0.176101 | 0.163350 | 0.174559 | 0.184268 | 0.182226 |
| B1 | 0.176254 | 0.163546 | 0.174677 | 0.184421 | 0.182374 |
| E00 | 0.183084 | 0.163007 | 0.181678 | 0.200155 | 0.187494 |
| E02 | 0.177753 | 0.159263 | 0.175318 | 0.192766 | 0.183667 |
| E04 | 0.174369 | 0.167339 | 0.173187 | 0.182692 | 0.174258 |
| E06 | 0.180227 | 0.162041 | 0.178942 | 0.194901 | 0.185025 |

No promoted candidate passes every pre-registered gate, so the v0.2 development G1
status remains failed:

- E02 improves J by 0.005330 and all four horizon means, but does not beat the
  better control on DEV_LONG.
- E04 improves J by 0.008715 and beats the controls on DEV_LONG, but regresses H1
  by 0.004332 and DEV_SHORT by 0.002451, both beyond their limits.
- E06 improves J by 0.002856 and all four horizon means, but does not beat the
  better control on DEV_LONG.

The next investment should therefore be frozen-history distribution adaptation,
using E02 and E04 as complementary comparisons. Historical target summaries remain
valuable (E03 is negative), while raw age columns and the current process/burden
combination are not robust. F1 process-change features should follow as isolated
increments; broad capacity or hyperparameter search should not precede those tests.

## Exploratory test_a results

Platform scores are supplied by the user and are not independently verified. The
platform exposes only the combined score, not per-target or per-sample errors.

| Candidate | Development H1 loss | Development H1 score | Platform score |
| --- | ---: | ---: | ---: |
| Frozen baseline | 0.163007 | 83.6993 | 81.4554 |
| E02 | 0.159263 | 84.0737 | 82.3290 |
| Iron 50/50; time E02 | 0.159273 | 84.0727 | 82.4431 |
| Uniform E02/E04 75/25 | 0.158526 | 84.1474 | 82.6221 |
| 75/25 plus time calibration | 0.158095 | 84.1905 | 82.7046 |

Across these five highly related candidates, Pearson correlation between the
development H1 score and platform score is `0.9958`; Spearman rank correlation is
`0.90`. This is high directional/ranking consistency. It is not high absolute-score
agreement: development H1 overstates the platform score by `1.49–2.24` points, with
a mean absolute gap of about `1.726` points.

These correlations are descriptive only. The sample is tiny, candidates share
models and predictions, weights were selected on development results, and the final
calibration uses early H1 OOF cells for fitting. The monotonic platform sequence
therefore does not establish independent generalization or justify additional
leaderboard parameter search.

The formally registered OPT-02 run remains G1 failed because none of E02, E04 or
E06 passes every gate. The later blends are promising derived experiments but have
not been registered and rerun through the authoritative acceptance report. OPT-03
frozen-history adaptation remains the next unimplemented phase.
