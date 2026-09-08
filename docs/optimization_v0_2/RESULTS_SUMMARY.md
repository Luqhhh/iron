# optimization-v0.2 results summary

Review date: 2026-09-08. Scope: authorized local development labels strictly before
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
frozen-history adaptation was therefore executed next.

## OPT-03 frozen-history adaptation

The two candidates were registered before the run. Each original training sample
was expanded into history views at ages `{0, 7, 30, 60, 90}` days. All five views
remain in one outer/inner partition and receive weight `0.2`, so the total weight
per original sample is one. Unavailable history rows are masked in full before
target summaries and counts are built.

| Candidate | J | H1 | H2 | H3 | H4 | DEV_LONG | DEV_SHORT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E00 | 0.183084 | 0.163007 | 0.181678 | 0.200155 | 0.187494 | 0.185646 | 0.171456 |
| E01 | 0.177559 | 0.161222 | 0.180352 | 0.191192 | 0.177470 | 0.181472 | 0.171330 |
| E02 | 0.177753 | 0.159263 | 0.175318 | 0.192766 | 0.183667 | 0.178950 | 0.163320 |
| E04 | 0.174369 | 0.167339 | 0.173187 | 0.182692 | 0.174258 | 0.173937 | 0.173907 |
| E07_FROZEN_E02 | 0.178193 | 0.162109 | 0.180739 | 0.190843 | 0.179081 | 0.181769 | 0.171019 |
| E08_FROZEN_E04 | 0.182899 | 0.163815 | 0.176719 | 0.186785 | 0.204277 | 0.185517 | 0.178898 |

G0 passed, E00 reproduced exactly, and protected labels were not read. G1 failed:
E07 improved all four horizon means and J by `0.004891` versus E00, but its
DEV_LONG loss did not beat the better median control (`0.181769` versus
`0.176101`). E08 failed five of six gates. E07 is the stronger OPT-03 candidate,
but it is not an accepted model-quality improvement.

At the user's explicit request, E07 was frozen as an exploratory test_a platform
probe despite the failed G1 gate. The release manifest labels this exception and
the user subsequently reported a platform score of `82.3610`. This is `0.9056`
above the frozen baseline and `0.0320` above E02, but `0.3436` below the incumbent
time-calibrated 75/25 blend. The result is consistent with the failed G1 decision:
OPT-03 does not replace the incumbent on current evidence. This does not reopen
adaptive platform tuning.

## OPT-04 signed process-change features

F1 adds three signed changes for each frozen hourly operation variable:
`latest−mean6h`, `latest−mean24h`, and `mean6h−mean24h`. These use the existing
as-of aggregates and introduce no new window or source.

Screening paired each increment with its unchanged route. E09 improved DEV_LONG
by `0.002929` versus E02 while regressing DEV_SHORT by only `0.001392`. E10
improved both folds versus E01. E11 improved DEV_SHORT but regressed DEV_LONG by
`0.004365` versus E04 and was not promoted.

| Candidate | J | H1 | H2 | H3 | H4 | DEV_LONG | DEV_SHORT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E00 | 0.183084 | 0.163007 | 0.181678 | 0.200155 | 0.187494 | 0.185646 | 0.171456 |
| E01 | 0.177559 | 0.161222 | 0.180352 | 0.191192 | 0.177470 | 0.181472 | 0.171330 |
| E02 | 0.177753 | 0.159263 | 0.175318 | 0.192766 | 0.183667 | 0.178950 | 0.163320 |
| E09_PROCESS_CHANGE_E02 | 0.173861 | 0.157199 | 0.172630 | 0.185171 | 0.180445 | 0.176021 | 0.164712 |
| E10_PROCESS_CHANGE_E01 | 0.175213 | 0.160900 | 0.175418 | 0.185402 | 0.179131 | 0.177931 | 0.170312 |

E09 passed all six frozen acceptance checks. Relative to E00, J improved by
`0.009222`, all four horizon means improved, and both target mean WMAPEs improved.
DEV_LONG narrowly but validly beat the better median control (`0.176021` versus
`0.176101`). E10 passed five checks but did not beat that control. OPT-04 therefore
has G0 and G1 PASS with E09 as the accepted candidate. Its test_a submission is
frozen; the user reported `82.8174`. This improves the previous platform incumbent
by `0.1128` and the frozen baseline by `1.3620`. E09 is now the platform incumbent,
with the platform result retained as external, non-independent evidence.

## OPT-05 registered derived candidates

OPT-05 evaluated the three fixed derivations from D010 in one full run with all
required controls and components. No weight, residual or candidate definition was
changed after reading the results.

| Candidate | J | H1 | H2 | H3 | H4 | DEV_LONG | DEV_SHORT delta vs E00 | G1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| E09_PROCESS_CHANGE_E02 | 0.173861 | 0.157199 | 0.172630 | 0.185171 | 0.180445 | 0.176021 | -0.006745 | PASS |
| E12_BLEND_E09_E04_80_20 | 0.171713 | 0.156806 | 0.170448 | 0.182095 | 0.177505 | 0.173574 | -0.007728 | PASS |
| E13_BLEND_E09_E10_50_50 | 0.173714 | 0.157973 | 0.173177 | 0.184684 | 0.179022 | 0.176371 | -0.005590 | FAIL |
| E14_TIMECAL_E09 | 0.171946 | 0.156641 | 0.171015 | 0.182513 | 0.177615 | 0.174315 | -0.007644 | PASS |

E12 achieved the lowest J and improved all four horizon means and both target mean
WMAPEs versus E00. Its J improvement was `0.011370`; its DEV_LONG result also beat
the better median control (`0.173574` versus `0.176101`). E14 passed all six gates
and was slightly better on H1, but E12 was selected by the pre-registered aggregate
objective. E13 failed only the DEV_LONG control comparison.

G0 is PASS: E00 reproduced the immutable reference with raw maximum absolute
difference `0.0` and byte-identical validation CSVs; protected labels were not read;
the locked Python 3.12 path passed 68 tests. G1 is independently PASS for E12. The
test_a package contains 335 validated rows. The user subsequently reported a
platform score of `82.9543`, improving E09 by `0.1369`, the previous derived
incumbent by `0.2497`, and the frozen baseline by `1.4989`. E12 is therefore the
platform incumbent, while the score remains external, non-independent evidence.

## OPT-06 target-level composition

OPT-06 ran from the clean pre-registration commit `a343e96`. E15 combines E12 iron
with E14 time; E16 applies the same frozen time residual after the E09/E04 blend.

| Candidate | J | H1 | H2 | H3 | H4 | DEV_LONG | DEV_SHORT delta vs E00 | G1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| E12_BLEND_E09_E04_80_20 | 0.171713 | 0.156806 | 0.170448 | 0.182095 | 0.177505 | 0.173574 | -0.007728 | PASS |
| E15_TARGETWISE_E12_E14 | 0.171057 | 0.156197 | 0.169569 | 0.181391 | 0.177069 | 0.173168 | -0.008517 | PASS |
| E16_TIMECAL_E12 | 0.169902 | 0.156308 | 0.168841 | 0.179582 | 0.174875 | 0.172042 | -0.008845 | PASS |

Both candidates passed all six gates. E16 achieved the lower aggregate J and the
best H2–H4, while E15 was lower only on H1. E16 improved J by `0.013182` versus E00,
improved all four horizons, and improved both target mean WMAPEs. It is selected by
the pre-registered equal-horizon objective.

G0 is PASS: E00 again reproduced with raw maximum absolute difference `0.0` and
byte-identical validation CSVs. The run and E16 release used a clean pre-registration
commit, protected labels were not read, and 69 locked Python 3.12 tests passed. G1
is separately PASS. The 335-row test_a package is frozen pending platform evidence.
