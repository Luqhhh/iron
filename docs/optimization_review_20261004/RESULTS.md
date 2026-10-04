# Completed SWA and TabM optimization review — 2026-10-04

Current platform representative remains EMA_TIME_Q75 at96.3920, user reported and not independently verified. Goals96.4→96.45→96.5. The table summarizes fixed timeA20 incremental score gains relative to the same Q75 reference. Two-development-seed means and four-seed means have different evidence depth and are not a single leaderboard. Four split seeds reuse the same official samples. Zero-fit candidates designed after viewing these seeds are post-selection exploration, not independent confirmation.

| Recipe | Evidence | Mean local increment | Interpretation |
|---|---|---:|---|
| SWA_WINDOW10 development | 2 development | +0.004443578049 | Both positive; original confirmation follows |
| SWA_WINDOW10 final | 4 | +0.002520555870 | 271828 negative; no formal promotion |
| SWA time-only selector | 2 development | +0.004245220784 | Paired mechanism negative; retain exploration |
| SWA prediction average | 4 seen, zero-fit | +0.002522413421 | Post-selection; nearly identical to parameter average |
| SWA cycle tail | 4 | +0.002917856711 | One seed slightly negative; platform96.3899 belowQ75 |
| SWA cycle time-weight | 2 development | +0.002349591806 | Weaker than cycle control; no confirmation |
| SWA time MAE | 2 development | +0.002903135387 | Weaker than original SWA; no confirmation |
| SWA MSE-to-MAE warmup | 2 development | +0.004423042484 | Both positive; very small negative mechanism; exploration |
| SWA last5epoch window | 4 seen, zero-fit | +0.002474578474 | Post-selection; one seed negative |
| SWA cycle mean2initializations | 2 development | +0.004249608893 | Positive gains but negative cycle mechanism |
| SWA updatewindow70 | 2 development | +0.004284018457 | User-retained exploration; no formal promotion |
| SWA time Huber | 4 | +0.003683563401 | Local formal; userreportedplatform96.3882 belowQ75 |
| JOINT_COV_MSE | 4 | +0.002991348251 | Allpositive,LCB+.002486665179; localformal, no platform feedback |
| Online headBootstrap | 4 | +0.004259479180 | Allpositive,LCB+.001941404281; localformal, no platform feedback |
| MIX50_A20 Bootstrap/Cov | 4 seen, zero-fit | +0.004102669673 | Post-selection; minimum+.003277209814/descriptiveLCB+.003081174811 |
| Bootstrap+EMA | 2 development | +0.004561824632 | Bothpositive; meanmechanism-.000080620067, retain exploration |
| Fixed per-fit Bootstrap | 2 development | +0.000751472955 | 42negative,mechanism-.003890971744; lowerpriority |

The original +0.004443578049 SWA development result remains valid. Its later confirmation weakened the four-seed result; the first two results were not erased. Huber is a concrete counterexample to local formal promotion implying platform improvement. The cycle and Huber userreported submissions scored96.3899 and96.3882, respectively, versus Q7596.3920. No other platform score is inferred; helpers never upload.

Prioritize online Bootstrap as the strongest completed four-seed time mean among these recent recipes, with Cov as a steadier complementary direction. MIX50_A20 has slightly lower mean than Bootstrap but better minimum and descriptive LCB on these seen splits; its correction correlations are imperfect, which motivates testing complementarity, not assuming it will transfer to training or platform. Keep BootstrapEMA/warmup/updatewindow70 as distinct exploration options despite small negative paired local differences. Fixed per-fit weight persistence is weaker in this exact recipe; this does not reject all bootstrap methods.

Next single authorized offline recipe is TABM_BOOTSTRAP_COV_V1: retain onlinePoisson1 sampling and originalrawTabM training, add the existing half-shrinkage covariance residual metric. Covariance uses actualfit rows only, separate selection/refit matrices, unchanged outputs. Compare against original onlineBootstrap at fixedA20 with complete two-seed development before conditional confirmation. No additional weight/shrinkage scan, no source/cache retry, no fullfit/package/upload/desktop release. Separate inference-time MIX50 exploration from this new training composition.

Sources are the immutable per-recipe RESULTS/STATUS documents and EVIDENCE_STATUS.json entries. Engineering G0/cold results and quality G1 decisions remain distinct. This summary does not overwrite old failure or promotion decisions.
