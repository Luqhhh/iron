# optimization-v0.12 / OPT-25 and OPT-26 frozen plan

Authorized 2026-09-11 from optimization-v0.11@9a9437e08026c8d69bc973aa35155dc2557751b8; local branch optimization-v0.12. This is registration, not a completed training report. V2/V3/V4/V5 stay closed. V1 (user-reported 83.0319) and R2 fallback (83.0207) stay unchanged. November is consumed; no retrospective outcome is an untouched holdout or independent platform confirmation. V5's failed gates and interval crossing zero do not prove significant worsening.

## Publication and evidence boundary

Only authorized local work. No public push of this branch: inherited Git history contains official data. No visibility change, force-push, history deletion, platform upload or desktop overwrite. Preserve existing code, evidence and hashes privately. Inspect remote refs, Actions metadata, logs/artifacts where accessible; explicitly record unavailable content. A separate remediation plan must preserve old-to-new identity mappings and acknowledge that prior downloads/clones cannot simply be recalled. Participant repository authorization does not establish organizer authorization to publish data. See DATA_PUBLICATION_REVIEW.md. The official page has distinct last-submission and best-submission clauses; keep their stage wording and recheck on submission day, without resolving them by assumption.

## Hypothesis and immutable intervention

Only test the fixed hypothesis that recent eligible training rows may represent the next period better. A 60-day half-life is pre-registered, not optimized or asserted to be a physical period. For each original E09/R2 training row at cutoff c, age_days=(c-reference_time).total_seconds()/86400, raw_weight=2**(-age_days/60), weight=raw_weight/mean(raw_weight). Require reference_time<c and label available_at<=c. Age never uses tap_end_time, targets, spout or test distribution. Keep all eligible original rows; no floor, clipping, recent-row truncation, target weighting or per-spout normalization. Same identities imply identical iron/time weights. Report ESS=(sum w)^2/sum(w^2), ESS/n, monthly weight shares and weight quantiles globally and by spout as diagnostics, not independent sample counts or search criteria.

Only the two direct E09/R2 targets get new CatBoost models. Preserve exact original schema/order/dtypes/values (no V5 trajectory columns), last30/last100 histories, as-of availability, training rows/order and all FROZEN_PARAMETERS (MAE, 800, depth 5, .03, L2 5, seed 2026, all other values). The sole fit change is sample_weight=weight. Weights are not prediction features or weighted history statistics. Do not modify DualTargetBaseline. No new E04/rate/q/original V1/R2 fit. Weighted training MAE is a changed proxy loss; evaluation remains unweighted official WMAPE.

## Isolated target branches and registered candidates

Clip direct outputs as before, then use original .8/(1-.8) arithmetic:

Ibase_w=.8*I09w+(1-.8)*I04; Tbase_w=.8*T09w+(1-.8)*T04.

dI=old_rate*old_R2_time-Ibase_w; dT=old_R2_iron/old_rate-Tbase_w. Original rate>1e-6 is required; otherwise both directions zero. Preserve finite/nonnegative validation; do not add rescue thresholds. Iw=max(0,Ibase_w+beta_I*dI), Tw=max(0,Tbase_w+beta_T*dT). No cross-input from new time to iron or new iron to time, no iteration or recursive history. Original V1 paths remain intact.

- V6I_RECENCY60_IRON: Iw, exact original V1 time.
- V6T_RECENCY60_TIME: exact original V1 iron, Tw.
- V6B_RECENCY60_BOTH: Iw, Tw, exactly the corresponding single-target outputs.

These are three fixed target ablations from one model pair, not independent evidence. On identical samples/denominators, delta E(B)=delta E(I)+delta E(T), and likewise J, tolerance 1e-12. Raw predictions/base and any other combinations are diagnostic only.

Each outer target coefficient is the original constrained scalar LAD in [0,1], minimum 100 earlier causal OOF rows, original zero-direction and smallest-minimizer tie rules, no intercept. Do not add recency weights to LAD (its internal absolute-direction weights remain). Coefficients are fixed across every horizon of one origin.

## OPT-25 zero-fit preflight

Freeze this plan, configuration, sources, data and old evidence hashes, then append the new access ledger. Verify exact starting commit, clean initial tree, locked environment and prior repaired local G0 evidence. Inventory/recover April–November original E09/E04/rate, OOF, all six original V1/R2 origins and final releases; missing assets BLOCK, never retrain replacements. Search local registries for the same full training/data/weight/parameter identity; an existing same-definition failure is not a new experiment.

Reproduce old inference with fit forbidden. Build all eight original matrices/weights, check no trajectory contamination, unchanged features, ID alignment, finite positive normalized weights, timezone/time/label boundaries and training distribution diagnostics. Background V1 errors by target/month/spout may be summarized after registration, as explicitly authorized consumed-data context; they cannot change this design. No test distribution selection, fitted test-vs-train classifier or weights calculated across test reference times. Original test_a cold reread is identity checking only.

## OPT-26 budget and chronology

Exactly eight monthly cutoffs April–November, two single-target CatBoost models per cutoff: 16 development fits. Expected original row counts 297,588,888,1180,1490,1803,2091,2424 are identity checks, never targets to pad/truncate toward. March initializes training. April/May OOF initializes June coefficients; June–November gives 12 scalar LAD fits. Synthetic micro fits are separately recorded. Three output combinations/scoring/bootstrap require zero fits. Failures retain all evidence; saved successful models and coefficients are restored rather than refitted. No half-life/seed/depth/feature/postprocessing scans.

OOF rows record sample ID, reference and label availability, fold cutoff, training/history maxima, original endpoints, new direct predictions, new bases and isolated directions. Each training/coefficient label is available by its own cutoff. Freeze every outer candidate prediction, ID set and hash before candidate scoring-label access. No feedback changes afterward.

Evaluation origins={6:4,7:4,8:4,9:3,10:2,11:1}: 18 cells plus DEV_LONG/SHORT. E=.5*(iron WMAPE+time WMAPE). J first averages origins inside each horizon, then equally averages four horizons; repeated origin/sample appearances are not independent rows.

## Frozen acceptance and deterministic choice

Every candidate requires all engineering/causal/identity and unchanged-target/additivity checks; delta J<=-.0005; H1 mean delta E<=-.0005; at least four strictly improving H1 origins; at least two of September/October/November; each changed target H1 mean WMAPE delta<=0; each H2/H3/H4 mean E delta<=+.0005 and each changed-target WMAPE delta<=+.0010; each DEV E delta<=+.0007. Deltas always candidate minus V1. V6B independently faces these gates; neither target may average worse on H1, but its two single-target versions need not each meet the total-gain minimum.

If none passes, FAIL_NO_RELEASE, close fixed recency60 and retain V1. Otherwise, among full passes find minimum J. Candidates with J<=J_min+.0001 are near ties; prefer fewer changed targets, then lower H1 mean E, then lexicographic candidate ID. At most one challenger. No new route based on month/spout/platform feedback, no alternative 30/90/120-day fit or shrinkage/clipping/residual search.

Diagnostics: per target/month/spout WMAPE, signed bias, error numerator/target denominator, prediction-change quantiles and rate fallback counts. Shared calendar-week paired bootstrap 1000 draws seed 2026, same week multiplicity for repeats, recomputed cell denominators, record invalid draws without replacing them. Consumed-data stability only; no unadjusted significance or expected platform gain claim and no relaxed gate when intervals cross zero.

## Required checks and conditional final stage

Test 60/120/180-day ratios, mean normalization, hour-level age, reference/availability roles, missing/future timestamps, invalid/nonpositive/nonfinite weights, duplicate IDs and misaligned y/weight, shuffled ID reconstruction with original training order preserved. Assert old schema/values, no V5 columns, only specified models receive weights, old fit calls forbidden, exact unchanged targets, isolated branches and E/J additivity. Persistence, independent cold replay, reversed and subset inference must pass with original tolerances; inference CatBoost/LAD fits zero. A synthetic all-ones comparison may use small synthetic models, never another competition-data control fit. Normalize timezone object representations to Asia/Shanghai without changing real instants or loosening tolerances; retain failures.

Only a full-pass selected candidate permits final target fits: 1+1 CatBoost/LAD for V6I or V6T; 2+2 for V6B. Total maxima 18 CatBoost and 14 LAD. Reuse final original identities, 2754 rows and cutoff 2024-12-01 01:44:00+08:00. Final coefficients use April–November new OOF and labels available by final cutoff, never in-sample final predictions. Self-contained bundle contains selected new targets and necessary old models/history/schema/weight/provenance; cold inputs omit official label/history paths. Check 335 test_a IDs/order/subsets, finite nonnegative values, exact three columns, UTF-8 six decimals and ZIP containing only result.csv, read back and validate. READY_CHALLENGER only; no active pointer/desktop change or upload. Original V1 ZIP SHA256 fdcbe03e8ea31577bfeed0c45cd0a0013bda42ab88d29db703fad9c2f7e557aa remains immutable.

Deliver manifest, P0/weight audits, 16 model fit records/bundles, OOF provenance, 12 beta records, all three grids, selection/failure, cold validation, locked tests, old release hashes, package presence/absence reason and RESULTS. Update only current README/status/index, retain historical reports. If engineering fails, fix implementation only and reuse completed fits; any extra competition fit needs another registration/budget, not this ceiling.

## Sources

- User specification dated 2026-09-11; repository starting SHA above, v0.11 RESULTS and original model/structural/component sources.
- [Official task, data constraints and distinct submission clauses](https://www.aicomp.cn/tracks/tracks-6/4177.html), checked 2026-09-11.
- [CatBoost fit/sample_weight](https://catboost.ai/docs/en/concepts/python-reference_catboostregressor_fit): omitted weights default to equal row weights.
- [CatBoost has_time](https://catboost.ai/docs/en/references/training-parameters/common): preserves specified object order during relevant stages, not loss decay.
- [GitHub sensitive-data removal guidance](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository): remediation requires coordination across history/copies; no history rewrite is executed here.
