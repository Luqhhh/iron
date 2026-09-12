# optimization-v0.14 / OPT-30–31

Registered from optimization-v0.13@2db6d5faae9c8568e01f5d75d145c17fc8404fb8 on local branch optimization-v0.14-h2-time. This is the fixed execution contract; results are recorded separately.

Only V7_H2_MATCHED_TIME is a candidate. D1_H1_SAME_CALENDAR is mandatory diagnostic control and cannot be released. Original E09/E04/rate models, feature schemas, source contracts, histories, cutoff, parameters, clipping and R2 0.8/(1-0.8) blend remain unchanged. V2–V6 remain closed. Iron copies the complete original V1 array exactly. Time is max(0, original_R2_time + beta * (original_R2_iron/original_rate-original_R2_time)); original rate validity and zero-direction fallback apply.

Seven genuine H2 forecasts use April–October month-start models to predict May–November, respectively. Matched H1 forecasts use May–November models on the same target-month IDs. Windows are Asia/Shanghai calendar months, left closed/right open. Forecast banks contain no targets. Model/schema/history/training/source digests and availability maxima certify each row. All original monthly H1 OOF and six outer V1/R2 arrays are replayed before fitting.

For June–November outer cutoffs, use completed May through prior-month windows and labels available at the outer cutoff, taken only from that cutoff's certified original history. Candidate and control share identical IDs, metadata and labels, at least 100 unique rows. The original single-scalar constrained LAD [0,1], smallest minimizer, is called once per role/origin, without additional time weights/intercept. One coefficient applies to all horizons of its origin. Intent files account for attempts before fitting; successful stored coefficients can be restored and verified without refitting. Failed attempts cannot be repeated.

Budget: CatBoost/other new base fits 0; V7 time LAD 6; D1 time LAD 6; iron LAD 0. Conditional final time LAD at most 1 only after historical acceptance AND official data/semantic/horizon compatibility. No final fit, bundle, ZIP or upload in the present pre-official-data execution. November and all evaluated labels are consumed retrospective development. Earlier outer labels may legally enter later rolling coefficients; no global untouched-label claim is made. Protected access uses configs/protection.yaml, frozen manifest and new append-only local/ledgers/optimization-v0.14-calibration.jsonl; old ledgers are immutable.

All V1/V7/D1 outer predictions and digests are saved before scoring-label access. Independent cold process reloads originals, replays both forecast banks and outer inputs, verifies stored LAD optimum/smallest-tie certificates, and checks exact new prediction serialization, reverse/chunk/subset/single inference and frozen iron. Every inference/cold/scoring fit attempt must be zero. Original prediction comparison tolerance is 1e-10; new scalar inference and unchanged iron are exact. E/J reconstruction tolerance is 1e-12. No tolerances will be widened after failures.

Acceptance is frozen in configs/optimization_v0_14/experiment.yaml: H2 mean delta E <= -0.0005; >=4/5 strictly improved; September/October/November target months >=2/3 improved; maximum single H2 delta E <= +0.0010; H2 V7 minus D1 <=0; original J delta <=+0.0002; H1/H3/H4 mean delta E each <=+0.0005; DEV_LONG/SHORT each <=+0.0007; all engineering/causal/identity checks and exact iron pass. Delta is candidate minus V1. Shared calendar-week paired bootstrap 1000, seed 2026, recomputed denominators, invalid draws counted without replacement; consumed-data stability description only. D1 scores and raw/base diagnostic outputs cannot become challengers.

Fail: FAIL_CLOSE_V7_RETAIN_V1, final fit/ZIP 0. Pass: DEV_ACCEPTED_PENDING_OFFICIAL_IDENTITY, not READY_CHALLENGER. Formal data identity/source semantics/final fitting/platform status remain separate. Official data changes block original-source publication, not historical research. No bias, per-spout/H3 coefficients, horizon routing, mixing, recency, trajectory or parameter search is authorized.

Implemented CLI (after synthetic tests pass, register clean source commit before execution):

```bash
uv sync --locked --extra dev --python 3.12
uv run --locked --python 3.12 pytest tests/test_horizon_calibration.py
uv run --locked --python 3.12 python scripts/optimization_v14_h2_calibration.py \
  --output local/runs/optimization-v0.14-opt30-r1
uv run --locked --python 3.12 pytest
uv run --locked --python 3.12 python scripts/check_no_private_artifacts.py
```

Runner invokes scripts/optimization_v14_cold_check.py --run automatically in an independent process before scoring. Run directory is exclusive and cannot be overwritten. Engineering fixes preserve failure evidence and restore already successful fits without adding attempts. Per-row banks/errors/labels/predictions/models/ledgers remain local. No public push, repository visibility change, history rewrite, desktop copy or active-release replacement is authorized by this registration.
