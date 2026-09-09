# optimization-v0.5 — OPT-14 first delivery

Base: `1f23f28817734c45af2e91b1dd1094108a122b25`, local branch `optimization-v0.5`.
This first delivery implements OPT-14 only. Old v0.4 code, configuration, outcomes,
and the R2 active release pointer remain unchanged. No push or platform upload is authorized.

The registration in configs/optimization_v0_5/experiment.yaml fixes P1/P2/P3,
conditional T1/T2 definitions, STAGE_A and GENERAL gates before new scoring.
T1/T2 are not executed in this delivery. They are eligible only if OPT-14 has no
STAGE_A winner. A winning OPT-14 candidate is eligible for a separate OPT-16 delivery;
it is not automatically trained, packaged, or activated here.

## Components and inference

- O0/H0: original E09/E04 at each monthly cutoff, loaded by v0.4 cache-reuse identities.
- OR/HR: R2 E09/E04, loaded by v0.4 fit-registry bundle hashes.
- A00 = .8 O0 + .2 H0; A11 = .8 OR + .2 HR.
- P1 = .8 OR + .2 H0; P2 = .8 O0 + .2 HR; P3 = OR.

Each component clips to nonnegative before blending; exported scoring values are not
rounded to six decimal places. Each model has its own schema and saved legal history.
The exporter uses metadata-only feature input and disables estimator fit and Context.fit.
A missing or mismatched component is an error, never an implicit retraining request.
The old feature-cache method is reused without constructing the old Context or calling fit.

Six origins produce 18 cells plus DEV_LONG/DEV_SHORT. Source error-file identities,
model hashes, schema, official-source hashes, legal training IDs, snapshot boundaries,
and original feature-code identities are checked. Every A00/A11 endpoint must match
its existing predictions within 1e-8 by sample ID. Prediction row reversal is also checked.

## Scoring and access

Official target columns are not reread. Scoring uses the declared old error CSVs;
loaded models supply their stored historical results. The new stage ledger records
both export and scoring. The old protected ledger remains unchanged, with global
holdout_consumed=true. November remains POST_HOLDOUT_CONSUMPTION.
Test_a metadata and platform results do not enter candidate selection.

STAGE_A uses R2 as reference: H1 improvement >=.001, >=4/6 improved origins,
J regression <=.0005, H2/H3/H4 regressions <=.0015, H1 target regressions <=.002,
DEV_LONG/DEV_SHORT regressions <=.001, and DEV_LONG beats the better simple control.
GENERAL is separately reported using the registered equal-horizon J rules.
Selection order is H1 loss, J, fewer components, then candidate ID.

D_O/D_H/interaction, conditional HR D_help, opposite-sign errors, cancellation amount,
per-target numerators/denominators/signed bias, month/spout breakdowns and weekly
paired intervals are retained. Weekly multiplicities are shared across repeated origins.
Intervals describe retrospective stability and do not adjust for candidate selection.

## Reproduction

Run from the repository root; output directories must be new:

```bash
uv run --locked --python 3.12 python -m bf_tap.optimization.component_export \
  --output local/runs/<new-export-id>
uv run --locked --python 3.12 python -m bf_tap.optimization.component_ablation \
  --export local/runs/<new-export-id> --output local/runs/<new-score-id>
```

The user-described standalone component_score.py was not attached as source; the
integrated scorer is component_ablation.py. Its real inference inputs come from
component_export.py, not reconstructed from an underdetermined blended prediction.
