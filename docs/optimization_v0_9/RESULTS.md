# optimization-v0.9 results

OPT-21 and OPT-22 are complete. G0 development execution passes; G1 fails for both registered candidates. Close ratio-structure expansion without further ratio/parameter scans. No final inverse-rate fit and no challenger are authorized by the failed gates. Current release remains V1 (user-reported test_a 83.0319), with original R2 retained as fallback. Independent cold audit is recorded in the completion evidence linked below.

## OPT-21: zero-fit target ablation

W0 uses exactly R2 iron and V1 time. It retains 91.4381% of V1’s development J gain: W0 improves J by 0.001150207582 versus R2, compared with V1’s 0.001257907868. This supports the time-branch diagnosis, but W0 is diagnostic only and cannot become a platform candidate. All 18 cells, six H1 origins, DEV_LONG/SHORT, per-target/per-spout WMAPE, signed bias and week bootstrap were recomputed from archived sample predictions. Zero model or coefficient fits.

## OPT-22: frozen comparison against V1

All deltas below are candidate minus V1; negative is improvement. J averages origins within each horizon, then equally averages the four horizons.

| Metric | V2_DUAL_RATIO | V3_TIME_STRUCTURAL | Frozen requirement |
| --- | ---: | ---: | --- |
| J | 0.165571855579 | 0.165679555865 | V1 = 0.165732536588 |
| J delta | -0.0001606810 | -0.0000529807 | ≤ −0.0005 |
| H1 E delta | -0.0000478643 | +0.0000337015 | ≤ 0.0002 |
| H1 time WMAPE delta | -0.0000957285 | -0.0000957285 | ≤ −0.0005 |
| H2 E delta | -0.0001984830 | +0.0002446145 | ≤ 0.0005 |
| H2 time WMAPE delta | -0.0003969660 | -0.0003969660 | ≤ 0.0005 |
| H3 E delta | -0.0003616057 | -0.0003678085 | ≤ 0.0005 |
| H3 time WMAPE delta | -0.0007232114 | -0.0007232114 | ≤ 0.0005 |
| H4 E delta | -0.0000347711 | -0.0001224304 | ≤ 0.0005 |
| H4 time WMAPE delta | -0.0000695421 | -0.0000695421 | ≤ 0.0005 |
| H1 non-worse origins | 3/6 | 1/6 | ≥ 4/6 |
| Sep/Oct/Nov H1 improved | 2/3 | 0/3 | ≥ 2/3 |
| DEV_LONG E delta | -0.0001708830 | +0.0015288190 | ≤ 0.0007 |
| DEV_SHORT E delta | -0.0001996795 | +0.0001566338 | ≤ 0.0007 |
| Iron identity | Exact V1 | Exact R2 | PASS for both |

V2 fails J improvement, H1 time improvement and H1 origin count. V3 additionally fails recent H1 and DEV stability. Neither candidate is selected; the V3 simplicity tie-break is inapplicable.

| H1 origin | V2 E delta | V3 E delta |
| --- | ---: | ---: |
| O202406_H1 | +0.0000530521 | +0.0011367397 |
| O202407_H1 | +0.0000480890 | +0.0002101617 |
| O202408_H1 | -0.0001104038 | -0.0023819971 |
| O202409_H1 | -0.0001020219 | +0.0004183568 |
| O202410_H1 | -0.0004113278 | +0.0004147620 |
| O202411_H1 | +0.0002354267 | +0.0004041860 |

## Bootstrap and limits

Paired calendar-week bootstrap uses one shared week multiplicity vector across origins and recomputes each cell’s target denominators. Of 1,000 registered draws (seed 2026), 975 have positive denominators for all units. V2−V1 J has percentile 95% interval [−0.0002743042, −0.0000549477]; V3−V1 J has [−0.0008661313, +0.0007423902]. Both candidates share H1 time interval [−0.0003707879, +0.0001654370]. These are consumed retrospective data, not selection-adjusted or independent confirmation. A small V2 aggregate benefit does not override the frozen minimum effect or origin-stability requirements; no leaderboard gain is inferred.

## Training and engineering

Eight new inverse-rate CatBoost MAE fits (April–November), six development time LAD scalars, zero new R2/rate fits, zero final fit and zero challenger. All eight q histories contain zero iron=0 rows; exclusion accounting is still enforced. q uses positive iron weights normalized by the positive training mean and exact frozen baseline parameters/E09 R2 history schema. V2 copies frozen V1 iron without refitting alpha; V3 copies R2 iron. Time uses q times R2 predicted iron. Coefficients use certified earlier OOF and labels available before each outer cutoff.

All six outer endpoints exactly reproduce archived R2/rate/V1 forecasts, q serialization is exact, and reversing inference order leaves predictions unchanged. Inference fit attempts are zero. Saved histories provide eligible training labels; official test labels and test inputs are not used. Original release configurations, ZIPs, desktop V1 ZIP and prior protected ledger remain unchanged. The locked Python 3.12 suite passes 209 tests.

## Reproduction and local evidence

```bash
.venv/bin/python -m bf_tap.optimization.target_ablation --output local/runs/optimization-v0.9-opt21-NEW
.venv/bin/python -m bf_tap.optimization.dual_ratio_run --ablation local/runs/optimization-v0.9-opt21-r1 --output local/runs/optimization-v0.9-opt22-NEW
.venv/bin/python scripts/optimization_v9_cold_check.py --source local/runs/optimization-v0.9-opt22-r1 --output local/runs/optimization-v0.9-opt22-cold-NEW
```

The first two commands are historical reproduction instructions; do not rerun training after closure without a new authorization/budget. Existing run directories are immutable.

- Registration: `configs/optimization_v0_9/experiment.yaml`, `access_scope.yaml`, `docs/optimization_v0_9/PLAN.md`.
- OPT-21: `local/runs/optimization-v0.9-opt21-r1/` (diagnosis, metrics, spout metrics, shared-week bootstrap).
- OPT-22: `local/runs/optimization-v0.9-opt22-r1/` (manifest, 8 models/OOF folds, 6 coefficient audit sets, candidate predictions, acceptance, spout metrics, signed bias and bootstrap).
- Cold audit: `local/runs/optimization-v0.9-opt22-cold-r1/validation.json`.
- Final completion receipt: `local/reports/optimization-v0.9-completion-r1.json`.
- Tests: `local/reports/pytest-optimization-v0.9-r1.xml`.

Next modeling phase is optimization-v0.10 causal OOF residual stacking; its model, capacity, minimal process inputs, budget and gates require a separate registration. No residual model has been trained in v0.9.
