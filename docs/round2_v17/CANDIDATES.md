# V17 frozen candidate register and disposition

The candidate pool, target-specific B0 references, 0.5 replacement slot and
tie order were frozen in `configs/round2_v17/SPEC.yaml` and
`configs/round2_v17/TIER_SPEC.yaml`. All numbers below are complete five-fold
local package-score point gains against B0, computed from row-level OOF vectors.
They are not platform results.

| Unit | Target | Seed 42 | Seed 3407 | Mean candidate score | Tier | Confirmation |
|---|---|---:|---:|---:|---|---|
| J1 iron-only epoch selection | iron | +0.004455 | −0.002513 | 96.248710 | exploration | Ineligible: one negative seed |
| J2 asymmetric auxiliary projection | iron | +0.001613 | +0.003280 | 96.250185 | formal | Selected; four-seed gate failed |
| P-LL joint control | iron | +0.010228 | −0.003488 | 96.251109 | exploration | Ineligible: one negative seed |
| P-LH joint dual scale | iron | −0.003145 | +0.002459 | 96.247396 | not shortlisted | Ineligible: one negative seed |
| P-LL single-output control | time | +0.002433 | +0.000706 | 96.249309 | exploration | Selected; local 96.25 gate failed |
| P-LH single-output dual scale | time | −0.013308 | −0.007462 | 96.237354 | not shortlisted | Ineligible: two negative seeds |

The tier policy's single exploration shortlist is **P-LL joint iron**, but it
fails the separate two-positive-seed prerequisite and is not a release
recommendation. P-LL time was confirmed as the frozen target-specific control;
confirmation is an experiment, not an additional exploration release slot.
No V17 unit is promoted or packaged.

The platform best remains user-reported V12 `96.3526`. The conditional
V12-iron/V7-time arithmetic `96.3679` has never been scored as a combined ZIP.
