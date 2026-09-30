# V48 results: longer training improves standalone error, not the blend

> Historical execution contract (reading update 2026-10-01): keep all measured failures, counts, frozen gates and source identities below. Subsequent user instructions resumed optimization, set milestones 96.4/96.45/96.5, changed monitoring to 600 seconds and removed future wall-clock budgets. Earlier pause/hourly/time-admission statements are not current instructions, and old failures are not relabeled as passes. See [current rules and status](../INDEX.md).

The frozen epoch-extension experiment is complete. Forty candidate outer units /
80 fits and 20 reused reference units passed independent audit. All 80 actual
old parameter/trace/checkpoint prefixes match V47 exactly; prefix prediction
difference is 0. No finalist, confirmation fit, full-data fit, package, desktop
write or upload. User-reported platform best remains 96.3727; milestones 96.4,
96.45 and 96.5 remain unmet.

## G1 quality and V47 comparison

Package score-point gains below use the unchanged current reference and an
inner-calibration-selected isolated-column blend. These local measurements
are not platform score forecasts.

| Target | Arm | Seed 42 | Seed 3407 | Mean | Mean change vs V47 |
|---|---|---:|---:|---:|---:|
| Iron | AFM2 control | -.002163824 | -.002152473 | -.002158148 | 0 |
| Iron | AHOFM4 candidate | -.001588775 | -.001099920 | -.001344348 | -.001119868 |
| Time | AFM2 control | 0 | 0 | 0 | 0 |
| Time | AHOFM4 candidate | 0 | -.005298953 | -.002649477 | -.002649477 |

AHOFM4 improves standalone WMAPE in all target/seed pairs:

| Target / seed | V47 (400 cap) | V48 (2000 cap) | Current reference |
|---|---:|---:|---:|
| Iron /42 | .05466582 | .05124198 | .03715193 |
| Iron /3407 | .05417682 | .05016826 | .03720431 |
| Time /42 | .06977119 | .05861087 | .03812708 |
| Time /3407 | .07058856 | .05922986 | .03816916 |

The standalone improvement does not produce an incremental blend gain.
AHOFM4 selects .05 in three iron and two time units; all other weights are
zero. Every nonzero blend weight loses on its outer fold. Neither arm has a
positive blended outer fold. AHOFM4 remains about 1.35-1.38x the incumbent
iron WMAPE and 1.54-1.55x its time WMAPE. AFM2 iron reproduces V47 entirely;
its time seed42 standalone error improves from .07984 to .07575 without
changing the selected zero blend weights.

The frozen confirmation command returned no_development_finalist and
confirmation_fits=0. Both derived split seeds remain unconsumed. Formal and
exploration shortlists are empty. No threshold or historical decision changed.

## The training cap is no longer the observed obstacle

All 40 selectors stopped before 2000 epochs through the unchanged patience
rule. AHOFM4 selected epochs 598-1370 for iron and 1084-1833 for time; latest
stopping epoch 1883. AFM2 selected 17-21 for iron and 22-1728 for time.
Thus this run has zero cap hits, resolving V47's measured truncation concern.
This does not establish a global optimization optimum or disprove all spline
models. It does remove the evidence-based reason for another epoch extension
of this recipe. Do not continue raising the cap, scan a duration grid, promote
the control or generate a package from standalone improvements.

## G0 execution, monitoring and audit

Locked Python 3.12 tests before execution: 1279 passed, 23 existing warnings.
Fixed synthetic admission passed with both actual 400-epoch prefixes exact.
Actual candidate event span 1319.7812 seconds (~22.00 minutes), peak worker RSS
430.75 MiB; conservative preflight projection was 2.0964 hours. No extra prefix
reproduction fits were used.

Scheduled observations at monotonic 83935.966767 and 84536.959900 seconds
(~600.993 seconds apart) showed 20/40 and 38/40 completed candidates, with
MainPID 195530 active/running. As-of-checkpoint fit ledgers had no failures.
Wall-clock times were 04:07:14 and 04:17:34 UTC; monitoring follows monotonic
time, not potentially adjusted wall-clock estimates. No intermediate health or
metric polling. A subscribed inotify terminal event records development exit 0
at 04:19:01 UTC and independent audit exit 0 at 04:19:13 UTC on 2026-09-29.
PID 0, active/exited, exit 0 then verified; the monitoring timer was stopped.
No live training or idle monitoring remains.

Independent audit covers all 60 units, 80 saved new models and 80 actual captured
V47 prefixes, training-only bases/normalization, fit/query row identities,
calibration epoch and blend selection, complete OOF reconstruction and gates.
Independent NumPy interpolation/ANOVA-DP, fresh-process, reversed and chunked
inference max difference 5.68434e-13 (<1e-8); pooled gain reproduction difference
1.81279e-16. Prefix parameters/traces exact; original predictions bit-identical.
Reference audit scope is original source/cache/data/fold/row identity and saved
endpoint arithmetic, without additional baseline fits. Engineering success does
not override model-quality failure.

Private evidence: local/runs/round2-v48/development-r1; v47-comparison.json
records exact comparison arithmetic without new fits.

| Artifact | SHA-256 |
|---|---|
| manifest.json | 94b7eef51c1100c302405997f7591933c3d5938eb97e869a99fe650d54fd2aa8 |
| summary.json | 7fc782208dc3b6dfc58ed639f6122b7ee294bc47c147881d8b554fe1ab86fc25 |
| audit.json | ac74101bc3b4e335cf7faad5da123e1a20a9a891a7bad43630773fe4bd4ba6fd |
| auditor source | 0004cfcfbfe86919cfaa0a7945b34b4f708142a0ec5fd2c743ccbc645bd7883e |
