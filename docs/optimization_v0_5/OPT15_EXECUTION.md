# OPT-15 continuation

The user authorized continuation after completed OPT-14 rejected P1/P2/P3.
The original five candidate definitions and both acceptance gates remain unchanged.
Only T1 (R2 main plus eight short/long-history differences) and T2 (R2 auxiliary
without two age columns) are fitted. Each changes one complete dual-target component;
the other component is verified and reused from R2. Development budget: 24 target fits.

New official label access is restricted using metadata to eligible training rows through
the November 1 cutoff. November evaluation labels come from the declared existing error
CSVs. All reads are post-holdout-consumption development and enter the v0.5 ledger;
the old protected lifecycle and ledger are unchanged. No test_a input is used in selection.

Each dual-target fit reserves two calls in an append-only locked budget ledger before
training. Repeated slots and exhausted budget are rejected, including after a failure.
Saved components carry variant/component/cutoff/schema/history/source/input identities.
Training uses the original per-sample history boundary; inference uses frozen snapshots.

If a T candidate passes the registered STAGE_A gate, OPT-16 freezes the winner and
fits only its changed final component, reusing the other final R2 component. The mixed
bundle describes each component's actual variant separately. Label-free cold prediction,
CSV/ZIP readback and exact identities must pass before READY_CHALLENGER. It is not
activated, copied onto the desktop or uploaded automatically.

After selection, B/C checks only verify independent prediction and calendar horizon
mapping. They do not score, create B/C submission ZIPs, or select new algorithms.

Commands (new output directories only):

```bash
uv run --locked --python 3.12 python -m bf_tap.optimization.component_followup --output local/runs/<new-opt15-run>
uv run --locked --python 3.12 python -m bf_tap.optimization.component_release_v5 release --run local/runs/<opt15-run> --output local/runs/<new-challenger-run>
uv run --locked --python 3.12 python scripts/optimization_v5_stage_check.py --opt15-run local/runs/<opt15-run> --output local/runs/<new-stage-check>
```

The release command rejects a failed STAGE_A gate. For an accepted challenger, the stage
check additionally takes `--challenger local/runs/<new-challenger-run>`.
## Completed execution and stopping decision

Real run: `local/runs/optimization-v0.5-opt15-r1`. All 24 development target fits
completed (12 per candidate); six cutoff training sizes were 888, 1180, 1490,
1803, 2091 and 2424. Saved-model reload and row-reversal predictions matched.
The 18 grid cells and both DEV folds completed with zero inference fit attempts.

Lower loss is better; positive deltas are regressions against R2.

|Candidate|H1 E|H1 delta vs R2|18-cell J|J delta vs R2|H1 origins improved|STAGE_A|GENERAL|
|---|---:|---:|---:|---:|---:|---|---|
|T1|0.1576164883|-0.0000357980|0.1678080995|+0.0008176551|3/6|FAIL|FAIL|
|T2|0.1569251903|-0.0007270960|0.1693229550|+0.0023325105|4/6|FAIL|FAIL|

T1 fails the H1 improvement, improved-origin count, J and far-horizon limits.
H4 regresses 0.0034004156. T2 fails the H1 improvement, J, far-horizon and DEV
limits; H2/H3/H4 regress 0.0023317800/0.0036994628/0.0040258954.
Its DEV_LONG regresses 0.0078608811 and no longer beats the better simple control.
Both pass the H1 per-target regression cap, which does not override other failures.
Relative to E12-raw, T1 H1/J deltas are +0.0014295032/-0.0005610458;
T2 deltas are +0.0007382053/+0.0009538096.

Six-origin H1 changes against R2:

|Origin|T1|T2|
|---|---:|---:|
|June|-0.0029468096|-0.0018347275|
|July|-0.0011339210|-0.0028077287|
|August|+0.0010362643|+0.0019981933|
|September|-0.0007154280|+0.0016596841|
|October|+0.0022166659|-0.0015758950|
|November|+0.0013284406|-0.0018021020|

Calendar-week bootstrap retains 982/1000 valid draws for each candidate, sharing
weights across repeated samples/origins and recalculating denominators. H1 95%
percentile intervals are [-0.0008994672, +0.0007687683] for T1 and
[-0.0022191969, +0.0010919731] for T2. J intervals are
[-0.0002409602, +0.0018279774] and [+0.0016509989, +0.0030200352].
These are retrospective stability diagnostics, not independent confirmation or
selection-adjusted evidence.

Full numerators, denominators, signed bias, per-target/spout/month diagnostics,
recent three-month rows, old 14-cell summaries and separate gate checks remain in
`all_units.csv`, `h1_six_origins.csv`, `group_metrics.json`,
`recent_three_months.csv`, `legacy_14_summary.json`, `acceptance.json` and
`paired_week_intervals.json` under the run directory.

**V05_CLOSED_KEEP_R2**: all five registered candidates failed STAGE_A and GENERAL.
OPT-16 release was not triggered: zero final fits and no challenger ZIP. The release
implementation is covered by refusal/identity tests, but has no real challenger
validation claim. R2 remains active; E16 remains the archived fallback. No threshold,
weight or candidate exception was introduced. No Git push, desktop replacement or
platform upload was performed.

Locked Python 3.12 suite: **161 passed** in 52.84 seconds,
`local/reports/pytest-optimization-v0.5-opt15-r1.xml`. Original v0.4 source,
active-release config and protected ledger are preserved. The append-only v0.5
fit ledger reserves all 24 target calls; rerunning training cannot silently exceed it.

B/C label-free cold-process checks passed with the incumbent R2: test_b 322 rows
(all H2), test_c 548 rows (303 H3 and 245 H4). Report:
`local/runs/optimization-v0.5-stage-check-r1/stage_mapping.json`. No labels were
scored and no B/C ZIP was created; these checks make no platform-quality claim.
