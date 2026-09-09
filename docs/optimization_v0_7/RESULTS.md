# optimization-v0.7 / OPT-19 results

**G0 PASS; G1 FAIL. Close pseudo-history. Keep R2 (user-reported 83.0207).**
All seven groups of the stricter frozen gates failed. No U1 challenger, new ZIP,
platform upload or new model fit was made. Old routing/blending searches remain
closed. The next model direction is v0.8 cross-target structural regression.

## Complete causal replay

Six saved R2 origin pairs were loaded with their original model/history/source
identities. Each trajectory starts at its cutoff and runs all intervening samples:
June/July/August four months each, September three, October two, November one.
These are 5,651 origin-sample predictions, with a separate complete reversed-input
rerun for every origin. DEV_LONG and DEV_SHORT use the same corresponding origin
paths. The 18 grid cells and two DEV windows are scored only after all rollouts
finish. No scoring target is passed to a feature builder or pseudo-row constructor.
These are consumed retrospective development results, not independent confirmation.

Delta below means U1 minus frozen U0/R2; positive is worse.

| Check | U1−U0 | Required | Result |
|---|---:|---|---|
| H1 mean E | +0.0023188674 | <= −0.0010 | FAIL |
| H1 improved origins | 3/6 | >= 5/6 | FAIL |
| Recent H1 | Sep/Oct worse; Nov better | all three improve | FAIL |
| H1 iron WMAPE | +0.0028691698 | <= +0.0010 | FAIL |
| H1 time WMAPE | +0.0017685650 | <= +0.0010 | FAIL |
| H2 mean E | +0.0064355508 | <= +0.0015 | FAIL |
| H3 mean E | +0.0054115503 | <= +0.0015 | FAIL |
| H4 mean E | +0.0048839496 | <= +0.0015 | FAIL |
| extended J | +0.0047624795 | <= 0 | FAIL |
| DEV_LONG E | +0.0035905067 | <= +0.0015 | FAIL |
| DEV_SHORT E | +0.0101478097 | <= +0.0015 | FAIL |

J rises from **0.16699044445671873** to **0.17175292397572037**.

| H1 origin | E delta |
|---|---:|
| June | −0.0018267647 |
| July | −0.0025628490 |
| August | +0.0061775681 |
| September | +0.0075122141 |
| October | +0.0064933601 |
| November | −0.0018803241 |

## Recursive state inspection

All outputs are finite and nonnegative. Monthly pseudo target median/p5/p95,
per-sample frozen-to-recursive changes and pending completion counts are retained.
There is no observed unbounded numerical explosion, but recursive inference makes
substantial systematic shifts and loses accuracy across every average horizon.
Across origins, mean iron shifts versus U0 are +8.38, +9.48, +28.06, +40.19,
+22.97 and +26.76 in chronological origin order. Maximum absolute sample changes
are 82.4864 iron and 13.1155 minutes. These are prediction shifts, not estimates
of signed error against truth or proof of the precise failure mechanism.

Maximum visible pseudo rows: 1,261. Maximum conservative ancestry depth: 904
(including the newly registered row); maximum depth visible at prediction: 903.
At most one earlier predicted completion was still pending at a prediction time.
The depth measure is an upper bound over visible ancestors, not a claim that all
ancestors causally affected every model split. No new weights, windows, partial
pseudo policies or month-specific choices are tried after observing these results.

## Engineering and delivery

- All six complete input reversals produce exactly identical predictions, audits
  and pseudo rows. Synthetic tests cover equal timestamp batches and zero duration.
- All 20 original archived R2 endpoints reproduce with maximum absolute difference
  1.7053025658242404e-13, within the frozen 1e-8 tolerance.
- A separate process loads the final R2 bundle and checks the first test_a timestamp
  against the unchanged v0.4 loader: U0 and U1 are exactly equal, with zero pseudo rows.
  This checks only the first timestamp; no full test_a challenger was generated.
- The independent audit reconstructs pseudo provenance, completion, visible state
  hashes, depths and pending overlaps from saved predictions and rows.
- Locked Python 3.12 suite: **182 passed**, 49.69 seconds;
  `local/reports/pytest-optimization-v0.7-r2.xml`.
- Experiment attempted/completed target fits: **0/0**. Base source contracts,
  active release and original protected ledger remain unchanged.
- Windows desktop was restored to R2 before replay. Its SHA-256 and the preserved
  repository ZIP both equal
  `e42602d3045e43b4b49dd1e1c104aa8e5c1f29c639ff5f892b3b07434ed9bbdf`.

Run: `local/runs/optimization-v0.7-opt19-r1/`.
Per-origin audit JSONL and pseudo rows are under `origins/`; all unit errors,
metrics, gates, endpoint comparisons and drift distributions remain local.
First-timestamp check: `local/runs/optimization-v0.7-first-timestamp-r1/`.
Independent audit: `local/reports/optimization-v0.7-audit-final-r1.json`.

Reproduction (use fresh output paths):

```bash
.venv/bin/python -m bf_tap.optimization.pseudo_evaluation --output local/runs/optimization-v0.7-opt19-r2
.venv/bin/python scripts/optimization_v7_first_timestamp.py --output local/runs/optimization-v0.7-first-timestamp-r2
.venv/bin/python scripts/optimization_v7_audit_check.py --run local/runs/optimization-v0.7-opt19-r2 --output local/reports/optimization-v0.7-audit-r2.json
```

## v0.8 handoff

Proceed with the user-specified average-rate target `tap_iron / tap_time_len` and
strictly temporal OOF rate/iron/time structural correction. Freeze zero-duration
handling, temporal OOF boundaries, correction formula and L1/WMAPE acceptance
before training. OOF must honor actual target availability; in-sample rate outputs
cannot be used as correction training features. Native MultiRMSE is not a justified
replacement for the official two-target L1/WMAPE objective. This phase does not
silently select those new modeling definitions or launch new fits.
